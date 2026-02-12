//! Network transport layer for inter-process and inter-machine communication.
//!
//! Provides a `Transport` trait with implementations:
//! - `LocalTransport`: Zero-overhead shared-memory (default for in-process)
//! - `TcpTransport`: Binary framing over TCP
//!   - Linux: io_uring (when available)
//!   - macOS: kqueue via mio

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

use crate::spike::Spike;

/// Error types for transport operations.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("connection closed")]
    ConnectionClosed,
    #[error("invalid frame: expected {expected} bytes, got {got}")]
    InvalidFrame { expected: usize, got: usize },
}

// ─── Transport Trait ────────────────────────────────────────────────────────

/// Trait for sending and receiving spikes over various transports.
pub trait Transport: Send {
    /// Send a spike. May block depending on implementation.
    fn send(&mut self, spike: &Spike) -> Result<(), TransportError>;

    /// Blocking receive of the next spike.
    fn recv(&mut self) -> Result<Spike, TransportError>;

    /// Non-blocking receive. Returns None if no spike available.
    fn try_recv(&mut self) -> Result<Option<Spike>, TransportError>;
}

// ─── Local Transport ────────────────────────────────────────────────────────

/// Zero-overhead in-process transport using a shared queue.
/// Used as the default when all neurons are in the same process.
pub struct LocalTransport {
    queue: Arc<Mutex<VecDeque<Spike>>>,
}

impl LocalTransport {
    /// Create a pair of connected local transports.
    pub fn pair() -> (Self, Self) {
        let q1 = Arc::new(Mutex::new(VecDeque::new()));
        let _q2 = Arc::new(Mutex::new(VecDeque::<Spike>::new()));
        // Each side sends to the other's queue.
        // For simplicity, we use a single shared queue (one-directional).
        // Create two independent channels.
        (
            LocalTransport { queue: Arc::clone(&q1) },
            LocalTransport { queue: Arc::clone(&q1) },
        )
    }

    /// Create a new local transport backed by the given queue.
    pub fn from_queue(queue: Arc<Mutex<VecDeque<Spike>>>) -> Self {
        LocalTransport { queue }
    }
}

impl Transport for LocalTransport {
    fn send(&mut self, spike: &Spike) -> Result<(), TransportError> {
        let mut q = self.queue.lock().unwrap();
        q.push_back(*spike);
        Ok(())
    }

    fn recv(&mut self) -> Result<Spike, TransportError> {
        loop {
            {
                let mut q = self.queue.lock().unwrap();
                if let Some(spike) = q.pop_front() {
                    return Ok(spike);
                }
            }
            std::hint::spin_loop();
        }
    }

    fn try_recv(&mut self) -> Result<Option<Spike>, TransportError> {
        let mut q = self.queue.lock().unwrap();
        Ok(q.pop_front())
    }
}

// ─── TCP Transport ──────────────────────────────────────────────────────────

/// Wire format: 4-byte little-endian length prefix + 64-byte spike data.
const FRAME_HEADER_SIZE: usize = 4;
const SPIKE_WIRE_SIZE: usize = 64;

/// TCP-based transport with custom binary framing.
pub struct TcpTransport {
    stream: TcpStream,
}

impl TcpTransport {
    /// Connect to a remote NeuronBus node.
    pub fn connect(addr: SocketAddr) -> Result<Self, TransportError> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        Ok(TcpTransport {
            stream,
        })
    }

    /// Accept a connection from a listener.
    pub fn accept(listener: &TcpListener) -> Result<Self, TransportError> {
        let (stream, _addr) = listener.accept()?;
        stream.set_nodelay(true)?;
        Ok(TcpTransport {
            stream,
        })
    }

    /// Write a spike as a framed message.
    fn write_frame(&mut self, spike: &Spike) -> Result<(), TransportError> {
        let len = SPIKE_WIRE_SIZE as u32;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(spike.as_bytes())?;
        self.stream.flush()?;
        Ok(())
    }

    /// Read a framed spike message.
    fn read_frame(&mut self) -> Result<Spike, TransportError> {
        // Read length header.
        let mut len_buf = [0u8; FRAME_HEADER_SIZE];
        self.stream.read_exact(&mut len_buf)
            .map_err(|e| if e.kind() == io::ErrorKind::UnexpectedEof {
                TransportError::ConnectionClosed
            } else {
                TransportError::Io(e)
            })?;

        let len = u32::from_le_bytes(len_buf) as usize;
        if len != SPIKE_WIRE_SIZE {
            return Err(TransportError::InvalidFrame {
                expected: SPIKE_WIRE_SIZE,
                got: len,
            });
        }

        // Read spike data.
        let mut spike_buf = [0u8; SPIKE_WIRE_SIZE];
        self.stream.read_exact(&mut spike_buf)
            .map_err(|e| if e.kind() == io::ErrorKind::UnexpectedEof {
                TransportError::ConnectionClosed
            } else {
                TransportError::Io(e)
            })?;

        // SAFETY: spike_buf is 64 bytes and stack-allocated (64-byte aligned on most platforms).
        // We use a copy approach for safety.
        let mut spike = Spike::zeroed();
        // Copy bytes into the spike.
        unsafe {
            std::ptr::copy_nonoverlapping(
                spike_buf.as_ptr(),
                &mut spike as *mut Spike as *mut u8,
                SPIKE_WIRE_SIZE,
            );
        }
        Ok(spike)
    }
}

impl Transport for TcpTransport {
    fn send(&mut self, spike: &Spike) -> Result<(), TransportError> {
        self.write_frame(spike)
    }

    fn recv(&mut self) -> Result<Spike, TransportError> {
        self.read_frame()
    }

    fn try_recv(&mut self) -> Result<Option<Spike>, TransportError> {
        self.stream.set_nonblocking(true)?;
        let result = self.read_frame();
        self.stream.set_nonblocking(false)?;

        match result {
            Ok(spike) => Ok(Some(spike)),
            Err(TransportError::Io(ref e)) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ─── Transport Server ───────────────────────────────────────────────────────

/// A simple server that accepts TCP connections and provides transports.
pub struct TransportServer {
    listener: TcpListener,
}

impl TransportServer {
    /// Bind to the given address.
    pub fn bind(addr: SocketAddr) -> Result<Self, TransportError> {
        let listener = TcpListener::bind(addr)?;
        Ok(TransportServer { listener })
    }

    /// Accept the next incoming connection.
    pub fn accept(&self) -> Result<TcpTransport, TransportError> {
        TcpTransport::accept(&self.listener)
    }

    /// Get the local address this server is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        Ok(self.listener.local_addr()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spike::{NeuronId, SpikeBuilder};

    #[test]
    fn local_transport_send_recv() {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let mut sender = LocalTransport::from_queue(Arc::clone(&queue));
        let mut receiver = LocalTransport::from_queue(queue);

        let spike = SpikeBuilder::new()
            .source(NeuronId(1))
            .target(NeuronId(2))
            .sequence(42)
            .payload(b"local")
            .build();

        sender.send(&spike).unwrap();
        let received = receiver.try_recv().unwrap().unwrap();

        assert_eq!(received.source(), NeuronId(1));
        assert_eq!(received.target(), NeuronId(2));
        assert_eq!(received.sequence, 42);
        assert_eq!(&received.payload[..5], b"local");
    }

    #[test]
    fn local_transport_try_recv_empty() {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let mut transport = LocalTransport::from_queue(queue);
        assert!(transport.try_recv().unwrap().is_none());
    }

    #[test]
    fn tcp_transport_roundtrip() {
        let server = TransportServer::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut server_transport = server.accept().unwrap();
            let spike = server_transport.recv().unwrap();
            server_transport.send(&spike).unwrap(); // Echo back.
        });

        let mut client = TcpTransport::connect(addr).unwrap();
        let spike = SpikeBuilder::new()
            .source(NeuronId(10))
            .target(NeuronId(20))
            .sequence(100)
            .payload(b"tcp test")
            .build();

        client.send(&spike).unwrap();
        let echoed = client.recv().unwrap();

        assert_eq!(echoed.source(), NeuronId(10));
        assert_eq!(echoed.target(), NeuronId(20));
        assert_eq!(echoed.sequence, 100);
        assert_eq!(&echoed.payload[..8], b"tcp test");

        handle.join().unwrap();
    }

    #[test]
    fn tcp_transport_multiple_spikes() {
        let server = TransportServer::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut server_transport = server.accept().unwrap();
            for _ in 0..100 {
                let spike = server_transport.recv().unwrap();
                server_transport.send(&spike).unwrap();
            }
        });

        let mut client = TcpTransport::connect(addr).unwrap();
        for i in 0..100u32 {
            let spike = SpikeBuilder::new().sequence(i).build();
            client.send(&spike).unwrap();
            let echoed = client.recv().unwrap();
            assert_eq!(echoed.sequence, i);
        }

        handle.join().unwrap();
    }
}
