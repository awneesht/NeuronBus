//! Lock-free ring buffer (LMAX Disruptor pattern).
//!
//! An Axon is a Single-Producer Multi-Consumer ring buffer for Spikes.
//! Each neuron owns one output Axon; consumers get AxonReader handles.
//!
//! The design follows the LMAX Disruptor:
//! - Power-of-2 capacity with bitmask indexing (no modulo)
//! - CachePadded cursors to prevent false sharing
//! - Single atomic Release store for publish (not CAS)
//! - Per-consumer read cursors with Acquire loads

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_utils::CachePadded;

use crate::spike::Spike;
use crate::wait::WaitStrategy;

/// Shared state between producer and consumers.
struct AxonInner {
    /// Write cursor - incremented by producer after writing.
    write_cursor: CachePadded<AtomicU64>,
    /// Pre-allocated spike buffer, 64-byte aligned.
    buffer: *mut Spike,
    /// Capacity (always power of 2).
    capacity: u64,
    /// Bitmask for fast index computation: capacity - 1.
    mask: u64,
}

// SAFETY: The buffer is only written by the single producer and read by consumers
// after the write cursor advances (enforced by Release/Acquire ordering).
unsafe impl Send for AxonInner {}
unsafe impl Sync for AxonInner {}

impl Drop for AxonInner {
    fn drop(&mut self) {
        // SAFETY: We allocated this buffer with the global allocator in Axon::new().
        unsafe {
            let layout = std::alloc::Layout::from_size_align(
                self.capacity as usize * std::mem::size_of::<Spike>(),
                64,
            )
            .unwrap();
            std::alloc::dealloc(self.buffer as *mut u8, layout);
        }
    }
}

/// Single-producer handle for publishing spikes to the ring buffer.
pub struct Axon {
    inner: Arc<AxonInner>,
    /// Local write sequence (not shared, only producer increments).
    write_seq: u64,
}

impl Axon {
    /// Create a new Axon with the given capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is not a power of 2 or is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0 && capacity.is_power_of_two(), "capacity must be a power of 2");

        // SAFETY: Allocate zeroed, 64-byte aligned buffer for Spikes.
        let layout = std::alloc::Layout::from_size_align(
            capacity * std::mem::size_of::<Spike>(),
            64,
        )
        .unwrap();
        let buffer = unsafe { std::alloc::alloc_zeroed(layout) as *mut Spike };
        assert!(!buffer.is_null(), "allocation failed");

        Axon {
            inner: Arc::new(AxonInner {
                write_cursor: CachePadded::new(AtomicU64::new(0)),
                buffer,
                capacity: capacity as u64,
                mask: (capacity - 1) as u64,
            }),
            write_seq: 0,
        }
    }

    /// Publish a single spike. Zero allocation, single atomic store.
    #[inline]
    pub fn publish(&mut self, spike: Spike) {
        let index = self.write_seq & self.inner.mask;
        // SAFETY: index is always within bounds (mask ensures it), and we are
        // the sole producer so no concurrent writes to this slot.
        unsafe {
            let slot = self.inner.buffer.add(index as usize);
            std::ptr::write(slot, spike);
        }
        self.write_seq += 1;
        // Release store ensures the spike data is visible before consumers see the new cursor.
        self.inner.write_cursor.store(self.write_seq, Ordering::Release);
    }

    /// Publish a batch of spikes with a single cursor update.
    #[inline]
    pub fn batch_publish(&mut self, spikes: &[Spike]) {
        for spike in spikes {
            let index = self.write_seq & self.inner.mask;
            // SAFETY: Same as publish - sole producer, index in bounds.
            unsafe {
                let slot = self.inner.buffer.add(index as usize);
                std::ptr::write(slot, *spike);
            }
            self.write_seq += 1;
        }
        // Single cursor update for the entire batch.
        self.inner.write_cursor.store(self.write_seq, Ordering::Release);
    }

    /// Create a reader handle for a consumer.
    pub fn reader(&self) -> AxonReader {
        AxonReader {
            inner: Arc::clone(&self.inner),
            read_cursor: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Current write sequence (number of spikes published so far).
    pub fn write_seq(&self) -> u64 {
        self.write_seq
    }
}

/// Per-consumer read handle for the ring buffer.
pub struct AxonReader {
    inner: Arc<AxonInner>,
    /// This consumer's read position.
    read_cursor: CachePadded<AtomicU64>,
}

impl AxonReader {
    /// Blocking read of the next spike. Uses the given wait strategy.
    #[inline]
    pub fn read_next(&self, strategy: &WaitStrategy) -> Spike {
        let current = self.read_cursor.load(Ordering::Relaxed);
        // Wait until producer has written past our position.
        strategy.wait_for(&self.inner.write_cursor, current);

        let index = current & self.inner.mask;
        // SAFETY: The write cursor has advanced past this position, so the data
        // is fully written. Index is in bounds via mask.
        let spike = unsafe {
            let slot = self.inner.buffer.add(index as usize);
            std::ptr::read(slot)
        };
        self.read_cursor.store(current + 1, Ordering::Release);
        spike
    }

    /// Non-blocking try-read. Returns None if no new spike available.
    #[inline]
    pub fn try_read_next(&self) -> Option<Spike> {
        let current = self.read_cursor.load(Ordering::Relaxed);
        let write_pos = self.inner.write_cursor.load(Ordering::Acquire);

        if write_pos <= current {
            return None;
        }

        let index = current & self.inner.mask;
        // SAFETY: Write cursor confirms data is written, index in bounds.
        let spike = unsafe {
            let slot = self.inner.buffer.add(index as usize);
            std::ptr::read(slot)
        };
        self.read_cursor.store(current + 1, Ordering::Release);
        Some(spike)
    }

    /// Read a batch of up to `max` spikes. Returns the spikes that were available.
    #[inline]
    pub fn read_batch(&self, max: usize) -> Vec<Spike> {
        let current = self.read_cursor.load(Ordering::Relaxed);
        let write_pos = self.inner.write_cursor.load(Ordering::Acquire);

        if write_pos <= current {
            return Vec::new();
        }

        let available = (write_pos - current) as usize;
        let count = available.min(max);
        let mut batch = Vec::with_capacity(count);

        for i in 0..count {
            let index = (current + i as u64) & self.inner.mask;
            // SAFETY: All positions up to write_pos have been written, index in bounds.
            let spike = unsafe {
                let slot = self.inner.buffer.add(index as usize);
                std::ptr::read(slot)
            };
            batch.push(spike);
        }

        self.read_cursor.store(current + count as u64, Ordering::Release);
        batch
    }

    /// Current read position.
    pub fn read_pos(&self) -> u64 {
        self.read_cursor.load(Ordering::Relaxed)
    }
}

impl Clone for AxonReader {
    fn clone(&self) -> Self {
        AxonReader {
            inner: Arc::clone(&self.inner),
            read_cursor: CachePadded::new(AtomicU64::new(
                self.read_cursor.load(Ordering::Relaxed),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spike::{NeuronId, SpikeBuilder};
    use crate::wait::WaitStrategy;

    #[test]
    fn single_producer_single_consumer() {
        let mut axon = Axon::new(1024);
        let reader = axon.reader();

        for i in 0..100u32 {
            let spike = SpikeBuilder::new()
                .source(NeuronId(1))
                .sequence(i)
                .build();
            axon.publish(spike);
        }

        for i in 0..100u32 {
            let spike = reader.try_read_next().unwrap();
            assert_eq!(spike.sequence, i);
            assert_eq!(spike.source(), NeuronId(1));
        }

        assert!(reader.try_read_next().is_none());
    }

    #[test]
    fn single_producer_multi_consumer() {
        let mut axon = Axon::new(1024);
        let reader1 = axon.reader();
        let reader2 = axon.reader();

        for i in 0..50u32 {
            let spike = SpikeBuilder::new().sequence(i).build();
            axon.publish(spike);
        }

        // Both readers should see all 50 spikes independently.
        for i in 0..50u32 {
            assert_eq!(reader1.try_read_next().unwrap().sequence, i);
            assert_eq!(reader2.try_read_next().unwrap().sequence, i);
        }

        assert!(reader1.try_read_next().is_none());
        assert!(reader2.try_read_next().is_none());
    }

    #[test]
    fn batch_publish_and_read() {
        let mut axon = Axon::new(1024);
        let reader = axon.reader();

        let spikes: Vec<Spike> = (0..10u32)
            .map(|i| SpikeBuilder::new().sequence(i).build())
            .collect();

        axon.batch_publish(&spikes);
        let batch = reader.read_batch(20);
        assert_eq!(batch.len(), 10);
        for (i, s) in batch.iter().enumerate() {
            assert_eq!(s.sequence, i as u32);
        }
    }

    #[test]
    fn blocking_read_with_strategy() {
        let mut axon = Axon::new(1024);
        let reader = axon.reader();

        let spike = SpikeBuilder::new().sequence(42).build();
        axon.publish(spike);

        let result = reader.read_next(&WaitStrategy::BusySpin);
        assert_eq!(result.sequence, 42);
    }

    #[test]
    fn cross_thread_spsc() {
        let mut axon = Axon::new(1 << 16);
        let reader = axon.reader();
        let count = 100_000u32;

        let producer = std::thread::spawn(move || {
            for i in 0..count {
                let spike = SpikeBuilder::new().sequence(i).build();
                axon.publish(spike);
            }
        });

        let consumer = std::thread::spawn(move || {
            let strategy = WaitStrategy::SpinLoopHint;
            for i in 0..count {
                let spike = reader.read_next(&strategy);
                assert_eq!(spike.sequence, i);
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn cross_thread_spmc() {
        let mut axon = Axon::new(1 << 16);
        let count = 50_000u32;
        let num_consumers = 4;

        let readers: Vec<_> = (0..num_consumers).map(|_| axon.reader()).collect();

        let producer = std::thread::spawn(move || {
            for i in 0..count {
                let spike = SpikeBuilder::new().sequence(i).build();
                axon.publish(spike);
            }
        });

        let consumers: Vec<_> = readers
            .into_iter()
            .map(|reader| {
                std::thread::spawn(move || {
                    let strategy = WaitStrategy::SpinLoopHint;
                    for i in 0..count {
                        let spike = reader.read_next(&strategy);
                        assert_eq!(spike.sequence, i);
                    }
                })
            })
            .collect();

        producer.join().unwrap();
        for c in consumers {
            c.join().unwrap();
        }
    }

    #[test]
    fn wrap_around() {
        // Small buffer to force wrap-around.
        let mut axon = Axon::new(8);
        let reader = axon.reader();

        for i in 0..32u32 {
            let spike = SpikeBuilder::new().sequence(i).build();
            axon.publish(spike);
            let read = reader.try_read_next().unwrap();
            assert_eq!(read.sequence, i);
        }
    }

    #[test]
    #[should_panic]
    fn non_power_of_two_panics() {
        Axon::new(100);
    }
}
