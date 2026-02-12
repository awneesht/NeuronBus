//! 64-byte cache-line-aligned binary message format.
//!
//! Spikes are the fundamental unit of communication in NeuronBus,
//! modeled after action potentials in biological neural networks.

use core::mem;

/// Unique identifier for a neuron (agent) in the bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NeuronId(pub u16);

impl NeuronId {
    /// Broadcast target - spike delivered to all connected neurons.
    pub const BROADCAST: NeuronId = NeuronId(0xFFFF);
}

/// Type of spike, analogous to neurotransmitter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpikeType {
    /// Increases target membrane potential (like glutamate).
    Excitatory = 0,
    /// Decreases target membrane potential (like GABA).
    Inhibitory = 1,
    /// Modulates synapse behavior without direct potential change (like dopamine).
    Modulatory = 2,
}

impl SpikeType {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => SpikeType::Excitatory,
            1 => SpikeType::Inhibitory,
            2 => SpikeType::Modulatory,
            _ => SpikeType::Excitatory,
        }
    }
}

/// Reference to data stored in the shared memory Arena.
/// Packed into the first 16 bytes of the spike payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ArenaRef {
    /// Byte offset into the arena.
    pub offset: u64,
    /// Length of the data in bytes.
    pub len: u32,
    /// Reference count slot index for deallocation.
    pub ref_slot: u32,
}

/// Marker byte at payload[0] indicating an ArenaRef is encoded.
const ARENA_REF_MARKER: u8 = 0xA0;

/// 64-byte cache-line-aligned spike (message).
///
/// Layout:
/// ```text
///   timestamp_ns: u64     (8B)  - nanosecond timestamp
///   sequence:     u32     (4B)  - monotonic per-source sequence
///   source:       u16     (2B)  - sender NeuronId
///   target:       u16     (2B)  - receiver NeuronId (0xFFFF = broadcast)
///   spike_type:   u8      (1B)  - Excitatory/Inhibitory/Modulatory
///   priority:     u8      (1B)  - 0-255
///   payload:      [u8;46] (46B) - inline data or ArenaRef
/// ```
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Spike {
    pub timestamp_ns: u64,
    pub sequence: u32,
    source: u16,
    target: u16,
    spike_type: u8,
    pub priority: u8,
    pub payload: [u8; 46],
}

// Compile-time assertions
const _: () = assert!(mem::size_of::<Spike>() == 64);
const _: () = assert!(mem::align_of::<Spike>() == 64);

impl Spike {
    /// Create a new zeroed spike.
    pub const fn zeroed() -> Self {
        Spike {
            timestamp_ns: 0,
            sequence: 0,
            source: 0,
            target: 0,
            spike_type: 0,
            priority: 0,
            payload: [0u8; 46],
        }
    }

    #[inline]
    pub fn source(&self) -> NeuronId {
        NeuronId(self.source)
    }

    #[inline]
    pub fn target(&self) -> NeuronId {
        NeuronId(self.target)
    }

    #[inline]
    pub fn set_source(&mut self, id: NeuronId) {
        self.source = id.0;
    }

    #[inline]
    pub fn set_target(&mut self, id: NeuronId) {
        self.target = id.0;
    }

    #[inline]
    pub fn spike_type(&self) -> SpikeType {
        SpikeType::from_u8(self.spike_type)
    }

    #[inline]
    pub fn set_spike_type(&mut self, t: SpikeType) {
        self.spike_type = t as u8;
    }

    /// Zero-copy view as bytes for network transmission.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Spike is repr(C, align(64)) with size 64, all bit patterns valid.
        unsafe { &*(self as *const Spike as *const [u8; 64]) }
    }

    /// Zero-copy reinterpret from bytes.
    ///
    /// # Safety
    /// `bytes` must be aligned to 64 bytes and contain a valid Spike.
    #[inline]
    pub unsafe fn from_bytes(bytes: &[u8; 64]) -> &Spike {
        // SAFETY: Caller guarantees alignment and validity.
        unsafe { &*(bytes.as_ptr() as *const Spike) }
    }

    /// Encode an ArenaRef into the payload.
    pub fn set_arena_ref(&mut self, arena_ref: ArenaRef) {
        self.payload[0] = ARENA_REF_MARKER;
        // SAFETY: ArenaRef is 16 bytes, fits in payload[1..17]
        let bytes = unsafe {
            core::slice::from_raw_parts(
                &arena_ref as *const ArenaRef as *const u8,
                mem::size_of::<ArenaRef>(),
            )
        };
        self.payload[1..1 + mem::size_of::<ArenaRef>()].copy_from_slice(bytes);
    }

    /// Try to decode an ArenaRef from the payload.
    pub fn arena_ref(&self) -> Option<ArenaRef> {
        if self.payload[0] != ARENA_REF_MARKER {
            return None;
        }
        // SAFETY: We check the marker and ArenaRef is repr(C) with no padding issues.
        let arena_ref = unsafe {
            core::ptr::read_unaligned(self.payload[1..].as_ptr() as *const ArenaRef)
        };
        Some(arena_ref)
    }

    /// Set inline payload data (up to 46 bytes).
    pub fn set_payload(&mut self, data: &[u8]) {
        let len = data.len().min(46);
        self.payload[..len].copy_from_slice(&data[..len]);
    }
}

impl core::fmt::Debug for Spike {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Spike")
            .field("timestamp_ns", &self.timestamp_ns)
            .field("sequence", &self.sequence)
            .field("source", &self.source())
            .field("target", &self.target())
            .field("spike_type", &self.spike_type())
            .field("priority", &self.priority)
            .finish()
    }
}

/// Ergonomic builder for constructing Spikes.
pub struct SpikeBuilder {
    spike: Spike,
}

impl SpikeBuilder {
    pub fn new() -> Self {
        SpikeBuilder {
            spike: Spike::zeroed(),
        }
    }

    pub fn timestamp(mut self, ns: u64) -> Self {
        self.spike.timestamp_ns = ns;
        self
    }

    pub fn sequence(mut self, seq: u32) -> Self {
        self.spike.sequence = seq;
        self
    }

    pub fn source(mut self, id: NeuronId) -> Self {
        self.spike.set_source(id);
        self
    }

    pub fn target(mut self, id: NeuronId) -> Self {
        self.spike.set_target(id);
        self
    }

    pub fn spike_type(mut self, t: SpikeType) -> Self {
        self.spike.set_spike_type(t);
        self
    }

    pub fn priority(mut self, p: u8) -> Self {
        self.spike.priority = p;
        self
    }

    pub fn payload(mut self, data: &[u8]) -> Self {
        self.spike.set_payload(data);
        self
    }

    pub fn arena_ref(mut self, r: ArenaRef) -> Self {
        self.spike.set_arena_ref(r);
        self
    }

    pub fn build(self) -> Spike {
        self.spike
    }
}

impl Default for SpikeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_size_and_alignment() {
        assert_eq!(mem::size_of::<Spike>(), 64);
        assert_eq!(mem::align_of::<Spike>(), 64);
    }

    #[test]
    fn spike_builder() {
        let spike = SpikeBuilder::new()
            .source(NeuronId(1))
            .target(NeuronId(2))
            .spike_type(SpikeType::Excitatory)
            .priority(128)
            .payload(b"hello")
            .build();

        assert_eq!(spike.source(), NeuronId(1));
        assert_eq!(spike.target(), NeuronId(2));
        assert_eq!(spike.spike_type(), SpikeType::Excitatory);
        assert_eq!(spike.priority, 128);
        assert_eq!(&spike.payload[..5], b"hello");
    }

    #[test]
    fn arena_ref_roundtrip() {
        let aref = ArenaRef {
            offset: 1024,
            len: 4096,
            ref_slot: 7,
        };
        let mut spike = Spike::zeroed();
        spike.set_arena_ref(aref);

        let decoded = spike.arena_ref().expect("should decode ArenaRef");
        assert_eq!(decoded.offset, 1024);
        assert_eq!(decoded.len, 4096);
        assert_eq!(decoded.ref_slot, 7);
    }

    #[test]
    fn no_arena_ref_when_not_set() {
        let spike = Spike::zeroed();
        assert!(spike.arena_ref().is_none());
    }

    #[test]
    fn as_bytes_roundtrip() {
        let spike = SpikeBuilder::new()
            .source(NeuronId(42))
            .target(NeuronId(99))
            .sequence(12345)
            .build();

        let bytes = spike.as_bytes();
        // SAFETY: bytes came from a valid Spike and is 64-byte aligned (on stack from Spike).
        let recovered = unsafe { Spike::from_bytes(bytes) };
        assert_eq!(recovered.source(), NeuronId(42));
        assert_eq!(recovered.target(), NeuronId(99));
        assert_eq!(recovered.sequence, 12345);
    }

    #[test]
    fn broadcast_target() {
        assert_eq!(NeuronId::BROADCAST, NeuronId(0xFFFF));
    }
}
