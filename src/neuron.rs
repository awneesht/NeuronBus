//! Agent abstraction modeled after biological neurons.
//!
//! Each AI agent is represented as a Neuron with an integrate-and-fire model:
//! incoming excitatory spikes increase membrane potential, inhibitory spikes
//! decrease it. When potential exceeds a threshold, the neuron "fires" and
//! enters a refractory period (backpressure).

use std::sync::Arc;

use crate::arena::Arena;
use crate::axon::{Axon, AxonReader};
use crate::spike::{NeuronId, Spike, SpikeBuilder, SpikeType};

// ─── Neuron Trait ───────────────────────────────────────────────────────────

/// Trait that AI agents implement to participate in the NeuronBus.
pub trait Neuron: Send {
    /// Called when a spike is delivered to this neuron.
    fn on_spike(&mut self, spike: &Spike, handle: &NeuronHandle);

    /// Called periodically (each tick of the cortex event loop).
    fn tick(&mut self, handle: &NeuronHandle);

    /// Agent capabilities for discovery.
    fn capabilities(&self) -> &[&str] {
        &[]
    }
}

// ─── Integrate-and-Fire State ───────────────────────────────────────────────

/// Integrate-and-fire model state for batching and backpressure.
#[derive(Debug, Clone)]
pub struct IntegrateFireState {
    /// Current membrane potential.
    pub potential: f32,
    /// Firing threshold - when potential exceeds this, the neuron fires.
    pub threshold: f32,
    /// Resting potential (potential decays toward this).
    pub rest: f32,
    /// Leak rate per tick (multiplicative decay toward rest).
    pub leak: f32,
    /// Refractory period in ticks (backpressure after firing).
    pub refractory_period: u32,
    /// Remaining refractory ticks (0 = ready to fire).
    refractory_remaining: u32,
    /// Whether the neuron fired on the last potential update.
    fired: bool,
}

impl IntegrateFireState {
    pub fn new(threshold: f32) -> Self {
        IntegrateFireState {
            potential: 0.0,
            threshold,
            rest: 0.0,
            leak: 0.95,
            refractory_period: 2,
            refractory_remaining: 0,
            fired: false,
        }
    }

    /// Integrate an incoming spike's effect on membrane potential.
    pub fn integrate(&mut self, spike: &Spike, weight: f32) {
        if self.refractory_remaining > 0 {
            return; // In refractory period, ignore inputs.
        }

        match spike.spike_type() {
            SpikeType::Excitatory => {
                self.potential += weight;
            }
            SpikeType::Inhibitory => {
                self.potential -= weight;
                self.potential = self.potential.max(self.rest - 1.0);
            }
            SpikeType::Modulatory => {
                // Modulatory spikes don't directly affect potential.
            }
        }
    }

    /// Check if the neuron should fire and apply leak/refractory.
    /// Returns true if the neuron fires this tick.
    pub fn tick(&mut self) -> bool {
        self.fired = false;

        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            return false;
        }

        if self.potential >= self.threshold {
            self.fired = true;
            self.potential = self.rest;
            self.refractory_remaining = self.refractory_period;
            return true;
        }

        // Leak: decay toward resting potential.
        self.potential = self.rest + (self.potential - self.rest) * self.leak;
        false
    }

    /// Whether currently in refractory period (backpressure).
    pub fn is_refractory(&self) -> bool {
        self.refractory_remaining > 0
    }

    /// Whether the neuron fired on the last tick.
    pub fn fired(&self) -> bool {
        self.fired
    }
}

impl Default for IntegrateFireState {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ─── Agent Card ─────────────────────────────────────────────────────────────

/// A2A-inspired capability discovery card for an agent.
#[derive(Debug, Clone)]
pub struct AgentCard {
    /// Human-readable name.
    pub name: String,
    /// Unique neuron ID assigned by the cortex.
    pub id: NeuronId,
    /// List of capabilities this agent advertises.
    pub capabilities: Vec<String>,
    /// Regions this agent belongs to (hierarchical topics).
    pub regions: Vec<String>,
}

impl AgentCard {
    pub fn new(name: impl Into<String>, id: NeuronId) -> Self {
        AgentCard {
            name: name.into(),
            id,
            capabilities: Vec::new(),
            regions: Vec::new(),
        }
    }

    pub fn with_capabilities(mut self, caps: Vec<String>) -> Self {
        self.capabilities = caps;
        self
    }

    pub fn with_regions(mut self, regions: Vec<String>) -> Self {
        self.regions = regions;
        self
    }

    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

// ─── Neuron Handle ──────────────────────────────────────────────────────────

/// Handle given to neurons for sending spikes and accessing the arena.
pub struct NeuronHandle {
    /// This neuron's ID.
    pub id: NeuronId,
    /// Output axon for publishing spikes.
    axon: Axon,
    /// Shared arena for large payloads.
    arena: Option<Arc<Arena>>,
    /// Sequence counter for outgoing spikes.
    sequence: u32,
    /// Quanta clock for timestamps.
    clock: quanta::Clock,
}

impl NeuronHandle {
    pub fn new(id: NeuronId, axon_capacity: usize, arena: Option<Arc<Arena>>) -> Self {
        NeuronHandle {
            id,
            axon: Axon::new(axon_capacity),
            arena,
            sequence: 0,
            clock: quanta::Clock::new(),
        }
    }

    /// Fire (publish) a spike to the output axon.
    #[inline]
    pub fn fire(&mut self, mut spike: Spike) {
        spike.set_source(self.id);
        spike.sequence = self.sequence;
        spike.timestamp_ns = self.clock.raw();
        self.sequence += 1;
        self.axon.publish(spike);
    }

    /// Fire a simple excitatory spike to a target.
    pub fn fire_to(&mut self, target: NeuronId, payload: &[u8]) {
        let spike = SpikeBuilder::new()
            .target(target)
            .spike_type(SpikeType::Excitatory)
            .payload(payload)
            .build();
        self.fire(spike);
    }

    /// Fire a broadcast spike.
    pub fn broadcast(&mut self, payload: &[u8]) {
        self.fire_to(NeuronId::BROADCAST, payload);
    }

    /// Get a reader handle for this neuron's output axon.
    pub fn reader(&self) -> AxonReader {
        self.axon.reader()
    }

    /// Access the shared arena (if available).
    pub fn arena(&self) -> Option<&Arena> {
        self.arena.as_deref()
    }

    /// Current outgoing sequence number.
    pub fn sequence(&self) -> u32 {
        self.sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrate_fire_basic() {
        let mut state = IntegrateFireState::new(1.0);

        // Sub-threshold excitation.
        let mut spike = Spike::zeroed();
        spike.set_spike_type(SpikeType::Excitatory);
        state.integrate(&spike, 0.5);
        assert!((state.potential - 0.5).abs() < f32::EPSILON);
        assert!(!state.tick());

        // Push over threshold.
        state.integrate(&spike, 0.6);
        assert!(state.potential > 1.0);
        assert!(state.tick()); // Should fire.

        // Should be in refractory period.
        assert!(state.is_refractory());
        assert!(!state.tick());
    }

    #[test]
    fn inhibitory_reduces_potential() {
        let mut state = IntegrateFireState::new(1.0);

        let mut exc = Spike::zeroed();
        exc.set_spike_type(SpikeType::Excitatory);
        state.integrate(&exc, 0.8);

        let mut inh = Spike::zeroed();
        inh.set_spike_type(SpikeType::Inhibitory);
        state.integrate(&inh, 0.5);

        assert!((state.potential - 0.3).abs() < 0.001);
    }

    #[test]
    fn refractory_blocks_integration() {
        let mut state = IntegrateFireState::new(1.0);
        state.refractory_remaining = 3;

        let mut spike = Spike::zeroed();
        spike.set_spike_type(SpikeType::Excitatory);
        state.integrate(&spike, 10.0);
        assert!((state.potential - 0.0).abs() < f32::EPSILON); // Should be ignored.
    }

    #[test]
    fn leak_decay() {
        let mut state = IntegrateFireState::new(1.0);
        state.potential = 0.5;
        state.tick(); // Doesn't fire, but leaks.
        assert!(state.potential < 0.5);
        assert!(state.potential > 0.0);
    }

    #[test]
    fn agent_card() {
        let card = AgentCard::new("test-agent", NeuronId(1))
            .with_capabilities(vec!["summarize".into(), "translate".into()])
            .with_regions(vec!["cortex/language".into()]);

        assert!(card.has_capability("summarize"));
        assert!(!card.has_capability("compute"));
        assert_eq!(card.regions.len(), 1);
    }

    #[test]
    fn neuron_handle_fire() {
        let mut handle = NeuronHandle::new(NeuronId(5), 1024, None);
        let reader = handle.reader();

        handle.fire_to(NeuronId(10), b"hello");
        handle.fire_to(NeuronId(11), b"world");

        let s1 = reader.try_read_next().unwrap();
        assert_eq!(s1.source(), NeuronId(5));
        assert_eq!(s1.target(), NeuronId(10));
        assert_eq!(s1.sequence, 0);
        assert_eq!(&s1.payload[..5], b"hello");

        let s2 = reader.try_read_next().unwrap();
        assert_eq!(s2.sequence, 1);
        assert_eq!(s2.target(), NeuronId(11));
    }

    #[test]
    fn neuron_handle_broadcast() {
        let mut handle = NeuronHandle::new(NeuronId(1), 1024, None);
        let reader = handle.reader();

        handle.broadcast(b"ping");
        let spike = reader.try_read_next().unwrap();
        assert_eq!(spike.target(), NeuronId::BROADCAST);
    }

    #[test]
    fn neuron_handle_with_arena() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let handle = NeuronHandle::new(NeuronId(1), 1024, Some(arena));
        assert!(handle.arena().is_some());
    }

    /// Simple test neuron that echoes back received spikes.
    struct EchoNeuron;

    impl Neuron for EchoNeuron {
        fn on_spike(&mut self, spike: &Spike, handle: &NeuronHandle) {
            let _ = (spike, handle); // Just verify the trait works.
        }

        fn tick(&mut self, _handle: &NeuronHandle) {}

        fn capabilities(&self) -> &[&str] {
            &["echo"]
        }
    }

    #[test]
    fn neuron_trait_impl() {
        let neuron = EchoNeuron;
        assert_eq!(neuron.capabilities(), &["echo"]);
    }
}
