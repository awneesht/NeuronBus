//! # NeuronBus
//!
//! Neuroscience-inspired ultra-fast messaging for AI agents.
//!
//! NeuronBus models itself on biological neural networks:
//! - **Spikes** = 64-byte cache-line-aligned binary messages
//! - **Axons** = LMAX Disruptor-style lock-free ring buffers
//! - **Synapses** = Connections with STDP (adaptive routing)
//! - **Neurons** = AI agents with integrate-and-fire batching
//! - **Cortex** = Topology manager with region-based routing
//! - **Arena** = Shared memory for large payloads (zero-copy)

pub mod arena;
pub mod axon;
pub mod cortex;
pub mod neuron;
pub mod spike;
pub mod synapse;
pub mod transport;
pub mod wait;

// Re-exports for convenience.
pub use arena::{Arena, ArenaError};
pub use axon::{Axon, AxonReader};
pub use cortex::{Cortex, Region};
pub use neuron::{AgentCard, IntegrateFireState, Neuron, NeuronHandle};
pub use spike::{ArenaRef, NeuronId, Spike, SpikeBuilder, SpikeType};
pub use synapse::{DendriticFilter, StdpParams, Synapse, SynapseBuilder, SynapseTable};
pub use transport::{LocalTransport, TcpTransport, Transport, TransportServer};
pub use wait::{WaitStrategy, WaitStrategyKind};

use std::sync::Arc;

/// Builder for constructing a NeuronBus instance.
pub struct NeuronBusBuilder {
    arena_capacity: usize,
    axon_capacity: usize,
    max_neurons: usize,
    wait_strategy: WaitStrategyKind,
}

impl NeuronBusBuilder {
    pub fn new() -> Self {
        NeuronBusBuilder {
            arena_capacity: 256 * 1024 * 1024, // 256MB
            axon_capacity: 1 << 20,             // 1M slots
            max_neurons: 1024,
            wait_strategy: WaitStrategyKind::SpinLoopHint,
        }
    }

    pub fn arena_capacity(mut self, bytes: usize) -> Self {
        self.arena_capacity = bytes;
        self
    }

    pub fn axon_capacity(mut self, slots: usize) -> Self {
        assert!(slots.is_power_of_two(), "axon_capacity must be power of 2");
        self.axon_capacity = slots;
        self
    }

    pub fn max_neurons(mut self, n: usize) -> Self {
        self.max_neurons = n;
        self
    }

    pub fn wait_strategy(mut self, kind: WaitStrategyKind) -> Self {
        self.wait_strategy = kind;
        self
    }

    pub fn build(self) -> NeuronBus {
        let arena = Arc::new(Arena::new(self.arena_capacity).expect("failed to create arena"));
        let cortex = Cortex::new(Arc::clone(&arena), self.max_neurons, self.axon_capacity);

        NeuronBus {
            cortex,
            arena,
            axon_capacity: self.axon_capacity,
            wait_strategy: self.wait_strategy,
        }
    }
}

impl Default for NeuronBusBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The main NeuronBus instance.
///
/// # Example
/// ```
/// use neuronbus::*;
///
/// let mut bus = NeuronBus::builder()
///     .arena_capacity(1024 * 1024)  // 1MB
///     .axon_capacity(1 << 16)       // 64K slots
///     .wait_strategy(WaitStrategyKind::SpinLoopHint)
///     .build();
/// ```
pub struct NeuronBus {
    pub cortex: Cortex,
    arena: Arc<Arena>,
    axon_capacity: usize,
    wait_strategy: WaitStrategyKind,
}

impl NeuronBus {
    /// Create a builder for configuring the bus.
    pub fn builder() -> NeuronBusBuilder {
        NeuronBusBuilder::new()
    }

    /// Register a neuron (agent) with the bus.
    pub fn register(
        &mut self,
        agent: Box<dyn Neuron>,
        card: AgentCard,
    ) -> NeuronId {
        self.cortex.register(agent, card)
    }

    /// Connect two neurons with a configurable synapse.
    pub fn connect(&mut self, pre: NeuronId, post: NeuronId) -> cortex::ConnectionBuilder<'_> {
        self.cortex.connect(pre, post)
    }

    /// Inject a spike directly into the bus for a given neuron.
    pub fn fire(&mut self, id: NeuronId, spike: Spike) {
        if let Some(handle) = self.cortex.handle_mut(id) {
            handle.fire(spike);
        }
    }

    /// Run the event loop for N ticks.
    pub fn run(&mut self, ticks: usize) {
        self.cortex.run(ticks);
    }

    /// Run a single tick of the event loop.
    pub fn tick(&mut self) {
        self.cortex.tick();
    }

    /// Discover agents by capability.
    pub fn discover(&self, capability: &str) -> Vec<&AgentCard> {
        self.cortex.discover(capability)
    }

    /// Access the shared arena.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Wait strategy configured for this bus.
    pub fn wait_strategy(&self) -> WaitStrategyKind {
        self.wait_strategy
    }

    /// Axon capacity configured for this bus.
    pub fn axon_capacity(&self) -> usize {
        self.axon_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct PingNeuron {
        received: Arc<AtomicU32>,
    }

    impl Neuron for PingNeuron {
        fn on_spike(&mut self, _spike: &Spike, _handle: &NeuronHandle) {
            self.received.fetch_add(1, Ordering::Relaxed);
        }
        fn tick(&mut self, _handle: &NeuronHandle) {}
    }

    #[test]
    fn builder_api() {
        let bus = NeuronBus::builder()
            .arena_capacity(1024 * 1024)
            .axon_capacity(1 << 16)
            .max_neurons(64)
            .wait_strategy(WaitStrategyKind::SpinLoopHint)
            .build();

        assert_eq!(bus.axon_capacity(), 1 << 16);
        assert_eq!(bus.wait_strategy(), WaitStrategyKind::SpinLoopHint);
    }

    #[test]
    fn end_to_end() {
        let mut bus = NeuronBus::builder()
            .arena_capacity(4096)
            .axon_capacity(1024)
            .max_neurons(16)
            .build();

        let count = Arc::new(AtomicU32::new(0));
        let ping = PingNeuron { received: count.clone() };
        let pong = PingNeuron { received: Arc::new(AtomicU32::new(0)) };

        let ping_id = bus.register(
            Box::new(ping),
            AgentCard::new("ping", NeuronId(0)),
        );
        let pong_id = bus.register(
            Box::new(pong),
            AgentCard::new("pong", NeuronId(0)),
        );

        bus.connect(pong_id, ping_id).weight(0.5).build();

        // Fire from pong -> ping.
        let spike = SpikeBuilder::new()
            .target(ping_id)
            .spike_type(SpikeType::Excitatory)
            .payload(b"hello")
            .build();
        bus.fire(pong_id, spike);

        bus.tick();

        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn discovery() {
        let mut bus = NeuronBus::builder()
            .arena_capacity(4096)
            .axon_capacity(1024)
            .max_neurons(16)
            .build();

        let agent = PingNeuron { received: Arc::new(AtomicU32::new(0)) };
        let card = AgentCard::new("translator", NeuronId(0))
            .with_capabilities(vec!["translate".into(), "summarize".into()]);
        bus.register(Box::new(agent), card);

        assert_eq!(bus.discover("translate").len(), 1);
        assert_eq!(bus.discover("nonexistent").len(), 0);
    }
}
