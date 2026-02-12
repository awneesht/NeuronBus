//! Topology manager and orchestrator.
//!
//! The Cortex manages all neurons, synapses, and routing within the bus.
//! It provides region-based routing (hierarchical topics like brain regions),
//! agent discovery, and coordinates STDP updates.

use std::collections::HashMap;
use std::sync::Arc;

use crate::arena::Arena;
use crate::axon::AxonReader;
use crate::neuron::{AgentCard, IntegrateFireState, Neuron, NeuronHandle};
use crate::spike::{NeuronId, Spike};
use crate::synapse::{Synapse, SynapseBuilder, SynapseTable};

// ─── Region ─────────────────────────────────────────────────────────────────

/// A hierarchical region (like a brain area) for organizing neurons.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Region(pub String);

impl Region {
    pub fn new(name: impl Into<String>) -> Self {
        Region(name.into())
    }

    /// Check if this region is a parent of another.
    /// e.g., "cortex" is parent of "cortex/language".
    pub fn is_parent_of(&self, other: &Region) -> bool {
        other.0.starts_with(&self.0) && other.0.len() > self.0.len()
    }
}

// ─── Neuron Entry ───────────────────────────────────────────────────────────

/// Internal bookkeeping for a registered neuron.
struct NeuronEntry {
    /// The agent implementation.
    agent: Box<dyn Neuron>,
    /// Integrate-and-fire state for this neuron.
    if_state: IntegrateFireState,
    /// Agent discovery card.
    card: AgentCard,
    /// Reader to observe this neuron's output.
    output_reader: AxonReader,
}

// ─── Connection Builder ─────────────────────────────────────────────────────

/// Builder for creating connections between neurons.
pub struct ConnectionBuilder<'a> {
    cortex: &'a mut Cortex,
    pre: NeuronId,
    post: NeuronId,
    weight: f32,
}

impl<'a> ConnectionBuilder<'a> {
    pub fn weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    pub fn build(self) {
        let synapse = SynapseBuilder::new(self.pre, self.post)
            .weight(self.weight)
            .build();
        self.cortex.synapses.insert(synapse);
    }
}

// ─── Cortex ─────────────────────────────────────────────────────────────────

/// The central orchestrator managing neurons, synapses, and routing.
pub struct Cortex {
    /// Registered neurons.
    neurons: HashMap<NeuronId, NeuronEntry>,
    /// Neuron handles (owned by cortex, passed by ref to agents).
    handles: HashMap<NeuronId, NeuronHandle>,
    /// Synapse table for connections.
    pub synapses: SynapseTable,
    /// Region membership: region -> list of neuron IDs.
    regions: HashMap<Region, Vec<NeuronId>>,
    /// Shared arena for large payloads.
    arena: Arc<Arena>,
    /// Default axon capacity for new neurons.
    axon_capacity: usize,
    /// Next neuron ID to assign.
    next_id: u16,
}

impl Cortex {
    /// Create a new Cortex with the given arena and max neuron count.
    pub fn new(arena: Arc<Arena>, max_neurons: usize, axon_capacity: usize) -> Self {
        let synapses = if max_neurons <= 1024 {
            SynapseTable::dense(max_neurons)
        } else {
            SynapseTable::sparse()
        };

        Cortex {
            neurons: HashMap::new(),
            handles: HashMap::new(),
            synapses,
            regions: HashMap::new(),
            arena,
            axon_capacity,
            next_id: 0,
        }
    }

    /// Register a new neuron (agent) with the cortex.
    /// Returns the assigned NeuronId.
    pub fn register(
        &mut self,
        agent: Box<dyn Neuron>,
        mut card: AgentCard,
    ) -> NeuronId {
        let id = NeuronId(self.next_id);
        self.next_id += 1;
        card.id = id;

        // Create the neuron's handle with its own output axon.
        let handle = NeuronHandle::new(id, self.axon_capacity, Some(Arc::clone(&self.arena)));
        let output_reader = handle.reader();

        // Register in regions.
        for region_name in &card.regions {
            let region = Region::new(region_name.clone());
            self.regions.entry(region).or_default().push(id);
        }

        let entry = NeuronEntry {
            agent,
            if_state: IntegrateFireState::default(),
            card,
            output_reader,
        };

        self.neurons.insert(id, entry);
        self.handles.insert(id, handle);
        id
    }

    /// Create a connection between two neurons.
    pub fn connect(&mut self, pre: NeuronId, post: NeuronId) -> ConnectionBuilder<'_> {
        ConnectionBuilder {
            cortex: self,
            pre,
            post,
            weight: 0.5,
        }
    }

    /// Add a synapse directly.
    pub fn add_synapse(&mut self, synapse: Synapse) {
        self.synapses.insert(synapse);
    }

    /// Discover agents by capability.
    pub fn discover(&self, capability: &str) -> Vec<&AgentCard> {
        self.neurons
            .values()
            .filter(|e| e.card.has_capability(capability))
            .map(|e| &e.card)
            .collect()
    }

    /// Discover agents in a region.
    pub fn discover_in_region(&self, region: &Region) -> Vec<&AgentCard> {
        match self.regions.get(region) {
            Some(ids) => ids
                .iter()
                .filter_map(|id| self.neurons.get(id))
                .map(|e| &e.card)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get an agent card by ID.
    pub fn agent_card(&self, id: NeuronId) -> Option<&AgentCard> {
        self.neurons.get(&id).map(|e| &e.card)
    }

    /// Route a spike from source through synapses to targets.
    /// Applies dendritic filters and synapse weights.
    pub fn route_spike(&mut self, spike: &Spike) {
        let source = spike.source();
        let target = spike.target();

        if target == NeuronId::BROADCAST {
            // Broadcast: deliver to all outgoing synapses from source.
            let outgoing: Vec<NeuronId> = self.synapses.outgoing(source)
                .iter()
                .filter(|s| s.passes_filters(spike))
                .map(|s| s.post)
                .collect();

            for post_id in outgoing {
                self.deliver_spike(spike, source, post_id);
            }
        } else {
            // Unicast: deliver through the specific synapse if it exists.
            self.deliver_spike(spike, source, target);
        }
    }

    fn deliver_spike(&mut self, spike: &Spike, pre: NeuronId, post: NeuronId) {
        let weight = self.synapses.get(pre, post)
            .map(|s| {
                // Record pre-spike for STDP.
                s.on_pre_spike(spike.timestamp_ns);
                s.weight()
            })
            .unwrap_or(1.0);

        if let Some(entry) = self.neurons.get_mut(&post) {
            // Integrate the spike with the synapse weight.
            entry.if_state.integrate(spike, weight);

            // Deliver to the agent.
            if let Some(handle) = self.handles.get(&post) {
                entry.agent.on_spike(spike, handle);
            }
        }
    }

    /// Run one tick of the event loop:
    /// 1. Collect spikes from all neuron output axons.
    /// 2. Route collected spikes through synapses.
    /// 3. Tick all neurons (integrate-and-fire, leak, refractory).
    /// 4. Call agent tick() for any that fired.
    pub fn tick(&mut self) {
        // Step 1: Collect all pending output spikes.
        let mut pending_spikes = Vec::new();
        for entry in self.neurons.values() {
            while let Some(spike) = entry.output_reader.try_read_next() {
                pending_spikes.push(spike);
            }
        }

        // Step 2: Route all collected spikes.
        for spike in &pending_spikes {
            self.route_spike(spike);
        }

        // Step 3 & 4: Tick all neurons.
        let neuron_ids: Vec<NeuronId> = self.neurons.keys().copied().collect();
        for id in neuron_ids {
            let fired = {
                let entry = self.neurons.get_mut(&id).unwrap();
                entry.if_state.tick()
            };

            if fired {
                // Record post-spike for STDP on all incoming synapses.
                let timestamp = quanta::Clock::new().raw();
                // Find all synapses where this neuron is post-synaptic.
                let pre_ids: Vec<NeuronId> = self.neurons.keys().copied().collect();
                for pre_id in &pre_ids {
                    if let Some(synapse) = self.synapses.get(*pre_id, id) {
                        synapse.on_post_spike(timestamp);
                    }
                }

                // Call the agent's tick with its handle.
                if let Some(handle) = self.handles.get(&id) {
                    let entry = self.neurons.get_mut(&id).unwrap();
                    entry.agent.tick(handle);
                }
            }
        }
    }

    /// Run the event loop for N ticks.
    pub fn run(&mut self, ticks: usize) {
        for _ in 0..ticks {
            self.tick();
        }
    }

    /// Prune weak synapses and return the number pruned.
    pub fn prune_synapses(&mut self) -> usize {
        self.synapses.prune()
    }

    /// Total number of registered neurons.
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Total number of synapses.
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Access the shared arena.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Get a mutable reference to a neuron handle for external spike injection.
    pub fn handle_mut(&mut self, id: NeuronId) -> Option<&mut NeuronHandle> {
        self.handles.get_mut(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::spike::SpikeBuilder;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    struct CountingNeuron {
        received: Arc<AtomicU32>,
    }

    impl CountingNeuron {
        fn new() -> (Self, Arc<AtomicU32>) {
            let count = Arc::new(AtomicU32::new(0));
            (CountingNeuron { received: count.clone() }, count)
        }
    }

    impl Neuron for CountingNeuron {
        fn on_spike(&mut self, _spike: &Spike, _handle: &NeuronHandle) {
            self.received.fetch_add(1, AtomicOrdering::Relaxed);
        }
        fn tick(&mut self, _handle: &NeuronHandle) {}
        fn capabilities(&self) -> &[&str] { &["count"] }
    }

    #[test]
    fn register_and_discover() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let mut cortex = Cortex::new(arena, 16, 1024);

        let (agent, _) = CountingNeuron::new();
        let card = AgentCard::new("counter", NeuronId(0))
            .with_capabilities(vec!["count".into(), "math".into()]);
        cortex.register(Box::new(agent), card);

        let found = cortex.discover("count");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "counter");

        assert!(cortex.discover("nonexistent").is_empty());
    }

    #[test]
    fn region_discovery() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let mut cortex = Cortex::new(arena, 16, 1024);

        let (a1, _) = CountingNeuron::new();
        let card1 = AgentCard::new("a1", NeuronId(0))
            .with_regions(vec!["cortex/language".into()]);
        cortex.register(Box::new(a1), card1);

        let (a2, _) = CountingNeuron::new();
        let card2 = AgentCard::new("a2", NeuronId(0))
            .with_regions(vec!["cortex/vision".into()]);
        cortex.register(Box::new(a2), card2);

        let lang = cortex.discover_in_region(&Region::new("cortex/language"));
        assert_eq!(lang.len(), 1);
        assert_eq!(lang[0].name, "a1");
    }

    #[test]
    fn spike_routing() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let mut cortex = Cortex::new(arena, 16, 1024);

        let (a1, _) = CountingNeuron::new();
        let (a2, count2) = CountingNeuron::new();

        let id1 = cortex.register(Box::new(a1), AgentCard::new("sender", NeuronId(0)));
        let id2 = cortex.register(Box::new(a2), AgentCard::new("receiver", NeuronId(0)));

        cortex.connect(id1, id2).weight(0.8).build();

        // Inject a spike from agent 1.
        {
            let handle = cortex.handle_mut(id1).unwrap();
            handle.fire_to(id2, b"test");
        }

        // Tick to process the spike.
        cortex.tick();

        assert_eq!(count2.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn broadcast_routing() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let mut cortex = Cortex::new(arena, 16, 1024);

        let (sender, _) = CountingNeuron::new();
        let (r1, count1) = CountingNeuron::new();
        let (r2, count2) = CountingNeuron::new();

        let sid = cortex.register(Box::new(sender), AgentCard::new("sender", NeuronId(0)));
        let r1id = cortex.register(Box::new(r1), AgentCard::new("r1", NeuronId(0)));
        let r2id = cortex.register(Box::new(r2), AgentCard::new("r2", NeuronId(0)));

        cortex.connect(sid, r1id).weight(0.5).build();
        cortex.connect(sid, r2id).weight(0.5).build();

        {
            let handle = cortex.handle_mut(sid).unwrap();
            handle.broadcast(b"hello all");
        }

        cortex.tick();

        assert_eq!(count1.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(count2.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn integrate_fire_in_cortex() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let mut cortex = Cortex::new(arena, 16, 1024);

        let (sender, _) = CountingNeuron::new();
        let (receiver, _) = CountingNeuron::new();

        let sid = cortex.register(Box::new(sender), AgentCard::new("s", NeuronId(0)));
        let rid = cortex.register(Box::new(receiver), AgentCard::new("r", NeuronId(0)));

        // High weight connection to push past threshold.
        cortex.connect(sid, rid).weight(2.0).build();

        // Fire enough spikes to exceed threshold.
        {
            let handle = cortex.handle_mut(sid).unwrap();
            handle.fire_to(rid, b"excite");
        }

        cortex.tick();
        // After routing and tick, receiver should have integrated the spike.
        // With weight 2.0 and threshold 1.0, it should fire on this tick.
    }

    #[test]
    fn synapse_count() {
        let arena = Arc::new(Arena::new(4096).unwrap());
        let mut cortex = Cortex::new(arena, 16, 1024);

        let (a1, _) = CountingNeuron::new();
        let (a2, _) = CountingNeuron::new();
        let (a3, _) = CountingNeuron::new();

        let id1 = cortex.register(Box::new(a1), AgentCard::new("a1", NeuronId(0)));
        let id2 = cortex.register(Box::new(a2), AgentCard::new("a2", NeuronId(0)));
        let id3 = cortex.register(Box::new(a3), AgentCard::new("a3", NeuronId(0)));

        cortex.connect(id1, id2).build();
        cortex.connect(id1, id3).build();
        cortex.connect(id2, id3).build();

        assert_eq!(cortex.synapse_count(), 3);
        assert_eq!(cortex.neuron_count(), 3);
    }
}
