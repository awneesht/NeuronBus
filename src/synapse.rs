//! Adaptive connections with Spike-Timing-Dependent Plasticity (STDP).
//!
//! Synapses connect neurons and modulate spike delivery based on learned weights.
//! The STDP rule strengthens connections where the pre-synaptic neuron fires just
//! before the post-synaptic neuron (potentiation), and weakens those where
//! post fires before pre (depression).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::spike::{NeuronId, Spike, SpikeType};

// ─── AtomicF32 ──────────────────────────────────────────────────────────────

/// Lock-free atomic f32 using CAS on AtomicU32 bit representation.
pub struct AtomicF32 {
    bits: AtomicU32,
}

impl AtomicF32 {
    pub fn new(val: f32) -> Self {
        AtomicF32 {
            bits: AtomicU32::new(val.to_bits()),
        }
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.bits.load(order))
    }

    #[inline]
    pub fn store(&self, val: f32, order: Ordering) {
        self.bits.store(val.to_bits(), order);
    }

    /// Atomically add `delta` to the current value via CAS loop.
    #[inline]
    pub fn fetch_add(&self, delta: f32, order: Ordering) -> f32 {
        loop {
            let current_bits = self.bits.load(Ordering::Relaxed);
            let current = f32::from_bits(current_bits);
            let new = current + delta;
            match self.bits.compare_exchange_weak(
                current_bits,
                new.to_bits(),
                order,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current,
                Err(_) => continue,
            }
        }
    }
}

// ─── STDP Parameters ────────────────────────────────────────────────────────

/// Parameters for the STDP learning rule.
#[derive(Debug, Clone, Copy)]
pub struct StdpParams {
    /// Potentiation amplitude (pre-before-post). Typically 0.01.
    pub a_plus: f32,
    /// Depression amplitude (post-before-pre). Typically 0.012 (> a_plus to prevent saturation).
    pub a_minus: f32,
    /// Potentiation time constant in nanoseconds.
    pub tau_plus_ns: f64,
    /// Depression time constant in nanoseconds.
    pub tau_minus_ns: f64,
    /// Minimum weight (below this, synapse is pruned).
    pub w_min: f32,
    /// Maximum weight.
    pub w_max: f32,
}

impl Default for StdpParams {
    fn default() -> Self {
        StdpParams {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus_ns: 20_000_000.0,  // 20ms
            tau_minus_ns: 20_000_000.0, // 20ms
            w_min: 0.001,
            w_max: 1.0,
        }
    }
}

// ─── Dendritic Filters ──────────────────────────────────────────────────────

/// Filters that synapses apply before delivering a spike.
#[derive(Debug, Clone)]
pub enum DendriticFilter {
    /// Only pass spikes of the given type.
    TypeFilter(SpikeType),
    /// Only pass spikes with priority >= threshold.
    PriorityThreshold(u8),
    /// Only pass spikes whose payload starts with the given prefix.
    PayloadPrefix(Vec<u8>),
    /// Custom filter function (boxed for trait-object compatibility).
    Custom(String),
}

impl DendriticFilter {
    /// Returns true if the spike passes this filter.
    pub fn passes(&self, spike: &Spike) -> bool {
        match self {
            DendriticFilter::TypeFilter(t) => spike.spike_type() == *t,
            DendriticFilter::PriorityThreshold(min) => spike.priority >= *min,
            DendriticFilter::PayloadPrefix(prefix) => spike.payload.starts_with(prefix),
            DendriticFilter::Custom(_) => true, // Custom filters are evaluated externally.
        }
    }
}

// ─── Synapse ────────────────────────────────────────────────────────────────

/// A connection between two neurons with adaptive weight.
pub struct Synapse {
    pub pre: NeuronId,
    pub post: NeuronId,
    /// Connection weight, updated by STDP.
    weight: AtomicF32,
    /// Last pre-synaptic spike time (for STDP).
    last_pre_ns: AtomicU64,
    /// Last post-synaptic spike time (for STDP).
    last_post_ns: AtomicU64,
    /// Optional filters to apply before delivery.
    pub filters: Vec<DendriticFilter>,
    /// STDP parameters for this synapse.
    pub stdp: StdpParams,
}

impl Synapse {
    pub fn new(pre: NeuronId, post: NeuronId, initial_weight: f32) -> Self {
        Synapse {
            pre,
            post,
            weight: AtomicF32::new(initial_weight),
            last_pre_ns: AtomicU64::new(u64::MAX),
            last_post_ns: AtomicU64::new(u64::MAX),
            filters: Vec::new(),
            stdp: StdpParams::default(),
        }
    }

    pub fn with_filters(mut self, filters: Vec<DendriticFilter>) -> Self {
        self.filters = filters;
        self
    }

    pub fn with_stdp(mut self, stdp: StdpParams) -> Self {
        self.stdp = stdp;
        self
    }

    /// Get current weight.
    #[inline]
    pub fn weight(&self) -> f32 {
        self.weight.load(Ordering::Relaxed)
    }

    /// Set weight directly (for initialization or testing).
    #[inline]
    pub fn set_weight(&self, w: f32) {
        self.weight.store(w.clamp(self.stdp.w_min, self.stdp.w_max), Ordering::Relaxed);
    }

    /// Check if a spike passes all dendritic filters.
    pub fn passes_filters(&self, spike: &Spike) -> bool {
        self.filters.iter().all(|f| f.passes(spike))
    }

    /// Record a pre-synaptic spike and apply STDP.
    /// Returns the weight delta applied.
    pub fn on_pre_spike(&self, timestamp_ns: u64) -> f32 {
        self.last_pre_ns.store(timestamp_ns, Ordering::Relaxed);

        // Check if there was a recent post-synaptic spike.
        let last_post = self.last_post_ns.load(Ordering::Relaxed);
        if last_post == u64::MAX {
            return 0.0;
        }

        // Pre after post -> depression (weaken).
        let dt = timestamp_ns as f64 - last_post as f64;
        if dt > 0.0 {
            let delta = -self.stdp.a_minus * (-dt / self.stdp.tau_minus_ns).exp() as f32;
            self.apply_weight_delta(delta);
            return delta;
        }

        0.0
    }

    /// Record a post-synaptic spike and apply STDP.
    /// Returns the weight delta applied.
    pub fn on_post_spike(&self, timestamp_ns: u64) -> f32 {
        self.last_post_ns.store(timestamp_ns, Ordering::Relaxed);

        // Check if there was a recent pre-synaptic spike.
        let last_pre = self.last_pre_ns.load(Ordering::Relaxed);
        if last_pre == u64::MAX {
            return 0.0;
        }

        // Post after pre -> potentiation (strengthen).
        let dt = timestamp_ns as f64 - last_pre as f64;
        if dt > 0.0 {
            let delta = self.stdp.a_plus * (-dt / self.stdp.tau_plus_ns).exp() as f32;
            self.apply_weight_delta(delta);
            return delta;
        }

        0.0
    }

    fn apply_weight_delta(&self, delta: f32) {
        let old = self.weight.fetch_add(delta, Ordering::Relaxed);
        let new = old + delta;
        // Clamp to [w_min, w_max].
        if new > self.stdp.w_max {
            self.weight.store(self.stdp.w_max, Ordering::Relaxed);
        } else if new < self.stdp.w_min {
            self.weight.store(self.stdp.w_min, Ordering::Relaxed);
        }
    }

    /// Returns true if the weight has fallen below the pruning threshold.
    pub fn should_prune(&self) -> bool {
        self.weight() <= self.stdp.w_min
    }
}

// ─── SynapseTable ───────────────────────────────────────────────────────────

/// Key for synapse lookup: (pre, post).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SynapseKey {
    pub pre: NeuronId,
    pub post: NeuronId,
}

/// Efficient storage for synapses. Uses dense array for small networks,
/// HashMap for larger ones.
pub enum SynapseTable {
    /// Flat array indexed by (pre * max_neurons + post). O(1) lookup.
    /// Used when max_neurons <= 1024.
    Dense {
        synapses: Vec<Option<Synapse>>,
        max_neurons: usize,
    },
    /// HashMap-based. Used for larger networks.
    Sparse {
        synapses: HashMap<SynapseKey, Synapse>,
    },
}

impl SynapseTable {
    /// Create a dense table for up to `max_neurons` neurons.
    pub fn dense(max_neurons: usize) -> Self {
        assert!(max_neurons <= 1024, "dense table limited to 1024 neurons");
        let size = max_neurons * max_neurons;
        let mut synapses = Vec::with_capacity(size);
        synapses.resize_with(size, || None);
        SynapseTable::Dense {
            synapses,
            max_neurons,
        }
    }

    /// Create a sparse table.
    pub fn sparse() -> Self {
        SynapseTable::Sparse {
            synapses: HashMap::new(),
        }
    }

    /// Insert a synapse.
    pub fn insert(&mut self, synapse: Synapse) {
        match self {
            SynapseTable::Dense {
                synapses,
                max_neurons,
            } => {
                let idx = synapse.pre.0 as usize * *max_neurons + synapse.post.0 as usize;
                if idx < synapses.len() {
                    synapses[idx] = Some(synapse);
                }
            }
            SynapseTable::Sparse { synapses } => {
                let key = SynapseKey {
                    pre: synapse.pre,
                    post: synapse.post,
                };
                synapses.insert(key, synapse);
            }
        }
    }

    /// Get a synapse by (pre, post).
    pub fn get(&self, pre: NeuronId, post: NeuronId) -> Option<&Synapse> {
        match self {
            SynapseTable::Dense {
                synapses,
                max_neurons,
            } => {
                let idx = pre.0 as usize * *max_neurons + post.0 as usize;
                synapses.get(idx).and_then(|s| s.as_ref())
            }
            SynapseTable::Sparse { synapses } => {
                synapses.get(&SynapseKey { pre, post })
            }
        }
    }

    /// Get all outgoing synapses from a neuron.
    pub fn outgoing(&self, pre: NeuronId) -> Vec<&Synapse> {
        match self {
            SynapseTable::Dense {
                synapses,
                max_neurons,
            } => {
                let start = pre.0 as usize * *max_neurons;
                let end = start + *max_neurons;
                synapses[start..end]
                    .iter()
                    .filter_map(|s| s.as_ref())
                    .collect()
            }
            SynapseTable::Sparse { synapses } => {
                synapses
                    .values()
                    .filter(|s| s.pre == pre)
                    .collect()
            }
        }
    }

    /// Remove synapses that should be pruned (weight below threshold).
    pub fn prune(&mut self) -> usize {
        let mut pruned = 0;
        match self {
            SynapseTable::Dense { synapses, .. } => {
                for slot in synapses.iter_mut() {
                    if let Some(s) = slot {
                        if s.should_prune() {
                            *slot = None;
                            pruned += 1;
                        }
                    }
                }
            }
            SynapseTable::Sparse { synapses } => {
                synapses.retain(|_, s| {
                    if s.should_prune() {
                        pruned += 1;
                        false
                    } else {
                        true
                    }
                });
            }
        }
        pruned
    }

    /// Total number of synapses.
    pub fn len(&self) -> usize {
        match self {
            SynapseTable::Dense { synapses, .. } => {
                synapses.iter().filter(|s| s.is_some()).count()
            }
            SynapseTable::Sparse { synapses } => synapses.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── Synapse Builder ────────────────────────────────────────────────────────

/// Builder for constructing synapses with a fluent API.
pub struct SynapseBuilder {
    pre: NeuronId,
    post: NeuronId,
    weight: f32,
    filters: Vec<DendriticFilter>,
    stdp: StdpParams,
}

impl SynapseBuilder {
    pub fn new(pre: NeuronId, post: NeuronId) -> Self {
        SynapseBuilder {
            pre,
            post,
            weight: 0.5,
            filters: Vec::new(),
            stdp: StdpParams::default(),
        }
    }

    pub fn weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    pub fn filter(mut self, f: DendriticFilter) -> Self {
        self.filters.push(f);
        self
    }

    pub fn stdp(mut self, params: StdpParams) -> Self {
        self.stdp = params;
        self
    }

    pub fn build(self) -> Synapse {
        Synapse::new(self.pre, self.post, self.weight)
            .with_filters(self.filters)
            .with_stdp(self.stdp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_f32_operations() {
        let a = AtomicF32::new(1.0);
        assert_eq!(a.load(Ordering::Relaxed), 1.0);

        a.store(2.5, Ordering::Relaxed);
        assert_eq!(a.load(Ordering::Relaxed), 2.5);

        let old = a.fetch_add(0.5, Ordering::Relaxed);
        assert_eq!(old, 2.5);
        assert_eq!(a.load(Ordering::Relaxed), 3.0);
    }

    #[test]
    fn stdp_potentiation() {
        // Pre fires at t=0, post fires at t=5ms -> should strengthen.
        let synapse = Synapse::new(NeuronId(0), NeuronId(1), 0.5);
        synapse.on_pre_spike(0);
        let delta = synapse.on_post_spike(5_000_000); // 5ms later
        assert!(delta > 0.0, "potentiation should increase weight");
        assert!(synapse.weight() > 0.5);
    }

    #[test]
    fn stdp_depression() {
        // Post fires at t=0, pre fires at t=5ms -> should weaken.
        let synapse = Synapse::new(NeuronId(0), NeuronId(1), 0.5);
        synapse.on_post_spike(0);
        let delta = synapse.on_pre_spike(5_000_000); // 5ms later
        assert!(delta < 0.0, "depression should decrease weight");
        assert!(synapse.weight() < 0.5);
    }

    #[test]
    fn stdp_asymmetry() {
        // Depression should be stronger than potentiation (a_minus > a_plus).
        let s1 = Synapse::new(NeuronId(0), NeuronId(1), 0.5);
        s1.on_pre_spike(0);
        let pot = s1.on_post_spike(5_000_000);

        let s2 = Synapse::new(NeuronId(0), NeuronId(1), 0.5);
        s2.on_post_spike(0);
        let dep = s2.on_pre_spike(5_000_000);

        assert!(dep.abs() > pot.abs(), "depression ({dep}) should be stronger than potentiation ({pot})");
    }

    #[test]
    fn weight_clamping() {
        let synapse = Synapse::new(NeuronId(0), NeuronId(1), 0.99);
        // Apply many potentiation events to try exceeding w_max.
        for i in 0..100 {
            synapse.on_pre_spike(i * 1_000_000);
            synapse.on_post_spike(i * 1_000_000 + 1_000_000);
        }
        assert!(synapse.weight() <= synapse.stdp.w_max);
    }

    #[test]
    fn dendritic_filter_type() {
        let filter = DendriticFilter::TypeFilter(SpikeType::Excitatory);
        let mut spike = Spike::zeroed();
        spike.set_spike_type(SpikeType::Excitatory);
        assert!(filter.passes(&spike));

        spike.set_spike_type(SpikeType::Inhibitory);
        assert!(!filter.passes(&spike));
    }

    #[test]
    fn dendritic_filter_priority() {
        let filter = DendriticFilter::PriorityThreshold(128);
        let mut spike = Spike::zeroed();
        spike.priority = 200;
        assert!(filter.passes(&spike));

        spike.priority = 50;
        assert!(!filter.passes(&spike));
    }

    #[test]
    fn dendritic_filter_payload_prefix() {
        let filter = DendriticFilter::PayloadPrefix(b"CMD:".to_vec());
        let mut spike = Spike::zeroed();
        spike.set_payload(b"CMD:run");
        assert!(filter.passes(&spike));

        spike.set_payload(b"DATA:foo");
        assert!(!filter.passes(&spike));
    }

    #[test]
    fn synapse_table_dense() {
        let mut table = SynapseTable::dense(16);
        table.insert(Synapse::new(NeuronId(0), NeuronId(1), 0.5));
        table.insert(Synapse::new(NeuronId(0), NeuronId(2), 0.3));
        table.insert(Synapse::new(NeuronId(1), NeuronId(0), 0.7));

        assert_eq!(table.len(), 3);
        assert!((table.get(NeuronId(0), NeuronId(1)).unwrap().weight() - 0.5).abs() < f32::EPSILON);
        assert!(table.get(NeuronId(2), NeuronId(0)).is_none());

        let outgoing = table.outgoing(NeuronId(0));
        assert_eq!(outgoing.len(), 2);
    }

    #[test]
    fn synapse_table_sparse() {
        let mut table = SynapseTable::sparse();
        table.insert(Synapse::new(NeuronId(100), NeuronId(200), 0.5));
        table.insert(Synapse::new(NeuronId(100), NeuronId(300), 0.3));

        assert_eq!(table.len(), 2);
        assert!(table.get(NeuronId(100), NeuronId(200)).is_some());
    }

    #[test]
    fn synapse_pruning() {
        let mut table = SynapseTable::dense(8);
        let s = Synapse::new(NeuronId(0), NeuronId(1), 0.0005); // Below w_min
        table.insert(s);
        table.insert(Synapse::new(NeuronId(0), NeuronId(2), 0.5));

        let pruned = table.prune();
        assert_eq!(pruned, 1);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn synapse_builder_api() {
        let synapse = SynapseBuilder::new(NeuronId(0), NeuronId(1))
            .weight(0.8)
            .filter(DendriticFilter::PriorityThreshold(10))
            .build();

        assert!((synapse.weight() - 0.8).abs() < f32::EPSILON);
        assert_eq!(synapse.filters.len(), 1);
    }
}
