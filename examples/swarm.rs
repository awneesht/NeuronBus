//! Swarm example: 100 neurons across 5 regions with STDP weight evolution.
//! Demonstrates adaptive routing where frequently-used paths strengthen.

use neuronbus::*;
use std::sync::{Arc, Mutex};

struct SwarmNeuron {
    name: String,
    spike_count: Arc<Mutex<u32>>,
}

impl Neuron for SwarmNeuron {
    fn on_spike(&mut self, _spike: &Spike, _handle: &NeuronHandle) {
        *self.spike_count.lock().unwrap() += 1;
    }

    fn tick(&mut self, _handle: &NeuronHandle) {}

    fn capabilities(&self) -> &[&str] {
        &["swarm"]
    }
}

fn main() {
    println!("=== NeuronBus Swarm: STDP Weight Evolution ===\n");

    let num_neurons = 100;
    let num_regions = 5;
    let neurons_per_region = num_neurons / num_regions;

    let mut bus = NeuronBus::builder()
        .arena_capacity(1024 * 1024)
        .axon_capacity(1 << 14)
        .max_neurons(128)
        .build();

    let regions = vec![
        "cortex/language",
        "cortex/vision",
        "cortex/motor",
        "cortex/memory",
        "cortex/reasoning",
    ];

    // Register neurons.
    let mut neuron_ids = Vec::new();
    let mut spike_counts = Vec::new();

    for (r, region_name) in regions.iter().enumerate() {
        for j in 0..neurons_per_region {
            let count = Arc::new(Mutex::new(0u32));
            let neuron = SwarmNeuron {
                name: format!("{region_name}/{j}"),
                spike_count: Arc::clone(&count),
            };
            spike_counts.push(count);

            let card = AgentCard::new(
                format!("neuron_{}_{}", r, j),
                NeuronId(0),
            )
            .with_capabilities(vec!["swarm".into()])
            .with_regions(vec![region_name.to_string()]);

            let id = bus.register(Box::new(neuron), card);
            neuron_ids.push(id);
        }
    }

    println!("Registered {} neurons in {} regions", neuron_ids.len(), num_regions);

    // Create connections: within regions (dense) and between regions (sparse).
    let mut connection_count = 0;
    for r in 0..num_regions {
        let start = r * neurons_per_region;
        let end = start + neurons_per_region;

        // Intra-region: connect each neuron to the next (ring topology).
        for i in start..end {
            let next = start + (i - start + 1) % neurons_per_region;
            bus.connect(neuron_ids[i], neuron_ids[next]).weight(0.5).build();
            connection_count += 1;
        }

        // Inter-region: connect first neuron of each region to first of next region.
        let next_region = (r + 1) % num_regions;
        let next_start = next_region * neurons_per_region;
        bus.connect(neuron_ids[start], neuron_ids[next_start]).weight(0.3).build();
        connection_count += 1;
    }

    println!("Created {connection_count} synapses\n");

    // Phase 1: Stimulate language region heavily.
    println!("Phase 1: Stimulating language region...");
    let lang_start = 0;
    for round in 0..50 {
        for i in 0..neurons_per_region {
            let spike = SpikeBuilder::new()
                .target(neuron_ids[lang_start + (i + 1) % neurons_per_region])
                .spike_type(SpikeType::Excitatory)
                .sequence(round * neurons_per_region as u32 + i as u32)
                .payload(b"stim")
                .build();
            bus.fire(neuron_ids[lang_start + i], spike);
        }
        bus.tick();
    }

    // Print spike counts per region.
    println!("\nSpike counts by region after stimulation:");
    for (r, region_name) in regions.iter().enumerate() {
        let start = r * neurons_per_region;
        let end = start + neurons_per_region;
        let total: u32 = spike_counts[start..end]
            .iter()
            .map(|c| *c.lock().unwrap())
            .sum();
        println!("  {region_name}: {total} spikes received");
    }

    // Phase 2: Check STDP weight evolution.
    println!("\nPhase 2: Checking synapse weights after STDP...");

    // Sample some synapses to show weight evolution.
    // The language region synapses should be stronger due to repeated activation.
    println!("\nSample synapse weights (language region, intra-region ring):");
    for i in 0..5.min(neurons_per_region) {
        let pre = neuron_ids[i];
        let post = neuron_ids[(i + 1) % neurons_per_region];
        if let Some(s) = bus.cortex.synapses.get(pre, post) {
            let w: f32 = s.weight();
            println!("  {} -> {}: weight = {:.4}", pre.0, post.0, w);
        }
    }

    println!("\nSample synapse weights (inter-region bridges):");
    for r in 0..num_regions {
        let start = r * neurons_per_region;
        let next_region = (r + 1) % num_regions;
        let next_start = next_region * neurons_per_region;
        let pre = neuron_ids[start];
        let post = neuron_ids[next_start];
        if let Some(s) = bus.cortex.synapses.get(pre, post) {
            let w: f32 = s.weight();
            println!(
                "  {} ({}) -> {} ({}): weight = {:.4}",
                pre.0, regions[r], post.0, regions[next_region], w
            );
        }
    }

    // Discovery test.
    println!("\nAgent discovery:");
    let swarm_agents = bus.discover("swarm");
    println!("  Found {} agents with 'swarm' capability", swarm_agents.len());

    let lang_agents = bus.cortex.discover_in_region(&Region::new("cortex/language"));
    println!("  Found {} agents in 'cortex/language'", lang_agents.len());

    println!("\n=== Done ===");
}
