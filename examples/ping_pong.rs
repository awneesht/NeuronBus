//! Ping-pong example: Two neurons exchanging spikes.
//! Demonstrates sub-microsecond round-trip latency.

use neuronbus::*;
use std::sync::{Arc, Mutex};

struct PingPongNeuron {
    #[allow(dead_code)]
    name: String,
    latencies: Arc<Mutex<Vec<u64>>>,
    clock: quanta::Clock,
}

impl Neuron for PingPongNeuron {
    fn on_spike(&mut self, spike: &Spike, _handle: &NeuronHandle) {
        let now = self.clock.raw();
        let latency = now.saturating_sub(spike.timestamp_ns);
        self.latencies.lock().unwrap().push(latency);
    }

    fn tick(&mut self, _handle: &NeuronHandle) {}

    fn capabilities(&self) -> &[&str] {
        &["ping_pong"]
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    println!("=== NeuronBus Ping-Pong Latency Test ===\n");

    let clock = quanta::Clock::new();
    let ping_latencies = Arc::new(Mutex::new(Vec::new()));
    let pong_latencies = Arc::new(Mutex::new(Vec::new()));

    let ping_neuron = PingPongNeuron {
        name: "Ping".into(),
        latencies: Arc::clone(&ping_latencies),
        clock: clock.clone(),
    };
    let pong_neuron = PingPongNeuron {
        name: "Pong".into(),
        latencies: Arc::clone(&pong_latencies),
        clock: clock.clone(),
    };

    let mut bus = NeuronBus::builder()
        .arena_capacity(1024 * 1024)
        .axon_capacity(1 << 16)
        .max_neurons(16)
        .build();

    let ping_id = bus.register(
        Box::new(ping_neuron),
        AgentCard::new("Ping", NeuronId(0))
            .with_capabilities(vec!["ping_pong".into()]),
    );
    let pong_id = bus.register(
        Box::new(pong_neuron),
        AgentCard::new("Pong", NeuronId(0))
            .with_capabilities(vec!["ping_pong".into()]),
    );

    // Connect bidirectionally.
    bus.connect(ping_id, pong_id).weight(1.0).build();
    bus.connect(pong_id, ping_id).weight(1.0).build();

    let rounds = 10_000;
    println!("Running {rounds} ping-pong rounds...\n");

    for i in 0..rounds {
        // Ping fires to pong.
        let spike = SpikeBuilder::new()
            .target(pong_id)
            .spike_type(SpikeType::Excitatory)
            .sequence(i)
            .payload(b"ping")
            .build();
        bus.fire(ping_id, spike);
        bus.tick();

        // Pong fires back to ping.
        let spike = SpikeBuilder::new()
            .target(ping_id)
            .spike_type(SpikeType::Excitatory)
            .sequence(i)
            .payload(b"pong")
            .build();
        bus.fire(pong_id, spike);
        bus.tick();
    }

    // Analyze latencies.
    let mut ping_lats = ping_latencies.lock().unwrap().clone();
    let mut pong_lats = pong_latencies.lock().unwrap().clone();
    ping_lats.sort();
    pong_lats.sort();

    println!("Pong -> Ping latencies ({} samples):", ping_lats.len());
    println!("  P50:  {} raw ticks", percentile(&ping_lats, 50.0));
    println!("  P99:  {} raw ticks", percentile(&ping_lats, 99.0));
    println!("  P999: {} raw ticks", percentile(&ping_lats, 99.9));

    println!("\nPing -> Pong latencies ({} samples):", pong_lats.len());
    println!("  P50:  {} raw ticks", percentile(&pong_lats, 50.0));
    println!("  P99:  {} raw ticks", percentile(&pong_lats, 99.0));
    println!("  P999: {} raw ticks", percentile(&pong_lats, 99.9));

    // Estimate nanoseconds (quanta raw ticks are ~ns on modern hardware).
    let total_roundtrips = ping_lats.len();
    let avg_ticks: f64 = ping_lats.iter().sum::<u64>() as f64 / total_roundtrips as f64;
    println!("\nAverage one-way latency: ~{:.0} raw ticks", avg_ticks);
    println!("Total roundtrips completed: {total_roundtrips}");
    println!("\n=== Done ===");
}
