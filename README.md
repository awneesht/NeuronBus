# NeuronBus

Neuroscience-inspired ultra-fast messaging library for AI agents in Rust.

NeuronBus models itself on biological neural networks to achieve millions of messages per second with adaptive, brain-inspired routing. No serialization overhead, no network brokers on the hot path — just lock-free ring buffers, shared memory, and spike-timing-dependent plasticity.

## Architecture

| Biological Concept | NeuronBus Component | Description |
|---|---|---|
| Action Potential | **Spike** | 64-byte cache-line-aligned binary message |
| Axon | **Axon** | LMAX Disruptor-style lock-free SPMC ring buffer |
| Synapse | **Synapse** | Adaptive connection with STDP weight learning |
| Neuron | **Neuron** | AI agent with integrate-and-fire batching |
| Brain Region | **Cortex** | Topology manager with region-based routing |
| Extracellular Space | **Arena** | mmap-backed shared memory for large payloads |

```
┌─────────────────────────────────────────────────┐
│                    Cortex                        │
│  ┌──────────┐  Synapse  ┌──────────┐            │
│  │ Neuron A ├──(w=0.8)──► Neuron B │            │
│  │  [Axon]  │   STDP    │  [Axon]  │            │
│  └────┬─────┘  learns   └────┬─────┘            │
│       │                      │                   │
│       └──────┬───────────────┘                   │
│              ▼                                   │
│         ┌─────────┐                              │
│         │  Arena   │  (shared memory, zero-copy) │
│         └─────────┘                              │
└─────────────────────────────────────────────────┘
```

## Key Features

- **Zero-copy messaging**: 64-byte spikes with no serialization — `as_bytes()` / `from_bytes()` directly
- **Lock-free ring buffers**: Single atomic store for publish, per-consumer read cursors, power-of-2 bitmask indexing
- **STDP adaptive routing**: Connections that work well get stronger; unused paths weaken and auto-prune
- **Integrate-and-fire backpressure**: Neurons accumulate potential from incoming spikes and enter refractory periods after firing
- **Shared memory arena**: mmap-backed bump allocator for payloads >46 bytes with reference counting
- **Region-based discovery**: Hierarchical topic routing and capability-based agent lookup
- **Network transport**: TCP with binary framing for cross-process communication

## Quick Start

```rust
use neuronbus::*;
use std::sync::{Arc, atomic::{AtomicU32, Ordering}};

struct MyAgent {
    received: Arc<AtomicU32>,
}

impl Neuron for MyAgent {
    fn on_spike(&mut self, spike: &Spike, _handle: &NeuronHandle) {
        self.received.fetch_add(1, Ordering::Relaxed);
        println!("Got spike: {:?}", &spike.payload[..5]);
    }
    fn tick(&mut self, _handle: &NeuronHandle) {}
}

fn main() {
    let mut bus = NeuronBus::builder()
        .arena_capacity(1024 * 1024)    // 1MB shared memory
        .axon_capacity(1 << 16)         // 64K ring buffer slots
        .max_neurons(64)
        .wait_strategy(WaitStrategyKind::SpinLoopHint)
        .build();

    let count = Arc::new(AtomicU32::new(0));

    let sender_id = bus.register(
        Box::new(MyAgent { received: Arc::new(AtomicU32::new(0)) }),
        AgentCard::new("sender", NeuronId(0))
            .with_capabilities(vec!["produce".into()]),
    );

    let receiver_id = bus.register(
        Box::new(MyAgent { received: count.clone() }),
        AgentCard::new("receiver", NeuronId(0))
            .with_capabilities(vec!["consume".into()]),
    );

    // Connect with adaptive synapse
    bus.connect(sender_id, receiver_id).weight(0.8).build();

    // Fire a spike
    let spike = SpikeBuilder::new()
        .target(receiver_id)
        .spike_type(SpikeType::Excitatory)
        .payload(b"hello")
        .build();
    bus.fire(sender_id, spike);

    // Process one tick
    bus.tick();

    assert_eq!(count.load(Ordering::Relaxed), 1);
}
```

## Benchmarks

Measured on Apple Silicon (release mode, criterion):

| Benchmark | Throughput |
|---|---|
| Spike creation | ~12.5ns per spike |
| Axon SPSC | ~98M spikes/sec |
| Axon SPMC (8 consumers) | ~109M elements/sec |
| Arena allocation | ~192M allocs/sec |

Run benchmarks yourself:

```bash
cargo bench
```

## Examples

**Ping-Pong** — Two neurons exchanging spikes, measures round-trip latency:

```bash
cargo run --example ping_pong
```

**Swarm** — 100 neurons across 5 brain regions with STDP weight evolution:

```bash
cargo run --example swarm
```

## Module Overview

| Module | Description |
|---|---|
| `spike` | 64-byte `#[repr(C, align(64))]` message with builder, ArenaRef encoding |
| `wait` | BusySpin, SpinLoopHint, YieldThenPark consumer wait strategies |
| `axon` | SPMC ring buffer — CachePadded cursors, batch publish, zero allocation |
| `arena` | mmap shared memory — atomic bump allocator, ref counting, epoch reset |
| `synapse` | STDP learning — AtomicF32 weights, dendritic filters, auto-pruning |
| `neuron` | Neuron trait, integrate-and-fire state, NeuronHandle, AgentCard |
| `cortex` | Topology manager — region routing, discovery, event-driven tick loop |
| `transport` | Local (shared queue) and TCP (binary framing) transports |

## How STDP Works

Spike-Timing-Dependent Plasticity adjusts synapse weights based on spike timing:

- **Pre fires before post** (causal) → weight increases (potentiation)
- **Post fires before pre** (anti-causal) → weight decreases (depression)
- Depression is slightly stronger than potentiation (`a_minus=0.012 > a_plus=0.01`) to prevent all-to-all saturation
- Weights below a minimum threshold trigger automatic synapse pruning

This means frequently-used, effective communication paths naturally strengthen while unused paths fade away.

## License

MIT
