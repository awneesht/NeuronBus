# NeuronBus - Project Conventions

## Architecture
Neuroscience-inspired messaging library for AI agents. Dependency graph:
- Level 0: spike.rs, wait.rs (no deps)
- Level 1: axon.rs, arena.rs (depend on spike)
- Level 2: synapse.rs (depends on spike)
- Level 3: neuron.rs (depends on all above)
- Level 4: cortex.rs, transport.rs (depends on all above)
- Level 5: lib.rs (public API)

## Conventions
- All unsafe code must have `// SAFETY:` comments
- Cache-line alignment (64 bytes) for hot data
- CachePadded (128 bytes) for atomics to prevent false sharing
- No allocations on the hot path (publish/consume spikes)
- Power-of-2 sizes for ring buffers (bitmask instead of modulo)
- Ordering: Release for writers, Acquire for readers, Relaxed only where proven safe

## Commands
- Build: `cargo build`
- Test: `cargo test`
- Bench: `cargo bench`
- Examples: `cargo run --example ping_pong`, `cargo run --example swarm`
- Miri: `cargo +nightly miri test` (optional UB check)

## Naming
- Biological terms: Spike (message), Axon (ring buffer), Synapse (connection), Neuron (agent), Cortex (orchestrator), Arena (shared memory)
