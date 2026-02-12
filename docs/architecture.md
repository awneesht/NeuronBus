# NeuronBus Architecture & Design

## 1. Overview

NeuronBus is a Rust library that fuses neuroscience-inspired communication patterns with systems-level performance optimizations to deliver ultra-fast messaging for AI agents. The core insight is that biological neural networks have evolved highly efficient, adaptive communication patterns that map surprisingly well onto high-performance computing primitives.

| Brain Concept | Computing Primitive | NeuronBus Component |
|---|---|---|
| Action potential | Fixed-size message | `Spike` (64 bytes) |
| Axon terminal | Ring buffer | `Axon` (LMAX Disruptor) |
| Synapse | Weighted connection | `Synapse` (STDP learning) |
| Neuron cell body | Agent process | `Neuron` (integrate-and-fire) |
| Brain region | Topic/namespace | `Region` (hierarchical routing) |
| Cerebral cortex | Orchestrator | `Cortex` (topology manager) |
| Extracellular fluid | Shared memory | `Arena` (mmap bump allocator) |

## 2. Design Principles

### 2.1 Zero Allocation on the Hot Path

The critical publish/consume path performs **zero heap allocations**. Spikes are written directly into pre-allocated ring buffer slots via raw pointer writes. This eliminates GC pauses (not applicable in Rust, but relevant for FFI consumers) and allocator contention.

### 2.2 Mechanical Sympathy

Every data structure is designed to work *with* the hardware, not against it:

- **Cache-line alignment**: `Spike` is exactly 64 bytes (`#[repr(C, align(64))]`), matching the L1 cache line size on x86 and ARM. One spike = one cache line fetch.
- **False sharing prevention**: Atomic cursors are wrapped in `CachePadded` (128 bytes) so that producer and consumer cursors never share a cache line.
- **Power-of-2 sizing**: Ring buffer capacity is always a power of 2, enabling bitmask indexing (`index = sequence & mask`) instead of expensive modulo division.
- **Sequential memory access**: Ring buffers are accessed sequentially, maximizing hardware prefetcher effectiveness.

### 2.3 Lock-Free by Default

All hot-path operations use lock-free algorithms:
- **Axon publish**: Single `AtomicU64::store(Release)` — not even a CAS
- **Axon consume**: `AtomicU64::load(Acquire)` + `AtomicU64::store(Release)`
- **Arena allocate**: `AtomicU64::fetch_add(Relaxed)` — lock-free bump pointer
- **Synapse weights**: `AtomicF32` via CAS loop on `AtomicU32` bit representation

Locks (`parking_lot::Mutex`) are only used in cold paths: neuron registration, topology changes, transport queues.

### 2.4 Adaptive Routing via STDP

Unlike static pub/sub systems, NeuronBus connections **learn**. Spike-Timing-Dependent Plasticity (STDP) adjusts synapse weights based on observed communication patterns, allowing the system to self-optimize its routing topology.

## 3. Module Dependency Graph

```
Level 0 (no deps):     spike.rs    wait.rs
                          │           │
Level 1 (spike):        axon.rs    arena.rs
                          │           │
Level 2 (spike):       synapse.rs ───┘
                          │
Level 3 (all above):   neuron.rs
                        ╱       ╲
Level 4:          cortex.rs   transport.rs
                        ╲       ╱
Level 5:             lib.rs
```

Each module only depends on modules from lower levels, forming a clean DAG with no circular dependencies.

## 4. Module Deep Dives

### 4.1 Spike (`src/spike.rs`)

The fundamental unit of communication. A spike is a 64-byte, cache-line-aligned, fixed-layout binary message.

**Memory Layout:**

```
Offset  Size  Field          Description
──────  ────  ─────          ───────────
0       8B    timestamp_ns   Nanosecond timestamp (quanta TSC)
8       4B    sequence       Monotonic per-source counter
12      2B    source         Sender NeuronId
14      2B    target         Receiver NeuronId (0xFFFF = broadcast)
16      1B    spike_type     Excitatory(0) / Inhibitory(1) / Modulatory(2)
17      1B    priority       0-255 priority level
18      46B   payload        Inline data OR ArenaRef for large payloads
──────  ────
Total:  64B   (= 1 cache line)
```

**Design decisions:**

- **Fixed size**: No length prefix, no serialization, no allocation. Spikes can be `memcpy`'d, sent over the wire as raw bytes, and reinterpreted with zero overhead.
- **`#[repr(C)]`**: Guarantees field ordering matches the layout above, enabling `as_bytes()` / `from_bytes()` zero-copy conversions.
- **`align(64)`**: Ensures each spike occupies exactly one cache line. When stored in a ring buffer, accessing spike N never pulls in spike N+1's data.
- **46-byte payload**: Enough for most agent messages (commands, small JSON, protobuf). For larger payloads, the first byte is set to `0xA0` (marker) followed by a 16-byte `ArenaRef` pointing to shared memory.

**Spike Types** mirror neurotransmitter categories:

| Type | Biological Analog | Effect on Target |
|---|---|---|
| Excitatory | Glutamate | Increases membrane potential |
| Inhibitory | GABA | Decreases membrane potential |
| Modulatory | Dopamine | Modulates synapse behavior |

### 4.2 Wait Strategies (`src/wait.rs`)

Control how consumers wait when the ring buffer is empty. Inspired by the LMAX Disruptor's `WaitStrategy` interface.

| Strategy | Mechanism | Latency | CPU Usage | Use Case |
|---|---|---|---|---|
| `BusySpin` | Tight loop on atomic load | <10ns | 100% core | Latency-critical, dedicated core |
| `SpinLoopHint` | `hint::spin_loop()` (PAUSE/YIELD) | ~10-50ns | High but yielding | Good default balance |
| `YieldThenPark` | Spin N times, then `thread::yield_now()` | ~100ns-1us | Low | Background consumers |

The `wait_for()` method blocks until the cursor advances past a given value. `try_wait()` provides non-blocking polling.

### 4.3 Axon (`src/axon.rs`) — The Critical Path

The performance-critical ring buffer. Implements a Single-Producer Multi-Consumer (SPMC) variant of the LMAX Disruptor pattern.

**Architecture:**

```
                    ┌─────────────────────────────────┐
                    │         AxonInner (shared)       │
                    │                                  │
  Producer          │  write_cursor: CachePadded<u64>  │
  (Axon)            │  ┌───┬───┬───┬───┬───┬───┬───┐  │
   │                │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │  │  ← 64-byte aligned
   │  publish()     │  │   │   │   │   │   │   │   │  │    Spike buffer
   └───────────►    │  └───┴───┴───┴───┴───┴───┴───┘  │
                    │                                  │
                    │  Consumer read cursors:          │
  Consumer A        │  reader_a: CachePadded<u64> ─────┤
  (AxonReader) ◄────┤  reader_b: CachePadded<u64> ─────┤
  Consumer B ◄──────┤                                  │
                    └─────────────────────────────────┘
```

**Key design decisions:**

1. **Single producer per axon**: Each neuron owns exactly one output axon. This means publish is a simple `atomic store(Release)` — no CAS, no retry loops, no contention. This is the single biggest performance win.

2. **Per-consumer read cursors**: Each `AxonReader` has its own `CachePadded<AtomicU64>` read position. Consumers never contend with each other. Cloning a reader creates an independent cursor starting at the clone's current position.

3. **Pre-allocated buffer**: The spike buffer is allocated once at construction with `alloc_zeroed` and 64-byte alignment. Buffer slots are written via `ptr::write`, avoiding any `Drop` overhead.

4. **Bitmask indexing**: `index = sequence & (capacity - 1)`. This compiles to a single AND instruction, vs. modulo which requires division.

5. **Batch publish**: `batch_publish()` writes N spikes with a single cursor update at the end, amortizing the atomic store cost.

**Publish hot path (assembly-level):**

```rust
pub fn publish(&mut self, spike: Spike) {
    let index = self.write_seq & self.inner.mask;  // AND instruction
    unsafe {
        let slot = self.inner.buffer.add(index);   // pointer arithmetic
        std::ptr::write(slot, spike);              // 64-byte memcpy
    }
    self.write_seq += 1;                           // local increment
    self.inner.write_cursor.store(                 // atomic store (Release)
        self.write_seq, Ordering::Release
    );
}
```

This is 4-5 instructions on the critical path. No branches, no allocations, no locks.

### 4.4 Arena (`src/arena.rs`)

Shared memory region for payloads exceeding 46 bytes. Uses mmap for zero-copy cross-process sharing.

**Memory Layout:**

```
┌──────────────────────────────────────────────┐
│ ArenaHeader (32B)                            │
│   alloc_offset: AtomicU64  (bump pointer)    │
│   capacity: u64                              │
│   epoch: AtomicU64                           │
│   ref_count_next: AtomicU32                  │
├──────────────────────────────────────────────┤
│ Reference Count Slots (256KB)                │
│   64K × AtomicU32 entries                    │
├──────────────────────────────────────────────┤
│ Data Region (user-specified capacity)        │
│   ┌─────┬─────┬─────┬────────────────────┐  │
│   │ A1  │ A2  │ A3  │     free space     │  │
│   └─────┴─────┴─────┴────────────────────┘  │
│                       ▲                      │
│                   alloc_offset               │
└──────────────────────────────────────────────┘
```

**Allocation algorithm:**

1. `fetch_add(aligned_len)` on the bump pointer — lock-free, O(1)
2. Allocate a ref count slot via `fetch_add(1)` on the slot counter
3. `memcpy` data into the allocated region
4. Return an `ArenaRef { offset, len, ref_slot }`

All allocations are 8-byte aligned. The bump allocator means allocation is wait-free (single atomic operation), but fragmentation is not handled — the arena is designed for epoch-based bulk reclamation via `reset()`.

**Variants:**

- **Anonymous**: `MmapMut::map_anon()` — in-process only
- **File-backed**: `MmapMut::map_mut(&file)` — cross-process sharing via filesystem path

### 4.5 Synapse (`src/synapse.rs`)

Adaptive connections implementing Spike-Timing-Dependent Plasticity (STDP).

**STDP Learning Rule:**

```
                    Potentiation
                    (strengthen)
        Δw          ╱╲
         │         ╱  ╲  a_plus = 0.01
         │        ╱    ╲
─────────┼───────╱──────╲────────── Δt (pre - post)
         │      ╱        ╲
         │     ╱  a_minus  ╲
         │    ╱   = 0.012   ╲
         │   ╱   Depression  ╲
         │  ╱    (weaken)     ╲
```

- **Pre fires before post** (Δt > 0): Weight increases by `a_plus × exp(-Δt/τ_plus)` — the connection is "useful" because the pre-synaptic neuron predicted the post-synaptic activation.
- **Post fires before pre** (Δt < 0): Weight decreases by `a_minus × exp(-Δt/τ_minus)` — the connection is "anti-causal" and should weaken.
- **Asymmetry** (`a_minus > a_plus`): Depression is slightly stronger than potentiation. This prevents all weights from saturating to maximum and maintains network sparsity.

**AtomicF32 Implementation:**

Since Rust's standard library doesn't provide `AtomicF32`, we implement it via CAS on the bit representation:

```rust
pub fn fetch_add(&self, delta: f32, order: Ordering) -> f32 {
    loop {
        let bits = self.bits.load(Relaxed);
        let new = f32::from_bits(bits) + delta;
        match self.bits.compare_exchange_weak(bits, new.to_bits(), order, Relaxed) {
            Ok(_) => return f32::from_bits(bits),
            Err(_) => continue,  // Retry on contention
        }
    }
}
```

**Dendritic Filters:**

Synapses can filter spikes before delivery:

| Filter | Description |
|---|---|
| `TypeFilter` | Only pass Excitatory/Inhibitory/Modulatory |
| `PriorityThreshold` | Only pass spikes with priority >= N |
| `PayloadPrefix` | Only pass spikes whose payload starts with a byte prefix |
| `Custom` | Named custom filter (evaluated externally) |

**SynapseTable Storage:**

| Network Size | Storage | Lookup | Memory |
|---|---|---|---|
| <= 1024 neurons | Dense (flat array) | O(1) indexed by `pre * N + post` | N^2 × sizeof(Synapse) |
| > 1024 neurons | Sparse (HashMap) | O(1) amortized | proportional to connections |

### 4.6 Neuron (`src/neuron.rs`)

The agent abstraction combining a trait interface with biological integrate-and-fire dynamics.

**Neuron Trait:**

```rust
pub trait Neuron: Send {
    fn on_spike(&mut self, spike: &Spike, handle: &NeuronHandle);
    fn tick(&mut self, handle: &NeuronHandle);
    fn capabilities(&self) -> &[&str] { &[] }
}
```

Agents implement this trait. `on_spike` is called for each delivered spike. `tick` is called each event loop iteration.

**Integrate-and-Fire Model:**

```
Potential
    ▲
    │          ╱╲  fire!
    │         ╱  │
  θ │────────╱───│──────── threshold
    │       ╱    │ refractory
    │      ╱     │  period
    │     ╱      │    │
    │    ╱ leak   │    │
    │   ╱  decay  │    │
    │──╱──────────┼────┼──────► time
    │             └────┘
  rest                    back to rest
```

1. **Integration**: Each incoming excitatory spike adds `weight` to the membrane potential. Inhibitory spikes subtract.
2. **Leak**: Each tick, potential decays toward rest: `potential = rest + (potential - rest) × leak_rate`
3. **Threshold**: When `potential >= threshold`, the neuron "fires" (triggers agent logic).
4. **Refractory period**: After firing, the neuron ignores input for N ticks. This provides natural backpressure — an overwhelmed agent simply stops accepting new work.

**NeuronHandle:**

The handle provides the agent's interface to the bus:
- `fire(spike)`: Publish a spike to the output axon (auto-stamps source ID, sequence, timestamp)
- `fire_to(target, payload)`: Convenience for simple excitatory spikes
- `broadcast(payload)`: Fire to all connected neurons
- `arena()`: Access shared memory for large payloads
- `reader()`: Get a read handle on this neuron's output axon

**AgentCard:**

Inspired by Google A2A's agent cards, provides capability-based discovery:

```rust
AgentCard {
    name: "translator",
    id: NeuronId(5),
    capabilities: ["translate", "summarize"],
    regions: ["cortex/language"],
}
```

### 4.7 Cortex (`src/cortex.rs`)

The topology manager and event loop orchestrator.

**Responsibilities:**

1. **Neuron lifecycle**: Register/manage agents, assign IDs, create output axons
2. **Synapse management**: Create/prune connections, maintain the SynapseTable
3. **Spike routing**: Collect output spikes, apply synapse filters/weights, deliver to targets
4. **STDP updates**: Record spike timing, update synapse weights
5. **Agent discovery**: Find agents by capability or region
6. **Event loop**: The `tick()` method drives the entire system

**Event Loop (single tick):**

```
┌─────────────────────────────────────────────────┐
│ 1. COLLECT: Read all pending spikes from every   │
│    neuron's output axon (non-blocking try_read)  │
├─────────────────────────────────────────────────┤
│ 2. ROUTE: For each collected spike:             │
│    a. Look up synapse(source, target)            │
│    b. Apply dendritic filters                    │
│    c. Record pre-spike for STDP                  │
│    d. Integrate spike into target's IF state     │
│    e. Call target agent's on_spike()             │
├─────────────────────────────────────────────────┤
│ 3. TICK: For each neuron:                        │
│    a. Apply leak decay                           │
│    b. Check if potential >= threshold (fire?)     │
│    c. If fired: record post-spike STDP,          │
│       enter refractory, call agent tick()        │
└─────────────────────────────────────────────────┘
```

**Region-Based Routing:**

Regions are hierarchical strings (like filesystem paths):

```
cortex/
├── language/
│   ├── neuron_0
│   └── neuron_1
├── vision/
│   └── neuron_2
└── motor/
    └── neuron_3
```

`Region::is_parent_of()` enables hierarchical matching. Discovery can find all agents in `cortex/language` or all agents under `cortex/`.

### 4.8 Transport (`src/transport.rs`)

Network layer for cross-process and cross-machine communication.

**Transport Trait:**

```rust
pub trait Transport: Send {
    fn send(&mut self, spike: &Spike) -> Result<(), TransportError>;
    fn recv(&mut self) -> Result<Spike, TransportError>;
    fn try_recv(&mut self) -> Result<Option<Spike>, TransportError>;
}
```

**LocalTransport:**

Uses a `Mutex<VecDeque<Spike>>` shared queue. Zero network overhead, useful for in-process communication between threads that don't share an axon.

**TcpTransport:**

Custom binary framing protocol:

```
┌──────────────┬────────────────────────────────────┐
│ Length (4B LE)│         Spike Data (64B)           │
└──────────────┴────────────────────────────────────┘
```

- Length is always 64 (for a bare spike)
- TCP_NODELAY enabled to minimize latency
- Spikes are copied byte-by-byte to/from the wire via `as_bytes()` / `ptr::copy_nonoverlapping`

**Platform-specific I/O (future):**
- Linux: `io_uring` for kernel-side polling (dependency included)
- macOS: `kqueue` via `mio` (dependency included)

Currently uses blocking std TCP; the platform-specific async backends are available as dependencies for future integration.

## 5. Memory Ordering Model

NeuronBus uses a carefully chosen memory ordering strategy:

| Operation | Ordering | Rationale |
|---|---|---|
| Axon publish (write cursor) | `Release` | Ensures spike data is visible before cursor advances |
| Axon consume (read write cursor) | `Acquire` | Pairs with producer's Release to see spike data |
| Axon consume (update read cursor) | `Release` | Not strictly necessary but prevents stale reads |
| Arena bump allocator | `Relaxed` | Each allocation gets unique offset; no ordering needed |
| Arena ref count init | `Release` | Ensures ref count is visible before ArenaRef is shared |
| Arena ref count decrement | `Release` | Ensures all prior accesses complete before potential dealloc |
| Synapse weight (AtomicF32) | `Relaxed` | Approximate weights are fine; no need for strong ordering |
| STDP timestamps | `Relaxed` | Approximate timing is sufficient for learning |

## 6. Safety and Correctness

### Unsafe Code Inventory

All unsafe blocks have `// SAFETY:` comments. The unsafe operations are:

1. **Axon buffer**: Raw `alloc_zeroed` / `dealloc` / `ptr::write` / `ptr::read` for the pre-allocated spike array. Safety relies on:
   - Single producer (no concurrent writes to same slot)
   - Acquire/Release ordering (data visible before cursor advances)
   - Bitmask indexing (always in bounds)

2. **Arena data access**: Raw pointer reads/writes into the mmap region. Safety relies on:
   - Atomic bump allocator (unique offsets per allocation)
   - Bounds checking on read

3. **Spike byte conversion**: `as_bytes()` / `from_bytes()` reinterpret casts. Safety relies on:
   - `#[repr(C)]` layout
   - All bit patterns valid for the field types

### Thread Safety

- `Axon` (producer) is `!Sync` — only one thread can publish
- `AxonReader` is `Send + !Sync` — each consumer runs on its own thread
- `AxonInner` is `Send + Sync` — shared immutable state + atomics
- `Arena` is `Send + Sync` — all mutations via atomics
- `Cortex` is `Send + !Sync` — single-threaded event loop

## 7. Error Handling Strategy

- **Hot path (axon publish/consume)**: No error returns. Panics only on programmer error (non-power-of-2 capacity). Ring buffer overwrite is by design (newest data overwrites oldest).
- **Arena**: Returns `Result<_, ArenaError>` for out-of-space and invalid reference conditions.
- **Transport**: Returns `Result<_, TransportError>` for I/O errors and protocol violations.
- **Cortex**: Silently drops spikes to unregistered neurons (defensive, no panic on stale IDs).

## 8. Extension Points

NeuronBus is designed for extensibility:

1. **Custom Neuron implementations**: Any `Neuron` trait impl can participate
2. **Custom DendriticFilters**: The `Custom(String)` variant allows external filter logic
3. **Custom Transports**: Implement the `Transport` trait for any network protocol
4. **Synapse table strategies**: Dense or Sparse, selectable at construction
5. **Wait strategies**: Pluggable consumer wait behavior
6. **File-backed arenas**: Cross-process shared memory via filesystem
