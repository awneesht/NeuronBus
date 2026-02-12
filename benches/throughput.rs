use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use neuronbus::*;

fn spike_creation(c: &mut Criterion) {
    c.bench_function("spike_create", |b| {
        b.iter(|| {
            black_box(SpikeBuilder::new()
                .source(NeuronId(1))
                .target(NeuronId(2))
                .spike_type(SpikeType::Excitatory)
                .priority(128)
                .payload(b"bench payload")
                .build())
        })
    });
}

fn axon_spsc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("axon_spsc");
    let count = 1_000_000u64;
    group.throughput(Throughput::Elements(count));

    group.bench_function("1M_spikes", |b| {
        b.iter(|| {
            let mut axon = Axon::new(1 << 20);
            let reader = axon.reader();

            for i in 0..count as u32 {
                let spike = SpikeBuilder::new().sequence(i).build();
                axon.publish(spike);
            }

            for _ in 0..count {
                black_box(reader.try_read_next().unwrap());
            }
        })
    });

    group.finish();
}

fn axon_spmc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("axon_spmc");
    let count = 100_000u64;
    let num_consumers = 8;
    group.throughput(Throughput::Elements(count * num_consumers));

    group.bench_function("8_consumers", |b| {
        b.iter(|| {
            let mut axon = Axon::new(1 << 20);
            let readers: Vec<_> = (0..num_consumers).map(|_| axon.reader()).collect();

            // Publish all spikes.
            for i in 0..count as u32 {
                let spike = SpikeBuilder::new().sequence(i).build();
                axon.publish(spike);
            }

            // Each consumer reads all spikes.
            for reader in &readers {
                for _ in 0..count {
                    black_box(reader.try_read_next().unwrap());
                }
            }
        })
    });

    group.finish();
}

fn arena_alloc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_alloc");
    let count = 10_000u64;
    group.throughput(Throughput::Elements(count));

    group.bench_function("10K_allocs", |b| {
        b.iter(|| {
            let arena = Arena::new(1 << 24).unwrap(); // 16MB
            for _ in 0..count {
                black_box(arena.write(b"benchmark payload data for arena").unwrap());
            }
        })
    });

    group.finish();
}

fn batch_vs_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("publish_mode");
    let count = 10_000;

    group.bench_function("individual", |b| {
        b.iter(|| {
            let mut axon = Axon::new(1 << 16);
            for i in 0..count {
                let spike = SpikeBuilder::new().sequence(i).build();
                axon.publish(spike);
            }
        })
    });

    group.bench_function("batch", |b| {
        b.iter(|| {
            let mut axon = Axon::new(1 << 16);
            let spikes: Vec<Spike> = (0..count)
                .map(|i| SpikeBuilder::new().sequence(i).build())
                .collect();
            axon.batch_publish(&spikes);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    spike_creation,
    axon_spsc_throughput,
    axon_spmc_throughput,
    arena_alloc_throughput,
    batch_vs_individual,
);
criterion_main!(benches);
