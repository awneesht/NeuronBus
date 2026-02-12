//! Wait strategies for ring buffer consumers.
//!
//! Inspired by the LMAX Disruptor's wait strategies, these control how
//! consumers wait for new data when the ring buffer is empty.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

/// Strategy kind for builder API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitStrategyKind {
    BusySpin,
    SpinLoopHint,
    YieldThenPark,
}

/// Wait strategies control how consumers wait for new spikes.
pub enum WaitStrategy {
    /// Pure spin on the atomic cursor. Lowest latency (<10ns) but burns CPU.
    /// Best for latency-critical consumers with dedicated cores.
    BusySpin,

    /// Uses `core::hint::spin_loop()` (PAUSE on x86, YIELD on ARM).
    /// Good balance of latency and CPU usage.
    SpinLoopHint,

    /// Spin for N iterations, then `thread::yield_now()`.
    /// Best for background consumers that don't need ultra-low latency.
    YieldThenPark {
        spin_count: u32,
    },
}

impl WaitStrategy {
    pub fn from_kind(kind: WaitStrategyKind) -> Self {
        match kind {
            WaitStrategyKind::BusySpin => WaitStrategy::BusySpin,
            WaitStrategyKind::SpinLoopHint => WaitStrategy::SpinLoopHint,
            WaitStrategyKind::YieldThenPark => WaitStrategy::YieldThenPark { spin_count: 1000 },
        }
    }

    /// Wait until `cursor` advances past `current_value`.
    /// Returns the new cursor value.
    #[inline]
    pub fn wait_for(&self, cursor: &AtomicU64, current_value: u64) -> u64 {
        match self {
            WaitStrategy::BusySpin => {
                loop {
                    let val = cursor.load(Ordering::Acquire);
                    if val > current_value {
                        return val;
                    }
                }
            }
            WaitStrategy::SpinLoopHint => {
                loop {
                    let val = cursor.load(Ordering::Acquire);
                    if val > current_value {
                        return val;
                    }
                    core::hint::spin_loop();
                }
            }
            WaitStrategy::YieldThenPark { spin_count } => {
                let mut spins = 0u32;
                loop {
                    let val = cursor.load(Ordering::Acquire);
                    if val > current_value {
                        return val;
                    }
                    spins += 1;
                    if spins < *spin_count {
                        core::hint::spin_loop();
                    } else {
                        thread::yield_now();
                        spins = 0;
                    }
                }
            }
        }
    }

    /// Non-blocking check. Returns Some(new_value) if cursor advanced.
    #[inline]
    pub fn try_wait(&self, cursor: &AtomicU64, current_value: u64) -> Option<u64> {
        let val = cursor.load(Ordering::Acquire);
        if val > current_value {
            Some(val)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn try_wait_returns_none_when_not_advanced() {
        let cursor = AtomicU64::new(5);
        let strategy = WaitStrategy::BusySpin;
        assert!(strategy.try_wait(&cursor, 5).is_none());
        assert!(strategy.try_wait(&cursor, 10).is_none());
    }

    #[test]
    fn try_wait_returns_value_when_advanced() {
        let cursor = AtomicU64::new(10);
        let strategy = WaitStrategy::SpinLoopHint;
        assert_eq!(strategy.try_wait(&cursor, 5), Some(10));
    }

    #[test]
    fn wait_for_returns_immediately_when_advanced() {
        let cursor = AtomicU64::new(10);
        let strategy = WaitStrategy::BusySpin;
        assert_eq!(strategy.wait_for(&cursor, 5), 10);
    }

    #[test]
    fn wait_for_spin_loop_hint() {
        let cursor = Arc::new(AtomicU64::new(0));
        let cursor2 = cursor.clone();

        let handle = thread::spawn(move || {
            let strategy = WaitStrategy::SpinLoopHint;
            strategy.wait_for(&cursor2, 0)
        });

        // Give the thread a moment to start spinning
        thread::sleep(std::time::Duration::from_millis(1));
        cursor.store(1, Ordering::Release);

        assert_eq!(handle.join().unwrap(), 1);
    }

    #[test]
    fn wait_for_yield_then_park() {
        let cursor = Arc::new(AtomicU64::new(0));
        let cursor2 = cursor.clone();

        let handle = thread::spawn(move || {
            let strategy = WaitStrategy::YieldThenPark { spin_count: 10 };
            strategy.wait_for(&cursor2, 0)
        });

        thread::sleep(std::time::Duration::from_millis(5));
        cursor.store(1, Ordering::Release);

        assert_eq!(handle.join().unwrap(), 1);
    }

    #[test]
    fn from_kind() {
        let _ = WaitStrategy::from_kind(WaitStrategyKind::BusySpin);
        let _ = WaitStrategy::from_kind(WaitStrategyKind::SpinLoopHint);
        let _ = WaitStrategy::from_kind(WaitStrategyKind::YieldThenPark);
    }
}
