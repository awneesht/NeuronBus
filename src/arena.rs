//! Shared memory arena for large payloads.
//!
//! When spike payloads exceed 46 bytes, data is stored in the Arena and
//! referenced via ArenaRef encoded in the spike payload. The Arena uses
//! mmap-backed memory with a lock-free bump allocator for zero-copy access.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use memmap2::MmapMut;

use crate::spike::ArenaRef;

/// Error types for Arena operations.
#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("arena out of space: requested {requested} bytes, {available} available")]
    OutOfSpace { requested: u64, available: u64 },
    #[error("invalid arena reference: offset={offset}, len={len}")]
    InvalidRef { offset: u64, len: u32 },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Header stored at the beginning of the arena mmap region.
/// Contains the bump allocator state and reference count slots.
#[repr(C)]
struct ArenaHeader {
    /// Next allocation offset (bump pointer).
    alloc_offset: AtomicU64,
    /// Total capacity of the data region.
    capacity: u64,
    /// Epoch counter for reset operations.
    epoch: AtomicU64,
    /// Number of ref count slots in use.
    ref_count_next: AtomicU32,
    _pad: [u8; 4],
}

const HEADER_SIZE: usize = std::mem::size_of::<ArenaHeader>();
const REF_COUNT_SLOTS: usize = 1 << 16; // 64K ref count slots
const REF_COUNTS_SIZE: usize = REF_COUNT_SLOTS * std::mem::size_of::<AtomicU32>();
const DATA_OFFSET: usize = HEADER_SIZE + REF_COUNTS_SIZE;

/// mmap-backed shared memory region for large payloads.
pub struct Arena {
    mmap: MmapMut,
    /// Total capacity of the data region.
    capacity: u64,
}

impl Arena {
    /// Create a new anonymous Arena with the given data capacity.
    pub fn new(capacity: usize) -> Result<Self, ArenaError> {
        let total_size = DATA_OFFSET + capacity;
        let mut mmap = MmapMut::map_anon(total_size)?;

        // Initialize header.
        // SAFETY: mmap region is at least HEADER_SIZE bytes, properly aligned for u64.
        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut ArenaHeader) };
        header.alloc_offset = AtomicU64::new(0);
        header.capacity = capacity as u64;
        header.epoch = AtomicU64::new(0);
        header.ref_count_next = AtomicU32::new(0);

        Ok(Arena {
            mmap,
            capacity: capacity as u64,
        })
    }

    /// Create a file-backed Arena for cross-process sharing.
    pub fn file_backed(path: &std::path::Path, capacity: usize) -> Result<Self, ArenaError> {
        let total_size = DATA_OFFSET + capacity;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize header.
        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut ArenaHeader) };
        header.alloc_offset = AtomicU64::new(0);
        header.capacity = capacity as u64;
        header.epoch = AtomicU64::new(0);
        header.ref_count_next = AtomicU32::new(0);

        Ok(Arena {
            mmap,
            capacity: capacity as u64,
        })
    }

    fn header(&self) -> &ArenaHeader {
        // SAFETY: Arena is always at least HEADER_SIZE bytes.
        unsafe { &*(self.mmap.as_ptr() as *const ArenaHeader) }
    }

    fn ref_counts_base(&self) -> *const AtomicU32 {
        // SAFETY: Ref counts start right after the header.
        unsafe { self.mmap.as_ptr().add(HEADER_SIZE) as *const AtomicU32 }
    }

    fn data_base(&self) -> *const u8 {
        // SAFETY: Data region starts at DATA_OFFSET.
        unsafe { self.mmap.as_ptr().add(DATA_OFFSET) }
    }

    /// Allocate space in the arena and write data. Returns an ArenaRef.
    ///
    /// Uses lock-free bump allocation via fetch_add.
    /// Allocations are 8-byte aligned.
    pub fn write(&self, data: &[u8]) -> Result<ArenaRef, ArenaError> {
        let len = data.len() as u64;
        // Round up to 8-byte alignment.
        let aligned_len = (len + 7) & !7;

        let header = self.header();
        let offset = header.alloc_offset.fetch_add(aligned_len, Ordering::Relaxed);

        if offset + aligned_len > self.capacity {
            // Roll back the allocation.
            header.alloc_offset.fetch_sub(aligned_len, Ordering::Relaxed);
            return Err(ArenaError::OutOfSpace {
                requested: len,
                available: self.capacity.saturating_sub(offset),
            });
        }

        // Allocate a ref count slot.
        let ref_slot = header.ref_count_next.fetch_add(1, Ordering::Relaxed);
        if (ref_slot as usize) >= REF_COUNT_SLOTS {
            header.ref_count_next.fetch_sub(1, Ordering::Relaxed);
            header.alloc_offset.fetch_sub(aligned_len, Ordering::Relaxed);
            return Err(ArenaError::OutOfSpace {
                requested: len,
                available: 0,
            });
        }

        // Initialize ref count to 1.
        // SAFETY: ref_slot is within bounds of the ref count array.
        unsafe {
            let rc = &*(self.ref_counts_base().add(ref_slot as usize));
            rc.store(1, Ordering::Release);
        }

        // SAFETY: We just allocated [offset, offset+len) and it's within bounds.
        // The mmap is writable. We use a raw pointer write because we only hold &self.
        // This is safe because each allocation gets a unique offset via atomic fetch_add.
        unsafe {
            let dst = (self.data_base() as *mut u8).add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        Ok(ArenaRef {
            offset,
            len: data.len() as u32,
            ref_slot,
        })
    }

    /// Read data from the arena by reference. Zero-copy slice into mmap.
    pub fn read(&self, arena_ref: &ArenaRef) -> Result<&[u8], ArenaError> {
        let end = arena_ref.offset + arena_ref.len as u64;
        if end > self.capacity {
            return Err(ArenaError::InvalidRef {
                offset: arena_ref.offset,
                len: arena_ref.len,
            });
        }

        // SAFETY: Offset and length are validated to be within the data region.
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.data_base().add(arena_ref.offset as usize),
                arena_ref.len as usize,
            )
        };
        Ok(slice)
    }

    /// Increment reference count for an ArenaRef.
    pub fn add_ref(&self, arena_ref: &ArenaRef) {
        if (arena_ref.ref_slot as usize) < REF_COUNT_SLOTS {
            // SAFETY: ref_slot is within bounds.
            unsafe {
                let rc = &*(self.ref_counts_base().add(arena_ref.ref_slot as usize));
                rc.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Decrement reference count. Returns true if this was the last reference.
    pub fn release(&self, arena_ref: &ArenaRef) -> bool {
        if (arena_ref.ref_slot as usize) >= REF_COUNT_SLOTS {
            return false;
        }
        // SAFETY: ref_slot is within bounds.
        unsafe {
            let rc = &*(self.ref_counts_base().add(arena_ref.ref_slot as usize));
            rc.fetch_sub(1, Ordering::Release) == 1
        }
    }

    /// Epoch-based reset: clears all allocations. Not safe to call while readers exist.
    pub fn reset(&self) {
        let header = self.header();
        header.alloc_offset.store(0, Ordering::Release);
        header.ref_count_next.store(0, Ordering::Release);
        header.epoch.fetch_add(1, Ordering::Release);
    }

    /// Current allocation offset (bytes used).
    pub fn used(&self) -> u64 {
        self.header().alloc_offset.load(Ordering::Relaxed)
    }

    /// Total data capacity.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Current epoch.
    pub fn epoch(&self) -> u64 {
        self.header().epoch.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_read() {
        let arena = Arena::new(1024).unwrap();
        let data = b"hello, arena!";
        let aref = arena.write(data).unwrap();
        let read_back = arena.read(&aref).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn multiple_allocations() {
        let arena = Arena::new(4096).unwrap();
        let refs: Vec<ArenaRef> = (0..10)
            .map(|i| {
                let data = format!("payload_{i}");
                arena.write(data.as_bytes()).unwrap()
            })
            .collect();

        for (i, aref) in refs.iter().enumerate() {
            let expected = format!("payload_{i}");
            let data = arena.read(aref).unwrap();
            assert_eq!(data, expected.as_bytes());
        }
    }

    #[test]
    fn alignment() {
        let arena = Arena::new(4096).unwrap();
        // Odd-sized allocation should still result in 8-byte aligned next offset.
        let _r1 = arena.write(&[1, 2, 3]).unwrap(); // 3 bytes -> rounds to 8
        let r2 = arena.write(&[4, 5]).unwrap();
        assert_eq!(r2.offset % 8, 0);
    }

    #[test]
    fn out_of_space() {
        let arena = Arena::new(64).unwrap();
        let _ = arena.write(&[0u8; 60]).unwrap();
        let result = arena.write(&[0u8; 60]);
        assert!(result.is_err());
    }

    #[test]
    fn reference_counting() {
        let arena = Arena::new(1024).unwrap();
        let aref = arena.write(b"test").unwrap();

        arena.add_ref(&aref); // refcount = 2
        assert!(!arena.release(&aref)); // refcount = 1
        assert!(arena.release(&aref)); // refcount = 0, last ref
    }

    #[test]
    fn epoch_reset() {
        let arena = Arena::new(1024).unwrap();
        let _ = arena.write(b"data").unwrap();
        assert!(arena.used() > 0);
        assert_eq!(arena.epoch(), 0);

        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.epoch(), 1);
    }

    #[test]
    fn concurrent_allocations() {
        let arena = std::sync::Arc::new(Arena::new(1 << 20).unwrap());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let arena = arena.clone();
                std::thread::spawn(move || {
                    let mut refs = Vec::new();
                    for i in 0..1000 {
                        let data = format!("thread{t}_item{i}");
                        let aref = arena.write(data.as_bytes()).unwrap();
                        refs.push((data, aref));
                    }
                    // Verify all our allocations.
                    for (expected, aref) in &refs {
                        let data = arena.read(aref).unwrap();
                        assert_eq!(data, expected.as_bytes());
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
    }
}
