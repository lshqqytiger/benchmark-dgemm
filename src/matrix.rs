use libnuma_sys::{numa_alloc_onnode, numa_free};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{mem, slice};

/// ???
const CHUNK_SIZE: usize = 2048;

enum Allocator {
    BOX,
    NUMA,
}

pub struct Matrix {
    allocator: Allocator,
    internal: *mut f64,
    len: usize,
}

impl Matrix {
    pub fn new(len: usize) -> Self {
        Matrix {
            allocator: Allocator::BOX,
            internal: Box::into_raw(Box::<[f64]>::new_uninit_slice(len)) as _,
            len,
        }
    }

    pub fn numa(len: usize, node: i32) -> Self {
        Matrix {
            allocator: Allocator::NUMA,
            internal: unsafe { numa_alloc_onnode(len * mem::size_of::<f64>(), node) as _ },
            len,
        }
    }

    pub fn as_ptr(&self) -> *const f64 {
        self.internal as _
    }

    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.internal as _
    }

    pub fn as_mut(&self) -> &mut [f64] {
        unsafe { slice::from_raw_parts_mut(self.internal, self.len) }
    }

    /// Originally written by Enoch Jung in C.
    pub fn fill(&self, seed: u64, min: f64, max: f64) {
        let mul = 192499u64;
        let add = 6837199u64;

        let scaling_factor = (max - min) / (u64::MAX as f64);
        self.as_mut()
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(tid, chunk)| {
                let mut value = (tid as u64 * 1034871 + 10581) * seed;

                for _ in 0..(50 + tid as u64) {
                    value = value.wrapping_mul(mul).wrapping_add(add);
                }

                for cell in chunk.iter_mut() {
                    value = value.wrapping_mul(mul).wrapping_add(add);
                    *cell = (value as f64) * scaling_factor + min;
                }
            });
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        match self.allocator {
            Allocator::BOX => unsafe {
                drop(Box::from_raw(self.internal));
            },
            Allocator::NUMA => unsafe {
                numa_free(self.internal as _, self.len * mem::size_of::<f64>());
            },
        }
    }
}
