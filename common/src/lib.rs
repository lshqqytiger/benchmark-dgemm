use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[inline(always)]
pub unsafe fn malloc<T>(size: usize) -> Box<[T]> {
    Box::<[T]>::new_uninit_slice(size).assume_init()
}

/// ???
const CHUNK_SIZE: usize = 2048;

/// Originally written by Enoch Jung in C.
pub fn fill_rand(size: usize, seed: u64, min: f64, max: f64) -> Box<[f64]> {
    let mul = 192499u64;
    let add = 6837199u64;

    let scaling_factor = (max - min) / (u64::MAX as f64);
    let mut matrix = unsafe { malloc::<f64>(size) };
    matrix
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
    matrix
}
