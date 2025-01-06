use argh::FromArgs;
use armpl_sys::{
    armpl_int_t, cblas_daxpy, cblas_dgemm, cblas_dnrm2, CBLAS_LAYOUT, CBLAS_TRANSPOSE,
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{ffi::c_double, fmt, fs, process, time};

#[derive(FromArgs)]
/// arguments
struct Arguments {
    /// the path to kernel source file.
    #[argh(positional)]
    kernel: String,

    #[argh(positional)]
    out: String,

    /// repeats
    #[argh(option, default = "10")]
    repeats: usize,

    /// skip verification
    #[argh(switch)]
    skip_verification: bool,

    /// layout
    #[argh(option, default = "CBLAS_LAYOUT::CblasRowMajor.0")]
    layout: u32,

    /// trans a
    #[argh(option, default = "CBLAS_TRANSPOSE::CblasNoTrans.0")]
    trans_a: u32,

    /// trans b
    #[argh(option, default = "CBLAS_TRANSPOSE::CblasNoTrans.0")]
    trans_b: u32,

    /// m
    #[argh(option, default = "10000")]
    m: usize,

    /// n
    #[argh(option, default = "10000")]
    n: usize,

    /// k
    #[argh(option, default = "10000")]
    k: usize,

    /// alpha
    #[argh(option, default = "1.0")]
    alpha: f64,

    /// beta
    #[argh(option, default = "1.0")]
    beta: f64,
}

struct Kernel<'a>(
    libloading::Symbol<
        'a,
        unsafe extern "C" fn(
            layout: CBLAS_LAYOUT,
            TransA: CBLAS_TRANSPOSE,
            TransB: CBLAS_TRANSPOSE,
            m: usize,
            n: usize,
            k: usize,
            alpha: c_double,
            A: *const c_double,
            lda: usize,
            B: *const c_double,
            ldb: usize,
            beta: c_double,
            C: *mut c_double,
            ldc: usize,
        ),
    >,
);

impl<'a> Kernel<'a> {
    fn run(
        &self,
        layout: CBLAS_LAYOUT,
        trans_a: CBLAS_TRANSPOSE,
        trans_b: CBLAS_TRANSPOSE,
        (m, n, k): (usize, usize, usize),
        a: &Box<[f64]>,
        lda: usize,
        b: &Box<[f64]>,
        ldb: usize,
        c: &mut Box<[f64]>,
        ldc: usize,
        alpha: f64,
        beta: f64,
    ) -> time::Duration {
        let a = a.as_ptr();
        let b = b.as_ptr();
        let c = c.as_mut_ptr();

        let start_time = time::Instant::now();
        unsafe {
            self.0(
                layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
        }
        time::Instant::now() - start_time
    }
}

#[inline(always)]
unsafe fn malloc<T>(size: usize) -> Box<[T]> {
    Box::<[T]>::new_uninit_slice(size).assume_init()
}

fn is_modified(kernel: &String, out: &String) -> bool {
    let source = fs::File::open(kernel)
        .expect("Error: kernel not found!")
        .metadata()
        .expect("Error: failed to query metadata");
    let out = fs::File::open(out);

    if out.is_err() {
        true
    } else if let Ok(out) = out.unwrap().metadata() {
        source.accessed().expect("Error: unsupported platform")
            > out.created().expect("Error: unsupported platform")
    } else {
        false
    }
}

fn build(kernel: &String, out: &String) -> process::ExitStatus {
    let mut compiler = process::Command::new("armclang")
        .args(["-O3", "-mcpu=native"])
        .arg("-fopenmp")
        .args(["-armpl", "-lm", "-lnuma"])
        .args(["-Wall", "-Werror"])
        .arg("-shared")
        .args(["-o", &out])
        .arg(&kernel)
        .spawn()
        .expect("Error: failed to run compiler");
    compiler
        .wait()
        .expect("Error: failed to wait compiler exit")
}

/// Originally written by Enoch Jung in C.
fn prepare(chunk_size: usize, size: usize, seed: u64, min: f64, max: f64) -> Box<[f64]> {
    let mul = 192499u64;
    let add = 6837199u64;

    let scaling_factor = (max - min) / (u64::MAX as f64);
    let mut matrix = unsafe { malloc::<f64>(size) };
    matrix
        .par_chunks_mut(chunk_size)
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

#[inline(always)]
fn ns_to_ms(ns: u128) -> f64 {
    ns as f64 / 1000.0 / 1000.0
}

struct Statistics {
    pub medium: u128,
    pub average: u128,
    pub maximum: u128,
    pub minimum: u128,
}

impl Statistics {
    fn from(mut records: Vec<u128>) -> Statistics {
        records.par_sort();
        let medium = records[records.len() / 2];
        let average = records.par_iter().sum::<u128>() / records.len() as u128;
        let maximum = *records.last().unwrap();
        let minimum = *records.first().unwrap();
        Statistics {
            medium,
            average,
            maximum,
            minimum,
        }
    }
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("Medium: {}ms\n", ns_to_ms(self.medium)))?;
        f.write_fmt(format_args!("Average: {}ms\n", ns_to_ms(self.average)))?;
        f.write_fmt(format_args!("Maximum: {}ms\n", ns_to_ms(self.maximum)))?;
        f.write_fmt(format_args!("Minimum: {}ms\n", ns_to_ms(self.minimum)))
    }
}

/// ???
static CHUNK_SIZE: usize = 2048;

fn main() {
    let args: Arguments = argh::from_env();

    if is_modified(&args.kernel, &args.out) {
        if !build(&args.kernel, &args.out).success() {
            eprintln!("Error: compilation failed!");
            return;
        }
    }

    let library = unsafe { libloading::Library::new(args.out) }
        .expect("Error: failed to load compiled object");
    let kernel = Kernel(
        unsafe { library.get(b"call_dgemm") }
            .expect("Error: compiled object does not contain symbol call_dgemm"),
    );

    let dimensions = (args.m, args.n, args.k);
    let (m, n, k) = dimensions;
    let a = prepare(CHUNK_SIZE, m * k, 100, 0.0, 2.0);
    let b = prepare(CHUNK_SIZE, k * n, 200, 0.0, 2.0);
    let mut c = unsafe { malloc::<f64>(m * n) };

    println!("M: {}, N: {}, K: {}", m, n, k);
    println!("alpha: {}, beta: {}", args.alpha, args.beta);

    let layout = CBLAS_LAYOUT(args.layout);
    let (trans_a, trans_b) = (CBLAS_TRANSPOSE(args.trans_a), CBLAS_TRANSPOSE(args.trans_b));

    println!("Layout: {}", layout);
    println!("TransA: {}", trans_a == CBLAS_TRANSPOSE::CblasTrans);
    println!("TransB: {}", trans_b == CBLAS_TRANSPOSE::CblasTrans);

    let lda = if (trans_a == CBLAS_TRANSPOSE::CblasTrans) != (layout == CBLAS_LAYOUT::CblasRowMajor)
    {
        k
    } else {
        m
    };
    let ldb = if (trans_b == CBLAS_TRANSPOSE::CblasTrans) != (layout == CBLAS_LAYOUT::CblasRowMajor)
    {
        n
    } else {
        k
    };
    let ldc = if layout == CBLAS_LAYOUT::CblasRowMajor {
        n
    } else {
        m
    };

    let mut records = Vec::with_capacity(args.repeats);
    for i in 0..args.repeats {
        let duration = kernel
            .run(
                layout, trans_a, trans_b, dimensions, &a, lda, &b, ldb, &mut c, ldc, args.alpha,
                args.beta,
            )
            .as_nanos();
        println!("Duration: {}ms", ns_to_ms(duration));
        records.push(duration);

        if i == 0 && !args.skip_verification {
            let difference = unsafe {
                let mut d = malloc::<f64>(m * n);
                cblas_dgemm(
                    layout,
                    trans_a,
                    trans_b,
                    m as _,
                    n as _,
                    k as _,
                    args.alpha,
                    a.as_ptr(),
                    lda as _,
                    b.as_ptr(),
                    ldb as _,
                    args.beta,
                    d.as_mut_ptr(),
                    ldc as _,
                );

                let n = (m * n) as armpl_int_t;
                cblas_daxpy(n, -1.0, c.as_ptr(), 1, d.as_mut_ptr(), 1);
                cblas_dnrm2(n, d.as_ptr(), 1)
            };
            if difference > 0.0001 {
                println!("WRONG RESULT!");
                process::exit(1)
            }
        }
    }

    let statistics = Statistics::from(records);
    println!("{}", statistics);
}
