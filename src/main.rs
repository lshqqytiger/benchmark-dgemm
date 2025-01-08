use argh::FromArgs;
use armpl_sys::{
    armpl_int_t, cblas_daxpy, cblas_dgemm, cblas_dnrm2, CBLAS_LAYOUT, CBLAS_TRANSPOSE,
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{ffi::c_double, fs, process, time};

#[derive(FromArgs)]
/// arguments
struct Arguments {
    /// path to kernel source file
    #[argh(positional)]
    kernel: String,

    #[argh(positional)]
    out: String,

    /// save benchmark result as
    #[argh(option)]
    save_as: Option<String>,

    /// not present: auto, true: recompile anyway, false: don't recompile
    #[argh(option)]
    compile: Option<bool>,

    /// compiler
    #[argh(option, default = "String::from(\"armclang\")")]
    compiler: String,

    /// repeats
    #[argh(option, default = "10")]
    repeats: usize,

    /// skip dgemm result verification
    #[argh(switch)]
    skip_verification: bool,

    /// layout
    #[argh(option, default = "String::from(\"ROW\")")]
    layout: String,

    /// transpose a
    #[argh(option, default = "CBLAS_TRANSPOSE::CblasNoTrans.0")]
    trans_a: u32,

    /// transpose b
    #[argh(option, default = "CBLAS_TRANSPOSE::CblasNoTrans.0")]
    trans_b: u32,

    /// m
    #[argh(option, default = "10000", short = 'm')]
    m: usize,

    /// n
    #[argh(option, default = "10000", short = 'n')]
    n: usize,

    /// k
    #[argh(option, default = "10000", short = 'k')]
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

fn build(compiler: String, kernel: &String, out: &String) -> process::ExitStatus {
    let mut compiler = process::Command::new(compiler)
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
    medium: u128,
    average: u128,
    maximum: u128,
    minimum: u128,
    deviation: f64,
}

impl Statistics {
    fn from(mut records: Vec<u128>) -> Statistics {
        records.par_sort();
        let medium = records[records.len() / 2];
        let average = records.par_iter().sum::<u128>() / records.len() as u128;
        let maximum = *records.last().unwrap();
        let minimum = *records.first().unwrap();
        let deviation = (records
            .iter()
            .map(|&x| ns_to_ms(x.abs_diff(average)).powi(2))
            .fold(0.0, |acc, x| acc + x)
            / records.len() as f64)
            .sqrt();
        Statistics {
            medium,
            average,
            maximum,
            minimum,
            deviation,
        }
    }

    #[inline]
    fn medium_as_ms(&self) -> f64 {
        ns_to_ms(self.medium)
    }

    #[inline]
    fn average_as_ms(&self) -> f64 {
        ns_to_ms(self.average)
    }

    #[inline]
    fn maximum_as_ms(&self) -> f64 {
        ns_to_ms(self.maximum)
    }

    #[inline]
    fn minimum_as_ms(&self) -> f64 {
        ns_to_ms(self.minimum)
    }
}

/// ???
static CHUNK_SIZE: usize = 2048;

fn main() {
    let args: Arguments = argh::from_env();

    if fs::File::open(&args.out)
        .and_then(|out| {
            let source = fs::File::open(&args.kernel)
                .expect("Error: kernel not found!")
                .metadata()
                .expect("Error: failed to query metadata");
            Ok(args.compile.unwrap_or(
                source.accessed().expect("Error: unsupported platform")
                    > out
                        .metadata()
                        .expect("Error: failed to query metadata")
                        .created()
                        .expect("Error: unsupported platform"),
            ))
        })
        .unwrap_or_else(|_| {
            if args.compile.is_some_and(|x| !x) {
                eprintln!("Error: could not find compiled object");
                process::exit(1)
            }
            true
        })
    {
        if !build(args.compiler, &args.kernel, &args.out).success() {
            eprintln!("Error: compilation failed!");
            process::exit(1)
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

    let layout = CBLAS_LAYOUT::try_from(args.layout.to_uppercase().as_str())
        .expect("Error: unexpected value for layout");
    let (trans_a, trans_b) = (
        CBLAS_TRANSPOSE::try_from(args.trans_a).expect("Error: unexpected value for transpose"),
        CBLAS_TRANSPOSE::try_from(args.trans_b).expect("Error: unexpected value for transpose"),
    );

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
        println!("Duration: {:.6}ms", ns_to_ms(duration));
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
    println!(
        "Medium\t {:.6}ms \t {}",
        statistics.medium_as_ms(),
        2.0 * (m * n * k) as f64 / statistics.medium as f64
    );
    println!("Average\t {:.6}ms", statistics.average_as_ms());
    println!(
        "Worst\t {:.6}ms \t {}",
        statistics.maximum_as_ms(),
        2.0 * (m * n * k) as f64 / statistics.maximum as f64
    );
    println!(
        "Best\t {:.6}ms \t {}",
        statistics.minimum_as_ms(),
        2.0 * (m * n * k) as f64 / statistics.minimum as f64
    );
    println!("Deviation\t {}", statistics.deviation);

    if let Some(_) = args.save_as.and_then(|x| fs::File::create(x).ok()) {
        // TODO
    }
}
