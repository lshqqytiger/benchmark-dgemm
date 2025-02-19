use argh::FromArgs;
use benchmark::*;
use library::{cblas_daxpy, cblas_dgemm, cblas_dnrm2, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use std::{ffi::c_double, fs, io::Write, path, process, sync, time};

trait IsErrOr<T> {
    fn is_err_or(self, f: impl FnOnce(T) -> bool) -> bool;
}

impl<T, E> IsErrOr<T> for Result<T, E> {
    fn is_err_or(self, f: impl FnOnce(T) -> bool) -> bool {
        match self {
            Ok(t) => f(t),
            Err(_) => true,
        }
    }
}

#[derive(FromArgs)]
/// arguments
struct Arguments {
    /// path to kernel source file
    #[argh(positional)]
    kernel: String,

    /// path to compiled binary
    #[argh(positional)]
    out: Option<String>,

    /// save benchmark result
    #[argh(option)]
    save_as: Option<String>,

    /// save benchmark history
    #[argh(option)]
    save_history_as: Option<String>,

    /// not present: auto, true: recompile anyway, false: don't recompile
    #[argh(option)]
    compile: Option<bool>,

    /// compiler
    #[argh(option, default = "Arguments::default_compiler()")]
    compiler: String,

    /// repeats
    #[argh(option, short = 'r', default = "10")]
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
    #[argh(option, short = 'm', default = "10000")]
    m: usize,

    /// n
    #[argh(option, short = 'n', default = "10000")]
    n: usize,

    /// k
    #[argh(option, short = 'k', default = "10000")]
    k: usize,

    /// alpha
    #[argh(option, default = "1.0")]
    alpha: f64,

    /// beta
    #[argh(option, default = "1.0")]
    beta: f64,
}

impl Arguments {
    fn default_compiler() -> String {
        #[cfg(target_arch = "arm")]
        return String::from("armclang");
        #[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
        return String::from("/usr/bin/clang");
    }
}

struct Kernel<'lib>(
    libloading::Symbol<
        'lib,
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

impl<'lib> Kernel<'lib> {
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
    ) -> common::Duration {
        let a = a.as_ptr();
        let b = b.as_ptr();
        let c = c.as_mut_ptr();

        let start_time = time::Instant::now();
        unsafe {
            self.0(
                layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
        }
        let end_time = time::Instant::now();
        common::Duration((end_time - start_time).as_nanos())
    }
}

fn check_args(args: &Arguments) {
    if args.repeats == 0 {
        eprintln!("Error: repeats should be signed integer that is not 0");
        process::exit(1)
    }
}

#[cfg(target_arch = "aarch64")]
fn build_extra_args(command: &mut process::Command) {
    command.arg("-armpl");
    command.arg("-mcpu=native");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn build_extra_args(command: &mut process::Command) {
    command.arg("-lmkl_rt");
    command.arg("-march=native");
}

fn build(compiler: String, kernel: &String, out: &String) -> process::ExitStatus {
    let mut command = process::Command::new(compiler);
    command.arg("-O3");
    command.arg("-fopenmp");
    command.args(["-lm", "-lnuma"]);
    build_extra_args(&mut command);
    command.args(["-Wall", "-Werror"]);
    command.arg("-shared");
    command.args(["-o", out]);
    command.arg(kernel);
    command.args(["-I", env!("PATH_INCLUDE")]);
    command
        .spawn()
        .expect("Error: failed to run compiler")
        .wait()
        .expect("Error: failed to wait compiler exit")
}

static FILENAME_TEMP: sync::LazyLock<String> = sync::LazyLock::new(|| ".temp".to_string());

fn main() {
    let args: Arguments = argh::from_env();
    check_args(&args);

    // these parts look really ugly, but they do what should be done.
    // out=Some, compile=Some(true) => build(out) then run(out),
    // out=Some, compile=Some(false) => run(out),
    // out=Some, compile=None => auto then run(out),
    // out=None, compile=Some(true) => build(".temp") then run(".temp"),
    // out=None, compile=Some(false) => run(kernel),
    // out=None, compile=None => build(".temp") then run(".temp"),
    let (out, compile) = args.out.as_ref().map_or_else(
        || {
            // out=None
            if args.compile.is_some_and(|x| !x) {
                // compile=Some(false)
                (&args.kernel, false)
            } else {
                // compile=Some(true) or compile=None
                (&*FILENAME_TEMP, true)
            }
        },
        |out| {
            // out=Some
            (
                out,
                args.compile.unwrap_or_else(|| {
                    // compile or auto
                    fs::File::open(out).is_err_or(|out| {
                        let source = fs::File::open(&args.kernel)
                            .expect("Error: kernel not found")
                            .metadata()
                            .expect("Error: failed to query metadata");
                        source.accessed().expect("Error: unsupported platform")
                            > out
                                .metadata()
                                .expect("Error: failed to query metadata")
                                .created()
                                .expect("Error: unsupported platform")
                    })
                }),
            )
        },
    );
    if compile && !build(args.compiler, &args.kernel, out).success() {
        eprintln!("Error: compilation failed");
        process::exit(1)
    }

    let library =
        unsafe { libloading::Library::new(out) }.expect("Error: failed to load compiled object");
    let kernel = Kernel(
        unsafe { library.get(b"call_dgemm") }
            .expect("Error: compiled object does not contain symbol call_dgemm"),
    );

    let dimensions = (args.m, args.n, args.k);
    let (m, n, k) = dimensions;
    println!("M: {}, N: {}, K: {}", m, n, k);
    println!("alpha: {:.4}, beta: {:.4}", args.alpha, args.beta);

    let layout = CBLAS_LAYOUT::try_from(args.layout.to_uppercase().as_str())
        .expect("Error: unexpected value for layout");
    println!("Layout: {}", layout);

    let transpose = (
        CBLAS_TRANSPOSE::try_from(args.trans_a).expect("Error: unexpected value for transpose"),
        CBLAS_TRANSPOSE::try_from(args.trans_b).expect("Error: unexpected value for transpose"),
    );
    let (trans_a, trans_b) = transpose;
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

    let a = utils::fill_rand(m * k, 100, 0.0, 2.0);
    let b = utils::fill_rand(k * n, 200, 0.0, 2.0);
    let mut c = unsafe { utils::malloc::<f64>(m * n) };

    let mut records = Vec::with_capacity(args.repeats);
    for i in 0..args.repeats {
        let duration = kernel.run(
            layout, trans_a, trans_b, dimensions, &a, lda, &b, ldb, &mut c, ldc, args.alpha,
            args.beta,
        );
        println!("Duration: {:.6}ms", duration.as_milis());
        records.push(duration);

        if i == 0 && !args.skip_verification {
            let difference = unsafe {
                let mut d = utils::malloc::<f64>(m * n);
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

                let n = (m * n) as _;
                cblas_daxpy(n, -1.0, c.as_ptr(), 1, d.as_mut_ptr(), 1);
                cblas_dnrm2(n, d.as_ptr(), 1)
            };
            if difference > 0.0001 {
                eprintln!("WRONG RESULT!");
                process::exit(1)
            }
        }
    }
    drop(library.close());
    let records = records;

    if out.as_ptr() == FILENAME_TEMP.as_ptr() {
        drop(fs::remove_file(&*FILENAME_TEMP));
    }

    let report = common::Report {
        name: path::PathBuf::from(args.kernel)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        dimensions,
        repeats: args.repeats,
        alpha: args.alpha,
        beta: args.beta,
        layout,
        transpose,
        statistics: common::Statistics::from(&records),
    };
    println!("{}", report.summary().unwrap());

    if let Some(mut file) = args.save_as.and_then(|x| fs::File::create(x).ok()) {
        file.write_all(
            serde_json::to_string(&report)
                .expect("Error: failed to serialize")
                .as_bytes(),
        )
        .expect("Error: failed to save benchmark report");
    }

    if let Some(mut file) = args.save_history_as.and_then(|x| fs::File::create(x).ok()) {
        file.write_all(
            records
                .into_iter()
                .map(|x| format!("{:.6}", x.as_milis()))
                .collect::<Vec<String>>()
                .join("\n")
                .as_bytes(),
        )
        .expect("Error: failed to save benchmark history");
    }
}
