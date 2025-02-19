use std::{env, path};

fn main() {
    dotenv::from_path(path::absolute(env::current_dir().unwrap().join("../.env")).unwrap())
        .unwrap();

    link(
        env::var("PATH_COMPILER").unwrap(),
        env::var("PATH_LIBRARY").unwrap(),
    );
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
fn link(path_compiler: String, path_library: String) {
    println!("cargo::rustc-link-lib=dylib=omp");
    println!("cargo::rustc-link-lib=dylib=flang");
    println!("cargo::rustc-link-lib=dylib=armpl_mp");
    println!("cargo::rustc-link-search=native={}/lib/", path_compiler);
    println!("cargo::rustc-link-search=native={}/lib/", path_library);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn link(path_compiler: String, path_library: String) {
    println!("cargo::rustc-link-lib=dylib=gomp");
    println!("cargo::rustc-link-lib=dylib=blas64");
    println!("cargo::rustc-link-search=native={}/", path_compiler);
    println!("cargo::rustc-link-search=native={}/", path_library);
}
