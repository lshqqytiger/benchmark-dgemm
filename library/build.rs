use std::{env, path};

fn main() {
    dotenv::from_path(path::absolute(env::current_dir().unwrap().join("../.env")).unwrap())
        .unwrap();

    #[cfg(feature = "link")]
    link(
        env::var("PATH_COMPILER").unwrap(),
        env::var("PATH_LIBRARY").unwrap(),
    );
}

#[cfg(all(target_arch = "aarch64", feature = "link"))]
fn link(path_compiler: String, path_library: String) {
    println!("cargo::rustc-link-lib=dylib=omp");
    println!("cargo::rustc-link-lib=dylib=flang");
    println!("cargo::rustc-link-lib=dylib=armpl_mp");
    println!("cargo::rustc-link-search=native={}/lib/", path_compiler);
    println!("cargo::rustc-link-search=native={}/lib/", path_library);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "link"))]
fn link(path_compiler: String, path_library: String) {
    println!("cargo::rustc-link-lib=dylib=gomp");
    println!("cargo::rustc-link-lib=dylib=mkl_rt");
    println!("cargo::rustc-link-search=native={}/", path_compiler);
    println!("cargo::rustc-link-search=native={}/", path_library);
}
