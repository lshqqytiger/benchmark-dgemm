fn main() {
    println!("cargo::rustc-link-lib=dylib=omp");
    println!("cargo::rustc-link-lib=dylib=flang");
    println!("cargo::rustc-link-lib=dylib=armpl_mp");
    println!("cargo::rustc-link-search=native=/opt/arm/arm-linux-compiler-24.10_Ubuntu-22.04/lib/");
    println!("cargo::rustc-link-search=native=/opt/arm/armpl-24.10.0_Ubuntu-22.04_arm-linux-compiler/lib/");
}
