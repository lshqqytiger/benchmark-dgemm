fn main() {
    common();
    #[cfg(target_arch = "arm")]
    armpl();
    #[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
    knl();
}

fn common() {
    println!("cargo::rustc-link-lib=dylib=omp");
}

fn armpl() {
    println!("cargo::rustc-link-lib=dylib=flang");
    println!("cargo::rustc-link-lib=dylib=armpl_mp");
    println!("cargo::rustc-link-search=native=/opt/arm/arm-linux-compiler-24.10_Ubuntu-22.04/lib/");
    println!("cargo::rustc-link-search=native=/opt/arm/armpl-24.10.0_Ubuntu-22.04_arm-linux-compiler/lib/");
}

fn knl() {}
