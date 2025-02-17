#[cfg(target_arch = "arm")]
mod armpl;
#[cfg(target_arch = "arm")]
pub use armpl::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};

#[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
mod knl;
