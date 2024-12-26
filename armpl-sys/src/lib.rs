pub mod armpl;
use std::fmt;

pub use armpl::*;

impl fmt::Display for CBLAS_LAYOUT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            CBLAS_LAYOUT::CblasColMajor => "Column-major",
            CBLAS_LAYOUT::CblasRowMajor => "Row-major",
            _ => unreachable!(),
        })
    }
}
