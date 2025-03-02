#[cfg(target_arch = "arm")]
mod armpl;
#[cfg(target_arch = "arm")]
pub use armpl::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};

#[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
mod mkl;
#[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
pub use mkl::{cblas_daxpy, cblas_dgemm, cblas_dnrm2, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use std::fmt;

impl fmt::Display for CBLAS_LAYOUT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            CBLAS_LAYOUT::CblasRowMajor => "row-major",
            CBLAS_LAYOUT::CblasColMajor => "column-major",
            _ => panic!(),
        })
    }
}

macro_rules! match_table {
    ($value:expr, $($candidate:expr => $result:expr), +) => {
        Ok(match $value {
            $($candidate => $result,)+
            x => {
                let mut builder = vec!["expected one of ["];
                let array = vec![$($candidate,)+].join(", ");
                builder.push(array.as_str());
                builder.push("], but got ");
                builder.push(x);
                return Err(builder.concat());
            },
        })
    };
}

impl<'a> TryFrom<&'a str> for CBLAS_LAYOUT {
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match_table!(
            value.to_uppercase().as_str(),
            "ROW" => CBLAS_LAYOUT::CblasRowMajor,
            "COL" => CBLAS_LAYOUT::CblasColMajor
        )
    }
}

impl<'a> TryFrom<&'a str> for CBLAS_TRANSPOSE {
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match_table!(
            value.to_uppercase().as_str(),
            "FALSE" => CBLAS_TRANSPOSE::CblasNoTrans,
            "TRUE" => CBLAS_TRANSPOSE::CblasTrans,
            "CONJ" => CBLAS_TRANSPOSE::CblasConjTrans
        )
    }
}
