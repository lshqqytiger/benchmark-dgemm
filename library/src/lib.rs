#[cfg(target_arch = "arm")]
mod armpl;
#[cfg(target_arch = "arm")]
pub use armpl::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};

#[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
mod mkl;
#[cfg_attr(target_arch = "x86", target_arch = "x86_64")]
pub use mkl::{cblas_daxpy, cblas_dgemm, cblas_dnrm2, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

use std::fmt;

pub struct UnexpectedValueError<T> {
    value: T,
    expected: Vec<T>,
}

impl<T: fmt::Display> fmt::Debug for UnexpectedValueError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected one of [{}], but got {}",
            self.expected
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", "),
            self.value
        )
    }
}

impl fmt::Display for CBLAS_LAYOUT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            CBLAS_LAYOUT::CblasRowMajor => "Row-major",
            CBLAS_LAYOUT::CblasColMajor => "Column-major",
            _ => panic!(),
        })
    }
}

macro_rules! match_table {
    ($value:expr, $($candidate:expr => $result:expr), +) => {
        match $value {
            $($candidate => Ok($result),)+
            x => Err(UnexpectedValueError {
                value: x,
                expected: vec![$($candidate,)+],
            }),
        }
    };
}

impl<'a> TryFrom<&'a str> for CBLAS_LAYOUT {
    type Error = UnexpectedValueError<&'a str>;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match_table!(
            value,
            "ROW" => CBLAS_LAYOUT::CblasRowMajor,
            "COL" => CBLAS_LAYOUT::CblasColMajor
        )
    }
}

impl TryFrom<u32> for CBLAS_TRANSPOSE {
    type Error = UnexpectedValueError<u32>;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match_table!(
            value,
            111 => CBLAS_TRANSPOSE::CblasNoTrans,
            112 => CBLAS_TRANSPOSE::CblasTrans,
            113 => CBLAS_TRANSPOSE::CblasConjTrans
        )
    }
}
