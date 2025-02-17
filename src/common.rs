use library::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Write};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Duration(pub(crate) u128);

impl Duration {
    pub(crate) const ZERO: Duration = Duration(0);
    pub(crate) const MIN: Duration = Duration::ZERO;
    pub(crate) const MAX: Duration = Duration(u128::MAX);
}

impl Duration {
    #[inline(always)]
    pub(crate) fn as_nanos(&self) -> u128 {
        self.0
    }

    #[inline(always)]
    pub(crate) fn as_milis(&self) -> f64 {
        self.0 as f64 / 1000.0 / 1000.0
    }
}

trait Average<T> {
    fn average(&self) -> Option<T>;
}

impl Average<f64> for Vec<f64> {
    /// Calculate average without overflow.
    fn average(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }

        let n = self.len() / 2;
        if n == 1 {
            return Some(if self.len() % 2 == 0 {
                self[0] / 2.0 + self[1] / 2.0
            } else {
                self[0]
            });
        }

        let average = (0..n)
            .into_par_iter()
            .map(|i| self[i * 2] / 2.0 + self[i * 2 + 1] / 2.0)
            .collect::<Vec<f64>>()
            .average()
            .unwrap();

        Some(if self.len() % 2 == 1 {
            // from "average = partial_average * ((n - 1) / n) + ((last + average) / 2) * (1 / n)"
            (average * (2 * n) as f64 + unsafe { self.last().unwrap_unchecked() })
                / (2 * n + 1) as f64
        } else {
            average
        })
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Statistics {
    pub medium: Option<Duration>,
    pub maximum: Duration,
    pub minimum: Duration,
    pub average: f64,
    pub deviation: f64,
}

impl Statistics {
    fn new() -> Self {
        Statistics {
            medium: None,
            maximum: Duration::ZERO,
            minimum: Duration::ZERO,
            average: 0.0,
            deviation: 0.0,
        }
    }
}

impl From<&Vec<Duration>> for Statistics {
    fn from(records: &Vec<Duration>) -> Self {
        assert_ne!(records.len(), 0);

        let vec = {
            let mut sorted = records.par_iter().collect::<Vec<&Duration>>();
            sorted.par_sort();
            sorted
        };
        let medium = Some(*vec[vec.len() / 2]);
        let maximum = **unsafe { vec.last().unwrap_unchecked() };
        let minimum = **unsafe { vec.first().unwrap_unchecked() };

        let vec = records
            .par_iter()
            .map(|x| x.as_milis())
            .collect::<Vec<f64>>();
        let average = {
            let average = vec.average();
            unsafe { average.unwrap_unchecked() }
        };
        let deviation = {
            let variances = vec
                .into_par_iter()
                .map(|x| (x - average).powi(2))
                .collect::<Vec<f64>>();
            let average = variances.average();
            unsafe { average.unwrap_unchecked() }.sqrt()
        };

        Statistics {
            medium,
            maximum,
            minimum,
            average,
            deviation,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Report {
    pub name: String,
    pub dimensions: (usize, usize, usize),
    pub alpha: f64,
    pub beta: f64,
    pub layout: CBLAS_LAYOUT,
    pub transpose: (CBLAS_TRANSPOSE, CBLAS_TRANSPOSE),
    pub statistics: Statistics,
}

impl Report {
    pub(crate) fn new() -> Self {
        Report {
            name: "".to_string(),
            dimensions: (0, 0, 0),
            alpha: 1.0,
            beta: 1.0,
            layout: CBLAS_LAYOUT::CblasRowMajor,
            transpose: (CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans),
            statistics: Statistics::new(),
        }
    }
}

impl Report {
    pub(crate) fn summary(&self) -> Result<String, fmt::Error> {
        let mut out = String::new();
        let ops = 2.0 * (self.dimensions.0 * self.dimensions.1 * self.dimensions.2) as f64;
        if let Some(medium) = self.statistics.medium {
            writeln!(
                &mut out,
                "Medium\t {:.6}ms \t {}",
                medium.as_milis(),
                ops / medium.as_nanos() as f64
            )?;
        }
        writeln!(&mut out, "Average\t {:.6}ms", self.statistics.average)?;
        writeln!(
            &mut out,
            "Worst\t {:.6}ms \t {}",
            self.statistics.maximum.as_milis(),
            ops / self.statistics.maximum.as_nanos() as f64
        )?;
        writeln!(
            &mut out,
            "Best\t {:.6}ms \t {}",
            self.statistics.minimum.as_milis(),
            ops / self.statistics.minimum.as_nanos() as f64
        )?;
        write!(&mut out, "Deviation\t {}", self.statistics.deviation)?;
        Ok(out)
    }

    pub(crate) fn full(&self) -> Result<String, fmt::Error> {
        let mut out = String::new();
        writeln!(&mut out, "=== {} ===", self.name)?;
        writeln!(
            &mut out,
            "M: {}, N: {}, K: {}",
            self.dimensions.0, self.dimensions.1, self.dimensions.2
        )?;
        writeln!(&mut out, "alpha: {:.4}, beta: {:.4}", self.alpha, self.beta)?;
        writeln!(&mut out, "Layout: {}", self.layout)?;
        writeln!(
            &mut out,
            "TransA: {}",
            self.transpose.0 == CBLAS_TRANSPOSE::CblasTrans
        )?;
        writeln!(
            &mut out,
            "TransB: {}",
            self.transpose.1 == CBLAS_TRANSPOSE::CblasTrans
        )?;
        out.write_str(self.summary()?.as_str())?;
        Ok(out)
    }
}
