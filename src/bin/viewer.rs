use argh::FromArgs;
use benchmark::*;
use std::{fs, io::Write, process};

#[derive(FromArgs)]
/// arguments
struct Arguments {
    #[argh(positional)]
    reports: Vec<String>,

    /// merge reports into one file
    #[argh(option, short = 'o')]
    out: Option<String>,
}

fn main() {
    let args: Arguments = argh::from_env();

    let mut reports = Vec::new();
    for report in args.reports {
        let glob = glob::glob(report.as_str());
        if glob.is_err() {
            continue;
        }
        for matched in glob.unwrap() {
            reports.push(
                serde_json::from_reader::<fs::File, common::Report>(
                    fs::File::open(matched.expect("Error: glob failed"))
                        .expect("Error: could not open file"),
                )
                .expect("Error: unknown format"),
            );
        }
    }
    let reports = reports;

    let mut report = common::Report {
        name: reports[0].name.clone(),
        dimensions: reports[0].dimensions,
        repeats: reports.iter().fold(0, |acc, x| acc + x.repeats),
        alpha: reports[0].alpha,
        beta: reports[0].beta,
        layout: reports[0].layout,
        transpose: reports[0].transpose,
        statistics: common::Statistics::new(),
    };

    for v in &reports[1..] {
        if v.dimensions != report.dimensions
            || v.alpha != report.alpha
            || v.beta != report.beta
            || v.layout != report.layout
            || v.transpose != report.transpose
        {
            eprintln!("Error: cannot merge reports that have different parameters.");
            process::exit(1)
        }
    }

    if reports.len() == 1 {
        report.statistics.medium = reports[0].statistics.medium;
    }

    {
        let mut maximum = common::Duration::MIN;
        for report in &reports {
            if maximum < report.statistics.maximum {
                maximum = report.statistics.maximum;
            }
        }
        report.statistics.maximum = maximum;
    }

    {
        let mut minimum = common::Duration::MAX;
        for report in &reports {
            if minimum > report.statistics.minimum {
                minimum = report.statistics.minimum;
            }
        }
        report.statistics.minimum = minimum;
    }

    report.statistics.average = reports.iter().fold(0.0, |acc, x| {
        acc + x.statistics.average * x.repeats as f64 / report.repeats as f64
    });

    // TODO: deviation

    if let Some(mut file) = args.out.and_then(|x| fs::File::create(x).ok()) {
        file.write_all(
            serde_json::to_string(&report)
                .expect("Error: failed to serialize")
                .as_bytes(),
        )
        .expect("Error: failed to save merged report");
        return;
    }

    println!("{}", report.full().unwrap());
}
