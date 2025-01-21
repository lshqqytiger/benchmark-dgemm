mod common;
use argh::FromArgs;
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

    let mut report = common::Report::new();
    report.name = reports[0].name.clone();
    report.dimensions = reports[0].dimensions;
    report.alpha = reports[0].alpha;
    report.beta = reports[0].beta;
    report.layout = reports[0].layout;
    report.transpose = reports[0].transpose;

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

    report.statistics.average = reports.iter().enumerate().fold(0.0, |acc, (i, report)| {
        if i == 0 {
            report.statistics.average
        } else {
            let i = i as f64;
            acc / i * (i - 1.0) + report.statistics.average / i as f64
        }
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
