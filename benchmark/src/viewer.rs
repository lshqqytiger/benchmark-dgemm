mod common;
use argh::FromArgs;
use std::fs;

#[derive(FromArgs)]
/// arguments
struct Arguments {
    /// path to report file
    #[argh(positional)]
    report: String,
}

fn main() {
    let args: Arguments = argh::from_env();
    let report = serde_json::from_reader::<fs::File, common::Report>(
        fs::File::open(args.report).expect("Error: could not open file"),
    )
    .expect("Error: unknown format");
    println!("{}", report.full().unwrap());
}
