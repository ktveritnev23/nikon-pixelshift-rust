use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        pixelshift::cli::run()
    } else {
        pixelshift::app::run()
    }
}
