//! `scrub-mcp [--port N]`: an MCP server on stdio that drives a running
//! Scrub. Point an agent's MCP config at this binary (NOTES.md says how).

fn main() {
    let mut port = scrub_api::port_from_env();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" => {
                port = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--port needs a number");
                    std::process::exit(2)
                })
            }
            _ => {
                eprintln!("usage: edify-mcp [--port N]   (default {}, or EDIFY_API_PORT)", scrub_api::DEFAULT_PORT);
                std::process::exit(2)
            }
        }
    }
    let stdin = std::io::stdin();
    if let Err(e) = scrub_api::mcp::Bridge::new(port).run(stdin.lock(), std::io::stdout()) {
        eprintln!("scrub-mcp: {e}");
        std::process::exit(1);
    }
}
