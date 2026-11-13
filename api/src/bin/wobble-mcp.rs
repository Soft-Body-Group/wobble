//! `wobble-mcp [--port N]`: an MCP server on stdio that drives a running
//! wobble (`wobble --api`). Point an agent's MCP config at this binary.

fn main() {
    let mut port = api::port_from_env();
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
                eprintln!("usage: wobble-mcp [--port N]   (default {}, or WOBBLE_API_PORT)", api::DEFAULT_PORT);
                std::process::exit(2)
            }
        }
    }
    let stdin = std::io::stdin();
    if let Err(e) = api::mcp::Bridge::new(port).run(stdin.lock(), std::io::stdout()) {
        eprintln!("wobble-mcp: {e}");
        std::process::exit(1);
    }
}
