//! The `wobble` binary. Everything lives in the `app` library (lib.rs), so
//! tests, tools and other front ends can link the same code.

fn main() -> anyhow::Result<()> {
    app::run()
}
