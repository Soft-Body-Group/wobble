//! Does the object-storage configuration work? `cargo run -p engine --example s3probe -- s3://bucket/prefix`
//! HEADs, LISTs and reads the first object, printing what came back and
//! the request count — the first thing to run when a Read of an `s3://`
//! path fails.
use engine::decode::ExrDecoder as _;
use engine::objstore::{self, Loc};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let arg = std::env::args().nth(1).expect("s3://bucket/prefix");
    let p = Path::new(&arg);
    let loc = Loc::parse(p)?;
    println!("bucket {} -> {}", loc.bucket, objstore::describe(&loc.bucket));
    match objstore::list(&loc) {
        Ok(v) => {
            println!("LIST ok: {} entries", v.len());
            for e in v.iter().take(5) {
                println!("  {}", e.display());
            }
            if let Some(first) = v.iter().find(|e| engine::decode::is_supported(e)) {
                let l = Loc::parse(first)?;
                println!("HEAD {}: {:?}", first.display(), objstore::head(&l)?);
                let mut r = engine::vfs::open(first)?;
                let mut head = [0u8; 16];
                std::io::Read::read_exact(&mut r, &mut head)?;
                println!("GET first 16 bytes: {head:02x?}");
                if let Ok(info) = engine::decode::AnyDecoder::default().read_header(first) {
                    println!("header: {}x{}", info.width, info.height);
                }
            }
        }
        Err(e) => println!("LIST failed: {e:#}"),
    }
    if !loc.key.is_empty() && !loc.key.ends_with('/') {
        println!("HEAD {}: {:?}", arg, objstore::head(&loc));
    }
    println!("{:?}", objstore::stats());
    Ok(())
}
