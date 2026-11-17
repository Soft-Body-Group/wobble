//! Generative-AI pipelines for wobble.
//!
//! A GenAI node carries a small step graph ([`pipeline::Pipeline`]): a
//! prompt, the node's own input as a reference, one or more model calls, an
//! Output. Pressing RUN turns it into a picture on disk ([`run`]), which the
//! node then plays back through a reader slot like any bake — so a comp
//! with a generated plate in it costs exactly what a comp with a Read costs.
//!
//! What this crate promises the app:
//!
//! - **Nothing runs, and nothing is billed, without a click.** There is no
//!   per-frame evaluation, no auto-run on open, no retry that re-bills.
//! - **The quote comes before the call** ([`run::quote`]), from the service
//!   itself, and the relay refuses a job priced above what was shown.
//! - **Re-running is free when nothing changed**: results are cached by a
//!   key over everything that shapes the pixels ([`run::plan`]).
//! - **A comp names catalog ids; the machine decides where they run and who
//!   pays** ([`config`]). No setup carries a URL or a key.
//! - **Licence is data** ([`catalog::RemoteModel::commercial`]), recorded on
//!   every take, and propagated by the engine like any research-only model.
//!
//! Services: fal with your own key ([`fal`]), the wobble relay with credits
//! ([`relay`]), and a local ComfyUI ([`comfy`]).

pub mod catalog;
pub mod comfy;
pub mod config;
pub mod fal;
pub mod history;
pub mod http;
pub mod image;
pub mod pipeline;
#[cfg(feature = "relay")]
pub mod relay;
pub mod run;
pub mod service;
pub mod templates;
pub mod vary;

pub use pipeline::{Op, Opts, Pipe, Pipeline, Step, Take};
pub use vary::Vary;
