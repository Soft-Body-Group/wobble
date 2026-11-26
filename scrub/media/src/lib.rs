//! Scrub's media layer: decode and cache, no UI.
//!
//! | module     | what it owns                                                  |
//! |------------|---------------------------------------------------------------|
//! | `frame`    | a decoded picture in its native planes (YCbCr stays YCbCr)    |
//! | `source`   | `SourceInfo`, the `Source` trait, `open` for any path          |
//! | `movie`    | FFmpeg: warm decoder session pool, zero-copy frames            |
//! | `audiofile`| sound files (WAV, MP3…) as clips; their picture is a waveform |
//! | `sequence` | image sequences from one sample file: EXR, DPX, PNG, JPEG      |
//! | `optimise` | transcode to ProRes beside the original or in a managed cache |
//! | `cache`    | the RAM frame cache, LRU by bytes, shared by every server      |
//! | `server`   | decode workers, prefetch around the playhead, cancellation     |

pub mod audio;
pub mod audiofile;
pub mod cache;
pub mod dpx;
pub mod fetch;
pub mod frame;
pub mod movie;
pub mod netcache;
pub mod netio;
pub mod optimise;
pub mod sequence;
pub mod server;
pub mod source;
pub mod sys;
pub mod thumb;

pub use cache::Cache;
pub use frame::{Frame, FrameRef, Layout, Matrix};
pub use server::{FrameServer, PlayState, Stats};
pub use source::{open, open_at, Source, SourceInfo, SourceRef};
