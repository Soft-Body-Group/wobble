//! Downloadable model weights.
//!
//! ML nodes need weights that are far too large to commit, and the old
//! answer — a text field pointing at a file you were expected to convert
//! yourself with a python script — is a developer's answer, not a user's.
//! This is the registry and the fetcher behind it: every ML node asks for a
//! `ModelSpec` by id, gets a `State` telling it whether the file is on disk,
//! and can start a resumable, verified `Download` that reports progress the
//! same way a bake does.
//!
//! # The convention for a node that needs weights
//!
//! A node DECLARES the catalog ids it needs in `NodeKind::models()` and
//! does nothing else about downloading. From that one arm it gets, for free:
//! the download gate at the top of its properties panel, a badge in the graph
//! while its weights are missing, a place in the "this setup needs N models"
//! prompt when a comp opens, and its licence tag (`cost::licence_of`).
//! Its inference code reaches a file through [`ready_path`] and nothing else,
//! and does no work (passes its input through) until that answers `Ok`.
//! CLAUDE.md "Adding a node that needs weights" is the checklist.
//!
//! Three rules the rest of the app relies on:
//!
//! - **Nothing downloads itself.** A fetch only ever starts from an explicit
//!   click. A missing model leaves its node inert with a labelled button,
//!   never a surprise 200 MB on someone's connection (manifesto rule 8: the
//!   slow path is conspicuous).
//! - **The cache is shared and content-addressed by hash.** Two nodes
//!   wanting the same weights download once, and a half-finished file
//!   resumes rather than restarting.
//! - **A file is only "ready" once its SHA-256 matches.** Downloads land on
//!   `<file>.part` and are renamed into place after verification, so an
//!   interrupted fetch can never be mistaken for a good model.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
#[cfg_attr(target_arch = "wasm32", allow(unused_imports))]
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// What the downloaded bytes actually are. Wobble's own containers load
/// directly; a foreign checkpoint is the upstream release, and whoever
/// consumes it says how it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A Wobble container (`.wifw`, `.wtpw`) — loads as-is.
    Native,
    /// A PyTorch `.pt`/`.pkl` checkpoint.
    Torch,
    /// A `.safetensors` file: no code in the format, read directly
    /// (`crate::safetensors`).
    Safetensors,
    /// A JAX/numpy `.npy` checkpoint.
    Numpy,
    /// An ONNX graph, run by the runtime rather than ported to WGSL.
    Onnx,
}

/// One downloadable file. Everything here is fixed at build time so the
/// panel can show the size and the licence BEFORE anything is fetched.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// Stable id — the setup file stores this, never a path.
    pub id: &'static str,
    /// What the panel calls it.
    pub label: &'static str,
    /// What it is for, as the Models panel groups it ("tracking",
    /// "segmentation"). A heading, not a type: nothing branches on it —
    /// which node uses a model is the node's own declaration.
    pub family: &'static str,
    pub format: Format,
    pub url: &'static str,
    /// Filename inside the cache directory.
    pub file: &'static str,
    /// Exact size, so the button can say what it will cost before you click.
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the finished file. Empty = unverified: the
    /// download still works, but `State::Ready` cannot be trusted, and the
    /// panel says so. Fill it in from `wobble-models hash <id>`.
    pub sha256: &'static str,
    /// Upstream licence, shown before the download starts.
    pub license: &'static str,
    /// Is that licence free for commercial use?
    ///
    /// Structured beside the free-text `license` because the graph REASONS
    /// about this: a node loading non-commercial weights is tagged, and
    /// everything downstream of it is warned (manifesto rule 10). Free text
    /// cannot be propagated; a bool can. Set it from the actual licence, and
    /// when in doubt set it false — an unnecessary warning costs a tooltip,
    /// a missing one costs somebody a delivery.
    pub commercial: bool,
    /// Where it comes from, shown next to the licence.
    pub source: &'static str,
    /// One line on what it is good for.
    pub about: &'static str,
}

impl ModelSpec {
    /// Where this model lives once fetched.
    pub fn path(&self) -> PathBuf {
        self.path_in(&models_dir())
    }

    /// The partial download alongside it.
    pub fn part_path(&self) -> PathBuf {
        self.part_path_in(&models_dir())
    }

    /// Same, against an explicit cache directory — the form the tests use,
    /// so checking cache behaviour never has to mutate the environment.
    pub fn path_in(&self, dir: &Path) -> PathBuf {
        dir.join(self.file)
    }

    pub fn part_path_in(&self, dir: &Path) -> PathBuf {
        let mut p = self.path_in(dir).into_os_string();
        p.push(".part");
        PathBuf::from(p)
    }

    /// Whether a hash was pinned at build time (see `sha256`).
    pub fn verifiable(&self) -> bool {
        !self.sha256.is_empty()
    }

    /// "218.9 MB" — for buttons and progress text.
    pub fn size_label(&self) -> String {
        human_bytes(self.bytes)
    }
}

/// The registry. Adding a model is adding a row here; nothing else in the
/// app knows a URL (manifesto rule 5 — registries over enums).
pub const CATALOG: &[ModelSpec] = &[
    ModelSpec {
        id: "searaft-m",
        label: "SEA-RAFT M",
        family: "motion",
        // The C+T checkpoint: TartanAir, FlyingChairs, FlyingThings. The
        // KITTI-tuned ones are better on driving footage and trained on
        // non-commercial data; this is the one with no such question.
        format: Format::Safetensors,
        url: "https://huggingface.co/MemorySlices/Tartan-C-T432x960-M/resolve/0ab5da9ac5e6b9c174f420b876ae3d85449c07dd/model.safetensors",
        file: "searaft_m_ct.safetensors",
        bytes: 78_778_760,
        sha256: "41f9a087377a4d3578e57e461cc07642a41dfe5fde992215d4d26288ddba27c7",
        license: "BSD-3-Clause (princeton-vl/SEA-RAFT)",
        commercial: true,
        source: "huggingface.co/MemorySlices (the authors)",
        about: "optical flow, accurate — the Flow node's default engine",
    },
    ModelSpec {
        id: "rife-4.6",
        label: "RIFE 4.6 (IFNet)",
        family: "motion",
        // The official train_log `flownet.pkl`, read in Rust by `torchpt`.
        // Upstream ships it only as a Google Drive zip; this is a
        // byte-identical copy on Hugging Face, pinned to a commit and to
        // the hash of the file the converter was validated against.
        // A mirror this project controls should replace it.
        format: Format::Torch,
        url: "https://huggingface.co/Jacid23/third-eye-models/resolve/96963d1f0cce7b02030ac892f0969941af9774ec/flownet.pkl",
        file: "rife46_flownet.pkl",
        bytes: 21_273_159,
        sha256: "008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db",
        license: "MIT (hzwer/Practical-RIFE)",
        commercial: true,
        source: "github.com/hzwer/Practical-RIFE (v4.6 weights)",
        about: "optical flow between frames — the Flow node's motion vectors",
    },
    ModelSpec {
        id: "tapir-causal",
        label: "TAPIR (causal)",
        family: "tracking",
        format: Format::Numpy,
        url: "https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy",
        file: "causal_tapir_checkpoint.npy",
        bytes: 127_043_639,
        sha256: "",
        license: "Apache-2.0 (google-deepmind/tapnet)",
        commercial: true,
        source: "storage.googleapis.com/dm-tapnet",
        about: "point tracking, online/streaming — the small one",
    },
    ModelSpec {
        id: "bootstapir-causal",
        label: "BootsTAPIR (causal)",
        family: "tracking",
        // The .npy (JAX) release rather than the .pt: same weights, and it
        // converts with numpy alone — no torch, which has no wheels for the
        // python this machine ships.
        format: Format::Numpy,
        url: "https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.npy",
        file: "causal_bootstapir_checkpoint.npy",
        bytes: 218_818_353,
        sha256: "7e1536f68d2ea5d90ac2b705ee621ae2874788b378596f90c7f24857146a5585",
        license: "Apache-2.0 (google-deepmind/tapnet)",
        commercial: true,
        source: "storage.googleapis.com/dm-tapnet",
        about: "point tracking, online/streaming — bootstrapped, the accurate one",
    },
    // SAM 2 comes as a PAIR: the encoder turns a frame into an embedding
    // (expensive, once per frame), the decoder turns that embedding plus your
    // clicks into a mask (cheap, once per edit). Both are needed for the node
    // to do anything, which is why they are listed together and the panel
    // shows the total.
    ModelSpec {
        id: "sam2-tiny-encoder",
        label: "SAM 2 tiny — encoder",
        family: "segmentation",
        format: Format::Onnx,
        url: "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_tiny.encoder.onnx",
        file: "sam2_hiera_tiny.encoder.onnx",
        bytes: 134_261_315,
        sha256: "",
        license: "Apache-2.0 (facebookresearch/sam2)",
        commercial: true,
        source: "huggingface.co/vietanhdev (ONNX export)",
        about: "frame -> embedding; the expensive half, cached per frame",
    },
    ModelSpec {
        id: "sam2-tiny-decoder",
        label: "SAM 2 tiny — decoder",
        family: "segmentation",
        format: Format::Onnx,
        url: "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_tiny.decoder.onnx",
        file: "sam2_hiera_tiny.decoder.onnx",
        bytes: 20_640_886,
        sha256: "",
        license: "Apache-2.0 (facebookresearch/sam2)",
        commercial: true,
        source: "huggingface.co/vietanhdev (ONNX export)",
        about: "embedding + clicks -> mask; cheap, runs on every edit",
    },
    // The Upscale node's three models, each read straight from the
    // upstream .pth by `torchpt` (no conversion) and run in our own WGSL.
    ModelSpec {
        id: "realesr-general-x4v3",
        label: "Real-ESRGAN compact (general x4 v3)",
        family: "upscaling",
        format: Format::Torch,
        url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        file: "realesr-general-x4v3.pth",
        bytes: 4_885_111,
        sha256: "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        license: "BSD-3-Clause (xinntao/Real-ESRGAN)",
        commercial: true,
        source: "github.com/xinntao/Real-ESRGAN releases",
        about: "x4 super-resolution, the fast general-purpose one",
    },
    // HDRCNN, from the authors' own site (no Hugging Face mirror): the
    // TensorLayer npz, read by `npy::read_npz`. The regularized one is the
    // node's default (steady over video, robust to JPEG).
    ModelSpec {
        id: "hdrcnn-compr-regularized",
        label: "HDRCNN (video, temporally regularized)",
        family: "highlights",
        format: Format::Numpy,
        url: "https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params_compr_regularized.npz",
        file: "hdrcnn_params_compr_regularized.npz",
        bytes: 177_599_660,
        sha256: "da7fe19280286ab3a145338d08b12905fd9bd785adbcf95c06c0e1986e5fdfed",
        license: "BSD-3-Clause (gabrieleilertsen/hdrcnn)",
        commercial: true,
        source: "computergraphics.on.liu.se/hdrcnn (the authors)",
        about: "clipped highlights reconstructed, steady frame to frame and robust to compression",
    },
    ModelSpec {
        id: "hdrcnn",
        label: "HDRCNN (original)",
        family: "highlights",
        format: Format::Numpy,
        url: "https://computergraphics.on.liu.se/hdrcnn/material/hdrcnn_params.npz",
        file: "hdrcnn_params.npz",
        bytes: 177_242_327,
        sha256: "1fa6d3d7a907cd48cbfbef4e572d140d74a2aa1b351b12c84a4b717bee655917",
        license: "BSD-3-Clause (gabrieleilertsen/hdrcnn)",
        commercial: true,
        source: "computergraphics.on.liu.se/hdrcnn (the authors)",
        about: "clipped highlights reconstructed, the paper's parameters: a little more detail on clean stills",
    },
    ModelSpec {
        id: "realesrgan-x4plus",
        label: "Real-ESRGAN x4plus",
        family: "upscaling",
        format: Format::Torch,
        url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        file: "RealESRGAN_x4plus.pth",
        bytes: 67_040_989,
        sha256: "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        license: "BSD-3-Clause (xinntao/Real-ESRGAN)",
        commercial: true,
        source: "github.com/xinntao/Real-ESRGAN releases",
        about: "x4 super-resolution, RRDB — the quality one, and slow",
    },
    // SPAN's official weights are only on Google Drive, which has no
    // stable direct URL; this is Philip Hofmann's universal x4 SPAN,
    // fine-tuned from the official spanx4_ch48, on a GitHub release.
    ModelSpec {
        id: "span-nomosuni-x4",
        label: "SPAN x4 (4xNomosUni_span_multijpg)",
        family: "upscaling",
        format: Format::Torch,
        url: "https://github.com/Phhofm/models/releases/download/4xNomosUni_span_multijpg/4xNomosUni_span_multijpg.pth",
        file: "4xNomosUni_span_multijpg.pth",
        bytes: 4_546_346,
        sha256: "3a9037c36de90e7825c030176c8e193dfd7897ef4b5df91b8bdc59ffb6ab65ca",
        license: "CC-BY-4.0 (Philip Hofmann; credit required). Architecture: SPAN, Apache-2.0",
        commercial: true,
        source: "github.com/Phhofm/models releases",
        about: "x4 super-resolution, SPAN — fast, trained with JPEG and resize damage",
    },
    // The Depth node's four networks (Video Depth Anything), each the
    // upstream .pth on the authors' Hugging Face, pinned to a commit. The
    // tool loads them (`torch.load(weights_only=True)`); the app only
    // passes the path. Licences read off each model card: the small ones
    // are Apache-2.0, the large ones CC-BY-NC-4.0.
    ModelSpec {
        id: "vda-small",
        label: "Video Depth Anything small",
        family: "depth",
        format: Format::Torch,
        url: "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/256875362cff76724b920335dfb4b29dd611f66e/video_depth_anything_vits.pth",
        file: "video_depth_anything_vits.pth",
        bytes: 116_440_756,
        sha256: "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
        license: "Apache-2.0 (DepthAnything/Video-Depth-Anything)",
        commercial: true,
        source: "huggingface.co/depth-anything (the authors)",
        about: "temporally steady relative depth for a clip — the Depth node's default",
    },
    ModelSpec {
        id: "vda-small-metric",
        label: "Video Depth Anything small, metric",
        family: "depth",
        format: Format::Torch,
        url: "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/273d090f2ce17df50c2872d82c8322c45da5b4dd/metric_video_depth_anything_vits.pth",
        file: "metric_video_depth_anything_vits.pth",
        bytes: 116_444_063,
        sha256: "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
        license: "Apache-2.0 (DepthAnything/Video-Depth-Anything)",
        commercial: true,
        source: "huggingface.co/depth-anything (the authors)",
        about: "temporally steady depth in metres (trained on Virtual KITTI + IRS)",
    },
    ModelSpec {
        id: "vda-large",
        label: "Video Depth Anything large",
        family: "depth",
        format: Format::Torch,
        url: "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/7aafbcb5c6af0bac741aad2b6471894fb4761afa/video_depth_anything_vitl.pth",
        file: "video_depth_anything_vitl.pth",
        bytes: 1_538_392_012,
        sha256: "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
        license: "CC-BY-NC-4.0 (DepthAnything/Video-Depth-Anything)",
        commercial: false,
        source: "huggingface.co/depth-anything (the authors)",
        about: "relative depth, the most accurate — research and non-commercial use only",
    },
    ModelSpec {
        id: "vda-large-metric",
        label: "Video Depth Anything large, metric",
        family: "depth",
        format: Format::Torch,
        url: "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/607fcdbd454b95c3bd39abbd3054142869a527d3/metric_video_depth_anything_vitl.pth",
        file: "metric_video_depth_anything_vitl.pth",
        bytes: 1_538_348_216,
        sha256: "24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620",
        license: "CC-BY-NC-4.0 (DepthAnything/Video-Depth-Anything)",
        commercial: false,
        source: "huggingface.co/depth-anything (the authors)",
        about: "depth in metres, the most accurate — research and non-commercial use only",
    },
    // The Depth node's Depth Pro 2 sizes (Apple) and the focal-length
    // network they share, each the safetensors on Apple's Hugging Face,
    // pinned to a commit. Every one is the Apple Machine Learning Research
    // Model licence: research only, NO commercial use (rule 10: tagged).
    ModelSpec {
        id: "dp2-s",
        label: "Depth Pro 2 S",
        family: "depth",
        format: Format::Safetensors,
        url: "https://huggingface.co/apple/DepthPro2-S/resolve/903a1c1ade918831d138adcc4aedbcc8f4da0078/model.safetensors",
        file: "depthpro2-s.safetensors",
        bytes: 97_264_508,
        sha256: "583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c",
        license: "Apple ML Research Model licence, research only (apple/DepthPro2-S)",
        commercial: false,
        source: "huggingface.co/apple (the authors)",
        about: "metric depth for a clip, the smallest and fastest (24M) — research and non-commercial use only",
    },
    ModelSpec {
        id: "dp2-b",
        label: "Depth Pro 2 B",
        family: "depth",
        format: Format::Safetensors,
        url: "https://huggingface.co/apple/DepthPro2-B/resolve/85e67fdfb7f5b36a6f6c4495e6789d331b1d8a04/model.safetensors",
        file: "depthpro2-b.safetensors",
        bytes: 386_076_596,
        sha256: "244ff85e0d9bde6e0f985de1c61420abc4c2110265b495c974556225ac2b60ed",
        license: "Apple ML Research Model licence, research only (apple/DepthPro2-B)",
        commercial: false,
        source: "huggingface.co/apple (the authors)",
        about: "metric depth for a clip (97M) — research and non-commercial use only",
    },
    ModelSpec {
        id: "dp2-l",
        label: "Depth Pro 2 L",
        family: "depth",
        format: Format::Safetensors,
        url: "https://huggingface.co/apple/DepthPro2-L/resolve/849dc5461267e0bc006ec63ab46920ea1ed98702/model.safetensors",
        file: "depthpro2-l.safetensors",
        bytes: 1_335_873_268,
        sha256: "d799d8a91f2e8dc91715f9b33a405a1ff0727212041d206ebbb04e6c1424ff9f",
        license: "Apple ML Research Model licence, research only (apple/DepthPro2-L)",
        commercial: false,
        source: "huggingface.co/apple (the authors)",
        about: "metric depth for a clip, sharper (334M) — research and non-commercial use only",
    },
    ModelSpec {
        id: "dp2-h",
        label: "Depth Pro 2 H+",
        family: "depth",
        format: Format::Safetensors,
        url: "https://huggingface.co/apple/DepthPro2-H/resolve/68426c789f75a7c760273e857222625b4fe69576/model.safetensors",
        file: "depthpro2-h.safetensors",
        bytes: 3_488_493_012,
        sha256: "cfc532336019abd2128b04be2c2967175522e46de69222ef72b87c6675ee3b17",
        license: "Apple ML Research Model licence, research only (apple/DepthPro2-H)",
        commercial: false,
        source: "huggingface.co/apple (the authors)",
        about: "metric depth for a clip, the most accurate (872M) — research and non-commercial use only",
    },
    ModelSpec {
        id: "dp2-focal",
        label: "Depth Pro 2 focal length",
        family: "depth",
        format: Format::Safetensors,
        url: "https://huggingface.co/apple/DepthPro2-focal/resolve/622659d44a2002cf9ae4333f2569afa938290d9a/model.safetensors",
        file: "depthpro2-focal.safetensors",
        bytes: 93_840_880,
        sha256: "8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86",
        license: "Apple ML Research Model licence, research only (apple/DepthPro2-focal)",
        commercial: false,
        source: "huggingface.co/apple (the authors)",
        about: "the focal-length estimate every Depth Pro 2 size needs (metres need a focal) — research and non-commercial use only",
    },
];

/// Look a model up by id.
pub fn spec(id: &str) -> Option<&'static ModelSpec> {
    CATALOG.iter().find(|m| m.id == id)
}

/// The panel's headings, in catalog order.
pub fn families() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for m in CATALOG {
        if !out.contains(&m.family) {
            out.push(m.family);
        }
    }
    out
}

/// The whole catalog.
pub fn all() -> &'static [ModelSpec] {
    CATALOG
}

/// Specs for a list of ids, deduplicated, in the order given. An id that is
/// not catalogued is skipped — `cost`'s tests fail on one a node declares.
pub fn specs<'a>(ids: impl IntoIterator<Item = &'a str>) -> Vec<&'static ModelSpec> {
    let mut out: Vec<&'static ModelSpec> = Vec::new();
    for id in ids {
        if let Some(m) = spec(id) {
            if !out.iter().any(|o| o.id == m.id) {
                out.push(m);
            }
        }
    }
    out
}

/// Those of `ids` that are not on disk and ready — what a node's gate and
/// the setup prompt offer to fetch. Size check only (see `state`).
pub fn missing<'a>(ids: impl IntoIterator<Item = &'a str>) -> Vec<&'static ModelSpec> {
    missing_in(&models_dir(), ids)
}

/// `missing` against an explicit cache directory.
pub fn missing_in<'a>(
    dir: &Path,
    ids: impl IntoIterator<Item = &'a str>,
) -> Vec<&'static ModelSpec> {
    specs(ids).into_iter().filter(|m| state_in(dir, m) != State::Ready).collect()
}

/// Total bytes of a set of specs — for "download all (430 MB)".
pub fn total_bytes(specs: &[&ModelSpec]) -> u64 {
    specs.iter().map(|m| m.bytes).sum()
}

/// The ONE way inference code gets at a weights file: the path, if and only
/// if the model is catalogued and ready. The error is written for a person —
/// a node shows it as its status line.
pub fn ready_path(id: &str) -> Result<PathBuf> {
    ready_path_in(&models_dir(), id)
}

/// `ready_path` against an explicit cache directory.
pub fn ready_path_in(dir: &Path, id: &str) -> Result<PathBuf> {
    let Some(m) = spec(id) else {
        bail!("no model \"{id}\" in the catalog");
    };
    match state_in(dir, m) {
        State::Ready => Ok(m.path_in(dir)),
        State::Partial { .. } => bail!("{} is part-downloaded — resume it in the node's panel", m.label),
        State::Bad(why) => bail!("{}: {why}", m.label),
        State::Missing => bail!("{} is not downloaded ({}) — the node's panel has the button", m.label, m.size_label()),
    }
}

/// Are all of these models free for commercial use? Uncatalogued ids answer
/// false: an unknown licence is not a permissive one.
pub fn all_commercial(ids: &[&str]) -> bool {
    ids.iter().all(|id| spec(id).is_some_and(|m| m.commercial))
}

/// Where downloaded weights live: `$WOBBLE_MODELS_DIR`, else
/// `~/.wobble/models` — beside `~/.wobble/plugins`, the other thing a user
/// installs rather than builds.
pub fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    // `paths::home` falls back to USERPROFILE: a Windows app launched from
    // Explorer has no HOME, and the cache landed beside the executable.
    match crate::paths::home() {
        Some(h) => h.join(".wobble").join("models"),
        None => PathBuf::from(".wobble").join("models"),
    }
}

/// What the cache holds for one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    /// Nothing on disk.
    Missing,
    /// An interrupted download that `fetch` will resume.
    Partial { have: u64 },
    /// On disk and the right size (and hash, when one is pinned).
    Ready,
    /// On disk and wrong — the message says how.
    Bad(String),
}

/// Cheap check: size only. Hashing 200 MB on every UI frame is exactly the
/// quiet slow path this project refuses, so `Ready` here means "the right
/// number of bytes"; the hash is checked once, when the download lands.
pub fn state(spec: &ModelSpec) -> State {
    state_in(&models_dir(), spec)
}

/// `state` against an explicit cache directory.
pub fn state_in(dir: &Path, spec: &ModelSpec) -> State {
    match std::fs::metadata(spec.path_in(dir)) {
        Ok(m) if spec.bytes == 0 || m.len() == spec.bytes => State::Ready,
        Ok(m) => State::Bad(format!(
            "{} on disk, expected {} — delete it and fetch again",
            human_bytes(m.len()),
            spec.size_label()
        )),
        Err(_) => match std::fs::metadata(spec.part_path_in(dir)) {
            Ok(m) => State::Partial { have: m.len() },
            Err(_) => State::Missing,
        },
    }
}

/// Progress from the worker thread.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
enum Msg {
    /// Bytes on disk so far.
    Progress(u64),
    Done,
    Error(String),
}

/// A running download. Shaped like the bake jobs so the panel polls it the
/// same way: drain each frame, read the fields, drop to cancel.
pub struct Download {
    rx: std::sync::mpsc::Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    pub spec: &'static ModelSpec,
    /// Bytes written, including whatever a resumed `.part` already held.
    pub have: u64,
    pub total: u64,
    pub finished: bool,
    pub error: Option<String>,
}

impl Drop for Download {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Download {
    /// Drain the worker. Call once a frame; returns true while it is still
    /// running (so the caller knows to keep repainting).
    pub fn poll(&mut self) -> bool {
        for msg in self.rx.try_iter() {
            match msg {
                Msg::Progress(n) => self.have = n,
                Msg::Done => {
                    self.have = self.total;
                    self.finished = true;
                }
                Msg::Error(e) => {
                    self.error = Some(e);
                    self.finished = true;
                }
            }
        }
        !self.finished
    }

    /// 0..1, or None when the server never said how big it was.
    pub fn fraction(&self) -> Option<f32> {
        (self.total > 0).then(|| (self.have as f32 / self.total as f32).clamp(0.0, 1.0))
    }

    /// Stop the worker. The `.part` file stays, so a later fetch resumes.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Start fetching a model. Resumes an existing `.part` with a Range request,
/// verifies the SHA-256 when one is pinned, and only then renames the file
/// into place — an interrupted fetch can never look like a good model.
pub fn fetch(spec: &'static ModelSpec) -> Download {
    fetch_into(models_dir(), spec)
}

/// `fetch` against an explicit cache directory — the form the tests drive,
/// so download behaviour can be checked without touching a real cache.
pub fn fetch_into(dir: PathBuf, spec: &'static ModelSpec) -> Download {
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    let have0 = match state_in(&dir, spec) {
        State::Partial { have } => have,
        _ => 0,
    };
    // No threads in a page (web/): there `run` only reports that downloads
    // are unavailable, so it may as well report it right here.
    #[cfg(not(target_arch = "wasm32"))]
    let spawn = |f| drop(std::thread::spawn(f));
    #[cfg(target_arch = "wasm32")]
    let spawn = |f: Box<dyn FnOnce()>| f();
    spawn(Box::new(move || {
        if let Err(e) = run(&dir, spec, &tx, &stop) {
            // A cancel unwinds through the same path; don't report it as a
            // failure the user has to read.
            if !stop.load(Ordering::Relaxed) {
                let _ = tx.send(Msg::Error(format!("{e:#}")));
            }
        } else {
            let _ = tx.send(Msg::Done);
        }
    }));
    Download {
        rx,
        cancel,
        spec,
        have: have0,
        total: spec.bytes,
        finished: false,
        error: None,
    }
}

/// The web build has no fetcher (no ureq on wasm32): say so as the job's error.
#[cfg(target_arch = "wasm32")]
fn run(_: &Path, _: &ModelSpec, _: &std::sync::mpsc::Sender<Msg>, _: &AtomicBool) -> Result<()> {
    bail!("model downloads are not available in the web build")
}

/// The worker body: resume, stream, hash, verify, rename.
#[cfg(not(target_arch = "wasm32"))]
fn run(
    dir: &Path,
    spec: &ModelSpec,
    tx: &std::sync::mpsc::Sender<Msg>,
    stop: &AtomicBool,
) -> Result<()> {
    let dest = spec.path_in(dir);
    let part = spec.part_path_in(dir);
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create {}", dir.display()))?;
    }
    // Resume only from a plausible partial: anything at or past the known
    // size is junk, and restarting is cheaper than explaining it.
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if spec.bytes > 0 && have >= spec.bytes {
        let _ = std::fs::remove_file(&part);
        have = 0;
    }

    let mut req = ureq::get(spec.url)
        // Transparent gzip would make Content-Length and the resume offset
        // disagree with the bytes on disk.
        .header("accept-encoding", "identity");
    if have > 0 {
        req = req.header("range", &format!("bytes={have}-"));
    }
    let resp = req.call().with_context(|| format!("GET {}", spec.url))?;
    let status = resp.status().as_u16();
    // 206 honours the resume; 200 means the server ignored it and is sending
    // the whole file, so the partial has to go.
    let resuming = status == 206;
    if !(status == 200 || status == 206) {
        bail!("{} returned HTTP {status}", spec.url);
    }
    if have > 0 && !resuming {
        have = 0;
    }
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|n| n + have)
        .unwrap_or(spec.bytes);
    if spec.bytes > 0 && total != spec.bytes {
        bail!(
            "{} is {} on the server, expected {} — the registry is out of date",
            spec.url,
            human_bytes(total),
            spec.size_label()
        );
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&part)
        .with_context(|| format!("open {}", part.display()))?;
    // Hash what we are resuming on top of, so the check at the end covers
    // the whole file and not just this session's bytes.
    let mut hasher = Sha256::new();
    if have > 0 {
        file.seek(SeekFrom::Start(0))?;
        let mut buf = vec![0u8; 1 << 20];
        let mut left = have;
        while left > 0 {
            let n = file.read(&mut buf[..(1 << 20).min(left as usize)])?;
            if n == 0 {
                // The file shrank under us; start clean rather than hash a
                // prefix that no longer describes what is there.
                have = 0;
                hasher = Sha256::new();
                break;
            }
            hasher.update(&buf[..n]);
            left -= n as u64;
        }
    }
    file.set_len(have)?;
    file.seek(SeekFrom::Start(have))?;

    let mut body = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut last_report = have;
    loop {
        if stop.load(Ordering::Relaxed) {
            file.flush()?;
            return Ok(());
        }
        let n = body.read(&mut buf).context("read response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write .part")?;
        hasher.update(&buf[..n]);
        have += n as u64;
        // One message per MB, not per chunk: the UI only redraws at frame
        // rate and the channel should not become the bottleneck.
        if have - last_report >= 1 << 20 {
            last_report = have;
            let _ = tx.send(Msg::Progress(have));
        }
    }
    file.flush()?;
    drop(file);
    let _ = tx.send(Msg::Progress(have));

    if total > 0 && have != total {
        bail!(
            "download ended early: {} of {}",
            human_bytes(have),
            human_bytes(total)
        );
    }
    if spec.verifiable() {
        let got = hex(&hasher.finalize());
        if got != spec.sha256 {
            // Keep the bad bytes out of the cache entirely — a wrong file
            // that resumes forever is worse than one that redownloads.
            let _ = std::fs::remove_file(&part);
            bail!("SHA-256 mismatch: got {got}, expected {}", spec.sha256);
        }
    }
    std::fs::rename(&part, &dest)
        .with_context(|| format!("move into place: {}", dest.display()))?;
    Ok(())
}

/// SHA-256 of a file already on disk — for filling in a registry row, and
/// for verifying a model the user placed by hand.
pub fn hash_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Sizes as a person reads them. Binary steps, decimal-ish labels — the same
/// lie every download UI tells, and the one users expect.
pub fn human_bytes(n: u64) -> String {
    const K: f64 = 1024.0;
    let n = n as f64;
    if n < K {
        return format!("{n:.0} B");
    }
    let units = ["KB", "MB", "GB", "TB"];
    let mut v = n / K;
    for (i, u) in units.iter().enumerate() {
        if v < K || i == units.len() - 1 {
            return if v < 10.0 {
                format!("{v:.1} {u}")
            } else {
                format!("{v:.0} {u}")
            };
        }
        v /= K;
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_files_do_not_collide() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|m| m.id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate model id in CATALOG");
        let mut files: Vec<&str> = CATALOG.iter().map(|m| m.file).collect();
        files.sort_unstable();
        files.dedup();
        assert_eq!(files.len(), n, "two models share a cache filename");
    }

    #[test]
    fn every_row_is_fetchable_and_labelled() {
        for m in CATALOG {
            assert!(!m.label.is_empty(), "{}: no label", m.id);
            assert!(!m.license.is_empty(), "{}: no licence to show", m.id);
            assert!(!m.about.is_empty(), "{}: nothing said about it", m.id);
            assert!(
                m.url.starts_with("https://"),
                "{}: weights must come over TLS",
                m.id
            );
            assert!(
                m.sha256.is_empty() || m.sha256.len() == 64,
                "{}: sha256 must be 64 hex chars or empty",
                m.id
            );
        }
    }

    #[test]
    fn part_file_sits_beside_the_real_one() {
        let m = &CATALOG[0];
        assert_eq!(
            m.part_path().file_name().unwrap().to_str().unwrap(),
            format!("{}.part", m.file)
        );
        assert_eq!(m.path().parent(), m.part_path().parent());
    }

    /// What a node's gate and the setup prompt see: missing dedups and
    /// skips unknown ids, `ready_path` refuses until the file is right, then
    /// hands back exactly the cache path. A sparse file of the catalogued
    /// size stands in for 200 MB of weights (state is a size check).
    #[test]
    fn missing_and_ready_path_follow_the_cache() {
        let dir = std::env::temp_dir()
            .join(format!("wobble-models-ready-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let id = "bootstapir-causal";
        let m = spec(id).unwrap();

        let miss = missing_in(&dir, [id, id, "no-such-model"]);
        assert_eq!(miss.len(), 1, "deduped, unknown id skipped");
        assert_eq!(total_bytes(&miss), m.bytes);
        let e = ready_path_in(&dir, id).unwrap_err().to_string();
        assert!(e.contains("not downloaded"), "{e}");
        assert!(ready_path_in(&dir, "no-such-model").is_err());

        let f = std::fs::File::create(m.path_in(&dir)).unwrap();
        f.set_len(12).unwrap();
        assert!(ready_path_in(&dir, id).is_err(), "wrong size is not ready");
        f.set_len(m.bytes).unwrap();
        assert_eq!(ready_path_in(&dir, id).unwrap(), m.path_in(&dir));
        assert!(missing_in(&dir, [id]).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_ids_are_not_commercial() {
        assert!(all_commercial(&[]), "no weights, no restriction");
        assert!(all_commercial(&["bootstapir-causal"]));
        assert!(!all_commercial(&["no-such-model"]));
    }

    #[test]
    fn state_reads_the_cache_without_hashing() {
        let dir = std::env::temp_dir().join(format!("wobble-models-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec = ModelSpec {
            id: "test",
            label: "test",
            family: "tracking",
            format: Format::Native,
            url: "https://example.invalid/x",
            file: "state-test.bin",
            bytes: 8,
            sha256: "",
            license: "-",
            commercial: true,
            source: "-",
            about: "-",
        };
        let _ = std::fs::remove_file(spec.path_in(&dir));
        let _ = std::fs::remove_file(spec.part_path_in(&dir));
        assert_eq!(state_in(&dir, &spec), State::Missing);
        std::fs::write(spec.part_path_in(&dir), b"abc").unwrap();
        assert_eq!(state_in(&dir, &spec), State::Partial { have: 3 });
        std::fs::write(spec.path_in(&dir), b"12345678").unwrap();
        assert_eq!(state_in(&dir, &spec), State::Ready);
        std::fs::write(spec.path_in(&dir), b"123").unwrap();
        assert!(matches!(state_in(&dir, &spec), State::Bad(_)));
        let _ = std::fs::remove_file(spec.path_in(&dir));
        let _ = std::fs::remove_file(spec.part_path_in(&dir));
    }

    #[test]
    fn hash_file_matches_a_known_vector() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("wobble-hash-{}.bin", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            hash_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn sizes_read_the_way_a_download_dialog_reads() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(218_887_028), "209 MB");
    }
}
