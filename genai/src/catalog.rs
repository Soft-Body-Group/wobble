//! Remote models: the registry a Generate step names by id.
//!
//! The sibling of `engine::models::CATALOG` for weights we never download.
//! Same three rules:
//!
//! - **A setup names an id, never an endpoint.** The endpoint, the task and
//!   the licence live here, so a comp cannot point the app at a server.
//! - **New endpoint = new id.** Never re-point an existing id at a different
//!   model: an old comp would silently re-render as something else on its
//!   next run.
//! - **`commercial` is the source of truth for rule 10**, read by
//!   `engine::cost::licence_of`. It answers "may the pixels this model
//!   returns THROUGH THIS PROVIDER be used commercially?" — which is not
//!   always the weights' licence (a provider can hold a commercial agreement
//!   the open weights do not grant). When in doubt it is `false`: an
//!   unnecessary warning costs a tooltip, a missing one costs a delivery.
//!
//! There are no prices here. A price in the binary is stale the day it
//! ships; the quote comes from the provider (fal's pricing API) or from the
//! relay, at the moment you are about to press run.

/// Which service answers an endpoint. Every entry here is fal-shaped today:
/// fal directly with your key, or the wobble relay with credits, which
/// forwards to the same endpoint and hands back the same output JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Service {
    Fal,
}

/// What a model does, which fixes its ports and how its request is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// prompt -> image
    TextToImage,
    /// prompt + image -> image
    Edit,
    /// prompt + image + mask (white = replace) -> image
    Inpaint,
    /// image -> a bigger image
    Upscale,
    /// image -> the same image with the background cut out (an alpha)
    Cutout,
    /// prompt + start frame -> a VIDEO (an mp4 take, played back as a
    /// movie). Offered only by a build that can decode one.
    ImageToVideo,
    /// image -> its depth, as a grey picture of RAW values (a data take:
    /// no sRGB transform on the way in). Relative, not metric.
    Depth,
    /// image -> its surface normals, RAW (`n * 0.5 + 0.5` in rgb, the
    /// usual encoding), a data take like Depth.
    Normals,
    /// image -> its albedo (the lighting taken out), a picture.
    Albedo,
    /// prompt + image -> the image relit as the prompt says.
    Relight,
    /// prompt + a CLIP (the node's input over a range) -> the clip edited
    /// as the prompt says. A video take.
    VideoEdit,
    /// prompt + clip + matte clip (white = regenerate) -> the clip with
    /// the matted region filled: object removal on video. A video take.
    VideoInpaint,
}

/// How a model's request is shaped, beyond its task. Most fal endpoints
/// share FLUX's field names; the ones that do not say so here, so the
/// differences live in one match in `run::request` rather than in a string
/// test on the endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// `image_url`, `seed`, `image_size {width, height}`.
    Flux,
    /// ByteDance Seedream: no `seed`, an edit takes `image_urls` (a list),
    /// and the output area must lie between 1024² and 2048² — we keep it
    /// at or under 1536², the per-image price's lower tier, so the quote is
    /// the bill.
    Seedream,
    /// ByteDance Seedance image-to-video: no `seed`, `duration` and
    /// `resolution` are strings, audio off (a take is pictures).
    Seedance { max_seconds: u32, top_res: u32 },
    /// Google's Gemini image models (Nano Banana): no width or height. The
    /// size is an `aspect_ratio` from a fixed list plus a `resolution`
    /// class ("1K", "2K"), and an edit takes `image_urls`. Takes a seed.
    Gemini,
    /// OpenAI's GPT Image: no `seed`, an edit or inpaint takes
    /// `image_urls` (+ `mask_url`, white = replace), and a text-to-image
    /// size must hold at least 655,360 px. Billed by tokens, which fal's
    /// pricing API does not turn into a price, so it quotes as unknown.
    GptImage,
    /// Microsoft MoGe (2 and 3): geometry from one image. We ask for the
    /// normals variant, skip the mesh and point-cloud exports (time, and
    /// files nobody reads), and take `normal_map`. Its `depth_map` is a
    /// COLOURISED visualisation, so it is never offered as depth.
    Moge,
    /// Marigold v2: one endpoint, `prediction_type` picks depth, normals or
    /// albedo; the result is `image`.
    Marigold { prediction: &'static str },
    /// fal's image preprocessors (Depth Anything v2, MiDaS): `image_url`
    /// only. MiDaS hands back `depth_map` AND `normal_map`.
    Preprocessor,
    /// IC-Light v2: `prompt`, `image_url`, `seed`; the light's side goes in
    /// extra (`initial_latent`: Left/Right/Top/Bottom).
    IcLight,
    /// fal's relighting app: `image_url` and a `lighting_style` from a
    /// fixed list, read off the prompt ("golden hour" -> golden_hour).
    LightingStyle,
    /// Kling O3 video edit: `prompt`, `video_url` (mp4/mov, 3-15 s, 720 px
    /// or more), source audio dropped. No seed.
    KlingEdit,
    /// Decart Lucy Edit: `prompt`, `video_url`, `seed`.
    LucyEdit,
    /// LTX 2.3 inpaint: `prompt`, `video_url`, `mask_video_url` (white =
    /// regenerate), `num_frames` (9-481) and `frames_per_second` from the
    /// clip, `seed`; audio and prompt expansion off.
    LtxInpaint,
}

impl Dialect {
    /// The megapixel range a text-to-image call must land in.
    pub fn mp_range(self) -> (f32, f32) {
        match self {
            // 1024² = 1.049 MP; 1536² = 2.359 MP (rounded to 16 px by
            // `image::fit_size`, so a hair inside both).
            Dialect::Seedream => (1.06, 2.34),
            // Below 2 MP asks for "1K", at or above for "2K".
            Dialect::Gemini => (0.5, 4.0),
            // 655,360 px is GPT Image's floor.
            Dialect::GptImage => (0.66, 4.0),
            _ => (0.1, 4.0),
        }
    }

    /// Does the request carry our authored seed? When not, the seed still
    /// keys the cache (a "new seed" is how you ask for a variation).
    pub fn sends_seed(self) -> bool {
        matches!(self, Dialect::Flux | Dialect::Gemini | Dialect::IcLight | Dialect::LucyEdit | Dialect::LtxInpaint)
    }

    /// Can this model take a clip of `frames` at `fps`, `w`×`h`? Said in
    /// the plan, before anything is sent: a provider that rejects the clip
    /// after upload has still cost the user the steps above it.
    pub fn check_clip(self, label: &str, frames: u32, fps: f32, w: u32, h: u32) -> Result<(), String> {
        let secs = frames as f32 / fps.max(1.0);
        match self {
            Dialect::KlingEdit => {
                if !(3.0..=15.0).contains(&secs) {
                    return Err(format!("{label} takes a 3-15 s clip; this one is {secs:.1} s"));
                }
                if w.min(h) < 720 {
                    return Err(format!(
                        "{label} needs a clip at least 720 px on its short side; the comp gives {w}x{h}"
                    ));
                }
            }
            Dialect::LtxInpaint if !(9..=481).contains(&frames) => {
                return Err(format!("{label} takes 9-481 frames; this clip is {frames}"));
            }
            _ => {}
        }
        Ok(())
    }

    /// The output field holding the picture, when it is not fal's usual
    /// `images[0]` / `image`.
    pub fn output_key(self, task: Task) -> Option<&'static str> {
        match (self, task) {
            (Dialect::Moge, _) => Some("normal_map"),
            (Dialect::Preprocessor, Task::Normals) => Some("normal_map"),
            (Dialect::Preprocessor, Task::Depth) => Some("depth_map"),
            _ => None,
        }
    }

    /// The relighting app's styles; a prompt naming one picks it.
    pub const LIGHTING_STYLES: [&'static str; 18] = [
        "natural", "studio", "golden_hour", "blue_hour", "dramatic", "soft", "hard", "backlight",
        "side_light", "front_light", "rim_light", "sunset", "sunrise", "neon", "candlelight",
        "moonlight", "spotlight", "ambient",
    ];

    /// The style a prompt names ("Golden hour, warm" -> golden_hour), else
    /// natural.
    pub fn lighting_style(prompt: &str) -> &'static str {
        let p = prompt.to_ascii_lowercase().replace(['-', '_'], " ");
        Self::LIGHTING_STYLES
            .iter()
            .copied()
            .filter(|s| p.contains(&s.replace('_', " ")))
            // "backlight" before "light": the longest name that matches.
            .max_by_key(|s| s.len())
            .unwrap_or("natural")
    }

    /// Nano Banana's `resolution` class for a megapixel budget.
    pub fn gemini_resolution(megapixels: f32) -> &'static str {
        if megapixels < 2.0 {
            "1K"
        } else {
            "2K"
        }
    }

    /// Nano Banana's `aspect_ratio`: the listed ratio nearest `w:h`.
    pub fn gemini_aspect(w: u32, h: u32) -> &'static str {
        const RATIOS: [(&str, f64); 14] = [
            ("21:9", 21.0 / 9.0),
            ("16:9", 16.0 / 9.0),
            ("3:2", 1.5),
            ("4:3", 4.0 / 3.0),
            ("5:4", 1.25),
            ("1:1", 1.0),
            ("4:5", 0.8),
            ("3:4", 0.75),
            ("2:3", 2.0 / 3.0),
            ("9:16", 9.0 / 16.0),
            ("4:1", 4.0),
            ("1:4", 0.25),
            ("8:1", 8.0),
            ("1:8", 0.125),
        ];
        let a = (w.max(1) as f64 / h.max(1) as f64).ln();
        RATIOS
            .iter()
            .min_by(|x, y| (x.1.ln() - a).abs().total_cmp(&(y.1.ln() - a).abs()))
            .unwrap()
            .0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RemoteModel {
    /// Stable id — the setup stores this.
    pub id: &'static str,
    pub label: &'static str,
    /// What the inner node says (a dozen characters).
    pub short: &'static str,
    pub service: Service,
    /// The service's own model path, e.g. `fal-ai/flux/schnell`.
    pub endpoint: &'static str,
    pub task: Task,
    pub dialect: Dialect,
    /// See the module docs. Structured, because the graph reasons about it.
    pub commercial: bool,
    /// Free text shown beside the model before anything is sent.
    pub licence: &'static str,
    pub about: &'static str,
}

/// The model a fresh GenAI node generates with.
pub const DEFAULT_MODEL: &str = "fal/flux-schnell";

pub const CATALOG: &[RemoteModel] = &[
    RemoteModel {
        id: "fal/flux-schnell",
        label: "FLUX.1 [schnell]",
        short: "flux schnell",
        service: Service::Fal,
        endpoint: "fal-ai/flux/schnell",
        dialect: Dialect::Flux,
        task: Task::TextToImage,
        commercial: true,
        licence: "Apache-2.0",
        about: "fast text-to-image, four steps; the cheap way to find a composition",
    },
    RemoteModel {
        id: "fal/flux-dev",
        label: "FLUX.1 [dev]",
        short: "flux dev",
        service: Service::Fal,
        endpoint: "fal-ai/flux/dev",
        dialect: Dialect::Flux,
        task: Task::TextToImage,
        // The open weights are under BFL's non-commercial licence. Whether
        // the provider's agreement covers your use is for you to check, so
        // the tag stays on until someone has.
        commercial: false,
        licence: "FLUX.1 [dev] Non-Commercial License (weights); check the provider's terms",
        about: "higher-quality text-to-image than schnell, slower",
    },
    RemoteModel {
        id: "fal/flux-kontext-pro",
        label: "FLUX.1 Kontext [pro]",
        short: "kontext",
        service: Service::Fal,
        endpoint: "fal-ai/flux-pro/kontext",
        dialect: Dialect::Flux,
        task: Task::Edit,
        commercial: true,
        licence: "Black Forest Labs API terms (commercial use through the API)",
        about: "edit a picture by instruction: \"make it night\", \"remove the sign\"",
    },
    RemoteModel {
        id: "fal/flux-fill-pro",
        label: "FLUX.1 Fill [pro]",
        short: "fill",
        service: Service::Fal,
        endpoint: "fal-ai/flux-pro/v1/fill",
        dialect: Dialect::Flux,
        task: Task::Inpaint,
        commercial: true,
        licence: "Black Forest Labs API terms (commercial use through the API)",
        about: "repaint the white of the matte from a prompt: clean plates, set extension",
    },
    RemoteModel {
        id: "fal/esrgan",
        label: "Real-ESRGAN",
        short: "esrgan",
        service: Service::Fal,
        endpoint: "fal-ai/esrgan",
        dialect: Dialect::Flux,
        task: Task::Upscale,
        commercial: true,
        licence: "BSD-3-Clause",
        about: "upscale by 2 or 4; invents detail, so judge it at 1:1",
    },
    RemoteModel {
        id: "fal/birefnet-v2",
        label: "BiRefNet v2",
        short: "cutout",
        service: Service::Fal,
        endpoint: "fal-ai/birefnet/v2",
        dialect: Dialect::Flux,
        task: Task::Cutout,
        commercial: true,
        licence: "MIT",
        about: "background removal: returns the picture with an alpha",
    },
    RemoteModel {
        id: "fal/seedream-5-pro",
        label: "Seedream 5 Pro",
        short: "seedream",
        service: Service::Fal,
        endpoint: "bytedance/seedream/v5/pro/text-to-image",
        dialect: Dialect::Seedream,
        task: Task::TextToImage,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "ByteDance's text-to-image: strong on text in the picture and layout; 1-2.3 MP",
    },
    RemoteModel {
        id: "fal/seedream-5-pro-edit",
        label: "Seedream 5 Pro edit",
        short: "seedream edit",
        service: Service::Fal,
        endpoint: "bytedance/seedream/v5/pro/edit",
        dialect: Dialect::Seedream,
        task: Task::Edit,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "edit a picture by instruction, ByteDance's take on Kontext",
    },
    RemoteModel {
        id: "fal/seedance-2-fast-i2v",
        label: "Seedance 2.0 fast",
        short: "seedance",
        service: Service::Fal,
        endpoint: "bytedance/seedance-2.0/fast/image-to-video",
        dialect: Dialect::Seedance { max_seconds: 15, top_res: 720 },
        task: Task::ImageToVideo,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "animate the input from a prompt: 4-15 s at 480p/720p, billed per second",
    },
    RemoteModel {
        id: "fal/seedance-2.5-i2v",
        label: "Seedance 2.5",
        short: "seedance 2.5",
        service: Service::Fal,
        endpoint: "bytedance/seedance-2.5/image-to-video",
        dialect: Dialect::Seedance { max_seconds: 30, top_res: 1080 },
        task: Task::ImageToVideo,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "the bigger Seedance: up to 30 s and 1080p; about twice 2.0 fast's price per second",
    },
    RemoteModel {
        id: "fal/kling-o3-pro-video-edit",
        label: "Kling O3 video edit",
        short: "kling edit",
        service: Service::Fal,
        endpoint: "fal-ai/kling-video/o3/pro/video-to-video/edit",
        dialect: Dialect::KlingEdit,
        task: Task::VideoEdit,
        commercial: true,
        licence: "Kling via fal API terms (commercial use through the API)",
        about: "edit the input clip by instruction: 3-15 s, 720 px or more, billed per second",
    },
    RemoteModel {
        id: "fal/lucy-edit-dev",
        label: "Lucy Edit",
        short: "lucy edit",
        service: Service::Fal,
        endpoint: "decart/lucy-edit/dev",
        dialect: Dialect::LucyEdit,
        task: Task::VideoEdit,
        // Decart's Dev weights are not released for commercial use and fal
        // states no terms: tagged until someone shows otherwise.
        commercial: false,
        licence: "Decart Lucy Edit Dev: no commercial terms stated (treat as research)",
        about: "cheap prompt edits of the input clip (clothes, colours, props), billed per second",
    },
    RemoteModel {
        id: "fal/ltx-2.3-inpaint",
        label: "LTX 2.3 video inpaint",
        short: "ltx inpaint",
        service: Service::Fal,
        endpoint: "fal-ai/ltx-2.3-quality/inpaint",
        dialect: Dialect::LtxInpaint,
        task: Task::VideoInpaint,
        // The LTX-2 community licence limits commercial use by revenue and
        // fal states no terms of its own.
        commercial: false,
        licence: "Lightricks LTX-2 community licence (commercial use limited by revenue)",
        about: "fill the matte's white over the input clip: object removal on video, 9-481 frames",
    },
    RemoteModel {
        id: "fal/nano-banana-pro",
        label: "Nano Banana Pro",
        short: "nano banana",
        service: Service::Fal,
        endpoint: "fal-ai/nano-banana-pro",
        dialect: Dialect::Gemini,
        task: Task::TextToImage,
        commercial: true,
        licence: "Google via fal API terms (commercial use through the API)",
        about: "Google's Gemini Pro image model: strong prompt following and text in the picture; 1K, or 2K from 2 MP",
    },
    RemoteModel {
        id: "fal/nano-banana-pro-edit",
        label: "Nano Banana Pro edit",
        short: "banana edit",
        service: Service::Fal,
        endpoint: "fal-ai/nano-banana-pro/edit",
        dialect: Dialect::Gemini,
        task: Task::Edit,
        commercial: true,
        licence: "Google via fal API terms (commercial use through the API)",
        about: "edit a picture by instruction with Nano Banana Pro; keeps the input's aspect",
    },
    RemoteModel {
        id: "fal/nano-banana-2",
        label: "Nano Banana 2",
        short: "nano banana 2",
        service: Service::Fal,
        endpoint: "fal-ai/nano-banana-2",
        dialect: Dialect::Gemini,
        task: Task::TextToImage,
        commercial: true,
        licence: "Google via fal API terms (commercial use through the API)",
        about: "the faster, cheaper Nano Banana; 1K, or 2K from 2 MP",
    },
    RemoteModel {
        id: "fal/nano-banana-2-edit",
        label: "Nano Banana 2 edit",
        short: "banana 2 edit",
        service: Service::Fal,
        endpoint: "fal-ai/nano-banana-2/edit",
        dialect: Dialect::Gemini,
        task: Task::Edit,
        commercial: true,
        licence: "Google via fal API terms (commercial use through the API)",
        about: "edit a picture by instruction, the faster, cheaper Nano Banana",
    },
    RemoteModel {
        id: "fal/gpt-image-2",
        label: "GPT Image 2",
        short: "gpt image",
        service: Service::Fal,
        endpoint: "openai/gpt-image-2",
        dialect: Dialect::GptImage,
        task: Task::TextToImage,
        commercial: true,
        licence: "OpenAI via fal API terms (commercial use through the API)",
        about: "OpenAI's image model. Billed by tokens, so no price up front; quality defaults to high, and extra {\"quality\": \"medium\"} costs far less",
    },
    RemoteModel {
        id: "fal/gpt-image-2-edit",
        label: "GPT Image 2 edit",
        short: "gpt edit",
        service: Service::Fal,
        endpoint: "openai/gpt-image-2/edit",
        dialect: Dialect::GptImage,
        task: Task::Edit,
        commercial: true,
        licence: "OpenAI via fal API terms (commercial use through the API)",
        about: "edit a picture by instruction with GPT Image; billed by tokens, so no price up front",
    },
    RemoteModel {
        id: "fal/gpt-image-2-inpaint",
        label: "GPT Image 2 inpaint",
        short: "gpt inpaint",
        service: Service::Fal,
        endpoint: "openai/gpt-image-2/edit",
        dialect: Dialect::GptImage,
        task: Task::Inpaint,
        commercial: true,
        licence: "OpenAI via fal API terms (commercial use through the API)",
        about: "repaint the white of the matte from a prompt with GPT Image; billed by tokens, so no price up front",
    },
    RemoteModel {
        id: "fal/moge-3-normals",
        label: "MoGe-3 Normals",
        short: "moge-3",
        service: Service::Fal,
        endpoint: "microsoft/moge-3",
        dialect: Dialect::Moge,
        task: Task::Normals,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "surface normals from one picture, Microsoft's MoGe-3 (refined); raw n*0.5+0.5, no sRGB",
    },
    RemoteModel {
        id: "fal/moge-2-normals",
        label: "MoGe-2 Normals",
        short: "moge-2",
        service: Service::Fal,
        endpoint: "microsoft/moge-2",
        dialect: Dialect::Moge,
        task: Task::Normals,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "surface normals from one picture, MoGe-2: cheaper than MoGe-3; raw n*0.5+0.5",
    },
    RemoteModel {
        id: "fal/marigold-depth",
        label: "Marigold Depth",
        short: "marigold depth",
        service: Service::Fal,
        endpoint: "fal-ai/marigold-v2",
        dialect: Dialect::Marigold { prediction: "depth" },
        task: Task::Depth,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "relative depth from one picture (Marigold v2), as raw grey; a viewable PNG, 8-bit",
    },
    RemoteModel {
        id: "fal/marigold-normals",
        label: "Marigold Normals",
        short: "marigold normals",
        service: Service::Fal,
        endpoint: "fal-ai/marigold-v2",
        dialect: Dialect::Marigold { prediction: "normals" },
        task: Task::Normals,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "surface normals from one picture (Marigold v2); raw, 8-bit",
    },
    RemoteModel {
        id: "fal/marigold-albedo",
        label: "Marigold Albedo",
        short: "marigold albedo",
        service: Service::Fal,
        endpoint: "fal-ai/marigold-v2",
        dialect: Dialect::Marigold { prediction: "albedo" },
        task: Task::Albedo,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "the picture with its lighting taken out (Marigold v2): a base for relighting",
    },
    RemoteModel {
        id: "fal/depth-anything-v2",
        label: "Depth Anything v2",
        short: "depth anything",
        service: Service::Fal,
        endpoint: "fal-ai/image-preprocessors/depth-anything/v2",
        dialect: Dialect::Preprocessor,
        task: Task::Depth,
        commercial: false,
        licence: "Depth Anything V2: Small is Apache-2.0, Base/Large CC-BY-NC-4.0; which fal runs is not stated",
        about: "relative depth (near is bright) from one picture, raw grey; fal gives no price up front",
    },
    RemoteModel {
        id: "fal/midas-depth",
        label: "MiDaS Depth",
        short: "midas depth",
        service: Service::Fal,
        endpoint: "fal-ai/image-preprocessors/midas",
        dialect: Dialect::Preprocessor,
        task: Task::Depth,
        commercial: true,
        licence: "MiDaS: MIT",
        about: "relative depth (near is bright) from one picture, raw grey; fal gives no price up front",
    },
    RemoteModel {
        id: "fal/midas-normals",
        label: "MiDaS Normals",
        short: "midas normals",
        service: Service::Fal,
        endpoint: "fal-ai/image-preprocessors/midas",
        dialect: Dialect::Preprocessor,
        task: Task::Normals,
        commercial: true,
        licence: "MiDaS: MIT",
        about: "normals derived from MiDaS depth, raw; rougher than MoGe or Marigold; no price up front",
    },
    RemoteModel {
        id: "fal/iclight-v2",
        label: "IC-Light v2",
        short: "ic-light",
        service: Service::Fal,
        endpoint: "fal-ai/iclight-v2",
        dialect: Dialect::IcLight,
        task: Task::Relight,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "relight a picture from a prompt (\"warm sunset light from the left\"); extra {\"initial_latent\": \"Left\"} sets the light's side",
    },
    RemoteModel {
        id: "fal/relighting",
        label: "Relighting",
        short: "relighting",
        service: Service::Fal,
        endpoint: "fal-ai/image-apps-v2/relighting",
        dialect: Dialect::LightingStyle,
        task: Task::Relight,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "relight a picture to a named style: natural, studio, golden hour, blue hour, dramatic, rim light, neon, moonlight…; the prompt picks it",
    },
];

pub fn get(id: &str) -> Option<&'static RemoteModel> {
    CATALOG.iter().find(|m| m.id == id)
}

/// The catalog models for a task, for the panel's model menu.
pub fn for_task(task: Task) -> impl Iterator<Item = &'static RemoteModel> {
    CATALOG.iter().filter(move |m| m.task == task)
}

impl Task {
    pub fn label(self) -> &'static str {
        match self {
            Task::TextToImage => "text to image",
            Task::Edit => "edit",
            Task::Inpaint => "inpaint",
            Task::Upscale => "upscale",
            Task::Cutout => "cutout",
            Task::ImageToVideo => "image to video",
            Task::Depth => "depth",
            Task::Normals => "normals",
            Task::Albedo => "albedo",
            Task::Relight => "relight",
            Task::VideoEdit => "video edit",
            Task::VideoInpaint => "video inpaint",
        }
    }

    /// Does this task return a video rather than a picture?
    pub fn is_video(self) -> bool {
        matches!(self, Task::ImageToVideo | Task::VideoEdit | Task::VideoInpaint)
    }

    /// Does this task READ a clip (the node's input over a range)?
    pub fn takes_clip(self) -> bool {
        matches!(self, Task::VideoEdit | Task::VideoInpaint)
    }

    /// Does this task hand back MEASUREMENTS rather than a picture? A data
    /// take plays back as stored values, with no sRGB transform.
    pub fn is_data(self) -> bool {
        matches!(self, Task::Depth | Task::Normals)
    }

    pub const ALL: [Task; 12] = [
        Task::TextToImage,
        Task::Edit,
        Task::Inpaint,
        Task::Upscale,
        Task::Cutout,
        Task::ImageToVideo,
        Task::Depth,
        Task::Normals,
        Task::Albedo,
        Task::Relight,
        Task::VideoEdit,
        Task::VideoInpaint,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_the_default_exists() {
        let mut ids: Vec<_> = CATALOG.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), CATALOG.len());
        assert!(get(DEFAULT_MODEL).is_some());
    }

    #[test]
    fn every_task_has_a_model() {
        for t in Task::ALL {
            assert!(for_task(t).next().is_some(), "{t:?}");
        }
    }

    #[test]
    fn no_endpoint_carries_a_scheme_or_host() {
        // An endpoint is a path on the service; the base URL is the app's.
        for m in CATALOG {
            assert!(!m.endpoint.contains("://") && !m.endpoint.starts_with('/'), "{}", m.id);
        }
    }

    #[test]
    fn dialects_match_their_tasks() {
        for m in CATALOG {
            let video = matches!(
                m.dialect,
                Dialect::Seedance { .. } | Dialect::KlingEdit | Dialect::LucyEdit | Dialect::LtxInpaint
            );
            assert_eq!(video, m.task.is_video(), "{}", m.id);
        }
        let (lo, hi) = Dialect::Seedream.mp_range();
        assert!(lo * 1e6 > 1024.0 * 1024.0 && hi * 1e6 < 1536.0 * 1536.0);
    }
}
