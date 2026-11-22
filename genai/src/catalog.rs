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
    /// a clip (the node's input over a range, or a video step's result)
    /// -> the same clip, bigger. A video take.
    VideoUpscale,
    /// prompt + image -> the image on a BIGGER canvas, the new border
    /// painted in. The picture stays in the middle.
    Outpaint,
    /// prompt + image + a SECOND picture (the node's fourth input) -> the
    /// image edited with the second to draw on: its colour, style, content.
    EditRef,
    /// prompt (which object) + image -> that object in 3D, as a Gaussian
    /// splat (`.ply`). Not a picture: the take becomes a Splat node.
    ImageTo3D,
    /// prompt -> a better prompt, by a language model. TEXT out: a step
    /// inside a pipeline, never a node of its own.
    Enhance,
    /// prompt (what to dwell on) + image -> the picture described, as a
    /// prompt. TEXT out, like Enhance.
    Describe,
}

/// One setting of a model the panel offers (`Dialect::params`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Param {
    /// The request field.
    pub key: &'static str,
    pub label: &'static str,
    /// (what the menu says, the JSON it sends). The first is the default.
    pub choices: &'static [(&'static str, &'static str)],
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
    /// `reference`: the reference-to-video endpoint, which takes the
    /// picture in `image_urls` (the prompt names it `@Image1`) rather than
    /// as the first frame.
    Seedance { max_seconds: u32, top_res: u32, reference: bool },
    /// Seedance 2.5's reference endpoint on its EDITING task: `prompt`,
    /// `video_urls` (one clip, 1.8-30.2 s, 24 fps or more), 720p, no sound;
    /// length and shape follow the clip.
    SeedanceEdit,
    /// Google's Gemini image models (Nano Banana): no width or height. The
    /// size is an `aspect_ratio` from a fixed list plus a `resolution`
    /// class ("1K", "2K"), and an edit takes `image_urls`. Takes a seed.
    Gemini,
    /// OpenAI's GPT Image: no `seed`, an edit or inpaint takes
    /// `image_urls` (+ `mask_url`, white = replace), and a text-to-image
    /// size must hold at least 655,360 px. Billed by tokens, which fal's
    /// pricing API does not turn into a price, so it quotes as unknown.
    GptImage,
    /// FLUX.1 Kontext [pro] multi: Kontext's fields, but the pictures go
    /// in `image_urls` (the first is the one edited, the rest it draws
    /// on). No size: the model follows its inputs.
    KontextMulti,
    /// Qwen Image Edit Plus (2509): `image_urls`, `seed`, and an
    /// `image_size {width, height}`, which we take from the comp's aspect
    /// (fal's default is a square).
    QwenEditPlus,
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
    /// Wan 2.7 video edit: `prompt`, `video_url` (2-10 s), `seed`;
    /// 720p (what our clips are; 1080p bills half as much again), the
    /// input's own length (`duration` 0), its own (silent) audio.
    Wan27Edit,
    /// Wan VACE 14B inpainting: `prompt`, `video_url`, `mask_video_url`
    /// (white = regenerate), `seed`; length and rate follow the clip
    /// (17-241 frames), prompt expansion off.
    WanVace,
    /// Bria Video Eraser (mask): `video_url`, `mask_video_url` (white =
    /// erase), both under 5 s; no prompt (the port is fed and ignored),
    /// no seed, sound off, no auto-trim (the plan refuses a long clip
    /// instead of fal cutting it behind the key's back).
    BriaVideoEraser,
    /// Topaz image upscale: `image_url`, `model` (which of Topaz's
    /// networks), `upscale_factor`. No seed. Face enhancement is OFF: on
    /// by default at Topaz, and a face quietly redrawn is not an upscale.
    Topaz { model: &'static str },
    /// Clarity upscaler: `image_url`, `upscale_factor`, `seed`; its
    /// creativity and resemblance ride in extra.
    Clarity,
    /// SeedVR2 on an image: `image_url`, `upscale_factor`, `seed`.
    SeedVr,
    /// Recraft's upscalers: `image_url` (a PNG, which is what we send) and
    /// nothing else. The factor is the model's own.
    Recraft,
    /// Topaz video upscale: `video_url`, `model`, `upscale_factor`; H.264
    /// out, because that is what every movie build decodes.
    TopazVideo { model: &'static str },
    /// SeedVR2 on a clip: `video_url`, `upscale_factor`, `seed`.
    SeedVrVideo,
    /// Bria Expand: `image_url`, `canvas_size`, `original_image_size`,
    /// `original_image_location` (the picture centred on the canvas),
    /// `prompt`, `seed`.
    BriaExpand,
    /// Bria background replace: `image_url`, `prompt`, `seed`; prompt
    /// refinement off (a rewrite would change the request behind the key).
    BriaBackground,
    /// A Gemini edit over TWO pictures (`image_urls: [image, second]`),
    /// with a fixed instruction ahead of the typed prompt: what makes a
    /// general edit model a ready-made node.
    GeminiRef { instruction: &'static str },
    /// SAM 3D Objects: `image_url`, `prompt` (the object to lift, found by
    /// SAM 3), `seed`; the splat is `gaussian_splat`.
    Sam3d,
    /// fal's any-llm (and its vision twin): `prompt`, `system_prompt`,
    /// `image_urls` for the vision one; the answer is `output`, a string.
    /// No seed: the authored seed keys the cache, so the same step gives
    /// the same words until you ask for another.
    Llm { system: &'static str },
}

/// What the Prompt Enhancer is told to be.
pub const ENHANCE: &str = "You rewrite prompts for image and video generation models. Expand the \
user's prompt into one vivid paragraph: subject, setting, composition, lighting, lens and mood. \
Keep every element the user named and invent nothing that contradicts it. Answer with the prompt \
only: no preamble, no quotation marks, no lists.";

/// What Describe is told to be.
pub const DESCRIBE: &str = "You describe images as prompts for image generation models. Write one \
paragraph that would recreate the image: subject, setting, composition, lighting, colour and \
style. Follow any instruction the user gives about what to focus on. Answer with the description \
only: no preamble, no quotation marks, no lists.";

/// What the Colour Transfer node asks for.
pub const COLOUR_TRANSFER: &str = "Recolour the first image so its colour palette, contrast and \
grade match the second image. Keep the first image's content, composition, framing and every \
detail exactly as they are; change only colour and tone. Take nothing else from the second image.";

impl Dialect {
    /// The prompt a request carries: the dialect's fixed instruction, then
    /// what was typed.
    pub fn instruct(self, typed: &str) -> String {
        match self {
            Dialect::GeminiRef { instruction } if !instruction.is_empty() => {
                if typed.trim().is_empty() {
                    instruction.to_string()
                } else {
                    format!("{instruction} {}", typed.trim())
                }
            }
            _ => typed.to_string(),
        }
    }
}

impl Dialect {
    /// The range an upscaler's factor may take, or `None` when the model
    /// picks its own (the panel then shows no slider).
    pub fn scale_range(self) -> Option<(f32, f32)> {
        match self {
            Dialect::Recraft => None,
            // How many times the canvas grows; 5000 px a side is Bria's top.
            Dialect::BriaExpand => Some((1.1, 3.0)),
            _ => Some((1.0, 4.0)),
        }
    }

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
            // Qwen-Image works around a megapixel.
            Dialect::QwenEditPlus => (0.5, 2.0),
            _ => (0.1, 4.0),
        }
    }

    /// The model's own settings the panel offers beyond seed and size: a
    /// request field, and the values it may take as JSON literals. The
    /// FIRST choice is what the request already says (nothing is stored
    /// for it); any other is written into the step's `extra`, which is
    /// merged over the request and is already part of the cache key.
    pub fn params(self) -> &'static [Param] {
        const ONOFF: &[(&str, &str)] = &[("off", "false"), ("on", "true")];
        const CODEC: Param =
            Param { key: "codec", label: "codec", choices: &[("auto", "\"auto\""), ("H264", "\"H264\""), ("H265", "\"H265\"")] };
        const BITRATE: Param =
            Param { key: "bitrate_mode", label: "bitrate", choices: &[("standard", "\"standard\""), ("high", "\"high\"")] };
        const AUDIO: Param = Param { key: "generate_audio", label: "generate audio", choices: ONOFF };
        const DRAFT: Param = Param { key: "draft", label: "draft (480p preview)", choices: ONOFF };
        match self {
            Dialect::Wan27Edit => &[
                Param { key: "resolution", label: "resolution", choices: &[("720p", "\"720p\""), ("1080p", "\"1080p\"")] },
                Param {
                    key: "duration",
                    label: "duration",
                    choices: &[
                        ("input's", "0"), ("2 s", "2"), ("3 s", "3"), ("4 s", "4"), ("5 s", "5"), ("6 s", "6"),
                        ("7 s", "7"), ("8 s", "8"), ("9 s", "9"), ("10 s", "10"),
                    ],
                },
                Param {
                    key: "aspect_ratio",
                    label: "aspect",
                    choices: &[
                        ("input's", "null"), ("16:9", "\"16:9\""), ("9:16", "\"9:16\""), ("1:1", "\"1:1\""),
                        ("4:3", "\"4:3\""), ("3:4", "\"3:4\""),
                    ],
                },
                Param { key: "audio_setting", label: "audio", choices: &[("original", "\"origin\""), ("auto", "\"auto\"")] },
            ],
            Dialect::Seedance { reference: true, .. } => &[
                DRAFT,
                Param { key: "resolution", label: "resolution", choices: &[("size above", "null"), ("4k", "\"4k\"")] },
                Param { key: "duration", label: "duration", choices: &[("seconds above", "null"), ("auto", "\"auto\"")] },
                Param {
                    key: "aspect_ratio",
                    label: "aspect",
                    choices: &[
                        ("auto", "\"auto\""), ("21:9", "\"21:9\""), ("16:9", "\"16:9\""), ("4:3", "\"4:3\""),
                        ("1:1", "\"1:1\""), ("3:4", "\"3:4\""), ("9:16", "\"9:16\""),
                    ],
                },
                AUDIO,
                BITRATE,
                CODEC,
            ],
            Dialect::SeedanceEdit => &[
                DRAFT,
                Param {
                    key: "resolution",
                    label: "resolution",
                    choices: &[("720p", "\"720p\""), ("480p", "\"480p\""), ("1080p", "\"1080p\""), ("4k", "\"4k\"")],
                },
                AUDIO,
                BITRATE,
                CODEC,
            ],
            _ => &[],
        }
    }

    /// Does the request carry our authored seed? When not, the seed still
    /// keys the cache (a "new seed" is how you ask for a variation).
    pub fn sends_seed(self) -> bool {
        matches!(
            self,
            Dialect::Flux
                | Dialect::Gemini
                | Dialect::IcLight
                | Dialect::LucyEdit
                | Dialect::LtxInpaint
                | Dialect::Seedance { reference: true, .. }
                | Dialect::WanVace
                | Dialect::Wan27Edit
                | Dialect::SeedanceEdit
                | Dialect::Clarity
                | Dialect::SeedVr
                | Dialect::SeedVrVideo
                | Dialect::BriaExpand
                | Dialect::BriaBackground
                | Dialect::GeminiRef { .. }
                | Dialect::Sam3d
                | Dialect::KontextMulti
                | Dialect::QwenEditPlus
        )
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
            Dialect::SeedanceEdit => {
                if !(1.8..=30.2).contains(&secs) {
                    return Err(format!("{label} takes a 1.8-30 s clip; this one is {secs:.1} s"));
                }
                if fps < 24.0 {
                    return Err(format!("{label} needs a clip at 24 fps or more; the comp is at {fps}"));
                }
            }
            Dialect::Wan27Edit if !(2.0..=10.0).contains(&secs) => {
                return Err(format!("{label} takes a 2-10 s clip; this one is {secs:.1} s"));
            }
            Dialect::WanVace if !(17..=241).contains(&frames) => {
                return Err(format!("{label} takes 17-241 frames; this clip is {frames}"));
            }
            Dialect::BriaVideoEraser if secs >= 5.0 => {
                return Err(format!("{label} takes a clip under 5 s; this one is {secs:.1} s"));
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
            (Dialect::Sam3d, _) => Some("gaussian_splat"),
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

impl RemoteModel {
    /// How many reference pictures this model takes AFTER its task's own
    /// image: the most `extra_ports` will ever grow to. The caps are the
    /// provider's documented totals less the main picture:
    /// - Seedream 5 edit: 10 (fal's schema: "up to 10 images").
    /// - Nano Banana Pro / 2 edit: 14 (Google's Gemini image docs; fal's
    ///   Nano Banana 2 page says the same).
    /// - GPT Image 2 edit: 16 (fal's schema, `maxItems: 16`).
    /// - Kontext multi, Qwen Edit Plus: no documented maximum, so 9, the
    ///   same as Seedance's reference (Qwen works best with 1-3).
    /// - Seedance reference: 9.
    pub fn max_refs(&self) -> usize {
        match (self.task, self.dialect) {
            (_, Dialect::Seedance { reference: true, .. }) => 8,
            (_, Dialect::Wan27Edit) => 1,
            (Task::Edit, Dialect::Seedream) => 9,
            (Task::Edit, Dialect::Gemini) => 13,
            (Task::Edit, Dialect::GptImage) => 15,
            (Task::Edit, Dialect::KontextMulti | Dialect::QwenEditPlus) => 8,
            _ => 0,
        }
    }

    /// Does this model take a GROWING list of reference ports (one free
    /// past the last wired), rather than a fixed set?
    pub fn grows_refs(&self) -> bool {
        self.max_refs() > 1
    }

    /// OPTIONAL ports this model has after its task's own: Wan 2.7's edit
    /// has one reference picture; a model that takes a list of them has
    /// `refs` already wired plus one free, so there is always somewhere to
    /// land the next (`Opts::refs`), up to `max_refs`.
    pub fn extra_ports(&self, refs: usize) -> &'static [&'static str] {
        const MORE: [&str; 15] = [
            "reference 2", "reference 3", "reference 4", "reference 5", "reference 6",
            "reference 7", "reference 8", "reference 9", "reference 10", "reference 11",
            "reference 12", "reference 13", "reference 14", "reference 15", "reference 16",
        ];
        match self.dialect {
            Dialect::Wan27Edit => &["reference"],
            _ => &MORE[..(refs + 1).min(self.max_refs()).min(MORE.len())],
        }
    }
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
        about: "fast text-to-image, four steps; the quick way to find a composition",
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
        id: "fal/flux-kontext-pro-multi",
        label: "FLUX.1 Kontext [pro] multi",
        short: "kontext multi",
        service: Service::Fal,
        endpoint: "fal-ai/flux-pro/kontext/multi",
        dialect: Dialect::KontextMulti,
        task: Task::Edit,
        commercial: true,
        licence: "Black Forest Labs API terms (commercial use through the API)",
        about: "Kontext over several pictures: edit the first, drawing on the references wired after it (a person, a product, a style)",
    },
    RemoteModel {
        id: "fal/qwen-image-edit-plus",
        label: "Qwen Image Edit Plus",
        short: "qwen edit+",
        service: Service::Fal,
        endpoint: "fal-ai/qwen-image-edit-plus",
        dialect: Dialect::QwenEditPlus,
        task: Task::Edit,
        commercial: true,
        licence: "Qwen-Image weights are Apache-2.0; fal API terms",
        about: "Alibaba's open edit model over several pictures (person + product, person + scene); best with 1-3",
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
        id: "fal/topaz-upscale",
        label: "Topaz Upscale",
        short: "topaz",
        service: Service::Fal,
        endpoint: "fal-ai/topaz/upscale/image",
        dialect: Dialect::Topaz { model: "Standard V2" },
        task: Task::Upscale,
        commercial: true,
        licence: "Topaz Labs via fal API terms (commercial use through the API)",
        about: "Topaz's faithful upscaler: sharper, not repainted; the one to try first on a plate",
    },
    RemoteModel {
        id: "fal/topaz-upscale-hifi",
        label: "Topaz Upscale High Fidelity",
        short: "topaz hifi",
        service: Service::Fal,
        endpoint: "fal-ai/topaz/upscale/image",
        dialect: Dialect::Topaz { model: "High Fidelity V2" },
        task: Task::Upscale,
        commercial: true,
        licence: "Topaz Labs via fal API terms (commercial use through the API)",
        about: "Topaz High Fidelity: keeps fine texture and grain structure, for an already clean source",
    },
    RemoteModel {
        id: "fal/topaz-recovery",
        label: "Topaz Recovery",
        short: "topaz recovery",
        service: Service::Fal,
        endpoint: "fal-ai/topaz/upscale/image",
        dialect: Dialect::Topaz { model: "Recovery V2" },
        task: Task::Upscale,
        commercial: true,
        licence: "Topaz Labs via fal API terms (commercial use through the API)",
        about: "Topaz Recovery: rebuilds a small or badly compressed picture; invents more, so check it",
    },
    RemoteModel {
        id: "fal/topaz-redefine",
        label: "Topaz Redefine",
        short: "topaz redefine",
        service: Service::Fal,
        endpoint: "fal-ai/topaz/upscale/image",
        dialect: Dialect::Topaz { model: "Redefine" },
        task: Task::Upscale,
        commercial: true,
        licence: "Topaz Labs via fal API terms (commercial use through the API)",
        about: "Topaz's creative upscaler: repaints detail as it enlarges; extra {\"creativity\": 3, \"prompt\": \"...\"} steers it",
    },
    RemoteModel {
        id: "fal/clarity-upscaler",
        label: "Clarity Upscaler",
        short: "clarity",
        service: Service::Fal,
        endpoint: "fal-ai/clarity-upscaler",
        dialect: Dialect::Clarity,
        task: Task::Upscale,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "creative upscale in the Magnific manner: adds detail that was never there; extra {\"creativity\": 0.35, \"resemblance\": 0.6} trades invention against likeness",
    },
    RemoteModel {
        id: "fal/seedvr2-upscale",
        label: "SeedVR2 Upscale",
        short: "seedvr2",
        service: Service::Fal,
        endpoint: "fal-ai/seedvr/upscale/image",
        dialect: Dialect::SeedVr,
        task: Task::Upscale,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "restoring upscale by a one-step diffusion model: strong on soft or noisy sources",
    },
    RemoteModel {
        id: "fal/recraft-crisp-upscale",
        label: "Recraft Crisp Upscale",
        short: "recraft crisp",
        service: Service::Fal,
        endpoint: "fal-ai/recraft/upscale/crisp",
        dialect: Dialect::Recraft,
        task: Task::Upscale,
        commercial: true,
        licence: "Recraft via fal API terms (commercial use through the API)",
        about: "a clean enlargement with sharper edges and no repainting; the size is the model's choice",
    },
    RemoteModel {
        id: "fal/bria-expand",
        label: "Outpaint",
        short: "outpaint",
        service: Service::Fal,
        endpoint: "fal-ai/bria/expand",
        dialect: Dialect::BriaExpand,
        task: Task::Outpaint,
        commercial: true,
        licence: "Bria via fal API terms (trained on licensed data; commercial use through the API)",
        about: "grow the canvas around the picture and paint what was outside the frame; the prompt may say what",
    },
    RemoteModel {
        id: "fal/bria-background-replace",
        label: "Background Replace",
        short: "new background",
        service: Service::Fal,
        endpoint: "fal-ai/bria/background/replace",
        dialect: Dialect::BriaBackground,
        task: Task::Edit,
        commercial: true,
        licence: "Bria via fal API terms (trained on licensed data; commercial use through the API)",
        about: "keep the subject, replace everything behind it with what the prompt describes",
    },
    RemoteModel {
        id: "fal/colour-transfer",
        label: "Colour Transfer",
        short: "colour transfer",
        service: Service::Fal,
        endpoint: "fal-ai/nano-banana-2/edit",
        dialect: Dialect::GeminiRef { instruction: COLOUR_TRANSFER },
        task: Task::EditRef,
        commercial: true,
        licence: "Google via fal API terms (commercial use through the API)",
        about: "give the picture the palette and grade of a second picture; a repaint at up to 2 MP, so for look development, not a final grade",
    },
    RemoteModel {
        id: "fal/nano-banana-pro-edit-ref",
        label: "Nano Banana Pro edit, two pictures",
        short: "banana edit 2",
        service: Service::Fal,
        endpoint: "fal-ai/nano-banana-pro/edit",
        dialect: Dialect::GeminiRef { instruction: "" },
        task: Task::EditRef,
        commercial: true,
        licence: "Google via fal API terms (commercial use through the API)",
        about: "edit a picture with a second one to draw on: its style, an object in it, its light; the prompt says how",
    },
    RemoteModel {
        id: "fal/sam-3d-objects",
        label: "SAM 3D Objects",
        short: "sam 3d",
        service: Service::Fal,
        endpoint: "fal-ai/sam-3/3d-objects",
        dialect: Dialect::Sam3d,
        task: Task::ImageTo3D,
        commercial: true,
        licence: "Meta SAM License via fal (commercial use allowed, with use restrictions)",
        about: "an object in the picture, named by the prompt (\"chair\"), as a 3D Gaussian splat: lands as a Splat node you can put a Camera on",
    },
    RemoteModel {
        id: "fal/prompt-enhance",
        label: "Prompt Enhancer",
        short: "enhance",
        service: Service::Fal,
        endpoint: "fal-ai/any-llm",
        dialect: Dialect::Llm { system: ENHANCE },
        task: Task::Enhance,
        commercial: true,
        licence: "the language model's own terms, through fal (commercial use through the API)",
        about: "rewrites a short prompt into a full one (subject, setting, light, lens) with a small language model; its text feeds the next step's prompt",
    },
    RemoteModel {
        id: "fal/describe",
        label: "Describe",
        short: "describe",
        service: Service::Fal,
        endpoint: "fal-ai/any-llm/vision",
        dialect: Dialect::Llm { system: DESCRIBE },
        task: Task::Describe,
        commercial: true,
        licence: "the language model's own terms, through fal (commercial use through the API)",
        about: "looks at the picture and writes it up as a prompt; its own prompt says what to dwell on (may be empty)",
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
        dialect: Dialect::Seedance { max_seconds: 15, top_res: 720, reference: false },
        task: Task::ImageToVideo,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "animate the input from a prompt: 4-15 s at 480p/720p",
    },
    RemoteModel {
        id: "fal/seedance-2.5-i2v",
        label: "Seedance 2.5",
        short: "seedance 2.5",
        service: Service::Fal,
        endpoint: "bytedance/seedance-2.5/image-to-video",
        dialect: Dialect::Seedance { max_seconds: 30, top_res: 1080, reference: false },
        task: Task::ImageToVideo,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "the bigger Seedance: up to 30 s and 1080p",
    },
    RemoteModel {
        id: "fal/seedance-2.5-reference",
        label: "Seedance 2.5 reference",
        short: "seedance ref",
        service: Service::Fal,
        endpoint: "bytedance/seedance-2.5/reference-to-video",
        dialect: Dialect::Seedance { max_seconds: 30, top_res: 1080, reference: true },
        task: Task::ImageToVideo,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "a clip made FROM the picture as a reference (a character, a product, a look), not starting on it: call it @Image1 in the prompt",
    },
    RemoteModel {
        id: "fal/seedance-2.5-video-edit",
        label: "Seedance 2.5 video edit",
        short: "seedance edit",
        service: Service::Fal,
        endpoint: "bytedance/seedance-2.5/reference-to-video",
        dialect: Dialect::SeedanceEdit,
        task: Task::VideoEdit,
        commercial: true,
        licence: "ByteDance via fal API terms (commercial use through the API)",
        about: "edit the input clip by instruction (change, add, remove): 2-30 s, 720p",
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
        about: "edit the input clip by instruction: 3-15 s, 720 px or more",
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
        about: "quick prompt edits of the input clip (clothes, colours, props)",
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
        id: "fal/wan-2.7-video-edit",
        label: "Wan 2.7 video edit",
        short: "wan 2.7 edit",
        service: Service::Fal,
        endpoint: "fal-ai/wan/v2.7/edit-video",
        dialect: Dialect::Wan27Edit,
        task: Task::VideoEdit,
        commercial: true,
        licence: "Wan 2.7 via fal API terms (commercial use through the API)",
        about: "edit the input clip by instruction (restyle, change, remove): 2-10 s, 720p",
    },
    RemoteModel {
        id: "fal/wan-vace-inpaint",
        label: "Wan VACE video inpaint",
        short: "wan vace",
        service: Service::Fal,
        endpoint: "fal-ai/wan-vace-14b/inpainting",
        dialect: Dialect::WanVace,
        task: Task::VideoInpaint,
        // Wan 2.1 (and VACE) are released under Apache 2.0.
        commercial: true,
        licence: "Wan 2.1 VACE (Apache 2.0)",
        about: "repaint the matte's white over the input clip from the prompt: edit or remove on video, 17-241 frames",
    },
    RemoteModel {
        id: "fal/bria-video-eraser",
        label: "Bria Video Eraser",
        short: "bria eraser",
        service: Service::Fal,
        endpoint: "bria/bria_video_eraser/erase/mask",
        dialect: Dialect::BriaVideoEraser,
        task: Task::VideoInpaint,
        // fal tags it "Commercial use"; Bria trains on licensed data.
        commercial: true,
        licence: "Bria via fal (tagged commercial use)",
        about: "erase what the matte's white covers from the input clip, no prompt: under 5 s",
    },
    RemoteModel {
        id: "fal/topaz-video-upscale",
        label: "Topaz Video Upscale",
        short: "topaz video",
        service: Service::Fal,
        endpoint: "fal-ai/topaz/upscale/video",
        dialect: Dialect::TopazVideo { model: "Proteus" },
        task: Task::VideoUpscale,
        commercial: true,
        licence: "Topaz Labs via fal API terms (commercial use through the API)",
        about: "upscale the input clip with Topaz Proteus: steady frame to frame, where a per-frame upscaler shimmers",
    },
    RemoteModel {
        id: "fal/seedvr2-video-upscale",
        label: "SeedVR2 Video Upscale",
        short: "seedvr2 video",
        service: Service::Fal,
        endpoint: "fal-ai/seedvr/upscale/video",
        dialect: Dialect::SeedVrVideo,
        task: Task::VideoUpscale,
        commercial: true,
        licence: "fal lists it for commercial use",
        about: "restore and upscale the input clip with SeedVR2: for soft, noisy or generated footage",
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
        about: "the faster Nano Banana; 1K, or 2K from 2 MP",
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
        about: "edit a picture by instruction, the faster Nano Banana",
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
        about: "OpenAI's image model; quality defaults to high, extra {\"quality\": \"medium\"} lowers it",
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
        about: "edit a picture by instruction with GPT Image",
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
        about: "repaint the white of the matte from a prompt with GPT Image",
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
        about: "surface normals from one picture, MoGe-2; raw n*0.5+0.5",
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
        about: "relative depth (near is bright) from one picture, raw grey",
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
        about: "relative depth (near is bright) from one picture, raw grey",
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
        about: "normals derived from MiDaS depth, raw; rougher than MoGe or Marigold",
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

/// The submenus of a GenAI Pipeline's step menu, in menu order. Every
/// [`Task::menu_group`] is one of these.
pub const MENU_GROUPS: [&str; 10] = [
    "image generation",
    "image edit",
    "upscale",
    "video generation",
    "video edit",
    "depth, normals, materials",
    "relight",
    "matte",
    "3D",
    "prompt tools",
];

impl Task {
    /// Which submenu of the step menu a model of this task sits in: fifty
    /// models in one column run off the screen, and a new model lands in
    /// the right place just by naming its task.
    pub fn menu_group(self) -> &'static str {
        match self {
            Task::TextToImage => "image generation",
            Task::Edit | Task::EditRef | Task::Inpaint | Task::Outpaint => "image edit",
            Task::Upscale | Task::VideoUpscale => "upscale",
            Task::ImageToVideo => "video generation",
            Task::VideoEdit | Task::VideoInpaint => "video edit",
            Task::Depth | Task::Normals | Task::Albedo => "depth, normals, materials",
            Task::Relight => "relight",
            Task::Cutout => "matte",
            Task::ImageTo3D => "3D",
            Task::Enhance | Task::Describe => "prompt tools",
        }
    }

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
            Task::VideoUpscale => "video upscale",
            Task::Outpaint => "outpaint",
            Task::EditRef => "edit with a second picture",
            Task::ImageTo3D => "image to 3D",
            Task::Enhance => "prompt enhancer",
            Task::Describe => "describe",
        }
    }

    /// Does this task return a video rather than a picture?
    pub fn is_video(self) -> bool {
        matches!(self, Task::ImageToVideo | Task::VideoEdit | Task::VideoInpaint | Task::VideoUpscale)
    }

    /// Does this task READ a clip (the node's input over a range)?
    pub fn takes_clip(self) -> bool {
        matches!(self, Task::VideoEdit | Task::VideoInpaint)
    }

    /// Does this task return a 3D capture (a splat file) rather than a
    /// picture? Its take is shown by a Splat node, not by the GenAI node.
    pub fn is_3d(self) -> bool {
        matches!(self, Task::ImageTo3D)
    }

    /// Does this task return TEXT? Such a model is a step inside a
    /// pipeline (it feeds another step's prompt), never a one-model node:
    /// a node's output is a picture.
    pub fn is_text(self) -> bool {
        matches!(self, Task::Enhance | Task::Describe)
    }

    /// Does this task make its input BIGGER? The node then takes the
    /// result's own size as its format, instead of stretching the result
    /// back over its input's.
    pub fn resizes(self) -> bool {
        matches!(self, Task::Upscale | Task::VideoUpscale | Task::Outpaint)
    }

    /// Does this task hand back MEASUREMENTS rather than a picture? A data
    /// take plays back as stored values, with no sRGB transform.
    pub fn is_data(self) -> bool {
        matches!(self, Task::Depth | Task::Normals)
    }

    pub const ALL: [Task; 18] = [
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
        Task::VideoUpscale,
        Task::Outpaint,
        Task::EditRef,
        Task::ImageTo3D,
        Task::Enhance,
        Task::Describe,
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
    fn every_task_is_in_a_listed_menu_group() {
        for t in Task::ALL {
            assert!(MENU_GROUPS.contains(&t.menu_group()), "{t:?}");
        }
        for g in MENU_GROUPS {
            assert!(Task::ALL.iter().any(|t| t.menu_group() == g), "{g} is empty");
        }
    }

    #[test]
    fn no_endpoint_carries_a_scheme_or_host() {
        // An endpoint is a path on the service; the base URL is the app's.
        for m in CATALOG {
            assert!(!m.endpoint.contains("://") && !m.endpoint.starts_with('/'), "{}", m.id);
        }
    }

    /// A 3D take becomes a Splat node, and a Splat node carries no licence
    /// tag: a research-only 3D model would lose its warning on the way.
    #[test]
    fn no_3d_model_is_research_only() {
        for m in CATALOG.iter().filter(|m| m.task.is_3d()) {
            assert!(m.commercial, "{}: give Splat a licence tag first", m.id);
        }
    }

    #[test]
    fn dialects_match_their_tasks() {
        for m in CATALOG {
            let video = matches!(
                m.dialect,
                Dialect::Seedance { .. }
                    | Dialect::KlingEdit
                    | Dialect::LucyEdit
                    | Dialect::LtxInpaint
                    | Dialect::BriaVideoEraser
                    | Dialect::WanVace
                    | Dialect::Wan27Edit
                    | Dialect::SeedanceEdit
                    | Dialect::TopazVideo { .. }
                    | Dialect::SeedVrVideo
            );
            assert_eq!(video, m.task.is_video(), "{}", m.id);
        }
        let (lo, hi) = Dialect::Seedream.mp_range();
        assert!(lo * 1e6 > 1024.0 * 1024.0 && hi * 1e6 < 1536.0 * 1536.0);
    }
}
