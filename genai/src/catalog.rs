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
}

impl Dialect {
    /// The megapixel range a text-to-image call must land in.
    pub fn mp_range(self) -> (f32, f32) {
        match self {
            // 1024² = 1.049 MP; 1536² = 2.359 MP (rounded to 16 px by
            // `image::fit_size`, so a hair inside both).
            Dialect::Seedream => (1.06, 2.34),
            _ => (0.1, 4.0),
        }
    }

    /// Does the request carry our authored seed? When not, the seed still
    /// keys the cache (a "new seed" is how you ask for a variation).
    pub fn sends_seed(self) -> bool {
        matches!(self, Dialect::Flux)
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
        }
    }

    /// Does this task return a video rather than a picture?
    pub fn is_video(self) -> bool {
        matches!(self, Task::ImageToVideo)
    }

    pub const ALL: [Task; 6] = [
        Task::TextToImage,
        Task::Edit,
        Task::Inpaint,
        Task::Upscale,
        Task::Cutout,
        Task::ImageToVideo,
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
            let video = matches!(m.dialect, Dialect::Seedance { .. });
            assert_eq!(video, m.task.is_video(), "{}", m.id);
        }
        let (lo, hi) = Dialect::Seedream.mp_range();
        assert!(lo * 1e6 > 1024.0 * 1024.0 && hi * 1e6 < 1536.0 * 1536.0);
    }
}
