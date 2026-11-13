//! Cryptomatte: the panel and the viewer click that picks an object. The
//! engine half (id encoding, manifest, uniform packing) is
//! `engine::cryptomatte`; this file is what needs the app shell — readbacks
//! and the node panel. The node's one input is the multi-channel cable; its
//! ranks are its own reader slots, minted by `rebuild_playback` and fed by
//! the reader-entries arm there, on the ChannelSelect precedent.
//!
//! Picking reads the pixel ON SCREEN, the Keyer's rule: the node has to be
//! the view target, where the fused pass writes rank 0's id chunks
//! (scaled into [0,1), exact in f16, or lit yellow for a picked object —
//! still exact) with the matte in alpha. One 1x1 readback per click; the
//! id decodes straight out of the texel.

use crate::theme;
use engine::cryptomatte::{self, CryptoDef, CryptomatteParams, MAX_PICKS, RANKS};
use engine::decode::ExrDecoder;
use engine::diskcache::DiskCache;
use engine::player::Sequence;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use web_time::{Duration, Instant};

/// What a click does with the object under it: the usual Cryptomatte hands —
/// ctrl-click (cmd on a Mac) ADDS it to the matte, ctrl+shift-click
/// REMOVES it, and a plain click does nothing (it stays free for the
/// viewer's own gestures).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickMode {
    Add,
    Remove,
}

impl PickMode {
    /// `None` = not a pick.
    pub fn from_mods(cmd: bool, shift: bool) -> Option<Self> {
        match (cmd, shift) {
            (true, true) => Some(PickMode::Remove),
            (true, false) => Some(PickMode::Add),
            _ => None,
        }
    }
}

struct PixelRead {
    buffer: wgpu::Buffer,
    node: u64,
    mode: PickMode,
    ready: Arc<AtomicBool>,
    mapped: bool,
}

/// A "bake all" in flight: the matte of every frame, decoded on background
/// threads straight into the NVMe tier. It is only a WARM-UP — any frame
/// the player asks for bakes itself the same way — so cancelling loses
/// nothing and a pick edit just makes it moot.
pub struct BakeJob {
    pub node: u64,
    layer: String,
    total: u64,
    done: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
}

/// What the panel's bake row shows.
pub struct BakeUi {
    /// Frames of THIS selection's matte already in the NVMe tier.
    pub have: u64,
    pub total: u64,
    /// `(done, total)` of a running bake-all for this selection.
    pub running: Option<(u64, u64)>,
    /// No NVMe cache directory: nothing persists between requests.
    pub no_disk: bool,
}

/// Where the NVMe tier keeps frame `i` of `seq` decoded as `layer`.
pub fn matte_file(disk: &DiskCache, seq: &Sequence, layer: &str, i: u64) -> Option<PathBuf> {
    let path = seq.frames.get(i as usize)?;
    disk.frame_file_layer_scaled(path, Some(layer), seq.file_frame(i), seq.proxy)
}

/// Everything the cryptomatte needs from the app shell, in one field.
#[derive(Default)]
pub struct CryptoState {
    /// Header metadata per frame file, read once: a header read per panel
    /// frame would be a header read on the UI thread every frame.
    manifests: HashMap<PathBuf, Vec<CryptoDef>>,
    /// A node the panel asked to become the view target (picking needs it).
    pub want_view: Option<u64>,
    /// A click on the image while a node is viewed: `(node, x, y, mode)` in
    /// output pixels. The encode step reads the texel; the poll step
    /// resolves it to a pick.
    pub click: Option<(u64, i32, i32, PickMode)>,
    /// The "add by name" text box.
    pub name_entry: String,
    /// The panel asked for a bake-all of this node (deferred: it needs the
    /// disk cache and the decoder, which the panel does not hold).
    pub want_bake: Option<u64>,
    bake: Option<BakeJob>,
    /// Baked-frame counts per node, recounted at most twice a second: a
    /// count is a stat per frame, and the panel draws every frame.
    counts: HashMap<u64, (Instant, String, u64)>,
    reads: Vec<PixelRead>,
}

impl CryptoState {
    /// The cryptomattes a frame file declares, cached by path.
    pub fn defs(&mut self, frame: &Path) -> &[CryptoDef] {
        self.manifests
            .entry(frame.to_path_buf())
            .or_insert_with(|| cryptomatte::defs(frame).unwrap_or_default())
    }

    /// Start warming every frame of `layer` for `node`. Replaces a running
    /// job. Frames already in the tier are skipped, so a second press (or a
    /// press after playing through half the shot) only does what is left.
    pub fn start_bake(
        &mut self,
        node: u64,
        seq: Arc<Sequence>,
        layer: String,
        disk: Arc<DiskCache>,
        decoder: Arc<dyn ExrDecoder>,
    ) {
        self.cancel_bake();
        let total = seq.frames.len() as u64;
        let done = Arc::new(AtomicU64::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        self.bake = Some(BakeJob {
            node,
            layer: layer.clone(),
            total,
            done: done.clone(),
            cancel: cancel.clone(),
        });
        crate::bg::spawn(move || {
            use rayon::prelude::*;
            let info = seq.info;
            (0..total).into_par_iter().for_each(|i| {
                if cancel.load(Ordering::Acquire) {
                    return;
                }
                let landed = (|| -> anyhow::Result<()> {
                    let Some(file) = matte_file(&disk, &seq, &layer, i) else { return Ok(()) };
                    if file.exists() {
                        return Ok(());
                    }
                    // Exactly what the player does on a miss: decode at the
                    // staging stride, write the tier's frame.
                    let mut owned = vec![half::f16::ZERO; info.row_stride_f16() * info.height as usize];
                    decoder.decode_frame_into(
                        &seq.frames[i as usize],
                        info,
                        info.row_stride_f16(),
                        Some(&layer),
                        seq.file_frame(i),
                        &mut owned,
                    )?;
                    disk.write_frame(&file, info, &owned)
                })();
                if let Err(e) = landed {
                    log::warn!("cryptomatte bake frame {i}: {e:#}");
                }
                done.fetch_add(1, Ordering::Release);
            });
        });
    }

    pub fn cancel_bake(&mut self) {
        if let Some(j) = self.bake.take() {
            j.cancel.store(true, Ordering::Release);
        }
    }

    /// The bake row's numbers for `node`'s current selection.
    pub fn bake_ui(
        &mut self,
        node: u64,
        seq: Option<&Sequence>,
        layer: &str,
        disk: Option<&DiskCache>,
    ) -> BakeUi {
        let total = seq.map(|s| s.frames.len() as u64).unwrap_or(0);
        // A job for another selection is moot; one that has finished is done.
        if self.bake.as_ref().is_some_and(|j| {
            j.node == node && (j.layer != layer || j.done.load(Ordering::Acquire) >= j.total)
        }) {
            self.cancel_bake();
            self.counts.remove(&node);
        }
        let running = self
            .bake
            .as_ref()
            .filter(|j| j.node == node)
            .map(|j| (j.done.load(Ordering::Acquire), j.total));
        let (Some(seq), Some(disk)) = (seq, disk) else {
            return BakeUi { have: 0, total, running, no_disk: disk.is_none() };
        };
        let fresh = self.counts.get(&node).is_some_and(|(t, l, _)| {
            l == layer && t.elapsed() < Duration::from_millis(500)
        });
        if !fresh {
            let have = (0..total)
                .filter(|i| matte_file(disk, seq, layer, *i).is_some_and(|f| f.exists()))
                .count() as u64;
            self.counts.insert(node, (Instant::now(), layer.to_string(), have));
        }
        let have = self.counts.get(&node).map(|c| c.2).unwrap_or(0);
        BakeUi { have, total, running, no_disk: false }
    }

    /// Queue the one-texel readback for a pending click.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        tex: &wgpu::Texture,
        out: (u32, u32),
    ) {
        let Some((node, x, y, mode)) = self.click.take() else { return };
        let (ow, oh) = out;
        if x < 0 || y < 0 || x >= ow as i32 || y >= oh as i32 {
            return;
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("crypto-pick"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x as u32, y: y as u32, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        self.reads.push(PixelRead {
            buffer,
            node,
            mode,
            ready: Arc::new(AtomicBool::new(false)),
            mapped: false,
        });
    }

    /// Landed clicks as `(node, id, mode)`. Call once a frame.
    pub fn poll(&mut self) -> Vec<(u64, u32, PickMode)> {
        for r in self.reads.iter_mut() {
            if !r.mapped {
                r.mapped = true;
                let flag = r.ready.clone();
                r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    if res.is_ok() {
                        flag.store(true, Ordering::Release);
                    }
                });
            }
        }
        let mut out = Vec::new();
        let mut done = Vec::new();
        for (i, r) in self.reads.iter().enumerate() {
            if !r.ready.load(Ordering::Acquire) {
                continue;
            }
            let view = r.buffer.slice(..).get_mapped_range().expect("a mapped buffer");
            let px: &[half::f16] = bytemuck::cast_slice(&view[..8]);
            let id = cryptomatte::decode_view([px[0].to_f32(), px[1].to_f32(), px[2].to_f32()]);
            drop(view);
            r.buffer.unmap();
            // Id 0 is the background (no object hashes to it: the spec
            // keeps ids out of the zero exponent band): a click there
            // picks nothing. Alpha is the matte, so it cannot say.
            if id != 0 {
                out.push((r.node, id, r.mode));
            }
            done.push(i);
        }
        for i in done.into_iter().rev() {
            self.reads.remove(i);
        }
        out
    }
}

/// The object name for an id: the manifest's, else the hex id.
pub fn name_of(defs: &[CryptoDef], layer: &str, id: u32) -> String {
    defs.iter()
        .filter(|d| layer.is_empty() || d.name == layer)
        .chain(defs.iter())
        .find_map(|d| d.name_of(id))
        .map(str::to_string)
        .unwrap_or_else(|| format!("{id:08x}"))
}

pub struct CryptoOut {
    /// Make this node the view target (picking needs it).
    pub want_view: bool,
    /// The crypto layer changed: the same slots now decode other pairs.
    pub layer_changed: bool,
    /// Warm every frame of the matte into the NVMe tier.
    pub bake_all: bool,
    pub bake_cancel: bool,
}

/// The node panel. `defs` is what the file behind the cable declares (empty
/// when nothing is wired or the file has no cryptomatte); `wired` is whether
/// the cable reaches a Read at all.
pub fn crypto_ui(
    ui: &mut egui::Ui,
    p: &mut CryptomatteParams,
    viewing: bool,
    defs: &[CryptoDef],
    wired: bool,
    name_entry: &mut String,
    bake: Option<&BakeUi>,
) -> CryptoOut {
    let mut out = CryptoOut {
        want_view: false,
        layer_changed: false,
        bake_all: false,
        bake_cancel: false,
    };

    // --- the cable ---------------------------------------------------------
    if defs.is_empty() {
        if wired {
            ui.weak("the file behind the cable has no cryptomatte metadata");
        } else {
            ui.colored_label(
                theme::error(),
                "not wired to a Load's channel output",
            );
            ui.weak("wire a Load (set to all-channels) into this node's input");
        }
    } else {
        if p.layer.is_empty() || !defs.iter().any(|d| d.name == p.layer) {
            p.layer = defs[0].name.clone();
            out.layer_changed = true;
        }
        ui.horizontal(|ui| {
            ui.label("layer");
            let before = p.layer.clone();
            egui::ComboBox::from_id_salt("crypto-layer")
                .selected_text(p.layer.clone())
                .show_ui(ui, |ui| {
                    for d in defs {
                        ui.selectable_value(&mut p.layer, d.name.clone(), &d.name);
                    }
                });
            if p.layer != before {
                out.layer_changed = true;
            }
        });
        ui.weak(format!(
            "{RANKS} ranks off the cable, decoded id-exact into this node's own readers"
        ));
    }
    ui.separator();

    // --- picking ---------------------------------------------------------
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.invert, "invert");
        if viewing {
            ui.weak(if cfg!(target_os = "macos") {
                "cmd-click an object to add it, cmd+shift-click removes; A shows the matte"
            } else {
                "ctrl-click an object to add it, ctrl+shift-click removes; A shows the matte"
            });
        } else if ui
            .button("view")
            .on_hover_text("view this node: picking reads the objects off the screen")
            .clicked()
        {
            out.want_view = true;
        }
    });
    ui.separator();

    // --- bake ------------------------------------------------------------
    // Seven f32 id layers a frame is this node's cost, so it is paid once:
    // the matte is its own reader, computed on the decode thread on a miss
    // and kept by the NVMe tier. Say so, with the numbers (manifesto 8).
    if let Some(b) = bake {
        if b.no_disk {
            ui.colored_label(
                theme::error(),
                "no NVMe cache directory: the matte is recomputed on every request",
            );
        } else if let Some((done, total)) = b.running {
            ui.horizontal(|ui| {
                let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(150.0)
                        .text(format!("baking {done}/{total}")),
                );
                if ui.button("cancel").clicked() {
                    out.bake_cancel = true;
                }
            });
        } else {
            ui.horizontal(|ui| {
                ui.monospace(format!("baked {}/{}", b.have, b.total));
                if b.have < b.total
                    && ui
                        .button("bake all")
                        .on_hover_text(
                            "frames bake themselves the first time they are asked for; \
                             this warms the rest in the background. A pick edit starts a \
                             fresh set — the old one stays in the cache until it ages out",
                        )
                        .clicked()
                {
                    out.bake_all = true;
                }
            });
        }
        ui.weak(if viewing {
            "viewed: live ids (7 layers a frame) so you can pick; downstream reads the bake"
        } else {
            "downstream reads ONE baked matte per frame, not the id layers"
        });
        ui.separator();
    }

    // --- picks -----------------------------------------------------------
    ui.horizontal(|ui| {
        ui.label(format!("picked ({}/{MAX_PICKS})", p.picks.len()));
        if !p.picks.is_empty() && ui.small_button("clear").clicked() {
            p.picks.clear();
        }
    });
    let mut remove = None;
    for (i, pk) in p.picks.iter().enumerate() {
        ui.horizontal(|ui| {
            if ui.small_button("×").clicked() {
                remove = Some(i);
            }
            ui.label(&pk.name).on_hover_text(format!("{:08x}", pk.id));
        });
    }
    if let Some(i) = remove {
        p.picks.remove(i);
    }
    ui.horizontal(|ui| {
        let edit = ui.add(
            egui::TextEdit::singleline(name_entry).hint_text("object name").desired_width(140.0),
        );
        let go = ui.button("add").clicked()
            || (edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
        if go && !name_entry.trim().is_empty() {
            let name = name_entry.trim().to_string();
            // A name the manifest knows uses ITS id, so a renderer that
            // hashed differently still resolves; anything else is hashed.
            let id = defs
                .iter()
                .find_map(|d| d.manifest.get(&name).copied())
                .unwrap_or_else(|| cryptomatte::id_of_name(&name));
            p.toggle(&name, id, true);
            name_entry.clear();
        }
    });
    // The manifest, as a list to tick from — the panel's answer to "what
    // is in this render".
    let names: Vec<&str> = defs
        .iter()
        .filter(|d| d.name == p.layer)
        .flat_map(|d| d.manifest.keys().map(String::as_str))
        .collect();
    if !names.is_empty() {
        let mut names = names;
        names.sort_unstable();
        egui::CollapsingHeader::new(format!("manifest ({})", names.len()))
            .id_salt("crypto-manifest")
            .show(ui, |ui| {
                egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                    for n in names {
                        let id = defs
                            .iter()
                            .find_map(|d| d.manifest.get(n).copied())
                            .unwrap_or(0);
                        let on = p.picks.iter().any(|pk| pk.id == id);
                        if ui.selectable_label(on, n).clicked() {
                            p.toggle(n, id, !on);
                        }
                    }
                });
            });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::EditorGraph;
    use engine::graph::NodeKind;
    use egui::Pos2;

    /// The cable is a lazy reference: lowering drops the node's input edge
    /// (the multi Read is not decoded on its account) and the engine graph
    /// instances the node's OWN reader set, one per source.
    #[test]
    fn lowering_drops_the_cable_and_instances_the_reader_set() {
        let mut g = EditorGraph::default();
        let read = g.add(
            {
                let mut k = NodeKind::read(0, "/x/crypto.exr");
                if let NodeKind::Read { multi, .. } = &mut k {
                    *multi = true;
                }
                k
            },
            Pos2::new(0.0, 0.0),
        );
        let mut p = CryptomatteParams { layer: "CryptoObject".into(), ..Default::default() };
        p.sources = (10..10 + cryptomatte::SOURCES).collect();
        let node = g.add(NodeKind::Cryptomatte(p), Pos2::new(0.0, 200.0));
        g.connect(node, 0, read);
        g.view = Some(node);
        assert_eq!(g.channel_source(node), Some(read));
        let eg = g.to_engine_graph();
        assert!(eg.input_of(node, 0).is_none(), "the cable edge is dropped");
        let inst = eg.reader_instances();
        let mut mine: Vec<usize> =
            inst.iter().filter(|i| i.node == node).map(|i| i.sub).collect();
        mine.sort_unstable();
        assert_eq!(mine, (0..cryptomatte::SOURCES).collect::<Vec<_>>());
        // (The multi Read itself still instances as an unconsumed root —
        // exactly as it does behind a ChannelSelect; that is the Read's own
        // account, not this node's.)
        // Each sub has its own key, so slots cannot collide.
        let keys: std::collections::HashSet<_> = inst.iter().map(|i| (i.node, i.key)).collect();
        assert_eq!(keys.len(), inst.len());
        // What each source decodes.
        let NodeKind::Cryptomatte(c) = &g.node(node).unwrap().kind else { panic!() };
        assert_eq!(c.source_layer(0), None);
        assert_eq!(c.source_layer(1).as_deref(), Some("CryptoObject00#0"));
        assert_eq!(c.source_layer(6).as_deref(), Some("CryptoObject02#1"));
        // Unminted renders nothing, but stays a node.
        assert!(!CryptomatteParams::default().minted());
    }

    /// "bake all" lands every frame of THIS selection's matte in the NVMe
    /// tier, the panel counts them, and another selection starts from zero.
    #[test]
    fn bake_all_warms_the_tier_per_selection() {
        let dir = engine::paths::temp_dir().join(format!("wobble-cryptobake-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let plate = dir.join("plate");
        cryptomatte::write_plate(&plate, 64, 36, 3).unwrap();
        let decoder: Arc<dyn ExrDecoder> = Arc::new(engine::decode::ExrsDecoder::default());
        let seq = Arc::new(Sequence::from_dir(&plate, decoder.as_ref()).unwrap());
        let disk = Arc::new(DiskCache::new(dir.join("cache"), 1 << 30).unwrap());
        let mut p = CryptomatteParams { layer: "CryptoObject".into(), ..Default::default() };
        p.toggle("cube", cryptomatte::id_of_name("cube"), true);
        let layer = p.matte_layer();

        let mut st = CryptoState::default();
        let ui = st.bake_ui(7, Some(&seq), &layer, Some(&disk));
        assert_eq!((ui.have, ui.total), (0, 3));
        st.start_bake(7, seq.clone(), layer.clone(), disk.clone(), decoder.clone());
        let t0 = Instant::now();
        loop {
            st.counts.clear(); // defeat the half-second throttle
            let ui = st.bake_ui(7, Some(&seq), &layer, Some(&disk));
            if ui.have == 3 && ui.running.is_none() {
                break;
            }
            assert!(t0.elapsed() < Duration::from_secs(30), "bake stuck at {}/3", ui.have);
            std::thread::sleep(Duration::from_millis(20));
        }
        // What landed is the matte: read one back through the tier.
        let f = matte_file(&disk, &seq, &layer, 0).unwrap();
        let info = seq.info;
        let mut px = vec![half::f16::ZERO; info.row_stride_f16() * info.height as usize];
        assert!(DiskCache::read_frame_into(&f, info, &mut px).unwrap());
        let at = |x: usize, y: usize| px[y * info.row_stride_f16() + x * 4].to_f32();
        assert_eq!(at(51, 18), 1.0, "inside the cube (0.8, 0.5)");
        assert_eq!(at(2, 2), 0.0, "background");
        // Another selection is another set of files.
        p.toggle("sphere", cryptomatte::id_of_name("sphere"), true);
        st.counts.clear();
        let ui = st.bake_ui(7, Some(&seq), &p.matte_layer(), Some(&disk));
        assert_eq!(ui.have, 0);
        // No cache directory: say so rather than count.
        assert!(st.bake_ui(7, Some(&seq), &layer, None).no_disk);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_resolve_through_the_manifest_or_fall_back_to_hex() {
        let mut m = HashMap::new();
        m.insert("cube".to_string(), 0x1234_5678u32);
        let defs = vec![CryptoDef { name: "CryptoObject".into(), manifest: m }];
        assert_eq!(name_of(&defs, "CryptoObject", 0x1234_5678), "cube");
        assert_eq!(name_of(&defs, "CryptoMaterial", 0x1234_5678), "cube", "other layers as fallback");
        assert_eq!(name_of(&defs, "", 0xdead_beef), "deadbeef");
    }
}
