//! Colour-keyer sampling: drag over the image to fit a sample.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Colour-keyer sampling: drag over the image to fit a sample.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn viewer_keyer_sample(
        &mut self,
        ui: &mut egui::Ui,
        cmd: bool,
        dropping: bool,
        nav: &egui::Response,
        owf: f32,
        ohf: f32,
        rect: egui::Rect,
        scrubbing: bool,
    ) {
        // Colour-keyer sampling, the Paint gesture's sibling: with a keyer
        // as the VIEW TARGET, left-drag over the image fits a sample to the
        // pixels you dragged over, alt-drag fits one that subtracts. It has
        // to be the view target because the pixels are read back off the
        // screen — which is exactly why the node shows its own input while
        // being viewed.
        let keyer_id = self.egraph.active_view().filter(|id| {
            !dropping
                && matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Keyer(_)))
        });
        if let Some(kid) = keyer_id {
            let started = nav.drag_started_by(egui::PointerButton::Primary);
            let dragging = nav.dragged_by(egui::PointerButton::Primary);
            if !scrubbing && !cmd && started {
                let (alt, shift) = ui.input(|i| (i.modifiers.alt, i.modifiers.shift));
                let sel = self.keyer.sel.get(&kid).copied().unwrap_or(0);
                let mut idx = sel;
                if let Some(editor::ENode { kind: NodeKind::Keyer(k), .. }) =
                    self.egraph.node_mut(kid)
                {
                    // A plain drag REFITS the selected sample; shift-drag
                    // adds a new one. Refitting is what you do over and over
                    // — the first sample is rarely right and you re-drag to
                    // correct it — so it gets the unmodified gesture, and
                    // adding a second colour to the key is the deliberate
                    // act that takes a modifier.
                    //
                    // A fresh node's placeholder blob is replaced either way:
                    // the first sample should BE the key, not join it.
                    let virgin = k.blobs.len() == 1
                        && k.blobs[0] == engine::keyer::KeyBlob::default();
                    let full = k.blobs.len() >= engine::keyer::MAX_BLOBS;
                    if !shift || virgin || full {
                        idx = sel.min(k.blobs.len().saturating_sub(1));
                    } else {
                        let seed = k.blobs.get(sel).copied().unwrap_or_default();
                        k.blobs.push(seed);
                        idx = k.blobs.len() - 1;
                    }
                }
                self.keyer.sel.insert(kid, idx);
                self.keyer.begin_drag(kid, alt, idx);
            }
            if self.keyer.dragging() && (started || dragging) {
                self.keyer.cursor = ui.ctx().pointer_latest_pos().map(|p| {
                    (
                        ((p.x - rect.min.x) / rect.width() * owf) as i32,
                        ((p.y - rect.min.y) / rect.height() * ohf) as i32,
                    )
                });
            }
            if self.keyer.dragging() && !dragging && !started {
                self.keyer.end_drag();
            }
        }
    }
}
