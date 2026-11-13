//! A key grabbed on a motion path: move that key, at its frame.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// A key grabbed on a motion path: move that key, at its frame.
    pub(crate) fn gizmo_path_edit(
        &mut self,
        path_edit: Option<(u64, f64, [f32; 2], (f32, f32))>,
        patch_driven: &Vec<(u64, anim::Channel)>,
    ) {
        // A key grabbed on a motion path: move THAT key, at its frame.
        // Through the same write-back as every other manipulator, so a
        // locked or cable-driven channel refuses it like anywhere else.
        if let Some((node, frame, at, size)) = path_edit {
            let autokey = self.autokey;
            if let Some(n) = self.egraph.node_mut(node) {
                let editor::ENode { kind, anim, .. } = n;
                let mut before = kind.clone();
                anim::apply_at(&mut before, anim, frame as f32);
                let mut after = before.clone();
                set_element_point(&mut after, size, at);
                let chans = element_channels(&after);
                keyed_write_back(
                    anim,
                    kind,
                    &before,
                    &after,
                    frame,
                    chans,
                    autokey,
                    &patch_driven,
                    node,
                );
            }
        }
    }
}
