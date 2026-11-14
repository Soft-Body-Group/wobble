//! Cryptomatte: hands the picker panel what it needs.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Cryptomatte: hands the picker panel what it needs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    cp: &mut engine::cryptomatte::CryptomatteParams,
    node: u64,
    viewing: bool,
    crypto_bake: &Option<cryptoui::BakeUi>,
    crypto_defs: &Vec<engine::cryptomatte::CryptoDef>,
    crypto_read: Option<u64>,
    crypto: &mut cryptoui::CryptoState,
    playback_dirty: &mut bool,
) {
    let matte_before = cp.matte_layer();
    let out = cryptoui::crypto_ui(
        ui,
        cp,
        viewing,
        &crypto_defs,
        crypto_read.is_some(),
        &mut crypto.name_entry,
        crypto_bake.as_ref(),
    );
    if out.want_view {
        crypto.want_view = Some(node);
    }
    if out.bake_all {
        crypto.want_bake = Some(node);
    }
    if out.bake_cancel {
        crypto.cancel_bake();
    }
    // A different selection is a different matte behind
    // the same reader slot. Viewed, that reader is not
    // in the rig (the live ids are) and the view hop
    // away rebuilds it anyway — no hitch per click.
    if !viewing && cp.matte_layer() != matte_before {
        *playback_dirty = true;
    }
    if out.layer_changed {
        // A different layer is different pixels behind
        // the same slots: rebuild the rig.
        *playback_dirty = true;
    }
}
