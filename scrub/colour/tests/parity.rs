//! Scrub always has parity with wobble's colour. These files are copies, so
//! the check is that they still are: a colour change in wobble's engine
//! fails here until it is ported.

use colour::color::{Colorspace, Curve, InputSpace, Primaries};
use colour::viewtf::{Look, ViewPreset, ViewTransform};

#[test]
fn the_shared_colour_files_are_wobbles() {
    assert_eq!(include_str!("../src/color.rs"), include_str!("../../../engine/src/color.rs"), "color.rs");
    assert_eq!(include_str!("../src/viewtf/aces.rs"), include_str!("../../../engine/src/viewtf/aces.rs"), "aces.rs");
    assert_eq!(include_str!("../src/viewtf/agx.rs"), include_str!("../../../engine/src/viewtf/agx.rs"), "agx.rs");
}

/// `viewtf.rs` differs by OpenDRT alone, wobble's GPLv3 feature.
#[test]
fn viewtf_differs_only_by_opendrt() {
    let wobble = include_str!("../../../engine/src/viewtf.rs")
        .replace(
            "#[cfg(feature = \"opendrt\")]\npub mod opendrt;\n\n#[cfg(not(feature = \"opendrt\"))]\n",
            "// (wobble's OpenDRT is GPLv3, behind a feature there: only its stub here.)\n",
        )
        .replace("cfg!(feature = \"opendrt\")", "false");
    assert_eq!(include_str!("../src/viewtf.rs"), wobble);
}

/// A project saved before these existed still loads, and the new picks
/// survive a save.
#[test]
fn projects_keep_loading() {
    let old: Colorspace = serde_json::from_str(r#"{"primaries":"Rec709","curve":"Srgb"}"#).unwrap();
    assert_eq!(old, Colorspace::SRGB);
    let t: ViewTransform = serde_json::from_str(r#""Aces13""#).unwrap();
    assert_eq!(t, ViewTransform::Aces13);
    for (_, cs) in Colorspace::PRESETS {
        let v = InputSpace::Explicit(cs);
        assert_eq!(serde_json::from_str::<InputSpace>(&serde_json::to_string(&v).unwrap()).unwrap(), v);
    }
    assert_eq!(Colorspace::ACES_OUTPUT_REC709.curve, Curve::AcesOutRec709);
    assert_eq!(Colorspace::DAVINCI_INTERMEDIATE.primaries, Primaries::DaVinciWideGamut);
}

#[test]
fn the_aces12_views_are_offered_by_name() {
    let named = |n: &str| ViewPreset::CATALOGUE.iter().find(|(l, _)| *l == n).map(|(_, p)| *p).unwrap();
    assert_eq!(named("sRGB (ACES)").transform, ViewTransform::Aces12);
    assert_eq!(named("Rec.709 (ACES)").transform, ViewTransform::Aces12);
    assert_eq!(ViewTransform::Aces12.looks(), &[Look::None]);
}
