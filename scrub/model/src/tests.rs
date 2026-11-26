use super::*;

const F24: Fps = Fps::new(24, 1);

fn seq() -> Sequence {
    let mut s = Sequence::new(F24);
    for (i, audio) in [(0, true), (1, false)] {
        s.add_media(Media { path: format!("/m{i}.mov").into(), name: format!("m{i}"), frames: 1000, fps: F24, has_audio: audio, width: 64, height: 36, generated: false, audio_only: false });
    }
    s
}

fn spans(s: &Sequence, t: usize) -> Vec<(i64, i64, usize, i64)> {
    s.tracks[t].clips.iter().map(|c| (c.start, c.len, c.media, c.src_in)).collect()
}

#[test]
fn overwrite_replaces_the_span_and_moves_nothing() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.overwrite(Source::new(1, 500, 20), 40, Targets::default());
    assert_eq!(spans(&s, V1), vec![(0, 40, 0, 0), (40, 20, 1, 500), (60, 40, 0, 60)]);
    // m1 has no audio: A1 keeps m0's audio across the whole length.
    assert_eq!(spans(&s, A1), vec![(0, 100, 0, 0)]);
    assert_eq!(s.duration(), 100);
    assert!(s.is_consistent());
}

#[test]
fn insert_ripples_every_track_and_cuts_across_the_point() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.insert(Source::new(1, 0, 10), 30, Targets { video: Some(V2), audio: None });
    assert_eq!(spans(&s, V1), vec![(0, 30, 0, 0), (40, 70, 0, 30)]);
    assert_eq!(spans(&s, V2), vec![(30, 10, 1, 0)]);
    assert_eq!(spans(&s, A1), vec![(0, 30, 0, 0), (40, 70, 0, 30)], "audio stays in sync");
    assert_eq!(s.duration(), 110);
    assert!(s.is_consistent());
}

#[test]
fn the_topmost_video_clip_plays() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.overwrite(Source::new(1, 200, 10), 50, Targets { video: Some(V3), audio: None });
    let (c, f) = s.video_at(55).unwrap();
    assert_eq!((c.media, f), (1, 205));
    let (c, f) = s.video_at(60).unwrap();
    assert_eq!((c.media, f), (0, 60));
    assert!(s.video_at(100).is_none());
}

#[test]
fn edit_points_step_through_every_cut() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.overwrite(Source::new(1, 0, 10), 120, Targets::default());
    assert_eq!(s.edit_points(), vec![0, 100, 120, 130]);
    assert_eq!(s.next_edit(100), Some(120));
    assert_eq!(s.prev_edit(100), Some(0));
    assert_eq!(s.next_edit(130), None);
}

#[test]
fn a_source_at_another_rate_plays_at_its_own_speed() {
    let mut s = seq();
    let m = s.add_media(Media { path: "/pal.mov".into(), name: "pal".into(), frames: 1000, fps: Fps::new(25, 1), has_audio: false, width: 64, height: 36, generated: false, audio_only: false });
    s.overwrite(Source::new(m, 0, 48), 0, Targets::default());
    assert_eq!(s.video_at(24).unwrap().1, 25, "one second in is source frame 25");
}

#[test]
fn ntsc_rates_convert_exactly_for_whole_seconds() {
    let ntsc = Fps::new(24000, 1001);
    assert_eq!(ntsc.convert(24000, Fps::new(48000, 1001)), 48000);
    assert_eq!(F24.convert(240, F24), 240);
}

#[test]
fn lift_leaves_a_gap_on_the_targets_only() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.lift(10, 20, Targets { video: Some(V1), audio: None });
    assert_eq!(spans(&s, V1), vec![(0, 10, 0, 0), (20, 80, 0, 20)]);
    assert_eq!(spans(&s, A1), vec![(0, 100, 0, 0)]);
}

#[test]
fn speed_changes_how_much_source_a_clip_covers() {
    let mut s = seq();
    s.overwrite(Source::new(0, 100, 48), 0, Targets::default());
    let len = s.set_speed(V1, 0, 2.0);
    assert_eq!(len, 24, "twice as fast, half as long");
    assert_eq!(s.video_at(10).unwrap().1, 120);
    assert_eq!(s.tracks[A1].clips[0].len, 24, "the linked audio follows");
    let len = s.set_speed(V1, 0, -1.0);
    assert_eq!(len, 48);
    assert_eq!(s.video_at(0).unwrap().1, 146, "backwards from the last frame it showed");
    assert_eq!(s.video_at(47).unwrap().1, 99);
    assert!(s.is_consistent());
}

#[test]
fn a_slower_clip_never_grows_into_the_next() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    s.overwrite(Source::new(1, 0, 10), 10, Targets::default());
    assert_eq!(s.set_speed(V1, 0, 0.5), 10);
    assert_eq!(s.video_at(9).unwrap().1, 4, "half speed: frame 9 shows source 4");
}

#[test]
fn fit_to_fill_stretches_the_source_to_the_record_range() {
    let mut s = seq();
    s.overwrite_fit(0, 0, 100, 0, 50, Targets::default());
    let c = &s.tracks[V1].clips[0];
    assert_eq!((c.len, c.speed), (50, 2.0));
    assert_eq!(s.video_at(49).unwrap().1, 98);
}

#[test]
fn lift_and_ripple_delete_take_the_linked_group() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    s.overwrite(Source::new(0, 50, 10), 10, Targets::default());
    s.overwrite(Source::new(0, 90, 10), 20, Targets::default());
    s.ripple_delete(V1, 1).unwrap();
    assert_eq!(spans(&s, V1), vec![(0, 10, 0, 0), (10, 10, 0, 90)]);
    assert_eq!(spans(&s, A1), vec![(0, 10, 0, 0), (10, 10, 0, 90)]);
    s.lift_clip(A1, 0);
    assert_eq!(spans(&s, V1), vec![(10, 10, 0, 90)]);
}

#[test]
fn ripple_delete_refuses_to_pull_another_track_out_of_sync() {
    let mut s = seq();
    s.overwrite(Source::new(1, 0, 20), 0, Targets { video: Some(V1), audio: None });
    s.overwrite(Source::new(1, 0, 5), 5, Targets { video: Some(V2), audio: None });
    assert!(s.ripple_delete(V2, 0).is_err());
    assert_eq!(s.tracks[V2].clips.len(), 1);
}

fn two_clips() -> Sequence {
    let mut s = seq();
    let t = Targets { video: Some(V1), audio: None };
    s.overwrite(Source::new(1, 100, 10), 0, t);
    s.overwrite(Source::new(1, 500, 10), 10, t);
    s
}

#[test]
fn roll_moves_the_cut_and_keeps_the_length() {
    let mut s = two_clips();
    assert_eq!(s.trim(V1, 10, Side::Roll, 3, false), Ok(3));
    assert_eq!(spans(&s, V1), vec![(0, 13, 1, 100), (13, 7, 1, 503)]);
    assert_eq!(s.duration(), 20);
}

#[test]
fn a_ripple_out_trim_moves_what_follows() {
    let mut s = two_clips();
    assert_eq!(s.trim(V1, 10, Side::Out, -4, true), Ok(-4));
    assert_eq!(spans(&s, V1), vec![(0, 6, 1, 100), (6, 10, 1, 500)]);
}

#[test]
fn a_plain_out_trim_cannot_overwrite_the_next_clip() {
    let mut s = two_clips();
    assert_eq!(s.trim(V1, 10, Side::Out, 5, false), Ok(0));
    assert_eq!(s.trim(V1, 10, Side::Out, -2, false), Ok(-2), "shortening leaves a gap");
    assert_eq!(spans(&s, V1), vec![(0, 8, 1, 100), (10, 10, 1, 500)]);
}

#[test]
fn a_ripple_in_trim_takes_the_head_and_closes_up() {
    let mut s = two_clips();
    assert_eq!(s.trim(V1, 10, Side::In, 4, true), Ok(4));
    assert_eq!(spans(&s, V1), vec![(0, 10, 1, 100), (10, 6, 1, 504)]);
}

#[test]
fn trims_stop_at_the_handles() {
    let mut s = seq();
    let t = Targets { video: Some(V1), audio: None };
    s.overwrite(Source::new(1, 2, 10), 0, t);
    assert_eq!(s.handles(&s.tracks[V1].clips[0]), (2, 988));
    assert_eq!(s.trim(V1, 0, Side::In, -5, true), Ok(-2));
    assert_eq!(s.tracks[V1].clips[0].src_in, 0);
}

#[test]
fn extend_moves_the_selected_side_to_the_playhead() {
    let mut s = two_clips();
    assert_eq!(s.extend(V1, 10, Side::Roll, 6, false), Ok(-4));
    assert_eq!(s.tracks[V1].clips[1].start, 6);
}

#[test]
fn slip_changes_the_source_not_the_place() {
    let mut s = two_clips();
    assert_eq!(s.slip(V1, 0, 7), 7);
    assert_eq!(spans(&s, V1)[0], (0, 10, 1, 107));
    assert_eq!(s.slip(V1, 0, -1000), -107, "clamped to the media");
}

#[test]
fn a_dissolve_mixes_both_clips_and_is_clamped_to_handles() {
    let mut s = two_clips();
    let d = s.add_dissolve(V1, 10, 8).unwrap();
    assert_eq!((d.before, d.after), (4, 4));
    match s.picture_at(10) {
        Picture::Mix(a, b, m) => {
            assert_eq!((a.frame, b.frame), (110, 500));
            assert!((m - 4.5 / 8.0).abs() < 1e-6);
        }
        p => panic!("expected a mix, got {p:?}"),
    }
    assert!(matches!(s.picture_at(5), Picture::One(_)));
    assert!(s.is_consistent());
    // No source before the second clip's in: only the half after the cut.
    let mut s = seq();
    let t = Targets { video: Some(V1), audio: None };
    s.overwrite(Source::new(1, 100, 10), 0, t);
    s.overwrite(Source::new(1, 0, 10), 10, t);
    let d = s.add_dissolve(V1, 10, 8).unwrap();
    assert_eq!((d.before, d.after), (0, 4));
}

#[test]
fn a_dissolve_goes_when_its_cut_does_and_rides_a_roll() {
    let mut s = two_clips();
    s.add_dissolve(V1, 10, 6).unwrap();
    s.trim(V1, 10, Side::Roll, 2, false).unwrap();
    assert_eq!(s.tracks[V1].dissolves[0].at, 12);
    s.lift(0, 20, Targets { video: Some(V1), audio: None });
    assert!(s.tracks[V1].dissolves.is_empty());
}

#[test]
fn a_sequence_round_trips_through_json_with_its_names() {
    let mut s = two_clips();
    s.add_dissolve(V1, 10, 6).unwrap();
    s.set_speed(V1, 0, 2.0);
    s.raster = Some((1920, 1080));
    let text = serde_json::to_string_pretty(&s).unwrap();
    let mut back: Sequence = serde_json::from_str(&text).unwrap();
    back.after_load().unwrap();
    assert_eq!(back, s);
    assert_eq!(back.tracks[A1].name, "A1");
}

#[test]
fn a_broken_document_is_refused_in_words() {
    let s = two_clips();
    let mut v: serde_json::Value = serde_json::to_value(&s).unwrap();
    v["tracks"][0]["clips"][0]["media"] = serde_json::json!(99);
    let mut back: Sequence = serde_json::from_value(v).unwrap();
    assert!(back.after_load().unwrap_err().contains("media"));
}

#[test]
fn reframe_fits_fills_zooms_and_moves() {
    let fit = Reframe::default().place((1920, 1080), (1080, 1920));
    assert_eq!(fit, (0.0, 960.0 - 303.75, 1080.0, 607.5));
    let fill = Reframe { fill: true, ..Default::default() }.place((1920, 1080), (1080, 1920));
    assert!((fill.3 - 1920.0).abs() < 1e-9 && fill.0 < 0.0, "covers the height, overflows the sides");
    let moved = Reframe { fill: true, x: 0.25, ..Default::default() }.place((1920, 1080), (1080, 1920));
    assert!((moved.0 - fill.0 - 270.0).abs() < 1e-9, "a quarter of the raster's width to the right");
    let zoomed = Reframe { zoom: 2.0, ..Default::default() }.place((1920, 1080), (1920, 1080));
    assert_eq!(zoomed, (-960.0, -540.0, 3840.0, 2160.0));
}

#[test]
fn a_reframe_rides_the_picture_and_the_linked_group_and_saves() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    let r = Reframe { fill: true, zoom: 1.2, x: -0.1, y: 0.05 };
    s.set_reframe(V1, 0, r);
    assert_eq!(s.tracks[A1].clips[0].reframe, r);
    match s.picture_at(3) {
        Picture::One(l) => assert_eq!(l.reframe, r),
        p => panic!("{p:?}"),
    }
    let back: Sequence = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.tracks[V1].clips[0].reframe, r);
}

#[test]
fn a_moved_group_overwrites_where_it_lands_and_its_sound_follows_to_a2() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    s.overwrite(Source::new(0, 100, 10), 10, Targets::default());
    // Move the first clip and its audio 15 frames later and up to V2.
    let g = s.group(V1, 0);
    assert_eq!(s.move_clips(&g, 15, 1, false), (15, 1));
    assert_eq!(spans(&s, V2), vec![(15, 10, 0, 0)]);
    assert_eq!(spans(&s, V1), vec![(10, 10, 0, 100)]);
    // The sound went to A2 with it; A1 keeps the second clip's sound.
    assert_eq!(spans(&s, A2), vec![(15, 10, 0, 0)]);
    assert_eq!(spans(&s, A1), vec![(10, 10, 0, 100)]);
    assert!(s.is_consistent());
}

#[test]
fn a_move_never_leaves_the_video_tracks_or_frame_0() {
    let mut s = seq();
    s.overwrite(Source::new(1, 0, 10), 5, Targets { video: Some(V2), audio: None });
    let g = s.group(V2, 0);
    assert_eq!(s.move_clips(&g, -50, 5, false), (-5, 1));
    assert_eq!(spans(&s, V3), vec![(0, 10, 1, 0)]);
}

#[test]
fn an_insert_move_ripples_every_track() {
    let mut s = seq();
    let t = Targets { video: Some(V1), audio: None };
    s.overwrite(Source::new(1, 0, 10), 0, t);
    s.overwrite(Source::new(1, 100, 10), 10, t);
    s.overwrite(Source::new(1, 200, 5), 30, t);
    // The last clip, insert-moved to frame 5: the first clip is cut, all after moves on.
    let g = s.group(V1, 2);
    s.move_clips(&g, -25, 0, true);
    assert_eq!(spans(&s, V1), vec![(0, 5, 1, 0), (5, 5, 1, 200), (10, 5, 1, 5), (15, 10, 1, 100)]);
    assert!(s.is_consistent());
}

#[test]
fn dissolve_sides_are_set_separately_and_clamped() {
    let mut s = two_clips();
    let d = s.set_dissolve_sides(V1, 10, 2, 6).unwrap();
    assert_eq!((d.before, d.after), (2, 6));
    let d = s.set_dissolve_sides(V1, 10, 50, 50).unwrap();
    assert_eq!((d.before, d.after), (10, 10), "clamped to the clips' lengths");
    assert!(s.set_dissolve_sides(V1, 10, 0, 0).is_err());
}

mod three_point {
    use super::*;

    fn r(src: Marks, sp: i64, rec: Marks, rp: i64) -> Option<Resolved> {
        resolve(src, sp, 1000, F24, rec, rp, F24)
    }
    fn m(i: Option<i64>, o: Option<i64>) -> Marks {
        Marks { inp: i, out: o }
    }

    #[test]
    fn source_in_and_out_at_the_record_playhead() {
        assert_eq!(r(m(Some(10), Some(19)), 0, m(None, None), 50), Some(Resolved { src_in: 10, rec_in: 50, len: 10 }));
    }

    #[test]
    fn record_in_and_out_set_the_length_from_the_source_in() {
        assert_eq!(r(m(Some(10), None), 0, m(Some(5), Some(9)), 0), Some(Resolved { src_in: 10, rec_in: 5, len: 5 }));
    }

    #[test]
    fn a_record_out_alone_back_times() {
        assert_eq!(r(m(Some(10), Some(19)), 0, m(None, Some(99)), 0), Some(Resolved { src_in: 10, rec_in: 90, len: 10 }));
    }

    #[test]
    fn a_source_out_alone_back_times_on_the_source() {
        assert_eq!(r(m(None, Some(49)), 0, m(Some(0), Some(9)), 0), Some(Resolved { src_in: 40, rec_in: 0, len: 10 }));
    }

    #[test]
    fn no_marks_takes_the_source_playhead_to_its_end() {
        assert_eq!(r(m(None, None), 990, m(None, None), 7), Some(Resolved { src_in: 990, rec_in: 7, len: 10 }));
    }

    #[test]
    fn back_timing_past_the_start_trims_the_head() {
        assert_eq!(r(m(Some(10), Some(19)), 0, m(None, Some(4)), 0), Some(Resolved { src_in: 15, rec_in: 0, len: 5 }));
    }

    #[test]
    fn four_marks_take_the_source_length() {
        assert_eq!(r(m(Some(0), Some(9)), 0, m(Some(100), Some(199)), 0), Some(Resolved { src_in: 0, rec_in: 100, len: 10 }));
    }

    #[test]
    fn an_edit_never_runs_past_the_source() {
        assert_eq!(r(m(Some(995), None), 0, m(Some(0), Some(99)), 0).unwrap().len, 5);
    }

    #[test]
    fn marks_keep_in_before_out() {
        let mut k = Marks::default();
        k.set_out(10);
        k.set_in(20);
        assert_eq!(k, m(Some(20), None));
    }
}

#[test]
fn sound_lands_on_the_video_targets_pair_and_music_on_m() {
    let mut s = seq();
    let mut t = Targets::default();
    t.set_video(Some(V2));
    assert_eq!(t.audio, Some(A2), "the sound follows the picture's track");
    s.overwrite(Source::new(0, 0, 10), 0, t);
    assert_eq!(spans(&s, V2), vec![(0, 10, 0, 0)]);
    assert_eq!(spans(&s, A2), vec![(0, 10, 0, 0)]);
    // On M the sound stays put when the picture target moves.
    t.toggle(M);
    t.set_video(Some(V3));
    assert_eq!(t.audio, Some(M));
    t.set_video(None);
    s.overwrite(Source::new(0, 100, 20), 20, t);
    assert_eq!(spans(&s, M), vec![(20, 20, 0, 100)]);
    assert!(s.tracks[V3].clips.is_empty());
    // Toggling the target off and back.
    t.toggle(M);
    assert_eq!(t.audio, None);
}

#[test]
fn a_moved_clip_takes_its_sound_to_the_paired_track() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    let g = s.group(V1, 0);
    let (_, dtr) = s.move_clips(&g, 5, 1, false);
    assert_eq!(dtr, 1);
    assert_eq!(spans(&s, V2), vec![(5, 10, 0, 0)]);
    assert_eq!(spans(&s, A2), vec![(5, 10, 0, 0)]);
    // Clamped at the top: V3 / A3.
    let g = s.group(V2, 0);
    let (_, dtr) = s.move_clips(&g, 0, 5, false);
    assert_eq!(dtr, 1);
    assert_eq!(spans(&s, A3), vec![(5, 10, 0, 0)]);
    // Sound alone goes down the audio tracks to M (rows move down = -).
    let (_, dtr) = s.move_clips(&[(A3, 0)], 0, -3, false);
    assert_eq!(dtr, -1);
    assert_eq!(spans(&s, M), vec![(5, 10, 0, 0)]);
    assert!(s.is_consistent());
}

#[test]
fn mute_and_solo_pick_what_the_mix_hears() {
    let mut s = seq();
    let heard = |s: &Sequence| s.audible().iter().map(|a| a.0).collect::<Vec<_>>();
    assert_eq!(heard(&s), vec![A1, A2, A3, M]);
    s.tracks[A2].mute = true;
    assert_eq!(heard(&s), vec![A1, A3, M]);
    s.tracks[M].solo = true;
    s.tracks[A2].solo = true;
    // Solo wins over mute.
    assert_eq!(heard(&s), vec![A2, M]);
    s.tracks[M].gain_db = -6.0;
    let g = s.audible().iter().find(|a| a.0 == M).unwrap().1;
    assert!((g - 0.501).abs() < 0.01);
}

#[test]
fn a_four_track_sequence_loads_with_a2_a3_and_m() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    let mut v = serde_json::to_value(&s).unwrap();
    v["tracks"].as_array_mut().unwrap().truncate(4);
    let mut old: Sequence = serde_json::from_value(v).unwrap();
    old.after_load().unwrap();
    assert_eq!(old.tracks.len(), 7);
    assert_eq!(old.tracks[M].name, "M");
    assert_eq!(spans(&old, A1), vec![(0, 10, 0, 0)]);
    // Old targets: a bool.
    let t: Targets = serde_json::from_str(r#"{"video":0,"audio":true}"#).unwrap();
    assert_eq!(t, Targets { video: Some(V1), audio: Some(A1) });
    let t: Targets = serde_json::from_str(r#"{"video":null,"audio":false}"#).unwrap();
    assert_eq!(t.audio, None);
}

#[test]
fn the_razor_cuts_picture_and_sound_together() {
    let mut s = seq();
    s.overwrite(Source::new(0, 100, 20), 0, Targets::default());
    assert!(s.razor(V1, 8));
    assert_eq!(spans(&s, V1), vec![(0, 8, 0, 100), (8, 12, 0, 108)]);
    assert_eq!(spans(&s, A1), vec![(0, 8, 0, 100), (8, 12, 0, 108)]);
    // On a cut or in a gap: nothing.
    assert!(!s.razor(V1, 8));
    assert!(!s.razor(V2, 5));
    // The halves are separate groups now.
    assert_eq!(s.group(V1, 0).len(), 2);
    s.overwrite(Source::new(1, 0, 20), 0, Targets { video: Some(V2), audio: None });
    assert_eq!(s.razor_tracks(&[V1, V2, A1], 4), 2, "V2 and V1 (A1 went with V1)");
    assert_eq!(spans(&s, V2).len(), 2);
    assert!(s.is_consistent());
}

#[test]
fn slide_moves_a_clip_and_its_neighbours_give_and_take() {
    let mut s = seq();
    for (src, at) in [(100, 0), (300, 10), (500, 20)] {
        s.overwrite(Source::new(1, src, 10), at, Targets { video: Some(V1), audio: None });
    }
    assert_eq!(s.slide(V1, 1, 3), 3);
    assert_eq!(spans(&s, V1), vec![(0, 13, 1, 100), (13, 10, 1, 300), (23, 7, 1, 503)]);
    assert_eq!(s.duration(), 30);
    // Clamped by the next clip's length (it keeps a frame).
    assert_eq!(s.slide(V1, 1, 50), 6);
    assert_eq!(spans(&s, V1)[2], (29, 1, 1, 509));
    // And back past where it began, limited by the first clip.
    assert_eq!(s.slide(V1, 1, -100), -18);
    assert_eq!(spans(&s, V1)[0], (0, 1, 1, 100));
    assert!(s.is_consistent());
}

#[test]
fn a_rate_stretch_keeps_the_source_and_changes_the_speed() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    s.overwrite(Source::new(0, 200, 10), 30, Targets::default());
    assert_eq!(s.stretch(V1, 0, Side::Out, 10), 20);
    let c = &s.tracks[V1].clips[0];
    assert_eq!((c.start, c.len, c.speed), (0, 20, 0.5));
    // The sound stretched with it, and the last frame is still source 9.
    assert_eq!(s.tracks[A1].clips[0].len, 20);
    assert_eq!(s.source_frame(c, 19), 9);
    // Stops at the next clip.
    assert_eq!(s.stretch(V1, 0, Side::Out, 100), 30);
    // From the head: the end stays.
    assert_eq!(s.stretch(V1, 1, Side::In, 5), 5);
    let c = &s.tracks[V1].clips[1];
    assert_eq!((c.start, c.len, c.speed), (35, 5, 2.0));
    assert!(s.is_consistent());
}

#[test]
fn duplicates_are_new_clips_with_a_new_link() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    let g = s.group(V1, 0);
    s.duplicate_clips(&g, 20, 1, false);
    assert_eq!(spans(&s, V1), vec![(0, 10, 0, 0)]);
    assert_eq!(spans(&s, V2), vec![(20, 10, 0, 0)]);
    assert_eq!(spans(&s, A2), vec![(20, 10, 0, 0)]);
    assert_ne!(s.tracks[V2].clips[0].link, s.tracks[V1].clips[0].link);
    assert_eq!(s.group(V2, 0).len(), 2);
}

#[test]
fn four_marks_insert_stretches_to_fit() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    s.insert_fit(1, 0, 20, 5, 10, Targets { video: Some(V1), audio: None });
    assert_eq!(s.tracks[V1].clips[1].speed, 2.0);
    assert_eq!(spans(&s, V1), vec![(0, 5, 0, 0), (5, 10, 1, 0), (15, 5, 0, 5)]);
    // Past the end: a gap is left.
    s.overwrite(Source::new(1, 0, 5), 40, Targets { video: Some(V1), audio: None });
    assert_eq!(s.duration(), 45);
    assert_eq!(s.tracks[V1].index_at(30), None);
}

#[test]
fn text_clips_draw_over_the_picture_below_and_share_a_style() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.ensure_styles();
    let sub = s.style_named("subtitle").unwrap();
    s.add_text(V2, 10, 20, "hello", sub).unwrap();
    s.add_text(V2, 30, 20, "world", sub).unwrap();
    // The footage still plays under the text.
    assert!(matches!(s.picture_at(15), Picture::One(l) if l.media == 0));
    let t = s.text_at(15);
    assert_eq!(t.len(), 1);
    assert_eq!((t[0].text.as_str(), t[0].opacity), ("hello", 1.0));
    assert!(s.text_at(5).is_empty());
    // One style for both: moving it moves both.
    s.styles[sub].y = 0.5;
    assert_eq!(s.text_at(40)[0].style.y, 0.5);
    assert_eq!(s.text_at(15)[0].style.y, 0.5);
    // Text BELOW a picture is covered.
    s.add_text(V1, 60, 10, "under", sub);
    s.overwrite(Source::new(0, 0, 100), 0, Targets { video: Some(V3), audio: None });
    assert!(s.text_at(65).is_empty());
    // Over black, a text clip alone draws on black.
    let mut b = seq();
    b.ensure_styles();
    b.add_text(V1, 0, 10, "alone", 0);
    assert!(matches!(b.picture_at(3), Picture::Black));
    assert_eq!(b.text_at(3).len(), 1);
    assert!(s.is_consistent());
}

#[test]
fn a_dissolve_between_text_clips_crossfades_them() {
    let mut s = seq();
    s.ensure_styles();
    s.add_text(V1, 0, 20, "a", 0);
    s.add_text(V1, 20, 20, "b", 0);
    // Text has endless handles: any length fits.
    let d = s.add_dissolve(V1, 20, 10).unwrap();
    assert_eq!(d.len(), 10);
    let mid = s.text_at(19);
    assert_eq!(mid.len(), 2);
    assert!((mid[0].opacity + mid[1].opacity - 1.0).abs() < 1e-6);
    assert!(mid[0].opacity > mid[1].opacity);
    assert!(matches!(s.picture_at(19), Picture::Black));
}

#[test]
fn text_clips_round_trip_and_old_documents_have_none() {
    let mut s = seq();
    s.ensure_styles();
    s.add_text(V2, 5, 10, "saved", 1);
    let back: Sequence = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.tracks[V2].clips[0].text.as_ref().unwrap().text, "saved");
    assert_eq!(back.styles.len(), 2);
    // A footage clip writes no text field at all.
    let mut f = seq();
    f.overwrite(Source::new(0, 0, 10), 0, Targets::default());
    assert!(!serde_json::to_string(&f).unwrap().contains("\"text\""));
}

#[test]
fn srt_parses_leniently_and_round_trips() {
    let src = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,500\r\nHello\r\nthere\r\n\r\n2\n00:00:03.000 --> 00:00:04,000 X1:0\nBye\n";
    let cues = parse_srt(src);
    assert_eq!(cues, vec![(1.0, 2.5, "Hello\nthere".to_string()), (3.0, 4.0, "Bye".to_string())]);
    assert_eq!(parse_srt(&format_srt(&cues)), cues);
}

#[test]
fn speed_about_a_point_keeps_that_point_still() {
    let mut s = seq();
    s.overwrite(Source::new(0, 100, 20), 10, Targets::default());
    // Playhead at 20 (source 110 there). Twice as fast: 10 frames long,
    // scaled about 20: 15..25, and frame 20 still shows source 110.
    let (st, len) = s.speed_about(V1, 0, 2.0, 20);
    assert_eq!((st, len), (15, 10));
    let c = &s.tracks[V1].clips[0];
    assert_eq!(s.source_frame(c, 20), 110);
    assert_eq!(s.tracks[A1].clips[0].start, 15);
    // Slower, against a neighbour: cut to the room there is.
    s.overwrite(Source::new(0, 500, 10), 30, Targets::default());
    let (st, len) = s.speed_about(V1, 0, 0.5, 15);
    assert_eq!(st, 15);
    assert_eq!(st + len, 30);
    assert!(s.is_consistent());
}

#[test]
fn a_wobble_clip_plays_its_active_render_with_the_same_source_frames() {
    let mut s = seq();
    s.overwrite(Source::new(0, 100, 20), 0, Targets { video: Some(V1), audio: None });
    let r1 = s.add_media(Media { path: "/w/v1/out.0080.exr".into(), name: "v1".into(), frames: 60, fps: F24, has_audio: false, width: 64, height: 36, generated: false, audio_only: false });
    s.tracks[V1].clips[0].wobble = Some(WobbleRef { uid: "u".into(), versions: vec![1, 2], active: 0, source: 0, first: 80, len: 60, renders: vec![WobbleRender { version: 1, media: r1, first: 80 }] });
    assert!(s.wobble_activate(V1, 0, 1));
    let c = &s.tracks[V1].clips[0];
    assert_eq!((c.media, c.src_in), (r1, 20));
    // Not rendered yet: the source plays.
    s.wobble_activate(V1, 0, 2);
    let c = &s.tracks[V1].clips[0];
    assert_eq!((c.media, c.src_in, c.wobble.as_ref().unwrap().active), (0, 100, 2));
    // Back and forth keeps the frames.
    s.wobble_activate(V1, 0, 1);
    s.wobble_activate(V1, 0, 1);
    assert_eq!(s.tracks[V1].clips[0].src_in, 20);
    let back: Sequence = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.tracks[V1].clips[0].wobble, s.tracks[V1].clips[0].wobble);
}

/// A 30 fps clip (media 2) in the 24 fps sequence.
fn seq_with_30() -> Sequence {
    let mut s = seq();
    s.add_media(Media { path: "/m30.mov".into(), name: "m30".into(), frames: 1000, fps: Fps::new(30, 1), has_audio: true, width: 64, height: 36, generated: false, audio_only: false });
    s
}

#[test]
fn conform_plays_every_source_frame_and_the_clip_grows() {
    let mut s = seq_with_30();
    // 60 record frames of 30 fps at real time cover 75 source frames,
    // skipping one in five.
    s.overwrite(Source::new(2, 100, 60), 0, Targets::default());
    let real: Vec<i64> = (0..10).map(|t| s.video_at(t).unwrap().1).collect();
    assert_ne!(real, (100..110).collect::<Vec<_>>(), "real time drops frames");
    let len = s.set_conform(V1, 0, true);
    assert_eq!(len, 75, "the same source, one frame each");
    let conf: Vec<i64> = (0..75).map(|t| s.video_at(t).unwrap().1).collect();
    assert_eq!(conf, (100..175).collect::<Vec<_>>(), "no drops, no repeats");
    assert!(s.tracks[A1].clips[0].conform && s.tracks[A1].clips[0].len == 75, "the linked audio follows");
    assert_eq!(s.tracks[V1].clips[0].realtime_speed(s.fps, Fps::new(30, 1)), 0.8);
    assert_eq!(s.set_conform(V1, 0, false), 60, "back to real time");
    assert_eq!(s.video_at(10).unwrap().1, real[0] + 12);
    // Speed on a conformed clip is against the conformed rate.
    s.set_conform(V1, 0, true);
    assert_eq!(s.set_speed(V1, 0, 2.0), 38);
    assert_eq!(s.video_at(1).unwrap().1, 102);
    assert!(s.tracks[V1].clips[0].conform, "speed keeps the conform");
    assert!(s.is_consistent());
}

#[test]
fn conform_is_a_no_op_at_the_same_rate() {
    let mut s = seq();
    s.overwrite(Source::new(0, 0, 48), 0, Targets::default());
    assert_eq!(s.set_conform(V1, 0, true), 48);
    assert_eq!(s.video_at(47).unwrap().1, 47);
    assert_eq!(s.tracks[V1].clips[0].realtime_speed(s.fps, F24), 1.0);
}

#[test]
fn a_conformed_sequence_fits_source_frames_one_to_one() {
    let mut s = seq_with_30();
    s.conform = true;
    let f = s.fit_source(2, 0, 50, 50).unwrap();
    assert!(f.conform && (f.speed - 1.0).abs() < 1e-12);
    s.overwrite(f, 0, Targets::default());
    assert_eq!((0..50).map(|t| s.video_at(t).unwrap().1).collect::<Vec<_>>(), (0..50).collect::<Vec<_>>());
}

#[test]
fn a_sound_file_lands_on_the_audio_track_alone() {
    let mut s = seq();
    let m = s.add_media(Media { path: "/song.mp3".into(), name: "song.mp3".into(), frames: 500, fps: F24, has_audio: true, width: 960, height: 540, generated: false, audio_only: true });
    s.overwrite(Source::new(0, 0, 100), 0, Targets::default());
    s.overwrite(Source::new(m, 0, 50), 20, Targets::default());
    // V1 keeps m0's picture right through; A1 takes the song over 20..70.
    assert_eq!(spans(&s, V1), vec![(0, 100, 0, 0)]);
    assert_eq!(spans(&s, A1), vec![(0, 20, 0, 0), (20, 50, m, 0), (70, 30, 0, 70)]);
    // Onto the music track with no video target at all.
    s.insert(Source::new(m, 100, 24), 0, Targets { video: None, audio: Some(M) });
    assert_eq!(spans(&s, M), vec![(0, 24, m, 100)]);
    assert!(s.tracks[V1].clips.iter().all(|c| c.media != m));
    assert!(s.is_consistent());
}
