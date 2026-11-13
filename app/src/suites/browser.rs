//! `--test-browser`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-browser driver: open at data/mix, select the first sequence
    /// row, and wait for its thumbnail to decode into a texture.
    pub(crate) fn drive_test_browser(&mut self) {
        let Some((mut frames, mut selected)) = self.test_browser else { return };
        frames += 1;
        if frames > 1800 {
            println!("BROWSER TEST FAIL: timeout (selected={selected})");
            std::process::exit(1);
        }
        if frames == 1 {
            self.browser = Some(browser::SeqBrowser::new(
                Some(0),
                std::path::PathBuf::from("data/mix"),
                self.decoder.clone(),
            ));
        } else if let Some(b) = &mut self.browser {
            let (n, thumb) = b.test_status();
            if thumb {
                // Columns: the sort and the duration floor act on exactly
                // the rows the listing draws.
                b.test_set_sort(browser::SortKey::Name, false);
                let by_name = b.test_visible();
                b.test_set_sort(browser::SortKey::Duration, true);
                let by_dur = b.test_visible();
                b.test_set_sort(browser::SortKey::Date, true);
                let by_date = b.test_visible();
                let longest = by_dur.first().map(|r| r.1).unwrap_or(0);
                b.test_set_min_frames(longest + 1);
                let above = b.test_visible().len();
                b.test_set_min_frames(longest);
                let at = b.test_visible().len();
                b.test_set_min_frames(0);
                b.test_set_sort(browser::SortKey::Name, false);
                let ok = !by_name.is_empty()
                    && by_name.windows(2).all(|w| browser::natural_cmp(&w[0].0, &w[1].0).is_le())
                    && by_dur.windows(2).all(|w| w[0].1 >= w[1].1)
                    && by_date.len() == by_name.len()
                    && above == 0
                    && at >= 1;
                println!(
                    "BROWSER TEST {}: {n} sequence rows scanned, thumbnail decoded, \
                     {} visible sorted by name/date/duration, duration floor keeps \
                     {at} at ≥{longest}f and {above} above it",
                    if ok { "PASS" } else { "FAIL" },
                    by_name.len(),
                );
                std::process::exit(if ok { 0 } else { 1 });
            }
            if n > 0 && !selected {
                b.test_select(0);
                selected = true;
            }
        } else if frames > 2 {
            println!("BROWSER TEST FAIL: browser closed unexpectedly");
            std::process::exit(1);
        }
        self.test_browser = Some((frames, selected));
    }
}
