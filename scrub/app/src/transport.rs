//! The playhead and how the keys move it: J / K / L shuttle (a repeated J
//! or L doubles the speed), space, frame steps, home / end, scrubbing.
//!
//! Playback runs on the wall clock: the playhead advances by real time ×
//! rate × speed whether or not frames keep up, and a frame that was due
//! but never shown is counted as DROPPED rather than stalling the clock.

pub const MAX_SPEED: f64 = 16.0;

pub struct Transport {
    /// The playhead as a fractional index (playback advances it smoothly).
    pub pos: f64,
    /// Signed shuttle speed; 0 is paused.
    pub speed: f64,
    pub frames: u64,
    pub fps: f64,
    pub looping: bool,
    /// Where the material ends when `frames` runs past it (the record side
    /// lets the playhead go past the last clip): forward playback that
    /// starts before it stops there, and End goes there.
    pub end: Option<u64>,
    /// The hand is dragging the playhead.
    pub scrubbing: bool,
    /// The frame the playhead wanted last redraw, and whether it was shown.
    due: Option<(u64, bool)>,
    pub dropped: u64,
    /// Playback started and the frames ahead are not in yet: the clock
    /// holds (pre-roll) until they are or `PREROLL_MAX` passes, so a start
    /// never drops its first frames.
    preroll: Option<std::time::Instant>,
}

/// Longest a start waits for its first frames.
pub const PREROLL_MAX: std::time::Duration = std::time::Duration::from_millis(500);
/// Frames ahead that must be ready before the clock runs.
pub const PREROLL_FRAMES: u64 = 6;

impl Transport {
    pub fn new(frames: u64, fps: f64) -> Self {
        Self { pos: 0.0, speed: 0.0, frames: frames.max(1), fps, looping: true, end: None, scrubbing: false, due: None, dropped: 0, preroll: None }
    }

    pub fn frame(&self) -> u64 {
        (self.pos.floor().max(0.0) as u64).min(self.frames - 1)
    }

    pub fn playing(&self) -> bool {
        self.speed != 0.0
    }

    pub fn dir(&self) -> i8 {
        if self.speed < 0.0 { -1 } else { 1 }
    }

    /// Frames the playhead moves between redraws, at least 1: what the
    /// prefetch should step by (a 60 Hz display at 4x skips frames).
    pub fn step(&self, display_hz: f64) -> u64 {
        ((self.speed.abs() * self.fps / display_hz).round() as u64).max(1)
    }

    pub fn shuttle_forward(&mut self) {
        self.start_if_stopped();
        self.speed = if self.speed > 0.0 { (self.speed * 2.0).min(MAX_SPEED) } else { 1.0 };
    }

    pub fn shuttle_back(&mut self) {
        self.start_if_stopped();
        self.speed = if self.speed < 0.0 { (self.speed * 2.0).max(-MAX_SPEED) } else { -1.0 };
    }

    fn start_if_stopped(&mut self) {
        if !self.playing() {
            self.preroll = Some(std::time::Instant::now());
            self.due = None;
        }
    }

    /// Set the speed (from stopped this pre-rolls).
    pub fn set_speed(&mut self, speed: f64) {
        self.start_if_stopped();
        self.speed = speed;
    }

    /// The frames playback will show first (for the pre-roll check).
    pub fn preroll_frames(&self, display_hz: f64) -> Option<Vec<i64>> {
        self.preroll?;
        let d = self.dir() as i64 * self.step(display_hz) as i64;
        Some((0..=PREROLL_FRAMES as i64).map(|k| self.frame() as i64 + k * d).collect())
    }

    pub fn prerolling(&self) -> bool {
        self.preroll.is_some()
    }

    /// The caller found the first frames ready (or not): end the pre-roll
    /// when they are, or when it has waited long enough.
    pub fn primed(&mut self, ready: bool) {
        if self.preroll.is_some_and(|t| ready || t.elapsed() >= PREROLL_MAX) {
            self.preroll = None;
        }
    }

    pub fn stop(&mut self) {
        self.preroll = None;
        self.speed = 0.0;
        self.pos = self.pos.floor();
    }

    pub fn toggle(&mut self) {
        if self.playing() {
            self.stop();
        } else {
            self.set_speed(1.0);
        }
    }

    /// The last frame End goes to.
    pub fn last(&self) -> u64 {
        self.end.unwrap_or(self.frames - 1).min(self.frames - 1)
    }

    pub fn step_by(&mut self, n: i64) {
        self.stop();
        self.seek((self.frame() as i64 + n).clamp(0, self.frames as i64 - 1) as u64);
    }

    pub fn seek(&mut self, f: u64) {
        self.pos = f.min(self.frames - 1) as f64;
        self.due = None;
    }

    /// Advance by `dt` seconds of wall time.
    pub fn tick(&mut self, dt: f64) {
        if !self.playing() || self.preroll.is_some() {
            return;
        }
        let n = self.frames as f64;
        let before = self.pos;
        self.pos += dt * self.fps * self.speed;
        if let Some(e) = self.end.filter(|_| !self.looping && self.speed > 0.0) {
            let e = e as f64;
            if before < e && self.pos >= e {
                self.pos = e;
                self.stop();
                return;
            }
        }
        if self.looping {
            self.pos = self.pos.rem_euclid(n);
        } else if self.pos < 0.0 || self.pos >= n {
            self.pos = self.pos.clamp(0.0, n - 1.0);
            self.stop();
        }
    }

    /// Record what this redraw showed for the frame that was due.
    pub fn account(&mut self, shown_exact: bool) {
        if self.preroll.is_some() {
            return;
        }
        let f = self.frame();
        match self.due {
            Some((d, seen)) if d != f => {
                if !seen && self.playing() {
                    self.dropped += 1;
                }
                self.due = Some((f, shown_exact));
            }
            Some((_, ref mut seen)) => *seen |= shown_exact,
            None => self.due = Some((f, shown_exact)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_l_doubles_and_j_reverses() {
        let mut t = Transport::new(100, 24.0);
        t.shuttle_forward();
        t.shuttle_forward();
        assert_eq!(t.speed, 2.0);
        t.shuttle_back();
        assert_eq!(t.speed, -1.0);
        for _ in 0..10 {
            t.shuttle_back();
        }
        assert_eq!(t.speed, -MAX_SPEED);
    }

    #[test]
    fn playback_runs_on_the_clock_and_loops() {
        let mut t = Transport::new(48, 24.0);
        t.toggle();
        t.primed(true);
        t.tick(0.5);
        assert_eq!(t.frame(), 12);
        t.tick(1.75);
        assert_eq!(t.frame(), 6, "wrapped past the end");
    }

    #[test]
    fn a_due_frame_never_shown_counts_as_dropped() {
        let mut t = Transport::new(48, 24.0);
        t.toggle();
        t.primed(true);
        t.account(false);
        t.tick(1.0 / 24.0);
        t.account(true);
        t.tick(1.0 / 24.0);
        t.account(false);
        assert_eq!(t.dropped, 1);
    }

    #[test]
    fn a_start_holds_the_clock_until_primed() {
        let mut t = Transport::new(48, 24.0);
        t.toggle();
        t.tick(0.5);
        assert_eq!(t.frame(), 0, "pre-rolling");
        t.primed(true);
        t.tick(0.5);
        assert_eq!(t.frame(), 12);
    }

    #[test]
    fn steps_stop_playback_and_clamp() {
        let mut t = Transport::new(10, 24.0);
        t.toggle();
        t.step_by(-5);
        assert!(!t.playing());
        assert_eq!(t.frame(), 0);
    }

    #[test]
    fn playback_stops_at_the_end_but_the_playhead_may_go_past_it() {
        let mut t = Transport::new(1000, 24.0);
        t.looping = false;
        t.end = Some(48);
        t.seek(40);
        t.set_speed(1.0);
        t.primed(true);
        t.tick(1.0);
        assert_eq!(t.frame(), 48);
        assert!(!t.playing());
        // Stepping and seeking go past it; End comes back to it.
        t.step_by(10);
        assert_eq!(t.frame(), 58);
        t.seek(500);
        assert_eq!(t.frame(), 500);
        assert_eq!(t.last(), 48);
    }
}
