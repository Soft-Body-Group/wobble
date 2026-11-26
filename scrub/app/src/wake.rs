//! Waking the idle event loop. The loop sleeps (`ControlFlow::Wait`) until
//! winit has something for it, so on its own it never noticed an egui
//! repaint asked for from another thread (a worker finishing) or a repaint
//! asked for later (`request_repaint_after`: a notice fading, a progress
//! bar): those waited for the next mouse move. (Scripting calls wake the
//! loop through its proxy directly; see `main.rs`.)
//!
//! The main window's egui context reports every repaint request here
//! ([`Wake::install`]): the request's deadline is kept, and the loop is woken
//! through its proxy. Between events the loop asks [`Wake::due`] whether a
//! repaint is due now and when to wake next, and sleeps until then. Nothing
//! asked, nothing scheduled: an idle app stays asleep.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::event_loop::EventLoopProxy;

#[derive(Clone, Default)]
pub struct Wake {
    deadline: Arc<Mutex<Option<Instant>>>,
}

impl Wake {
    /// Route `ctx`'s repaint requests here, waking the loop through `proxy`.
    /// The callback runs under egui's own lock, possibly on any thread: it
    /// must not touch `ctx`, and it only notes a time and sends an event.
    pub fn install(&self, ctx: &egui::Context, proxy: EventLoopProxy<()>) {
        let me = self.clone();
        ctx.set_request_repaint_callback(move |info| {
            me.ask(Instant::now(), info.delay);
            let _ = proxy.send_event(());
        });
    }

    /// A repaint wanted `delay` after `now` (the earliest wish wins).
    pub fn ask(&self, now: Instant, delay: Duration) {
        // egui says "never" with a huge delay.
        let Some(at) = now.checked_add(delay) else { return };
        if let Ok(mut d) = self.deadline.lock() {
            *d = Some(d.map_or(at, |t| t.min(at)));
        }
    }

    /// At `now`: whether a repaint is due (it is then taken), and the next
    /// deadline to sleep until, if any.
    pub fn due(&self, now: Instant) -> (bool, Option<Instant>) {
        let Ok(mut d) = self.deadline.lock() else { return (false, None) };
        match *d {
            Some(t) if t <= now => {
                *d = None;
                (true, None)
            }
            other => (false, other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_earliest_wish_wins_and_is_taken_once() {
        let w = Wake::default();
        let t0 = Instant::now();
        assert_eq!(w.due(t0), (false, None), "nothing asked: sleep");
        w.ask(t0, Duration::from_millis(500));
        w.ask(t0, Duration::from_millis(100));
        w.ask(t0, Duration::from_millis(300));
        let at = t0 + Duration::from_millis(100);
        assert_eq!(w.due(t0), (false, Some(at)), "sleep until the earliest");
        assert_eq!(w.due(at), (true, None), "due");
        assert_eq!(w.due(at), (false, None), "taken");
        w.ask(t0, Duration::MAX);
        assert_eq!(w.due(t0), (false, None), "never is never");
        w.ask(t0, Duration::ZERO);
        assert_eq!(w.due(t0), (true, None), "now is now");
    }

    /// A repaint asked from another thread (a worker finishing) lands.
    #[test]
    fn a_request_from_another_thread_is_seen() {
        let w = Wake::default();
        let ctx = egui::Context::default();
        let me = w.clone();
        ctx.set_request_repaint_callback(move |info| me.ask(Instant::now(), info.delay));
        // Past the first frames egui wants anyway: idle.
        for _ in 0..3 {
            ctx.begin_pass(Default::default());
            let _ = ctx.end_pass();
        }
        w.due(Instant::now() + Duration::from_secs(3600));
        assert!(!w.due(Instant::now()).0, "idle");
        let c = ctx.clone();
        std::thread::spawn(move || c.request_repaint()).join().unwrap();
        assert!(w.due(Instant::now()).0);
    }
}
