//! Animation primitives: easing curves and time-based interpolation.
//!
//! The compositor free-runs at refresh rate (every vblank renders), so an
//! animation is just a function of time: record when it started, and each
//! frame read the monotonic clock to get progress in `[0, 1]`, shape it
//! through an easing [`Curve`], and interpolate. Nothing here schedules
//! repaints — the render loop already does that — and nothing holds GPU
//! resources; this module is pure maths so it's cheap and unit-testable.
//!
//! Time is `f64` seconds relative to the renderer's start instant (see
//! `render::Renderer::start`), which is the single clock all animations
//! share.

/// A timing curve mapping linear progress `x ∈ [0, 1]` to eased output.
///
/// Named curves match the CSS/`cubic-bezier` definitions so configs read
/// the way people expect; [`Curve::Bezier`] exposes the four control
/// points for full control (the two implicit endpoints are `(0,0)` and
/// `(1,1)`, as in CSS).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Curve {
    /// Constant-rate: output == input.
    Linear,
    /// Accelerate from rest (`cubic-bezier(0.42, 0, 1, 1)`).
    EaseIn,
    /// Decelerate to rest (`cubic-bezier(0, 0, 0.58, 1)`) — the snappy,
    /// "settles into place" feel; the default for most window motion.
    EaseOut,
    /// Accelerate then decelerate (`cubic-bezier(0.42, 0, 0.58, 1)`).
    EaseInOut,
    /// Arbitrary cubic Bézier control points `(x1, y1, x2, y2)`.
    Bezier(f64, f64, f64, f64),
}

impl Curve {
    /// Resolve to the cubic-Bézier control points the named curves stand
    /// for. `Linear` has no Bézier form (it's handled directly in
    /// [`Self::eval`]); we give it the identity points for completeness.
    fn control_points(self) -> (f64, f64, f64, f64) {
        match self {
            Self::Linear => (0.0, 0.0, 1.0, 1.0),
            Self::EaseIn => (0.42, 0.0, 1.0, 1.0),
            Self::EaseOut => (0.0, 0.0, 0.58, 1.0),
            Self::EaseInOut => (0.42, 0.0, 0.58, 1.0),
            Self::Bezier(x1, y1, x2, y2) => (x1, y1, x2, y2),
        }
    }

    /// Map linear progress `x` (clamped to `[0, 1]`) to the eased value.
    /// Endpoints are exact (`0 → 0`, `1 → 1`).
    pub fn eval(self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        if matches!(self, Self::Linear) {
            return x;
        }
        let (x1, y1, x2, y2) = self.control_points();
        let t = bezier_t_for_x(x, x1, x2);
        cubic_bezier(t, y1, y2)
    }
}

/// One axis of a cubic Bézier with implicit endpoints 0 and 1:
/// `B(t) = 3(1-t)²·t·p1 + 3(1-t)·t²·p2 + t³`.
fn cubic_bezier(t: f64, p1: f64, p2: f64) -> f64 {
    let u = 1.0 - t;
    3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t
}

/// Derivative of [`cubic_bezier`] w.r.t. `t` — for Newton's method.
fn cubic_bezier_dt(t: f64, p1: f64, p2: f64) -> f64 {
    let u = 1.0 - t;
    3.0 * u * u * p1 + 6.0 * u * t * (p2 - p1) + 3.0 * t * t * (1.0 - p2)
}

/// Solve `B_x(t) = x` for the Bézier parameter `t`, given the x control
/// points. Newton–Raphson with a bisection fallback — the same approach
/// browsers use for `cubic-bezier()`. `x1`/`x2` are the x coordinates of
/// the two control points; the curve must be monotonic in x for a unique
/// solution (CSS requires `x ∈ [0, 1]`, which we don't re-validate here).
fn bezier_t_for_x(x: f64, x1: f64, x2: f64) -> f64 {
    // Newton–Raphson: fast when the slope is well-behaved.
    let mut t = x;
    for _ in 0..8 {
        let err = cubic_bezier(t, x1, x2) - x;
        if err.abs() < 1e-6 {
            return t;
        }
        let d = cubic_bezier_dt(t, x1, x2);
        if d.abs() < 1e-6 {
            break; // flat slope — bisect instead
        }
        t -= err / d;
    }
    // Bisection fallback, guaranteed to converge on a monotonic curve.
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    t = x;
    for _ in 0..20 {
        let v = cubic_bezier(t, x1, x2);
        if (v - x).abs() < 1e-6 {
            break;
        }
        if v < x {
            lo = t;
        } else {
            hi = t;
        }
        t = lo.midpoint(hi);
    }
    t
}

/// A running animation: when it started, how long it lasts, and the curve
/// shaping it. All times are seconds on the shared renderer clock.
#[derive(Debug, Clone, Copy)]
pub struct Animation {
    start: f64,
    duration: f64,
    curve: Curve,
}

impl Animation {
    /// Start an animation at `now` lasting `duration` seconds. A
    /// non-positive duration yields an animation that is immediately
    /// [`done`](Self::done) (value pinned at `1.0`).
    pub fn start(now: f64, duration: f64, curve: Curve) -> Self {
        Self {
            start: now,
            duration: duration.max(0.0),
            curve,
        }
    }

    /// Linear progress in `[0, 1]` (`1.0` once finished or zero-length).
    pub fn progress(&self, now: f64) -> f64 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        ((now - self.start) / self.duration).clamp(0.0, 1.0)
    }

    /// Eased value in `[0, 1]` for `now`.
    pub fn value(&self, now: f64) -> f64 {
        self.curve.eval(self.progress(now))
    }

    /// Whether the animation has reached its end at `now`.
    pub fn done(&self, now: f64) -> bool {
        self.progress(now) >= 1.0
    }
}

/// Linear interpolation `a → b` by `t` (caller supplies eased `t`).
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_are_exact() {
        for c in [
            Curve::Linear,
            Curve::EaseIn,
            Curve::EaseOut,
            Curve::EaseInOut,
            Curve::Bezier(0.1, 0.7, 0.1, 1.0),
        ] {
            assert!((c.eval(0.0) - 0.0).abs() < 1e-6, "{c:?} at 0");
            assert!((c.eval(1.0) - 1.0).abs() < 1e-6, "{c:?} at 1");
        }
    }

    #[test]
    fn input_is_clamped() {
        assert_eq!(Curve::EaseOut.eval(-1.0), 0.0);
        assert_eq!(Curve::EaseOut.eval(2.0), 1.0);
    }

    #[test]
    fn linear_is_identity() {
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((Curve::Linear.eval(x) - x).abs() < 1e-9);
        }
    }

    #[test]
    fn ease_out_is_ahead_of_linear_in_the_middle() {
        // Decelerating curves cover ground early, so at the midpoint the
        // eased value sits above the linear diagonal.
        let v = Curve::EaseOut.eval(0.5);
        assert!(v > 0.5, "ease-out midpoint {v} should exceed 0.5");
    }

    #[test]
    fn ease_in_lags_linear_in_the_middle() {
        let v = Curve::EaseIn.eval(0.5);
        assert!(v < 0.5, "ease-in midpoint {v} should be below 0.5");
    }

    #[test]
    fn bezier_solver_round_trips() {
        // For a known monotonic curve, eval is monotonic increasing.
        let c = Curve::Bezier(0.25, 0.1, 0.25, 1.0);
        let mut prev = -1.0;
        for i in 0..=20 {
            let v = c.eval(f64::from(i) / 20.0);
            assert!(v >= prev - 1e-9, "not monotonic at {i}: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn animation_progress_and_done() {
        let a = Animation::start(10.0, 2.0, Curve::Linear);
        assert_eq!(a.progress(10.0), 0.0);
        assert_eq!(a.progress(11.0), 0.5);
        assert_eq!(a.progress(12.0), 1.0);
        assert_eq!(a.progress(99.0), 1.0);
        assert!(!a.done(11.0));
        assert!(a.done(12.0));
    }

    #[test]
    fn zero_duration_is_instantly_done() {
        let a = Animation::start(5.0, 0.0, Curve::EaseOut);
        assert_eq!(a.value(5.0), 1.0);
        assert!(a.done(5.0));
    }

    #[test]
    fn lerp_basic() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(100.0, 200.0, 0.0), 100.0);
        assert_eq!(lerp(100.0, 200.0, 1.0), 200.0);
    }
}
