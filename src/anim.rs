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
    /// A damped harmonic oscillator — the physical spring a bezier can only
    /// imitate. Overshoot falls out of the maths instead of being dialled in
    /// by hand, and undershoot/settling look right because they *are* right.
    ///
    /// Only the damping ratio `ζ = damping / 2√(stiffness · mass)` affects
    /// the shape: the response is run over its own settling time so the
    /// animation's `duration` still says how long it takes. Two springs with
    /// the same ζ are the same curve. That is why `mass` exists at all —
    /// it moves ζ, and it lets numbers be copied verbatim from configs that
    /// specify all three.
    Spring {
        mass: f64,
        stiffness: f64,
        damping: f64,
    },
}

/// Control points of the straight line from `(0,0)` to `(1,1)`.
const IDENTITY_POINTS: (f64, f64, f64, f64) = (0.0, 0.0, 1.0, 1.0);

impl Curve {
    /// Resolve to the cubic-Bézier control points the named curves stand
    /// for. `Linear` has no Bézier form (it's handled directly in
    /// [`Self::eval`]); we give it the identity points for completeness.
    fn control_points(self) -> (f64, f64, f64, f64) {
        match self {
            // Neither of these has a Bézier form, and `eval` handles both
            // before it reaches here — the identity keeps the match total
            // for any future caller.
            Self::Linear | Self::Spring { .. } => IDENTITY_POINTS,
            Self::EaseIn => (0.42, 0.0, 1.0, 1.0),
            Self::EaseOut => (0.0, 0.0, 0.58, 1.0),
            Self::EaseInOut => (0.42, 0.0, 0.58, 1.0),
            Self::Bezier(x1, y1, x2, y2) => (x1, y1, x2, y2),
        }
    }

    /// Map linear progress `x` (clamped to `[0, 1]`) to the eased value.
    /// Endpoints are exact (`0 → 0`, `1 → 1`).
    ///
    /// The *output* is not confined to `[0, 1]`: a [`Curve::Bezier`] whose
    /// `y` control points sit outside it overshoots and settles back, which
    /// is how a window springs slightly past its target. Only the `x` points
    /// are constrained (the config rejects others), because the curve must
    /// stay monotonic in `x` to be solvable.
    ///
    /// Callers interpolating geometry want that overshoot. Callers driving an
    /// *opacity* must clamp — see [`Animation::value`].
    pub fn eval(self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        if matches!(self, Self::Linear) {
            return x;
        }
        if let Self::Spring {
            mass,
            stiffness,
            damping,
        } = self
        {
            return spring_eval(x, mass, stiffness, damping);
        }
        let (x1, y1, x2, y2) = self.control_points();
        let t = bezier_t_for_x(x, x1, x2);
        cubic_bezier(t, y1, y2)
    }
}

/// Step response of a damped spring, normalised so `x ∈ [0, 1]` spans the
/// time it takes to settle and `x = 1` lands exactly on the target.
///
/// The three regimes are the standard closed forms, in `τ = ω₀·t`:
/// underdamped rings, critically damped is the fastest approach without
/// ringing, overdamped crawls in. Only ζ selects between them.
fn spring_eval(x: f64, mass: f64, stiffness: f64, damping: f64) -> f64 {
    // A zero or negative mass/stiffness is a config that means nothing
    // physical; clamp rather than produce NaN and paint garbage.
    let m = mass.max(1e-6);
    let k = stiffness.max(1e-6);
    let zeta = (damping.max(0.0) / (2.0 * (k * m).sqrt())).min(SPRING_MAX_ZETA);

    // How far to run it, in τ. The envelope decays as e^(-ζτ) when
    // underdamped and as e^(-rτ) with the slower root when overdamped, so
    // running until it is within `SPRING_SETTLE_TOL` covers both.
    //
    // The tolerance is tight on purpose. The textbook 2% criterion puts the
    // end of the run *exactly on the first overshoot peak* for a typical
    // ζ≈0.78 spring — and the endpoint correction below then cancels the
    // overshoot completely, turning a spring into an ease. Settling to 0.1%
    // instead leaves the peak at ~60% of the run, with the ring-down after
    // it visible, which is the whole reason to use a spring.
    let rate = if zeta < 1.0 {
        zeta.max(SPRING_MIN_ZETA)
    } else {
        zeta - (zeta * zeta - 1.0).sqrt()
    };
    let settle = SPRING_SETTLE_TOL / rate.max(SPRING_MIN_ZETA);

    let at = |tau: f64| -> f64 {
        if (zeta - 1.0).abs() < 1e-6 {
            // Critically damped: y = 1 - e^(-τ)(1 + τ).
            return 1.0 - (-tau).exp() * (1.0 + tau);
        }
        if zeta < 1.0 {
            // Underdamped: decaying oscillation about the target.
            let wd = (1.0 - zeta * zeta).sqrt();
            let env = (-zeta * tau).exp();
            return 1.0 - env * ((wd * tau).cos() + (zeta / wd) * (wd * tau).sin());
        }
        // Overdamped: two real roots, no overshoot at all.
        let s = (zeta * zeta - 1.0).sqrt();
        let (r1, r2) = (-zeta + s, -zeta - s);
        1.0 - (r2 * (r1 * tau).exp() - r1 * (r2 * tau).exp()) / (r2 - r1)
    };

    // Whatever the envelope still has left at `settle` is spread linearly
    // across the run, so the endpoints are exact (0 → 0, 1 → 1) without
    // rescaling — which would have flattened the overshoot that is the
    // entire reason to use a spring.
    let residual = 1.0 - at(settle);
    at(x * settle) + residual * x
}

/// Time-constants to run a spring for, `ln(1/tol)` with `tol = 0.1%`. See
/// the note in [`spring_eval`] for why this is not the textbook 2%.
const SPRING_SETTLE_TOL: f64 = 6.907_755;

/// Past this the spring is so overdamped it is indistinguishable from a slow
/// ease, and the arithmetic starts losing precision.
const SPRING_MAX_ZETA: f64 = 20.0;
/// Below this the spring would ring for far longer than any sane animation,
/// so the settling time is capped instead of running to infinity.
const SPRING_MIN_ZETA: f64 = 0.05;

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

    /// Eased value for `now`.
    ///
    /// Usually in `[0, 1]`, but an overshoot curve deliberately exceeds it
    /// near the end (see [`Curve::eval`]) — which is the point for position
    /// and size, and out of range for anything feeding an alpha channel.
    /// Use [`Self::alpha`] for those.
    pub fn value(&self, now: f64) -> f64 {
        self.curve.eval(self.progress(now))
    }

    /// [`Self::value`] clamped to `[0, 1]` and narrowed to `f32`, for use as
    /// an opacity.
    ///
    /// An overshoot curve would otherwise hand the renderer an alpha above
    /// `1.0` (a fade-in) or below `0.0` (a fade-out), which blends as a
    /// bright flash rather than as the spring it was meant to be.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "clamped to [0,1] first; f32 is exact enough for an opacity"
    )]
    pub fn alpha(&self, now: f64) -> f32 {
        self.value(now).clamp(0.0, 1.0) as f32
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
// These tests assert exact arithmetic identities — a clamp returning
// *exactly* 0.0, a linear curve at its endpoints, `lerp` at t=0 and t=1.
// Every value involved is exactly representable, and comparing with an
// epsilon instead would weaken the tests to the point of uselessness: a
// clamp that came back 1e-9 short is precisely the bug they exist to
// catch, and a tolerance would wave it through.
#[allow(clippy::float_cmp, reason = "the exactness is the assertion")]
mod tests {
    use super::*;

    /// Hyprland's `easy` spring, whose numbers people copy verbatim. The
    /// textbook step response for its damping ratio (0.7845) overshoots by
    /// 1.88%; ours must agree, or "copy the numbers from Hyprland" quietly
    /// produces a different feel.
    #[test]
    fn the_reference_spring_overshoots_by_the_textbook_amount() {
        let c = Curve::Spring {
            mass: 1.0,
            stiffness: 238.119_1,
            damping: 24.212_793_33,
        };
        let peak = (0..=2000)
            .map(|i| c.eval(f64::from(i) / 2000.0))
            .fold(f64::MIN, f64::max);
        assert!(
            (peak - 1.0188).abs() < 0.004,
            "expected ~1.88% overshoot, got {:.2}%",
            (peak - 1.0) * 100.0
        );
    }

    /// Every regime must start at 0, end exactly at 1, and stay finite.
    /// The endpoint especially: a window that settles at 0.98 of its target
    /// is a window in the wrong place.
    #[test]
    fn springs_are_well_behaved_in_all_three_regimes() {
        let spring = |damping| Curve::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping,
        };
        // zeta = damping / (2*sqrt(100)) = damping/20
        for (name, c) in [
            ("underdamped", spring(4.0)),  // zeta 0.2
            ("critical", spring(20.0)),    // zeta 1.0
            ("overdamped", spring(60.0)),  // zeta 3.0
            ("undamped-ish", spring(0.0)), // zeta 0, clamped internally
            ("absurdly overdamped", spring(100_000.0)),
        ] {
            assert!((c.eval(0.0)).abs() < 1e-9, "{name}: must start at 0");
            assert!((c.eval(1.0) - 1.0).abs() < 1e-9, "{name}: must land on 1");
            for i in 0..=200 {
                let v = c.eval(f64::from(i) / 200.0);
                assert!(v.is_finite(), "{name}: non-finite at {i}");
                assert!((-1.0..=3.0).contains(&v), "{name}: wild value {v} at {i}");
            }
        }
    }

    /// Overdamped springs approach without ever passing the target — that
    /// is the defining property, and the thing you pick one for.
    #[test]
    fn an_overdamped_spring_never_overshoots() {
        let c = Curve::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping: 60.0,
        };
        for i in 0..=1000 {
            let v = c.eval(f64::from(i) / 1000.0);
            assert!(v <= 1.0 + 1e-9, "overshot to {v}");
        }
    }

    /// The Hyprland-style springy feel is a bezier whose final y control
    /// point sits above 1: the value rises past the target mid-flight and
    /// settles back exactly onto it. Both halves matter — overshooting and
    /// *landing*.
    #[test]
    fn an_overshoot_bezier_exceeds_one_then_lands() {
        let c = Curve::Bezier(0.05, 0.9, 0.1, 1.05);
        let peak = (0..=100)
            .map(|i| c.eval(f64::from(i) / 100.0))
            .fold(f64::MIN, f64::max);
        assert!(peak > 1.0, "expected overshoot, peaked at {peak}");
        assert!((c.eval(1.0) - 1.0).abs() < 1e-6, "must settle exactly on 1");
    }

    /// …and that overshoot must never reach an alpha channel.
    #[test]
    fn alpha_is_clamped_even_for_overshoot_curves() {
        let a = Animation::start(0.0, 1.0, Curve::Bezier(0.05, 0.9, 0.1, 1.4));
        for i in 0..=100 {
            let t = f64::from(i) / 100.0;
            let alpha = a.alpha(t);
            assert!((0.0..=1.0).contains(&alpha), "alpha {alpha} at t={t}");
        }
    }

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
