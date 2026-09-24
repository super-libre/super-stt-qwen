// SPDX-License-Identifier: GPL-3.0-only
//! What a load reports while it runs: the `phase`, `step` and `progress`
//! fields of `GET /v1/status` while `state` is `loading`.
//!
//! The daemon fails a load whose `step` and `progress` both stand still for
//! 120 seconds once it has reported progress at all, so every step here is
//! measured by something that moves while the work does: the bytes of the
//! checkpoint read so far; on a first load, the entries `CubeCL` writes to its
//! kernel cache as it compiles and tunes; on a later one, the warm-up runs
//! done.
//!
//! Adapted from the Voxtral backend's module of the same name.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

/// How often a running load samples its progress. The daemon polls the
/// status every 500 ms.
const SAMPLE_EVERY: Duration = Duration::from_millis(500);

/// The load as a whole: the title the app gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The first load of this model with this build: its kernels are compiled
    /// and tuned, which takes minutes.
    InitialSetup,
    /// Every load after that.
    Loading,
}

impl Phase {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::InitialSetup => "initial_setup",
            Phase::Loading => "loading",
        }
    }
}

/// What the load is doing now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    LoadingWeights,
    /// The warm-up of an initial setup, which compiles and tunes every kernel.
    BuildingKernels,
    /// The warm-up of a later load, which finds them cached.
    WarmingUp,
}

impl Step {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Step::LoadingWeights => "loading_weights",
            Step::BuildingKernels => "building_kernels",
            Step::WarmingUp => "warming_up",
        }
    }
}

/// One sample of a load's progress.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Report {
    pub phase: Option<Phase>,
    pub step: Option<Step>,
    /// The fraction of `step` done, below 1 until the step ends.
    pub progress: Option<f32>,
}

/// How a step's progress is measured.
#[derive(Clone, Debug)]
pub enum Measure {
    /// `done` of a known `total`: bytes of the checkpoint, or warm-up runs.
    /// The counter is shared with whatever does the work.
    Count { done: Arc<AtomicU64>, total: u64 },
    /// Entries written to the kernel cache since the step began, against how
    /// many a first load's warm-up is expected to write.
    CacheEntries { start: u64, expected: u64 },
}

impl Measure {
    /// The entries a warm-up is about to write, counted from now.
    #[must_use]
    pub fn cache_entries(expected: u64) -> Self {
        Measure::CacheEntries {
            start: cache_entries(),
            expected,
        }
    }

    fn fraction(&self) -> f32 {
        match self {
            Measure::Count { done, total } => {
                #[allow(clippy::cast_precision_loss)]
                let fraction = done.load(Ordering::Relaxed) as f64 / (*total).max(1) as f64;
                #[allow(clippy::cast_possible_truncation)]
                let fraction = fraction.min(0.999) as f32;
                fraction
            }
            Measure::CacheEntries { start, expected } => {
                estimate(cache_entries().saturating_sub(*start), *expected)
            }
        }
    }
}

/// Entries in `CubeCL`'s kernel cache: compiled kernels and tuning results,
/// both of which a first load writes as it goes. The throughput probes are
/// left out; they are written once, before any kernel.
pub fn cache_entries() -> u64 {
    burn::cubecl::environment::namespaces()
        .iter()
        .filter(|n| !n.namespace.starts_with("throughput/"))
        .map(|n| n.entries)
        .sum()
}

/// `done` of about `expected`, as a fraction that stays below 1 and keeps
/// moving however far the estimate is off.
///
/// Linear up to 90% of `expected`; from there it closes on 1 without reaching
/// it, reading 95% at `expected` itself. A card that writes more entries than
/// estimated slows the bar near its end rather than pinning it, which the
/// daemon would take for a stall.
fn estimate(done: u64, expected: u64) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let (done, expected) = (done as f64, expected.max(1) as f64);
    let knee = 0.9 * expected;
    let fraction = if done <= knee {
        done / expected
    } else {
        let tail = 0.1 * expected;
        1.0 - 0.1 * tail / (done - knee + tail)
    };
    #[allow(clippy::cast_possible_truncation)]
    let fraction = fraction as f32;
    fraction
}

/// Samples a load's progress into `sink` every [`SAMPLE_EVERY`] until
/// [`Tracker::finish`].
pub struct Tracker<'a> {
    sink: &'a (dyn Fn(Report) + Sync),
    phase: Phase,
    current: Mutex<Option<(Step, Measure)>>,
    done: Mutex<bool>,
    finished: Condvar,
}

impl<'a> Tracker<'a> {
    pub fn new(phase: Phase, sink: &'a (dyn Fn(Report) + Sync)) -> Self {
        sink(Report {
            phase: Some(phase),
            ..Report::default()
        });
        Self {
            sink,
            phase,
            current: Mutex::new(None),
            done: Mutex::new(false),
            finished: Condvar::new(),
        }
    }

    /// Begin `step`, reporting it at once.
    pub fn enter(&self, step: Step, measure: Measure) {
        *self.current.lock().unwrap_or_else(PoisonError::into_inner) = Some((step, measure));
        self.sample();
    }

    fn sample(&self) {
        let current = self
            .current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let (step, progress) = match current {
            Some((step, measure)) => (Some(step), Some(measure.fraction())),
            None => (None, None),
        };
        (self.sink)(Report {
            phase: Some(self.phase),
            step,
            progress,
        });
    }

    /// Sample until [`Self::finish`]. Run on a thread of its own, beside the
    /// load it watches.
    pub fn run(&self) {
        let mut done = self.done.lock().unwrap_or_else(PoisonError::into_inner);
        while !*done {
            (done, _) = self
                .finished
                .wait_timeout(done, SAMPLE_EVERY)
                .unwrap_or_else(PoisonError::into_inner);
            if !*done {
                drop(done);
                self.sample();
                done = self.done.lock().unwrap_or_else(PoisonError::into_inner);
            }
        }
    }

    /// Stop [`Self::run`], at once rather than at its next sample.
    pub fn finish(&self) {
        *self.done.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.finished.notify_all();
    }
}

/// Stops a [`Tracker`] when dropped, so a load that panics or fails still
/// ends its sampler.
pub struct Finish<'a, 'b>(pub &'a Tracker<'b>);

impl Drop for Finish<'_, '_> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_estimate_is_linear_then_closes_on_one() {
        assert!(estimate(0, 100).abs() < f32::EPSILON);
        assert!((estimate(45, 100) - 0.45).abs() < 1e-6);
        assert!((estimate(90, 100) - 0.9).abs() < 1e-6);
        assert!((estimate(100, 100) - 0.95).abs() < 1e-6);
        let mut last = 0.0;
        // Out to ten times the estimate, where each entry still moves the
        // fraction by more than an f32 can resolve.
        for done in 0..1_000 {
            let fraction = estimate(done, 100);
            assert!(
                fraction > last || done == 0,
                "{done}: {fraction} after {last}"
            );
            assert!(fraction < 1.0, "{done}: {fraction}");
            last = fraction;
        }
    }

    #[test]
    fn a_count_stops_short_of_the_end_until_the_step_does() {
        let done = Arc::new(AtomicU64::new(50));
        let measure = Measure::Count {
            done: Arc::clone(&done),
            total: 100,
        };
        assert!((measure.fraction() - 0.5).abs() < 1e-6);
        done.store(100, Ordering::Relaxed);
        assert!(measure.fraction() < 1.0);
    }

    #[test]
    fn a_tracker_reports_each_step_as_it_begins() {
        let reports = Mutex::new(Vec::new());
        let sink = |r: Report| reports.lock().unwrap().push(r);
        let tracker = Tracker::new(Phase::InitialSetup, &sink);
        let done = Arc::new(AtomicU64::new(25));
        tracker.enter(
            Step::LoadingWeights,
            Measure::Count {
                done: Arc::clone(&done),
                total: 100,
            },
        );
        std::thread::scope(|scope| {
            scope.spawn(|| tracker.run());
            done.store(75, Ordering::Relaxed);
            std::thread::sleep(SAMPLE_EVERY * 2);
            let _finish = Finish(&tracker);
        });
        let reports = reports.into_inner().unwrap();
        assert_eq!(reports[0].phase, Some(Phase::InitialSetup));
        assert_eq!(reports[0].step, None);
        assert_eq!(reports[1].step, Some(Step::LoadingWeights));
        assert_eq!(reports[1].progress, Some(0.25));
        assert_eq!(reports.last().unwrap().progress, Some(0.75));
    }
}
