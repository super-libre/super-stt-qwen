// SPDX-License-Identifier: GPL-3.0-only
//! 16 kHz samples to the log-mel spectrogram the audio tower reads.
//!
//! This is `transformers`' `WhisperFeatureExtractor` — the feature extractor
//! the checkpoints name in `preprocessor_config.json` and the reference ran —
//! computed the way its torch path computes it:
//!
//! - frames are centred on their hop: the signal is reflect-padded by half a
//!   window on both sides, a 400-sample periodic Hann window slides by 160, and
//!   the last frame is dropped, so `n` samples give `n / 160` frames;
//! - the power spectrum is `|X|²` over the 201 bins of the real FFT, projected
//!   onto 128 Slaney-normalized mel filters between 0 and 8 kHz;
//! - `log10` of that, floored at `1e-10` and then at eight below the loudest
//!   value of the whole clip, and mapped by `(x + 4) / 4`.
//!
//! The reference pads a batch to its longest clip, so a single clip is not
//! padded at all — unlike Whisper and Voxtral, which read fixed 30-second
//! windows. Whatever length arrives is the length encoded.
//!
//! Plain Rust on the host: a few milliseconds per second of audio, next to a
//! model that takes far longer.

use std::sync::Arc;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
/// Bins of the real FFT of one frame.
const N_FREQS: usize = N_FFT / 2 + 1;

/// A mel filter bank, stored sparsely: every triangular filter is zero outside
/// a short run of frequency bins, and skipping the zeros changes no sum, since
/// adding an exact zero is exact.
#[derive(Debug, Clone)]
pub struct MelFilters {
    /// Per filter, the first bin it covers and its weights from there on.
    filters: Vec<(usize, Vec<f32>)>,
}

impl MelFilters {
    /// The Slaney-scale, Slaney-normalized filter bank of `transformers`'
    /// `mel_filter_bank(1 + n_fft // 2, n_mels, 0, 8000, 16000, "slaney",
    /// "slaney")`, computed in f64 and narrowed to f32 as the extractor's
    /// torch path narrows it.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn slaney(n_mels: usize) -> Self {
        let mel_min = hertz_to_mel(0.0);
        let mel_max = hertz_to_mel(f64::from(SAMPLE_RATE) / 2.0);
        // `np.linspace(mel_min, mel_max, n_mels + 2)`, mapped back to hertz.
        let edges: Vec<f64> = (0..n_mels + 2)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
                mel_to_hertz(mel)
            })
            .collect();
        // `np.linspace(0, sampling_rate // 2, n_freqs)`.
        let nyquist = f64::from(SAMPLE_RATE / 2);
        let fft_freqs: Vec<f64> = (0..N_FREQS)
            .map(|i| nyquist * i as f64 / (N_FREQS - 1) as f64)
            .collect();
        let filters = (0..n_mels)
            .map(|m| {
                let (lower, center, upper) = (edges[m], edges[m + 1], edges[m + 2]);
                let enorm = 2.0 / (upper - lower);
                let weights: Vec<f32> = fft_freqs
                    .iter()
                    .map(|&f| {
                        let down = (f - lower) / (center - lower);
                        let up = (upper - f) / (upper - center);
                        (down.min(up).max(0.0) * enorm) as f32
                    })
                    .collect();
                let first = weights.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = weights.iter().rposition(|&w| w != 0.0).map_or(0, |l| l + 1);
                (first, weights[first..last.max(first)].to_vec())
            })
            .collect();
        Self { filters }
    }

    /// The number of mel bins.
    #[must_use]
    pub fn bins(&self) -> usize {
        self.filters.len()
    }
}

/// `hertz_to_mel(freq, mel_scale="slaney")`: linear below 1 kHz, logarithmic
/// above.
fn hertz_to_mel(freq: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * logstep
    } else {
        3.0 * freq / 200.0
    }
}

/// The inverse of [`hertz_to_mel`].
fn mel_to_hertz(mel: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// The frames `samples` samples give: one per hop, the last one dropped.
#[must_use]
pub fn frame_count(samples: usize) -> usize {
    samples / HOP_LENGTH
}

/// The fewest samples the centred framing can pad: reflection needs more than
/// half a window of signal to mirror.
pub const MIN_SAMPLES: usize = N_FFT / 2 + 1;

/// The log-mel spectrogram of `samples` at 16 kHz: `filters.bins()` rows of
/// [`frame_count`] values, row-major.
///
/// # Panics
/// If `samples` is shorter than [`MIN_SAMPLES`]; the caller pads a short clip
/// before it gets here.
#[must_use]
pub fn log_mel_spectrogram(samples: &[f32], filters: &MelFilters) -> Vec<f32> {
    assert!(
        samples.len() >= MIN_SAMPLES,
        "{} samples is too short to frame",
        samples.len()
    );
    let frames = frame_count(samples.len());
    let n_mels = filters.bins();

    // Reflect padding by half a window, `torch.stft(center=True)`'s default.
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| samples[i]));
    padded.extend_from_slice(samples);
    let n = samples.len();
    padded.extend((0..pad).map(|i| samples[n - 2 - i]));

    // `torch.hann_window(400)`: the periodic window.
    #[allow(clippy::cast_precision_loss)]
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos())
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);

    // Frames are independent, so they are split across threads in contiguous
    // runs. Which thread computes a frame does not change a bit of it.
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(8)
        .min(frames.div_ceil(256).max(1));
    let per_thread = frames.div_ceil(threads).max(1);
    let runs: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..frames)
            .step_by(per_thread)
            .map(|start| {
                let end = (start + per_thread).min(frames);
                let (padded, window, fft) = (&padded, &window, Arc::clone(&fft));
                s.spawn(move || {
                    let mut out = Vec::with_capacity((end - start) * n_mels);
                    let mut buffer = vec![Complex32::default(); N_FFT];
                    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
                    let mut power = vec![0f32; N_FREQS];
                    for frame in start..end {
                        let offset = frame * HOP_LENGTH;
                        for (j, slot) in buffer.iter_mut().enumerate() {
                            *slot = Complex32::new(window[j] * padded[offset + j], 0.0);
                        }
                        fft.process_with_scratch(&mut buffer, &mut scratch);
                        for (p, x) in power.iter_mut().zip(&buffer) {
                            *p = x.norm_sqr();
                        }
                        for (first, weights) in &filters.filters {
                            let sum: f32 = weights
                                .iter()
                                .zip(&power[*first..])
                                .map(|(w, p)| w * p)
                                .sum();
                            out.push(sum.max(1e-10).log10());
                        }
                    }
                    (start, out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a mel thread panicked"))
            .collect()
    });

    // Frame-major runs into the mel-major layout the encoder reads.
    let mut mel = vec![0f32; n_mels * frames];
    for (start, run) in runs {
        for (offset, frame) in run.chunks_exact(n_mels).enumerate() {
            for (m, &v) in frame.iter().enumerate() {
                mel[m * frames + start + offset] = v;
            }
        }
    }

    let floor = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for v in &mut mel {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
    mel
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Filters of the bank `transformers` builds, printed from
    /// `WhisperFeatureExtractor(feature_size=128).mel_filters` in float32:
    /// the two lowest, narrower than a bin, which land on bin 1 alone, one on
    /// the linear part of the scale and the highest, on the logarithmic part.
    #[test]
    fn the_filter_bank_matches_the_reference() {
        let bank = MelFilters::slaney(128);
        assert_eq!(bank.bins(), 128);
        let dense = |m: usize| {
            let (first, weights) = &bank.filters[m];
            let mut row = vec![0f32; N_FREQS];
            row[*first..first + weights.len()].copy_from_slice(weights);
            row
        };
        let nonzero = |row: &[f32]| -> Vec<usize> {
            row.iter()
                .enumerate()
                .filter(|(_, w)| **w != 0.0)
                .map(|(i, _)| i)
                .collect()
        };
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-7 * b.abs().max(1e-3);

        let row = dense(0);
        assert_eq!(nonzero(&row), vec![1]);
        assert!(close(row[1], 0.012_373_987), "{}", row[1]);

        let row = dense(1);
        assert_eq!(nonzero(&row), vec![1]);
        assert!(close(row[1], 0.030_392_565), "{}", row[1]);

        let row = dense(40);
        assert_eq!(nonzero(&row), vec![24]);
        assert!(close(row[24], 0.040_376_365), "{}", row[24]);

        let row = dense(127);
        assert_eq!(nonzero(&row), (191..=199).collect::<Vec<_>>());
        for (bin, expected) in [
            (191, 0.000_475_695_1),
            (192, 0.001_617_171_7),
            (193, 0.002_758_648_5),
        ] {
            assert!(close(row[bin], expected), "bin {bin}: {}", row[bin]);
        }
    }

    #[test]
    fn frames_are_one_per_hop_with_the_last_dropped() {
        assert_eq!(frame_count(176_000), 1100);
        assert_eq!(frame_count(16_000), 100);
        assert_eq!(frame_count(16_159), 100);
        assert_eq!(frame_count(16_160), 101);
        let mel = log_mel_spectrogram(&vec![0.01; 16_160], &MelFilters::slaney(128));
        assert_eq!(mel.len(), 128 * 101);
    }

    /// Silence is the floor everywhere: the loudest value is `log10(1e-10)`
    /// and the floor eight below it never binds.
    #[test]
    fn silence_maps_to_one_value() {
        let mel = log_mel_spectrogram(&vec![0.0; 4_000], &MelFilters::slaney(128));
        let expected = (-10.0 + 4.0) / 4.0;
        assert!(mel.iter().all(|&v| (v - expected).abs() < 1e-6));
    }

    /// The dynamic range is clamped to eight decades below the peak.
    #[test]
    fn the_range_is_eight_below_the_peak() {
        #[allow(clippy::cast_precision_loss)]
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mel = log_mel_spectrogram(&tone, &MelFilters::slaney(128));
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = mel.iter().copied().fold(f32::INFINITY, f32::min);
        assert!((max - min - 2.0).abs() < 1e-5, "range {}", max - min);
    }
}
