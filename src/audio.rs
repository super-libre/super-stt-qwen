// SPDX-License-Identifier: GPL-3.0-only
//! What happens to a request's samples before the model sees them.
//!
//! The same three steps the reference applies in
//! `qwen_asr.inference.utils.normalize_audio_input` and
//! `split_audio_into_chunks`, in the same order:
//!
//! 1. resample to 16 kHz, the only rate the feature extractor reads;
//! 2. scale a clip whose peak exceeds full scale back into range, and clip;
//! 3. cut a recording longer than [`MAX_CHUNK_SECONDS`] into pieces at its
//!    quietest points, each transcribed on its own and the texts joined.
//!
//! The daemon sends 16 kHz already, so the first is a guard rather than a
//! path; it is here because the contract lets a request name another rate.

// A single `start..end` range in a list of them.
#![allow(clippy::single_range_in_vec_init)]

use crate::qwen3::audio::{MIN_SAMPLES, SAMPLE_RATE};

/// The longest piece of audio transcribed in one prompt: twenty minutes, the
/// reference's `MAX_ASR_INPUT_SECONDS`.
pub const MAX_CHUNK_SECONDS: f64 = 1200.0;

/// The shortest piece transcribed: half a second, the reference's
/// `MIN_ASR_INPUT_SECONDS`, to which it pads the pieces a split leaves short.
///
/// Also applied to a whole clip that short, which the reference passes
/// through as it is. It cannot always: under a hundredth of a second there is
/// not a single mel frame to compute, and the reference fails the request.
/// Trailing silence is what a short clip is padded with either way, so this
/// transcribes it instead.
pub const MIN_CHUNK_SECONDS: f64 = 0.5;

/// Resample `samples` from `rate` to 16 kHz and bring it into `[-1, 1]`.
#[must_use]
pub fn normalize(samples: Vec<f32>, rate: u32) -> Vec<f32> {
    let mut samples = if rate == SAMPLE_RATE {
        samples
    } else {
        resample(&samples, rate, SAMPLE_RATE)
    };
    // `float_range_normalize`: a clip decoded from integers or mixed hot can
    // peak above 1, and the whole of it is scaled down rather than clipped.
    let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    if peak > 1.0 {
        for s in &mut samples {
            *s /= peak;
        }
    }
    for s in &mut samples {
        *s = s.clamp(-1.0, 1.0);
    }
    samples
}

/// Band-limited resampling by a Kaiser-windowed sinc, to `ceil(n * to / from)`
/// samples — the length `librosa.resample` returns.
///
/// Not bit-for-bit the reference's `soxr_hq`, and it does not need to be: the
/// daemon captures at 16 kHz, so this runs only for a caller that chose
/// another rate, and a transcription does not hinge on the last decibel of
/// stopband.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    const ZERO_CROSSINGS: f64 = 24.0;
    const ROLLOFF: f64 = 0.945;
    const BETA: f64 = 8.6;
    if samples.is_empty() || from == 0 || from == to {
        return samples.to_vec();
    }
    let ratio = f64::from(to) / f64::from(from);
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    // The filter's cutoff, relative to the input rate's Nyquist: the lower of
    // the two rates, less a roll-off band.
    let cutoff = ROLLOFF * ratio.min(1.0);
    let half_width = ZERO_CROSSINGS / cutoff;
    let i0_beta = bessel_i0(BETA);
    (0..out_len)
        .map(|i| {
            let center = i as f64 / ratio;
            let first = (center - half_width).ceil().max(0.0) as usize;
            let last = ((center + half_width).floor() as usize).min(samples.len() - 1);
            let mut acc = 0.0;
            for (j, &s) in samples.iter().enumerate().take(last + 1).skip(first) {
                let t = j as f64 - center;
                let x = t * cutoff;
                let sinc = if x.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let r = t / half_width;
                let window = bessel_i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
                acc += f64::from(s) * cutoff * sinc * window;
            }
            acc as f32
        })
        .collect()
}

/// The zeroth-order modified Bessel function of the first kind, by its power
/// series, which converges in a few dozen terms for the arguments a Kaiser
/// window takes.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..64 {
        let k = f64::from(k);
        term *= (half / k) * (half / k);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

/// Where `samples` is cut, as `start..end` ranges covering it exactly, each at
/// most [`MAX_CHUNK_SECONDS`] long bar the search slack below.
///
/// The reference's `split_audio_into_chunks`: each cut aims at the maximum
/// length, then moves to the quietest point within five seconds either side —
/// the minimum of a 100 ms moving sum of `|x|`, then the quietest sample
/// inside that window — so a cut lands in a pause rather than a word.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn split_points(samples: &[f32]) -> Vec<std::ops::Range<usize>> {
    let rate = f64::from(SAMPLE_RATE);
    let total = samples.len();
    let max_len = (MAX_CHUNK_SECONDS * rate) as usize;
    if total <= max_len {
        return vec![0..total];
    }
    let expand = (5.0 * rate) as usize;
    let win = ((0.1 * rate) as usize).max(4);

    let mut ranges = Vec::new();
    let mut start = 0;
    while total - start > max_len {
        let cut = start + max_len;
        let left = cut.saturating_sub(expand).max(start);
        let right = (cut + expand).min(total);
        let boundary = if right - left <= win {
            cut
        } else {
            let seg = &samples[left..right];
            // The first window with the smallest sum, as `np.argmin` picks it.
            let mut sum: f32 = seg[..win].iter().map(|s| s.abs()).sum();
            let (mut best, mut best_sum) = (0, sum);
            for pos in 1..=seg.len() - win {
                sum += seg[pos + win - 1].abs() - seg[pos - 1].abs();
                if sum < best_sum {
                    best = pos;
                    best_sum = sum;
                }
            }
            let inner = seg[best..best + win]
                .iter()
                .enumerate()
                .fold((0, f32::INFINITY), |(at, min), (i, s)| {
                    if s.abs() < min {
                        (i, s.abs())
                    } else {
                        (at, min)
                    }
                })
                .0;
            left + best + inner
        };
        let boundary = boundary.max(start + 1).min(total);
        ranges.push(start..boundary);
        start = boundary;
    }
    ranges.push(start..total);
    ranges
}

/// `samples` zero-padded to at least [`MIN_CHUNK_SECONDS`].
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn pad_short(samples: &[f32]) -> Vec<f32> {
    let min = ((MIN_CHUNK_SECONDS * f64::from(SAMPLE_RATE)) as usize).max(MIN_SAMPLES);
    let mut padded = samples.to_vec();
    if padded.len() < min {
        padded.resize(min, 0.0);
    }
    padded
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Reads a mono 16-bit PCM WAV into f32 samples in `[-1, 1)`, the one
    /// format the fixtures use. Chunks other than `fmt ` and `data` are
    /// skipped.
    pub(crate) fn read_wav(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let mut at = 12;
        let mut format = None;
        while at + 8 <= bytes.len() {
            let id = &bytes[at..at + 4];
            let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            let body = &bytes[at + 8..at + 8 + len];
            match id {
                b"fmt " => {
                    let channels = u16::from_le_bytes([body[2], body[3]]);
                    let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                    let bits = u16::from_le_bytes([body[14], body[15]]);
                    format = Some((channels, rate, bits));
                }
                b"data" => {
                    assert_eq!(format, Some((1, SAMPLE_RATE, 16)), "mono 16 kHz s16le");
                    return body
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
                        .collect();
                }
                _ => {}
            }
            at += 8 + len + (len & 1);
        }
        panic!("{} has no data chunk", path.display());
    }

    #[test]
    fn a_hot_clip_is_scaled_rather_than_clipped() {
        let out = normalize(vec![0.5, -2.0, 1.0], SAMPLE_RATE);
        assert_eq!(out, vec![0.25, -1.0, 0.5]);
        let quiet = normalize(vec![0.5, -0.25], SAMPLE_RATE);
        assert_eq!(quiet, vec![0.5, -0.25]);
    }

    #[test]
    fn resampling_keeps_a_tone_and_sets_the_length() {
        #[allow(clippy::cast_precision_loss)]
        let tone = |rate: u32, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
                .collect()
        };
        let input = tone(48_000, 48_000);
        let out = resample(&input, 48_000, SAMPLE_RATE);
        assert_eq!(out.len(), 16_000);
        let expected = tone(SAMPLE_RATE, 16_000);
        // Away from the edges, where the filter runs off the signal.
        let worst = out[200..15_800]
            .iter()
            .zip(&expected[200..15_800])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-2, "the tone moved by {worst}");
        // Upsampling rounds the length up, as `np.ceil` does.
        assert_eq!(
            resample(&tone(8_000, 1_001), 8_000, SAMPLE_RATE).len(),
            2_002
        );
    }

    #[test]
    fn a_recording_under_the_limit_is_one_piece() {
        assert_eq!(split_points(&[0.1; 1000]), [(0..1000)]);
    }

    /// A long recording is cut in its quiet stretch, not at the limit.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn a_long_recording_is_cut_where_it_is_quiet() {
        let rate = SAMPLE_RATE as usize;
        let max = MAX_CHUNK_SECONDS as usize * rate;
        let mut samples = vec![0.5f32; max + 30 * rate];
        // A second of silence three seconds before the limit.
        let quiet = max - 3 * rate;
        samples[quiet..quiet + rate].fill(0.0);
        let ranges = split_points(&samples);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[1].end, samples.len());
        assert_eq!(ranges[0].end, ranges[1].start);
        assert!(
            (quiet..quiet + rate).contains(&ranges[0].end),
            "cut at {} rather than in the silence at {quiet}",
            ranges[0].end
        );
    }

    #[test]
    fn a_short_clip_is_padded_to_half_a_second() {
        assert_eq!(pad_short(&[0.1; 10]).len(), 8_000);
        assert_eq!(pad_short(&vec![0.1; 9_000]).len(), 9_000);
    }

    #[test]
    fn resampling_nothing_or_to_the_same_rate_copies() {
        assert!(resample(&[], 48_000, SAMPLE_RATE).is_empty());
        assert_eq!(resample(&[0.5, -0.5], 0, SAMPLE_RATE), vec![0.5, -0.5]);
        assert_eq!(
            resample(&[0.5, -0.5], SAMPLE_RATE, SAMPLE_RATE),
            vec![0.5, -0.5]
        );
    }

    #[test]
    fn the_fixture_reads_as_eleven_seconds_in_range() {
        let samples =
            read_wav(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/jfk.wav"));
        assert_eq!(samples.len() / SAMPLE_RATE as usize, 11);
        assert!(samples.iter().all(|s| (-1.0..1.0).contains(s)));
    }
}
