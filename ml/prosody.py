"""Channel-robust voice features: pitch contour, voicing, loudness dynamics, duration.

A room or microphone reshapes the spectrum but leaves pitch, timing, and relative
loudness movement almost untouched, so these should transfer across recording
sessions better than spectral features. The encoder also never sees duration
(short clips are tiled to 3 s), so this adds information it lacks.

  python prosody.py     # -> data/cache/prosody.npy (one row per clip in meta.csv)
"""
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

from data import CACHE, Clips
from model import SAMPLE_RATE

NAMES = [
    "log_dur", "voiced_frac", "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope", "f0_start", "f0_end",
    "f0_absdelta", "f0_jumps", "f0_curve", "rms_std", "rms_slope", "rms_peak_pos", "rms_peaks", "attack", "decay",
    "hnr", "flat_mean", "flat_std", "zcr_mean", "zcr_std", "tilt_delta", "centroid_cv",
]


def features(wav):
    n = len(wav)
    if n < 2048:
        wav = np.pad(wav, (0, 2048 - n))
    hop = 160
    f0 = librosa.yin(wav, fmin=80, fmax=1200, sr=SAMPLE_RATE, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=wav, n_fft=1024, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=1024, hop_length=hop)[0]
    cen = librosa.feature.spectral_centroid(y=wav, sr=SAMPLE_RATE, n_fft=1024, hop_length=hop)[0]
    m = min(len(f0), len(rms), len(flat))
    f0, rms, flat, zcr, cen = f0[:m], rms[:m], flat[:m], zcr[:m], cen[:m]
    voiced = (rms > 0.1 * rms.max()) & (flat < 0.3) & (f0 > 85) & (f0 < 1150)
    st = 12 * np.log2(np.maximum(f0, 1) / 100.0)  # semitones re 100 Hz
    t = np.linspace(0, 1, m)
    if voiced.sum() >= 4:
        v, tv = st[voiced], t[voiced]
        slope, inter = np.polyfit(tv, v, 1)
        curve = np.polyfit(tv, v, 2)[0]
        d = np.abs(np.diff(v))
        pitch = [v.mean(), v.std(), np.percentile(v, 5), np.percentile(v, 95), np.percentile(v, 95) - np.percentile(v, 5), slope,
                 v[: max(1, len(v) // 5)].mean(), v[-max(1, len(v) // 5):].mean(), d.mean(), (d > 2).mean(), curve]
    else:
        pitch = [0.0] * 11
    logr = np.log(rms + 1e-5)
    env = rms / (rms.max() + 1e-9)
    peaks = ((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > 0.5)).sum() if m > 2 else 0
    peak_at = int(env.argmax())
    ac = librosa.autocorrelate(wav[: SAMPLE_RATE * 2])
    lo, hi = SAMPLE_RATE // 1000, SAMPLE_RATE // 80
    r = ac[lo:hi].max() / (ac[0] + 1e-9) if len(ac) > hi else 0.0
    hnr = 10 * np.log10(max(r, 1e-3) / max(1 - r, 1e-3))
    half = m // 2
    return np.array([
        np.log(n / SAMPLE_RATE), voiced.mean(), *pitch, logr.std(), np.polyfit(t, logr, 1)[0], peak_at / m, peaks / (n / SAMPLE_RATE),
        peak_at / m * (n / SAMPLE_RATE), (1 - peak_at / m) * (n / SAMPLE_RATE), hnr, np.log(flat + 1e-6).mean(), np.log(flat + 1e-6).std(),
        zcr.mean(), zcr.std(), np.log(cen[half:].mean() + 1) - np.log(cen[:half].mean() + 1) if half else 0.0, cen.std() / (cen.mean() + 1e-9),
    ], dtype=np.float32)


def _chunk(args):
    lo, hi = args
    clips = Clips()
    return np.stack([features(clips.wav(i)) for i in range(lo, hi)])


if __name__ == "__main__":
    n = len(Clips().meta)
    parts = [(i, min(n, i + 500)) for i in range(0, n, 500)]
    with ProcessPoolExecutor(6) as ex:
        X = np.concatenate(list(ex.map(_chunk, parts)))
    X = np.nan_to_num(X)
    np.save(CACHE / "prosody.npy", X)
    print("prosody features", X.shape)
