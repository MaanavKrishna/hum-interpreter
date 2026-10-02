"""Hum encoder: raw waveform -> L2-normalised embedding.

The log-mel front end is expressed as fixed convolutions so the whole network
exports to plain ONNX ops. The browser feeds raw samples and gets exactly the
features the model was trained on, with no signal processing in JavaScript.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FFT = 400
HOP = 160
N_MELS = 64
EMB_DIM = 128


def _mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE, fmin=50.0, fmax=7600.0):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    points = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = points[i], points[i + 1], points[i + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


class ConvLogMel(nn.Module):
    """Log-mel spectrogram built from a strided Conv1d holding a windowed Fourier basis."""

    def __init__(self, band_norm=False):
        super().__init__()
        self.band_norm = band_norm
        n_bins = N_FFT // 2 + 1
        n = np.arange(N_FFT)
        window = 0.5 - 0.5 * np.cos(2 * math.pi * n / N_FFT)
        k = np.arange(n_bins)[:, None]
        angle = 2 * math.pi * k * n[None, :] / N_FFT
        basis = np.concatenate([np.cos(angle), -np.sin(angle)], axis=0) * window[None, :]
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32).unsqueeze(1))
        self.register_buffer("mel", torch.tensor(_mel_filterbank()).unsqueeze(-1))
        self.n_bins = n_bins

    def forward(self, wav):  # wav: (B, T)
        x = F.conv1d(wav.unsqueeze(1), self.basis, stride=HOP)
        power = x[:, : self.n_bins] ** 2 + x[:, self.n_bins :] ** 2
        mel = F.conv1d(power, self.mel)
        # Clamp: batched CPU convolution can return tiny negatives for near-silent bands.
        logmel = torch.log(mel.clamp_min(0.0) + 1e-6)
        if self.band_norm:
            # Ablation: subtract each mel band's mean over time to cancel a fixed
            # room/microphone filter. It did not help on held-out sessions.
            logmel = logmel - logmel.mean(dim=2, keepdim=True)
        else:
            logmel = logmel - logmel.mean(dim=(1, 2), keepdim=True)
        std = logmel.std(dim=(1, 2), keepdim=True)
        return (logmel / (std + 1e-5)).unsqueeze(1)  # (B, 1, mels, frames)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class HumEncoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, width=32, dropout=0.2, band_norm=False):
        super().__init__()
        self.frontend = ConvLogMel(band_norm)
        self.features = nn.Sequential(
            _block(1, width), _block(width, width * 2), _block(width * 2, width * 4), _block(width * 4, width * 8)
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(width * 8 * 2, emb_dim)

    def forward(self, wav, spec_augment=None):
        x = self.frontend(wav)
        if spec_augment is not None:
            x = spec_augment(x)
        x = self.features(x)
        x = x.mean(dim=2)  # pool frequency
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)  # pool time: mean + max
        return F.normalize(self.proj(self.dropout(x)), dim=1)


def fit_length(wav: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    """Tile short clips, centre-crop long ones. Mirrored exactly in web/js/audio.js."""
    if len(wav) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(wav) < n:
        wav = np.tile(wav, int(np.ceil(n / len(wav))))[:n]
    elif len(wav) > n:
        start = (len(wav) - n) // 2
        wav = wav[start : start + n]
    return wav.astype(np.float32)
