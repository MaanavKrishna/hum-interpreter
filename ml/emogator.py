"""Pretrain the Hum encoder on EmoGator (32,130 vocal bursts, 357 speakers, 30 emotions).

  python emogator.py cache          # decode MP3 -> 16 kHz int16 cache
  python emogator.py pretrain 20    # -> runs/pretrain_emogator.pt

Files are named NNNNNN-EE-I.mp3: speaker, emotion category, take. Validation
uses held-out speakers so the logged accuracy is not a memory test.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from data import DATA
from model import CLIP_SAMPLES, EMB_DIM, SAMPLE_RATE, HumEncoder, fit_length
from train import DEVICE, RUNS, augment_wave, spec_augment, supcon

SRC = DATA / "EmoGator" / "data" / "mp3"
CACHE = DATA / "cache"


def build_cache():
    files = sorted(SRC.glob("*.mp3"))
    chunks, lengths, speakers, emotions = [], [], [], []
    for i, f in enumerate(files):
        try:
            wav, sr = sf.read(f, dtype="float32", always_2d=True)
        except Exception:
            continue
        wav = torch.from_numpy(wav.mean(axis=1))
        if sr != SAMPLE_RATE:
            wav = AF.resample(wav, sr, SAMPLE_RATE)
        wav = wav.numpy()[: CLIP_SAMPLES * 2]  # nothing longer than 6 s is needed
        peak = np.abs(wav).max() if len(wav) else 0
        if peak < 1e-4 or len(wav) < 1600:
            continue
        chunks.append((wav / peak * 0.9 * 32767).astype(np.int16))
        lengths.append(len(wav))
        spk, emo, _ = f.stem.split("-")
        speakers.append(int(spk))
        emotions.append(int(emo) - 1)
        if i % 4000 == 0:
            print(f"{i}/{len(files)}", flush=True)
    np.save(CACHE / "emogator_audio.npy", np.concatenate(chunks))
    np.savez(CACHE / "emogator_meta.npz", lengths=lengths, speaker=speakers, emotion=emotions)
    print("cached", len(lengths), "clips,", round(sum(lengths) / SAMPLE_RATE / 3600, 1), "hours")


def pretrain(epochs, bs=64, lr=2e-3, seed=7):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    audio = np.load(CACHE / "emogator_audio.npy", mmap_mode="r")
    meta = np.load(CACHE / "emogator_meta.npz")
    off = np.concatenate([[0], np.cumsum(meta["lengths"])])
    y, spk = meta["emotion"], meta["speaker"]
    wav = lambda i: np.asarray(audio[off[i] : off[i + 1]], dtype=np.float32) / 32767.0
    val = np.where(spk % 10 == 0)[0]
    trn = np.where(spk % 10 != 0)[0]
    model = HumEncoder().to(DEVICE)
    head = nn.Linear(EMB_DIM, 30, bias=False).to(DEVICE)
    steps = len(trn) // bs
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0
        for _ in range(steps):
            pick = rng.choice(trn, bs)
            x = torch.from_numpy(np.stack([augment_wave(wav(i), rng) for i in pick])).to(DEVICE)
            t = torch.from_numpy(y[pick]).long().to(DEVICE)
            emb = model(x, spec_augment)
            loss = F.cross_entropy(emb @ F.normalize(head.weight, dim=1).T * 16.0, t, label_smoothing=0.1) + 0.5 * supcon(emb, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        hit = 0
        with torch.no_grad():
            for i in range(0, len(val), 256):
                idx = val[i : i + 256]
                x = torch.from_numpy(np.stack([fit_length(wav(j)) for j in idx])).to(DEVICE)
                hit += ((model(x) @ F.normalize(head.weight, dim=1).T).argmax(1).cpu().numpy() == y[idx]).sum()
        print(f"emogator epoch {ep + 1}/{epochs} loss {tot / steps:.3f} held-out-speaker acc {hit / len(val):.3f} (chance 0.033) {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), RUNS / "pretrain_emogator.pt")


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        build_cache()
    else:
        pretrain(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
