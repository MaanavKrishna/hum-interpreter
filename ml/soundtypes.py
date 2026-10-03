"""What kind of sound is it? An objective layer under Hum's meaning hint.

Eleven types from three open corpora, each with held-out people or recordings:

  VocalSound   laughter, sigh, cough, throat clearing, sneeze, sniff   (speaker split from the authors)
  Nonspeech7k  breathing, crying, laughing, coughing, screaming, sneezing, yawning   (authors' test set,
               minus 32 clips whose source recording also appears in train)
  EmoGator     "other voice": bursts labelled confusion, interest, neutral, realization  (held-out speakers)

  python soundtypes.py cache          # -> data/cache/types_*.npy
  python soundtypes.py train 25       # -> runs/types.pt, runs/types.json
"""
import csv
import json
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from sklearn.metrics import confusion_matrix, f1_score

from data import CACHE, DATA
from model import CLIP_SAMPLES, EMB_DIM, SAMPLE_RATE, HumEncoder, fit_length
from train import DEVICE, RUNS, augment_wave, spec_augment, supcon

TYPES = ["laugh", "cry", "scream", "cough", "sneeze", "sniff", "sigh", "throat clearing", "yawn", "breathing", "other voice"]
VS = {"laughter": "laugh", "sigh": "sigh", "cough": "cough", "throatclearing": "throat clearing", "sneeze": "sneeze", "sniff": "sniff"}
NS = {"breath": "breathing", "crying": "cry", "laugh": "laugh", "cough": "cough", "screaming": "scream", "sneeze": "sneeze", "yawn": "yawn", "yawm": "yawn"}
OTHER_VOICE = {5, 17, 18, 21}  # EmoGator: confusion, interest, neutral, realization


def sources():
    """(path, type, split, corpus) for every clip."""
    rows = []
    for split, name in (("train", "tr"), ("val", "val"), ("test", "te")):
        for d in json.load(open(DATA / "vocalsound" / "datafiles" / f"{name}.json"))["data"]:
            f = d["wav"].split("/")[-1]
            rows.append((DATA / "vocalsound" / "audio_16k" / f, VS[f.rsplit("_", 1)[1][:-4]], split, "vocalsound"))
    tr = list(csv.reader(open(DATA / "nonspeech7k" / "metadata of train set .csv")))[1:]
    te = list(csv.reader(open(DATA / "nonspeech7k" / "metadata of test set.csv")))[1:]
    train_ids = sorted({r[1] for r in tr})
    val_ids = set(train_ids[::10])
    for r in tr:
        rows.append((DATA / "nonspeech7k" / "train" / r[0], NS[r[4]], "val" if r[1] in val_ids else "train", "nonspeech7k"))
    for r in te:
        if r[1] not in set(train_ids):
            rows.append((DATA / "nonspeech7k" / "test" / r[0], NS[r[4]], "test", "nonspeech7k"))
    for f in sorted((DATA / "EmoGator" / "data" / "mp3").glob("*.mp3")):
        spk, emo, _ = f.stem.split("-")
        if int(emo) in OTHER_VOICE:
            s = int(spk) % 10
            rows.append((f, "other voice", "test" if s == 0 else "val" if s == 1 else "train", "emogator"))
    return rows


def build_cache():
    rows = sources()
    chunks, lengths, keep = [], [], []
    for n, (path, typ, split, corpus) in enumerate(rows):
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            continue
        wav = torch.from_numpy(wav.mean(axis=1))
        if sr != SAMPLE_RATE:
            wav = AF.resample(wav, sr, SAMPLE_RATE)
        wav = wav.numpy()[: CLIP_SAMPLES * 2]
        peak = np.abs(wav).max() if len(wav) else 0
        if peak < 1e-4 or len(wav) < 1600:
            continue
        chunks.append((wav / peak * 0.9 * 32767).astype(np.int16))
        lengths.append(len(wav))
        keep.append((TYPES.index(typ), split, corpus))
        if n % 5000 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    np.save(CACHE / "types_audio.npy", np.concatenate(chunks))
    np.savez(CACHE / "types_meta.npz", lengths=lengths, label=[k[0] for k in keep], split=[k[1] for k in keep], corpus=[k[2] for k in keep])
    lab, spl = np.array([k[0] for k in keep]), np.array([k[1] for k in keep])
    for s in ("train", "val", "test"):
        print(s, {TYPES[i]: int(((lab == i) & (spl == s)).sum()) for i in range(len(TYPES))})


class TypeModel(nn.Module):
    """Hum encoder + linear layer over sound types. Exported to ONNX as one graph."""

    def __init__(self):
        super().__init__()
        self.encoder = HumEncoder()
        self.head = nn.Linear(EMB_DIM, len(TYPES))

    def forward(self, wav, spec_augment=None):
        emb = self.encoder(wav, spec_augment)
        return emb, self.head(emb) * 1.0


def evaluate(model, wav, idx, y, bs=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x = torch.from_numpy(np.stack([fit_length(wav(j)) for j in idx[i : i + bs]])).to(DEVICE)
            probs.append(torch.softmax(model(x)[1], 1).cpu().numpy())
    p = np.concatenate(probs)
    pred = p.argmax(1)
    return float((pred == y).mean()), float(f1_score(y, pred, average="macro")), pred, p


def train(epochs, bs=64, lr=1e-3, seed=7):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    audio = np.load(CACHE / "types_audio.npy", mmap_mode="r")
    meta = np.load(CACHE / "types_meta.npz")
    off = np.concatenate([[0], np.cumsum(meta["lengths"])])
    y, split, corpus = meta["label"], meta["split"], meta["corpus"]
    wav = lambda i: np.asarray(audio[off[i] : off[i + 1]], dtype=np.float32) / 32767.0
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    model = TypeModel().to(DEVICE)
    init = RUNS / "pretrain_emogator.pt"
    if init.exists():
        model.encoder.load_state_dict(torch.load(init, map_location=DEVICE))
    # balance types: rare ones (yawn, sneeze) are sampled more often
    w = 1.0 / np.bincount(y[tr], minlength=len(TYPES))[y[tr]] ** 0.7
    w /= w.sum()
    steps = len(tr) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    best, t0 = -1, time.time()
    for ep in range(epochs):
        model.train()
        tot = 0
        for _ in range(steps):
            pick = rng.choice(tr, bs, p=w)
            x = torch.from_numpy(np.stack([augment_wave(wav(i), rng) for i in pick])).to(DEVICE)
            t = torch.from_numpy(y[pick]).long().to(DEVICE)
            emb, logits = model(x, spec_augment)
            loss = F.cross_entropy(logits, t, label_smoothing=0.05) + 0.3 * supcon(emb, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        acc, f1, _, _ = evaluate(model, wav, va, y[va])
        print(f"types epoch {ep + 1}/{epochs} loss {tot / steps:.3f} val acc {acc:.3f} macro-F1 {f1:.3f} {time.time() - t0:.0f}s", flush=True)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), RUNS / "types.pt")
    model.load_state_dict(torch.load(RUNS / "types.pt", map_location=DEVICE))
    acc, f1, pred, _ = evaluate(model, wav, te, y[te])
    per_corpus = {c: float((pred[corpus[te] == c] == y[te][corpus[te] == c]).mean()) for c in np.unique(corpus[te])}
    cm = confusion_matrix(y[te], pred, labels=range(len(TYPES)))
    recall = {TYPES[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(TYPES)) if cm[i].sum()}
    out = {"types": TYPES, "test_acc": acc, "test_f1": f1, "per_corpus_acc": per_corpus, "recall": recall,
           "n_test": int(len(te)), "confusion": cm.tolist()}
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=1))
    json.dump(out, open(RUNS / "types.json", "w"), indent=1)


if __name__ == "__main__":
    build_cache() if sys.argv[1] == "cache" else train(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
