"""Fine-tune a pretrained speech encoder end to end with Hum's loss.

  python finetune.py distilhubert 12
  python finetune.py whisper-tiny 12

Checkpoint selection uses calib sessions only; test sessions are scored once
per epoch for the log but never used to choose anything.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data import DATA, ROOT, Clips
from model import EMB_DIM, fit_length
from splits import SEED, eval_labels, fixed_split
from train import DEVICE, RUNS, augment_wave, supcon

os.environ.setdefault("HF_HOME", str(DATA / "hf"))
from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel  # noqa: E402


class Backbone(nn.Module):
    """Pretrained encoder -> learned mix of layers -> mean over time -> 128-d unit vector."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "whisper-tiny":
            self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
            self.enc = WhisperModel.from_pretrained("openai/whisper-tiny").encoder
            self.enc.embed_positions = nn.Embedding.from_pretrained(self.enc.embed_positions.weight[:150].clone())
            self.enc.config.max_source_positions = 150
            dim, layers = 384, 5
        else:
            self.enc = AutoModel.from_pretrained("ntu-spml/distilhubert")
            self.enc.feature_extractor._freeze_parameters()
            dim, layers = 768, 3
        self.mix = nn.Parameter(torch.zeros(layers))
        self.proj = nn.Linear(dim, EMB_DIM)

    def forward(self, wav):
        if self.name == "whisper-tiny":
            f = self.fe(list(wav.cpu().numpy()), sampling_rate=16000, return_tensors="np").input_features[:, :, :300]
            hs = self.enc(torch.from_numpy(f).to(wav.device), output_hidden_states=True).hidden_states
        else:
            wav = (wav - wav.mean(1, keepdim=True)) / (wav.std(1, keepdim=True) + 1e-7)
            hs = self.enc(wav, output_hidden_states=True).hidden_states
        w = torch.softmax(self.mix, 0)
        x = sum(wi * h for wi, h in zip(w, hs)).mean(1)
        return F.normalize(self.proj(x), dim=1)


@torch.no_grad()
def embed(model, clips, idx, bs=64):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        w = torch.from_numpy(np.stack([fit_length(clips.wav(j)) for j in idx[i : i + bs]])).to(DEVICE)
        out.append(model(w).float().cpu())
    return torch.cat(out).numpy()


def proto_scores(model, clips, meta):
    """Mean macro-F1 over people on calib and on test, prototypes from train sessions."""
    cal, tst, per = [], [], {}
    for p, g in meta.groupby("person"):
        E = embed(model, clips, g.index.values)
        pos = {i: n for n, i in enumerate(g.index)}
        labels = eval_labels(g)
        tr = g[(g.split == "train") & g.label.isin(labels)]
        P = np.stack([E[[pos[i] for i in tr.index[tr.label == l]]].mean(0) for l in labels])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for split, acc in (("calib", cal), ("test", tst)):
            s = g[(g.split == split) & g.label.isin(labels)]
            pred = [labels[k] for k in (E[[pos[i] for i in s.index]] @ P.T).argmax(1)]
            acc.append(f1_score(s.label, pred, average="macro"))
        per[p] = float(tst[-1])
    return float(np.mean(cal)), float(np.mean(tst)), per


def main(name, epochs, bs=32):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    clips = Clips()
    meta = fixed_split(clips.usable())
    rows = meta[meta.split == "train"]
    key = (rows.person + "|" + rows.label).values
    classes = sorted(set(key))
    y_all = np.array([classes.index(k) for k in key])
    idx_all = rows.index.values
    w = 1.0 / np.bincount(y_all)[y_all] ** 0.75
    w /= w.sum()
    model = Backbone(name).to(DEVICE)
    head = nn.Linear(EMB_DIM, len(classes), bias=False).to(DEVICE)
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": 5e-5},
         {"params": [model.mix, *model.proj.parameters(), *head.parameters()], "lr": 1e-3}],
        weight_decay=1e-2,
    )
    steps = len(rows) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 1e-3], total_steps=epochs * steps, pct_start=0.1)
    best, log, t0 = -1, [], time.time()
    for ep in range(epochs):
        model.train()
        tot = 0
        for _ in range(steps):
            pick = rng.choice(len(rows), bs, p=w)
            wav = torch.from_numpy(np.stack([augment_wave(clips.wav(idx_all[k]), rng) for k in pick])).to(DEVICE)
            y = torch.from_numpy(y_all[pick]).to(DEVICE)
            emb = model(wav)
            loss = F.cross_entropy(emb @ F.normalize(head.weight, dim=1).T * 16.0, y, label_smoothing=0.1) + 0.5 * supcon(emb, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        cal, tst, per = proto_scores(model, clips, meta)
        log.append({"epoch": ep + 1, "loss": tot / steps, "calib": cal, "test": tst, "per_person_test": per})
        print(f"{name} epoch {ep + 1}/{epochs} loss {tot / steps:.3f} calib {cal:.3f} test {tst:.3f} {time.time() - t0:.0f}s", flush=True)
        if cal > best:
            best = cal
            torch.save(model.state_dict(), RUNS / f"ft_{name}.pt")
    chosen = max(log, key=lambda r: r["calib"])
    print(f"{name}: chosen epoch {chosen['epoch']} by calib ({chosen['calib']:.3f}); test {chosen['test']:.3f}")
    json.dump({"log": log, "chosen": chosen}, open(RUNS / f"ft_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
