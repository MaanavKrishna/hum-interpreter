"""Train the Hum encoder.

  python train.py shipped 40     # all 8 people, train sessions only -> runs/shipped.pt
  python train.py lopo 15        # 8 encoders, each never hears one person -> runs/lopo.json
  python train.py seed 40 11     # extra ensemble member -> runs/seed_11.pt
  python train.py xsession 24    # ablation: cross-session contrastive + per-band norm

Loss: cross-entropy over (person, meaning) classes + supervised contrastive.
Scoring always uses the head that ships: cosine similarity to per-meaning prototypes.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from data import ROOT, Clips
from model import CLIP_SAMPLES, EMB_DIM, HumEncoder, fit_length
from splits import SEED, eval_labels, fixed_split

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RUNS = ROOT / "runs"


def augment_wave(wav, rng):
    if len(wav) > 800 and rng.random() < 0.5:  # random start so tiling phase varies
        wav = np.roll(wav, rng.integers(len(wav)))
    wav = fit_length(wav) * rng.uniform(0.3, 1.0)
    if rng.random() < 0.5:
        wav = wav + rng.normal(0, rng.uniform(0.001, 0.02), len(wav))
    return wav.astype(np.float32)


def spec_augment(x):
    """Mask random frequency bands and time spans per example."""
    b, _, m, t = x.shape
    x = x.clone()
    for i in range(b):
        for _ in range(2):
            f0 = np.random.randint(0, m - 8)
            x[i, :, f0 : f0 + np.random.randint(1, 9)] = 0
            t0 = np.random.randint(0, t - 30)
            x[i, :, :, t0 : t0 + np.random.randint(1, 31)] = 0
    return x


def supcon(emb, y, sess=None, temp=0.1):
    """Supervised contrastive loss. With `sess`, positives must come from a
    different recording session where one exists (the cross-session ablation)."""
    sim = emb @ emb.T / temp
    eye = torch.eye(len(y), dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    if sess is not None:
        cross = pos & (sess[:, None] != sess[None, :])
        keep = torch.where(cross.any(1, keepdim=True), cross, pos)
        sim = sim.masked_fill(pos & ~keep, -1e9)
        pos = keep
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


@torch.no_grad()
def embed_rows(model, clips, idx, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        w = np.stack([fit_length(clips.wav(j)) for j in idx[i : i + bs]])
        out.append(model(torch.from_numpy(w).to(DEVICE)).cpu())
    return torch.cat(out).numpy()


def prototype_predict(Etr, ytr, Ete):
    labels = sorted(set(ytr))
    P = np.stack([Etr[np.array(ytr) == l].mean(0) for l in labels])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return [labels[k] for k in (Ete @ P.T).argmax(1)]


def score_person(model, clips, g):
    labels = eval_labels(g)
    g = g[g.label.isin(labels)]
    tr, te = g[g.split == "train"], g[g.split == "test"]
    pred = prototype_predict(embed_rows(model, clips, tr.index.values), tr.label.tolist(), embed_rows(model, clips, te.index.values))
    return {"acc": float(accuracy_score(te.label, pred)), "f1": float(f1_score(te.label, pred, average="macro")), "n_test": len(te)}


def train_encoder(clips, rows, epochs=40, bs=64, lr=2e-3, tag="", cross_session=False, seed=SEED):
    """cross_session=True is the ablation: batches draw each meaning from several
    sessions, positives are cross-session only, and the front end normalises per band."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    key = (rows.person + "|" + rows.label).values
    classes = sorted(set(key))
    cid = {c: i for i, c in enumerate(classes)}
    sid = {s: i for i, s in enumerate(sorted(set(rows.session)))}
    y_all = np.array([cid[c] for c in key])
    s_all = np.array([sid[s] for s in rows.session])
    idx_all = rows.index.values
    by_class = {c: {} for c in range(len(classes))}
    for n, (c, s) in enumerate(zip(y_all, s_all)):
        by_class[c].setdefault(s, []).append(n)
    # Sample classes close to evenly: rare meanings matter as much as common ones.
    w = 1.0 / np.bincount(y_all)[y_all] ** 0.75
    w /= w.sum()
    model = HumEncoder(band_norm=cross_session).to(DEVICE)
    head = nn.Linear(EMB_DIM, len(classes), bias=False).to(DEVICE)
    steps = len(rows) // bs
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0
        for _ in range(steps):
            if cross_session:
                pick = []
                for c in rng.choice(len(classes), bs // 4, replace=len(classes) < bs // 4):
                    sess = list(by_class[c])
                    pick += [rng.choice(by_class[c][s]) for s in rng.choice(sess, 4, replace=len(sess) < 4)]
                pick = np.array(pick)
            else:
                pick = rng.choice(len(rows), bs, p=w)
            wav = torch.from_numpy(np.stack([augment_wave(clips.wav(idx_all[k]), rng) for k in pick])).to(DEVICE)
            y = torch.from_numpy(y_all[pick]).to(DEVICE)
            emb = model(wav, spec_augment)
            logits = emb @ F.normalize(head.weight, dim=1).T * 16.0
            con = supcon(emb, y, torch.from_numpy(s_all[pick]).to(DEVICE)) if cross_session else 0.5 * supcon(emb, y)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1) + con
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"{tag} epoch {ep + 1}/{epochs} loss {tot / steps:.3f} {time.time() - t0:.0f}s", flush=True)
    return model


def main(mode, epochs, seed=SEED):
    clips = Clips()
    meta = fixed_split(clips.usable())
    RUNS.mkdir(exist_ok=True)
    if mode == "seed":  # extra ensemble member: same recipe, different initialisation and batches
        model = train_encoder(clips, meta[meta.split == "train"], epochs, tag=f"seed-{seed}", seed=seed)
        res = {p: score_person(model, clips, g) for p, g in meta.groupby("person")}
        torch.save(model.state_dict(), RUNS / f"seed_{seed}.pt")
        print(f"seed {seed} mean F1", round(np.mean([r["f1"] for r in res.values()]), 3))
        json.dump(res, open(RUNS / f"seed_{seed}.json", "w"), indent=1)
        return
    if mode in ("shipped", "xsession"):
        model = train_encoder(clips, meta[meta.split == "train"], epochs, tag=mode, cross_session=mode == "xsession")
        res = {p: score_person(model, clips, g) for p, g in meta.groupby("person")}
        torch.save(model.state_dict(), RUNS / f"{mode}.pt")
    else:
        res = {}
        for p, g in meta.groupby("person"):
            model = train_encoder(clips, meta[meta.person != p], epochs, tag=f"lopo-{p}")
            res[p] = score_person(model, clips, g)
            # Keep this person's embeddings from an encoder that never heard them.
            np.save(RUNS / f"lopo_emb_{p}.npy", embed_rows(model, clips, g.index.values))
            print(p, res[p], flush=True)
    for p, r in res.items():
        print(p, round(r["acc"], 3), round(r["f1"], 3))
    print(mode, "mean acc", round(np.mean([r["acc"] for r in res.values()]), 3), "mean F1", round(np.mean([r["f1"] for r in res.values()]), 3))
    json.dump(res, open(RUNS / f"{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40, int(sys.argv[3]) if len(sys.argv) > 3 else SEED)
