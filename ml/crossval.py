"""Five-fold, session-grouped cross-validation of the whole pipeline.

The single train/calib/test split scores each person on a handful of sessions,
and calib and test disagree by up to 0.4 AUC for one person: too noisy to rank
ideas. Here every clip is tested exactly once, by an encoder that never trained
on its session, and the encoder gets 80% of sessions instead of 60%.

  python crossval.py 30          # trains 5 encoders, resumable; -> runs/cv_emb_{k}.npy
  python crossval.py score       # -> runs/cv.json
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from context import parse
from data import MIN_PER_LABEL, ROOT, Clips
from splits import SEED
from train import RUNS, embed_rows, train_encoder

K = 5
DISTRESS = {"frustrated", "dysregulated", "protest", "dysregulation-sick", "dysregulation-bathroom"}


def folds(meta):
    """fold id per usable clip: whole sessions, stratified by meaning, within each person."""
    fold = np.zeros(len(meta), dtype=int)
    for _, g in meta.groupby("person"):
        loc = meta.index.get_indexer(g.index)
        sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
        for k, (_, te) in enumerate(sgkf.split(g, g["label"], g["session"])):
            fold[loc[te]] = k
    return fold


def run(epochs):
    clips = Clips()
    meta = clips.usable()
    fold = folds(meta)
    init = RUNS / "pretrain_emogator.pt"
    for k in range(K):
        out = RUNS / f"cv_emb_{k}.npy"
        if out.exists():
            continue
        model = train_encoder(clips, meta[fold != k], epochs, lr=1e-3, tag=f"cv-{k}", init=init if init.exists() else None)
        np.save(out, embed_rows(model, clips, meta.index.values))
        print(f"fold {k} done", flush=True)


def alert_states(sessions, t, upset, window=45.0, decay=20.0, threshold=0.5):
    """The app's rule, replayed: alert is on after a sound if at least two sounds fall in
    the last `window` seconds and their recency-weighted upset probability is >= threshold."""
    on = np.zeros(len(t), dtype=bool)
    for s in np.unique(sessions):
        k = np.where(sessions == s)[0]
        k = k[np.argsort(t[k])]
        for a, i in enumerate(k):
            m = k[: a + 1][t[i] - t[k[: a + 1]] <= window]
            w = np.exp(-(t[i] - t[m]) / decay)
            on[i] = len(m) >= 2 and (w * upset[m]).sum() / w.sum() >= threshold
    return on


def score():
    clips = Clips()
    meta = clips.usable()
    fold = folds(meta)
    start = np.array([parse(f)[1] for f in meta.file])
    temps = {p["id"]: p["temperature"] for p in json.load(open(ROOT / "web" / "data" / "bundle.json"))["people"]}
    E = [np.load(RUNS / f"cv_emb_{k}.npy") for k in range(K)]
    people = {}
    for p, g in meta.groupby("person"):
        loc = meta.index.get_indexer(g.index)
        y, f = g.label.values, fold[loc]
        # Meanings scored: enough examples, and used in at least three sessions. A meaning
        # heard in one session only cannot be both taught and tested with sessions kept apart.
        vc = g.label.value_counts()
        ns = g.groupby("label").session.nunique()
        scored = sorted(l for l in vc.index if vc[l] >= MIN_PER_LABEL and ns[l] >= 3)
        pred = np.empty(len(g), dtype=object)
        second = np.empty(len(g), dtype=object)
        distress = np.full(len(g), np.nan)
        for k in range(K):
            tr, te = f != k, f == k
            counts = {l: (y[tr] == l).sum() for l in set(y[tr])}
            labels = sorted(l for l, c in counts.items() if c >= MIN_PER_LABEL // 2)
            P = np.stack([E[k][loc[tr & (y == l)]].mean(0) for l in labels])
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            S = E[k][loc[te]] @ P.T
            use = [i for i, l in enumerate(labels) if l in scored]
            order = np.argsort(-S[:, use], 1)
            pred[te] = [labels[use[i]] for i in order[:, 0]]
            second[te] = [labels[use[i]] for i in order[:, 1]]
            pr = np.exp(S / temps[p])  # the temperature the app uses for this voice
            pr /= pr.sum(1, keepdims=True)
            isd = np.array([l in DISTRESS for l in labels])
            if isd.any() and not isd.all():
                distress[te] = pr[:, isd].sum(1)
        # score meanings with enough examples overall, as in the fixed split
        keep = np.isin(y, scored)
        majority = vc[scored].idxmax()
        yd = np.isin(y, list(DISTRESS)).astype(int)
        ok = ~np.isnan(distress)
        alert = None
        if ok.all() and 0 < yd.mean() < 1:
            on = alert_states(g.session.values, start[loc], distress)
            alert = {"recall": float(on[yd == 1].mean()), "false_alarm": float(on[yd == 0].mean())}
        per_fold = [f1_score(y[keep & (f == k)], pred[keep & (f == k)], average="macro") for k in range(K)]
        people[p] = {
            "n": int(keep.sum()),
            "labels": int(len(set(y[keep]))),
            "f1": float(f1_score(y[keep], pred[keep], average="macro")),
            "acc": float(accuracy_score(y[keep], pred[keep])),
            "top2": float(np.mean((pred[keep] == y[keep]) | (second[keep] == y[keep]))),
            "majority_f1": float(f1_score(y[keep], [majority] * keep.sum(), average="macro")),
            "majority_acc": float(np.mean(y[keep] == majority)),
            "fold_f1": [float(x) for x in per_fold],
            "distress_auc": float(roc_auc_score(yd[ok], distress[ok])) if ok.any() and 0 < yd[ok].mean() < 1 else None,
            "distress_rate": float(yd.mean()),
            "alert": alert,
        }
        print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in people[p].items() if k != "fold_f1"})
    mean = lambda key: float(np.mean([v[key] for v in people.values() if v[key] is not None]))
    fold_means = np.mean([v["fold_f1"] for v in people.values()], axis=0)
    out = {
        "people": people,
        "f1": mean("f1"), "acc": mean("acc"), "top2": mean("top2"),
        "majority_f1": mean("majority_f1"), "majority_acc": mean("majority_acc"),
        "distress_auc": mean("distress_auc"),
        "alert_recall": float(np.mean([v["alert"]["recall"] for v in people.values() if v["alert"]])),
        "alert_false_alarm": float(np.mean([v["alert"]["false_alarm"] for v in people.values() if v["alert"]])),
        "fold_f1_mean": [float(x) for x in fold_means], "fold_f1_sd": float(np.std(fold_means)),
        "n": int(sum(v["n"] for v in people.values())),
    }
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "people"})
    json.dump(out, open(RUNS / "cv.json", "w"), indent=1)


if __name__ == "__main__":
    score() if sys.argv[1] == "score" else run(int(sys.argv[1]))
