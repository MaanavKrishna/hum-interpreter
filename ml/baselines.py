"""Baselines on the fixed session split: majority class and MFCC + logistic regression.
Also reports the same MFCC model under a random split to show how much leakage inflates scores."""
import json

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import ROOT, Clips
from model import SAMPLE_RATE
from splits import SEED, eval_labels, fixed_split


def mfcc_features(wav):
    if len(wav) < 2048:
        wav = np.pad(wav, (0, 2048 - len(wav)))
    m = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=20, n_fft=400, hop_length=160)
    d = librosa.feature.delta(m, width=min(9, m.shape[1] // 2 * 2 - 1)) if m.shape[1] >= 5 else np.zeros_like(m)
    return np.concatenate([m.mean(1), m.std(1), d.std(1)])


def score(y, p):
    return {"acc": float(accuracy_score(y, p)), "f1": float(f1_score(y, p, average="macro"))}


def main():
    clips = Clips()
    meta = fixed_split(clips.usable())
    X = {i: mfcc_features(clips.wav(i)) for i in meta.index}
    out = {}
    for person, g in meta.groupby("person"):
        labels = eval_labels(g)
        g = g[g.label.isin(labels)]
        tr, te = g[g.split == "train"], g[g.split == "test"]
        Xtr, Xte = np.stack([X[i] for i in tr.index]), np.stack([X[i] for i in te.index])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        lr.fit(Xtr, tr.label)
        majority = [tr.label.value_counts().idxmax()] * len(te)
        # Same model, random split: what you would report if you ignored sessions.
        Xa = np.stack([X[i] for i in g.index])
        a, b, ya, yb = train_test_split(Xa, g.label.values, test_size=0.25, stratify=g.label.values, random_state=SEED)
        leak = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)).fit(a, ya)
        out[person] = {
            "labels": labels,
            "n_train": len(tr),
            "n_test": len(te),
            "majority": score(te.label, majority),
            "mfcc_lr": score(te.label, lr.predict(Xte)),
            "mfcc_lr_random_split": score(yb, leak.predict(b)),
        }
        print(person, len(labels), "labels", {k: round(v["f1"], 3) for k, v in out[person].items() if isinstance(v, dict)})
    for k in ["majority", "mfcc_lr", "mfcc_lr_random_split"]:
        print(k, "mean acc", round(np.mean([o[k]["acc"] for o in out.values()]), 3), "mean F1", round(np.mean([o[k]["f1"] for o in out.values()]), 3))
    (ROOT / "runs").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "runs" / "baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
