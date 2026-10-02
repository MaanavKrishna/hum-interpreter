"""Paired bootstrap: is Hum's macro-F1 gap over each comparison model real?

Resamples test clips within each person (2,000 draws), recomputes the mean
macro-F1 over people for both models, and reports the gap with a 95% interval.

  python significance.py
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import mfcc_features
from data import ROOT, Clips
from model import HumEncoder
from splits import SEED, eval_labels, fixed_split
from train import DEVICE, RUNS, embed_rows, prototype_predict

NAMES = {"whisper-tiny": "Whisper-tiny encoder, frozen", "wav2vec2-base": "wav2vec 2.0 base, frozen", "distilhubert": "DistilHuBERT, frozen"}


def main():
    clips = Clips()
    meta = fixed_split(clips.usable())
    pos = {i: n for n, i in enumerate(meta.index)}
    model = HumEncoder().to(DEVICE)
    model.load_state_dict(torch.load(RUNS / "shipped.pt", map_location=DEVICE))
    probe = json.load(open(RUNS / "probe.json"))
    feats = {}
    for key, rows in probe.items():
        layer = max(rows, key=lambda r: r["calib"])["layer"]
        E = np.load(RUNS / "probe" / f"{key}.npy").astype(np.float32)[:, layer]
        E -= E.mean(0, keepdims=True)
        feats[NAMES[key]] = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    truth, preds = {}, {"Hum": {}, "MFCC + logistic regression": {}, **{n: {} for n in feats}}
    for person, g in meta.groupby("person"):
        labels = eval_labels(g)
        g = g[g.label.isin(labels)]
        tr, te = g[g.split == "train"], g[g.split == "test"]
        truth[person] = te.label.values
        preds["Hum"][person] = np.array(prototype_predict(embed_rows(model, clips, tr.index.values), tr.label.tolist(), embed_rows(model, clips, te.index.values)))
        for name, E in feats.items():
            preds[name][person] = np.array(prototype_predict(E[[pos[i] for i in tr.index]], tr.label.tolist(), E[[pos[i] for i in te.index]]))
        X = lambda rows: np.stack([mfcc_features(clips.wav(i)) for i in rows.index])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)).fit(X(tr), tr.label)
        preds["MFCC + logistic regression"][person] = lr.predict(X(te))

    rng = np.random.default_rng(SEED)
    draws = {p: rng.integers(0, len(y), (2000, len(y))) for p, y in truth.items()}

    def boot(name):
        return np.mean([[f1_score(truth[p][d], preds[name][p][d], average="macro") for d in draws[p]] for p in truth], axis=0)

    hum = boot("Hum")
    out = {"hum": {"f1": float(np.mean([f1_score(truth[p], preds["Hum"][p], average="macro") for p in truth])), "ci": [float(x) for x in np.percentile(hum, [2.5, 97.5])]}, "gaps": []}
    for name in preds:
        if name == "Hum":
            continue
        gap = hum - boot(name)
        row = {"vs": name, "gap": float(gap.mean()), "ci": [float(x) for x in np.percentile(gap, [2.5, 97.5])], "p_hum_better": float((gap > 0).mean())}
        out["gaps"].append(row)
        print(f"Hum - {name}: {row['gap']:+.3f}  95% CI [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]  Hum better in {row['p_hum_better']:.0%} of draws")
    print("Hum macro-F1", round(out["hum"]["f1"], 3), "95% CI", [round(x, 3) for x in out["hum"]["ci"]])
    json.dump(out, open(RUNS / "significance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
