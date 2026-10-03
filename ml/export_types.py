"""Export the sound-type model for the browser and relate sound types to meanings.

  python export_types.py

- web/model/types.onnx: raw 16 kHz audio in, probabilities over sound types out.
- For every ReCANVo clip, the detected sound type. From train sessions only, a
  per-person table: when this person made a laugh, what did the caregiver say it meant?
  The app shows that table as "When Voice 16 laughed before, it meant delighted 9 of 10 times".
- Checks whether sound type helps the meaning hint, on the cross-validation folds.
- runs/types_bundle.json is merged into web/data/bundle.json by evaluate.py (and here).
"""
import json

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from crossval import K, folds
from data import MIN_PER_LABEL, ROOT, Clips
from model import CLIP_SAMPLES, fit_length
from soundtypes import TYPES, TypeModel
from splits import fixed_split
from train import DEVICE, RUNS

WEB = ROOT / "web"


class Probs(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, wav):
        return torch.softmax(self.m(wav)[1], 1)


def export(model):
    m = Probs(model).cpu().eval()
    x = torch.randn(1, CLIP_SAMPLES)
    path = WEB / "model" / "types.onnx"
    torch.onnx.export(m, (x,), path, input_names=["wav"], output_names=["probs"], opset_version=17, dynamo=False)
    diff = float(np.abs(ort.InferenceSession(str(path)).run(None, {"wav": x.numpy()})[0] - m(x).detach().numpy()).max())
    print(f"types.onnx {path.stat().st_size / 1e6:.1f} MB, max diff {diff:.1e}")


@torch.no_grad()
def type_probs(model, clips, idx, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        x = torch.from_numpy(np.stack([fit_length(clips.wav(j)) for j in idx[i : i + bs]])).to(DEVICE)
        out.append(torch.softmax(model(x)[1], 1).cpu().numpy())
    return np.concatenate(out)


def main():
    model = TypeModel().to(DEVICE)
    model.load_state_dict(torch.load(RUNS / "types.pt", map_location=DEVICE))
    export(model)
    model.to(DEVICE)  # export moved the shared weights to the CPU
    res = json.load(open(RUNS / "types.json"))

    clips = Clips()
    meta = fixed_split(clips.usable())
    P = type_probs(model, clips, meta.index.values)
    top = P.argmax(1)
    sure = P.max(1) >= 0.5
    meta = meta.assign(type=[TYPES[t] if s else None for t, s in zip(top, sure)])
    print("ReCANVo clips by detected type:", meta.type.value_counts(dropna=False).to_dict())
    print("type by meaning (row share):")
    import pandas as pd
    print(pd.crosstab(meta.label, meta.type.fillna("unclear"), normalize="index").round(2).to_string())

    patterns = {}
    for p, g in meta[meta.split == "train"].groupby("person"):
        tab = g.dropna(subset=["type"]).groupby(["type", "label"]).size()
        patterns[p] = {}
        for (t, l), n in tab.items():
            patterns[p].setdefault(t, {})[l] = int(n)

    # Does sound type help the meaning hint? Cross-validation folds, prototype scores + type log-probs.
    usable = clips.usable()
    fold = folds(usable)
    E = [np.load(RUNS / f"cv_emb_{k}.npy") for k in range(K)]
    Pu = type_probs(model, clips, usable.index.values)
    gain = {"hum": [], "hum+type": []}
    for p, g in usable.groupby("person"):
        loc = usable.index.get_indexer(g.index)
        y, f = g.label.values, fold[loc]
        vc, ns = g.label.value_counts(), g.groupby("label").session.nunique()
        scored = sorted(l for l in vc.index if vc[l] >= MIN_PER_LABEL and ns[l] >= 3)
        pa, pb = np.empty(len(g), dtype=object), np.empty(len(g), dtype=object)
        for k in range(K):
            tr, te = (f != k) & np.isin(y, scored), f == k
            Pm = np.stack([E[k][loc[tr & (y == l)]].mean(0) for l in scored])
            Pm /= np.linalg.norm(Pm, axis=1, keepdims=True)
            z = E[k][loc[te]] @ Pm.T / 0.1
            lh = z - np.log(np.exp(z).sum(1, keepdims=True))
            clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(np.log(Pu[loc[tr]] + 1e-4), [scored.index(l) for l in y[tr]])
            lt = np.log(clf.predict_proba(np.log(Pu[loc[te]] + 1e-4)) + 1e-6)
            pa[te] = [scored[i] for i in lh.argmax(1)]
            pb[te] = [scored[i] for i in (lh + 0.5 * lt).argmax(1)]
        keep = np.isin(y, scored)
        gain["hum"].append(f1_score(y[keep], pa[keep], average="macro"))
        gain["hum+type"].append(f1_score(y[keep], pb[keep], average="macro"))
    fusion = {k: float(np.mean(v)) for k, v in gain.items()}
    print("meaning macro-F1 five-fold:", fusion, "per person", {k: np.round(v, 2).tolist() for k, v in gain.items()})

    out = {"names": TYPES, "recall": res["recall"], "acc": res["test_acc"], "f1": res["test_f1"], "n_test": res["n_test"],
           "perCorpus": res["per_corpus_acc"], "patterns": patterns, "fusion": fusion,
           "recanvo": meta.type.fillna("unclear").value_counts().to_dict()}
    json.dump(out, open(RUNS / "types_bundle.json", "w"), indent=1)
    bpath = WEB / "data" / "bundle.json"
    b = json.load(open(bpath))
    merge_into(b, out)
    json.dump(b, open(bpath, "w"), separators=(",", ":"))
    print("bundle updated")


def merge_into(bundle, t):
    bundle["types"] = {k: t[k] for k in ("names", "recall", "acc", "f1", "n_test", "perCorpus", "fusion", "recanvo")}
    for p in bundle["people"]:
        p["patterns"] = t["patterns"].get(p["id"], {})


if __name__ == "__main__":
    main()
