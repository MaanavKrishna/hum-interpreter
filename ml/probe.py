"""Compare frozen pretrained encoders, layer by layer.

For each model and each hidden layer: mean-pool over time, L2-normalise, build
per-meaning prototypes from train sessions, score on calib (to choose a layer)
and on test (to report). Embeddings are cached in runs/probe/.

  python probe.py
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from data import DATA, ROOT, Clips
from model import fit_length
from splits import eval_labels, fixed_split

os.environ.setdefault("HF_HOME", str(DATA / "hf"))
from transformers import AutoFeatureExtractor, AutoModel, WhisperFeatureExtractor, WhisperModel  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = ROOT / "runs" / "probe"
MODELS = {
    "distilhubert": "ntu-spml/distilhubert",
    "whisper-tiny": "openai/whisper-tiny",
    "wav2vec2-base": "facebook/wav2vec2-base",
    # wav2vec 2.0 pretrained on 125 h of nonverbal vocalizations. Its unlabeled pretraining
    # audio included ReCANVo, so its score here has a small transductive advantage.
    "voc2vec": "alkiskoudounas/voc2vec-ls-pt",
}


def load(name):
    repo = MODELS[name]
    if name.startswith("whisper"):
        fe = WhisperFeatureExtractor.from_pretrained(repo)
        enc = WhisperModel.from_pretrained(repo).encoder
        frames = 150  # 3 s; Whisper's positions are sinusoidal, so the first 150 are valid alone
        enc.embed_positions = torch.nn.Embedding.from_pretrained(enc.embed_positions.weight[:frames].clone())
        enc.config.max_source_positions = frames

        def run(wavs):
            f = fe(list(wavs), sampling_rate=16000, return_tensors="np").input_features[:, :, : frames * 2]
            return enc(torch.from_numpy(f).to(DEVICE), output_hidden_states=True).hidden_states

        return enc.to(DEVICE).eval(), run
    fe = AutoFeatureExtractor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(DEVICE).eval()

    def run(wavs):
        x = fe(list(wavs), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        return model(x, output_hidden_states=True).hidden_states

    return model, run


@torch.no_grad()
def extract(name, clips, idx, bs=32):
    path = OUT / f"{name}.npy"
    if path.exists():
        return np.load(path)
    _, run = load(name)
    out = []
    for i in range(0, len(idx), bs):
        hs = run(np.stack([fit_length(clips.wav(j)) for j in idx[i : i + bs]]))
        out.append(torch.stack([h.mean(1) for h in hs], 1).float().cpu().numpy())  # (B, layers, dim)
        if i % 1600 == 0:
            print(name, i, flush=True)
    E = np.concatenate(out)
    np.save(path, E.astype(np.float16))
    return E


def proto_f1(E, g, pos, fit, score, labels):
    tr, te = g[g.split.isin(fit) & g.label.isin(labels)], g[g.split.isin(score) & g.label.isin(labels)]
    Etr, Ete = E[[pos[i] for i in tr.index]], E[[pos[i] for i in te.index]]
    P = np.stack([Etr[(tr.label == l).values].mean(0) for l in labels])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return f1_score(te.label, [labels[k] for k in (Ete @ P.T).argmax(1)], average="macro")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = Clips()
    meta = fixed_split(clips.usable())
    pos = {i: n for n, i in enumerate(meta.index)}
    report = {}
    for name in MODELS:
        E = extract(name, clips, meta.index.values).astype(np.float32)
        E -= E.mean(0, keepdims=True)  # centre each layer's features
        E /= np.linalg.norm(E, axis=2, keepdims=True) + 1e-9
        rows = []
        for layer in range(E.shape[1]):
            cal, tst = [], []
            for _, g in meta.groupby("person"):
                labels = eval_labels(g)
                cl = [l for l in labels if ((g.split == "calib") & (g.label == l)).any()]
                cal.append(proto_f1(E[:, layer], g, pos, ["train"], ["calib"], cl))
                tst.append(proto_f1(E[:, layer], g, pos, ["train"], ["test"], labels))
            rows.append({"layer": layer, "calib": float(np.mean(cal)), "test": float(np.mean(tst)), "per_person_test": [round(float(x), 3) for x in tst]})
            print(name, "layer", layer, "calib", round(rows[-1]["calib"], 3), "test", round(rows[-1]["test"], 3), flush=True)
        report[name] = rows
    json.dump(report, open(ROOT / "runs" / "probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
