"""Score the shipped encoder, calibrate each voice, and write everything the web app needs.

  python evaluate.py

Outputs web/model/hum.onnx, web/data/bundle.json, web/clips/*, runs/results.json.
"""
import json
import shutil

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, recall_score

from data import ROOT, Clips
from model import CLIP_SAMPLES, SAMPLE_RATE, HumEncoder, fit_length
from splits import SEED, eval_labels, fixed_split
from train import DEVICE, RUNS, embed_rows

WEB = ROOT / "web"
COVERAGE = 0.8  # app default; 0.7 and 0.9 are reported alongside
MIN_PROTO = 15
CLIPS_PER_PERSON = 30

MEANINGS = {
    "delighted": ("Delighted", "Happy or excited about something.", "#e0a100"),
    "frustrated": ("Frustrated", "Something is wrong, blocked, or not going their way.", "#cf3f2b"),
    "request": ("Request", "They want something. Offer choices.", "#2a7bd6"),
    "selftalk": ("Self-talk", "Content and vocalizing to themselves. No response needed.", "#7c869c"),
    "social": ("Social", "They want to connect or get your attention.", "#16a077"),
    "dysregulated": ("Dysregulated", "Overwhelmed. Lower the noise and give space.", "#8b3ab5"),
    "yes": ("Yes", "Agreeing or accepting.", "#5a9e2b"),
    "affectionate": ("Affectionate", "Showing warmth toward someone.", "#de5f98"),
    "laughter": ("Laughter", "Laughing.", "#ee7a1c"),
    "happy": ("Happy", "Calm, pleased.", "#c9b400"),
    "help": ("Help", "Asking for help with something.", "#0f8b8d"),
    "more": ("More", "Wants more of what just happened.", "#3d5a9e"),
    "protest": ("Protest", "Objecting. They do not want this.", "#a8324a"),
    "bathroom": ("Bathroom", "Needs the bathroom.", "#4f7f8a"),
    "dysregulation-sick": ("Unwell", "Distressed because they feel sick.", "#6d4c7d"),
}


def prototypes(E, y, labels):
    return {l: E[np.array(y) == l].sum(0) for l in labels}


def probs(E, protos, labels, T):
    P = np.stack([protos[l] / np.linalg.norm(protos[l]) for l in labels])
    z = E @ P.T / T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_temperature(E, y, protos, labels):
    yi = np.array([labels.index(l) for l in y])
    best = min(np.geomspace(0.02, 0.25, 30), key=lambda T: -np.log(probs(E, protos, labels, T)[np.arange(len(yi)), yi] + 1e-9).mean())
    return float(best)


def conformal_q(p, yi, coverage=COVERAGE):
    s = np.sort(1 - p[np.arange(len(yi)), yi])
    k = min(len(s) - 1, int(np.ceil((len(s) + 1) * coverage)) - 1)
    return float(s[k])


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro"))


def few_shot_curve(E, g, ks, trials=20):
    """Macro-F1 on test sessions after teaching k random sounds per meaning."""
    labels = eval_labels(g)
    pos = {i: n for n, i in enumerate(g.index)}
    tr, te = g[(g.split != "test") & g.label.isin(labels)], g[(g.split == "test") & g.label.isin(labels)]
    Ete = E[[pos[i] for i in te.index]]
    rng = np.random.default_rng(SEED)
    out = []
    for k in ks:
        f = []
        for _ in range(trials):
            pick = np.concatenate([rng.choice(tr.index[tr.label == l], min(k, (tr.label == l).sum()), replace=False) for l in labels])
            pr = prototypes(E[[pos[i] for i in pick]], g.loc[pick].label.tolist(), labels)
            f.append(macro_f1(te.label, [labels[j] for j in probs(Ete, pr, labels, 0.1).argmax(1)]))
        out.append(float(np.mean(f)))
    return out


def active_curve(E, g, budgets, trials=10):
    """Random labelling vs asking about the sounds the model is least sure of."""
    labels = eval_labels(g)
    pos = {i: n for n, i in enumerate(g.index)}
    tr, te = g[(g.split != "test") & g.label.isin(labels)], g[(g.split == "test") & g.label.isin(labels)]
    Ete = E[[pos[i] for i in te.index]]
    pool_idx = tr.index.values
    Epool = E[[pos[i] for i in pool_idx]]
    ypool = tr.label.values
    res = {"random": np.zeros(len(budgets)), "active": np.zeros(len(budgets))}
    for t in range(trials):
        rng = np.random.default_rng(SEED + t)
        seed = [rng.choice(np.where(ypool == l)[0]) for l in labels]
        for mode in res:
            chosen = list(seed)
            for b, budget in enumerate(budgets):
                while len(chosen) < budget:
                    rest = np.setdiff1d(np.arange(len(pool_idx)), chosen)
                    if mode == "random":
                        chosen.append(rng.choice(rest))
                    else:
                        pr = prototypes(Epool[chosen], ypool[chosen].tolist(), labels)
                        p = probs(Epool[rest], pr, labels, 0.1)
                        chosen.append(rest[(-(p * np.log(p + 1e-9)).sum(1)).argmax()])
                pr = prototypes(Epool[chosen], ypool[chosen].tolist(), labels)
                res[mode][b] += macro_f1(te.label, [labels[j] for j in probs(Ete, pr, labels, 0.1).argmax(1)]) / trials
    return res


def conformal_table(conf_data, targets=(0.7, 0.8, 0.9)):
    """Coverage and set size on test sessions for several targets, thresholds fit on calib."""
    rows = []
    for target in targets:
        cov, sizes, one_ok = [], [], []
        for pca, yca, pk, yk in conf_data:
            q = conformal_q(pca, yca, target)
            inset = pk >= 1 - q
            inset[np.arange(len(yk)), pk.argmax(1)] = True
            cov += inset[np.arange(len(yk)), yk].tolist()
            sizes += inset.sum(1).tolist()
            one_ok += (pk.argmax(1) == yk)[inset.sum(1) == 1].tolist()
        sizes = np.array(sizes)
        rows.append({
            "target": target, "coverage": float(np.mean(cov)), "setSize": float(sizes.mean()),
            "single": float((sizes == 1).mean()), "singleAcc": float(np.mean(one_ok)) if one_ok else 0.0,
            "upToTwo": float((sizes <= 2).mean()),
        })
    return {**next(r for r in rows if r["target"] == COVERAGE), "table": rows}


def cv_summary():
    """Five-fold session-grouped cross-validation (crossval.py), trimmed for the app."""
    path = RUNS / "cv.json"
    if not path.exists():
        return None
    cv = json.load(open(path))
    keep = ("f1", "acc", "top2", "majority_f1", "majority_acc", "distress_auc", "alert_recall", "alert_false_alarm", "fold_f1_sd", "n")
    people = [
        {"id": p, "labels": v["labels"], "n": v["n"], "f1": v["f1"], "top2": v["top2"], "auc": v["distress_auc"],
         "recall": v["alert"]["recall"] if v["alert"] else None, "falseAlarm": v["alert"]["false_alarm"] if v["alert"] else None}
        for p, v in cv["people"].items()
    ]
    return {**{k: cv[k] for k in keep}, "people": people}


def pretrained_rows():
    """Frozen pretrained encoders with the same prototype head; layer chosen on calib sessions."""
    path = RUNS / "probe.json"
    if not path.exists():
        return []
    names = {"wav2vec2-base": "wav2vec 2.0 base, frozen", "distilhubert": "DistilHuBERT, frozen", "whisper-tiny": "Whisper-tiny encoder, frozen", "voc2vec": "voc2vec (nonverbal-vocalization model), frozen"}
    out = []
    for key, rows in json.load(open(path)).items():
        best = max(rows, key=lambda r: r["calib"])
        out.append({"name": names[key], "f1": best["test"], "kind": "", "layer": best["layer"]})
    return sorted(out, key=lambda r: r["f1"])


def export_onnx(model):
    (WEB / "model").mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    x = torch.randn(1, CLIP_SAMPLES)
    path = WEB / "model" / "hum.onnx"
    torch.onnx.export(model, (x,), path, input_names=["wav"], output_names=["emb"], opset_version=17, dynamo=False)
    got = ort.InferenceSession(str(path)).run(None, {"wav": x.numpy()})[0]
    diff = float(np.abs(got - model(x).detach().numpy()).max())
    print(f"onnx exported, {path.stat().st_size / 1e6:.1f} MB, max abs diff vs torch {diff:.2e}")
    assert diff < 1e-3


def main():
    clips = Clips()
    meta = fixed_split(clips.usable())
    model = HumEncoder().to(DEVICE)
    model.load_state_dict(torch.load(RUNS / "shipped.pt", map_location=DEVICE))
    base = json.load(open(RUNS / "baselines.json"))
    lopo = json.load(open(RUNS / "lopo.json")) if (RUNS / "lopo.json").exists() else None

    if (WEB / "clips").exists():
        shutil.rmtree(WEB / "clips")
    rng = np.random.default_rng(SEED)
    people, per_person, temps = [], [], []
    conf_data = []
    ks, budgets = [1, 2, 5, 10, 20], [10, 20, 40, 80]
    curves, actives = [], []
    for person, g in meta.groupby("person"):
        E = embed_rows(model, clips, g.index.values)
        pos = {i: n for n, i in enumerate(g.index)}
        tr, ca, te = (g[g.split == s] for s in ("train", "calib", "test"))
        counts = tr.label.value_counts()
        # A voice ships a meaning only if the caregiver used it in at least two train
        # sessions: one session cannot separate the meaning from that day's room.
        nsess = tr.groupby("label").session.nunique()
        labels = sorted(l for l in counts.index if counts[l] >= MIN_PROTO and nsess[l] >= 2)
        Etr = E[[pos[i] for i in tr.index]]
        protos = prototypes(Etr, tr.label.tolist(), labels)

        ca = ca[ca.label.isin(labels)]
        Eca = E[[pos[i] for i in ca.index]]
        T = fit_temperature(Eca, ca.label.tolist(), protos, labels)
        yca = np.array([labels.index(l) for l in ca.label])
        q = conformal_q(probs(Eca, protos, labels, T), yca)
        temps.append(T)

        # Headline score: same meanings and test rows as the baselines.
        ev = eval_labels(g)
        tee = te[te.label.isin(ev)]
        pe = probs(E[[pos[i] for i in tee.index]], prototypes(Etr, tr.label.tolist(), ev), ev, T)
        pred = [ev[j] for j in pe.argmax(1)]
        f1, acc = macro_f1(tee.label, pred), float(accuracy_score(tee.label, pred))
        rec = recall_score(tee.label, pred, labels=ev, average=None, zero_division=0)

        # Keep calib and test probabilities to study conformal behaviour at several targets.
        tk = te[te.label.isin(labels)]
        pk = probs(E[[pos[i] for i in tk.index]], protos, labels, T)
        yk = np.array([labels.index(l) for l in tk.label])
        conf_data.append((probs(Eca, protos, labels, T), yca, pk, yk))

        # Clips to ship: test sessions only, spread across meanings.
        chosen = []
        per = max(2, CLIPS_PER_PERSON // len(labels))
        for l in labels:
            cand = tk.index[(tk.label == l) & (tk.seconds >= 0.4) & (tk.seconds <= 3.0)].values
            chosen += rng.choice(cand, min(per, len(cand)), replace=False).tolist()
        out_dir = WEB / "clips" / person
        out_dir.mkdir(parents=True)
        clip_list = []
        for n, i in enumerate(chosen):
            sf.write(out_dir / f"{n:02d}.wav", clips.wav(i), SAMPLE_RATE, subtype="PCM_16")
            clip_list.append({"file": f"clips/{person}/{n:02d}.wav", "label": g.loc[i, "label"]})
        by_label = {}
        for n, c in enumerate(clip_list):
            by_label.setdefault(c["label"], []).append(n)
        samples = [v[j] for j in range(2) for v in by_label.values() if len(v) > j][:8]
        rng.shuffle(samples)

        # Voice map: sessions the encoder never trained on.
        mg = g[(g.split != "train") & g.label.isin(labels)]
        if len(mg) > 450:
            keep = set(chosen) | set(rng.choice(mg.index.values, 450, replace=False).tolist())
            mg = mg[mg.index.isin(keep)]
        xy = TSNE(2, metric="cosine", init="pca", perplexity=min(30, len(mg) // 4), random_state=SEED).fit_transform(E[[pos[i] for i in mg.index]])
        clip_of = {i: n for n, i in enumerate(chosen)}
        vmap = [[round(float(x), 2), round(float(y), 2), labels.index(l), clip_of.get(i, -1)] for (x, y), l, i in zip(xy, mg.label, mg.index)]

        people.append({
            "id": person,
            "sums": {l: [round(float(v), 4) for v in protos[l]] for l in labels},
            "counts": {l: int(counts[l]) for l in labels},
            "temperature": T,
            "qhat": q,
            "clips": clip_list,
            "samples": [int(s) for s in samples],
            "map": vmap,
            "mapLabels": labels,
            "perLabel": {l: {"recall": float(r), "n": int((tee.label == l).sum())} for l, r in zip(ev, rec)},
        })
        per_person.append({
            "id": person, "labels": len(ev), "n_test": len(tee),
            "majority": base[person]["majority"]["f1"], "mfcc": base[person]["mfcc_lr"]["f1"],
            "hum": f1, "hum_acc": acc, "lopo": lopo[person]["f1"] if lopo else 0.0,
        })
        Ecold = np.load(RUNS / f"lopo_emb_{person}.npy") if lopo else E
        curves.append(few_shot_curve(Ecold, g, ks))
        a = active_curve(Ecold, g, budgets)
        actives.append([a["random"], a["active"]])
        print(person, "T", round(T, 3), "q", round(q, 3), "F1", round(f1, 3), "acc", round(acc, 3), "labels", labels, flush=True)

    mean = lambda k: float(np.mean([p[k] for p in per_person]))
    leak = float(np.mean([b["mfcc_lr_random_split"]["f1"] for b in base.values()]))
    actives = np.mean(actives, axis=0)
    results = {
        "nTest": int(sum(p["n_test"] for p in per_person)),
        "summary": [
            {"name": "Always guess the commonest", "f1": mean("majority"), "kind": ""},
            {"name": "MFCC + logistic regression", "f1": mean("mfcc"), "kind": ""},
            *pretrained_rows(),
            {"name": "Hum, never heard this person", "f1": mean("lopo"), "kind": "ours"},
            {"name": "Hum", "f1": mean("hum"), "kind": "ours"},
            {"name": "MFCC, sessions mixed (inflated)", "f1": leak, "kind": "warn"},
        ],
        "summaryNote": "",
        "perPerson": per_person,
        "humAcc": mean("hum_acc"),
        "conformal": conformal_table(conf_data),
        "curve": {"ks": ks, "f1": np.mean(curves, axis=0).round(4).tolist()},
        "active": {"ks": budgets, "random": actives[0].round(4).tolist(), "active": actives[1].round(4).tolist()},
        "lopoReady": bool(lopo),
        "cv": cv_summary(),
        "attempts": json.load(open(ROOT / "ml" / "attempts.json"))["rows"],
        "xsession": float(np.mean([r["f1"] for r in json.load(open(RUNS / "xsession.json")).values()])),
    }
    results["summaryNote"] = (
        f"The striped bar is the same MFCC model scored the common way, with clips from one session on both sides of the split. "
        f"It looks {leak / mean('mfcc'):.1f} times better than it is. Every other bar keeps sessions apart."
    )
    json.dump(results, open(RUNS / "results.json", "w"), indent=1)
    bundle = {
        "coverage": COVERAGE,
        "defaultTemperature": float(np.median(temps)),
        "meanings": {k: {"title": t, "plain": p, "color": c} for k, (t, p, c) in MEANINGS.items()},
        "people": people,
        "results": results,
    }
    if (RUNS / "types_bundle.json").exists():
        from export_types import merge_into
        merge_into(bundle, json.load(open(RUNS / "types_bundle.json")))
    (WEB / "data").mkdir(exist_ok=True)
    json.dump(bundle, open(WEB / "data" / "bundle.json", "w"), separators=(",", ":"))
    export_onnx(model)
    print(json.dumps({k: v for k, v in results.items() if k not in ("perPerson", "summaryNote")}, indent=1))
    clips_mb = sum(f.stat().st_size for f in (WEB / "clips").rglob("*.wav")) / 1e6
    print(f"bundle {(WEB / 'data' / 'bundle.json').stat().st_size / 1e6:.2f} MB, clips {clips_mb:.1f} MB")


if __name__ == "__main__":
    main()
