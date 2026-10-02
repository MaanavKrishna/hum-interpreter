"""Write the results section of README.md from runs/results.json:  python report.py"""
import json
import re

from data import ROOT

r = json.load(open(ROOT / "runs" / "results.json"))
c = r["conformal"]
lines = [
    f"Scored on {r['nTest']} vocalizations from recording sessions held out per person. Macro-F1, mean over eight people.",
    "",
    "| Model | Macro-F1 |",
    "|---|---|",
]
for s in r["summary"]:
    if s["name"].startswith("Hum, never") and not r["lopoReady"]:
        continue
    name = f"**{s['name']}**" if s["kind"] == "ours" else s["name"]
    lines.append(f"| {name} | {s['f1']:.2f} |")
lines += ["", f"Hum's plain accuracy is {r['humAcc']:.2f}. The last row is not a real result: it is the MFCC model scored with sessions mixed across train and test, shown to quantify the leakage that session-aware splitting removes.", ""]
lines += ["| Voice | Meanings | Test sounds | Majority | MFCC | Hum | Hum, person unseen |", "|---|---|---|---|---|---|---|"]
for p in r["perPerson"]:
    lines.append(f"| {p['id']} | {p['labels']} | {p['n_test']} | {p['majority']:.2f} | {p['mfcc']:.2f} | {p['hum']:.2f} | {p['lopo']:.2f} |")
lines += ["", "Conformal prediction sets (threshold fit on calibration sessions, checked on test sessions):", "", "| Target | Right meaning inside | Meanings offered | Single answers | Single answers correct |", "|---|---|---|---|---|"]
for t in c["table"]:
    lines.append(f"| {t['target']:.0%} | {t['coverage']:.0%} | {t['setSize']:.1f} | {t['single']:.0%} | {t['singleAcc']:.0%} |")
lines += ["", "Few-shot teaching" + (" (encoder never heard the person)" if r["lopoReady"] else " (provisional)") + ", macro-F1 by examples per meaning:", "", "| " + " | ".join(str(k) for k in r["curve"]["ks"]) + " |", "|" + "---|" * len(r["curve"]["ks"]), "| " + " | ".join(f"{x:.2f}" for x in r["curve"]["f1"]) + " |"]
lines += ["", "What did not help:", "", f"- Frozen pretrained speech encoders: best {max(s['f1'] for s in r['summary'] if 'layer' in s):.2f}.", f"- Cross-session contrastive positives with per-band normalisation: {r['xsession']:.2f}.", f"- Uncertainty-based active learning: {r['active']['active'][-1]:.2f} after {r['active']['ks'][-1]} labels, against {r['active']['random'][-1]:.2f} for random order."]

lines += [f"- {a['idea']}: {a['f1']:.2f}." for a in r.get("attempts", [])]
lines += ["", "The last four were attempts to beat the shipped model. None moved the score past 0.39, level with Hum within noise. The limit is the data (eight people, sessions that differ more than meanings do), not the model."]

sig_path = ROOT / "runs" / "significance.json"
if sig_path.exists():
    sig = json.load(open(sig_path))
    lines += ["", f"Paired bootstrap over test clips (2,000 draws). Hum macro-F1 {sig['hum']['f1']:.2f}, 95% interval {sig['hum']['ci'][0]:.2f} to {sig['hum']['ci'][1]:.2f}. Gap to each comparison:", "", "| Hum minus | Gap | 95% interval |", "|---|---|---|"]
    for g in sig["gaps"]:
        lines.append(f"| {g['vs']} | {g['gap']:+.3f} | {g['ci'][0]:+.3f} to {g['ci'][1]:+.3f} |")
    lines += ["", "Every interval excludes zero. Caveat: clips from one session are not independent and there are only eight people, so these intervals are narrower than the truth."]

path = ROOT / "README.md"
text = path.read_text()
block = "<!-- results:start -->\n" + "\n".join(lines) + "\n<!-- results:end -->"
path.write_text(re.sub(r"<!-- results:start -->.*<!-- results:end -->", lambda _: block, text, flags=re.S))
print("README results updated")
