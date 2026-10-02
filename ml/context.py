"""Label-free context features, all of which the app can compute live.

Time of day, and the rhythm of vocalising: gap since the previous sound, how
many sounds in the last 30 s and 2 min, how much of the last 30 s was voiced,
and minutes since the recording began.

  python context.py     # -> data/cache/context.npy (one row per clip in meta.csv)
"""
import re

import numpy as np

from data import CACHE, Clips

NUM = r"(\d+(?:\.\d+)?)"
PATTERN = re.compile(rf"(\d{{6}})_(\d{{2}})(\d{{2}})[^_]*_(\d+)-(\d+)-{NUM}--(\d+)-(\d+)-{NUM}")


def parse(name):
    m = PATTERN.match(name)
    start = int(m[4]) * 3600 + int(m[5]) * 60 + float(m[6])
    end = int(m[7]) * 3600 + int(m[8]) * 60 + float(m[9])
    return int(m[2]) + int(m[3]) / 60, start, end


def build():
    meta = Clips().meta.copy()
    P = np.array([parse(f) for f in meta.file])
    meta["hour0"], meta["t"], meta["te"] = P[:, 0], P[:, 1], P[:, 2]
    C = np.zeros((len(meta), 9), dtype=np.float32)
    for _, g in meta.groupby("session"):
        g = g.sort_values("t")
        t, te, ix = g.t.values, g.te.values, g.index.values
        hour = (g.hour0.values + t / 3600) % 24
        gap = np.r_[600, np.clip(t[1:] - te[:-1], 0, 600)]
        recent = lambda ti, w: (t < ti) & (t >= ti - w)
        n30 = np.array([recent(ti, 30).sum() for ti in t])
        n120 = np.array([recent(ti, 120).sum() for ti in t])
        voiced30 = np.array([np.clip(te[recent(ti, 30)] - t[recent(ti, 30)], 0, None).sum() for ti in t])
        C[ix] = np.c_[
            np.sin(hour / 24 * 2 * np.pi), np.cos(hour / 24 * 2 * np.pi), np.sin(hour / 12 * 2 * np.pi), np.cos(hour / 12 * 2 * np.pi),
            np.log1p(gap), n30, np.log1p(n120), voiced30, np.log1p(t / 60),
        ]
    np.save(CACHE / "context.npy", C)
    return C


if __name__ == "__main__":
    print("context features", build().shape)
