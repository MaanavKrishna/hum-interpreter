"""Session-aware splits.

Random splits put clips from the same recording on both sides and overstate
accuracy. Every split here keeps whole recording sessions together.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from data import MIN_PER_LABEL

SEED = 7
MIN_TEST = 5


def fixed_split(meta):
    """Adds a `split` column, by whole sessions per person.

    test   ~25%  never touched until scoring
    calib  ~15%  not used to train the encoder; fits temperature and conformal threshold
    train  ~60%  trains the encoder and builds the prototypes
    """
    meta = meta.copy()
    meta["split"] = "train"
    for person, g in meta.groupby("person"):
        outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        rest, test = next(outer.split(g, g["label"], g["session"]))
        meta.loc[g.index[test], "split"] = "test"
        r = g.iloc[rest]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        _, calib = next(inner.split(r, r["label"], r["session"]))
        meta.loc[r.index[calib], "split"] = "calib"
    return meta


def eval_labels(g):
    """Meanings of one person that can be scored: enough examples on both sides."""
    tr = g[g.split == "train"].label.value_counts()
    te = g[g.split == "test"].label.value_counts()
    return sorted(l for l in tr.index if tr[l] >= MIN_PER_LABEL // 2 and te.get(l, 0) >= MIN_TEST)


def grouped_folds(g, n_splits=5):
    """Session-grouped CV folds for one person, used in leave-one-person-out runs."""
    k = min(n_splits, g["session"].nunique())
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    return list(sgkf.split(g, g["label"], g["session"]))
