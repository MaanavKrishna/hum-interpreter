"""Load ReCANVo, resample to 16 kHz mono, cache, and expose metadata.

Filenames look like `200126_2142_00-13-04.06--00-13-04.324.wav`: recording date,
recording start time, then the clip's offsets inside that recording. The first
two fields identify a recording session, which is what evaluation must group by:
clips from one session share a microphone position, a room, and usually a mood.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

from model import SAMPLE_RATE

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MIN_PER_LABEL = 20  # a meaning needs this many examples for a person to be evaluated


def load_meta() -> pd.DataFrame:
    df = pd.read_csv(DATA / "dataset_file_directory.csv")
    df.columns = ["file", "person", "label"]
    df["session"] = df["person"] + "_" + df["file"].str.split("_").str[:2].str.join("_")
    return df


def build_cache():
    df = load_meta()
    CACHE.mkdir(exist_ok=True)
    chunks, lengths = [], []
    for i, f in enumerate(df["file"]):
        wav, sr = sf.read(DATA / f, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        if sr != SAMPLE_RATE:
            wav = AF.resample(wav, sr, SAMPLE_RATE)
        wav = wav.numpy()
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.9
        chunks.append((wav * 32767).astype(np.int16))
        lengths.append(len(wav))
        if i % 1000 == 0:
            print(f"{i}/{len(df)}", flush=True)
    np.save(CACHE / "audio.npy", np.concatenate(chunks))
    np.save(CACHE / "lengths.npy", np.array(lengths))
    df["seconds"] = np.array(lengths) / SAMPLE_RATE
    df.to_csv(CACHE / "meta.csv", index=False)
    print("cached", len(df), "clips,", round(sum(lengths) / SAMPLE_RATE / 60, 1), "minutes")


class Clips:
    """All clips in memory as int16, with per-clip slices."""

    def __init__(self):
        self.meta = pd.read_csv(CACHE / "meta.csv")
        self.audio = np.load(CACHE / "audio.npy")
        lengths = np.load(CACHE / "lengths.npy")
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])

    def wav(self, i: int) -> np.ndarray:
        return self.audio[self.offsets[i] : self.offsets[i + 1]].astype(np.float32) / 32767.0

    def usable(self) -> pd.DataFrame:
        """Rows whose (person, label) has enough examples to learn and test on."""
        m = self.meta
        counts = m.groupby(["person", "label"])["file"].transform("count")
        return m[counts >= MIN_PER_LABEL]


if __name__ == "__main__":
    build_cache()
    c = Clips()
    u = c.usable()
    print(u.groupby("person").agg(clips=("file", "count"), labels=("label", "nunique"), sessions=("session", "nunique")))
    print(u.groupby(["person", "label"])["session"].nunique().unstack(0).fillna(0).astype(int))
