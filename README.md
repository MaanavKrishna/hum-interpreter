# Hum

**A personal interpreter for nonverbal vocalizations.**

Some people communicate with sounds instead of words: many autistic people, and
people with cerebral palsy or certain genetic conditions. A parent knows that one
sound means "I want that" and another means "too loud, get me out". A new
teacher, a substitute nurse, or a babysitter does not, and needs get missed.

Hum learns one person's sounds from the people who know them, then helps
everyone else. It runs entirely in the browser. Audio never leaves the device.

Built for the ML Empowerment Build Challenge 3.0.

**Live demo:** https://maanavkrishna.github.io/hum-interpreter/

## What it does

| | |
|---|---|
| **Listen** | Tap, let the person vocalize, tap again. Hum answers with the likely meaning. |
| **Honest answers** | Hum returns a *set* of meanings sized by conformal prediction. One meaning when it is sure, two when torn, "maybe" when it is not. It does not bluff. |
| **Teach** | One tap on the right meaning updates that person's model on the spot. No server, no retraining. |
| **Start a new voice** | Teach Hum any voice from zero, including your own. Three examples per meaning are enough to try it; accuracy keeps improving with more. |
| **Voice map** | Every sound of one person, laid out by similarity and coloured by meaning. |
| **Hand-off file** | Save a taught voice as a 4 KB file and load it on another device, so the parent teaches once and the sitter's phone understands. The file holds embeddings, not audio. |
| **Passport** | A printable page for a new caregiver: each meaning, example sounds, what helps, and how far to trust Hum on it. |
| **How well it works** | The full evaluation, including the parts that are unflattering. |

## Run it locally

```bash
python serve.py
```

Then open http://localhost:8765. No build step. The trained model
(`web/model/hum.onnx`, 5.7 MB) and sample clips are in the repository.

## How it works

1. **Raw audio in.** 16 kHz mono, 3 seconds (short sounds are tiled, long ones cropped).
2. **Front end inside the network.** The log-mel spectrogram is computed by fixed
   convolutions holding a windowed Fourier basis, so the exported ONNX file takes
   raw samples. The browser does no signal processing, and its features match
   training to within 2e-7.
3. **Encoder.** A 1.2 M-parameter CNN trained on ReCANVo with cross-entropy over
   (person, meaning) classes plus a supervised contrastive loss, with waveform and
   spectrogram augmentation. Output: a 128-number description of the sound.
4. **Personal head.** Each meaning is the mean of its examples' embeddings.
   Classifying is a cosine similarity; teaching is adding one vector. That is
   what makes learning on the device instant.
5. **Conformal calibration.** Per person, a threshold is fit on sessions held out
   from all training, so the set of meanings Hum offers contains the right one at
   a chosen rate. For voices taught in the browser, the same threshold is
   re-derived by leave-one-out over the taught examples.

## Results

<!-- results:start -->
Scored on 2244 vocalizations from recording sessions held out per person. Macro-F1, mean over eight people.

| Model | Macro-F1 |
|---|---|
| Always guess the commonest | 0.15 |
| MFCC + logistic regression | 0.30 |
| DistilHuBERT, frozen | 0.32 |
| wav2vec 2.0 base, frozen | 0.33 |
| Whisper-tiny encoder, frozen | 0.34 |
| **Hum, never heard this person** | 0.34 |
| **Hum** | 0.38 |
| MFCC, sessions mixed (inflated) | 0.47 |

Hum's plain accuracy is 0.50. The last row is not a real result: it is the MFCC model scored with sessions mixed across train and test, shown to quantify the leakage that session-aware splitting removes.

| Voice | Meanings | Test sounds | Majority | MFCC | Hum | Hum, person unseen |
|---|---|---|---|---|---|---|
| P01 | 4 | 436 | 0.18 | 0.20 | 0.43 | 0.25 |
| P02 | 4 | 183 | 0.22 | 0.19 | 0.11 | 0.22 |
| P03 | 5 | 208 | 0.17 | 0.30 | 0.45 | 0.38 |
| P05 | 5 | 272 | 0.08 | 0.51 | 0.53 | 0.52 |
| P06 | 4 | 130 | 0.16 | 0.24 | 0.35 | 0.41 |
| P08 | 5 | 720 | 0.06 | 0.28 | 0.31 | 0.18 |
| P11 | 5 | 88 | 0.15 | 0.21 | 0.23 | 0.26 |
| P16 | 4 | 207 | 0.15 | 0.46 | 0.60 | 0.46 |

Conformal prediction sets (threshold fit on calibration sessions, checked on test sessions):

| Target | Right meaning inside | Meanings offered | Single answers | Single answers correct |
|---|---|---|---|---|
| 70% | 80% | 2.6 | 30% | 72% |
| 80% | 88% | 3.2 | 17% | 83% |
| 90% | 94% | 3.8 | 4% | 89% |

Few-shot teaching (encoder never heard the person), macro-F1 by examples per meaning:

| 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| 0.21 | 0.23 | 0.26 | 0.29 | 0.32 |

What did not help:

- Frozen pretrained speech encoders: best 0.34.
- Cross-session contrastive positives with per-band normalisation: 0.33.
- Uncertainty-based active learning: 0.30 after 80 labels, against 0.32 for random order.
- DistilHuBERT fine-tuned end to end (finetune.py; run stopped at epoch 8 of 12, epoch 7 chosen on calib): 0.38.
- Ensemble of three Hum encoders with different seeds (train.py seed): 0.38.
- Adding the previous sounds in the session as context (window chosen on calib): 0.38.
- Hum and frozen Whisper-tiny combined: 0.39.

The last four were attempts to beat the shipped model. None moved the score past 0.39, level with Hum within noise. The limit is the data (eight people, sessions that differ more than meanings do), not the model.

Paired bootstrap over test clips (2,000 draws). Hum macro-F1 0.38, 95% interval 0.35 to 0.40. Gap to each comparison:

| Hum minus | Gap | 95% interval |
|---|---|---|
| MFCC + logistic regression | +0.076 | +0.046 to +0.105 |
| DistilHuBERT, frozen | +0.056 | +0.028 to +0.083 |
| Whisper-tiny encoder, frozen | +0.038 | +0.009 to +0.067 |
| wav2vec 2.0 base, frozen | +0.047 | +0.020 to +0.074 |

Every interval excludes zero. Caveat: clips from one session are not independent and there are only eight people, so these intervals are narrower than the truth.
<!-- results:end -->

### Reading these numbers

- **This is a hard problem and the scores are modest.** Hum roughly doubles
  always-guess-the-commonest-meaning and scores above MFCC features and three
  large pretrained speech models (gaps of 0.04 to 0.08, bootstrap intervals
  excluding zero), but it is right about half the time. It is a hint.
- **Why it is hard.** Eight people, a few hundred to 1,700 clips each, labelled
  live by caregivers; the same meaning sounds different on different days.
- **Session leakage is real.** Scoring with clips from one recording on both
  sides of the split makes the same model look far better. Every headline number
  here keeps whole sessions apart.
- **Honesty is the feature.** Because accuracy is modest, the product never
  forces a single guess. When Hum commits to one meaning it is usually right;
  otherwise it says what it is choosing between.
- **A new person starts weak.** With an encoder that never heard them, one
  example per meaning gives 0.21 and twenty give 0.32. Teaching helps steadily;
  it is not instant.
- **People differ.** For some voices Hum is useful; for one (P02) it is worse than
  guessing. The passport tells the caregiver which meanings to trust.

## Limits and care

- Eight people is a small sample. Nothing here shows Hum generalises to the wider
  population.
- Hum does not diagnose anything and does not replace the people who know the
  communicator. Their read always wins.
- A wrong "content" when someone is in distress has a cost. That is why answers
  come as sets, and why the passport states reliability per meaning.
- Sample clips are real vocalizations from the public ReCANVo dataset, shared by
  families for research under CC BY 3.0 US. They are included with attribution
  and only from held-out sessions.
- Voices you teach are stored in your browser's local storage as embeddings, not
  audio.

## Repository

```
ml/model.py       encoder with convolutional log-mel front end
ml/data.py        ReCANVo loading, 16 kHz cache, session ids
ml/splits.py      session-aware train / calib / test splits
ml/baselines.py   majority and MFCC baselines, plus the leakage demonstration
ml/probe.py       frozen wav2vec 2.0, DistilHuBERT, Whisper-tiny comparison
ml/train.py       encoder training; leave-one-person-out; cross-session ablation; extra seeds
ml/finetune.py    end-to-end fine-tuning of DistilHuBERT or Whisper-tiny (did not help)
ml/attempts.json  scores of later attempts to raise accuracy
ml/evaluate.py    scoring, calibration, conformal sets, ONNX export, app bundle
ml/significance.py  paired bootstrap of Hum against each comparison model
ml/report.py      writes the results section of this file
web/              the app: static files, onnxruntime-web, no backend
```

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt transformers
mkdir -p data && curl -L -o data/ReCANVo.zip "https://zenodo.org/api/records/5786860/files/ReCANVo.zip/content"
cd data && unzip -q ReCANVo.zip && cd ../ml
../.venv/bin/python data.py          # cache audio
../.venv/bin/python baselines.py
../.venv/bin/python probe.py         # downloads three pretrained models
../.venv/bin/python train.py shipped 40
../.venv/bin/python train.py xsession 24
../.venv/bin/python train.py lopo 15
../.venv/bin/python evaluate.py && ../.venv/bin/python significance.py && ../.venv/bin/python report.py
```

Training takes about 20 minutes for the shipped encoder and about 2 hours for
leave-one-person-out on an Apple M1 Pro.

## Credit

- **ReCANVo**: Johnson, K. T., Narain, J., Quatieri, T., Maes, P., Picard, R. W.
  "ReCANVo: A database of real-world communicative and affective nonverbal
  vocalizations." *Scientific Data* 10, 523 (2023).
  Data: https://doi.org/10.5281/zenodo.5786859, CC BY 3.0 US.
- The idea of personalised models built from caregivers' live labels comes from
  the MIT Media Lab **Commalla** project. Hum is an independent student project
  and is not affiliated with it.
- onnxruntime-web (MIT), PyTorch, scikit-learn, librosa, Hugging Face Transformers.
- Typefaces: Atkinson Hyperlegible (Braille Institute) and Bricolage Grotesque.

Code: MIT licence.
