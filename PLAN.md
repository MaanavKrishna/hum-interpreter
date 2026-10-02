# Hum — build plan

A personal interpreter for nonverbal vocalizations, so that someone who has just
met a minimally speaking person can understand them the way their family does.

## Product

| Piece | What it does | Why it is new |
|---|---|---|
| Listen | Mic or sample clip in, likely meaning out | Runs fully in the browser; audio never leaves the device |
| Honest answers | Returns a *set* of plausible meanings sized by a conformal guarantee, or "not sure" | Never forces a single guess on a caregiver |
| Teach | Caregiver taps the right meaning; the personal model updates instantly | Few-shot learning on device, no retraining, no server |
| Voice map | 2-D map of one person's sounds, coloured by meaning | Shows that each person has their own "language" |
| Passport | One-page hand-off for a new teacher, nurse, or sitter | Turns a model into something a human can read |

## Model

1. Raw 16 kHz waveform in. Log-mel front end is built from fixed convolutions
   inside the network, so the exported ONNX file needs no JS signal processing
   and browser output matches training exactly.
2. Compact CNN encoder trained with supervised contrastive + cross-entropy loss
   across participants, with augmentation.
3. Per-person prototype head on the embedding. Adding a label = updating a mean
   vector, which is why learning in the browser is instant.
4. Split conformal calibration per person for prediction sets.

## Evaluation (all reported honestly in README)

- Per-person macro-F1 and accuracy, session-aware splits where filenames allow.
- Baselines: majority class, MFCC + logistic regression.
- Leave-one-person-out: encoder never saw the person; only the prototype head adapts.
- Learning curve: accuracy vs labelled examples per meaning (1, 2, 5, 10, 20).
- Conformal coverage and mean set size.
- Active learning vs random labelling (tested; did not beat random, so it is not in the app).
- Frozen pretrained encoders (wav2vec 2.0, DistilHuBERT, Whisper-tiny) as comparison.

## Stack

- `ml/` PyTorch training, evaluation, ONNX export.
- `web/` static site, onnxruntime-web, no build step, no backend.
- `data/` ReCANVo (not committed).

## Order of work

1. Data audit and splits: done
2. Baselines and pretrained comparison: done
3. Encoder training + evaluation: done
4. Export + browser parity check: done (browser embedding equals Python to 4 decimals)
5. Web app: done; microphone path still needs a manual test on a real device
6. README and submission draft: done; leave-one-person-out numbers fill in when that run ends
7. Still to do by hand: record demo video, deploy static site, submit on Devpost
