# Devpost submission draft

Numbers below match README.md as of the final evaluation run. If you retrain,
rerun `ml/report.py` and update them.

## Links

- Live demo: https://maanavkrishna.github.io/hum-interpreter/
- Code: https://github.com/MaanavKrishna/hum-interpreter

## Title

Hum: a personal interpreter for nonverbal vocalizations

## Tagline

Helps a new caregiver understand someone who communicates with sounds, not words. Runs on the device.

## Problem

Many people are minimally speaking: they communicate with vocalizations rather
than words. Family members learn what each sound means. Everyone else (a new
teacher, a substitute nurse, a babysitter) starts from nothing, so requests go
unanswered and distress gets read as misbehaviour.

## Solution

Hum learns one person's sounds from the people who know them and shares that
knowledge with everyone else. Tap Listen, and Hum says what the sound most
likely means. If it is unsure, it says so and names the options. One tap from a
caregiver teaches it. A printable passport carries the result to the next
caregiver.

## Key features

- Listening and learning happen in the browser. Audio is never uploaded.
- Answers are prediction sets with a statistical coverage guarantee (conformal
  prediction), not forced single guesses.
- A caregiver can start a brand-new voice from zero; it improves with each sound taught.
- Hand-off file: a taught voice saves to a 4 KB file and loads on another device.
- Voice map: one person's sounds laid out by similarity.
- Communication passport: per-meaning examples, notes on what helps, and how far
  to trust the model.
- An evidence page inside the app with every score, including the bad ones.

## How we built it

- Data: ReCANVo, 7,077 vocalizations from 8 minimally speaking people, labelled
  live by caregivers (MIT, CC BY 3.0 US).
- Model: 1.2 M-parameter CNN in PyTorch. The log-mel front end is built from
  fixed convolutions so the ONNX export takes raw audio and the browser matches
  training exactly.
- Training: cross-entropy plus supervised contrastive loss over (person,
  meaning) classes.
- Personal head: one prototype per meaning; teaching is adding a vector.
- Calibration: split conformal prediction per person.
- App: static HTML, CSS, JavaScript with onnxruntime-web. No backend.

## Results

Tested on 2,244 vocalizations from recording sessions the model never heard.
Macro-F1, averaged over eight people:

| Model | Macro-F1 |
|---|---|
| Always guess the commonest meaning | 0.15 |
| MFCC + logistic regression | 0.30 |
| DistilHuBERT, frozen | 0.32 |
| wav2vec 2.0 base, frozen | 0.33 |
| Whisper-tiny encoder, frozen | 0.34 |
| Hum, encoder never heard this person | 0.34 |
| **Hum** | **0.38** |

- Hum's plain accuracy is 0.50. It is a hint, not an answer.
- Hum's lead over each comparison model is 0.04 to 0.08 macro-F1; paired
  bootstrap 95% intervals exclude zero (with the caveat that eight people is few).
- Leakage: the same MFCC model scores 0.47 if clips from one session sit on both
  sides of the split, against 0.30 when sessions are kept apart.
- Conformal sets at an 80% target: the right meaning is inside 88% of the time,
  3.2 meanings offered on average. 17% of answers are a single meaning, and
  those are right 83% of the time.
- A brand-new voice: 0.21 with one example per meaning, 0.32 with twenty.
- Per person, Hum ranges from 0.11 to 0.60. For one voice it is worse than guessing.

## Challenges

- Accuracy is modest and uneven across people. We chose to design around that
  rather than hide it.
- Random train/test splits leak the recording session and inflate scores. We
  rebuilt the evaluation around whole sessions.
- Seven attempts to raise accuracy failed to help: three frozen pretrained
  speech models, fine-tuning one of them, a cross-session contrastive loss, a
  three-model ensemble, session context, and active learning. We report them all.

## What is next

- Test with families and speech-language pathologists before any real use.
- Use context (time of day, activity) alongside sound.
- More people. Eight is too few to claim generality.

## Technologies

Python, PyTorch, torchaudio, scikit-learn, librosa, Hugging Face Transformers,
ONNX, onnxruntime-web, JavaScript, HTML, CSS.

## Target users

Parents and caregivers of minimally speaking people, and the teachers, nurses,
therapists, and sitters who meet them for the first time.

## Demo video script (about 2 minutes)

1. (15 s) The problem, one sentence, over the Listen screen.
2. (30 s) Pick Voice 16. Play two held-out recordings. Show one confident answer
   and one "X or Y" answer, each with the caregiver's label revealed.
3. (30 s) Start a new voice. Record three of your own sounds for two meanings.
   Record a fourth; Hum recognises it. Say: no upload, learned on this laptop.
4. (15 s) Voice map: click dots, hear how meanings cluster.
5. (15 s) Passport: type one "what helps" note, show print preview.
6. (20 s) Evidence page: the bar chart, the striped inflated bar, the conformal
   table. Say the accuracy number out loud and why sets matter.

## Credit line to include

Built on ReCANVo (Johnson, Narain, Quatieri, Maes, Picard, 2023). Inspired by the
MIT Media Lab Commalla project; not affiliated.
