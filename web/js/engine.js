// Hum engine: ONNX encoder + per-person prototype head + conformal prediction sets.
// Everything here runs on the device. No audio or embedding is sent anywhere.

export const SAMPLE_RATE = 16000;
export const CLIP_SAMPLES = 48000;

let session = null;

export async function loadModel(url = 'model/hum.onnx') {
  if (session) return session;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  return session;
}

// Tile short clips, centre-crop long ones. Mirrors fit_length() in ml/model.py.
export function fitLength(wav, n = CLIP_SAMPLES) {
  const out = new Float32Array(n);
  if (wav.length === 0) return out;
  if (wav.length < n) {
    for (let i = 0; i < n; i++) out[i] = wav[i % wav.length];
  } else {
    const start = Math.floor((wav.length - n) / 2);
    out.set(wav.subarray(start, start + n));
  }
  return out;
}

// Second model: what kind of sound it is (laugh, cry, cough...). Independent of any voice.
let typeSession = null;

export async function soundType(wav, types) {
  if (!typeSession) typeSession = await ort.InferenceSession.create('model/types.onnx', { executionProviders: ['wasm'] });
  const input = new ort.Tensor('float32', fitLength(wav), [1, CLIP_SAMPLES]);
  const p = (await typeSession.run({ wav: input })).probs.data;
  return types.map((t, i) => ({ type: t, p: p[i] })).sort((a, b) => b.p - a.p);
}

export async function embed(wav) {
  const s = await loadModel();
  const input = new ort.Tensor('float32', fitLength(wav), [1, CLIP_SAMPLES]);
  const out = await s.run({ wav: input });
  return Float32Array.from(out.emb.data);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normalise(v) {
  const n = Math.sqrt(dot(v, v)) || 1;
  return v.map((x) => x / n);
}

// A voice is one person's model: a running sum of embeddings per meaning.
// Teaching is adding a vector, which is why it is instant and needs no server.
export class Voice {
  constructor({ id, name, sums = {}, counts = {}, temperature = 0.1, qhat = null, coverage = 0.9 }) {
    this.id = id;
    this.name = name;
    this.sums = {};
    for (const [k, v] of Object.entries(sums)) this.sums[k] = Array.from(v);
    this.counts = { ...counts };
    this.temperature = temperature;
    this.qhat = qhat;
    this.coverage = coverage;
  }

  get labels() {
    return Object.keys(this.sums).filter((k) => this.counts[k] > 0);
  }

  get totalExamples() {
    return Object.values(this.counts).reduce((a, b) => a + b, 0);
  }

  teach(label, emb) {
    if (!this.sums[label]) {
      this.sums[label] = new Array(emb.length).fill(0);
      this.counts[label] = 0;
    }
    for (let i = 0; i < emb.length; i++) this.sums[label][i] += emb[i];
    this.counts[label] += 1;
  }

  // Returns meanings ranked by probability, plus the conformal set.
  interpret(emb) {
    const labels = this.labels;
    if (labels.length === 0) return { ranked: [], set: [], state: 'empty' };
    const sims = labels.map((l) => dot(emb, normalise(this.sums[l])));
    const logits = sims.map((s) => s / this.temperature);
    const max = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - max));
    const z = exps.reduce((a, b) => a + b, 0);
    const ranked = labels
      .map((label, i) => ({ label, p: exps[i] / z, sim: sims[i] }))
      .sort((a, b) => b.p - a.p);

    // Split conformal: keep every meaning whose probability clears 1 - qhat.
    // With a calibrated qhat the true meaning is in the set ~coverage of the time.
    let set;
    if (this.qhat == null) {
      set = ranked.filter((r, i) => i === 0 || r.p > 0.25);
    } else {
      set = ranked.filter((r) => r.p >= 1 - this.qhat);
      if (set.length === 0) set = [ranked[0]];
    }
    const state = labels.length === 1 ? 'single' : set.length === 1 ? 'confident' : set.length === 2 ? 'between' : 'unsure';
    return { ranked, set, state };
  }

  toJSON() {
    return {
      id: this.id,
      name: this.name,
      sums: this.sums,
      counts: this.counts,
      temperature: this.temperature,
      qhat: this.qhat,
      coverage: this.coverage,
    };
  }
}

// Uncertainty used to decide which unlabeled sounds are worth a caregiver's tap.
export function entropy(ranked) {
  return -ranked.reduce((s, r) => s + (r.p > 0 ? r.p * Math.log(r.p) : 0), 0);
}
