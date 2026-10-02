// Audio capture and decoding at 16 kHz mono.
import { SAMPLE_RATE } from './engine.js';

let ctx = null;
function audioContext() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  return ctx;
}

function toMono(buffer) {
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const ch = buffer.getChannelData(c);
    for (let i = 0; i < ch.length; i++) out[i] += ch[i] / buffer.numberOfChannels;
  }
  return out;
}

async function decode(arrayBuffer) {
  const buffer = await audioContext().decodeAudioData(arrayBuffer);
  if (buffer.sampleRate === SAMPLE_RATE) return toMono(buffer);
  // Browser ignored the requested rate: resample offline.
  const frames = Math.ceil((buffer.duration * SAMPLE_RATE));
  const off = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const src = off.createBufferSource();
  src.buffer = buffer;
  src.connect(off.destination);
  src.start();
  return toMono(await off.startRendering());
}

export async function loadFile(file) {
  return trimSilence(await decode(await file.arrayBuffer()));
}

export async function loadClip(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url}`);
  return decode(await res.arrayBuffer());
}

export function play(wav) {
  const c = audioContext();
  const buffer = c.createBuffer(1, wav.length, SAMPLE_RATE);
  buffer.copyToChannel(wav, 0);
  const src = c.createBufferSource();
  src.buffer = buffer;
  src.connect(c.destination);
  src.start();
  return src;
}

// Cut leading and trailing quiet so a recording looks like the dataset's
// pre-segmented vocalizations. Returns null if nothing rose above the floor.
export function trimSilence(wav, { frame = 320, floorRatio = 0.12, pad = 1600 } = {}) {
  const n = Math.floor(wav.length / frame);
  if (n === 0) return null;
  const rms = new Float32Array(n);
  let peak = 0;
  for (let f = 0; f < n; f++) {
    let s = 0;
    for (let i = 0; i < frame; i++) s += wav[f * frame + i] ** 2;
    rms[f] = Math.sqrt(s / frame);
    if (rms[f] > peak) peak = rms[f];
  }
  if (peak < 0.01) return null;
  const thr = peak * floorRatio;
  let first = 0;
  let last = n - 1;
  while (first < n && rms[first] < thr) first++;
  while (last > first && rms[last] < thr) last--;
  const start = Math.max(0, first * frame - pad);
  const end = Math.min(wav.length, (last + 1) * frame + pad);
  if (end - start < SAMPLE_RATE * 0.25) return null;
  return wav.slice(start, end);
}

// Microphone recorder. onLevel receives 0..1 while recording so the UI can draw it.
export class Recorder {
  constructor({ maxSeconds = 4, onLevel = () => {} } = {}) {
    this.maxSeconds = maxSeconds;
    this.onLevel = onLevel;
    this.active = false;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const c = audioContext();
    await c.resume();
    this.source = c.createMediaStreamSource(this.stream);
    this.analyser = c.createAnalyser();
    this.analyser.fftSize = 1024;
    this.source.connect(this.analyser);
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.done = new Promise((resolve) => (this.recorder.onstop = resolve));
    this.recorder.start();
    this.active = true;
    this.timer = setTimeout(() => this.active && this.onMax && this.onMax(), this.maxSeconds * 1000);
    const buf = new Float32Array(this.analyser.fftSize);
    const tick = () => {
      if (!this.active) return;
      this.analyser.getFloatTimeDomainData(buf);
      let s = 0;
      for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
      this.onLevel(Math.min(1, Math.sqrt(s / buf.length) * 6));
      requestAnimationFrame(tick);
    };
    tick();
  }

  async stop() {
    if (!this.active) return null;
    this.active = false;
    clearTimeout(this.timer);
    this.recorder.stop();
    await this.done;
    this.stream.getTracks().forEach((t) => t.stop());
    this.source.disconnect();
    this.onLevel(0);
    const blob = new Blob(this.chunks, { type: this.recorder.mimeType });
    const wav = await decode(await blob.arrayBuffer());
    return trimSilence(wav);
  }
}

// ---------- continuous listening ----------

// Cuts a live stream into separate vocalizations by loudness. Pure logic, so it can
// be tested by pushing recorded audio through it. Feed 16 kHz samples in any chunk size.
export class Segmenter {
  constructor({ frame = 320, hangFrames = 20, preFrames = 8, minSeconds = 0.3, maxSeconds = 4 } = {}) {
    Object.assign(this, { frame, hangFrames, preFrames, minSeconds, maxSeconds });
    this.pending = new Float32Array(0);
    this.pre = []; // recent quiet frames, kept so the start of a sound is not clipped
    this.active = null; // frames of the sound in progress
    this.quiet = 0;
    this.floor = 0.005; // running estimate of room noise
    this.level = 0;
  }

  push(samples) {
    const merged = new Float32Array(this.pending.length + samples.length);
    merged.set(this.pending);
    merged.set(samples, this.pending.length);
    const done = [];
    let i = 0;
    for (; i + this.frame <= merged.length; i += this.frame) {
      const f = merged.subarray(i, i + this.frame);
      let s = 0;
      for (let k = 0; k < f.length; k++) s += f[k] * f[k];
      const rms = Math.sqrt(s / f.length);
      this.level = rms;
      const loud = rms > Math.max(0.012, this.floor * 4);
      if (!this.active) {
        // The floor follows quiet frames down quickly and creeps up slowly.
        this.floor = rms < this.floor ? rms * 0.5 + this.floor * 0.5 : this.floor * 0.995 + rms * 0.005;
        if (loud) {
          this.active = [...this.pre, f.slice()];
          this.quiet = 0;
        } else {
          this.pre.push(f.slice());
          if (this.pre.length > this.preFrames) this.pre.shift();
        }
        continue;
      }
      this.active.push(f.slice());
      this.quiet = loud ? 0 : this.quiet + 1;
      const seconds = (this.active.length * this.frame) / SAMPLE_RATE;
      if (this.quiet >= this.hangFrames || seconds >= this.maxSeconds) {
        const keep = this.active.length - Math.max(0, this.quiet - 4);
        if ((keep * this.frame) / SAMPLE_RATE >= this.minSeconds) {
          const out = new Float32Array(keep * this.frame);
          for (let k = 0; k < keep; k++) out.set(this.active[k], k * this.frame);
          done.push(out);
        }
        this.active = null;
        this.pre = [];
      }
    }
    this.pending = merged.slice(i);
    return done;
  }
}

const TAP = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
}
registerProcessor('tap', Tap);`;

// Keeps the microphone open and calls onSound(wav) for each vocalization it hears.
export class LiveListener {
  constructor({ onSound, onLevel = () => {} }) {
    this.onSound = onSound;
    this.onLevel = onLevel;
    this.segmenter = new Segmenter();
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const c = audioContext();
    await c.resume();
    const url = URL.createObjectURL(new Blob([TAP], { type: 'application/javascript' }));
    await c.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    this.source = c.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(c, 'tap');
    const ratio = c.sampleRate / SAMPLE_RATE;
    this.node.port.onmessage = (e) => {
      let chunk = e.data;
      if (ratio !== 1) {
        // Browser refused 16 kHz: pick nearest samples. Crude, but only loudness and a short clip depend on it.
        const out = new Float32Array(Math.floor(chunk.length / ratio));
        for (let i = 0; i < out.length; i++) out[i] = chunk[Math.floor(i * ratio)];
        chunk = out;
      }
      for (const wav of this.segmenter.push(chunk)) this.onSound(wav);
      this.onLevel(Math.min(1, this.segmenter.level * 6));
    };
    this.source.connect(this.node);
  }

  stop() {
    this.node?.port.close();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.onLevel(0);
  }
}
