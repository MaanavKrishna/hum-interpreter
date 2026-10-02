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
