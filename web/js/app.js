import { Voice, embed, loadModel, soundType } from './engine.js';
import { LiveListener, Recorder, loadClip, loadFile, play } from './audio.js';

const $ = (id) => document.getElementById(id);
const SPARE_COLOURS = ['#0f8b8d', '#b5651d', '#6a4c93', '#3d7a2a', '#c2185b', '#5d6d7e'];
const STORE = 'hum.voices.v1';
const NOTES = 'hum.notes.v1';
const ALERTS = 'hum.alerts.v1';
const PATTERNS = 'hum.patterns.v1'; // voice -> sound type -> meaning -> count, from what caregivers taught
const LOG = 'hum.log.v1'; // day -> sound type -> count, for sounds heard while listening
// Meanings that raise the alert unless the caregiver says otherwise.
const UPSET = new Set(['frustrated', 'dysregulated', 'protest', 'dysregulation-sick', 'dysregulation-bathroom', 'pain', 'scared', 'upset', 'distress']);
const ALERT_WINDOW = 45; // seconds of recent sounds the alert looks at
const ALERT_DECAY = 20;

let bundle;
let voices = []; // { voice: Voice, person: bundle person or null, examples: [] for custom }
let current;
let lastEmb = null;
let lastType = null;
let recorder = null;
let live = null;
let recent = []; // { at, label, sure, upset } for sounds heard while listening continuously

// ---------- storage (best effort: the app works without it) ----------
function readStore(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode or full: keep going in memory */
  }
}

// ---------- meanings ----------
function meaning(label) {
  const known = bundle.meanings[label];
  if (known) return known;
  let h = 0;
  for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return { title: label.charAt(0).toUpperCase() + label.slice(1), plain: '', color: SPARE_COLOURS[h % SPARE_COLOURS.length] };
}
const dot = (label) => `<span class="dot" style="background:${meaning(label).color}"></span>`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// ---------- voices ----------
// For a voice taught in the browser, recompute the conformal threshold by
// leave-one-out over its own examples, so honesty does not depend on our dataset.
function recalibrate(entry) {
  const ex = entry.examples;
  const v = entry.voice;
  v.qhat = null;
  if (ex.length < 12 || v.labels.length < 2) return;
  const scores = [];
  for (let i = 0; i < ex.length; i++) {
    if (v.counts[ex[i].label] < 2) continue;
    const loo = new Voice({ id: 'loo', name: '', temperature: v.temperature });
    ex.forEach((e, j) => j !== i && loo.teach(e.label, e.emb));
    const hit = loo.interpret(ex[i].emb).ranked.find((r) => r.label === ex[i].label);
    scores.push(1 - (hit ? hit.p : 0));
  }
  if (scores.length < 10) return;
  scores.sort((a, b) => a - b);
  const k = Math.min(scores.length - 1, Math.ceil((scores.length + 1) * v.coverage) - 1);
  v.qhat = scores[k];
}

function buildVoices() {
  voices = bundle.people.map((p) => ({
    person: p,
    examples: null,
    voice: new Voice({ id: p.id, name: `Voice ${p.id.slice(1)}`, sums: p.sums, counts: p.counts, temperature: p.temperature, qhat: p.qhat, coverage: bundle.coverage }),
  }));
  for (const saved of readStore(STORE, [])) voices.push(customEntry(saved));
}

// A voice taught or imported in this browser. `base` holds prototypes that arrived
// in a voice file; `examples` are sounds taught here, kept so the threshold can be refit.
function customEntry({ id, name, examples = [], base = null }) {
  const voice = new Voice({
    id,
    name,
    sums: base?.sums ?? {},
    counts: base?.counts ?? {},
    temperature: base?.temperature ?? bundle.defaultTemperature,
    qhat: base?.qhat ?? null,
    coverage: bundle.coverage,
  });
  examples.forEach((e) => voice.teach(e.label, e.emb));
  const entry = { person: null, examples, base, voice };
  if (!base) recalibrate(entry);
  return entry;
}

function saveCustom() {
  writeStore(STORE, voices.filter((v) => !v.person).map((v) => ({ id: v.voice.id, name: v.voice.name, examples: v.examples, base: v.base })));
}

function fillVoicePicker() {
  const sel = $('voice');
  sel.innerHTML = '';
  const recorded = document.createElement('optgroup');
  recorded.label = 'Recorded voices (ReCANVo)';
  const yours = document.createElement('optgroup');
  yours.label = 'Voices you taught';
  for (const v of voices) {
    const o = new Option(v.person ? `${v.voice.name}, ${v.voice.labels.length} meanings` : v.voice.name, v.voice.id);
    (v.person ? recorded : yours).append(o);
  }
  sel.append(recorded);
  if (yours.children.length) sel.append(yours);
  sel.append(new Option('Start a new voice', '__new'));
  sel.value = current.voice.id;
}

function selectVoice(id) {
  if (id === '__new') {
    const name = (window.prompt('Name this voice') || '').trim();
    if (!name) {
      $('voice').value = current.voice.id;
      return;
    }
    const entry = customEntry({ id: `custom-${Date.now()}`, name });
    voices.push(entry);
    saveCustom();
    current = entry;
  } else {
    current = voices.find((v) => v.voice.id === id);
  }
  lastEmb = null;
  recent = [];
  renderRecent();
  fillVoicePicker();
  renderIdle();
  renderSamples();
  renderMap();
  renderPassport();
}

// ---------- listen ----------
function renderIdle() {
  renderType(null);
  $('bars').hidden = true;
  $('truth').hidden = true;
  $('teach').hidden = true;
  if (current.voice.labels.length === 0) {
    $('answer-main').textContent = 'Nothing taught yet.';
    $('answer-sub').textContent = `Tap Listen and record one of ${current.voice.name}'s sounds, or your own. Then tell Hum what it meant. Two or three examples of each meaning are enough to start.`;
  } else {
    $('answer-main').textContent = 'Every sound means something.';
    $('answer-sub').textContent = `${current.voice.name} has ${current.voice.totalExamples} taught sounds across ${current.voice.labels.length} meanings. Tap Listen, or play a recording.`;
  }
}

function renderSamples() {
  const wrap = $('samples-wrap');
  const box = $('samples');
  box.innerHTML = '';
  wrap.hidden = !current.person;
  if (!current.person) return;
  current.person.samples.forEach((clipIndex, i) => {
    const b = document.createElement('button');
    b.textContent = `Recording ${i + 1}`;
    b.addEventListener('click', async () => {
      box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      const clip = current.person.clips[clipIndex];
      await hear(await loadClip(clip.file), { playIt: true, truth: clip.label });
    });
    box.append(b);
  });
}

function renderAnswer(result, truth) {
  document.querySelectorAll('.pattern').forEach((p) => p.remove());
  const names = result.set.map((r) => meaning(r.label).title);
  const main = $('answer-main');
  const sub = $('answer-sub');
  if (result.state === 'empty') return renderIdle();
  if (result.state === 'single') {
    main.textContent = `${names[0]}?`;
    sub.textContent = 'Hum only knows one meaning for this voice so far, so it cannot compare. Teach a second meaning.';
  } else if (result.state === 'confident') {
    main.textContent = `${names[0]}.`;
    sub.textContent = meaning(result.set[0].label).plain || 'Hum is fairly sure about this one.';
  } else if (result.state === 'between') {
    main.textContent = `${names[0]} or ${names[1].toLowerCase()}.`;
    sub.textContent = 'It could be either. Look at what is happening around them to decide.';
  } else {
    main.textContent = `Maybe ${names[0].toLowerCase()}.`;
    sub.textContent = `Hum is not sure. It could also be ${names.slice(1, -1).map((n) => n.toLowerCase()).join(', ')}${names.length > 2 ? ' or ' : ''}${names.at(-1).toLowerCase()}. Go by what is happening around them, and teach Hum this sound if you know.`;
  }

  const inSet = new Set(result.set.map((r) => r.label));
  const bars = $('bars');
  bars.hidden = result.ranked.length < 2;
  bars.innerHTML = result.ranked
    .map(
      (r) => `<div class="bar ${inSet.has(r.label) ? '' : 'out'}">
        <span>${dot(r.label)}${esc(meaning(r.label).title)}</span>
        <span class="bar-track"><span class="bar-fill" data-w="${(r.p * 100).toFixed(1)}" style="background:${meaning(r.label).color}"></span></span>
        <span class="bar-pct">${Math.round(r.p * 100)}%</span>
      </div>`
    )
    .join('');
  requestAnimationFrame(() => bars.querySelectorAll('.bar-fill').forEach((f) => (f.style.width = `${f.dataset.w}%`)));

  const t = $('truth');
  t.hidden = !truth;
  if (truth) {
    const ok = inSet.has(truth);
    const known = current.voice.labels.includes(truth);
    t.innerHTML = `The caregiver labelled this ${dot(truth)}<strong>${esc(meaning(truth).title.toLowerCase())}</strong>. ${
      !known ? 'Hum was never taught that meaning for this voice.' : ok ? (result.set.length === 1 ? 'Hum agreed.' : 'It was among the meanings Hum offered.') : 'Hum missed it.'
    }`;
  }
  renderTeach();
}

function renderTeach() {
  $('teach').hidden = false;
  const box = $('teach-buttons');
  box.innerHTML = '';
  const options = current.voice.labels.length ? current.voice.labels : [];
  for (const label of options) {
    const b = document.createElement('button');
    b.innerHTML = `${dot(label)}${esc(meaning(label).title)}`;
    b.addEventListener('click', () => teach(label));
    box.append(b);
  }
  $('teach-h').textContent = options.length ? 'What did it mean?' : 'Tell Hum what that sound meant';
  $('teach-hint').textContent = 'One tap teaches Hum. It learns on this device straight away.';
}

function teach(label) {
  if (!lastEmb) return;
  label = label.trim().toLowerCase();
  if (!label) return;
  current.voice.teach(label, lastEmb);
  if (lastType) rememberPattern(lastType, label);
  if (!current.person) {
    current.examples.push({ label, emb: Array.from(lastEmb, (x) => +x.toFixed(4)) });
    if (!current.base) recalibrate(current);
    saveCustom();
  }
  const n = current.voice.counts[label];
  renderAnswer(current.voice.interpret(lastEmb), null);
  $('teach-hint').textContent = `Taught. Hum now has ${n} ${n === 1 ? 'example' : 'examples'} of ${meaning(label).title.toLowerCase()} for ${current.voice.name}.${current.person ? ' Changes to recorded voices last until you reload.' : ''}`;
  fillVoicePicker();
  renderPassport();
}

// ---------- sound type: the objective layer ----------
// Body sounds are not attempts to communicate, so Hum names them and does not guess a meaning.
const BODY = new Set(['cough', 'sneeze', 'sniff', 'throat clearing', 'yawn', 'breathing']);
const PLURAL = { cry: 'cries', laugh: 'laughs', 'other voice': 'other voice sounds', breathing: 'breathing sounds', 'throat clearing': 'throat clearings' };
const plural = (t, n = 2) => (n === 1 ? t : PLURAL[t] ?? `${t}s`);
const named = (t) => (t === 'other voice' ? 'another voice sound' : t === 'breathing' ? 'breathing' : `${/^[aeiou]/.test(t) ? 'an' : 'a'} ${t}`);

function renderType(ranked) {
  const el = $('sound-type');
  if (!ranked) {
    el.hidden = true;
    return;
  }
  const top = ranked[0];
  el.hidden = false;
  const reliable = bundle.types.recall[top.type];
  el.classList.toggle('unclear', top.p < 0.5);
  el.textContent =
    top.p < 0.5
      ? `Sound type unclear: ${top.type} or ${ranked[1].type}`
      : `Sound: ${named(top.type)}, ${Math.round(top.p * 100)}% sure${reliable ? `. In testing Hum named ${plural(top.type)} right ${Math.round(reliable * 100)}% of the time` : ''}`;
}

// Per-voice patterns: what this sound type has meant for this person before.
function patternsFor(voiceId) {
  const saved = readStore(PATTERNS, {})[voiceId] ?? {};
  const base = current.person?.patterns ?? {};
  const out = {};
  for (const src of [base, saved])
    for (const [type, labels] of Object.entries(src))
      for (const [label, n] of Object.entries(labels)) {
        out[type] ??= {};
        out[type][label] = (out[type][label] ?? 0) + n;
      }
  return out;
}

function patternLine(type) {
  const row = patternsFor(current.voice.id)[type];
  if (!row) return '';
  const total = Object.values(row).reduce((a, b) => a + b, 0);
  if (total < 3) return '';
  const [label, n] = Object.entries(row).sort((a, b) => b[1] - a[1])[0];
  return `Past ${plural(type)} from ${current.voice.name} meant ${meaning(label).title.toLowerCase()} ${n} of ${total} times.`;
}

function rememberPattern(type, label) {
  const all = readStore(PATTERNS, {});
  const v = (all[current.voice.id] ??= {});
  v[type] ??= {};
  v[type][label] = (v[type][label] ?? 0) + 1;
  writeStore(PATTERNS, all);
}

function logSound(type) {
  const day = new Date().toISOString().slice(0, 10);
  const all = readStore(LOG, {});
  all[day] ??= {};
  all[day][type] = (all[day][type] ?? 0) + 1;
  writeStore(LOG, all);
  renderToday();
}

function renderToday() {
  const day = new Date().toISOString().slice(0, 10);
  const counts = readStore(LOG, {})[day] ?? {};
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  $('today-wrap').hidden = entries.length === 0;
  $('today').textContent = entries.map(([t, n]) => `${n} ${plural(t, n)}`).join(', ');
}

function exportLog() {
  const rows = [['date', 'sound type', 'count']];
  for (const [day, counts] of Object.entries(readStore(LOG, {}))) for (const [t, n] of Object.entries(counts)) rows.push([day, t, n]);
  const url = URL.createObjectURL(new Blob([rows.map((r) => r.join(',')).join('\n')], { type: 'text/csv' }));
  Object.assign(document.createElement('a'), { href: url, download: 'hum-sound-log.csv' }).click();
  URL.revokeObjectURL(url);
}

// ---------- alert: several upset sounds close together ----------
function alertsOn(label) {
  const saved = readStore(ALERTS, {})[`${current.voice.id}|${label}`];
  return saved ?? UPSET.has(label);
}

function upsetProbability(result) {
  return result.ranked.filter((r) => alertsOn(r.label)).reduce((s, r) => s + r.p, 0);
}

// Recent sounds vote, newer ones count more. One sound alone never raises the alert.
function renderRecent() {
  const now = Date.now() / 1000;
  recent = recent.filter((r) => now - r.at <= 300).slice(-12);
  $('recent-wrap').hidden = recent.length === 0;
  $('recent').innerHTML = recent
    .map((r) => `<li class="${r.sure ? '' : 'unsure'}">${dot(r.label)}${r.type ? `${esc(r.type)}: ` : ''}${esc(meaning(r.label).title.toLowerCase())}${r.sure ? '' : '?'}</li>`)
    .join('');
  const win = recent.filter((r) => now - r.at <= ALERT_WINDOW);
  const weights = win.map((r) => Math.exp(-(now - r.at) / ALERT_DECAY));
  const score = win.length ? win.reduce((s, r, i) => s + weights[i] * r.upset, 0) / weights.reduce((a, b) => a + b, 0) : 0;
  const box = $('alert');
  box.hidden = !(win.length >= 2 && score >= 0.5);
  if (!box.hidden) box.innerHTML = `${esc(current.voice.name)} may be upset.<span>${win.length} sounds in the last ${ALERT_WINDOW} seconds lean that way. Check on them.</span>`;
}

async function hear(wav, { playIt = false, truth = null, track = false } = {}) {
  const ear = $('ear');
  if (!wav) {
    $('answer-main').textContent = 'Too quiet.';
    $('answer-sub').textContent = 'Hum could not pick out a sound. Move the device closer and try again.';
    $('bars').hidden = true;
    $('truth').hidden = true;
    return;
  }
  ear.disabled = true;
  try {
    if (playIt) play(wav);
    const [emb, types] = await Promise.all([embed(wav), soundType(wav, bundle.types.names).catch(() => null)]);
    lastEmb = emb;
    lastType = types && types[0].p >= 0.5 ? types[0].type : null;
    const result = current.voice.interpret(lastEmb);
    renderAnswer(result, truth);
    renderType(types);
    if (types && BODY.has(types[0].type) && types[0].p >= 0.6) {
      const t = types[0].type;
      $('answer-main').textContent = `${named(t).replace(/^./, (c) => c.toUpperCase())}.`;
      $('answer-sub').textContent = 'A body sound, not an attempt to communicate, so Hum does not guess a meaning. It is counted in today\'s log while listening.';
      $('bars').hidden = true;
    }
    if (lastType && !BODY.has(lastType) && patternLine(lastType)) $('answer-sub').insertAdjacentHTML('afterend', `<p class="pattern answer-sub">${esc(patternLine(lastType))}</p>`);
    if (track && lastType) logSound(lastType);
    if (track && result.ranked.length) {
      recent.push({ at: Date.now() / 1000, label: result.ranked[0].label, type: lastType, sure: result.set.length === 1, upset: upsetProbability(result) });
      renderRecent();
    }
    if (window.matchMedia('(max-width: 760px)').matches) $('answer').scrollIntoView({ block: 'start', behavior: 'smooth' });
    if (current.voice.labels.length === 0) {
      $('answer-main').textContent = 'Heard it.';
      $('answer-sub').textContent = 'Now type what that sound meant, for example "happy" or "wants water".';
      $('bars').hidden = true;
      renderTeach();
    }
  } catch (err) {
    $('answer-main').textContent = 'The model did not load.';
    $('answer-sub').textContent = `Check your connection and reload the page. (${err.message})`;
  } finally {
    ear.disabled = false;
  }
}

async function toggleLive() {
  const btn = $('live');
  const ear = $('ear');
  if (live) {
    live.stop();
    live = null;
    btn.setAttribute('aria-pressed', 'false');
    btn.textContent = 'Keep listening';
    ear.disabled = false;
    $('ear-hint').textContent = 'Tap, let them vocalize, tap again. Up to 3 seconds.';
    return;
  }
  try {
    let busy = false;
    live = new LiveListener({
      onLevel: (l) => ear.style.setProperty('--level', l.toFixed(3)),
      onSound: async (wav) => {
        if (busy) return; // one sound at a time; the next is picked up when this one is answered
        busy = true;
        await hear(wav, { track: true });
        ear.disabled = true;
        busy = false;
      },
    });
    await live.start();
    btn.setAttribute('aria-pressed', 'true');
    btn.textContent = 'Stop listening';
    ear.disabled = true;
    $('ear-hint').textContent = 'Listening on its own. The ring moves with the sound it hears.';
    $('answer-main').textContent = 'Listening.';
    $('answer-sub').textContent = 'Hum will answer each sound it hears. Leave this open nearby.';
    $('bars').hidden = true;
    $('truth').hidden = true;
  } catch {
    live = null;
    $('answer-main').textContent = 'No microphone.';
    $('answer-sub').textContent = 'Allow microphone access in your browser, or play one of the recordings instead.';
  }
}

async function toggleEar() {
  const ear = $('ear');
  if (recorder) {
    const r = recorder;
    recorder = null;
    ear.setAttribute('aria-pressed', 'false');
    ear.querySelector('.ear-label').textContent = 'Listen';
    $('samples').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
    await hear(await r.stop());
    return;
  }
  try {
    recorder = new Recorder({ maxSeconds: 3, onLevel: (l) => ear.style.setProperty('--level', l.toFixed(3)) });
    recorder.onMax = toggleEar;
    await recorder.start();
    ear.setAttribute('aria-pressed', 'true');
    ear.querySelector('.ear-label').textContent = 'Stop';
  } catch {
    recorder = null;
    $('answer-main').textContent = 'No microphone.';
    $('answer-sub').textContent = 'Allow microphone access in your browser, or play one of the recordings instead.';
  }
}

// ---------- voice map ----------
function renderMap() {
  const svg = $('map-svg');
  const legend = $('map-legend');
  const note = $('map-note');
  svg.innerHTML = '';
  legend.innerHTML = '';
  if (!current.person) {
    note.textContent = 'The map is drawn for the eight recorded voices. Pick one at the top.';
    return;
  }
  const p = current.person;
  const xs = p.map.map((m) => m[0]);
  const ys = p.map.map((m) => m[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x) => 24 + ((x - x0) / (x1 - x0 || 1)) * 552;
  const sy = (y) => 24 + ((y - y0) / (y1 - y0 || 1)) * 392;
  const ns = 'http://www.w3.org/2000/svg';
  for (const [x, y, li, ci] of p.map) {
    const c = document.createElementNS(ns, 'circle');
    const label = p.mapLabels[li];
    c.setAttribute('cx', sx(x).toFixed(1));
    c.setAttribute('cy', sy(y).toFixed(1));
    c.setAttribute('r', ci >= 0 ? 5.5 : 3.5);
    c.setAttribute('fill', meaning(label).color);
    c.setAttribute('fill-opacity', ci >= 0 ? 1 : 0.55);
    if (ci >= 0) {
      c.classList.add('playable');
      c.setAttribute('tabindex', '0');
      c.setAttribute('role', 'button');
      c.setAttribute('aria-label', `Play a ${meaning(label).title.toLowerCase()} sound`);
      const go = async () => play(await loadClip(p.clips[ci].file));
      c.addEventListener('click', go);
      c.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && go());
    }
    const t = document.createElementNS(ns, 'title');
    t.textContent = meaning(label).title;
    c.append(t);
    svg.append(c);
  }
  legend.innerHTML = p.mapLabels.map((l) => `<div>${dot(l)}${esc(meaning(l).title)}</div>`).join('');
  note.textContent = 'Outlined dots can be played. Positions come from the model\'s 128-number description of each sound, flattened to two dimensions, so distances are approximate.';
}

// ---------- passport ----------
function renderPassport() {
  const v = current.voice;
  const notes = readStore(NOTES, {});
  $('passport-title').textContent = `${v.name}: communication passport`;
  const cvp = bundle.results.cv?.people.find((p) => p.id === current.person?.id);
  $('alert-note').textContent = !cvp || cvp.recall == null
    ? 'Upset alert: not yet measured for this voice. Treat it as a prompt to look, nothing more.'
    : cvp.recall >= 0.6 && cvp.falseAlarm <= 0.1
      ? `Upset alert: dependable for this voice. In testing it caught about ${Math.round(cvp.recall * 10)} in 10 upset sounds and raised a false alarm on ${Math.max(1, Math.round(cvp.falseAlarm * 100))} in 100 calm ones.`
      : `Upset alert: not dependable for this voice (caught ${Math.round(cvp.recall * 100)}% of upset sounds, false alarms on ${Math.round(cvp.falseAlarm * 100)}% of calm ones). Do not rely on it.`;
  const body = $('passport-body');
  if (v.labels.length === 0) {
    body.innerHTML = '<p>Teach this voice a few sounds on the Listen tab and its passport will appear here.</p>';
    return;
  }
  body.innerHTML = '';
  const labels = [...v.labels].sort((a, b) => v.counts[b] - v.counts[a]);
  for (const label of labels) {
    const m = meaning(label);
    const row = document.createElement('div');
    row.className = 'meaning-row';
    const per = current.person?.perLabel?.[label];
    let rel = `${v.counts[label]} taught ${v.counts[label] === 1 ? 'sound' : 'sounds'}.`;
    if (per) {
      const tenth = Math.round(per.recall * 10);
      rel += tenth >= 7 ? ` Hum recognises this about ${tenth} times in 10.` : tenth >= 4 ? ` Hum recognises this about ${tenth} times in 10, so check the context.` : ` Hum often misses this one. Trust your own read.`;
    }
    const key = `${v.id}|${label}`;
    const pats = patternsFor(v.id);
    const byType = Object.entries(pats).map(([t, ls]) => [t, ls[label] ?? 0]).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
    const typed = byType.reduce((s, [, n]) => s + n, 0);
    const usual = typed >= 3 ? ` Usually sounds like: ${byType.slice(0, 2).map(([t, n]) => `${t} (${n} of ${typed})`).join(', ')}.` : '';
    row.innerHTML = `
      <div><div class="meaning-name">${dot(label)}${esc(m.title)}</div><p class="hint">${esc(m.plain)}${esc(usual)}</p></div>
      <div><label class="hint" for="n-${esc(key)}">What helps</label><textarea id="n-${esc(key)}" placeholder="For example: offer the picture board, then wait.">${esc(notes[key] || '')}</textarea></div>
      <div><div class="play-row samples"></div><p class="reliability">${rel}</p><label class="alert-toggle"><input type="checkbox" ${alertsOn(label) ? 'checked' : ''}> Alert me when several sounds in a row mean this</label></div>`;
    row.querySelector('.alert-toggle input').addEventListener('change', (e) => {
      const all = readStore(ALERTS, {});
      all[key] = e.target.checked;
      writeStore(ALERTS, all);
    });
    row.querySelector('textarea').addEventListener('input', (e) => {
      const all = readStore(NOTES, {});
      all[key] = e.target.value;
      writeStore(NOTES, all);
    });
    const plays = row.querySelector('.play-row');
    (current.person?.clips ?? [])
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => c.label === label)
      .slice(0, 3)
      .forEach(({ c }, k) => {
        const b = document.createElement('button');
        b.textContent = `Hear example ${k + 1}`;
        b.addEventListener('click', async () => play(await loadClip(c.file)));
        plays.append(b);
      });
    body.append(row);
  }
}

// ---------- voice files: how a parent hands a voice to another device ----------
function voiceNotes(id) {
  const all = readStore(NOTES, {});
  return Object.fromEntries(Object.entries(all).filter(([k]) => k.startsWith(`${id}|`)).map(([k, v]) => [k.slice(id.length + 1), v]));
}

function exportVoice() {
  const v = current.voice;
  const file = {
    format: 'hum-voice',
    version: 1,
    name: v.name,
    sums: Object.fromEntries(Object.entries(v.sums).map(([k, a]) => [k, a.map((x) => +x.toFixed(4))])),
    counts: v.counts,
    temperature: v.temperature,
    qhat: v.qhat,
    notes: voiceNotes(v.id),
    alerts: Object.fromEntries(v.labels.map((l) => [l, alertsOn(l)])),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(file)], { type: 'application/json' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: `${v.name.replace(/[^\w-]+/g, '-').toLowerCase()}.hum.json` });
  a.click();
  URL.revokeObjectURL(url);
  $('voice-file-status').textContent = `Saved ${a.download}. It holds numbers that describe the sounds, not recordings.`;
}

async function importVoice(file) {
  const status = $('voice-file-status');
  let data;
  try {
    data = JSON.parse(await file.text());
  } catch {
    data = null;
  }
  const dim = bundle.people[0] ? Object.values(bundle.people[0].sums)[0].length : 0;
  const ok = data?.format === 'hum-voice' && data.sums && data.counts && Object.values(data.sums).every((a) => Array.isArray(a) && a.length === dim && a.every(Number.isFinite));
  if (!ok) {
    status.textContent = 'That is not a Hum voice file, or it was made with a different version of the model.';
    return;
  }
  const id = `custom-${Date.now()}`;
  const name = String(data.name || 'Imported voice').slice(0, 40);
  const entry = customEntry({ id, name, base: { sums: data.sums, counts: data.counts, temperature: data.temperature, qhat: data.qhat } });
  voices.push(entry);
  const notes = readStore(NOTES, {});
  for (const [label, text] of Object.entries(data.notes ?? {})) notes[`${id}|${label}`] = String(text).slice(0, 2000);
  writeStore(NOTES, notes);
  const alerts = readStore(ALERTS, {});
  for (const [label, on] of Object.entries(data.alerts ?? {})) alerts[`${id}|${label}`] = Boolean(on);
  writeStore(ALERTS, alerts);
  saveCustom();
  selectVoice(id);
  $('voice-file-status').textContent = `Loaded ${name} with ${entry.voice.labels.length} meanings. It is ready on the Listen tab.`;
}

// ---------- evidence ----------
function hbar(name, value, cls = '') {
  return `<div class="hbar ${cls}"><span>${name}</span><span class="hbar-track"><span class="hbar-fill" style="display:block;width:${(value * 100).toFixed(1)}%"></span></span><span class="hbar-val">${value.toFixed(2)}</span></div>`;
}

function typesBlock(t) {
  if (!t) return '';
  const pct = (x) => `${Math.round(x * 100)}%`;
  const rows = t.names
    .filter((n) => t.recall[n] != null)
    .map((n) => `<tr><td>${esc(n)}</td><td>${pct(t.recall[n])}</td></tr>`)
    .join('');
  const fused = t.fusion ? ` Adding the sound type to the meaning hint moved five-fold macro-F1 from ${t.fusion.hum.toFixed(2)} to ${t.fusion['hum+type'].toFixed(2)}.` : '';
  return `
    <div class="ev-block">
      <h2>What kind of sound: the part that is not guesswork</h2>
      <p>Meaning depends on context a microphone cannot hear. The kind of sound does not: a laugh is a laugh. Hum's second model names ${t.names.length} kinds of sound, trained on VocalSound, Nonspeech7k and EmoGator and tested on ${t.n_test.toLocaleString()} clips from people and recordings it never heard.</p>
      ${hbar('Accuracy', t.acc, 'ours')}
      ${hbar('Macro-F1', t.f1, 'ours')}
      <table class="ev" style="margin-top:.8rem"><thead><tr><th>Sound</th><th>Named correctly</th></tr></thead><tbody>${rows}</tbody></table>
      <p style="margin-top:.8rem">This model was tested on adults and children in everyday recordings, not on the eight ReCANVo voices, whose sounds have no type labels. Treat its answers for those voices as a good guess, not a measured fact.${fused}</p>
    </div>`;
}

function cvBlock(cv) {
  if (!cv) return '';
  const pct = (x) => (x == null ? 'n/a' : `${Math.round(x * 100)}%`);
  const rows = cv.people
    .map((p) => `<tr><td>Voice ${p.id.slice(1)}</td><td>${p.labels}</td><td>${p.f1.toFixed(2)}</td><td>${pct(p.top2)}</td><td>${p.auc == null ? 'n/a' : p.auc.toFixed(2)}</td><td>${pct(p.recall)}</td><td>${pct(p.falseAlarm)}</td></tr>`)
    .join('');
  return `
    <div class="ev-block">
      <h2>The stricter check: every session tested once</h2>
      <p>One split tests each person on a handful of sessions, which is a noisy estimate. So we also trained five encoders, each leaving out a different fifth of every person's sessions, and scored all ${cv.n} sounds by an encoder that never heard their session. More meanings per person are scored here, so the task is harder.</p>
      ${hbar('Always guess the commonest', cv.majority_f1)}
      ${hbar('Hum, five-fold', cv.f1, 'ours')}
      <p style="margin-top:.8rem">Macro-F1 ${cv.f1.toFixed(2)} (folds vary by about ${cv.fold_f1_sd.toFixed(2)}). The right meaning is among Hum's top two ${pct(cv.top2)} of the time. On plain accuracy Hum only ties guessing the commonest meaning (${pct(cv.acc)} against ${pct(cv.majority_acc)}): its advantage is recognising the rarer meanings a guesser never names. This lower number is the one to trust.</p>
    </div>
    <div class="ev-block">
      <h2>The upset alert</h2>
      <p>While listening continuously, Hum raises an alert when at least two sounds in the last 45 seconds lean upset. Replaying that exact rule over every session, five-fold:</p>
      <table class="ev"><thead><tr><th>Voice</th><th>Meanings</th><th>Macro-F1</th><th>Top two</th><th>Upset vs not (AUC)</th><th>Upset sounds caught</th><th>False alarms</th></tr></thead><tbody>${rows}</tbody></table>
      <p style="margin-top:.8rem">For voices 05, 08 and 16 the alert catches about seven in ten upset sounds with almost no false alarms. For the others it is not dependable: it misses most upset sounds, or (voice 03) cries wolf. The passport says which kind each voice is.</p>
    </div>`;
}

function renderEvidence() {
  const r = bundle.results;
  const c = r.conformal;
  const rows = r.perPerson
    .map((p) => `<tr><td>Voice ${p.id.slice(1)}</td><td>${p.labels}</td><td>${p.n_test}</td><td>${p.majority.toFixed(2)}</td><td>${p.mfcc.toFixed(2)}</td><td>${p.hum.toFixed(2)}</td><td>${p.lopo.toFixed(2)}</td></tr>`)
    .join('');
  $('evidence-body').innerHTML = `
    <div class="ev-block">
      <h2>Telling meanings apart</h2>
      <p>Macro-F1 averaged over eight people (1.00 is perfect; it weighs rare meanings as much as common ones). Tested on ${r.nTest} vocalizations from held-out sessions.</p>
      ${r.summary.map((s) => hbar(s.name, s.f1, s.kind)).join('')}
      <p style="margin-top:.8rem">${r.summaryNote}</p>
    </div>
    <div class="ev-block">
      <h2>Person by person</h2>
      <p>People differ a lot. Hum is a hint, and for some voices a weak one.</p>
      <table class="ev"><thead><tr><th>Voice</th><th>Meanings</th><th>Test sounds</th><th>Always guess commonest</th><th>MFCC baseline</th><th>Hum</th><th>Hum, never heard this person</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
    ${typesBlock(bundle.types)}
    ${cvBlock(r.cv)}
    <div class="ev-block">
      <h2>Saying "not sure" when it should</h2>
      <p>With accuracy this modest, a single forced guess would mislead. Hum instead offers a set of meanings sized, by split conformal prediction, so the right one is inside about ${Math.round(c.target * 100)}% of the time. Thresholds were fit on sessions used for nothing else, then checked on test sessions:</p>
      <table class="ev"><thead><tr><th>Target</th><th>Right meaning inside</th><th>Meanings offered</th><th>Single answers</th><th>Single answers correct</th></tr></thead><tbody>
      ${c.table.map((t) => `<tr><td>${Math.round(t.target * 100)}%${t.target === c.target ? ' (app)' : ''}</td><td>${Math.round(t.coverage * 100)}%</td><td>${t.setSize.toFixed(1)}</td><td>${Math.round(t.single * 100)}%</td><td>${Math.round(t.singleAcc * 100)}%</td></tr>`).join('')}</tbody></table>
      <p style="margin-top:.8rem">When Hum does commit to one meaning, it is right ${Math.round(c.singleAcc * 100)}% of the time. That is the answer a new caregiver can lean on; the rest are honest maybes.</p>
    </div>
    <div class="ev-block">
      <h2>How many sounds a new family must teach</h2>
      <p>${r.lopoReady ? 'The encoder has never heard this person.' : 'Provisional: uses the shipped encoder.'} The caregiver labels a few sounds per meaning; macro-F1 on that person's held-out sessions:</p>
      ${r.curve.ks.map((k, i) => hbar(`${k} per meaning`, r.curve.f1[i], 'ours')).join('')}
      <p style="margin-top:.8rem">A brand-new voice starts weak and improves with every example taught: from ${r.curve.f1[0].toFixed(2)} with one sound per meaning to ${r.curve.f1.at(-1).toFixed(2)} with ${r.curve.ks.at(-1)}. Hum gets useful slowly, so keep teaching it.</p>
    </div>
    <div class="ev-block">
      <h2>What did not help</h2>
      <p>Reported because the next team should not repeat it. Nothing moved the score past 0.39. The limit is the data, eight people whose recording sessions differ more than their meanings do, not the model.</p>
      <table class="ev"><thead><tr><th>Idea</th><th>Macro-F1</th></tr></thead><tbody>
      <tr><td>Large pretrained speech models, frozen (best of three)</td><td>${Math.max(...r.summary.filter((s) => s.layer !== undefined).map((s) => s.f1)).toFixed(2)}</td></tr>
      <tr><td>Contrastive loss with cross-session positives and per-band normalisation</td><td>${r.xsession.toFixed(2)}</td></tr>
      <tr><td>Asking about the least certain sounds, after ${r.active.ks.at(-1)} labels (random order: ${r.active.random.at(-1).toFixed(2)})</td><td>${r.active.active.at(-1).toFixed(2)}</td></tr>
      ${(r.attempts ?? []).map((a) => `<tr><td>${a.idea.replace(/ \(.*?\)/g, '')}</td><td>${a.f1.toFixed(2)}</td></tr>`).join('')}
      <tr><td>Hum as shipped</td><td>${r.summary.find((s) => s.name === 'Hum').f1.toFixed(2)}</td></tr></tbody></table>
    </div>`;
}

// ---------- routing ----------
function route() {
  const tab = (location.hash || '#listen').slice(1);
  const valid = ['listen', 'map', 'passport', 'evidence'].includes(tab) ? tab : 'listen';
  document.querySelectorAll('.view').forEach((v) => (v.hidden = v.id !== valid));
  document.querySelectorAll('.tabs a').forEach((a) => (a.dataset.tab === valid ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
}

async function main() {
  window.addEventListener('hashchange', route);
  route();
  bundle = await (await fetch('data/bundle.json')).json();
  buildVoices();
  // Open on the voice Hum understands best; every voice's score is on the evidence tab.
  const best = [...bundle.results.perPerson].sort((x, y) => y.hum - x.hum)[0].id;
  current = voices.find((v) => v.voice.id === best) ?? voices[0];
  fillVoicePicker();
  renderIdle();
  renderSamples();
  renderMap();
  renderPassport();
  renderEvidence();
  $('voice').addEventListener('change', (e) => selectVoice(e.target.value));
  $('ear').addEventListener('click', toggleEar);
  $('live').addEventListener('click', toggleLive);
  $('export-log').addEventListener('click', exportLog);
  renderToday();
  $('file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await hear(await loadFile(file), { playIt: true });
    } catch {
      $('answer-main').textContent = 'Could not read that file.';
      $('answer-sub').textContent = 'Try a WAV, MP3, or M4A recording.';
    }
    e.target.value = '';
  });
  $('print').addEventListener('click', () => window.print());
  $('export-voice').addEventListener('click', exportVoice);
  $('import-voice').addEventListener('change', (e) => {
    if (e.target.files[0]) importVoice(e.target.files[0]);
    e.target.value = '';
  });
  $('new-meaning').addEventListener('submit', (e) => {
    e.preventDefault();
    teach($('new-meaning-input').value);
    $('new-meaning-input').value = '';
  });
  loadModel().catch(() => {});
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
}

main();
