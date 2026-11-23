// Sound effects, made on the spot with Web Audio (no files to load). Muted per device (the choice is
// remembered); silent until the page has been touched, as browsers require.
//
// Each cue is a few notes played on small instruments (`VOICES`): a chime, a mallet, a plucked string,
// brass, a drum, a gust of air, struck tin. They are built from sine partials, filtered noise and
// filtered saws with real envelopes, through a little room reverb, so they sound like toys and bells
// rather than a 1980s computer (Kevin, 2026-11-18: the bare square and sawtooth beeps were "too 8 bit").
// `cueNotes` is the score, kept apart from the Web Audio so it can be checked without a browser.

export type Cue = 'jump' | 'right' | 'wrong' | 'pass' | 'doom' | 'tick' | 'fanfare' | 'ding' | 'thud' | 'go' | 'horn' | 'clank' | 'bell' | 'whoosh';
export type Voice = 'chime' | 'mallet' | 'pluck' | 'brass' | 'drum' | 'air' | 'tin';
/** A note: instrument, frequency (Hz), start (s), length (s), loudness (0 to 1, default 1). */
export type Note = [Voice, number, number, number, number?];

const KEY = 'jb_muted';

export function muted(): boolean {
  try { return localStorage.getItem(KEY) === '1'; } catch { return false; }
}
export function setMuted(m: boolean): void {
  try { localStorage.setItem(KEY, m ? '1' : '0'); } catch { /* private mode: muting just isn't remembered */ }
}

const CUES: Record<Cue, Note[]> = {
  // the Gremlin hops; a right answer; a wrong one; it is passed on; it goes berserk
  jump: [['mallet', 587, 0, 0.16, 0.8], ['mallet', 880, 0.08, 0.22]],
  right: [['chime', 784, 0, 0.45, 0.8], ['chime', 1175, 0.09, 0.6]],
  wrong: [['pluck', 233, 0, 0.22], ['pluck', 175, 0.13, 0.32], ['drum', 82, 0.13, 0.2, 0.6]],
  pass: [['air', 900, 0, 0.2, 0.7], ['mallet', 880, 0.02, 0.14, 0.6], ['mallet', 587, 0.1, 0.2, 0.7]],
  doom: [['brass', 147, 0, 0.5], ['brass', 104, 0.28, 0.65], ['drum', 55, 0.62, 0.9], ['air', 260, 0.6, 0.8, 0.8], ['brass', 73, 0.62, 0.9, 0.9]],
  // a clock's tick; a win
  tick: [['mallet', 1568, 0, 0.07, 0.75], ['tin', 2600, 0, 0.03, 0.25]],
  fanfare: [['brass', 523, 0, 0.13], ['brass', 659, 0.12, 0.13], ['brass', 784, 0.24, 0.13], ['brass', 1047, 0.36, 0.6], ['brass', 784, 0.36, 0.6, 0.6], ['brass', 659, 0.36, 0.6, 0.5], ['chime', 2093, 0.36, 0.9, 0.5]],
  // a sleigh bell (a ringer), a ring in the snow, the start, time's up
  ding: [['chime', 1319, 0, 0.6], ['chime', 1976, 0.015, 0.4, 0.45]],
  thud: [['drum', 105, 0, 0.18]],
  go: [['brass', 523, 0, 0.13, 0.9], ['brass', 784, 0.12, 0.4], ['chime', 1568, 0.12, 0.7, 0.4]],
  horn: [['brass', 233, 0, 0.7], ['brass', 175, 0, 0.7, 0.8], ['brass', 117, 0.04, 0.68, 0.8]],
  // tin on tin, the ring bell, a punch through the air
  clank: [['tin', 430, 0, 0.16], ['drum', 150, 0, 0.09, 0.6]],
  bell: [['chime', 660, 0, 1.1], ['tin', 1320, 0, 0.12, 0.35], ['chime', 660, 0.24, 1.3], ['tin', 1320, 0.24, 0.12, 0.35]],
  whoosh: [['air', 700, 0, 0.2]],
};

/** A cue's notes, `pitch` times higher (a run of ringers climbs). */
export function cueNotes(cue: Cue, pitch = 1): Note[] {
  return CUES[cue].map(([v, f, at, len, level]) => [v, f * pitch, at, len, level ?? 1]);
}

// ---- the instruments ----------------------------------------------------------------------------
interface Rig { ctx: AudioContext; out: GainNode; wet: GainNode; noise: AudioBuffer }
let rig: Rig | null = null;

function getRig(): Rig | null {
  if (rig) return rig;
  const AC = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AC) return null;
  const ctx = new AC();
  const master = ctx.createDynamicsCompressor();          // several cues at once don't clip
  master.threshold.value = -14; master.ratio.value = 4; master.attack.value = 0.003; master.release.value = 0.2;
  master.connect(ctx.destination);
  const out = ctx.createGain(); out.gain.value = 0.55; out.connect(master);
  // a small room: a second of decaying noise as the reverb's impulse
  const sr = ctx.sampleRate, n = Math.floor(sr * 1.1);
  const impulse = ctx.createBuffer(2, n, sr);
  for (let ch = 0; ch < 2; ch++) { const d = impulse.getChannelData(ch); for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n) ** 3.2; }
  const room = ctx.createConvolver(); room.buffer = impulse;
  const wet = ctx.createGain(); wet.gain.value = 0.2;
  wet.connect(room).connect(master);
  const noise = ctx.createBuffer(1, sr, sr);
  const nd = noise.getChannelData(0); for (let i = 0; i < sr; i++) nd[i] = Math.random() * 2 - 1;
  rig = { ctx, out, wet, noise };
  return rig;
}

/** A gain that swells to `peak` in `attack` s and dies away by `t + len`, wired to the output and the room. */
function env(r: Rig, t: number, len: number, peak: number, attack = 0.004, room = 1): GainNode {
  const g = r.ctx.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(Math.max(0.0002, peak), t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + Math.max(len, attack + 0.02));
  g.connect(r.out);
  if (room > 0) { const s = r.ctx.createGain(); s.gain.value = room; g.connect(s).connect(r.wet); }
  return g;
}
function osc(r: Rig, type: OscillatorType, f: number, t: number, len: number, to: AudioNode, detune = 0): OscillatorNode {
  const o = r.ctx.createOscillator();
  o.type = type; o.frequency.setValueAtTime(f, t); o.detune.value = detune;
  o.connect(to); o.start(t); o.stop(t + len + 0.05);
  return o;
}
function hiss(r: Rig, t: number, len: number, to: AudioNode): void {
  const s = r.ctx.createBufferSource();
  s.buffer = r.noise; s.loop = true; s.connect(to); s.start(t, Math.random() * 0.5); s.stop(t + len + 0.05);
}

const VOICES: Record<Voice, (r: Rig, f: number, t: number, len: number, level: number) => void> = {
  // a small bell: sine partials a bell has (not a neat harmonic stack), the high ones dying first
  chime(r, f, t, len, level) {
    [[1, 1, 1], [2.76, 0.42, 0.55], [5.4, 0.22, 0.32], [8.93, 0.1, 0.18]].forEach(([ratio, gain, life]) => {
      if (f * ratio < 15000) osc(r, 'sine', f * ratio, t, len * life, env(r, t, len * life, 0.3 * gain * level, 0.002));
    });
  },
  // a wooden bar: the note, a short octave-and-a-bit knock on top
  mallet(r, f, t, len, level) {
    osc(r, 'sine', f, t, len, env(r, t, len, 0.42 * level, 0.002, 0.7));
    osc(r, 'sine', f * 4, t, len * 0.25, env(r, t, len * 0.25, 0.12 * level, 0.001, 0.4));
    osc(r, 'sine', f * 9.2, t, 0.03, env(r, t, 0.03, 0.05 * level, 0.001, 0));
  },
  // a plucked string: bright at first, mellowing as it rings
  pluck(r, f, t, len, level) {
    const lp = r.ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 2;
    lp.frequency.setValueAtTime(Math.min(f * 9, 9000), t); lp.frequency.exponentialRampToValueAtTime(f * 1.6, t + len);
    lp.connect(env(r, t, len, 0.5 * level, 0.003, 0.6));
    osc(r, 'sawtooth', f, t, len, lp); osc(r, 'triangle', f * 0.5, t, len, lp);
  },
  // brass: two saws a hair apart behind a filter that opens as the note speaks, a little vibrato once it holds
  brass(r, f, t, len, level) {
    const lp = r.ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 1.2;
    lp.frequency.setValueAtTime(f * 1.4, t); lp.frequency.linearRampToValueAtTime(Math.min(f * 6, 7000), t + Math.min(0.07, len * 0.4)); lp.frequency.linearRampToValueAtTime(f * 3, t + len);
    const g = r.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(0.2 * level, t + Math.min(0.035, len * 0.3));
    g.gain.setValueAtTime(0.2 * level, t + len * 0.75); g.gain.exponentialRampToValueAtTime(0.0001, t + len + 0.06);
    g.connect(r.out); const send = r.ctx.createGain(); send.gain.value = 1.1; g.connect(send).connect(r.wet);
    lp.connect(g);
    const voices = [osc(r, 'sawtooth', f, t, len + 0.06, lp, -7), osc(r, 'sawtooth', f, t, len + 0.06, lp, 7)];
    if (len > 0.3) {
      const lfo = r.ctx.createOscillator(), depth = r.ctx.createGain();
      lfo.frequency.value = 5.5; depth.gain.setValueAtTime(0, t); depth.gain.linearRampToValueAtTime(f * 0.006, t + 0.25);
      lfo.connect(depth); voices.forEach((o) => depth.connect(o.frequency)); lfo.start(t); lfo.stop(t + len + 0.1);
    }
  },
  // a soft drum: a tone that drops as it lands, and the slap of the skin
  drum(r, f, t, len, level) {
    const o = osc(r, 'sine', f * 2.4, t, len, env(r, t, len, 0.7 * level, 0.002, 0.5));
    o.frequency.exponentialRampToValueAtTime(f, t + 0.06);
    const lp = r.ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 1400;
    lp.connect(env(r, t, 0.04, 0.3 * level, 0.001, 0.3)); hiss(r, t, 0.04, lp);
  },
  // air: a band of noise sweeping down (something going past)
  air(r, f, t, len, level) {
    const bp = r.ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 1.1;
    bp.frequency.setValueAtTime(f * 2.2, t); bp.frequency.exponentialRampToValueAtTime(Math.max(80, f * 0.45), t + len);
    bp.connect(env(r, t, len, 0.5 * level, len * 0.3, 0.8)); hiss(r, t, len, bp);
  },
  // struck tin: partials that don't agree with each other, and the tick of the strike
  tin(r, f, t, len, level) {
    [[1, 1], [1.52, 0.7], [2.33, 0.55], [3.71, 0.35], [5.18, 0.2]].forEach(([ratio, gain], i) => {
      if (f * ratio < 15000) osc(r, i ? 'sine' : 'triangle', f * ratio, t, len / (1 + i * 0.25), env(r, t, len / (1 + i * 0.25), 0.2 * gain * level, 0.001, 0.6));
    });
    const hp = r.ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 3500;
    hp.connect(env(r, t, 0.025, 0.2 * level, 0.001, 0.2)); hiss(r, t, 0.025, hp);
  },
};

/** `pitch` multiplies every note (a run of ringers climbs). */
export function play(cue: Cue, pitch = 1): void {
  if (muted() || typeof window === 'undefined') return;
  try {
    const r = getRig();
    if (!r) return;
    if (r.ctx.state === 'suspended') void r.ctx.resume();
    const t0 = r.ctx.currentTime + 0.005;
    for (const [voice, f, at, len, level] of cueNotes(cue, pitch)) VOICES[voice](r, f, t0 + at, len, level ?? 1);
  } catch { /* no sound is fine */ }
}

/** A short buzz on phones that can (Android; iPhones ignore it). */
export function buzz(pattern: number | number[]): void {
  try { navigator.vibrate?.(pattern); } catch { /* fine */ }
}
