// Little sound effects, made on the spot with Web Audio (no files to load). Muted per device (the
// choice is remembered); silent until the page has been touched, as browsers require.

export type Cue = 'jump' | 'right' | 'wrong' | 'pass' | 'doom' | 'tick';

const KEY = 'jb_muted';
let ctx: AudioContext | null = null;

export function muted(): boolean {
  try { return localStorage.getItem(KEY) === '1'; } catch { return false; }
}
export function setMuted(m: boolean): void {
  try { localStorage.setItem(KEY, m ? '1' : '0'); } catch { /* private mode: muting just isn't remembered */ }
}

/** Each cue: notes as [frequency Hz, start s, length s, wave]. */
const CUES: Record<Cue, [number, number, number, OscillatorType][]> = {
  jump: [[520, 0, 0.09, 'square'], [780, 0.09, 0.12, 'square']],
  right: [[660, 0, 0.08, 'triangle'], [990, 0.08, 0.16, 'triangle']],
  wrong: [[220, 0, 0.18, 'sawtooth'], [180, 0.12, 0.2, 'sawtooth']],
  pass: [[880, 0, 0.06, 'triangle'], [660, 0.06, 0.06, 'triangle'], [440, 0.12, 0.1, 'triangle']],
  doom: [[160, 0, 0.35, 'sawtooth'], [110, 0.25, 0.5, 'sawtooth'], [70, 0.6, 0.7, 'square']],
  tick: [[1200, 0, 0.03, 'square']],
};

export function play(cue: Cue): void {
  if (muted() || typeof window === 'undefined') return;
  try {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return;
    ctx = ctx || new AC();
    if (ctx.state === 'suspended') void ctx.resume();
    const t0 = ctx.currentTime;
    for (const [f, at, len, wave] of CUES[cue]) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = wave;
      o.frequency.setValueAtTime(f, t0 + at);
      g.gain.setValueAtTime(0.0001, t0 + at);
      g.gain.exponentialRampToValueAtTime(0.18, t0 + at + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + at + len);
      o.connect(g).connect(ctx.destination);
      o.start(t0 + at);
      o.stop(t0 + at + len + 0.02);
    }
  } catch { /* no sound is fine */ }
}

/** A short buzz on phones that can (Android; iPhones ignore it). */
export function buzz(pattern: number | number[]): void {
  try { navigator.vibrate?.(pattern); } catch { /* fine */ }
}
