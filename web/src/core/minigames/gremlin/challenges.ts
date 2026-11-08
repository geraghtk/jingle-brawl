// The holder's challenge before they may pass the Gremlin on. A registry: a game draws each new
// challenge from the kinds its host chose. The answer is the server's: it is checked there, and the
// holder's phone is the only screen that sees the challenge itself (`view`).
//   quick_math      an arithmetic problem; type the answer
//   memory_flash    a short row of symbols flashes up, then hides; tap them back in order
//   perfect_timing  a marker swings back and forth; stop it inside the band. The phone says when it
//                   tapped (server time); the server recomputes where the marker was then from its
//                   own start and speed, trusting that time only within a short latency window.
import type { Rng } from '../../rng';

export interface Challenge {
  kind: string;
  issuedAt: number;
  prompt: string;            // quick_math's problem ('' for the others)
  answer: unknown;           // secret: the expected answer (quick_math: a number; memory_flash: indices)
  params?: TimingParams | MemoryParams;
}
export interface TimingParams { startAt: number; periodMs: number; band: [number, number] }
export interface MemoryParams { sequence: number[]; showMs: number }

export interface ChallengeKind {
  kind: string;
  label: string;            // for the TV: "Solving a quick math problem"
  make(rng: Rng, now: number): Challenge;
  /** Normalise what a phone sent; null if it isn't an answer of this kind. */
  parse(raw: unknown): unknown | null;
  check(c: Challenge, answer: unknown, now: number): boolean;
  /** What the holder's phone shows (never the TV's, never anyone else's). */
  view(c: Challenge): Record<string, unknown>;
  /** A right answer at `now`, for the simulated hands (null: not right now, e.g. the marker is off the band). */
  solve(c: Challenge, now: number): unknown | null;
  /** A wrong one. */
  miss(c: Challenge, now: number): unknown;
}

const int = (rng: Rng, lo: number, hi: number) => lo + Math.floor(rng() * (hi - lo + 1));

export const quickMath: ChallengeKind = {
  kind: 'quick_math',
  label: 'a quick math problem',
  make(rng, now) {
    const op = int(rng, 0, 2);
    const c = (prompt: string, answer: number): Challenge => ({ kind: 'quick_math', prompt, answer, issuedAt: now });
    if (op === 0) { const a = int(rng, 2, 20), b = int(rng, 2, 20); return c(`${a} + ${b}`, a + b); }
    if (op === 1) { const a = int(rng, 6, 25), b = int(rng, 2, a - 1); return c(`${a} - ${b}`, a - b); }
    const a = int(rng, 2, 9), b = int(rng, 2, 9);
    return c(`${a} × ${b}`, a * b);
  },
  parse: (raw) => (typeof raw === 'number' && Number.isFinite(raw) ? raw : null),
  check: (c, answer) => answer === c.answer,
  view: (c) => ({ prompt: c.prompt }),
  solve: (c) => c.answer,
  miss: (c) => (c.answer as number) + 1,
};

/** The symbols memory_flash shows (the phone draws them as stickers; alt text = the emoji). */
export const SYMBOLS = ['🎁', '🎄', '🦌', '🎅', '🧸', '🎲'];   // all six are sticker icons (ui/icons.tsx)
export const MEMORY_LENGTH = 4;
export const MEMORY_SHOW_MS = 2600;

export const memoryFlash: ChallengeKind = {
  kind: 'memory_flash',
  label: 'a memory test',
  make(rng, now) {
    const sequence: number[] = [];
    while (sequence.length < MEMORY_LENGTH) {
      const n = int(rng, 0, SYMBOLS.length - 1);
      if (n !== sequence.at(-1)) sequence.push(n);        // no symbol twice in a row: easier to read at a glance
    }
    return { kind: 'memory_flash', prompt: '', answer: sequence, issuedAt: now, params: { sequence, showMs: MEMORY_SHOW_MS } };
  },
  parse: (raw) => (Array.isArray(raw) && raw.length <= 12 && raw.every((n) => Number.isInteger(n) && n >= 0 && n < SYMBOLS.length) ? raw as number[] : null),
  check: (c, answer) => {
    const want = c.answer as number[], got = answer as number[];
    return got.length === want.length && got.every((n, i) => n === want[i]);
  },
  view: (c) => ({ sequence: (c.params as MemoryParams).sequence, showMs: (c.params as MemoryParams).showMs, length: MEMORY_LENGTH }),
  solve: (c) => c.answer,
  miss: (c) => (c.answer as number[]).map((n, i) => (i === 0 ? (n + 1) % SYMBOLS.length : n)),
};

/** perfect_timing: how long before the marker starts (the phone has it by then), how much of a tap's
 *  reported time the server will take on trust, and a little give at the band's edges. */
export const TIMING_LEAD_MS = 700;
export const TIMING_LATENCY_MS = 700;
export const TIMING_TOLERANCE = 0.03;
export const TIMING_BAND = 0.18;

/** Where the marker is at `t` (0 = left, 1 = right): it swings at a steady speed, end to end and back. */
export function markerAt(p: TimingParams, t: number): number | null {
  if (t < p.startAt) return null;
  const phase = ((t - p.startAt) % p.periodMs) / p.periodMs;
  return phase < 0.5 ? phase * 2 : 2 - phase * 2;
}

/** The tap time the server uses: the phone's own (server-clock) time if it is believable (not in the
 *  future, not older than the latency window), else the moment the press arrived. */
export function tapTime(claimed: unknown, now: number): number {
  return typeof claimed === 'number' && Number.isFinite(claimed) && claimed <= now + 50 && claimed >= now - TIMING_LATENCY_MS ? claimed : now;
}

export const perfectTiming: ChallengeKind = {
  kind: 'perfect_timing',
  label: 'a timing test',
  make(rng, now) {
    const periodMs = int(rng, 1300, 1800);
    const lo = 0.12 + rng() * (1 - 0.24 - TIMING_BAND);
    const params: TimingParams = { startAt: now + TIMING_LEAD_MS, periodMs, band: [lo, lo + TIMING_BAND] };
    return { kind: 'perfect_timing', prompt: '', answer: null, issuedAt: now, params };
  },
  parse: (raw) => {
    const r = raw as { at?: unknown } | null;
    return r && typeof r === 'object' ? { at: typeof r.at === 'number' && Number.isFinite(r.at) ? r.at : null } : null;
  },
  check: (c, answer, now) => {
    const p = c.params as TimingParams;
    const x = markerAt(p, tapTime((answer as { at: number | null }).at, now));
    return x !== null && x >= p.band[0] - TIMING_TOLERANCE && x <= p.band[1] + TIMING_TOLERANCE;
  },
  view: (c) => ({ ...(c.params as TimingParams) }),
  solve(c, now) {
    // the latest moment within the latency window when the marker was in the band's middle part
    const p = c.params as TimingParams;
    for (let t = now; t >= Math.max(p.startAt, now - TIMING_LATENCY_MS + 10); t -= 10) {
      const x = markerAt(p, t);
      if (x !== null && x > p.band[0] + 0.02 && x < p.band[1] - 0.02) return { at: t };
    }
    return null;
  },
  miss(c, now) {
    const p = c.params as TimingParams;
    for (let t = now; t >= now - TIMING_LATENCY_MS + 10; t -= 10) {
      const x = markerAt(p, t);
      if (x === null || x < p.band[0] - 0.1 || x > p.band[1] + 0.1) return { at: t };
    }
    return { at: p.startAt - 1 };
  },
};

export const CHALLENGES: Record<string, ChallengeKind> = {
  quick_math: quickMath, memory_flash: memoryFlash, perfect_timing: perfectTiming,
};
export const ALL_CHALLENGES = Object.keys(CHALLENGES);

/** A new challenge of one of the game's kinds (a different kind from the last one when there is a choice). */
export function nextChallenge(kinds: string[], rng: Rng, now: number, last?: string): Challenge {
  const pool = kinds.length > 1 && last ? kinds.filter((k) => k !== last) : kinds;
  return CHALLENGES[pool[Math.floor(rng() * pool.length)]].make(rng, now);
}
