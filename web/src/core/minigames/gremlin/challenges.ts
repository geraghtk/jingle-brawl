// The holder's challenge before they may pass the Gremlin on. A registry, so more can be added
// (perfect_timing and memory_flash are next, M3); quick_math is the first. The answer is the
// server's: it is checked there and never shown on any screen but as the prompt.
import type { Rng } from '../../rng';

export interface Challenge { kind: string; prompt: string; answer: number; issuedAt: number }

export interface ChallengeKind {
  kind: string;
  label: string;
  make(rng: Rng, now: number): Challenge;
  check(c: Challenge, answer: unknown): boolean;
}

const int = (rng: Rng, lo: number, hi: number) => lo + Math.floor(rng() * (hi - lo + 1));

export const quickMath: ChallengeKind = {
  kind: 'quick_math',
  label: 'Quick math',
  make(rng, now) {
    const op = int(rng, 0, 2);
    if (op === 0) { const a = int(rng, 2, 20), b = int(rng, 2, 20); return { kind: 'quick_math', prompt: `${a} + ${b}`, answer: a + b, issuedAt: now }; }
    if (op === 1) { const a = int(rng, 6, 25), b = int(rng, 2, a - 1); return { kind: 'quick_math', prompt: `${a} - ${b}`, answer: a - b, issuedAt: now }; }
    const a = int(rng, 2, 9), b = int(rng, 2, 9);
    return { kind: 'quick_math', prompt: `${a} × ${b}`, answer: a * b, issuedAt: now };
  },
  check: (c, answer) => typeof answer === 'number' && Number.isFinite(answer) && answer === c.answer,
};

export const CHALLENGES: Record<string, ChallengeKind> = { quick_math: quickMath };
