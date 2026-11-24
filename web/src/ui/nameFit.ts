// Type that fits the name. A name drawn large (a duel corner, the verdict, the Party MVP) gets a
// scale from its own shape: the longest word must fit one line and the whole name the lines the
// place has, so a long name comes out smaller and whole instead of breaking mid-word, running off
// the card or ending in "...". A name breaks only at a space or after a hyphen.
// The place says how many average letters fit one of its lines at full size (`chars`); widths are
// estimated (a capital and an "m" are wide, an "i" is narrow; one average letter is about 0.54 of
// the font size in the display face), so no screen has to measure anything: chars = room / (size * 0.54).
// DOM-free: `ui/FittedName.tsx` draws it.

export interface FitSpec {
  /** Average letters that fit one line of the place at full size. */
  chars: number;
  /** Lines the place has room for (default 2). */
  lines?: number;
  /** The smallest scale worth reading (default 0.4); only a name that cannot fit even then may break inside a word. */
  min?: number;
}

export interface Fit {
  /** Multiply the place's font size by this (1 = full size). */
  scale: number;
  /** False only for a name no scale down to `min` can fit: the one case that may break inside a word. */
  whole: boolean;
}

const WIDE = /[mwMW@%]/;
const NARROW = /[iljtfrI.,:;'!|()\[\] -]/;

/** How wide a string sets, in average lower-case letters. */
export function nameWidth(s: string): number {
  let w = 0;
  for (const ch of s) {
    if (ch === ' ') w += 0.5;
    else if (WIDE.test(ch)) w += 1.5;
    else if (NARROW.test(ch)) w += 0.6;
    else if (/\d/.test(ch)) w += 1.05;
    else if (ch !== ch.toLowerCase()) w += 1.1;           // a capital
    else if (ch.charCodeAt(0) > 0x2e7f) w += 1.9;         // CJK, emoji: about a square
    else w += 1;
  }
  return w;
}

/** The pieces a name may break between: at spaces, and after a hyphen ("Mary-" "Jane"). */
export function nameWords(name: string): string[] {
  return name.trim().split(/\s+/).flatMap((w) => w.split(/(?<=-)(?=.)/)).filter(Boolean);
}

/** Lines a name takes when a line holds `cap` letters; Infinity when one word alone is too long. */
function linesAt(name: string, cap: number): number {
  let lines = 1, used = 0;
  for (const raw of name.trim().split(/\s+/)) {
    const parts = raw.split(/(?<=-)(?=.)/);
    parts.forEach((part, i) => {
      const w = nameWidth(part);
      if (w > cap) { lines = Infinity; return; }
      const gap = used > 0 && i === 0 ? 0.5 : 0;         // a space before a new word, none after a hyphen
      if (used > 0 && used + gap + w > cap) { lines += 1; used = w; }
      else used += gap + w;
    });
  }
  return lines;
}

const STEP = 0.04;

/** The scale at which `name` fits a place of `chars` letters by `lines` lines. */
export function fitName(name: string, spec: FitSpec): Fit {
  const lines = Math.max(1, spec.lines ?? 2);
  const min = spec.min ?? 0.4;
  const text = (name || '').trim();
  if (!text) return { scale: 1, whole: true };
  for (let scale = 1; scale >= min - 1e-9; scale -= STEP) {
    if (linesAt(text, spec.chars / scale) <= lines) return { scale: Math.round(scale * 100) / 100, whole: true };
  }
  // still too long at the smallest size: it takes more lines; only a word longer than a line has to break
  const whole = linesAt(text, spec.chars / min) !== Infinity;
  return { scale: Math.round(min * 100) / 100, whole };
}
