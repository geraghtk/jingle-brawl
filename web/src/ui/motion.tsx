// The motion kit's two moving parts (the rest is ui/motion.css): paper confetti, and a number that
// counts up to its value. Both are for show only: nothing reads them back.
import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import './motion.css';

const PAPER = ['#ff3366', '#ffc21a', '#18b88a', '#3b9cf5', '#ffffff', '#b565f0'];

/** Where each piece of paper falls: fixed by `seed`, so a re-render does not reshuffle the air. */
export function confettiPieces(n: number, seed = 1): { x: number; w: number; t: number; d: number; sway: number; spin: number; c: string }[] {
  let s = (seed * 2654435761) >>> 0 || 1;
  const r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  return Array.from({ length: n }, (_, i) => ({
    x: Math.round(r() * 1000) / 10, w: 8 + Math.round(r() * 10), t: 2.2 + Math.round(r() * 20) / 10, d: Math.round(r() * 14) / 10,
    sway: Math.round(r() * 160 - 80), spin: Math.round(360 + r() * 900) * (r() < 0.5 ? -1 : 1), c: PAPER[i % PAPER.length],
  }));
}

/** Paper falling over the box it is in (give that box `position: relative`). `forever`: it keeps falling. */
export function Confetti({ n = 60, seed = 1, forever = false }: { n?: number; seed?: number; forever?: boolean }) {
  const pieces = useMemo(() => confettiPieces(n, seed), [n, seed]);
  return (
    <div className={`mo-confetti${forever ? ' forever' : ''}`} aria-hidden data-testid="confetti">
      {pieces.map((p, i) => (
        <i key={i} style={{ ['--x']: `${p.x}%`, ['--w']: `${p.w}px`, ['--t']: `${p.t}s`, ['--d']: `${p.d}s`, ['--sway']: `${p.sway}px`, ['--spin']: `${p.spin}deg`, ['--c']: p.c } as CSSProperties} />
      ))}
    </div>
  );
}

/** The number a count-up shows `ms` after it started (eases out: quick at first, settling on the value). */
export function countAt(from: number, to: number, ms: number, over: number): number {
  if (over <= 0 || ms >= over) return to;
  const k = 1 - (1 - Math.max(0, ms) / over) ** 3;
  return Math.round(from + (to - from) * k);
}

/** A number that counts up to `value` (from `from`, or from what it showed last). Screen readers get the value itself. */
export function CountUp({ value, from, ms = 900, delay = 0, prefix = '', className }: { value: number; from?: number; ms?: number; delay?: number; prefix?: string; className?: string }) {
  const start = useRef(from ?? value);
  const [shown, setShown] = useState(from ?? value);
  useEffect(() => {
    const a = start.current;
    if (a === value) { setShown(value); return; }
    let raf = 0;
    const t0 = performance.now() + delay;
    const step = (t: number) => {
      const n = countAt(a, value, t - t0, ms);
      setShown(n);
      if (t - t0 < ms) raf = requestAnimationFrame(step); else start.current = value;
    };
    raf = requestAnimationFrame(step);
    return () => { cancelAnimationFrame(raf); start.current = value; };
  }, [value, ms, delay]);
  return <span className={className} aria-label={`${prefix}${value}`}><span aria-hidden>{prefix}{shown}</span></span>;
}
