// The duel wheel: every option as a slice, spun to the one the host's spin landed on. Every
// screen animates the same spin from the same server-time start, so the TV and the phones stop
// together; a screen that joins late just shows where it stopped.
import { useEffect, useState } from 'react';
import { SPIN_MS } from '../core/duelgame';
import type { WheelSpin } from '../core/duelgame';
import { GAMES } from '../core/party';
import './wheel.css';

const FILLS = ['#ff3366', '#18b88a', '#ffc53d', '#38b6e6', '#0d1018', '#8b5cf6'];
const INK_ON = new Set(['#ffc53d']); // dark text on light slices
const TURNS = 6;

export function DuelWheel({ spin, offset, size = 300 }: { spin: WheelSpin; offset: number; size?: number }) {
  const n = spin.options.length;
  const seg = 360 / n;
  const final = TURNS * 360 - (spin.index + 0.5) * seg;
  const spunAt = typeof spin.spunAt === 'number' ? spin.spunAt : Date.now() + offset;
  const left = Math.max(0, SPIN_MS - (Date.now() + offset - spunAt));
  const reduced = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const [angle, setAngle] = useState(left > 300 && !reduced ? 0 : final);
  const [done, setDone] = useState(angle === final);
  useEffect(() => {
    if (angle === final) return;
    const raf = requestAnimationFrame(() => setAngle(final)); // start the transition after first paint
    const id = setTimeout(() => setDone(true), left);
    return () => { cancelAnimationFrame(raf); clearTimeout(id); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const fill = (i: number) => FILLS[(n % FILLS.length === 1 && i === n - 1 ? i + 1 : i) % FILLS.length];
  const slice = (i: number) => {
    const a0 = ((i * seg - 90) * Math.PI) / 180, a1 = (((i + 1) * seg - 90) * Math.PI) / 180;
    const large = seg > 180 ? 1 : 0;
    return `M0 0 L${(96 * Math.cos(a0)).toFixed(2)} ${(96 * Math.sin(a0)).toFixed(2)} A96 96 0 ${large} 1 ${(96 * Math.cos(a1)).toFixed(2)} ${(96 * Math.sin(a1)).toFixed(2)} Z`;
  };
  const label = (o: WheelSpin['options'][0]) => {
    const text = o.kind === 'app' && o.app ? `${GAMES[o.app]?.emoji ?? ''} ${o.name}` : o.name;
    return text.length > 17 ? `${text.slice(0, 16)}…` : text;
  };
  const winner = spin.options[spin.index];
  return (
    <div className={`duel-wheel${done ? ' stopped' : ' spinning'}`} style={{ width: size }} data-testid="duel-wheel" aria-live="polite">
      <div className="wheel-pointer" aria-hidden />
      <svg viewBox="-100 -100 200 200" width={size} height={size} role="img" aria-label={done ? `The wheel landed on ${winner.name}` : 'The wheel is spinning'}>
        <g className="wheel-disc" style={{ transform: `rotate(${angle}deg)`, transitionDuration: `${left}ms` }}>
          <circle r="99" fill="#0d1018" />
          {spin.options.map((o, i) => {
            const mid = i * seg + seg / 2;
            const f = fill(i);
            const flip = mid > 180; // left half: turn the label so it reads left-to-right, not upside down
            return (
              <g key={i} className={done && i === spin.index ? 'wheel-win' : undefined}>
                <path d={slice(i)} fill={f} stroke="#0d1018" strokeWidth="1.6" />
                <text transform={`rotate(${mid - 90}) translate(56 0)${flip ? ' rotate(180)' : ''}`} textAnchor="middle" dominantBaseline="middle"
                  fill={INK_ON.has(f) ? '#0d1018' : '#fff'} fontSize={n > 8 ? 8 : 9.5} fontWeight={700}>{label(o)}</text>
              </g>
            );
          })}
          <circle r="13" fill="#fff" stroke="#0d1018" strokeWidth="3" />
        </g>
      </svg>
      <div className="wheel-result" data-testid="wheel-result">{done ? winner.name : 'Spinning…'}</div>
    </div>
  );
}
