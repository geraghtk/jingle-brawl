// Reindeer Ring Toss on the TV (and the host console): the arena itself. The reindeer is drawn from
// the game's own pegs (the antler tips ARE the scoring zones), each ring flies to where the server
// landed it (engines/projectile: real gravity, drag, a spin) from the moment it was thrown, by the room's clock, and stays
// where it landed in its thrower's colour. Scores and rings left beside it.
import { useEffect, useRef, useState } from 'react';
import { arcAt, arcFor } from '../core/minigames/engines/projectile';
import type { FlightParams } from '../core/minigames/engines/projectile';
import type { Peg } from '../core/minigames/engines/targets';
import { useServerOffset } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { play } from '../ui/sound';
import '../ui/ringtoss.css';
import { useHost } from './HostContext';

export interface RingThrow { n: number; pid: string; at: number; stage: 'practice' | 'throwing' | 'tiebreak'; angle: number; power: number; x: number; z: number; ms: number; pegId: string | null; points: number }
export interface RingView {
  stage: 'practice' | 'throwing' | 'tiebreak' | 'done';
  stageEndsAt: number; pool: string[]; left: Record<string, number>; scores: Record<string, number>; tiebreakRound: number;
  rings: number; practiceThrows: number; flight: FlightParams; pegs: Peg[]; throws: RingThrow[];
  me?: string; myLeft?: number; myScore?: number; nextAt?: number;
}

/** A view as it arrives: empty lists and maps don't travel (the finished game has no rings left). */
export const ringView = (raw: unknown): RingView => {
  const v = (raw || {}) as RingView;
  return { ...v, pool: v.pool || [], left: v.left || {}, scores: v.scores || {}, throws: v.throws || [], pegs: v.pegs || [] };
};

export const RING_COLORS = ['#ff3366', '#18b88a', '#8a7bff', '#ffb000', '#38b6e6', '#ff7a59', '#c04fd6', '#2f6fde', '#7cc242', '#e0457b', '#00a3a3', '#9b6b3c'];
export const colorIndex = (participants: string[], pid: string) => Math.max(0, participants.indexOf(pid)) % RING_COLORS.length;
export const ringColor = (participants: string[], pid: string) => RING_COLORS[colorIndex(participants, pid)];

/** The scene: world x (across) and z (into the scene) to the picture, with a little perspective. */
export const W = 1000, H = 620;
// The view: SX across, SZ into the scene (squeezed: the reindeer stands at depth 1.65, still big, in a
// snowdrift; the longest throw lands just behind it), HZ up. The physics is in
// world units (x, z) only; heights are only for drawing.
const SX = 1290, SZ = 86, HZ = 420;
export function toScreen(x: number, z: number, h = 0): { sx: number; sy: number; k: number } {
  const k = 1 / (1 + 0.35 * z);                       // further away: smaller
  return { sx: W / 2 + x * SX * k, sy: H - 30 - z * SZ * k - h * HZ * k, k };
}

/** When each ring's flight starts on this screen: the room's throw time, or, if the throw reached us
 *  while the ring was already in the air (the network takes a moment), when it did, so every ring
 *  flies its whole arc. A throw first seen after it landed just lies where it landed. */
function useFlightStarts(key: string, throws: RingThrow[], serverNow: number): (t: RingThrow) => number {
  const seen = useRef(new Map<string, number>());
  for (const t of throws) {
    const k = `${key}:${t.n}`;
    if (!seen.current.has(k)) seen.current.set(k, serverNow < t.at + visMs(t) ? Math.max(t.at, serverNow - 40) : t.at);
  }
  return (t) => seen.current.get(`${key}:${t.n}`) ?? t.at;
}

/** The scene's clock: a few times a second, and exactly when the next ring lands. */
function useSceneClock(offset: number, nextAt: number | null): number {
  const [now, setNow] = useState(() => Date.now() + offset);
  useEffect(() => { const id = setInterval(() => setNow(Date.now() + offset), 250); return () => clearInterval(id); }, [offset]);
  useEffect(() => {
    if (nextAt == null) return;
    const id = setTimeout(() => setNow(Date.now() + offset), Math.max(0, nextAt - (Date.now() + offset)) + 4);
    return () => clearTimeout(id);
  }, [nextAt, offset]);
  return now;
}

export function RingScene({ v, participants, offset, instance = '', compact = false, onLand }: {
  v: RingView; participants: string[]; offset: number; instance?: string; compact?: boolean; onLand?: (t: RingThrow) => void;
}) {
  const startOf = useFlightStarts(instance, v.throws, Date.now() + offset);
  const flightNodes = useRef<FlightNodes>(new Map());
  const ends = v.throws.map((t) => startOf(t) + visMs(t));
  const probe = Date.now() + offset;
  const upcoming = ends.filter((e) => e > probe);
  const now = useSceneClock(offset, upcoming.length ? Math.min(...upcoming) : null);
  const landed = v.throws.filter((t) => now >= startOf(t) + visMs(t));
  const flying = v.throws.filter((t) => now < startOf(t) + visMs(t));
  const landedN = landed.length;
  const lastLanded = landed.at(-1);
  const seenLanded = useRef(landedN);
  useEffect(() => { if (landedN > seenLanded.current && lastLanded) onLand?.(lastLanded); seenLanded.current = landedN; }, [landedN]); // eslint-disable-line react-hooks/exhaustive-deps

  const peg = (id: string) => v.pegs.find((p) => p.id === id);
  const misses = landed.filter((t) => !t.pegId && now - (startOf(t) + visMs(t)) < MISS_FADE_MS);   // a miss fades away
  const ringers = landed.filter((t) => t.pegId && peg(t.pegId) && now - (startOf(t) + visMs(t)) < RINGER_MS).map((t) => {
    const p = peg(t.pegId!)!;
    const stack = landed.filter((u) => u.pegId === t.pegId && u.n < t.n).length;
    const at = restOn(p, stack);
    return { t, at };
  });
  const color = (t: RingThrow) => colorIndex(participants, t.pid);
  // where each ringer still in the air will end up: on its tip, above the ones already there
  const ringerEnds = new Map<number, { x: number; z: number; h: number }>();
  for (const t of flying) {
    const p = t.pegId ? peg(t.pegId) : undefined;
    if (p) ringerEnds.set(t.n, { x: p.x, z: p.z, h: restHeight(p, v.throws.filter((u) => u.pegId === t.pegId && u.n < t.n).length) });
  }
  return (
    <svg className={`rt-scene${compact ? ' compact' : ''}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The reindeer and the rings thrown so far" data-testid="ringtoss-scene">
      <defs>
        {RING_COLORS.map((c, i) => (
          <linearGradient key={c} id={`rt-rg-${i}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={shade(c, 0.45)} /><stop offset="0.45" stopColor={c} /><stop offset="1" stopColor={shade(c, -0.45)} />
          </linearGradient>
        ))}
        <radialGradient id="rt-smoke"><stop offset="0" stopColor="#8e9db3" stopOpacity="0.95" /><stop offset="0.5" stopColor="#a9b6c8" stopOpacity="0.8" /><stop offset="1" stopColor="#c9d2de" stopOpacity="0" /></radialGradient>
        <radialGradient id="rt-flash"><stop offset="0" stopColor="#fff" stopOpacity="1" /><stop offset="0.4" stopColor="#fff6c8" stopOpacity="0.85" /><stop offset="1" stopColor="#fff6c8" stopOpacity="0" /></radialGradient>
        <linearGradient id="rt-snow" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#ffffff" /><stop offset="1" stopColor="#d9e5f3" /></linearGradient>
        {/* a ringer's front half: the same ring, cut along its middle (no seams) */}
        {ringers.map(({ t, at }) => <clipPath key={t.n} id={`rt-front-${instance}-${t.n}`}><rect x={0} y={at.sy} width={W} height={H} /></clipPath>)}
      </defs>
      <image className="rt-backdrop" href={BACKDROP} x={0} y={0} width={W} height={H} preserveAspectRatio="xMidYMid slice" />
      {/* misses behind the reindeer, whole ringers (their backs show behind the antler), the reindeer, then
          each ringer's front half over the antler, then the misses in front */}
      {misses.filter((t) => t.z >= DEER_Z).map((t) => <Ring key={t.n} at={toScreen(t.x, t.z)} color={color(t)} practice={t.stage === 'practice'} ground hidePuff />)}
      {ringers.map(({ t, at }) => <Ring key={`b${t.n}`} at={at} color={color(t)} practice={t.stage === 'practice'} ringer />)}
      <FlightsBehind key={`b${instance}`} throws={flying} nodes={flightNodes} participants={participants} />
      <Reindeer pegs={v.pegs} compact={compact} />
      {ringers.map(({ t, at }) => <g key={`f${t.n}`} clipPath={`url(#rt-front-${instance}-${t.n})`}><Ring at={at} color={color(t)} practice={t.stage === 'practice'} ringer /></g>)}
      {ringers.map(({ t, at }) => <Burst key={`s${t.n}`} at={at} color={RING_COLORS[color(t)]} points={t.points} />)}
      {misses.filter((t) => t.z < DEER_Z).map((t) => <Ring key={t.n} at={toScreen(t.x, t.z)} color={color(t)} practice={t.stage === 'practice'} ground />)}
      {/* the smoke of a ring that came down behind the reindeer rises in front of it, where it shows */}
      {misses.filter((t) => t.z >= DEER_Z).map((t) => { const at = toScreen(t.x, t.z); return <Puff key={`p${t.n}`} sx={at.sx} sy={at.sy} r={ringR(at.k)} color={RING_COLORS[color(t)]} />; })}
      {/* rings in the air: moved every frame without React */}
      <Flights key={instance} throws={flying} startOf={startOf} offset={offset} participants={participants} ends={ringerEnds} nodes={flightNodes} />
    </svg>
  );
}

/** Where a ringer rests: a little down its prong (stacking further down), or down the hat's cone to its
 *  brim (the ring is wider than the cone). */
function restOn(p: Peg, stack: number): { sx: number; sy: number; k: number } {
  const tip = toScreen(p.x, p.z, TIP_H[p.id] ?? 1.15);
  return { ...tip, sy: tip.sy + (p.id === 'golden' ? HAT_DROP + stack * 6 : PRONG_DROP + stack * 7) };
}
/** The same, as a height above the ground at the peg (a ringer's flight ends there, not on the ground). */
function restHeight(p: Peg, stack: number): number {
  const ground = toScreen(p.x, p.z, 0), r = restOn(p, stack);
  return (ground.sy - r.sy) / (HZ * ground.k);
}

/** A missed ring goes up in a puff of snow dust as it lands: gone after this long. */
const MISS_FADE_MS = 1300;

/** A colour lighter (amt > 0) or darker (amt < 0): the ring's shading. */
function shade(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const ch = (v: number) => Math.round(amt > 0 ? v + (255 - v) * amt : v * (1 + amt));
  return `#${[n >> 16, (n >> 8) & 255, n & 255].map((v) => ch(v).toString(16).padStart(2, '0')).join('')}`;
}

/** The reindeer (tools/art/reindeer.py, variant g seed 4, its left half mirrored so both sides score
 *  alike), standing chest-deep in a snowdrift (drawn here, in front of it) at depth 1.65: the 1088 x 791
 *  art at 0.58 from (184.5, 127.6). In the art the far prongs' tips are at (84, 32), the tall ones at
 *  (197, 100), the hat's tip at (544, 30); the pegs (ringtoss.ts PEGS) are under them, TIP_H is how high. */
const DEER = { src: 'art/ringtoss/reindeer.webp', x: 184.5, y: 127.6, w: 631, h: 458.8 };
const TIP_H: Record<string, number> = { 'left-outer': 1.329, 'right-outer': 1.329, 'left-inner': 1.181, 'right-inner': 1.181, golden: 1.333 };
const BACKDROP = 'art/ringtoss/backdrop.webp';
/** A ring on the hat slides down the cone to where the cone fills it (px below the tip); on a prong it
 *  slides a little way down. */
const HAT_DROP = 67;
const PRONG_DROP = 10;
/** Rings on the ground nearer than this are in front of the reindeer; further ones behind it. */
const DEER_Z = 1.65;
/** A ring this high is as far up as rings go (its shadow is faintest then). */
const HIGH = 1.55;
/** A ring's size: RING_R as it comes up from the bottom of the screen, about 22 at the reindeer (k^2.5: it
 *  shrinks quickly, so it reads as going away). */
const RING_R = 70;
export const ringR = (k: number) => RING_R * k ** 2.5;
/** A ringer bursts into stars in its thrower's colour as it lands, and is gone after this long. */
const RINGER_MS = 1300;

function Reindeer({ pegs, compact }: { pegs: Peg[]; compact: boolean }) {
  return (
    <g className="rt-deer" aria-hidden>
      <image href={DEER.src} x={DEER.x} y={DEER.y} width={DEER.w} height={DEER.h} />
      <Drift />
      <Knobs pegs={pegs} compact={compact} />
    </g>
  );
}

/** The snowdrift the reindeer stands in (over its chest): flat-shaded like the art, a soft shadow under. */
function Drift() {
  const d = 'M300,566 C306,520 336,500 372,498 C390,474 428,462 458,474 C478,456 524,454 544,472 C578,460 616,474 630,498 C668,500 696,522 700,566 Z';
  return (
    <g className="rt-drift" transform="translate(0 36)">
      <ellipse cx={500} cy={566} rx={214} ry={16} fill="#1d2a3a" opacity={0.14} />
      <path d={d} fill="url(#rt-snow)" stroke="#8ea6c4" strokeWidth={3} strokeLinejoin="round" />
      <path d="M352,528 C392,516 430,520 452,532 M548,520 C580,512 614,516 640,532" fill="none" stroke="#c9d7ea" strokeWidth={5} strokeLinecap="round" />
      <path d="M392,492 C410,482 428,480 444,486" fill="none" stroke="#fff" strokeWidth={6} strokeLinecap="round" opacity={0.9} />
    </g>
  );
}

/** The scoring spots on the antler tips (the hat's tip is its own), with their points. */
function Knobs({ pegs, compact }: { pegs: Peg[]; compact: boolean }) {
  const knob = (p: Peg) => toScreen(p.x, p.z, TIP_H[p.id] ?? 1.15);
  return (
    <>
      {pegs.filter((p) => p.id !== 'golden').map((p) => { const k = knob(p); return <circle key={p.id} className="rt-knob" cx={k.sx} cy={k.sy} r={5} fill="#fff" stroke="#5a3418" strokeWidth={2.5} />; })}
      {!compact && pegs.map((p) => { const k = knob(p); return <text key={`l${p.id}`} x={k.sx} y={k.sy - 16} textAnchor="middle" className="rt-peg-pts">{p.points}</text>; })}
    </>
  );
}

/** A ring with some body: shaded top to bottom, a darker inner edge, a shine along the top; on the
 *  ground, a soft shadow under it. (A ringer is drawn whole behind the antler and again, cut along its
 *  middle, over it: so it hangs round the prong with no seams.) */
function Ring({ at, color, practice = false, ground = false, ringer = false, hidePuff = false }: {
  at: { sx: number; sy: number; k: number }; color: number; practice?: boolean; ground?: boolean; ringer?: boolean;
  /** Its smoke is drawn elsewhere (a ring behind the reindeer: the smoke rises in front, where it shows). */
  hidePuff?: boolean;
}) {
  const r = ringR(at.k), ry = r * 0.42, sw = r * 0.24;
  const { sx, sy } = at;
  const top = (rx: number, ryy: number) => `M${sx - rx},${sy} A${rx},${ryy} 0 0 1 ${sx + rx},${sy}`;
  return (
    <g opacity={practice ? 0.55 : 1}>
      <g className={`rt-ring${ground ? ' miss' : ringer ? ' ringer' : ''}`}>
        {ground && <ellipse cx={sx} cy={sy + sw * 0.55} rx={r * 1.04} ry={ry * 1.1} fill="none" stroke="#1d2a3a" strokeOpacity={0.22} strokeWidth={sw * 1.5} />}
        <ellipse cx={sx} cy={sy} rx={r} ry={ry} fill="none" stroke={`url(#rt-rg-${color})`} strokeWidth={sw} />
        <path d={top(r - sw * 0.38, ry - sw * 0.3)} fill="none" stroke="#000" strokeOpacity={0.22} strokeWidth={sw * 0.22} />
        <path d={`M${sx - r * 0.55},${sy - ry * 0.92} A${r},${ry} 0 0 1 ${sx + r * 0.25},${sy - ry * 1.02}`} fill="none" stroke="#fff" strokeOpacity={0.6} strokeWidth={sw * 0.25} strokeLinecap="round" />
      </g>
      {ground && !hidePuff && <Puff sx={sx} sy={sy} r={r} color={RING_COLORS[color]} />}
    </g>
  );
}

/** A ringer's moment, right as it lands: a flash, then stars burst outward in its thrower's colour, each
 *  spinning as it flies, with gold twinkles; its points rise. (Positions are offsets the CSS animates to.) */
function Burst({ at, color, points }: { at: { sx: number; sy: number; k: number }; color: string; points: number }) {
  const r = Math.max(ringR(at.k), 26) * 1.6, n = 12;   // a good burst even far away
  const star = (R: number) => {
    const pts = Array.from({ length: 10 }, (_, i) => {
      const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? R * 0.48 : R;
      return `${(Math.cos(a) * rr).toFixed(1)},${(Math.sin(a) * rr).toFixed(1)}`;
    });
    return `M${pts.join('L')}Z`;
  };
  const twinkle = (R: number) => `M0,${-R} Q0,0 ${R},0 Q0,0 0,${R} Q0,0 ${-R},0 Q0,0 0,${-R}Z`;
  return (
    <g className="rt-burst" aria-hidden transform={`translate(${at.sx.toFixed(1)} ${at.sy.toFixed(1)})`}>
      <circle className="rt-flash" r={r * 1.1} fill="url(#rt-flash)" />
      {Array.from({ length: n }, (_, i) => {
        const a = (i / n) * Math.PI * 2 + (i % 2) * 0.2, d = r * (2.2 + (i % 3) * 0.5);
        return (
          <g key={i} className="rt-fly" style={{ ['--dx' as string]: `${(Math.cos(a) * d).toFixed(1)}px`, ['--dy' as string]: `${(Math.sin(a) * d * 0.8 - r * 0.4).toFixed(1)}px`, animationDelay: `${(i % 3) * 25}ms` }}>
            <path className="rt-spin" d={star(r * (0.3 + (i % 3) * 0.08))} fill={color} stroke="#fff" strokeWidth={2} strokeLinejoin="round"
              style={{ animationDirection: i % 2 ? 'reverse' : 'normal' }} />
          </g>
        );
      })}
      {Array.from({ length: 6 }, (_, i) => {
        const a = (i / 6) * Math.PI * 2 + 0.5, d = r * (1.3 + (i % 2) * 0.9);
        return <path key={`t${i}`} className="rt-twinkle" d={twinkle(r * 0.22)} fill="#ffe27a"
          transform={`translate(${(Math.cos(a) * d).toFixed(1)} ${(Math.sin(a) * d * 0.8 - r * 0.3).toFixed(1)})`} style={{ animationDelay: `${80 + i * 45}ms` }} />;
      })}
      {points > 0 && <text y={-r * 0.7} textAnchor="middle" className="rt-plus" fill={color}>+{points}</text>}
    </g>
  );
}

/** A missed ring, right as it lands: it crumples into bits in its thrower's colour that scatter and fall,
 *  under a soft puff of snow dust (shaded grey-blue, no outlines: it lands on snow). */
function Puff({ sx, sy, r: ring, color }: { sx: number; sy: number; r: number; color: string }) {
  const r = Math.max(ring, 34) * 1.6;   // a good cloud even for a ring far away
  const blobs = [[-0.55, 0.05, 0.62], [0.5, 0.0, 0.66], [0, -0.2, 0.78], [-0.2, 0.25, 0.5], [0.3, 0.28, 0.52]];
  return (
    <g className="rt-puff" aria-hidden transform={`translate(${sx.toFixed(1)} ${sy.toFixed(1)})`}>
      {blobs.map(([x, y, s], i) => (
        <circle key={i} className="rt-smoke" cx={x * r} cy={y * r * 0.5} r={s * r} fill="url(#rt-smoke)"
          style={{ ['--rise' as string]: `${(-r * (0.5 + i * 0.12)).toFixed(1)}px`, ['--drift' as string]: `${(x * r * 0.6).toFixed(1)}px`, animationDelay: `${i * 30}ms` }} />
      ))}
      {bits(sx, sy, ring, color)}
    </g>
  );
}

/** A missed ring's pieces as it crumples: curved chips of its band, each its own size, shade, throw and
 *  spin, up and out then falling (gravity: the CSS eases each one up, then down). Seeded by where it landed,
 *  so every ring breaks differently but the same way on each render. */
function bits(sx: number, sy: number, ring: number, color: string) {
  let seed = (Math.round(sx * 7) * 73856093) ^ (Math.round(sy * 7) * 19349663);
  const rnd = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const r = Math.max(ring, 26), R = Math.max(ring, 32) * 1.4, n = 10 + Math.floor(rnd() * 5);   // R: how far they fly
  return Array.from({ length: n }, (_, i) => {
    const a = -Math.PI / 2 + (rnd() - 0.5) * 2.8;                 // mostly up, some sideways
    const v = R * (0.6 + rnd() * 1.4);                             // how hard it flies
    const up = Math.min(-R * 0.25, Math.sin(a) * v * 1.1), dx = Math.cos(a) * v * 1.3;
    const down = r * (0.3 + rnd() * 0.7);                          // where it ends, below where it started
    const rr = r * (0.5 + rnd() * 0.4), span = 0.4 + rnd() * 0.8, w = Math.max(3, r * (0.14 + rnd() * 0.12));
    const arc = `M${(rr * Math.cos(-span / 2)).toFixed(1)},${(rr * Math.sin(-span / 2)).toFixed(1)} A${rr.toFixed(1)},${rr.toFixed(1)} 0 0 1 ${(rr * Math.cos(span / 2)).toFixed(1)},${(rr * Math.sin(span / 2)).toFixed(1)}`;
    const vars = {
      ['--dx' as string]: `${dx.toFixed(1)}px`, ['--up' as string]: `${up.toFixed(1)}px`, ['--down' as string]: `${down.toFixed(1)}px`,
      ['--rot' as string]: `${Math.round((rnd() - 0.5) * 1100)}deg`, ['--dur' as string]: `${Math.round(650 + rnd() * 550)}ms`,
      ['--delay' as string]: `${Math.round(260 + rnd() * 90)}ms`,
    };
    return (
      <g key={`b${i}`} className="rt-bit-x" style={vars}>
        <g className="rt-bit-y">
          <path className="rt-bit" d={arc} transform={`rotate(${Math.round(rnd() * 360)})`} fill="none" stroke={shade(color, (rnd() - 0.5) * 0.7)}
            strokeWidth={w.toFixed(1)} strokeLinecap="round" />
        </g>
      </g>
    );
  });
}

/** How long a ring is in the air on screen: its arc's time (engines/projectile arcFor: real gravity, so a
 *  longer, higher throw is up longer), but down before the room moves on (it waits the flight's ms + 300).
 *  A ringer's arc ends on its tip. Drawing only: the room's landing and score don't change. */
export const visMs = (t: { x: number; z: number; ms: number; pegId?: string | null }) =>
  Math.min(t.ms + 250, arcFor(Math.hypot(t.x, t.z), t.pegId ? TIP_H[t.pegId] ?? 1.15 : 0).ms);

/** A ring in the air spins and wobbles as it flies (a thrown ring is never quite flat): how far it's tipped
 *  (degrees) and how open it looks (its height over its width), `ms` into its flight. Each ring its own. */
function tumble(n: number, ms: number, angle: number): { deg: number; open: number } {
  const s = ms / 1000, ph = n * 1.7;
  return { deg: angle * 40 + 12 * Math.sin(s * 9.5 + ph), open: 0.42 + 0.13 * Math.sin(s * 15 + ph * 0.6) };
}

type FlightNodes = Map<number, { ring: SVGEllipseElement | null; shadow: SVGEllipseElement | null; back?: SVGEllipseElement | null }>;

/** The copies of the rings in the air that show once they're behind the reindeer (drawn before it). */
function FlightsBehind({ throws, nodes, participants }: { throws: RingThrow[]; nodes: { current: FlightNodes }; participants: string[] }) {
  return (
    <g className="rt-flights behind">
      {throws.map((t) => (
        <ellipse key={t.n} ref={(e) => { const n = nodes.current.get(t.n); nodes.current.set(t.n, { ring: n?.ring ?? null, shadow: n?.shadow ?? null, back: e }); }}
          className="rt-ring flying" fill="none" stroke={`url(#rt-rg-${colorIndex(participants, t.pid)})`} style={{ visibility: 'hidden' }} />
      ))}
    </g>
  );
}

function Flights({ throws, startOf, offset, participants, ends, nodes }: {
  throws: RingThrow[]; startOf: (t: RingThrow) => number; offset: number; participants: string[];
  /** A ringer's flight ends on its peg's tip: where (x, z) and how high. */
  ends: Map<number, { x: number; z: number; h: number }>;
  /** Shared with FlightsBehind (its back copies). */
  nodes: { current: FlightNodes };
}) {
  const live = useRef({ throws, startOf, offset, ends });
  live.current = { throws, startOf, offset, ends };
  const place = (t: RingThrow, at: number) => {
    const n = nodes.current.get(t.n);
    if (!n?.ring || !n.shadow) return;
    // live.current: the frame loop outlives renders, so it must read this render's starts (a stale
    // closure timed a replay's ring #1 from the last game's ring #1: it was "already down")
    const dur = visMs(t), since = at - live.current.startOf(t);
    const u = Math.min(1, Math.max(0, since / dur));
    // real gravity (engines/projectile arcAt): quick out of the hand, top late, a steep drop; a ringer drops onto its tip
    const end = live.current.ends.get(t.n) ?? { x: t.x, z: t.z, h: 0 };
    const f = arcAt(arcFor(Math.hypot(end.x, end.z), end.h), u);
    const x = end.x * f.along, z = end.z * f.along, hh = f.h;
    const c = toScreen(x, z, hh), g = toScreen(x, z, 0);
    const tb = tumble(t.n, Math.max(0, since), t.angle);
    const spin = `rotate(${tb.deg.toFixed(1)} ${c.sx.toFixed(1)} ${c.sy.toFixed(1)})`;
    // past the reindeer: behind it (the copy in the back layer shows, the front one hides)
    const behind = z > DEER_Z + 0.01;
    n.ring.style.visibility = n.shadow.style.visibility = behind ? 'hidden' : '';
    if (n.back) {
      n.back.style.visibility = behind ? '' : 'hidden';
      n.back.setAttribute('cx', String(c.sx)); n.back.setAttribute('cy', String(c.sy));
      n.back.setAttribute('rx', String(ringR(c.k))); n.back.setAttribute('ry', String(ringR(c.k) * tb.open)); n.back.setAttribute('stroke-width', String(ringR(c.k) * 0.22));
      n.back.setAttribute('transform', spin);
    }
    const r = ringR(c.k);
    n.ring.setAttribute('cx', String(c.sx)); n.ring.setAttribute('cy', String(c.sy));
    n.ring.setAttribute('rx', String(r)); n.ring.setAttribute('ry', String(r * tb.open)); n.ring.setAttribute('stroke-width', String(r * 0.22));
    n.ring.setAttribute('transform', spin);
    n.shadow.setAttribute('cx', String(g.sx)); n.shadow.setAttribute('cy', String(g.sy));
    const lift = Math.min(1, hh / HIGH);
    n.shadow.setAttribute('rx', String(ringR(g.k) * 0.9 * (1 - 0.6 * lift))); n.shadow.setAttribute('ry', String(ringR(g.k) * 0.3 * (1 - 0.6 * lift)));
    n.shadow.setAttribute('opacity', String(0.28 * (1 - 0.8 * lift)));
  };
  useEffect(() => {
    let raf = 0;
    const frame = () => {
      const { throws: ts, offset: off } = live.current;
      const at = Date.now() + off;
      for (const t of ts) place(t, at);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <g className="rt-flights">
      {throws.map((t) => (
        <g key={t.n}>
          <ellipse ref={(e) => { nodes.current.set(t.n, { ...nodes.current.get(t.n), shadow: e, ring: nodes.current.get(t.n)?.ring ?? null }); if (e) place(t, Date.now() + offset); }} fill="#1d2a3a" opacity={0} />
          <ellipse ref={(e) => { nodes.current.set(t.n, { ...nodes.current.get(t.n), ring: e, shadow: nodes.current.get(t.n)?.shadow ?? null }); if (e) place(t, Date.now() + offset); }}
            className="rt-ring flying" fill="none" stroke={`url(#rt-rg-${colorIndex(participants, t.pid)})`} />
        </g>
      ))}
    </g>
  );
}

/** The TV / console: the scene, the scores, the clock. Sounds on the TV only. */
export function RingTossStage() {
  const { game, tv } = useHost();
  const offset = useServerOffset();
  const av = game.arenaView!;
  const v = ringView(av.view);
  const [now, setNow] = useState(() => Date.now() + offset);
  useEffect(() => { const id = setInterval(() => setNow(Date.now() + offset), 250); return () => clearInterval(id); }, [offset]);
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const duel: string[] = av.duelists || [];
  const order = [...av.participants].sort((a, b) => (v.scores[b] || 0) - (v.scores[a] || 0));
  const secs = Math.max(0, Math.ceil((v.stageEndsAt - now) / 1000));
  return (
    <div className="rt-stage" data-testid="ringtoss-stage">
      <div className="rt-banner" data-testid="ringtoss-stage-label">
        {v.stage === 'practice' ? `Practice throws (not scored): ${secs}s`
          : v.stage === 'throwing' ? `Throw! ${secs}s left`
          : v.stage === 'tiebreak' ? `Level! One more ring each: ${duel.map(name).join(' vs ')}`
          : 'All rings thrown'}
      </div>
      <div className="rt-layout">
        <RingScene v={v} participants={av.participants} offset={offset} instance={av.instanceId} onLand={(t) => { if (tv && t.points > 0) play('right'); }} />
        <ol className="rt-board" aria-label="Scores">
          {order.map((pid) => {
            const p = game.players?.[pid];
            return (
              <li key={pid} className={duel.includes(pid) ? 'duelist' : ''} data-testid={`ringtoss-score-${pid}`}>
                <span className="rt-swatch" style={{ borderColor: ringColor(av.participants, pid) }} />
                {p && <Avatar name={p.name} avatar={p.avatar} size={30} />}
                <span className="rt-name">{name(pid)}{duel.includes(pid) ? ' ⚔️' : ''}</span>
                <span className="rt-left" aria-label={`${v.left[pid] || 0} rings left`}>{'◯'.repeat(Math.min(10, v.left[pid] || 0))}</span>
                <span className="rt-pts">{v.scores[pid] || 0}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
