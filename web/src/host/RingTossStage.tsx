// Reindeer Ring Toss on the TV (and the host console): the arena itself. The reindeer is drawn from
// the game's own pegs (the antler tips ARE the scoring zones), each ring flies on the same flight
// the server used (engines/projectile) from the moment it was thrown, by the room's clock, and stays
// where it landed in its thrower's colour. Scores and rings left beside it.
import { useEffect, useRef, useState } from 'react';
import type { Launch } from '../core/minigames/engines/gesture';
import { flightAt } from '../core/minigames/engines/projectile';
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
        <linearGradient id="rt-snow" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#ffffff" /><stop offset="1" stopColor="#d9e5f3" /></linearGradient>
        {/* a ringer's front half: the same ring, cut along its middle (no seams) */}
        {ringers.map(({ t, at }) => <clipPath key={t.n} id={`rt-front-${instance}-${t.n}`}><rect x={0} y={at.sy} width={W} height={H} /></clipPath>)}
      </defs>
      <image className="rt-backdrop" href={BACKDROP} x={0} y={0} width={W} height={H} preserveAspectRatio="xMidYMid slice" />
      {/* misses behind the reindeer, whole ringers (their backs show behind the antler), the reindeer, then
          each ringer's front half over the antler, then the misses in front */}
      {misses.filter((t) => t.z >= DEER_Z).map((t) => <Ring key={t.n} at={toScreen(t.x, t.z)} color={color(t)} practice={t.stage === 'practice'} ground />)}
      {ringers.map(({ t, at }) => <Ring key={`b${t.n}`} at={at} color={color(t)} practice={t.stage === 'practice'} ringer />)}
      <FlightsBehind key={`b${instance}`} throws={flying} nodes={flightNodes} participants={participants} />
      <Reindeer pegs={v.pegs} compact={compact} />
      {ringers.map(({ t, at }) => <g key={`f${t.n}`} clipPath={`url(#rt-front-${instance}-${t.n})`}><Ring at={at} color={color(t)} practice={t.stage === 'practice'} ringer /></g>)}
      {ringers.map(({ t, at }) => <Burst key={`s${t.n}`} at={at} color={RING_COLORS[color(t)]} points={t.points} />)}
      {misses.filter((t) => t.z < DEER_Z).map((t) => <Ring key={t.n} at={toScreen(t.x, t.z)} color={color(t)} practice={t.stage === 'practice'} ground />)}
      {/* rings in the air: moved every frame without React */}
      <Flights key={instance} throws={flying} startOf={startOf} offset={offset} flight={v.flight} participants={participants} ends={ringerEnds} nodes={flightNodes} />
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

/** A missed ring lands, then goes up in a puff of smoke (only ringers stay): gone after this long. */
const MISS_FADE_MS = 1500;

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
/** A throw that reaches the reindeer goes over its antlers (this high at its depth), then comes down;
 *  no arc goes higher than ARC_MAX (the top of the screen). */
const CLEAR_H = 1.63, ARC_MAX = 1.55;
/** A ring's size: near the thrower RING_R, smaller with distance (k squared: it reads as further away). */
const RING_R = 54.7;
export const ringR = (k: number) => RING_R * k * k;
/** A ringer sits there a moment, then bursts into stars in its thrower's colour and is gone. */
const RINGER_MS = 1600;

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
function Ring({ at, color, practice = false, ground = false, ringer = false }: {
  at: { sx: number; sy: number; k: number }; color: number; practice?: boolean; ground?: boolean; ringer?: boolean;
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
      {ground && <Puff sx={sx} sy={sy} r={r} color={RING_COLORS[color]} />}
    </g>
  );
}

/** A ringer's moment: it bursts into stars in its thrower's colour, and its points float up. */
function Burst({ at, color, points }: { at: { sx: number; sy: number; k: number }; color: string; points: number }) {
  const r = Math.max(ringR(at.k), 24) * 1.3, n = 10;   // a good burst even far away
  const star = (cx: number, cy: number, R: number) => Array.from({ length: 10 }, (_, i) => {
    const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? R * 0.45 : R;
    return `${(cx + Math.cos(a) * rr).toFixed(1)},${(cy + Math.sin(a) * rr).toFixed(1)}`;
  }).join(' ');
  return (
    <g className="rt-burst" aria-hidden>
      {Array.from({ length: n }, (_, i) => {
        const a = (i / n) * Math.PI * 2 + 0.3;
        return <polygon key={i} points={star(at.sx, at.sy, r * (0.42 + (i % 2) * 0.16))} fill={color} stroke="#fff" strokeWidth={1.5}
          style={{ ['--dx' as string]: `${Math.cos(a) * r * 2.6}px`, ['--dy' as string]: `${Math.sin(a) * r * 1.8 - r}px` }} />;
      })}
      {points > 0 && <text x={at.sx} y={at.sy - r * 0.6} textAnchor="middle" className="rt-plus" fill={color}>+{points}</text>}
    </g>
  );
}

/** A missed ring disintegrates: it breaks into bits in its thrower's colour that scatter, under a puff of
 *  grey-blue smoke (grey, not white: it lands on snow). */
function Puff({ sx, sy, r: ring, color }: { sx: number; sy: number; r: number; color: string }) {
  const n = 8, r = Math.max(ring, 34) * 1.6;   // a good cloud even for a ring far away
  return (
    <g className="rt-puff" aria-hidden>
      {Array.from({ length: 9 }, (_, i) => {
        const a = (i / 9) * Math.PI * 2 + 0.2;
        return <rect key={`b${i}`} className="rt-bit" x={sx + Math.cos(a) * ring * 0.8 - 3} y={sy + Math.sin(a) * ring * 0.35 - 4} width={11} height={8} rx={3}
          fill={color} style={{ ['--dx' as string]: `${Math.cos(a) * r * 1.6}px`, ['--dy' as string]: `${Math.sin(a) * r * 0.7 - r * 0.9}px` }} />;
      })}
      {Array.from({ length: n }, (_, i) => {
        const a = (i / n) * Math.PI * 2;
        const x = sx + Math.cos(a) * r * 0.8, y = sy + Math.sin(a) * r * 0.35;
        return <circle key={i} cx={x} cy={y} r={r * (0.34 + (i % 3) * 0.08)} fill={i % 2 ? '#c3cddc' : '#d9e1ec'} stroke="#8396b0" strokeWidth={2.5}
          style={{ ['--dx' as string]: `${Math.cos(a) * r * 0.5}px`, animationDelay: `${(i % 3) * 40}ms` }} />;
      })}
      <circle cx={sx} cy={sy - r * 0.1} r={r * 0.5} fill="#ced8e5" stroke="#8396b0" strokeWidth={2.5} />
    </g>
  );
}

/** How long a ring is in the air on screen: a throw that reaches the reindeer takes a little longer (its
 *  arc goes up over the antlers), so its drop down behind it is unhurried (drawing only: the room's landing and score don't change). */
export const visMs = (t: { x: number; z: number; ms: number }) => (Math.hypot(t.x, t.z) >= DEER_Z - 0.12 ? Math.round(t.ms * 1.2) : t.ms);

/** How high a ring is, `u` of the way through its flight (drawing only: where it lands is the room's).
 *  One smooth curve, its top at `apex` of the way: sin(pi * u^p). A ringer rises over its tip and
 *  settles onto it; a throw that reaches the reindeer clears its antlers (as far as the screen allows)
 *  and comes down behind it at a steady rate; a short one is the plain arc. */
function arcHeight(t: RingThrow, u: number, plain: number, endH: number): number {
  const curve = (apex: number) => Math.sin(Math.PI * u ** (Math.log(0.5) / Math.log(apex)));
  if (endH > 0) return 0.4 * curve(0.5) + endH * u * u * (3 - 2 * u);
  const range = Math.hypot(t.x, t.z);
  if (range >= DEER_Z - 0.12) {
    const apex = 0.58, at = Math.min(0.95, DEER_Z / range);
    const needed = CLEAR_H / Math.max(0.2, Math.sin(Math.PI * at ** (Math.log(0.5) / Math.log(apex))));
    return Math.min(ARC_MAX, needed) * curve(apex);
  }
  return plain;
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

function Flights({ throws, startOf, offset, flight, participants, ends, nodes }: {
  throws: RingThrow[]; startOf: (t: RingThrow) => number; offset: number; flight: FlightParams; participants: string[];
  /** A ringer's flight ends on its peg's tip: where (x, z) and how high. */
  ends: Map<number, { x: number; z: number; h: number }>;
  /** Shared with FlightsBehind (its back copies). */
  nodes: { current: FlightNodes };
}) {
  const live = useRef({ throws, startOf, offset, flight, ends });
  live.current = { throws, startOf, offset, flight, ends };
  const place = (t: RingThrow, at: number) => {
    const n = nodes.current.get(t.n);
    if (!n?.ring || !n.shadow) return;
    // live.current: the frame loop outlives renders, so it must read this render's starts (a stale
    // closure timed a replay's ring #1 from the last game's ring #1: it was "already down")
    const u = Math.min(1, Math.max(0, (at - live.current.startOf(t)) / visMs(t)));
    const f = flightAt({ angle: t.angle, power: t.power } as Launch, u * t.ms, live.current.flight);
    const end = live.current.ends.get(t.n);
    const x = t.x * u + (end ? (end.x - t.x) * u : 0), z = t.z * u + (end ? (end.z - t.z) * u : 0);
    const hh = arcHeight(t, u, f.h, end?.h ?? 0);
    const c = toScreen(x, z, hh), g = toScreen(x, z, 0);
    // past the reindeer: behind it (the copy in the back layer shows, the front one hides)
    const behind = z > DEER_Z + 0.01;
    n.ring.style.visibility = n.shadow.style.visibility = behind ? 'hidden' : '';
    if (n.back) {
      n.back.style.visibility = behind ? '' : 'hidden';
      n.back.setAttribute('cx', String(c.sx)); n.back.setAttribute('cy', String(c.sy));
      n.back.setAttribute('rx', String(ringR(c.k))); n.back.setAttribute('ry', String(ringR(c.k) * 0.42)); n.back.setAttribute('stroke-width', String(ringR(c.k) * 0.22));
    }
    const r = ringR(c.k);
    n.ring.setAttribute('cx', String(c.sx)); n.ring.setAttribute('cy', String(c.sy));
    n.ring.setAttribute('rx', String(r)); n.ring.setAttribute('ry', String(r * 0.42)); n.ring.setAttribute('stroke-width', String(r * 0.22));
    n.shadow.setAttribute('cx', String(g.sx)); n.shadow.setAttribute('cy', String(g.sy));
    const lift = Math.min(1, hh / CLEAR_H);
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
