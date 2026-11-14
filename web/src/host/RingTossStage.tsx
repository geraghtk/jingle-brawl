// Reindeer Ring Toss on the TV (and the host console): the arena itself. The reindeer is drawn from
// the game's own pegs (the antler tips ARE the scoring zones), each ring flies to where the server
// landed it (engines/projectile: real gravity, drag, a spin) from the moment it was thrown, by the room's clock, and stays
// where it landed in its thrower's colour. Scores and rings left beside it.
import { useEffect, useRef, useState } from 'react';
import { arcAt, arcFor } from '../core/minigames/engines/projectile';
import type { Peg } from '../core/minigames/engines/targets';
import { finalCount, leaderOf, STREAK_MIN, streakAt, streakCall, streakPitch } from '../core/minigames/ringtoss/flair';
import { ringView } from '../core/minigames/ringtoss/view';
import type { RingThrow, RingView } from '../core/minigames/ringtoss/view';
import { useServerNow, useServerOffset } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { colorIndex, RING_COLORS, ringColor } from '../ui/ringtoss/colors';
import { DEER, TIP_H, visMs } from '../ui/ringtoss/scene';
import { SetPips } from '../ui/ringtoss/SetPips';
import { play } from '../ui/sound';
import '../ui/ringtoss.css';
import { useHost } from './HostContext';

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

export function RingScene({ v, participants, offset, instance = '', compact = false, onLand, nameOf }: {
  v: RingView; participants: string[]; offset: number; instance?: string; compact?: boolean; onLand?: (t: RingThrow) => void;
  /** Whose ring it was, under its points (the TV: a dozen colours are hard to tell apart). */
  nameOf?: (pid: string) => string;
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
  // whose ringer, and their run as it stood when it landed
  const tag = (t: RingThrow) => [nameOf ? short(nameOf(t.pid)) : '', streakCall(streakAt(landed, t.pid, t.n))].filter(Boolean).join(' · ');
  const cheer = ringers.at(-1)?.t;
  const splash = useSplash(v.stage);
  const count = finalCount(v.stage, v.stageEndsAt, now);
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
        <radialGradient id="rt-nose"><stop offset="0" stopColor="#fff3d6" stopOpacity="1" /><stop offset="0.35" stopColor="#ff5a4a" stopOpacity="0.9" /><stop offset="1" stopColor="#ff2a2a" stopOpacity="0" /></radialGradient>
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
      <Snow n={26} seed={7} far />
      <Reindeer pegs={v.pegs} cheer={cheer?.n} big={!!cheer?.bonus} />
      {ringers.map(({ t, at }) => <g key={`f${t.n}`} clipPath={`url(#rt-front-${instance}-${t.n})`}><Ring at={at} color={color(t)} practice={t.stage === 'practice'} ringer /></g>)}
      {ringers.map(({ t, at }) => <Burst key={`s${t.n}`} at={at} color={RING_COLORS[color(t)]} points={t.points} tag={tag(t)} />)}
      {ringers.filter(({ t }) => t.bonus).map(({ t }) => <Confetti key={`c${t.n}`} seed={t.n} />)}
      {/* a full set: across the top of the scene, in the thrower's colour */}
      {ringers.filter(({ t }) => t.bonus).map(({ t }) => (
        <text key={`set${t.n}`} x={W / 2} y={92} textAnchor="middle" className="rt-fullset" fill={RING_COLORS[color(t)]} data-testid="ringtoss-fullset">Full set! +{t.bonus}</text>
      ))}
      {misses.filter((t) => t.z < DEER_Z).map((t) => <Ring key={t.n} at={toScreen(t.x, t.z)} color={color(t)} practice={t.stage === 'practice'} ground />)}
      {/* the smoke of a ring that came down behind the reindeer rises in front of it, where it shows */}
      {misses.filter((t) => t.z >= DEER_Z).map((t) => { const at = toScreen(t.x, t.z); return <Puff key={`p${t.n}`} sx={at.sx} sy={at.sy} r={ringR(at.k)} color={RING_COLORS[color(t)]} />; })}
      {/* rings in the air: moved every frame without React */}
      <Flights key={instance} throws={flying} startOf={startOf} offset={offset} participants={participants} ends={ringerEnds} nodes={flightNodes} />
      <Snow n={12} seed={31} />
      {count != null && <text key={`n${count}`} x={W / 2} y={H / 2 + 20} textAnchor="middle" className="rt-count" data-testid="ringtoss-count">{count}</text>}
      {splash && <text key={`sp${splash.key}`} x={W / 2} y={H / 2 - 10} textAnchor="middle" className="rt-splash" data-testid="ringtoss-splash">{splash.text}</text>}
    </svg>
  );
}

const short = (name: string) => (name.length > 14 ? `${name.slice(0, 13)}…` : name);

/** The big word across the scene as the game moves on: the scored round starts, a tiebreak, time's up.
 *  Only for a change this screen saw happen (a TV opened mid-round shows none). */
const SPLASH_MS = 1500;
function useSplash(stage: RingView['stage']): { text: string; key: number } | null {
  const was = useRef(stage);
  const [splash, setSplash] = useState<{ text: string; key: number } | null>(null);
  useEffect(() => {
    const from = was.current;
    was.current = stage;
    if (from === stage) return;
    const text = stage === 'throwing' ? 'Go!' : stage === 'tiebreak' ? 'Tiebreak!' : stage === 'done' ? 'Time!' : '';
    if (!text) return;
    setSplash((s) => ({ text, key: (s?.key ?? 0) + 1 }));
    const id = setTimeout(() => setSplash(null), SPLASH_MS);
    return () => clearTimeout(id);
  }, [stage]);
  return splash;
}

/** A small seeded random stream (the same snow and confetti on every render). */
function seeded(seed: number): () => number {
  let s = seed | 0;
  return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Snow drifting down the scene, always: small slow flakes far off (behind the reindeer), a few big
 *  quick ones close to. Each falls on its own time and sways its own way (CSS). */
function Snow({ n, seed, far = false }: { n: number; seed: number; far?: boolean }) {
  const flakes = useRef<{ x: number; r: number; dur: number; delay: number; sway: number }[]>();
  if (!flakes.current) {
    const rnd = seeded(seed);
    flakes.current = Array.from({ length: n }, () => ({
      x: rnd() * W, r: far ? 2 + rnd() * 2.5 : 4.5 + rnd() * 4, dur: (far ? 11 : 6.5) + rnd() * 6, delay: -rnd() * 17, sway: (rnd() - 0.5) * (far ? 90 : 170),
    }));
  }
  return (
    <g className={`rt-snowfall${far ? ' far' : ''}`} aria-hidden>
      {flakes.current.map((f, i) => (
        <circle key={i} className="rt-flake" cx={f.x.toFixed(1)} cy={-12} r={f.r.toFixed(1)} fill="#fff"
          style={{ ['--dur' as string]: `${f.dur.toFixed(2)}s`, ['--delay' as string]: `${f.delay.toFixed(2)}s`, ['--sway' as string]: `${f.sway.toFixed(0)}px` }} />
      ))}
    </g>
  );
}

/** A full set: confetti in every ring colour falls the height of the scene. */
function Confetti({ seed }: { seed: number }) {
  const rnd = seeded(seed * 977 + 5);
  return (
    <g className="rt-confetti" aria-hidden>
      {Array.from({ length: 46 }, (_, i) => {
        const w = 9 + rnd() * 9;
        return (
          <g key={i} className="rt-conf-fall" transform={`translate(${(rnd() * W).toFixed(0)} -30)`}
            style={{ ['--dur' as string]: `${(1.5 + rnd() * 0.9).toFixed(2)}s`, ['--delay' as string]: `${(rnd() * 0.35).toFixed(2)}s`, ['--sway' as string]: `${((rnd() - 0.5) * 220).toFixed(0)}px` }}>
            <rect className="rt-conf" x={-w / 2} y={-w / 4} width={w} height={w / 2} rx={1.5} fill={RING_COLORS[i % RING_COLORS.length]}
              style={{ ['--rot' as string]: `${Math.round((rnd() - 0.5) * 1400)}deg` }} />
          </g>
        );
      })}
    </g>
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

/** The reindeer's nose in the scene (the art's (543, 541)): it lights up for a ringer. */
const NOSE = { x: DEER.x + 543 * 0.58, y: DEER.y + 541 * 0.58, rx: 40, ry: 24 };

/** The reindeer, pleased with a ringer (`cheer`: the ring's number): a little hop and its nose glows; a
 *  full set (`big`) gets a bigger one. The hop is small, so the tips stay under their rings. */
function Reindeer({ pegs, cheer, big = false }: { pegs: Peg[]; cheer?: number; big?: boolean }) {
  const body = useRef<SVGGElement>(null);
  useEffect(() => {
    if (cheer == null) return;
    const up = big ? -11 : -5;
    try {
      body.current?.animate?.(
        [{ transform: 'translateY(0)' }, { transform: `translateY(${up}px)`, offset: 0.35 }, { transform: 'translateY(0)', offset: 0.7 }, { transform: `translateY(${up / 3}px)`, offset: 0.85 }, { transform: 'translateY(0)' }],
        { duration: big ? 620 : 380, easing: 'ease-out' },
      );
    } catch { /* a browser without it: the nose still glows */ }
  }, [cheer]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <g className="rt-deer" aria-hidden>
      <g ref={body}>
        <image href={DEER.src} x={DEER.x} y={DEER.y} width={DEER.w} height={DEER.h} />
        {cheer != null && <ellipse key={cheer} className={`rt-nose${big ? ' big' : ''}`} cx={NOSE.x} cy={NOSE.y} rx={NOSE.rx} ry={NOSE.ry} fill="url(#rt-nose)" data-testid="ringtoss-nose" />}
      </g>
      <Drift />
      <Knobs pegs={pegs} />
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

/** The scoring spots: a white dot on each antler tip and on the hat's tip. Every one is worth 1: no numbers. */
function Knobs({ pegs }: { pegs: Peg[] }) {
  const knob = (p: Peg) => toScreen(p.x, p.z, TIP_H[p.id] ?? 1.15);
  return (
    <>
      {pegs.map((p) => { const k = knob(p); return <circle key={p.id} className="rt-knob" cx={k.sx} cy={k.sy} r={5} fill="#fff" stroke="#5a3418" strokeWidth={2.5} />; })}
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
function Burst({ at, color, points, tag = '' }: { at: { sx: number; sy: number; k: number }; color: string; points: number; tag?: string }) {
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
      {tag && <text y={-r * 0.7 - 34} textAnchor="middle" className="rt-plus rt-who" fill={color} data-testid="ringtoss-who">{tag}</text>}
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

/** A missed ring shatters: its band snaps into 4 to 6 chunks (together the whole ring), each a curved slab
 *  with jagged broken ends, that burst outward from where they were on the ring, up and then falling
 *  (gravity: the CSS eases each one up, then down), spinning; and a few small splinters. Seeded by where it
 *  landed, so every ring breaks differently but the same way on each render. */
function bits(sx: number, sy: number, ring: number, color: string) {
  let seed = (Math.round(sx * 7) * 73856093) ^ (Math.round(sy * 7) * 19349663);
  const rnd = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const r = Math.max(ring, 22), w = Math.max(5, r * 0.24), ro = r + w / 2, R = Math.max(ring, 32) * 1.4;   // R: how far they fly
  const pt = (rad: number, a: number, cx: number) => `${(rad * Math.cos(a) - cx).toFixed(1)},${(rad * Math.sin(a)).toFixed(1)}`;
  /** A chunk of the band from angle -span/2 to span/2, centred on itself, its two ends broken jaggedly. */
  const slab = (span: number) => {
    const a0 = -span / 2, a1 = span / 2, cx = ro - w / 2, jag = (w / ro) * 0.9, pts: string[] = [];
    for (let k = 0; k <= 6; k++) pts.push(pt(ro, a0 + (span * k) / 6, cx));                       // the outer edge
    for (const f of [0.3, 0.55, 0.8]) pts.push(pt(ro - w * f, a1 + (rnd() - 0.5) * 2 * jag, cx));   // a broken end
    for (let k = 6; k >= 0; k--) pts.push(pt(ro - w, a0 + (span * k) / 6, cx));                   // the inner edge
    for (const f of [0.8, 0.55, 0.3]) pts.push(pt(ro - w * f, a0 + (rnd() - 0.5) * 2 * jag, cx));   // the other broken end
    return pts.join(' ');
  };
  /** A splinter: a thin jagged sliver. */
  const splinter = (len: number) => {
    const t = len * (0.25 + rnd() * 0.2);
    return [[-len / 2, 0], [-len * 0.1, -t], [len * 0.15, -t * 0.4], [len / 2, 0], [len * 0.05, t * 0.7]].map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  };
  const fly = (a: number, hard: number) => {
    const v = R * hard;
    return {
      ['--dx' as string]: `${(Math.cos(a) * v * 1.2).toFixed(1)}px`,
      ['--up' as string]: `${Math.min(-R * 0.3, (Math.sin(a) * 0.5 - 0.6) * v).toFixed(1)}px`,   // outward, and up
      ['--down' as string]: `${(r * (0.3 + rnd() * 0.8)).toFixed(1)}px`,
      ['--rot' as string]: `${Math.round((rnd() - 0.5) * 900)}deg`, ['--dur' as string]: `${Math.round(700 + rnd() * 500)}ms`,
      ['--delay' as string]: `${Math.round(260 + rnd() * 60)}ms`,
    };
  };
  // the chunks: random shares of the whole circle, each where it was on the ring (squashed like the ring)
  const k = 4 + Math.floor(rnd() * 3), cuts = Array.from({ length: k }, () => 0.6 + rnd()), total = cuts.reduce((a, b) => a + b, 0);
  let at = rnd() * Math.PI * 2;
  const chunks = cuts.map((c, i) => {
    const span = (c / total) * Math.PI * 2 - 0.12, mid = at + span / 2 + 0.06;
    at += (c / total) * Math.PI * 2;
    const ox = Math.cos(mid) * (ro - w / 2), oy = Math.sin(mid) * (ro - w / 2) * 0.42;
    return (
      <g key={`c${i}`} transform={`translate(${ox.toFixed(1)} ${oy.toFixed(1)})`}>
        <g className="rt-bit-x" style={fly(Math.atan2(oy, ox), 0.5 + rnd() * 0.9)}>
          <g className="rt-bit-y">
            <polygon className="rt-bit" points={slab(span)} transform={`rotate(${((mid * 180) / Math.PI).toFixed(0)}) scale(1 ${(0.5 + Math.abs(Math.cos(mid)) * 0.5).toFixed(2)})`}
              fill={shade(color, (rnd() - 0.5) * 0.4)} stroke={shade(color, -0.45)} strokeWidth={1.2} strokeLinejoin="miter" />
          </g>
        </g>
      </g>
    );
  });
  const splinters = Array.from({ length: 4 + Math.floor(rnd() * 4) }, (_, i) => {
    const a = rnd() * Math.PI * 2, ox = Math.cos(a) * ro * 0.9, oy = Math.sin(a) * ro * 0.38;
    return (
      <g key={`s${i}`} transform={`translate(${ox.toFixed(1)} ${oy.toFixed(1)})`}>
        <g className="rt-bit-x" style={fly(a, 0.8 + rnd() * 1.1)}>
          <g className="rt-bit-y">
            <polygon className="rt-bit" points={splinter(w * (0.8 + rnd() * 1.2))} transform={`rotate(${Math.round(rnd() * 360)})`}
              fill={shade(color, (rnd() - 0.5) * 0.6)} stroke={shade(color, -0.45)} strokeWidth={0.8} />
          </g>
        </g>
      </g>
    );
  });
  return [...chunks, ...splinters];
}

/** A ring in the air spins and wobbles as it flies (a thrown ring is never quite flat): how far it's tipped
 *  (degrees) and how open it looks (its height over its width), `ms` into its flight. Each ring its own. */
function tumble(n: number, ms: number, angle: number): { deg: number; open: number } {
  const s = ms / 1000, ph = n * 1.7;
  return { deg: angle * 40 + 12 * Math.sin(s * 9.5 + ph), open: 0.42 + 0.13 * Math.sin(s * 15 + ph * 0.6) };
}

/** A ring's trail: how faint each copy is, and how far apart in time. */
const TRAIL = [0.34, 0.22, 0.13, 0.06];
const TRAIL_GAP_MS = 34;

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
  /** Each ring's trail: fainter copies of it where it was a moment ago (worked out, not remembered). */
  const trails = useRef(new Map<number, (SVGEllipseElement | null)[]>());
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
    (trails.current.get(t.n) || []).forEach((e, i) => {
      if (!e) return;
      const ago = since - (i + 1) * TRAIL_GAP_MS;
      const pf = arcAt(arcFor(Math.hypot(end.x, end.z), end.h), Math.max(0, ago) / dur);
      const pz = end.z * pf.along;
      if (ago <= 0 || u >= 1 || pz > DEER_Z + 0.01) { e.style.visibility = 'hidden'; return; }
      const pc = toScreen(end.x * pf.along, pz, pf.h), pt = tumble(t.n, ago, t.angle), pr = ringR(pc.k);
      e.style.visibility = '';
      e.setAttribute('cx', String(pc.sx)); e.setAttribute('cy', String(pc.sy));
      e.setAttribute('rx', String(pr)); e.setAttribute('ry', String(pr * pt.open)); e.setAttribute('stroke-width', String(pr * 0.2));
      e.setAttribute('transform', `rotate(${pt.deg.toFixed(1)} ${pc.sx.toFixed(1)} ${pc.sy.toFixed(1)})`);
    });
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
          {TRAIL.map((o, i) => (
            <ellipse key={i} ref={(e) => { const a = trails.current.get(t.n) || []; a[i] = e; trails.current.set(t.n, a); }}
              className="rt-trail" fill="none" stroke={RING_COLORS[colorIndex(participants, t.pid)]} opacity={o} style={{ visibility: 'hidden' }} />
          ))}
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
  const now = useServerNow();
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const duel: string[] = av.duelists || [];
  const order = [...av.participants].sort((a, b) => (v.scores[b] || 0) - (v.scores[a] || 0));
  const secs = secsLeft(v.stageEndsAt, now);
  const leader = leaderOf(v.scores, av.participants);
  const hurry = v.stage === 'throwing' && secs <= 10;
  // out loud on the TV: the start, the last five seconds, time's up
  const count = finalCount(v.stage, v.stageEndsAt, now);
  const heard = useRef({ stage: v.stage, count });
  useEffect(() => {
    const was = heard.current;
    heard.current = { stage: v.stage, count };
    if (!tv) return;
    if (was.stage !== v.stage) { if (v.stage === 'throwing') play('go'); else if (v.stage === 'done') play('horn'); else if (v.stage === 'tiebreak') play('jump'); }
    else if (count != null && count !== was.count) play('tick', count === 1 ? 1.5 : 1);
  }, [v.stage, count, tv]);
  // a ringer chimes higher the longer its thrower's run; a miss thumps in the snow
  const landSound = (t: RingThrow) => {
    if (!tv) return;
    if (t.bonus) play('fanfare');
    else if (t.pegId) play('ding', streakPitch(streakAt(v.throws, t.pid, t.n)));
    else play('thud');
  };
  return (
    <div className="rt-stage" data-testid="ringtoss-stage">
      <div className={`rt-banner${hurry ? ' hurry' : ''}`} data-testid="ringtoss-stage-label">
        {v.stage === 'practice' ? `Practice throws (not scored): ${secs}s`
          : v.stage === 'throwing' ? `Throw! ${secs}s left`
          : v.stage === 'tiebreak' ? `Level! One more ring each: ${duel.map(name).join(' vs ')}`
          : 'All rings thrown'}
      </div>
      <div className="rt-layout">
        <RingScene v={v} participants={av.participants} offset={offset} instance={av.instanceId} onLand={landSound} nameOf={name} />
        <ol className="rt-board" aria-label="Scores">
          {order.map((pid) => {
            const p = game.players?.[pid];
            const run = streakAt(v.throws, pid);
            return (
              <li key={pid} className={`${duel.includes(pid) ? 'duelist' : ''}${pid === leader ? ' leader' : ''}`} data-testid={`ringtoss-score-${pid}`}>
                {pid === leader && <Crown />}
                <span className="rt-swatch" style={{ borderColor: ringColor(av.participants, pid) }} />
                {p && <Avatar name={p.name} avatar={p.avatar} size={30} />}
                <span className="rt-name">{name(pid)}{duel.includes(pid) ? ' ⚔️' : ''}</span>
                <SetPips v={v} pid={pid} color={ringColor(av.participants, pid)} />
                {run >= STREAK_MIN && <span className="rt-run" aria-label={`${run} in a row`} data-testid={`ringtoss-run-${pid}`}><Flame />{run}</span>}
                <span key={v.scores[pid] || 0} className={`rt-pts${v.scores[pid] ? ' pop' : ''}`}>{v.scores[pid] || 0}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

/** The leader's crown and a run's flame on the board: drawn here (no emoji, so they look the same everywhere). */
function Crown() {
  return (
    <svg className="rt-crown" viewBox="0 0 24 18" role="img" aria-label="In the lead" data-testid="ringtoss-leader">
      <path d="M2,16 L1,4 L7,9 L12,1 L17,9 L23,4 L22,16 Z" fill="#ffc21a" stroke="#a86a00" strokeWidth={1.6} strokeLinejoin="round" />
      <circle cx={12} cy={11.5} r={1.8} fill="#ff3366" />
    </svg>
  );
}
function Flame() {
  return (
    <svg className="rt-flame" viewBox="0 0 16 20" aria-hidden>
      <path d="M8,1 C9,5 14,7 14,13 C14,17 11,19 8,19 C5,19 2,17 2,13 C2,10 4,9 5,6 C6,8 7,8 8,1 Z" fill="#ff7a1a" stroke="#c23b00" strokeWidth={1.2} strokeLinejoin="round" />
      <path d="M8,10 C9,12 11,13 11,15 C11,17 9.5,18 8,18 C6.5,18 5,17 5,15 C5,13.5 7,13 8,10 Z" fill="#ffd84a" />
    </svg>
  );
}
