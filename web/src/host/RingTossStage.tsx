// Reindeer Ring Toss on the TV (and the host console): the arena itself. The reindeer is drawn from
// the game's own pegs (the antler tips ARE the scoring zones), each ring flies to where the server
// landed it (engines/projectile: real gravity, drag, a spin) from the moment it was thrown, by the room's clock, and stays
// where it landed in its thrower's colour. Scores and rings left beside it. The scene's maths and
// measurements are ui/ringtoss/scene.ts, its effects ui/ringtoss/effects.tsx, the rings in the air
// ui/ringtoss/Flights.tsx.
import { useEffect, useRef, useState } from 'react';
import type { Peg } from '../core/minigames/engines/targets';
import { finalCount, leaderOf, STREAK_MIN, streakAt, streakCall, streakPitch } from '../core/minigames/ringtoss/flair';
import { ringView } from '../core/minigames/ringtoss/view';
import type { RingThrow, RingView } from '../core/minigames/ringtoss/view';
import { useServerNow, useServerOffset } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { colorIndex, RING_COLORS, ringColor } from '../ui/ringtoss/colors';
import { Burst, Confetti, Puff, Ring, shade, Snow } from '../ui/ringtoss/effects';
import { Flights, FlightsBehind } from '../ui/ringtoss/Flights';
import type { FlightNodes } from '../ui/ringtoss/Flights';
import { BACKDROP, DEER, DEER_Z, H, MISS_FADE_MS, NOSE, restHeight, restOn, RINGER_MS, ringR, TIP_H, toScreen, visMs, W } from '../ui/ringtoss/scene';
import { SetPips } from '../ui/ringtoss/SetPips';
import { play } from '../ui/sound';
import '../ui/ringtoss.css';
import { useHost } from './HostContext';

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

export function RingScene({ v, participants, offset, instance = '', onLand, nameOf }: {
  v: RingView; participants: string[]; offset: number; instance?: string; onLand?: (t: RingThrow) => void;
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
    <svg className="rt-scene" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The reindeer and the rings thrown so far" data-testid="ringtoss-scene">
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
