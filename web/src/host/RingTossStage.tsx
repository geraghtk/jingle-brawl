// Reindeer Ring Toss on the TV (and the host console): the arena itself. The reindeer is drawn from
// the game's own pegs (the antler tips ARE the scoring zones), each ring flies on the same flight
// the server used (engines/projectile) from the moment it was thrown, by the room's clock, and stays
// where it landed in its thrower's colour. Scores and rings left beside it.
import { useEffect, useState } from 'react';
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
export const ringColor = (participants: string[], pid: string) => RING_COLORS[Math.max(0, participants.indexOf(pid)) % RING_COLORS.length];

/** The scene: world x (across) and z (into the scene) to the picture, with a little perspective. */
export const W = 1000, H = 620;
const SX = 1500, SZ = 400;
export function toScreen(x: number, z: number, h = 0): { sx: number; sy: number; k: number } {
  const k = 1 / (1 + 0.35 * z);                       // further away: smaller
  return { sx: W / 2 + x * SX * k, sy: H - 30 - z * SZ * k - h * 420 * k, k };
}

export function RingScene({ v, participants, now, compact = false }: { v: RingView; participants: string[]; now: number; compact?: boolean }) {
  const head = toScreen(0, 1.42), nose = toScreen(0, 1.36);
  const peg = (id: string) => v.pegs.find((p) => p.id === id)!;
  const knob = (p: Peg) => toScreen(p.x, p.z, 0.18);
  const antler = (side: 'left' | 'right') => {
    const outer = knob(peg(`${side}-outer`)), inner = knob(peg(`${side}-inner`));
    const base = toScreen(side === 'left' ? -0.07 : 0.07, 1.44, 0.1);
    const mid = toScreen(side === 'left' ? -0.16 : 0.16, 1.52, 0.14);
    return `M${base.sx},${base.sy} Q${mid.sx},${mid.sy} ${inner.sx},${inner.sy} M${mid.sx},${mid.sy} L${outer.sx},${outer.sy}`;
  };
  const gold = knob(peg('golden'));
  const goldBase = toScreen(0, 1.47, 0.12);
  const landed = v.throws.filter((t) => now >= t.at + t.ms);
  const flying = v.throws.filter((t) => now >= t.at && now < t.at + t.ms);
  return (
    <svg className={`rt-scene${compact ? ' compact' : ''}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The reindeer and the rings thrown so far" data-testid="ringtoss-scene">
      <defs>
        <radialGradient id="rt-floor" cx="50%" cy="85%" r="75%"><stop offset="0" stopColor="#fff" /><stop offset="1" stopColor="#dfe8f2" /></radialGradient>
      </defs>
      <rect width={W} height={H} fill="url(#rt-floor)" />
      {/* the throw line */}
      <line x1={toScreen(-0.6, 0).sx} y1={toScreen(0, 0).sy} x2={toScreen(0.6, 0).sx} y2={toScreen(0, 0).sy} stroke="#c7d2e0" strokeWidth={4} strokeDasharray="14 10" />
      {/* the zones, faintly: where a ring counts */}
      {v.pegs.map((p) => {
        const c = toScreen(p.x, p.z);
        return <ellipse key={p.id} cx={c.sx} cy={c.sy} rx={p.rx * SX * c.k} ry={p.rz * SZ * c.k * 0.5} fill={p.id === 'golden' ? '#ffd76a' : '#9ad9c4'} opacity={0.35} />;
      })}
      {/* rings on the floor (misses) under the reindeer */}
      {landed.filter((t) => !t.pegId).map((t) => <Ring key={t.n} t={t} color={ringColor(participants, t.pid)} practice={t.stage === 'practice'} />)}
      {/* the reindeer */}
      <g className="rt-deer" aria-hidden>
        <path d={antler('left')} stroke="#7a4b2a" strokeWidth={16} fill="none" strokeLinecap="round" />
        <path d={antler('right')} stroke="#7a4b2a" strokeWidth={16} fill="none" strokeLinecap="round" />
        <line x1={goldBase.sx} y1={goldBase.sy} x2={gold.sx} y2={gold.sy} stroke="#7a4b2a" strokeWidth={12} strokeLinecap="round" />
        <ellipse cx={head.sx} cy={head.sy - 10} rx={70} ry={58} fill="#a8683a" stroke="#3b2414" strokeWidth={6} />
        <ellipse cx={head.sx - 80} cy={head.sy - 40} rx={30} ry={14} fill="#a8683a" stroke="#3b2414" strokeWidth={5} transform={`rotate(-25 ${head.sx - 80} ${head.sy - 40})`} />
        <ellipse cx={head.sx + 80} cy={head.sy - 40} rx={30} ry={14} fill="#a8683a" stroke="#3b2414" strokeWidth={5} transform={`rotate(25 ${head.sx + 80} ${head.sy - 40})`} />
        <circle cx={head.sx - 24} cy={head.sy - 22} r={9} fill="#1b1b1b" /><circle cx={head.sx + 24} cy={head.sy - 22} r={9} fill="#1b1b1b" />
        <circle cx={head.sx - 21} cy={head.sy - 25} r={3} fill="#fff" /><circle cx={head.sx + 27} cy={head.sy - 25} r={3} fill="#fff" />
        <circle cx={nose.sx} cy={nose.sy + 8} r={20} fill="#ff3b3b" stroke="#3b2414" strokeWidth={5} />
        {v.pegs.map((p) => { const k = knob(p); return <circle key={p.id} cx={k.sx} cy={k.sy} r={p.id === 'golden' ? 15 : 12} fill={p.id === 'golden' ? '#ffcc33' : '#d9b48f'} stroke="#3b2414" strokeWidth={4} />; })}
        {v.pegs.map((p) => { const k = knob(p); return <text key={`l${p.id}`} x={k.sx} y={k.sy - 22} textAnchor="middle" className="rt-peg-pts">{p.points}</text>; })}
      </g>
      {/* ringers: stacked on their peg */}
      {landed.filter((t) => t.pegId).map((t) => {
        const p = peg(t.pegId!);
        const stack = landed.filter((u) => u.pegId === t.pegId && u.n < t.n).length;
        return <Ring key={t.n} t={{ ...t, x: p.x, z: p.z }} lift={0.12 + stack * 0.025} color={ringColor(participants, t.pid)} practice={t.stage === 'practice'} />;
      })}
      {/* rings in the air */}
      {flying.map((t) => {
        const f = flightAt({ angle: t.angle, power: t.power } as Launch, now - t.at, v.flight);
        return <Ring key={t.n} t={{ ...t, x: f.x, z: f.z }} lift={f.h} color={ringColor(participants, t.pid)} flying />;
      })}
    </svg>
  );
}

function Ring({ t, color, lift = 0, flying = false, practice = false }: { t: { x: number; z: number }; color: string; lift?: number; flying?: boolean; practice?: boolean }) {
  const c = toScreen(t.x, t.z, lift);
  const r = 46 * c.k;
  return <ellipse className={flying ? 'rt-ring flying' : 'rt-ring'} cx={c.sx} cy={c.sy} rx={r} ry={r * 0.42} fill="none" stroke={color} strokeWidth={10 * c.k}
    opacity={practice ? 0.4 : 1} strokeDasharray={practice ? `${8 * c.k} ${6 * c.k}` : undefined} />;
}

/** The TV / console: the scene, the scores, the clock. Sounds on the TV only. */
export function RingTossStage() {
  const { game, tv } = useHost();
  const offset = useServerOffset();
  const av = game.arenaView!;
  const v = ringView(av.view);
  const [now, setNow] = useState(() => Date.now() + offset);
  useEffect(() => {
    let raf = 0;
    const tick = () => { setNow(Date.now() + offset); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [offset]);
  const last = v.throws.at(-1);
  const landedN = v.throws.filter((t) => now >= t.at + t.ms).length;
  useEffect(() => { if (tv && last && now >= last.at + last.ms && last.points > 0) play('right'); }, [tv, landedN]); // eslint-disable-line react-hooks/exhaustive-deps
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
        <RingScene v={v} participants={av.participants} now={now} />
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
