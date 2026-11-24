// Elf Tower Wars on the TV (and the host console): the battlefield. Team A's workshop on the left, team
// B's on the right, three lanes between them; each side's towers on its spots, the toys marching both
// ways (glided between the room's steps from each one's speed), a tower's snowballs on whatever it is
// aiming at, a shield where one is up; and over it each team's workshop health, its players (each
// with their own energy) and the clock. Sounds on the TV only.
import { useEffect, useRef } from 'react';
import { armed, guardTarget, healthOf, powerOf, reachOf, targetOf, unitAt } from '../core/minigames/engines/lanes';
import { energyOf, fieldClock, towerView } from '../core/minigames/towers/view';
import type { TowerTeamView } from '../core/minigames/towers/view';
import { useFrameNow } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { play } from '../ui/sound';
import { LaneGate, TOWER_COLORS, TowerArt, UnitArt, WorkshopArt, artSrc, laneHasGate } from '../ui/towers/art';
import { recoil, shotEvery, shotPhase, stagger } from '../ui/towers/motion';
import '../ui/motion.css';
import '../ui/arena.css';
import '../ui/towers.css';
import { useHost } from './HostContext';

/** Across the field, in % of its width: where a point `d` of the way from `side`'s workshop is. */
export const fieldX = (side: number, d: number) => (side === 0 ? 9 + d * 82 : 91 - d * 82);

/** Where the paths run, in % of the field's height (room above the top one for its towers). */
export const laneTop = (lane: number, lanes: number) => ((lane + 0.78) / (lanes + 0.22)) * 100;
/** Where a lane's gate stands, as a point along the lane from its own workshop (just short of the lane's end). */
export const GATE_AT = -0.012;
/** The middle path runs out from behind the workshop: its gate stands where the path clears the building, and a toy is
 *  drawn from there (they used to slide out from under the house: Kevin, 2026-11-23). */
export const MID_GATE_AT = 0.105;
export const gateAt = (lane: number, lanes: number) => (laneHasGate(lane, lanes) ? GATE_AT : MID_GATE_AT);
/** Further down the field is nearer the camera: it is drawn over what is behind it. */
const zAt = (y: number) => 10 + Math.round(y / 2);
/** The scenery: what, across (%), down (%), how tall (cqh). Between the paths, clear of the building spots. */
const SCENERY: [string, number, number, number][] = [
  ['pine', 18, 13, 9.5], ['pine', 33, 12, 8.5], ['snowman', 50, 13, 8.5], ['pine', 67, 12, 8.5], ['pine', 82, 13, 9.5],
  ['gifts', 33, 44, 7.5], ['pine', 50, 45, 11], ['cane', 67, 44, 8.5],
  ['cane', 33, 76, 8.5], ['snowman', 50, 77, 9.5], ['gifts', 67, 76, 7.5],
  ['pine', 22, 103, 10], ['pine', 78, 103, 10],
];

export function TowerStage() {
  const { game, tv } = useHost();
  const live = useFrameNow();
  const av = game.arenaView!;
  const now = av.pausedAt ?? live;                       // paused: the battle stands as it was
  const v = towerView(av.view);
  const f = v.field, p = v.rules;
  // while everyone plans the field stands still (as the last march left it); while it marches it is now
  const fnow = fieldClock(v, now);
  const planning = v.stage !== 'done' && v.phase === 'plan';
  const name = (id: string) => game.players?.[id]?.name || 'Someone';

  // out loud on the TV: each new thing once
  const newest = f.events.at(-1)?.n ?? 0;
  const heard = useRef({ n: newest, stage: v.stage as string, phase: `${v.round}|${v.phase}`, primed: false });
  useEffect(() => {
    const was = heard.current;
    heard.current = { n: Math.max(newest, was.n), stage: v.stage, phase: `${v.round}|${v.phase}`, primed: true };
    if (!tv) return;
    if (!was.primed) { play('go'); return; }
    for (const e of f.events) {
      if (e.n <= was.n) continue;
      const pitch = e.side === 0 ? 1 : 0.84;
      if (e.kind === 'arrived') { play('thud', 0.7 * pitch); play('clank', 0.7 * pitch); }
      else if (e.kind === 'died') play('tick', 0.6 * pitch);
      else if (e.kind === 'blocked') play('ding', 0.7 * pitch);
      else if (e.kind === 'built') play('right', pitch);
      else if (e.kind === 'shield') play('whoosh', pitch);
      else if (e.kind === 'patch') play('ding', 1.2 * pitch);
      else if (e.kind === 'zap') play('pass', 1.3 * pitch);
      else if (e.kind === 'armed') play('tick', 0.5 * pitch);
      else if (e.kind === 'combo') { play('whoosh', 0.6 * pitch); play('thud', 0.6 * pitch); }
    }
    if (was.stage !== v.stage) { if (v.stage === 'done') play('fanfare'); else if (v.stage === 'overtime') play('bell'); }
    // the two halves of a round: a horn as the armies march out, a bell when it is time to plan again
    else if (was.phase !== `${v.round}|${v.phase}` && v.stage !== 'done') play(v.phase === 'march' ? 'horn' : 'bell');
  }, [newest, v.stage, v.phase, v.round, tv]); // eslint-disable-line react-hooks/exhaustive-deps

  const secs = secsLeft(v.phaseEndsAt, now);
  const winner = v.stage === 'done' ? v.teams.find((t) => t.teamId === v.winner) : null;
  const laneY = (lane: number) => laneTop(lane, p.lanes);
  /** Who is here and has not said Done yet (the plan waits for them, or its clock). */
  const waiting = v.teams.flatMap((t) => t.active.filter((id) => !t.done.includes(id)));

  // a toy that is gone before it got there was knocked over: a puff where it last stood
  // (and the toy itself, tipping over backwards in it)
  const stood = useRef(new Map<string, { x: number; lane: number; side: number; d: number; kind: string }>());
  const puffs = useRef<{ key: string; x: number; lane: number; side: number; kind: string; at: number }[]>([]);
  const here = new Map(f.units.map((u) => { const d = unitAt(u, f.asOf, fnow); return [u.id, { x: fieldX(u.side, d), lane: u.lane, side: u.side, d, kind: u.kind }] as const; }));
  for (const [id, was] of stood.current) if (!here.has(id) && was.d < 0.9) puffs.current.push({ key: id, ...was, at: now });
  stood.current = here;
  puffs.current = puffs.current.filter((q) => now - q.at < 700 && now >= q.at);
  const just = (kind: string, ms: number, mine: (side: number) => boolean) => f.events.some((e) => e.kind === kind && mine(e.side) && now - e.at >= 0 && now - e.at < ms);
  const hit = (side: number) => f.events.some((e) => e.kind === 'arrived' && e.side !== side && now - e.at < 500);
  /** A toy has just left this workshop: it gives a little heave. */
  const sent = (side: number) => just('sent', 320, (s) => s === side);

  const panel = (t: TowerTeamView, i: number) => {
    const s = f.sides[i];
    return (
      <div className={`tw-team ${i === 0 ? 'left' : 'right'}${winner === t ? ' won' : winner ? ' lost' : ''}`} style={{ ['--team' as string]: TOWER_COLORS[i] }} data-testid={`tower-team-${t.teamId}`}>
        <div className="tw-team-head">
          <span className="tw-team-name">Team {t.teamId}</span>
          <span className="tw-hp" role="img" aria-label={`Workshop ${Math.ceil(s.hp)} of ${p.baseHp}`} data-testid={`tower-hp-${t.teamId}`}>
            <span className="tw-hp-bar"><i style={{ width: `${(s.hp / p.baseHp) * 100}%` }} /></span><b>{Math.ceil(s.hp)}</b>
          </span>
        </div>
        <div className="tw-crew">
          {t.members.map((id) => {
            const pl = game.players?.[id];
            return (
              <span key={id} className={`tw-job${t.active.includes(id) ? '' : ' away'}${planning && t.done.includes(id) ? ' done' : planning && t.active.includes(id) ? ' thinking' : ''}`} data-testid={`tower-member-${id}`}>
                <span className="tw-who on" title={name(id)}><Avatar name={pl?.name || id} avatar={pl?.avatar} size={30} /></span>
                {t.members.length <= 3 && <i>{name(id)}</i>}
                <b className="tw-energy">{planning && t.done.includes(id) ? '✓' : Math.floor(energyOf(t, id))}</b>
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className={`tw-stage${v.stage === 'done' ? ' over' : ''}${av.pausedAt != null || planning ? ' held' : ''}${planning ? ' planning' : ''}`} data-testid="tower-stage">
      <div className="tw-hud">
        {v.teams[0] && panel(v.teams[0], 0)}
        <div className="tw-clock" data-testid="tower-clock">
          <span className="tw-clock-label" data-testid="tower-round">{v.stage === 'done' ? 'Battle over' : v.stage === 'overtime' ? 'Extra round' : v.round >= v.rounds ? 'Last round' : `Round ${v.round}`}</span>
          {v.stage !== 'done' && <span key={`${v.round}${v.phase}`} className={`tw-phase ${v.phase}`} data-testid="tower-phase">{planning ? 'Plan' : 'March!'}</span>}
          {/* (a march has no clock: it lasts until the last toy is home or down) */}
          {planning && <span className={`tw-secs${secs <= 3 ? ' hurry' : ''}`}>{secs}</span>}
          {planning && <span className="tw-waiting" data-testid="tower-waiting">{waiting.length ? `Planning: ${waiting.length > 3 ? `${waiting.length} still to go` : waiting.map(name).join(', ')}` : 'Here they come!'}</span>}
        </div>
        {v.teams[1] && panel(v.teams[1], 1)}
      </div>
      <div className="tw-field" style={{ ['--lanes' as string]: p.lanes }} data-testid="tower-field">
        {Array.from({ length: p.lanes }, (_, lane) => <div key={lane} className="tw-lane" style={{ top: `${laneY(lane)}%` }} />)}
        {/* the scenery: off the paths and clear of the building spots */}
        {SCENERY.map(([what, x, y, size], i) => <img key={i} className="tw-prop" src={artSrc(what)} alt="" draggable={false} style={{ left: `${x}%`, top: `${y}%`, height: `${size}cqh`, zIndex: zAt(y) }} />)}
        {[0, 1].map((side) => {
          const health = f.sides[side].hp / p.baseHp;
          return (
            <div key={side} className={`tw-base side-${side}${hit(side) ? ' hit' : sent(side) ? ' sent' : ''}${health <= 0 ? ' down' : ''}`} style={{ ['--team' as string]: TOWER_COLORS[side] }} data-testid={`tower-base-${side}`}>
              <WorkshopArt color={TOWER_COLORS[side]} health={health} flip={side === 1} style={{ width: '100%', height: '100%' }} />
              {health < 0.6 && <span className={`tw-smoke${health < 0.3 ? ' thick' : ''}`} aria-hidden><i /><i /><i />{health <= 0 && <><i /><i /></>}</span>}
              {/* it has fallen: a cloud of dust rolls out from under it, once */}
              {health <= 0 && <span className="tw-rubble" aria-hidden><i /><i /><i /><i /><i /><i /></span>}
            </div>
          );
        })}
        {/* a gate at each end of every path: its toys come out of it */}
        {[0, 1].flatMap((side) => Array.from({ length: p.lanes }, (_, lane) => (
          true ? <div key={`gate${side}-${lane}`} className={`tw-gate side-${side}`} style={{ left: `${fieldX(side, gateAt(lane, p.lanes))}%`, top: `${laneY(lane)}%`, zIndex: zAt(laneY(lane)) + 3 }} data-testid={`tower-gate-${side}-${lane}`}><LaneGate color={TOWER_COLORS[side]} style={{ width: '100%', height: '100%' }} /></div>
          : null)))}
        {/* what each team has lined up at its door for the next march (everyone can see it: plan against it) */}
        {planning && [0, 1].flatMap((side) => Array.from({ length: p.lanes }, (_, lane) => {
          const mine = v.queue.filter((q) => q.side === side && q.lane === lane);
          if (!mine.length) return null;
          return (
            <div key={`q${side}-${lane}`} className={`tw-queue side-${side}`} style={{ left: `${fieldX(side, 0.085)}%`, top: `${laneY(lane)}%`, ['--team' as string]: TOWER_COLORS[side] }} data-testid={`tower-queue-${side}-${lane}`}>
              {[...new Set(mine.map((q) => q.unit))].map((unit) => {
                const n = mine.filter((q) => q.unit === unit).length;
                return <span key={unit} className="tw-queued"><UnitArt kind={unit} color={TOWER_COLORS[side]} flip={side === 1} style={{ width: '100%', height: '100%' }} />{n > 1 && <b>{n}</b>}</span>;
              })}
            </div>
          );
        }))}
        {/* the spots (on the verge above each path), and what is built on them */}
        {[0, 1].flatMap((side) => Array.from({ length: p.lanes }, (_, lane) => p.slots.map((q, slot) => {
          const ti = f.sides[side].towers.findIndex((x) => x.lane === lane && x.slot === slot), t = f.sides[side].towers[ti];
          const firing = !!t && v.stage !== 'done' && !planning && !!p.towers[t.kind].dps && !!targetOf(f, side, t, p);
          // it kicks back as each of its shots leaves (the same loop the shot below is drawn from)
          const kick = t && firing ? recoil(shotPhase(now, ti, shotEvery(t.kind === 'cannon', powerOf(t, p)))) : 0;
          return (
            <div key={`${side}-${lane}-${slot}`} className={`tw-spot${t ? ' built' : ''}`} style={{ left: `${fieldX(side, q)}%`, top: `${laneY(lane)}%`, zIndex: zAt(laneY(lane)) - 2, ['--team' as string]: TOWER_COLORS[side] }} data-testid={`tower-spot-${side}-${lane}-${slot}`}>
              {t && <span key={`${t.kind}-${t.level || 1}`} className={`tw-tower${firing ? ' firing' : ''}${(t.level || 1) > 1 ? ' up' : ''}`} style={{ ['--grow' as string]: 1 + 0.07 * ((t.level || 1) - 1), ['--kick' as string]: kick.toFixed(3), ['--dir' as string]: side === 0 ? 1 : -1 }}><TowerArt kind={t.kind} level={t.level || 1} color={TOWER_COLORS[side]} flip={side === 1} style={{ width: '100%', height: '100%' }} /></span>}
              {t && (t.level || 1) > 1 && <span className="tw-stars" aria-label={`level ${t.level}`}>{'★'.repeat((t.level || 1) - 1)}</span>}
              {/* just built or made better: how far it reaches, for a moment */}
              {t && f.events.some((e) => e.kind === 'built' && e.side === side && e.lane === lane && (e.what || '').split(':')[1] === String(slot) && now - e.at >= 0 && now - e.at < 1600)
                && <span className="tw-reach" style={{ width: `${reachOf(t, p) * 2 * 82 * (100 / 14.4)}%` }} aria-hidden />}
            </div>
          );
        })))}
        {/* what each tower is doing: a frost lamp's chill on the path, a snowball (or a peppermint) lobbed at whatever a tower is aiming at */}
        {[0, 1].flatMap((side) => f.sides[side].towers.map((t, i) => {
          const x = fieldX(side, p.slots[t.slot]), key = `${side}-${t.lane}-${t.slot}`;
          if (t.kind === 'frost') return <span key={key} className="tw-frost" style={{ left: `${x}%`, top: `${laneY(t.lane)}%`, width: `${reachOf(t, p) * 2 * 82}%` }} />;
          const aim = targetOf(f, side, t, p);
          if (!aim || v.stage === 'done' || planning) return null;                // (no shot hangs in the air while everyone plans)
          const heavy = t.kind === 'cannon', every = shotEvery(heavy, powerOf(t, p));
          const to = fieldX(1 - side, unitAt(aim, f.asOf, fnow));
          const k = shotPhase(now, i, every);                                // one shot in the air at a time, from the tower's top to the toy
          const y = -11 * (1 - k) + (aim.side === 0 ? -5 : 0) * k - 9 * 4 * k * (1 - k);
          return (
            <span key={key} className={`tw-ball${heavy ? ' heavy' : ''}`} style={{ left: `${x + (to - x) * k}%`, top: `calc(${laneY(t.lane)}% + ${y}cqh)`, transform: `translate(-50%, -50%) rotate(${k * 540}deg) scale(${0.8 + 0.4 * Math.sin(Math.PI * k)})` }} />
          );
        }))}
        {/* a workshop looks after itself a little: its elves throw at whatever has got close */}
        {v.stage !== 'done' && !planning && [0, 1].flatMap((side) => Array.from({ length: p.lanes }, (_, lane) => {
          const aim = guardTarget(f, side, lane, p);
          if (!aim) return null;
          const from = fieldX(side, 0.02), to = fieldX(1 - side, unitAt(aim, f.asOf, fnow));
          const k = ((now + (side * 3 + lane) * 113) % 380) / 380;
          const y = -6 * (1 - k) + (aim.side === 0 ? -5 : 0) * k - 5 * 4 * k * (1 - k);
          return <span key={`g${side}-${lane}`} className="tw-ball" style={{ left: `${from + (to - from) * k}%`, top: `calc(${laneY(lane)}% + ${y}cqh)`, transform: `translate(-50%, -50%) rotate(${k * 540}deg) scale(${0.7 + 0.3 * Math.sin(Math.PI * k)})` }} />;
        }))}
        {/* the avalanche: armed by one (it rumbles over that team's half of the path, waiting for a second), then down it comes */}
        {[0, 1].map((side) => (v.stage !== 'done' && armed(f.sides[side], now, p)
          ? <div key={`arm${side}`} className="tw-armed" style={{ left: `${fieldX(side, p.comboReach / 2)}%`, top: `${laneY(f.sides[side].armLane)}%`, width: `${p.comboReach * 82}%`, ['--team' as string]: TOWER_COLORS[side] }} data-testid={`tower-armed-${side}`}><b>Avalanche! One more!</b></div> : null))}
        {f.events.filter((e) => e.kind === 'combo' && now - e.at >= 0 && now - e.at < 900).map((e) => (
          <div key={e.n} className="tw-avalanche" style={{ left: `${fieldX(e.side, p.comboReach / 2)}%`, top: `${laneY(e.lane ?? 1)}%`, width: `${p.comboReach * 82}%` }} aria-hidden><i /><i /><i /><i /><i /><i /></div>
        ))}
        {/* a shield across a path at a workshop */}
        {[0, 1].map((side) => (now < f.sides[side].shieldUntil && f.sides[side].shieldLane >= 0
          ? <div key={side} className={`tw-shield side-${side}`} style={{ left: `${fieldX(side, 0.075)}%`, top: `${laneY(f.sides[side].shieldLane)}%`, ['--team' as string]: TOWER_COLORS[side] }} data-testid={`tower-shield-${side}`} /> : null))}
        {/* the toys: each side keeps to its own side of the path, so the armies pass */}
        {f.units.map((u) => {
          // still behind its gate (the top and bottom paths): not out yet, so not drawn standing in the snow beside it
          if (unitAt(u, f.asOf, fnow) < gateAt(u.lane, p.lanes)) return null;
          return (
            <div key={u.id} className={`tw-unit ${u.kind} side-${u.side}`} style={{ left: `${fieldX(u.side, unitAt(u, f.asOf, fnow))}%`, top: `${laneY(u.lane)}%`, zIndex: zAt(laneY(u.lane)) + (u.side === 0 ? 0 : 1), ['--team' as string]: TOWER_COLORS[u.side] }} data-testid={`tower-unit-${u.id}`}>
              <span className="tw-toy" style={{ animationDelay: `-${stagger(u.id, 800)}ms` }}><UnitArt kind={u.kind} color={TOWER_COLORS[u.side]} flip={u.side === 1} style={{ width: '100%', height: '100%' }} /></span>
              {healthOf(u, p) < 1 && <span className="tw-unit-hp" data-testid={`tower-unit-hp-${u.id}`}><i style={{ width: `${healthOf(u, p) * 100}%` }} /></span>}
            </div>
          );
        })}
        {/* a toy knocked over: a puff of snow where it stood */}
        {puffs.current.map((q) => (
          <div key={`f${q.key}`} className={`tw-unit tw-fallen ${q.kind} side-${q.side}`} style={{ left: `${q.x}%`, top: `${laneY(q.lane)}%`, ['--dir' as string]: q.side === 0 ? 1 : -1 }} aria-hidden>
            <span className="tw-toy"><UnitArt kind={q.kind} color={TOWER_COLORS[q.side]} flip={q.side === 1} style={{ width: '100%', height: '100%' }} /></span>
          </div>
        ))}
        {puffs.current.map((q) => <span key={q.key} className="tw-puff" style={{ left: `${q.x}%`, top: `${laneY(q.lane)}%`, ['--team' as string]: TOWER_COLORS[q.side] }} aria-hidden><i /><i /><i /><i /><i /></span>)}
        {/* a toy that got through: a bang at the workshop and what it cost */}
        {f.events.filter((e) => e.kind === 'arrived' && now - e.at >= 0 && now - e.at < 900).map((e) => (
          <span key={e.n} className="tw-bang" style={{ left: `${fieldX(e.side, 0.97)}%`, top: `${laneY(e.lane ?? 1)}%` }} aria-hidden><b>-{p.units[e.what || 'soldier']?.damage ?? 1}</b></span>
        ))}
        {f.events.filter((e) => e.kind === 'blocked' && now - e.at >= 0 && now - e.at < 600).map((e) => (
          <span key={e.n} className="tw-clink" style={{ left: `${fieldX(e.side, 0.925)}%`, top: `${laneY(e.lane ?? 1)}%` }} aria-hidden />
        ))}
      </div>
    </div>
  );
}
