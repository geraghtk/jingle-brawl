// Nutcracker Knockout on the runtime: the room decides every hit, a part comes off per hit and the
// head last, best of three, the bell and sudden death, only the duelists fight, and the duel gets
// its winner.
import { describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { DEFAULT_FIGHT as F } from '../engines/fight';
import { arenaAct, arenaDeadline, arenaOf, arenaOffers, handoff, startArena } from '../runtime';
import { DEFAULTS, lost, PARTS, robots } from './robots';
import type { RobotState } from './robots';
import { robotView } from './view';

const st = (g: Game) => arenaOf(g)!.state as RobotState;
const ctx = (now: number, seed = 3) => ({ now, rng: rng(seed) });

/** p1 (fighter 0) against p2; p3 watches. */
function fight(config: Record<string, unknown> = {}): Game {
  const g = table(3);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const out = startArena(tx, 'nutcracker_knockout', { mode: 'direct_duel', participants: ['p1', 'p2'], duelists: ['p1', 'p2'], config }, ctx(NOW));
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
const act = (g: Game, who: string, action: unknown, now: number) => arenaAct(g, who, { action }, ctx(now));
const did = (g: Game, who: string, action: unknown, now: number): Game => {
  const plan = act(g, who, action, now);
  expect(plan.error, `${who} ${JSON.stringify(action)}`).toBeUndefined();
  return apply(g, plan);
};
/** Run the room's next deadline (at its own time, or `now`). */
function due(g: Game, now?: number): Game {
  const d = arenaOf(g)!.deadlines[0];
  return apply(g, arenaDeadline(g, d.id, ctx(now ?? d.at)));
}
const bell = (g: Game) => due(g);                                  // intro → fighting, ko → next round
/** `who` lands `n` clean punches on a robot that just stands there; returns the game and the time. */
function batter(g: Game, who: string, n: number, from: number): [Game, number] {
  let now = from;
  for (let i = 0; i < n && st(g).stage === 'fighting'; i++) {
    g = did(g, who, { kind: 'punch', hand: i % 2 ? 'R' : 'L' }, now);
    g = due(g);                                                     // it lands
    now += F.windupMs + F.recoverMs + 10;
  }
  return [g, now];
}

describe('Nutcracker Knockout', () => {
  it('a round starts with a moment to get ready: nothing can be thrown until the bell', () => {
    let g = fight();
    expect(st(g)).toMatchObject({ stage: 'intro', round: 1, order: ['p1', 'p2'], wins: { p1: 0, p2: 0 } });
    expect(act(g, 'p1', { kind: 'punch', hand: 'L' }, NOW + 10).error).toBe('Wait for the bell');
    g = bell(g);
    expect(st(g).stage).toBe('fighting');
    expect(st(g).stageEndsAt).toBe(NOW + DEFAULTS.introMs + DEFAULTS.roundMs);
  });

  it('only the two duelists fight, with a punch or a block and nothing else', () => {
    const g = bell(fight());
    const now = NOW + DEFAULTS.introMs + 100;
    expect(act(g, 'p3', { kind: 'punch', hand: 'L' }, now).error).toBe("You're not in this game");
    expect(act(g, 'p1', { kind: 'punch', hand: 'X' }, now).error).toBe('Left or right?');
    expect(act(g, 'p1', { kind: 'guard', on: 'yes' }, now).error).toBe('Guard up or down?');
    expect(act(g, 'p1', { kind: 'win' }, now).error).toBe('Punch or block');
    expect(act(g, 'p1', { kind: 'punch', hand: 'L', damage: 99, lands: 0 }, now).error).toBeUndefined();   // extras are ignored: the room times it
  });

  it('the room lands the punch, a wind-up later; a part comes off for each hit, in order, the head last', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p1', { kind: 'punch', hand: 'L' }, t0);
    expect(st(g).fight.f[0].punch).toMatchObject({ hand: 'L', at: t0, lands: t0 + F.windupMs });
    expect(arenaOf(g)!.deadlines).toEqual([{ id: 'tick', at: t0 + F.windupMs }]);
    expect(act(g, 'p1', { kind: 'punch', hand: 'R' }, t0 + 50).error).toBe('Not yet');          // one punch at a time
    g = due(g);
    expect(st(g).fight.f[1].health).toBe(F.health - 1);
    const tv = robotView((projectGame(g, { kind: 'tv' }, t0 + F.windupMs).arenaView as { view: unknown }).view);
    expect(tv.fighters[1].lost).toEqual(['hat']);
    expect(tv.fighters[0].lost).toEqual([]);
    expect(tv.events.at(-1)).toMatchObject({ kind: 'hit', by: 0, hand: 'L' });
    // the order of the parts, and the head is the last one
    expect(PARTS.at(-1)).toBe('head');
    expect(PARTS.length).toBe(F.health);
    expect(lost(F.health)).toEqual([]);
    expect(lost(1)).toEqual(PARTS.slice(0, -1));
    expect(lost(0).at(-1)).toBe('head');
  });

  it('a block stops the punch; the view shows the fight as it stands at that moment, with no new message', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p2', { kind: 'guard', on: true }, t0);
    g = did(g, 'p1', { kind: 'punch', hand: 'R' }, t0 + 400);
    expect(act(g, 'p2', { kind: 'punch', hand: 'L' }, t0 + 450).error).toBe('Drop your guard to punch');
    // a screen asking after the punch was due sees it blocked, though the room's timer hasn't run yet
    const late = robotView((projectGame(g, { kind: 'phone', pid: 'p2' }, t0 + 400 + F.windupMs + 5).arenaView as { view: unknown }).view);
    expect(late.events.at(-1)).toMatchObject({ kind: 'blocked', by: 0 });
    expect(late.fighters[1].health).toBe(F.health);
    expect(late.mySeat).toBe(1);
    g = due(g);
    expect(st(g).fight.f[0].busy).toBe('blocked');
    g = did(g, 'p2', { kind: 'guard', on: false }, t0 + 400 + F.windupMs + 20);                  // dropping a guard is always allowed
  });

  it('the last hit takes the head: the round is over, then the next round starts fresh', () => {
    let g = bell(fight());
    let now = NOW + DEFAULTS.introMs + 100;
    [g, now] = batter(g, 'p1', F.health, now);
    expect(st(g)).toMatchObject({ stage: 'ko', wins: { p1: 1, p2: 0 }, last: { round: 1, winner: 'p1', loser: 'p2', how: 'knockout' } });
    expect(st(g).fight.f[1].health).toBe(0);
    expect(act(g, 'p2', { kind: 'punch', hand: 'L' }, now).error).toBe('Wait for the bell');
    const tv = robotView((projectGame(g, { kind: 'tv' }, now).arenaView as { view: unknown }).view);
    expect(tv.fighters[1].lost.at(-1)).toBe('head');
    g = bell(g);                                                    // the head has flown: round 2
    expect(st(g)).toMatchObject({ stage: 'intro', round: 2 });
    expect(st(g).fight.f.map((f) => f.health)).toEqual([F.health, F.health]);
    expect(arenaOf(g)!.result ?? null).toBeNull();
  });

  it('best of three: two rounds win the match and the duel', () => {
    let g = bell(fight());
    let now = NOW + DEFAULTS.introMs + 100;
    [g, now] = batter(g, 'p2', F.health, now);                      // p2 takes round 1
    g = bell(bell(g));
    now = st(g).stageEndsAt - DEFAULTS.roundMs + 100;
    [g, now] = batter(g, 'p1', F.health, now);                      // p1 takes round 2
    g = bell(bell(g));
    expect(st(g).round).toBe(3);
    now = st(g).stageEndsAt - DEFAULTS.roundMs + 100;
    [g] = batter(g, 'p2', F.health, now);                           // p2 takes the decider
    expect(arenaOf(g)!.result ?? null).toBeNull();                  // not until the knockout has been seen
    g = bell(g);
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', winners: ['p2'], losers: ['p1'], duel: { winningDuelistId: 'p2', losingDuelistId: 'p1' } });
    expect(st(g).stage).toBe('done');
    const back = apply(g, handoff(g));
    expect(back.currentTurn?.duelGameWinner).toBe('p2');
  });

  it('the bell: more parts left wins the round; level is sudden death, where one hit takes the head', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    [g] = batter(g, 'p1', 1, t0);
    g = due(g, st(g).stageEndsAt);
    expect(st(g)).toMatchObject({ stage: 'ko', last: { winner: 'p1', how: 'time' } });

    let h = bell(fight());
    h = due(h, st(h).stageEndsAt);                                  // nobody landed a thing
    expect(st(h)).toMatchObject({ stage: 'fighting', sudden: true });
    expect(st(h).fight.f.map((f) => f.health)).toEqual([1, 1]);
    const t1 = st(h).stageEndsAt - DEFAULTS.suddenMs + 50;
    [h] = batter(h, 'p2', 1, t1);
    expect(st(h)).toMatchObject({ stage: 'ko', last: { winner: 'p2', how: 'knockout' } });
  });

  it('two robots that never swing: the match ends unresolved and the host decides (nobody wins by default)', () => {
    let g = bell(fight());
    g = due(g, st(g).stageEndsAt);                                  // level: sudden death
    g = due(g, st(g).stageEndsAt);                                  // and nothing there either
    const r = arenaOf(g)!.result!;
    expect(r.status).toBe('requires_tiebreaker');
    expect(r.winners).toEqual([]);
    expect(apply(g, handoff(g)).currentTurn?.duelGameWinner ?? null).toBeNull();
  });

  it('a punch that arrives after the round was decided changes nothing', () => {
    let g = bell(fight({ fight: { health: 1 } }));
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p1', { kind: 'punch', hand: 'L' }, t0);
    // p2's punch reaches the room after p1's has landed, before the room's own timer ran: the head is already off
    const late = act(g, 'p2', { kind: 'punch', hand: 'L' }, t0 + F.windupMs + 5);
    expect(late.error).toBe('Wait for the bell');
    g = due(g);
    expect(st(g)).toMatchObject({ stage: 'ko', last: { winner: 'p1' } });
  });

  it('the settings are held to what a robot is: as many hits as it has parts', () => {
    expect(robots.validateConfig({ fight: { health: 99 } }).fight.health).toBe(PARTS.length);
    expect(robots.validateConfig({ fight: { health: 0 } }).fight.health).toBe(PARTS.length);
    expect(robots.validateConfig({ fight: { health: 3 } }).fight.health).toBe(3);
    expect(() => robots.validateConfig({ roundsToWin: 0 })).toThrow();
    expect(robots.canAlwaysResolve('direct_duel', DEFAULTS, 2)).toEqual({ ok: true });
    expect(robots.canAlwaysResolve('party_with_duelists', DEFAULTS, 5).ok).toBe(false);
  });

  it('as a duel game: only the duelists are in it, however many are at the table', () => {
    const g0 = table(6, { duelPick: 'defender' });
    g0.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Nutcracker Knockout', kind: 'app', app: 'nutcracker_knockout', via: 'picker' } };
    const g = apply(g0, startDuelApp(g0, { now: NOW, rng: rng(2) }));
    expect(arenaOf(g)).toMatchObject({ mode: 'direct_duel', participants: ['p2', 'p1'], duelists: ['p2', 'p1'] });
    expect(st(g).order).toEqual(['p2', 'p1']);
  });

  it('simulated hands can play a whole match to a result', () => {
    let g = fight();
    let now = NOW;
    for (let step = 0; step < 20_000 && !arenaOf(g)!.result; step++) {
      now += 40;
      const d = arenaOf(g)!.deadlines[0];
      if (d && d.at <= now) { g = apply(g, arenaDeadline(g, d.id, ctx(now, step))); continue; }
      for (const who of ['p1', 'p2']) {
        const [offer] = arenaOffers(g, who, now, rng(step * 7 + (who === 'p1' ? 1 : 2)));
        if (!offer) continue;
        const plan = act(g, who, offer, now);
        expect(plan.error, JSON.stringify(offer)).toBeUndefined();   // what is offered can be pressed
        g = apply(g, plan);
      }
    }
    const r = arenaOf(g)!.result!;
    expect(r).toBeTruthy();
    expect(['resolved', 'requires_tiebreaker']).toContain(r.status);
    expect(arenaOffers(g, 'p3', now, rng(1))).toEqual([]);
  });
});
