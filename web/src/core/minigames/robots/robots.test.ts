// Nutcracker Knockout on the runtime: the room decides every blow, a part takes two blows (loose,
// then off) and the head goes last, best of three, the bell and sudden death, only the duelists
// fight, and the duel gets its winner.
import { describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { DEFAULT_FIGHT as F } from '../engines/fight';
import { arenaAct, arenaDeadline, arenaOf, arenaOffers, handoff, startArena } from '../runtime';
import { DEFAULTS, HITS_PER_PART, loose, lost, PARTS, robots } from './robots';
import type { RobotState } from './robots';
import { robotView } from './view';

const st = (g: Game) => arenaOf(g)!.state as RobotState;
const ctx = (now: number, seed = 3) => ({ now, rng: rng(seed) });
const HEALTH = PARTS.length * HITS_PER_PART;
const JAB = { kind: 'punch', punch: 'jab' }, HEAVY = { kind: 'punch', punch: 'heavy' };

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
const view = (g: Game, now: number, who?: string) => robotView((projectGame(g, who ? { kind: 'phone', pid: who } : { kind: 'tv' }, now).arenaView as { view: unknown }).view);
/** `who` lands `n` clean jabs on a robot that just stands there; returns the game and the time. */
function batter(g: Game, who: string, n: number, from: number): [Game, number] {
  let now = from;
  for (let i = 0; i < n && st(g).stage === 'fighting'; i++) {
    g = did(g, who, JAB, now);
    g = due(g);                                                     // it lands
    now += F.jabMs + F.recoverMs + 10;
  }
  return [g, now];
}

describe('Nutcracker Knockout', () => {
  it('a round starts with a moment to get ready: nothing can be thrown until the bell', () => {
    let g = fight();
    expect(st(g)).toMatchObject({ stage: 'intro', round: 1, order: ['p1', 'p2'], wins: { p1: 0, p2: 0 } });
    expect(act(g, 'p1', JAB, NOW + 10).error).toBe('Wait for the bell');
    g = bell(g);
    expect(st(g).stage).toBe('fighting');
    expect(st(g).stageEndsAt).toBe(NOW + DEFAULTS.introMs + DEFAULTS.roundMs);
    expect(st(g).fight.f.map((f) => f.health)).toEqual([HEALTH, HEALTH]);
  });

  it('only the two duelists fight, with a jab, a haymaker, a block or a dodge and nothing else', () => {
    const g = bell(fight());
    const now = NOW + DEFAULTS.introMs + 100;
    expect(act(g, 'p3', JAB, now).error).toBe("You're not in this game");
    expect(act(g, 'p1', { kind: 'punch', punch: 'uppercut' }, now).error).toBe('A jab or a haymaker?');
    expect(act(g, 'p1', { kind: 'guard', on: 'yes' }, now).error).toBe('Guard up or down?');
    expect(act(g, 'p1', { kind: 'win' }, now).error).toBe('Punch, block or dodge');
    expect(act(g, 'p1', { kind: 'punch', punch: 'jab', damage: 99, lands: 0 }, now).error).toBeUndefined();   // extras are ignored: the room times it
    expect(act(g, 'p1', { kind: 'dodge' }, now).error).toBeUndefined();
  });

  it('a part takes two blows: the first knocks it loose, the second knocks it off; the head is the last', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p1', JAB, t0);
    expect(st(g).fight.f[0].punch).toMatchObject({ kind: 'jab', at: t0, lands: t0 + F.jabMs });
    expect(arenaOf(g)!.deadlines).toEqual([{ id: 'tick', at: t0 + F.jabMs }]);
    expect(act(g, 'p1', JAB, t0 + 50).error).toBe('Not yet');        // one punch at a time
    g = due(g);
    let tv = view(g, t0 + F.jabMs);
    expect(tv.fighters[1]).toMatchObject({ health: HEALTH - 1, lost: [], loose: 'hat' });
    expect(tv.fighters[0].loose).toBeUndefined();
    expect(tv.events.at(-1)).toMatchObject({ kind: 'hit', by: 0, punch: 'jab' });
    [g] = batter(g, 'p1', 1, t0 + F.jabMs + F.recoverMs + 10);
    tv = view(g, t0 + 5000);
    expect(tv.fighters[1].lost).toEqual(['hat']);
    expect(tv.fighters[1].loose).toBeUndefined();
    expect(tv.parts).toBe(PARTS.length);
    // the arithmetic of parts
    expect(PARTS.at(-1)).toBe('head');
    expect([lost(HEALTH), loose(HEALTH)]).toEqual([[], null]);
    expect([lost(HEALTH - 1), loose(HEALTH - 1)]).toEqual([[], 'hat']);
    expect([lost(HEALTH - 2), loose(HEALTH - 2)]).toEqual([['hat'], null]);
    expect([lost(1), loose(1)]).toEqual([PARTS.slice(0, -1), 'head']);
    expect([lost(0).at(-1), loose(0)]).toEqual(['head', null]);
    expect(lost(2).includes('head')).toBe(false);                    // the head only goes with the last blow
  });

  it('a haymaker takes a whole part at once, but lands much later', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p1', HEAVY, t0);
    expect(st(g).fight.f[0].punch).toMatchObject({ kind: 'heavy', lands: t0 + F.heavyMs });
    g = due(g);
    expect(view(g, t0 + F.heavyMs).fighters[1]).toMatchObject({ health: HEALTH - 2, lost: ['hat'] });
  });

  it('a block stops a jab; the view shows the fight as it stands at that moment, with no new message', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p2', { kind: 'guard', on: true }, t0);
    g = did(g, 'p1', JAB, t0 + 400);
    expect(act(g, 'p2', JAB, t0 + 450).error).toBe('Drop your guard to punch');
    expect(act(g, 'p2', { kind: 'dodge' }, t0 + 450).error).toBe('Drop your guard to dodge');
    // a screen asking after the punch was due sees it blocked, though the room's timer hasn't run yet
    const late = view(g, t0 + 400 + F.jabMs + 5, 'p2');
    expect(late.events.at(-1)).toMatchObject({ kind: 'blocked', by: 0 });
    expect(late.fighters[1].health).toBe(HEALTH);
    expect(late.mySeat).toBe(1);
    g = due(g);
    expect(st(g).fight.f[0].busy).toBe('blocked');
    g = did(g, 'p2', { kind: 'guard', on: false }, t0 + 400 + F.jabMs + 20);                    // dropping a guard is always allowed
  });

  it('a haymaker smashes through a block; a dodge makes it miss and leaves the puncher open', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    g = did(g, 'p2', { kind: 'guard', on: true }, t0);
    g = due(did(g, 'p1', HEAVY, t0 + 100));
    expect(st(g).fight.events.at(-1)).toMatchObject({ kind: 'smash', by: 0 });
    expect(st(g).fight.f[1]).toMatchObject({ health: HEALTH - F.smashDamage, busy: 'guard_break' });

    let h = bell(fight());
    h = did(h, 'p1', HEAVY, t0);
    h = did(h, 'p2', { kind: 'dodge' }, t0 + F.heavyMs - 150);       // timed to it
    expect(view(h, t0 + F.heavyMs - 100, 'p2').fighters[1].dodge).toBeTruthy();
    h = due(h);
    expect(st(h).fight.events.at(-1)).toMatchObject({ kind: 'dodged', by: 0, punch: 'heavy' });
    expect(st(h).fight.f[1].health).toBe(HEALTH);
    expect(st(h).fight.f[0].busy).toBe('whiff');
    expect(act(h, 'p1', JAB, t0 + F.heavyMs + 300).error).toBe('Not yet');                       // still stumbling
  });

  it('the last blow takes the head: the round is over, then the next round starts fresh', () => {
    let g = bell(fight());
    let now = NOW + DEFAULTS.introMs + 100;
    [g, now] = batter(g, 'p1', HEALTH - 1, now);
    expect(st(g).stage).toBe('fighting');
    expect(view(g, now).fighters[1]).toMatchObject({ lost: PARTS.slice(0, -1), loose: 'head' });   // everything but the head, and that is loose
    [g, now] = batter(g, 'p1', 1, now);
    expect(st(g)).toMatchObject({ stage: 'ko', wins: { p1: 1, p2: 0 }, last: { round: 1, winner: 'p1', loser: 'p2', how: 'knockout' } });
    expect(act(g, 'p2', JAB, now).error).toBe('Wait for the bell');
    expect(view(g, now).fighters[1].lost.at(-1)).toBe('head');
    g = bell(g);                                                    // the head has flown: round 2
    expect(st(g)).toMatchObject({ stage: 'intro', round: 2 });
    expect(st(g).fight.f.map((f) => f.health)).toEqual([HEALTH, HEALTH]);
    expect(arenaOf(g)!.result ?? null).toBeNull();
  });

  it('a round is no longer over in a handful of blows: twelve clean jabs, or six haymakers', () => {
    expect(HEALTH).toBe(12);
    expect(Math.ceil(HEALTH / F.jabDamage)).toBe(12);
    expect(Math.ceil(HEALTH / F.heavyDamage)).toBe(6);
    // even unanswered, that is several seconds of punching; a real exchange is much longer
    expect(12 * (F.jabMs + F.recoverMs)).toBeGreaterThan(7000);
    expect(DEFAULTS.roundMs).toBe(60_000);
  });

  it('best of three: two rounds win the match and the duel', () => {
    let g = bell(fight());
    let now = NOW + DEFAULTS.introMs + 100;
    [g, now] = batter(g, 'p2', HEALTH, now);                        // p2 takes round 1
    g = bell(bell(g));
    now = st(g).stageEndsAt - DEFAULTS.roundMs + 100;
    [g, now] = batter(g, 'p1', HEALTH, now);                        // p1 takes round 2
    g = bell(bell(g));
    expect(st(g).round).toBe(3);
    now = st(g).stageEndsAt - DEFAULTS.roundMs + 100;
    [g] = batter(g, 'p2', HEALTH, now);                             // p2 takes the decider
    expect(arenaOf(g)!.result ?? null).toBeNull();                  // not until the knockout has been seen
    g = bell(g);
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', winners: ['p2'], losers: ['p1'], duel: { winningDuelistId: 'p2', losingDuelistId: 'p1' } });
    expect(st(g).stage).toBe('done');
    const back = apply(g, handoff(g));
    expect(back.currentTurn?.duelGameWinner).toBe('p2');
  });

  it('the bell: more health left wins the round; level is sudden death, where one blow takes the head', () => {
    let g = bell(fight());
    const t0 = NOW + DEFAULTS.introMs + 100;
    [g] = batter(g, 'p1', 1, t0);
    g = due(g, st(g).stageEndsAt);
    expect(st(g)).toMatchObject({ stage: 'ko', last: { winner: 'p1', how: 'time' } });

    let h = bell(fight());
    h = due(h, st(h).stageEndsAt);                                  // nobody landed a thing
    expect(st(h)).toMatchObject({ stage: 'fighting', sudden: true });
    expect(st(h).fight.f.map((f) => f.health)).toEqual([1, 1]);
    expect(view(h, st(h).stageEndsAt - DEFAULTS.suddenMs + 10).fighters[0].lost).toEqual(PARTS.slice(0, -1));
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
    let g = bell(fight());
    let now = NOW + DEFAULTS.introMs + 100;
    [g, now] = batter(g, 'p1', HEALTH - 1, now);                    // one blow from the end
    g = did(g, 'p1', JAB, now);
    // p2's punch reaches the room after p1's has landed, before the room's own timer ran: the head is already off
    expect(act(g, 'p2', JAB, now + F.jabMs + 5).error).toBe('Wait for the bell');
    g = due(g);
    expect(st(g)).toMatchObject({ stage: 'ko', last: { winner: 'p1' } });
  });

  it('the settings are held to what a robot is: so many blows for each of its parts', () => {
    expect(robots.validateConfig({}).fight.health).toBe(HEALTH);
    expect(robots.validateConfig({ fight: { health: 99 } }).fight.health).toBe(HEALTH);          // health is never set directly
    expect(robots.validateConfig({ hitsPerPart: 1 }).fight.health).toBe(PARTS.length);
    expect(robots.validateConfig({ hitsPerPart: 99 }).fight.health).toBe(PARTS.length * 4);
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

  it('simulated hands can play a whole match to a result, and what is offered can always be pressed', () => {
    let g = fight();
    let now = NOW;
    const used = new Set<string>();
    for (let step = 0; step < 40_000 && !arenaOf(g)!.result; step++) {
      now += 40;
      const d = arenaOf(g)!.deadlines[0];
      if (d && d.at <= now) { g = apply(g, arenaDeadline(g, d.id, ctx(now, step))); continue; }
      for (const who of ['p1', 'p2']) {
        const [offer] = arenaOffers(g, who, now, rng(step * 7 + (who === 'p1' ? 1 : 2))) as { kind: string; punch?: string }[];
        if (!offer) continue;
        used.add(offer.punch || offer.kind);
        const plan = act(g, who, offer, now);
        expect(plan.error, JSON.stringify(offer)).toBeUndefined();
        g = apply(g, plan);
      }
    }
    const r = arenaOf(g)!.result!;
    expect(r).toBeTruthy();
    expect(['resolved', 'requires_tiebreaker']).toContain(r.status);
    expect([...used].sort()).toEqual(['dodge', 'guard', 'heavy', 'jab']);                       // they use everything there is
    expect(arenaOffers(g, 'p3', now, rng(1))).toEqual([]);
  });
});
