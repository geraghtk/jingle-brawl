// The Great Gift Conveyor on the runtime: two captain-led teams, the same orders for both, gifts
// passing phone to phone with one holder at a time, only right gifts counting, stowaways to shoo,
// the bell and overtime, the winning captain winning the duel, and rewards only for the winning team.
import { describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { assignTeams } from '../engines/teams';
import { arenaAct, arenaDeadline, arenaOf, handoff, startArena } from '../runtime';
import { conveyor, STEPS } from './conveyor';
import type { ConveyorAction, ConveyorState } from './conveyor';

const st = (g: Game) => arenaOf(g)!.state as ConveyorState;
const ctx = (now: number) => ({ now, rng: rng(4) });

function play(n: number, config: Record<string, unknown> = {}): Game {
  let g = table(n);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const teams = assignTeams(['p1', 'p2'], Object.keys(g.players!), rng(1));
  const out = startArena(tx, 'gift_conveyor', { mode: 'captain_teams', participants: Object.keys(g.players!), duelists: ['p1', 'p2'], config: { teams, stowawayChance: 0, ...config } }, ctx(NOW));
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
const offer = (g: Game, who: string, now: number) => conveyor.offers({ ...ctx(now), instanceId: 'x', mode: 'captain_teams', participants: [], connected: () => true }, st(g), who)[0];
const act = (g: Game, who: string, action: unknown, now = NOW + 1000, aid?: string) => arenaAct(g, who, { action, aid }, ctx(now));
/** Everyone on team `ti` does the right thing with whatever is on their table, once. */
function workTeam(g: Game, ti: number, now: number): Game {
  for (const p of st(g).teams[ti].members) {
    const o = offer(g, p, now);
    if (o) g = apply(g, act(g, p, o, now));
  }
  return g;
}
const deliveredBy = (g: Game) => st(g).teams.map((t) => t.delivered);

describe('The Great Gift Conveyor', () => {
  it('captains lead opposite teams; the same number of stations for both (the smaller team); a gift on the first table', () => {
    const g = play(5);
    const [a, b] = st(g).teams;
    expect([a.captainId, b.captainId]).toEqual(['p1', 'p2']);
    expect(a.line.stations.length).toBe(b.line.stations.length);
    expect(a.line.stations.length).toBe(Math.min(a.members.length, b.members.length));
    expect(Object.values(a.floor.items)).toHaveLength(1);
    expect(Object.values(a.floor.items)[0].holder).toBe(a.line.workers[0][0]);
  });

  it('a stuck station holds the line up: never more gifts on the floor than stations', () => {
    let g = play(4, { quota: 9 });                             // 2 a side: two stations each
    let now = NOW + 500;
    const a = st(g).teams[0];
    const first = a.line.workers[0][0];
    // only the first station works; the second never does
    for (let i = 0; i < 30; i++) { const o = offer(g, first, now); if (o) g = apply(g, act(g, first, o, now)); now += 50; }
    expect(Object.keys(st(g).teams[0].floor.items).length).toBeLessThanOrEqual(a.line.stations.length);
  });

  it('both teams work the same orders in the same order', () => {
    const g = play(4);
    const [a, b] = st(g).teams;
    const first = (t: typeof a) => Object.values(t.floor.items)[0].data.n;
    expect(first(a)).toBe(first(b));
    expect(st(g).orders.length).toBeGreaterThanOrEqual(st(g).cfg.quota * 2);
  });

  it('a gift goes phone to phone, one holder at a time; a resend or the wrong phone does nothing', () => {
    let g = play(4);
    const a = st(g).teams[0];
    const [w0, w1] = a.line.workers.map((w) => w[0]);
    const gift = Object.values(a.floor.items)[0];
    const pick = offer(g, w0, NOW + 500) as ConveyorAction;
    expect(act(g, w1, pick).error).toMatch(/isn't on your table/);
    g = apply(g, act(g, w0, pick, NOW + 500, 'tap-1'));
    expect(act(g, w0, pick, NOW + 600, 'tap-1').error).toMatch(/Already done/);
    const moved = st(g).teams[0].floor.items[gift.id];
    expect(moved.step).toBe(1);
    expect(moved.holder === w0 || moved.holder === w1 || moved.holder === null).toBe(true);
    expect(Object.values(st(g).teams[0].floor.items).filter((it) => it.id === gift.id)).toHaveLength(1);
  });

  it('only a right gift counts: a wrong toy loaded is a wrong gift, not a delivery; sending it back is a fix', () => {
    let g = play(2);                                         // one each: a captain works the whole line
    const me = 'p1';
    const it0 = Object.values(st(g).teams[0].floor.items)[0];
    const o = st(g).orders[it0.data.n];
    const wrongToy = o.options.toy.find((t) => t !== o.toy)!;
    g = apply(g, act(g, me, { kind: 'pick', item: it0.id, choice: wrongToy }));
    g = apply(g, act(g, me, { kind: 'wrap', item: it0.id, choice: o.paper }));
    g = apply(g, act(g, me, { kind: 'label', item: it0.id, choice: o.tag }));
    expect(offer(g, me, NOW + 1000)).toEqual({ kind: 'back', item: it0.id });  // the checker should catch it
    g = apply(g, act(g, me, { kind: 'load', item: it0.id }));
    expect(st(g).teams[0]).toMatchObject({ delivered: 0, wrong: 1 });
    // the next one, wrong again but sent back: a fix, and it comes round again empty
    const it1 = Object.values(st(g).teams[0].floor.items)[0];
    const o1 = st(g).orders[it1.data.n];
    g = apply(g, act(g, me, { kind: 'pick', item: it1.id, choice: o1.options.toy.find((t) => t !== o1.toy)! }));
    g = apply(g, act(g, me, { kind: 'wrap', item: it1.id, choice: o1.paper }));
    g = apply(g, act(g, me, { kind: 'label', item: it1.id, choice: o1.tag }));
    g = apply(g, act(g, me, { kind: 'back', item: it1.id }));
    expect(st(g).teams[0].floor.items[it1.id]).toMatchObject({ step: 0, data: { toy: null } });
    expect(st(g).teams[0].fixes.p1).toBe(1);
    expect(act(g, me, { kind: 'pick', item: it1.id, choice: 99 }).error).toMatch(/Pick one of these/);
  });

  it('a stowaway must be shooed before the gift can go on the sleigh', () => {
    let g = play(2, { stowawayChance: 1 });
    const me = 'p1';
    const it = Object.values(st(g).teams[0].floor.items)[0];
    const o = st(g).orders[it.data.n];
    for (const [kind, choice] of [['pick', o.toy], ['wrap', o.paper], ['label', o.tag]] as const) g = apply(g, act(g, me, { kind, item: it.id, choice }));
    expect(act(g, me, { kind: 'load', item: it.id }).error).toMatch(/Shoo the (raccoon|penguin) out first/);
    g = apply(g, act(g, me, { kind: 'shoo', item: it.id }));
    g = apply(g, act(g, me, { kind: 'load', item: it.id }));
    expect(st(g).teams[0].delivered).toBe(1);
  });

  it("state that lost its nulls (as stored, or as a screen's view gets it) plays the same: a shooed box can be loaded", () => {
    let g = play(2, { stowawayChance: 1 });
    const me = 'p1';
    const it0 = Object.values(st(g).teams[0].floor.items)[0];
    const o = st(g).orders[it0.data.n];
    for (const [kind, choice] of [['pick', o.toy], ['wrap', o.paper], ['label', o.tag]] as const) g = apply(g, act(g, me, { kind, item: it0.id, choice }));
    g = apply(g, act(g, me, { kind: 'shoo', item: it0.id }));
    const dropNulls = (v: unknown): unknown => (Array.isArray(v) ? v.map(dropNulls) : v && typeof v === 'object'
      ? Object.fromEntries(Object.entries(v).filter(([, x]) => x !== null).map(([k, x]) => [k, dropNulls(x)])) : v);
    g = { ...g, arena: dropNulls(g.arena) as Game['arena'] };
    expect(st(g).teams[0].floor.items[it0.id].data.stowaway).toBeUndefined();
    g = apply(g, act(g, me, { kind: 'load', item: it0.id }));
    expect(st(g).teams[0].delivered).toBe(1);
  });

  it('first to the quota wins: the winning captain wins the duel; the winning team alone gets rewards, by contribution', () => {
    let g = play(6, { quota: 2 });
    let now = NOW + 500;
    for (let i = 0; i < 200 && !arenaOf(g)!.result; i++) { g = workTeam(g, 0, now); now += 50; }
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
    const teamA = st(g).teams[0].members, teamB = st(g).teams[1].members;
    expect(r.rewards!.map((x) => x.playerId).sort()).toEqual([...teamA].sort());
    expect(r.rewards!.every((x) => x.points >= 2)).toBe(true);
    expect(r.rewards!.some((x) => teamB.includes(x.playerId))).toBe(false);
    // the experience applies them, once
    const before = Object.fromEntries(teamA.map((p) => [p, g.players![p].partyPoints || 0]));
    const back = apply(g, handoff(g));
    for (const x of r.rewards!) expect(back.players![x.playerId].partyPoints).toBe(before[x.playerId] + x.points);
    expect(back.currentTurn?.duelGameWinner).toBe('p1');
    expect(handoff(back).error).toBeTruthy();
  });

  it('the bell: more gifts wins; level means overtime; still level, fewer wrong gifts; still level, the host', () => {
    let g = play(2, { quota: 9 });
    let d = arenaOf(g)!.deadlines.find((x) => x.id === 'end')!;
    g = apply(g, arenaDeadline(g, 'end', ctx(d.at)));
    expect(st(g).stage).toBe('overtime');
    d = arenaOf(g)!.deadlines.find((x) => x.id === 'end')!;
    g = apply(g, arenaDeadline(g, 'end', ctx(d.at)));
    expect(arenaOf(g)!.result).toMatchObject({ status: 'requires_tiebreaker' });
    expect(arenaOf(g)!.result!.rewards).toBeUndefined();
  });

  it("a dropped phone's gift goes to a teammate at the same station; with nobody there, it waits", () => {
    let g = play(7);                                          // 4 against 3: three stations, two people on the bigger team's first
    const big = st(g).teams.findIndex((t) => t.members.length === 4);
    const crew = st(g).teams[big].line.workers[0];
    expect(crew).toHaveLength(2);
    const holder = Object.values(st(g).teams[big].floor.items)[0].holder!;
    g.players![holder].connected = false;
    let tick = arenaOf(g)!.deadlines.find((x) => x.id === 'tick')!;
    g = apply(g, arenaDeadline(g, 'tick', ctx(tick.at)));
    expect(Object.values(st(g).teams[big].floor.items)[0].holder).toBe(crew.find((p) => p !== holder));
    // the other team's first station has one person: if they drop, the gift waits in the tray
    const small = 1 - big;
    const solo = st(g).teams[small].line.workers[0][0];
    g.players![solo].connected = false;
    tick = arenaOf(g)!.deadlines.find((x) => x.id === 'tick')!;
    g = apply(g, arenaDeadline(g, 'tick', ctx(tick.at)));
    expect(Object.values(st(g).teams[small].floor.items).find((it) => it.step === 0)!.holder).toBeNull();
  });

  it("each phone sees its own job (the order and its choices); the TV sees both factories, no one's job", () => {
    const g = play(4);
    const w = st(g).teams[0].line.workers[0][0];
    const mine = projectGame(g, { kind: 'phone', pid: w }, NOW).arenaView!.view as { job: { step: string; options: { toy: number[] } } | null; myTeam: string };
    expect(mine.myTeam).toBe('A');
    expect(mine.job!.step).toBe(STEPS[0]);
    expect(mine.job!.options.toy).toHaveLength(4);
    const tv = projectGame(g, { kind: 'tv' }, NOW).arenaView!.view as { teams: { delivered: number; gifts: unknown[] }[]; job?: unknown };
    expect(tv.teams).toHaveLength(2);
    expect(tv.job).toBeUndefined();
  });

  it('the simulated hands can play a whole round to a result', () => {
    let g = play(5, { quota: 3 });
    let now = NOW + 300;
    for (let i = 0; i < 400 && !arenaOf(g)!.result; i++) { g = workTeam(g, i % 2, now); now += 40; }
    expect(arenaOf(g)!.result?.status).toBe('resolved');
    expect(Math.max(...deliveredBy(g))).toBe(3);
  });

  it('starts as a duel game: captains are the duelists, everyone at the table is dealt in, and who played together is remembered', () => {
    const g0 = table(6, { duelPick: 'off', duelAppGames: ['gift_conveyor'], voteWindow: 0 });
    g0.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'The Great Gift Conveyor', kind: 'app', app: 'gift_conveyor', via: 'picker' } };
    const g = apply(g0, startDuelApp(g0, { now: NOW, rng: rng(2) }));
    const a = arenaOf(g)!;
    expect(a.mode).toBe('captain_teams');
    expect(st(g).teams.map((t) => t.captainId)).toEqual(['p2', 'p1']);
    expect(st(g).teams.flatMap((t) => t.members).sort()).toEqual(['p1', 'p2', 'p3', 'p4', 'p5', 'p6']);
    expect(Object.values(g.teamPairs || {}).reduce((n, x) => n + x, 0)).toBe(6);     // 3 + 3 pairs within two teams of three
  });
});
