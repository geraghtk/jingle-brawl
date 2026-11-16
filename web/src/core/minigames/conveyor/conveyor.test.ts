// The Great Gift Conveyor on the runtime: two captain-led teams, each with a belt of unfinished toys
// rolling past its stations; a piece can only be fitted while the toy is at your station; a finished
// toy reaches the sleigh, anything else the bin; a belt speeds up with its own team's deliveries;
// the bell, overtime, the winning captain winning the duel, a point for the winning team only.
import { describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { posAt, zoneOf } from '../engines/belt';
import { assignTeams } from '../engines/teams';
import { arenaAct, arenaDeadline, arenaOf, arenaPause, arenaResume, handoff, startArena } from '../runtime';
import { conveyor, crewFor, DEFAULTS, PARTS, speedAt, VARIANTS, zones } from './conveyor';
import type { ConveyorAction, ConveyorState } from './conveyor';
import { conveyorView } from './view';

const st = (g: Game) => arenaOf(g)!.state as ConveyorState;
const ctx = (now: number) => ({ now, rng: rng(4) });
const ZONES = zones(DEFAULTS);

function play(n: number, config: Record<string, unknown> = {}, away: string[] = []): Game {
  const g = table(n);
  for (const id of away) g.players![id].connected = false;
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const teams = assignTeams(['p1', 'p2'], Object.keys(g.players!), rng(1));
  const out = startArena(tx, 'gift_conveyor', { mode: 'captain_teams', participants: Object.keys(g.players!), duelists: ['p1', 'p2'], config: { teams, stowawayChance: 0, ...config } }, ctx(NOW));
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
const mctx = (g: Game, now: number) => ({ ...ctx(now), instanceId: 'x', mode: 'captain_teams' as const, participants: [], duelists: ['p1', 'p2'] as [string, string], connected: (p: string) => g.players?.[p]?.connected !== false });
const offer = (g: Game, who: string, now: number) => conveyor.offers(mctx(g, now), st(g), who)[0] as ConveyorAction | undefined;
const act = (g: Game, who: string, action: unknown, now: number, aid?: string) => arenaAct(g, who, { action, aid }, ctx(now));
/** Everything the room's clock owes by `now`, in order. */
function tick(g: Game, now: number): Game {
  for (let i = 0; i < 5000 && !arenaOf(g)!.result; i++) {
    const d = [...arenaOf(g)!.deadlines].sort((a, b) => a.at - b.at)[0];
    if (!d || d.at > now) break;
    g = apply(g, arenaDeadline(g, d.id, ctx(d.at)));
  }
  return g;
}
/** From `from` to `to`, the listed teams' players each do the right thing with whatever is at their station
 *  (`slip`: fit a wrong piece instead). Returns the game. */
function work(g: Game, teams: number[], from: number, to: number, slip = false, noShoo = false): Game {
  for (let now = from; now <= to && !arenaOf(g)!.result; now += 100) {
    g = tick(g, now);
    if (arenaOf(g)!.result) break;
    for (const ti of teams) for (const p of st(g).teams[ti].members) {
      let o = offer(g, p, now);
      if (!o || (noShoo && o.kind === 'shoo')) continue;
      if (slip && o.kind === 'fit') o = { ...o, variant: (o.variant + 1) % VARIANTS };
      g = apply(g, act(g, p, o, now));
    }
  }
  return g;
}
const tv = (g: Game, now: number) => conveyorView((projectGame(g, { kind: 'tv' }, now).arenaView as { view: unknown }).view);
const phone = (g: Game, who: string, now: number) => conveyorView((projectGame(g, { kind: 'phone', pid: who }, now).arenaView as { view: unknown }).view);
const counts = (g: Game) => st(g).teams.map((t) => [t.delivered, t.rejected]);
/** When the first toy of a belt at level 0 is in the middle of station `k`. */
const atStation = (k: number) => NOW + (((ZONES[k].from + ZONES[k].to) / 2) / DEFAULTS.baseSpeed) * 1000;

describe('The Great Gift Conveyor', () => {
  it('captains lead opposite teams; every station on both belts has someone; a toy is on each belt at once', () => {
    const g = play(6);
    const [a, b] = st(g).teams;
    expect([a.captainId, b.captainId]).toEqual(['p1', 'p2']);
    for (const t of [a, b]) {
      expect(t.crew).toHaveLength(PARTS.length);
      expect(t.crew.every((c) => c.length >= 1 && c.every((p) => t.members.includes(p)))).toBe(true);
      expect(new Set(t.crew.flat())).toEqual(new Set(t.members));       // and everyone has a station
    }
    const first = tick(g, NOW);
    expect(st(first).teams.map((t) => t.belt.items.map((it) => it.id))).toEqual([['A-0'], ['B-0']]);
  });

  it('a small team doubles up its stations; a big team shares them', () => {
    expect(crewFor(['a', 'b'])).toEqual([['a'], ['b'], ['a'], ['b']]);
    expect(crewFor(['a'])).toEqual([['a'], ['a'], ['a'], ['a']]);
    expect(crewFor(['a', 'b', 'c', 'd', 'e', 'f'])).toEqual([['a', 'e'], ['b', 'f'], ['c'], ['d']]);
    const g = play(4);                                                // two a side
    expect(phone(g, 'p1', NOW).myStations).toHaveLength(2);
    expect(phone(g, 'p1', NOW).myTeam).toBe('A');
  });

  it('both teams get the same orders in the same order', () => {
    const g = work(play(6), [], NOW, NOW + 30_000);
    const wants = st(g).teams.map((t) => t.belt.items.map((it) => it.data.want.join('')));
    expect(wants[0]).toEqual(wants[1]);
    expect(wants[0].length).toBeGreaterThan(1);
  });

  it('a screen draws a toy where it is now from where it was and how fast the belt runs', () => {
    const g = tick(play(6), NOW);
    const v = tv(g, NOW);
    const t = v.teams[0];
    expect(t).toMatchObject({ speed: DEFAULTS.baseSpeed, asOf: NOW, level: 0 });
    expect(t.toys[0]).toMatchObject({ id: 'A-0', pos: 0, stowaway: false, fitted: PARTS.map(() => -1) });
    expect(posAt(t, t.toys[0].pos, NOW + 3000)).toBeCloseTo(0.3);
    expect(v.zones).toHaveLength(PARTS.length);
    expect(v.parts).toEqual([...PARTS]);
  });

  it('a piece can only be fitted while the toy is at YOUR station', () => {
    const g = play(6);
    const a = st(g).teams[0];
    const [first, second] = [a.crew[0][0], a.crew[1][0]];
    expect(first).not.toBe(second);
    // before it gets there
    expect(act(tick(g, NOW + 300), first, { kind: 'fit', toy: 'A-0', variant: 0 }, NOW + 300).error).toBe('It is between stations');
    const t0 = atStation(0);
    const at0 = tick(g, t0);
    expect(act(at0, second, { kind: 'fit', toy: 'A-0', variant: 0 }, t0).error).toBe('That is not your station');
    expect(act(at0, st(g).teams[1].members[0], { kind: 'fit', toy: 'A-0', variant: 0 }, t0).error).toBe('That one has gone by');   // the other team's belt
    expect(act(at0, first, { kind: 'fit', toy: 'A-0', variant: VARIANTS }, t0).error).toBe('Pick one of the pieces');
    expect(act(at0, first, { kind: 'fit', toy: 'A-0', variant: 1.5 }, t0).error).toBe('Pick one of the pieces');
    expect(act(at0, first, { kind: 'shoo', toy: 'A-0' }, t0).error).toBe('Nothing to shoo');
    expect(act(at0, first, { kind: 'load', toy: 'A-0' }, t0).error).toBe('Fit a piece, or shoo');
    const fitted = apply(at0, act(at0, first, { kind: 'fit', toy: 'A-0', variant: 2 }, t0));
    expect(st(fitted).teams[0].belt.items[0].data).toMatchObject({ fitted: [2, -1, -1, -1], by: [first, '', '', ''] });
    expect(act(fitted, first, { kind: 'fit', toy: 'A-0', variant: 1 }, t0 + 50).error).toBe('That piece is already on');
    // once it has rolled on, the first station can't reach it; the second can
    const t1 = atStation(1);
    const at1 = tick(fitted, t1);
    expect(act(at1, first, { kind: 'fit', toy: 'A-0', variant: 0 }, t1).error).toBe('That is not your station');
    expect(act(at1, second, { kind: 'fit', toy: 'A-0', variant: 0 }, t1).error).toBeUndefined();
  });

  it('a toy with every piece right goes on the sleigh; one nobody finished, or with a wrong piece, goes in the bin', () => {
    const idle = work(play(6), [], NOW, NOW + 11_000);                 // nobody touches it: it rides to the end
    expect(counts(idle)).toEqual([[0, 1], [0, 1]]);
    const right = work(play(6), [0], NOW, NOW + 11_000);
    expect(counts(right)).toEqual([[1, 0], [0, 1]]);
    const wrong = work(play(6), [0], NOW, NOW + 11_000, true);
    expect(counts(wrong)).toEqual([[0, 1], [0, 1]]);
    // credit: a share of the delivered toy to each who fitted a piece of it
    const credit = st(right).teams[0].credit;
    expect(Object.values(credit).reduce((n, x) => n + x, 0)).toBeCloseTo(1);
  });

  it('a belt speeds up with its OWN team\'s deliveries; the other team\'s belt does not', () => {
    const g = work(play(6, { timeLevelMs: 10_000_000 }), [0], NOW, NOW + 22_000);
    const [a, b] = tv(g, NOW + 22_000).teams;
    expect(a.delivered).toBeGreaterThanOrEqual(DEFAULTS.perLevel);
    expect(a.level).toBe(Math.floor(a.delivered / DEFAULTS.perLevel));
    expect(a.speed).toBeCloseTo(speedAt(DEFAULTS, a.level));
    expect(a.speed).toBeGreaterThan(DEFAULTS.baseSpeed);
    expect([b.level, b.speed]).toEqual([0, DEFAULTS.baseSpeed]);
    // toys keep their spacing on the faster belt (they enter by distance, not by time)
    const pos = st(g).teams[0].belt.items.map((it) => it.pos).sort((x, y) => y - x);
    for (let i = 1; i < pos.length; i++) expect(pos[i - 1] - pos[i]).toBeCloseTo(DEFAULTS.gap, 5);
  });

  it('both belts creep up as the round goes on, to a top speed', () => {
    const g = work(play(6, { roundMs: 400_000 }), [], NOW, NOW + 2 * DEFAULTS.timeLevelMs + 500);
    expect(tv(g, NOW + 2 * DEFAULTS.timeLevelMs + 500).teams.map((t) => t.level)).toEqual([2, 2]);
    expect(st(g).teams[0].belt.speed).toBeCloseTo(speedAt(DEFAULTS, 2));
    const late = work(g, [], NOW + 2 * DEFAULTS.timeLevelMs + 600, NOW + 300_000);
    expect(tv(late, NOW + 300_000).teams.map((t) => t.level)).toEqual([DEFAULTS.maxLevel, DEFAULTS.maxLevel]);
    // even at top speed a toy is at a station long enough to press a button
    expect((DEFAULTS.zone / speedAt(DEFAULTS, DEFAULTS.maxLevel)) * 1000).toBeGreaterThan(700);
  });

  it('first to the quota wins: the winning captain wins the duel; the winning team alone gets a point each', () => {
    let g = work(play(6, { quota: 2 }), [0], NOW, NOW + 40_000);
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', reason: 'first to 2 toys', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
    const teamA = st(g).teams[0].members, teamB = st(g).teams[1].members;
    expect(r.rewards!.map((x) => x.playerId).sort()).toEqual([...teamA].sort());
    expect(r.rewards!.map((x) => x.points)).toEqual(teamA.map(() => 1));
    expect(r.rewards!.some((x) => teamB.includes(x.playerId))).toBe(false);
    expect(st(g).stage).toBe('done');
    expect(tv(g, NOW + 50_000).teams[0]).toMatchObject({ speed: 0, toys: [] });   // the belts stop
    expect(act(g, 'p1', { kind: 'fit', toy: 'A-9', variant: 0 }, NOW + 50_000).error).toBeTruthy();
    const before = Object.fromEntries(teamA.map((p) => [p, g.players![p].partyPoints || 0]));
    g = apply(g, handoff(g));
    for (const p of teamA) expect(g.players![p].partyPoints).toBe(before[p] + 1);
    expect(g.currentTurn?.duelGameWinner).toBe('p1');
    expect(handoff(g).error).toBeTruthy();
  });

  it('the bell: more toys on the sleigh wins', () => {
    const g = work(play(6, { quota: 99 }), [1], NOW, NOW + DEFAULTS.roundMs + 200);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', reason: 'more toys on the sleigh at the bell', duel: { winningDuelistId: 'p2' } });
  });

  it('level at the bell: overtime, and the first team ahead wins', () => {
    let g = work(play(6, { quota: 99 }), [], NOW, NOW + DEFAULTS.roundMs + 200);      // nobody delivers: level
    expect(st(g)).toMatchObject({ stage: 'overtime' });
    expect(st(g).endsAt).toBe(NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs);
    expect(arenaOf(g)!.result ?? null).toBeNull();
    g = work(g, [1], NOW + DEFAULTS.roundMs + 300, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', reason: 'ahead in overtime', duel: { winningDuelistId: 'p2' } });
  });

  it('still level after overtime: fewer rejects; a dead level goes to the host', () => {
    const dead = work(play(6, { quota: 99 }), [], NOW, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs + 200);
    expect(arenaOf(dead)!.result).toMatchObject({ status: 'requires_tiebreaker', winners: [] });
    expect(apply(dead, handoff(dead)).currentTurn?.duelGameWinner ?? null).toBeNull();
    // team A wastes one more toy than team B (a piece fitted that makes no difference to B): decided on rejects
    let g = work(play(6, { quota: 99 }), [], NOW, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs - 300);
    const s = st(g);
    const fewer: ConveyorState = { ...s, teams: [{ ...s.teams[0], rejected: s.teams[0].rejected + 1 }, s.teams[1]] };
    g = { ...g, arena: { ...arenaOf(g)!, state: fewer } } as Game;
    g = tick(g, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs + 200);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', reason: 'level after overtime: fewer rejects', duel: { winningDuelistId: 'p2' } });
  });

  it('a raccoon on the belt: shooed off at any of your stations, or it makes off with a toy', () => {
    // every order after the first two is a raccoon; team A fits pieces but lets the raccoons ride
    const cfg = { stowawayChance: 1, quota: 99, timeLevelMs: 10_000_000 };
    const g = work(play(6, cfg), [0], NOW, NOW + 13_000, false, true);
    expect(st(g).teams[0].delivered).toBe(1);                       // the first toy; the second is nearly there
    const coon = st(g).teams[0].belt.items.find((it) => it.data.stowaway)!;
    expect(coon).toBeTruthy();
    expect(tv(g, NOW + 13_000).teams[0].toys.find((y) => y.id === coon.id)!.stowaway).toBe(true);
    // nobody can fit a piece on it (where it is NOW: the stored position is as of the belt's last event)
    const belt = st(g).teams[0].belt;
    const k = zoneOf(ZONES, posAt(belt, coon.pos, NOW + 13_000));
    expect(k).toBeGreaterThanOrEqual(0);
    const who = st(g).teams[0].crew[k][0];
    expect(act(g, who, { kind: 'fit', toy: coon.id, variant: 0 }, NOW + 13_000).error).toBe('That is a raccoon: shoo it off!');
    expect(act(g, st(g).teams[0].members.find((p) => !st(g).teams[0].crew[k].includes(p))!, { kind: 'shoo', toy: coon.id }, NOW + 13_000).error).toBe('That is not your station');
    // left alone, the raccoons ride to the end and each takes a toy off the sleigh
    const robbed = work(g, [0], NOW + 13_100, NOW + 24_000, false, true);
    expect(st(robbed).teams[0].stolen).toBeGreaterThanOrEqual(1);
    expect(st(robbed).teams[0].delivered).toBeLessThan(2);          // it delivered two in all; a raccoon never takes more than there is
    // shooed (the hands do it when one reaches a station of theirs), the sleigh keeps its toys
    const shooed = apply(g, act(g, who, { kind: 'shoo', toy: coon.id }, NOW + 13_000));
    expect(st(shooed).teams[0].belt.items.some((it) => it.id === coon.id)).toBe(false);
    const safe = work(shooed, [0], NOW + 13_100, NOW + 24_000);
    expect(st(safe).teams[0]).toMatchObject({ stolen: 0, delivered: 2 });
    // a team with nothing on the sleigh loses nothing
    const nothing = work(play(6, cfg), [], NOW, NOW + 40_000);
    expect(st(nothing).teams[1].delivered).toBe(0);
    expect(st(nothing).teams[1].stolen).toBeGreaterThan(0);
  });

  it('a phone that dropped out: a teammate may work its station', () => {
    const g0 = play(6);
    const a = st(g0).teams[0];
    const gone = a.crew[1][0], mate = a.members.find((p) => p !== gone && !a.crew[1].includes(p))!;
    const t1 = atStation(1);
    const there = tick(g0, t1);
    expect(act(there, mate, { kind: 'fit', toy: 'A-0', variant: 0 }, t1).error).toBe('That is not your station');
    const dropped = { ...there, players: { ...there.players!, [gone]: { ...there.players![gone], connected: false } } };
    expect(act(dropped, mate, { kind: 'fit', toy: 'A-0', variant: 0 }, t1).error).toBeUndefined();
    expect(act(dropped, st(g0).teams[1].members[0], { kind: 'fit', toy: 'A-0', variant: 0 }, t1).error).toBeTruthy();   // not the other team
  });

  it('a late press changes nothing: the room judges where the toy is on its own clock', () => {
    const g = play(6);
    const first = st(g).teams[0].crew[0][0];
    const leftZone = NOW + ((ZONES[0].to + 0.01) / DEFAULTS.baseSpeed) * 1000;
    expect(act(g, first, { kind: 'fit', toy: 'A-0', variant: 0 }, leftZone).error).toBe('It is between stations');
    // and a press for a toy that fell off the end
    expect(act(g, first, { kind: 'fit', toy: 'A-0', variant: 0 }, NOW + 11_000).error).toBe('That one has gone by');
  });

  it('the host pauses: the belts stand still on every screen, and pick up where they were', () => {
    let g = work(play(6, { quota: 99 }), [0], NOW, NOW + 6000);
    const before = tv(g, NOW + 6000).teams[0];
    const at = (t: typeof before, now: number) => t.toys.map((y) => Math.round(posAt(t, y.pos, now) * 1000));
    const where = at(before, NOW + 6000);
    const left = st(g).endsAt - (NOW + 6000);
    g = apply(g, arenaPause(g, NOW + 6000));
    // a minute into the pause every screen still gets the belts as they stood, and is told the clock stands
    const held = projectGame(g, { kind: 'tv' }, NOW + 66_000).arenaView!;
    expect(held.pausedAt).toBe(NOW + 6000);
    const frozen = conveyorView(held.view).teams[0];
    expect(at(frozen, held.pausedAt!)).toEqual(where);
    expect(act(g, 'p1', { kind: 'fit', toy: 'A-0', variant: 0 }, NOW + 30_000).error).toMatch(/paused/);
    // resumed two minutes later: nothing moved meanwhile, the round has as long left as it had
    g = apply(g, arenaResume(g, NOW + 126_000));
    const after = projectGame(g, { kind: 'tv' }, NOW + 126_000).arenaView!;
    expect(after.pausedAt).toBeUndefined();
    expect(at(conveyorView(after.view).teams[0], NOW + 126_000)).toEqual(where);
    expect(st(g).endsAt - (NOW + 126_000)).toBe(left);
    expect(arenaOf(g)!.result ?? null).toBeNull();
    // and it carries on: a toy that was 3 s from the end reaches it 3 s after the resume, not at once
    const on = work(g, [0], NOW + 126_000, NOW + 126_000 + 6000);
    expect(st(on).teams[0].delivered).toBe(1);
    expect(tv(on, NOW + 132_000).teams.map((t) => t.level)).toEqual([0, 0]);   // the round is 12 s old, not two minutes
  });

  it('the bell is its deadline coming due: the room\'s clock decides, whatever the state had noted', () => {
    let g = work(play(6, { quota: 99 }), [0], NOW, NOW + 12_000);
    expect(st(g).teams[0].delivered).toBe(1);
    const end = arenaOf(g)!.deadlines.find((d) => d.id === 'end')!;
    // the host's console (or a test) brings the bell forward
    g = { ...g, arena: { ...arenaOf(g)!, deadlines: [{ id: 'end', at: NOW + 13_000 }] } } as Game;
    expect(end.at).toBeGreaterThan(NOW + 13_000);
    g = apply(g, arenaDeadline(g, 'end', ctx(NOW + 13_000)));
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', reason: 'more toys on the sleigh at the bell' });
  });

  it('settling the belts in one go or step by step gives the same game', () => {
    const fine = work(play(6, { quota: 99 }), [0, 1], NOW, NOW + 40_000);
    const coarse = tick(work(play(6, { quota: 99 }), [0, 1], NOW, NOW + 40_000), NOW + 40_000);
    expect(counts(coarse)).toEqual(counts(fine));
  });

  it('as a duel game: the duelists captain, everyone at the table is dealt onto a team', () => {
    const g0 = table(6, { duelPick: 'defender' });
    g0.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'The Great Gift Conveyor', kind: 'app', app: 'gift_conveyor', via: 'picker' } };
    const g = apply(g0, startDuelApp(g0, { now: NOW, rng: rng(2) }));
    expect(arenaOf(g)!.mode).toBe('captain_teams');
    expect(st(g).teams.map((t) => t.captainId)).toEqual(['p2', 'p1']);
    expect(st(g).teams.flatMap((t) => t.members).sort()).toEqual(['p1', 'p2', 'p3', 'p4', 'p5', 'p6']);
    expect(Object.values(g.teamPairs || {}).reduce((n, x) => n + x, 0)).toBe(6);     // 3 + 3 pairs within two teams of three
  });

  it('bad settings are refused', () => {
    expect(() => conveyor.validateConfig({ quota: 0 })).toThrow();
    expect(() => conveyor.validateConfig({ zone: 0.5 })).toThrow();            // stations would overlap
    expect(() => conveyor.validateConfig({ baseSpeed: 0 })).toThrow();
    expect(conveyor.validateConfig({}).quota).toBe(DEFAULTS.quota);
  });
});
