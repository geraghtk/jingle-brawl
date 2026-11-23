// Mecha Santa Mayhem on the runtime: two captain-led teams, one robot each, four jobs dealt to
// whoever is on the team; only your own job's buttons work (a dropped phone's job opens to its team);
// first over the line wins and its captain wins the duel; the bell, overtime, a point for the winning
// team only; pause; the same game however the clock is stepped.
import { describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { posAt } from '../engines/walker';
import { assignTeams } from '../engines/teams';
import { arenaAct, arenaDeadline, arenaOf, arenaPause, arenaResume, handoff, startArena } from '../runtime';
import { CLASSIC_COURSE, DEFAULTS, mecha } from './mecha';
import { courseFaults } from './course';
import { NEEDS } from './view';
import type { MechaAction, MechaState } from './mecha';
import { JOBS, mechaView } from './view';

const st = (g: Game) => arenaOf(g)!.state as MechaState;
const ctx = (now: number) => ({ now, rng: rng(4) });
const OPEN = { obstacles: [], length: 20 };

function play(n: number, config: Record<string, unknown> = {}, away: string[] = []): Game {
  const g = table(n);
  for (const id of away) g.players![id].connected = false;
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const teams = assignTeams(['p1', 'p2'], Object.keys(g.players!), rng(1));
  const out = startArena(tx, 'mecha_santa', { mode: 'captain_teams', participants: Object.keys(g.players!), duelists: ['p1', 'p2'], config: { teams, ...config } }, ctx(NOW));
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
const mctx = (g: Game, now: number) => ({ ...ctx(now), instanceId: 'x', mode: 'captain_teams' as const, participants: [], duelists: ['p1', 'p2'] as [string, string], connected: (p: string) => g.players?.[p]?.connected !== false });
const offers = (g: Game, who: string, now: number) => mecha.offers(mctx(g, now), st(g), who) as MechaAction[];
const act = (g: Game, who: string, action: unknown, now: number) => arenaAct(g, who, { action }, ctx(now));
/** Everything the room's clock owes by `now`, in order. */
function tick(g: Game, now: number): Game {
  for (let i = 0; i < 5000 && !arenaOf(g)!.result; i++) {
    const d = [...arenaOf(g)!.deadlines].sort((a, b) => a.at - b.at)[0];
    if (!d || d.at > now) break;
    g = apply(g, arenaDeadline(g, d.id, ctx(d.at)));
  }
  return g;
}
/** From `from` to `to`, the listed teams' players each do what their job calls for. */
function work(g: Game, teams: number[], from: number, to: number, skip: (a: MechaAction) => boolean = () => false): Game {
  for (let now = from; now <= to && !arenaOf(g)!.result; now += 60) {
    g = tick(g, now);
    if (arenaOf(g)!.result) break;
    for (const ti of teams) for (const p of st(g).teams[ti].members) {
      const o = offers(g, p, now).filter((a) => !skip(a))[0];
      if (o) g = apply(g, act(g, p, o, now));
    }
  }
  return g;
}
const tv = (g: Game, now: number) => mechaView((projectGame(g, { kind: 'tv' }, now).arenaView as { view: unknown }).view);
const phone = (g: Game, who: string, now: number) => mechaView((projectGame(g, { kind: 'phone', pid: who }, now).arenaView as { view: unknown }).view);

describe('Mecha Santa Mayhem', () => {
  it('two teams under the captains, a robot each; every job has someone, whatever the team size', () => {
    for (const n of [4, 5, 7, 10]) {
      const s = st(play(n));
      expect(s.teams.map((t) => t.captainId)).toEqual(['p1', 'p2']);
      for (const t of s.teams) {
        expect(t.crew).toHaveLength(JOBS.length);
        for (const c of t.crew) expect(c.length).toBeGreaterThan(0);
        expect([...new Set(t.crew.flat())].sort()).toEqual([...t.members].sort());   // everyone has a job
      }
    }
    const two = st(play(4)).teams[0];
    expect(two.crew).toEqual([[two.members[0]], [two.members[1]], [two.members[0]], [two.members[1]]]);   // a leg and one more each
  });

  it('only your own job: the left leg cannot step the right one or push the balance', () => {
    const g = play(8);
    const t = st(g).teams[0];
    const [left, right, balance, arms] = t.crew.map((c) => c[0]);
    expect(act(g, right, { kind: 'step', leg: 0 }, NOW + 100).error).toMatch(/not your job/);
    expect(act(g, left, { kind: 'lean', dir: 1 }, NOW + 100).error).toMatch(/not your job/);
    expect(act(g, balance, { kind: 'smash' }, NOW + 100).error).toMatch(/not your job/);
    expect(act(g, arms, { kind: 'smash' }, NOW + 100).error).toBeUndefined();                // theirs: the arms swing even at nothing (and it lurches)
    expect(st(apply(g, act(g, arms, { kind: 'smash' }, NOW + 100))).teams[0].walker.events.at(-1)).toMatchObject({ kind: 'whiff' });
    expect(act(g, left, { kind: 'jump', leg: 0 }, NOW + 100).error).toBeUndefined();          // and a leg may jump with no gap in sight
    expect(act(g, left, { kind: 'step', leg: 0 }, NOW + 100).error).toBeUndefined();
    expect(act(g, left, { kind: 'step', leg: 2 }, NOW + 100).error).toMatch(/Not a robot move/);
    expect(act(g, 'nobody', { kind: 'step', leg: 0 }, NOW + 100).error).toBeTruthy();
  });

  it('a step moves only your own team’s robot, and the TV sees it part way through the step', () => {
    let g = play(8, OPEN);
    const left = st(g).teams[0].crew[0][0];
    g = apply(g, act(g, left, { kind: 'step', leg: 0 }, NOW + 100));
    const v = tv(g, NOW + 100 + DEFAULTS.walk.stepMs / 2);
    expect(posAt(v.teams[0].walker, NOW + 100 + DEFAULTS.walk.stepMs / 2)).toBeCloseTo(DEFAULTS.walk.stride / 2);
    expect(v.teams[1].walker.pos).toBe(0);
    expect(v.teams[0].walker.events.at(-1)).toMatchObject({ kind: 'step', leg: 0 });
    expect(arenaOf(g)!.deadlines.find((d) => d.id === 'tick')!.at).toBe(NOW + 100 + DEFAULTS.walk.stepMs);   // the room wakes for the landing
  });

  it("a phone knows its team and its jobs; nothing about the robots is secret", () => {
    const g = play(8);
    const t = st(g).teams[1];
    const p = phone(g, t.crew[2][0], NOW + 50);
    expect(p).toMatchObject({ myTeam: 'B', myJobs: [2] });
    expect(p.teams).toHaveLength(2);
    expect(p.course.obstacles.length).toBeGreaterThan(5);                    // (a course of its own: below)
    expect(tv(g, NOW + 50).myTeam ?? null).toBeNull();
    const small = play(4);
    expect(phone(small, st(small).teams[0].members[0], NOW).myJobs).toEqual([0, 2]);
  });

  it('first over the line wins: its captain wins the duel, its team a Party point each, the other team none', () => {
    let g = play(8, OPEN);
    g = work(g, [0], NOW + 100, NOW + 60_000);
    const r = arenaOf(g)!.result!;
    const [a, b] = st(g).teams;
    expect(r).toMatchObject({ status: 'resolved', reason: 'first over the line', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
    expect(st(g)).toMatchObject({ stage: 'done', winner: 'A' });
    expect([...r.winners].sort()).toEqual([...a.members].sort());
    expect(r.rewards!.map((x) => x.playerId).sort()).toEqual([...a.members].sort());
    expect(r.rewards!.every((x) => x.points === 1)).toBe(true);
    expect(r.rewards!.some((x) => b.members.includes(x.playerId))).toBe(false);
    expect(r.performance!.find((x) => x.playerId === a.crew[0][0])!.contribution).toBeGreaterThan(0);
    // the hand-off pays it once
    g.currentTurn = { ...g.currentTurn!, phase: 'minigame_arena' };
    const after = apply(g, handoff(g));
    expect(after.players![a.members[0]].partyPoints || 0).toBe(1);
    expect(after.players![b.members[0]].partyPoints || 0).toBe(0);
    expect(after.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p1' });
  });

  it('the whole course (walls, gaps, boosts) can be walked by a team doing its jobs', () => {
    let g = play(8, { obstacles: CLASSIC_COURSE });
    g = work(g, [0], NOW + 100, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs + 1000);
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', reason: 'first over the line', duel: { winningDuelistId: 'p1' } });
    const w = st(g).teams[0].walker;
    expect(Object.keys(w.hits).sort()).toEqual(['w1', 'w2', 'w3']);
    expect(w.taken.length).toBeGreaterThan(0);
    expect(w.pos).toBe(DEFAULTS.length);
  });

  it('the bell: further along wins; a team that never moves loses to one that took a step', () => {
    let g = play(4, OPEN);
    const left = st(g).teams[1].crew[0][0];
    g = apply(g, act(g, left, { kind: 'step', leg: 0 }, NOW + 100));
    g = tick(g, NOW + DEFAULTS.roundMs);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', reason: 'further along at the bell', duel: { winningDuelistId: 'p2' } });
  });

  it('dead level at the bell: a little longer, once; still level: the host decides, nobody is paid', () => {
    let g = play(4, OPEN);
    g = tick(g, NOW + DEFAULTS.roundMs);
    expect(arenaOf(g)!.result).toBeUndefined();
    expect(st(g)).toMatchObject({ stage: 'overtime', endsAt: NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs });
    const moved = work(g, [0], NOW + DEFAULTS.roundMs + 100, NOW + DEFAULTS.roundMs + 3000);
    expect(arenaOf(tick(moved, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs))!.result).toMatchObject({ status: 'resolved', reason: 'further along after overtime', duel: { winningDuelistId: 'p1' } });
    g = tick(g, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs);
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'requires_tiebreaker', winners: [] });
    expect(r.rewards).toBeUndefined();
  });

  it("a dropped phone's job is open to its team, and only then", () => {
    const g0 = play(8);
    const t = st(g0).teams[0];
    const [left, right] = [t.crew[0][0], t.crew[1][0]];
    expect(act(g0, right, { kind: 'step', leg: 0 }, NOW + 100).error).toMatch(/not your job/);
    const g = play(8, {}, [left]);
    expect(act(g, right, { kind: 'step', leg: 0 }, NOW + 100).error).toBeUndefined();
    expect(offers(g, right, NOW + 100)).toContainEqual({ kind: 'step', leg: 0 });
    expect(act(g, st(g).teams[1].crew[1][0], { kind: 'step', leg: 0 }, NOW + 100).error).toMatch(/not your job/);   // the other team's: never
  });

  it('two who share a job may both press whenever (nothing takes turns for them): they must agree, or the leg goes twice and it stumbles', () => {
    let g = play(10, OPEN);                                            // five a side: the left leg is shared
    const t = st(g).teams[0];
    const [first, second] = t.crew[0];
    expect(t.crew[0]).toHaveLength(2);
    expect(act(g, second, { kind: 'step', leg: 0 }, NOW + 100).error).toBeUndefined();   // either of them
    g = apply(g, act(g, first, { kind: 'step', leg: 0 }, NOW + 100));
    g = tick(g, NOW + 1000);
    g = apply(g, act(g, second, { kind: 'step', leg: 0 }, NOW + 1000));                  // "I thought you had it": the same leg again
    expect(st(g).teams[0].walker).toMatchObject({ busy: 'stumble' });
    // (a simulated pair take it in turns of their own accord, so bots sharing a leg do not trip each other up)
    const fresh = play(10, OPEN);
    expect(offers(fresh, first, NOW + 100)).toContainEqual({ kind: 'step', leg: 0 });
    expect(offers(fresh, second, NOW + 100)).toEqual([]);
  });

  it('a team of two or three has fewer hands for the same jobs: its robot strides a little further', () => {
    const small = st(play(5, OPEN));                                   // two against three
    const sizes = small.teams.map((t) => t.members.length);
    expect([...sizes].sort()).toEqual([2, 3]);
    small.teams.forEach((t) => expect(t.walk.stride).toBeCloseTo(DEFAULTS.walk.stride * DEFAULTS.lightFoot[t.members.length]));
    expect(st(play(8)).teams[0].walk.stride).toBe(DEFAULTS.walk.stride);
    expect(tv(play(5), NOW).teams[0].walk.stride).toBeGreaterThan(DEFAULTS.walk.stride);
  });

  it('a pause holds everything: the robots stand where they were, the bell and a step in the air wait', () => {
    let g = play(8, OPEN);
    const left = st(g).teams[0].crew[0][0];
    g = apply(g, act(g, left, { kind: 'step', leg: 0 }, NOW + 1000));
    g = apply(g, arenaPause(g, NOW + 1200));
    const held = projectGame(g, { kind: 'tv' }, NOW + 50_000).arenaView!;
    expect(held.pausedAt).toBe(NOW + 1200);
    expect(posAt(mechaView(held.view).teams[0].walker, held.pausedAt!)).toBeCloseTo(DEFAULTS.walk.stride * 200 / DEFAULTS.walk.stepMs);
    g = apply(g, arenaResume(g, NOW + 61_200));
    expect(st(g).endsAt).toBe(NOW + DEFAULTS.roundMs + 60_000);
    expect(st(g).teams[0].walker.move!.lands).toBe(NOW + 1000 + DEFAULTS.walk.stepMs + 60_000);
  });

  it('settling in one go or step by step gives the same game', () => {
    let a = play(8), b = play(8);
    const left = st(a).teams[0].crew[0][0];
    a = apply(a, act(a, left, { kind: 'step', leg: 0 }, NOW + 500));
    b = apply(b, act(b, left, { kind: 'step', leg: 0 }, NOW + 500));
    a = tick(a, NOW + 40_000);
    for (let t = NOW + 600; t <= NOW + 40_000; t += 333) b = tick(b, t);
    b = tick(b, NOW + 40_000);
    expect(tv(a, NOW + 40_000)).toEqual(tv(b, NOW + 40_000));
  });

  it('starts as the duel game with four at the table: captains and teams, never a head to head', () => {
    const g = table(4, { duelPick: 'off', duelAppGames: ['mecha_santa'], voteWindow: 0 });
    g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Mecha Santa Mayhem', kind: 'app', app: 'mecha_santa', via: 'picker' } };
    const started = apply(g, startDuelApp(g, ctx(NOW)));
    expect(arenaOf(started)).toMatchObject({ gameId: 'mecha_santa', mode: 'captain_teams', duelists: ['p2', 'p1'] });
    expect(st(started).teams.map((t) => t.captainId)).toEqual(['p2', 'p1']);
    const tx = new Tx(table(2));
    expect(startArena(tx, 'mecha_santa', { mode: 'direct_duel', participants: ['p1', 'p2'], duelists: ['p1', 'p2'] }, ctx(NOW)).error).toBeTruthy();
  });

  it('refuses settings that make no course', () => {
    expect(() => mecha.validateConfig({ length: 0 })).toThrow();
    expect(() => mecha.validateConfig({ obstacles: [{ id: 'g', kind: 'gap', at: 90, width: 10 }] })).toThrow();
    expect(mecha.validateConfig({}).walk.stride).toBe(DEFAULTS.walk.stride);
  });
});

describe('Mecha Santa: a new course every race, with climbs, ropes and ice', () => {
  it('every race gets a course of its own that keeps the rules; a course given by hand is used as it is', () => {
    const a = st(play(8)).course;
    expect(courseFaults(a.obstacles, a.length)).toEqual([]);
    for (const kind of ['wall', 'gap', 'ice', 'climb', 'rope', 'boost']) expect(a.obstacles.some((o) => o.kind === kind), kind).toBe(true);
    expect(tv(play(8), NOW).course.obstacles).toEqual(a.obstacles);            // the screens get it (one course: both robots are drawn on it)
    expect(st(play(8, { obstacles: CLASSIC_COURSE })).course.obstacles).toEqual(CLASSIC_COURSE);
    expect(st(play(8, OPEN)).course.obstacles).toEqual([]);
  });

  it('a team doing its jobs gets over generated courses: up the climbs, across the ropes, over the ice', () => {
    for (let seed = 1; seed <= 6; seed++) {
      const g0 = table(8);
      g0.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
      const tx = new Tx(g0);
      startArena(tx, 'mecha_santa', { mode: 'captain_teams', participants: Object.keys(g0.players!), duelists: ['p1', 'p2'], config: { teams: assignTeams(['p1', 'p2'], Object.keys(g0.players!), rng(1)) } }, { now: NOW, rng: rng(seed * 31) });
      let g = apply(g0, tx.plan());
      g = work(g, [0], NOW + 100, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs + 1000);
      expect(arenaOf(g)!.result, `seed ${seed}`).toMatchObject({ status: 'resolved', reason: 'first over the line' });
      const w = st(g).teams[0].walker;
      for (const o of st(g).course.obstacles.filter((x) => x.kind === 'climb')) expect(w.taken, `seed ${seed}`).toContain(o.id);
    }
  });

  it('the phones are told what is coming once it is close (what it is, never when), and only then', () => {
    const course = [{ id: 'c1', kind: 'climb', at: 20 }, { id: 'r1', kind: 'rope', at: 40, width: 7 }];
    let g = play(8, { obstacles: course, length: 60 });
    expect(tv(g, NOW + 50).teams[0].ahead ?? null).toBeNull();                 // 20 away: nothing yet
    g = work(g, [0], NOW + 100, NOW + 4000);
    const at = NOW + 4100;
    const v = phone(g, st(g).teams[0].crew[3][0], at);
    expect(v.teams[0].ahead).toEqual({ id: 'c1', kind: 'climb' });
    expect(v.teams[1].ahead ?? null).toBeNull();                               // the other robot has not moved
    expect(NEEDS.climb).toEqual([3, 0, 1, 2]);                                 // arms first, then the legs; the balance too
    expect(NEEDS.ice).toEqual([2]);
    expect(Object.keys(v.teams[0].ahead!).sort()).toEqual(['id', 'kind']);     // nothing in it says when: no time, no "now"
  });

  it('a rope needs a width; the new walk numbers are checked; a course is generated unless one is given', () => {
    expect(() => mecha.validateConfig({ obstacles: [{ id: 'r', kind: 'rope', at: 30 }] })).toThrow();
    expect(() => mecha.validateConfig({ walk: { rungs: 0 } })).toThrow();
    expect(mecha.validateConfig({}).generate).toBe(true);
    expect(mecha.validateConfig({ obstacles: [] }).generate).toBe(false);
  });
});
