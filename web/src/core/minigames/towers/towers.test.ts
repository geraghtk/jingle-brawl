// Elf Tower Wars on the runtime, and its lane battle underneath: toys march and are stopped by towers,
// snowballs and a shield, and a workshop throws at what gets close; nobody has a job: every player has
// every move and their own energy, and a team earns the same whatever its size; a workshop falling
// ends it for its captain; the bell, overtime, a point for the winning team only; pause; the same
// battle however the clock is stepped.
import { describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { advance, arm, armed, build, DEFAULT_LANES as P, energyAt, guardTarget, newField, PACK_GAP, powerOf, send, shield, spend, targetOf, unitAt, zap } from '../engines/lanes';
import type { Field } from '../engines/lanes';
import { assignTeams } from '../engines/teams';
import { arenaAct, arenaDeadline, arenaOf, arenaPause, arenaResume, handoff, startArena } from '../runtime';
import { DEFAULTS, handOffers, incomeOf, toughAt, towers } from './towers';
import type { TowerState } from './towers';
import { energyOf, fieldClock, towerView } from './view';

const must = (d: { field?: Field; error?: string }): Field => { if (!d.field) throw new Error(d.error); return d.field; };
/** The battle with no workshop throwing (what the towers alone do). */
const Q = { ...P, guard: 0 };

describe('the lane battle (engines/lanes)', () => {
  it('a toy marches down its lane at its speed; a screen glides it between the steps', () => {
    let f = must(send(newField(NOW), 0, 1, 'soldier', NOW));
    f = advance(f, NOW + 2000);
    expect(f.units[0]).toMatchObject({ side: 0, lane: 1, kind: 'soldier' });
    expect(f.units[0].pos).toBeCloseTo(P.units.soldier.speed * 2);
    expect(unitAt(f.units[0], f.asOf, f.asOf + 500)).toBeCloseTo(f.units[0].pos + P.units.soldier.speed * 0.5);
    expect(send(f, 0, 1, 'soldier', NOW + 100).error).toMatch(/busy/);                // the lane takes one at a time
    expect(send(f, 0, 2, 'soldier', NOW + 100).error).toBeUndefined();
    expect(send(must(send(f, 0, 1, 'soldier', NOW + 2000)), 0, 1, 'soldier', NOW + 2100).error).toMatch(/busy/);
  });

  it('a toy that arrives hurts the other base; one that meets a shield there is stopped', () => {
    const sent = must(send(newField(NOW), 0, 0, 'tank', NOW));
    const through = advance(sent, NOW + 30_000);
    expect(through.sides[1].hp).toBe(P.baseHp - P.units.tank.damage);
    expect(through.units).toEqual([]);
    expect(through.events.at(-1)).toMatchObject({ kind: 'arrived', side: 0, lane: 0 });
    const almost = advance(sent, NOW + 17_000);
    const shut = advance(must(shield(almost, 1, 0, NOW + 17_000)), NOW + 30_000);
    expect(shut.sides[1].hp).toBe(P.baseHp);
    expect(shut.events.at(-1)).toMatchObject({ kind: 'blocked' });
    expect(shield(shut, 1, 1, NOW + 20_000).error).toMatch(/recharging/);
  });

  it('a tower hurts the toy nearest its base in reach; two snow towers stop a lone soldier, one does not', () => {
    const one = must(build(newField(NOW), 1, 0, 0, 'snow', NOW));
    const two = must(build(one, 1, 0, 1, 'snow', NOW));
    expect(build(must(build(one, 1, 0, 1, 'cannon', NOW)), 1, 0, 1, 'snow', NOW).error).toMatch(/built there/);   // not a cheaper one over a dearer
    expect(must(build(two, 1, 0, 1, 'cannon', NOW)).sides[1].towers).toEqual([{ lane: 0, slot: 0, kind: 'snow' }, { lane: 0, slot: 1, kind: 'cannon' }]);   // a better one in its place
    expect(advance(must(send(one, 0, 0, 'soldier', NOW)), NOW + 30_000, Q).sides[1].hp).toBe(P.baseHp - 1);
    const held = advance(must(send(two, 0, 0, 'soldier', NOW)), NOW + 30_000, Q);
    expect(held.sides[1]).toMatchObject({ hp: P.baseHp, stopped: 1 });
    // in another lane the towers do nothing
    expect(advance(must(send(two, 0, 2, 'soldier', NOW)), NOW + 30_000).sides[1].hp).toBe(P.baseHp - 1);
    const mid = advance(must(send(must(send(two, 0, 0, 'soldier', NOW)), 0, 0, 'tank', NOW + 600)), NOW + 6500);
    expect(targetOf(mid, 1, mid.sides[1].towers[1])?.kind).toBe('soldier');   // the one further along
  });

  it('a workshop throws at what gets close: a lone soldier limps in, one a tower has softened does not', () => {
    const lone = advance(must(send(newField(NOW), 0, 0, 'soldier', NOW)), NOW + 9000);
    expect(guardTarget(lone, 1, 0)?.id).toBe(lone.units[0].id);                      // the last stretch: the workshop is throwing at it
    expect(guardTarget(lone, 1, 1)).toBeNull();
    expect(lone.units[0].hp).toBeLessThan(P.units.soldier.hp);
    expect(advance(lone, NOW + 30_000).sides[1].hp).toBe(P.baseHp - 1);              // hurt, but in
    const one = must(build(newField(NOW), 1, 0, 0, 'cannon', NOW));
    expect(advance(must(send(one, 0, 0, 'soldier', NOW)), NOW + 30_000).sides[1]).toMatchObject({ hp: P.baseHp, stopped: 1 });
    expect(guardTarget(advance(must(send(newField(NOW), 0, 0, 'soldier', NOW)), NOW + 5000), 1, 0)).toBeNull();   // not from across the field
  });

  it('what stops what: a pack swamps a tower that hits one at a time, and a cannon (it hits a crowd) stops the lot', () => {
    const pack = must(send(newField(NOW), 0, 0, 'mouse', NOW));
    expect(pack.units).toHaveLength(P.units.mouse.pack!);
    expect(pack.units.map((u) => u.pos)).toEqual([2 * PACK_GAP, PACK_GAP, 0]);         // one behind the other
    expect(pack.sides[0].sent).toBe(1);                                                // one card, one send
    const through = (field: Field) => P.baseHp - advance(must(send(field, 0, 0, 'mouse', NOW)), NOW + 30_000, Q).sides[1].hp;
    const snow = must(build(newField(NOW), 1, 0, 0, 'snow', NOW)), cannon = must(build(newField(NOW), 1, 0, 0, 'cannon', NOW));
    expect(through(newField(NOW))).toBe(3);
    expect(through(snow)).toBeGreaterThanOrEqual(1);                                   // it gets one, the rest are past
    expect(through(cannon)).toBe(0);
    // and the other way round: one sturdy toy shrugs the cannon off, where a pack did not
    expect(advance(must(send(cannon, 0, 0, 'tank', NOW)), NOW + 40_000, Q).sides[1].hp).toBe(P.baseHp - P.units.tank.damage);
  });

  it('a tower built again on its own spot gets better, up to a top level; a slowing one reaches further', () => {
    let f = must(build(newField(NOW), 1, 0, 0, 'snow', NOW));
    f = must(build(f, 1, 0, 0, 'snow', NOW));
    expect(f.sides[1].towers).toEqual([{ lane: 0, slot: 0, kind: 'snow', level: 2 }]);
    expect(powerOf(f.sides[1].towers[0])).toBeCloseTo(1 + P.levelUp);
    expect(f.events.at(-1)).toMatchObject({ kind: 'built', what: 'snow:0:2' });
    // one snow tower does not stop a soldier (above); the same tower a level up does
    expect(advance(must(send(f, 0, 0, 'soldier', NOW)), NOW + 30_000, Q).sides[1]).toMatchObject({ hp: P.baseHp, stopped: 1 });
    f = must(build(f, 1, 0, 0, 'snow', NOW));
    expect(build(f, 1, 0, 0, 'snow', NOW).error).toMatch(/as good as it gets/);
    expect(must(build(f, 1, 0, 0, 'cannon', NOW)).sides[1].towers).toEqual([{ lane: 0, slot: 0, kind: 'cannon' }]);   // a dearer kind starts again at level 1
  });

  it('the avalanche takes two: one arms it on a lane, a different player joins within a moment, and everything on that half is hit', () => {
    let f = newField(NOW);
    for (const [at, lane] of [[0, 0], [600, 0], [1200, 1]] as const) f = must(send(advance(f, NOW + at), 0, lane, 'soldier', NOW + at));
    f = advance(f, NOW + 7000);                                                        // on side 1's half now
    const one = must(arm(f, 1, 0, 'ann', NOW + 7000));
    expect(armed(one.sides[1], NOW + 7100)).toBe(true);
    expect(one.units).toHaveLength(3);                                                 // nothing yet
    expect(arm(one, 1, 0, 'ann', NOW + 7100).error).toMatch(/second pair of hands/);   // not the same player twice
    const off = must(arm(one, 1, 0, 'bob', NOW + 7500));
    expect(off.units.map((u) => u.lane)).toEqual([1]);                                 // that lane only
    expect(off.sides[1]).toMatchObject({ stopped: 2, armLane: -1, comboAt: NOW + 7500 + P.comboEveryMs });
    expect(off.events.at(-1)).toMatchObject({ kind: 'combo', lane: 0, what: '2' });
    expect(arm(off, 1, 1, 'ann', NOW + 8000).error).toMatch(/Not ready/);
    // too late, or on another lane: it only arms again (and waits there)
    expect(armed(must(arm(one, 1, 0, 'bob', NOW + 7000 + P.comboMs)).sides[1], NOW + 7000 + P.comboMs + 1)).toBe(true);
    expect(must(arm(one, 1, 1, 'bob', NOW + 7500)).sides[1]).toMatchObject({ armLane: 1, armBy: 'bob' });
    expect(must(arm(f, 1, 0, 'ann', NOW + 7000, P, true)).units).toHaveLength(1);       // nobody else on the team: one press is enough
  });

  it('a frost lamp slows what is in its reach; a snowball hurts the lead toy on your half, not before', () => {
    const lamp = must(build(newField(NOW), 1, 0, 1, 'frost', NOW));
    const f = advance(must(send(lamp, 0, 0, 'soldier', NOW)), NOW + 6000);
    expect(f.units[0].v).toBeCloseTo(P.units.soldier.speed * P.towers.frost.slow);
    expect(zap(advance(must(send(newField(NOW), 0, 0, 'soldier', NOW)), NOW + 1000), 1, 0, NOW + 1000).error).toMatch(/Nothing in reach/);
    const near = advance(must(send(newField(NOW), 0, 0, 'soldier', NOW)), NOW + 6000);
    const hit = must(zap(near, 1, 0, NOW + 6000));
    expect(hit.units[0].hp).toBe(P.units.soldier.hp - P.zap);
    expect(zap(hit, 1, 0, NOW + 6100).error).toMatch(/Easy/);
  });

  it('energy refills by the clock up to a cap; you cannot spend what is not there', () => {
    const u = { energy: 1, asOf: NOW };
    expect(energyAt(u, NOW + 3000, P.purse)).toBeCloseTo(4);
    expect(energyAt(u, NOW + 600_000, P.purse)).toBe(P.purse.max);
    expect(spend(u, 5, NOW + 3000, P.purse)).toBeNull();
    expect(spend(u, 4, NOW + 3000, P.purse)).toEqual({ energy: 0, asOf: NOW + 3000 });
  });

  it('bringing the battle up to date in one go or bit by bit gives the same battle', () => {
    let f = must(build(newField(NOW), 1, 0, 0, 'snow', NOW));
    for (const [at, lane, kind] of [[0, 0, 'soldier'], [700, 0, 'tank'], [900, 1, 'soldier']] as const) f = must(send(advance(f, NOW + at), 0, lane, kind, NOW + at));
    const once = advance(f, NOW + 25_000);
    let slow = f;
    for (let t = NOW + 900; t <= NOW + 25_000; t += 73) slow = advance(slow, t);
    expect(advance(slow, NOW + 25_000)).toEqual(once);
  });
});

const st = (g: Game) => arenaOf(g)!.state as TowerState;
const ctx = (now: number) => ({ now, rng: rng(4) });
function play(n: number, config: Record<string, unknown> = {}, away: string[] = []): Game {
  const g = table(n);
  for (const id of away) g.players![id].connected = false;
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const teams = assignTeams(['p1', 'p2'], Object.keys(g.players!), rng(1));
  const out = startArena(tx, 'elf_tower_wars', { mode: 'captain_teams', participants: Object.keys(g.players!), duelists: ['p1', 'p2'], config: { teams, ...config } }, ctx(NOW));
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
const act = (g: Game, who: string, action: unknown, now: number) => arenaAct(g, who, { action }, ctx(now));
function tick(g: Game, now: number): Game {
  for (let i = 0; i < 9000 && !arenaOf(g)!.result; i++) {
    const d = [...arenaOf(g)!.deadlines].sort((a, b) => a.at - b.at)[0];
    if (!d || d.at > now) break;
    g = apply(g, arenaDeadline(g, d.id, ctx(d.at)));
  }
  return g;
}
const tv = (g: Game, now: number) => towerView((projectGame(g, { kind: 'tv' }, now).arenaView as { view: unknown }).view);
const phone = (g: Game, who: string, now: number) => towerView((projectGame(g, { kind: 'phone', pid: who }, now).arenaView as { view: unknown }).view);
/** A player on team `ti` (not its captain by default). */
const on = (g: Game, ti: number, i = 1) => st(g).teams[ti].members[i];
const energy = (g: Game, who: string) => { const s = st(g); return energyOf(s.teams.find((t) => t.members.includes(who))!, who); };
/** When this half of the round ends at the latest. */
const ends = (g: Game) => st(g).phaseEndsAt;
/** On to the march of the round in hand (the plan runs out its clock). */
const march = (g: Game) => tick(g, ends(g));

describe('Elf Tower Wars: plan, then watch', () => {
  it('two teams under the captains; it opens with a plan: the field stands still and everyone has their share of the round\'s energy, near enough the same for a team whatever its size', () => {
    const g = play(5);                                                         // three against two
    const s = st(g);
    expect(s).toMatchObject({ stage: 'running', phase: 'plan', round: 1, phaseAt: NOW, phaseEndsAt: NOW + DEFAULTS.planMs });
    expect(s.teams.map((t) => t.captainId)).toEqual(['p1', 'p2']);
    const [big, small] = s.teams[0].members.length === 3 ? s.teams : [s.teams[1], s.teams[0]];
    // (the bigger team gets a little more in all, `crowd` for each extra player: a share split more ways buys less)
    expect(energy(g, big.members[0])).toBeCloseTo((incomeOf(s, 1) + DEFAULTS.crowd) / 3);
    expect(energy(g, small.members[0])).toBeCloseTo(incomeOf(s, 1) / 2);
    expect(big.members.reduce((n, m) => n + energy(g, m), 0)).toBeCloseTo(small.members.reduce((n, m) => n + energy(g, m), 0) + DEFAULTS.crowd);
    // nothing trickles in by the clock
    expect(energyOf(tv(g, NOW + 5000).teams[0], s.teams[0].members[0])).toBeCloseTo(energy(g, s.teams[0].members[0]));
    expect(tv(g, NOW)).toMatchObject({ phase: 'plan', round: 1, rounds: DEFAULTS.rounds, phaseEndsAt: NOW + DEFAULTS.planMs });
    expect(phone(g, 'p1', NOW).myTeam).toBe('A');
    // a big table: nobody's share falls under the least
    const crowd = st(play(12));
    expect(incomeOf(crowd, 1) / 6).toBeGreaterThanOrEqual(DEFAULTS.minShare);
  });

  it('planning: a tower goes up at once, toys are lined up at the door and do not move; each paid from your own energy; reactions wait for the march', () => {
    let g = play(4);
    const who = on(g, 0), mate = st(g).teams[0].members[0];
    const before = energy(g, who);
    g = apply(g, act(g, who, { kind: 'build', lane: 0, slot: 0, tower: 'snow' }, NOW + 1000));
    expect(st(g).field.sides[0].towers).toEqual([{ lane: 0, slot: 0, kind: 'snow' }]);
    expect(energy(g, who)).toBeCloseTo(before - P.towers.snow.cost);
    expect(energy(g, mate)).toBeCloseTo(before);                               // a teammate's is their own
    g = apply(g, act(g, who, { kind: 'send', lane: 1, unit: 'soldier' }, NOW + 2000));
    expect(st(g).queue).toEqual([{ side: 0, lane: 1, unit: 'soldier', by: who }]);
    expect(st(g).field.units).toEqual([]);                                     // not out yet
    expect(tv(g, NOW + 2500).queue).toHaveLength(1);                           // every screen sees what is lined up
    expect(act(g, who, { kind: 'send', lane: 1, unit: 'tank' }, NOW + 2500).error).toMatch(/Not enough energy/);
    expect(act(g, mate, { kind: 'zap', lane: 1 }, NOW + 2500).error).toMatch(/Wait for the march/);
    expect(act(g, mate, { kind: 'shield', lane: 1 }, NOW + 2500).error).toMatch(/Wait for the march/);
    expect(act(g, 'nobody', { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 2500).error).toBeTruthy();
    // a door only holds so many
    let full = play(2, { income: 200, rules: { purse: { start: 0, rate: 0, max: 200 } } });
    for (let i = 0; i < DEFAULTS.maxQueue; i++) full = apply(full, act(full, 'p1', { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 100 + i));
    expect(act(full, 'p1', { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 900).error).toMatch(/door is full/);
    expect(act(full, 'p1', { kind: 'send', lane: 1, unit: 'soldier' }, NOW + 900).error).toBeUndefined();
  });

  it('the march: both teams\' toys come out together, one behind the other, and walk; planning moves are refused; the phones have their reactions', () => {
    let g = play(4);
    const a = on(g, 0), b = on(g, 1);
    g = apply(g, act(g, a, { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 1000));
    g = apply(g, act(g, a, { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 1500));
    g = apply(g, act(g, b, { kind: 'send', lane: 2, unit: 'mouse' }, NOW + 2000));
    g = march(g);
    const s = st(g), T = NOW + DEFAULTS.planMs;
    expect(s).toMatchObject({ phase: 'march', round: 1, phaseAt: T, phaseEndsAt: T + DEFAULTS.marchMs });
    expect(s.queue).toEqual([]);
    expect(s.field.asOf).toBe(T);
    const mine = s.field.units.filter((u) => u.side === 0);
    expect(mine.map((u) => u.pos)).toEqual([0, -(PACK_GAP + PACK_GAP * 1.5)]);  // the second starts behind the first, inside the workshop
    expect(s.field.units.filter((u) => u.side === 1)).toHaveLength(P.units.mouse.pack!);
    expect(act(g, a, { kind: 'send', lane: 0, unit: 'soldier' }, T + 500).error).toMatch(/next round/);
    expect(act(g, a, { kind: 'build', lane: 0, slot: 0, tower: 'snow' }, T + 500).error).toMatch(/next round/);
    // they walk: a screen glides them from the field's own clock
    const later = tv(g, T + 3000);
    expect(later.phase).toBe('march');
    expect(later.field.units.find((u) => u.side === 0)!.pos).toBeCloseTo(P.units.soldier.speed * 3, 1);
    expect(fieldClock(later, T + 3000)).toBe(T + 3000);
  });

  it('the march ends and the next plan begins: what is still walking stands where it is, the next round\'s energy arrives on top of what was kept', () => {
    let g = play(4);
    const a = on(g, 0);
    g = apply(g, act(g, a, { kind: 'send', lane: 0, unit: 'tank' }, NOW + 1000));          // slow: still out when the march ends
    const kept = energy(g, a);
    g = march(g);
    g = tick(g, ends(g));
    const s = st(g), T2 = NOW + DEFAULTS.planMs + DEFAULTS.marchMs;
    expect(s).toMatchObject({ phase: 'plan', round: 2, phaseAt: T2, phaseEndsAt: T2 + DEFAULTS.planMs });
    const tank = s.field.units[0];
    expect(tank.pos).toBeCloseTo(P.units.tank.speed * DEFAULTS.marchMs / 1000, 1);
    expect(energy(g, a)).toBeCloseTo(kept + incomeOf(s, 2) / 2);
    expect(incomeOf(s, 2)).toBe(DEFAULTS.income + DEFAULTS.incomeStep);
    // while everyone plans it does not move, on the room's side or on a screen
    const v = tv(g, T2 + 6000);
    expect(v.field.units[0].pos).toBeCloseTo(tank.pos);
    expect(fieldClock(v, T2 + 6000)).toBe(v.field.asOf);
    // and the next march takes it up from there
    g = march(g);
    expect(st(g).field.asOf).toBe(T2 + DEFAULTS.planMs);
    expect(tv(g, T2 + DEFAULTS.planMs + 2000).field.units[0].pos).toBeGreaterThan(tank.pos + 0.05);
  });

  it('nobody waits for nothing: a plan ends once everyone here has said done, and a march with an empty field ends early', () => {
    let g = play(4);
    const everyone = st(g).teams.flatMap((t) => t.members);
    for (const [i, p] of everyone.entries()) {
      expect(st(g).phaseEndsAt).toBe(NOW + DEFAULTS.planMs);                    // not until the last one
      g = apply(g, act(g, p, { kind: 'done' }, NOW + 2000 + i * 100));
    }
    expect(act(g, everyone[0], { kind: 'done' }, NOW + 2500).error).toMatch(/Already/);
    expect(st(g).phaseEndsAt).toBe(NOW + 2300 + DEFAULTS.doneMs);
    expect(tv(g, NOW + 2400).teams[0].done.length).toBe(2);
    g = tick(g, NOW + 2300 + DEFAULTS.doneMs);
    expect(st(g)).toMatchObject({ phase: 'march', round: 1 });
    // nothing was sent: the march is over as soon as it decently can be
    const T = NOW + 2300 + DEFAULTS.doneMs;
    g = tick(g, T + DEFAULTS.minMarchMs + 500);
    expect(st(g)).toMatchObject({ phase: 'plan', round: 2 });
    expect(st(g).phaseAt).toBeLessThan(T + DEFAULTS.marchMs);
    // a phone that is away is not waited for
    let away = play(4, {}, ['p4']);
    for (const p of st(away).teams.flatMap((t) => t.active)) away = apply(away, act(away, p, { kind: 'done' }, NOW + 1000));
    expect(st(away).phaseEndsAt).toBe(NOW + 1000 + DEFAULTS.doneMs);
  });

  it('a workshop that falls loses: the other captain wins the duel, their team a Party point each', () => {
    let g = play(4, { rules: { baseHp: 2, guard: 0 } });
    const a = on(g, 0);
    g = apply(g, act(g, a, { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 500));
    g = apply(g, act(g, a, { kind: 'send', lane: 1, unit: 'soldier' }, NOW + 600));
    g = march(g);
    g = tick(g, ends(g));
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', reason: 'the other workshop fell', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
    expect(st(g)).toMatchObject({ stage: 'done', winner: 'A' });
    expect(r.rewards!.map((x) => x.playerId).sort()).toEqual([...st(g).teams[0].members].sort());
    g.currentTurn = { ...g.currentTurn!, phase: 'minigame_arena' };
    const after = apply(g, handoff(g));
    expect(after.players![st(g).teams[0].members[0]].partyPoints || 0).toBe(1);
    expect(after.players![st(g).teams[1].members[0]].partyPoints || 0).toBe(0);
    expect(after.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p1' });
  });

  it('after the last round the stronger workshop wins; level: one more round, once; still level: the host decides, nobody is paid', () => {
    const quick = { rounds: 2, planMs: 1000, marchMs: 12_000, rules: { guard: 0 } };
    let g = play(4, quick);
    g = apply(g, act(g, on(g, 0), { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 100));
    for (let i = 0; i < 4 && !arenaOf(g)!.result; i++) g = tick(g, ends(g));
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', reason: 'the stronger workshop after the last round', duel: { winningDuelistId: 'p1' } });
    // nobody does anything: level after the last round, so one more; level again: the host
    let idle = play(4, quick);
    for (let i = 0; i < 12 && st(idle).stage === 'running'; i++) idle = tick(idle, Math.min(ends(idle), st(idle).phase === 'march' ? st(idle).phaseAt + DEFAULTS.minMarchMs : ends(idle)));
    expect(st(idle)).toMatchObject({ stage: 'overtime', round: 3, phase: 'plan' });
    expect(tv(idle, ends(idle) - 10).rounds).toBe(3);
    for (let i = 0; i < 4 && !arenaOf(idle)!.result; i++) idle = tick(idle, ends(idle));
    const r = arenaOf(idle)!.result!;
    expect(r).toMatchObject({ status: 'requires_tiebreaker', winners: [] });
    expect(r.rewards ?? []).toEqual([]);
  });

  it('toys sent in later rounds are sturdier, so it cannot end in a stand-off', () => {
    let g = play(4, { planMs: 1000, marchMs: 5000, minMarchMs: 5000 });
    const a = on(g, 0);
    g = apply(g, act(g, a, { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 100));
    g = march(g);
    const first = st(g).field.units[0].hp;
    g = tick(g, ends(g));
    g = apply(g, act(g, a, { kind: 'send', lane: 1, unit: 'soldier' }, st(g).phaseAt + 100));
    g = march(g);
    const second = st(g).field.units.find((u) => u.lane === 1)!.hp;
    expect(first).toBe(P.units.soldier.hp);
    expect(second).toBeCloseTo(P.units.soldier.hp * toughAt(st(g), 2));
    expect(toughAt(st(g), 2)).toBeGreaterThan(1);
  });

  it('two of a team set off the avalanche between them during a march, each paying; a snowball and a shield are march moves too', () => {
    let g = play(4, { income: 40 });
    const [a, b] = st(g).teams[1].members, them = on(g, 0);
    for (const lane of [0, 0, 0]) g = apply(g, act(g, them, { kind: 'send', lane, unit: 'soldier' }, NOW + 100));
    g = march(g);
    const T = st(g).phaseAt;
    g = tick(g, T + 6000);                                                     // on team B's half now
    expect(st(g).field.units.filter((u) => u.side === 0)).toHaveLength(3);
    const before = energy(g, a);
    g = apply(g, act(g, a, { kind: 'combo', lane: 0 }, T + 6000));
    expect(st(g).field.units.filter((u) => u.side === 0)).toHaveLength(3);     // it waits for a second pair of hands
    expect(tv(g, T + 6100).field.sides[1]).toMatchObject({ armLane: 0, armBy: a });
    expect(act(g, a, { kind: 'shield', lane: 1 }, T + 6200).error).toBeUndefined();
    expect(act(g, a, { kind: 'zap', lane: 0 }, T + 6200).error).toBeUndefined();
    g = apply(g, act(g, b, { kind: 'combo', lane: 0 }, T + 6500));
    expect(st(g).field.units.filter((u) => u.side === 0)).toHaveLength(0);
    expect(energy(g, a)).toBeCloseTo(before - P.comboCost);
    expect(energy(g, b)).toBeCloseTo(before - P.comboCost);
  });

  it("a phone that has dropped gets no share of the next round's energy: it goes to the teammates who are here", () => {
    let g = play(4);
    const [x, y] = st(g).teams[0].members;
    g = { ...g, players: { ...g.players, [y]: { ...g.players![y], connected: false } } };
    const keptX = energy(g, x), keptY = energy(g, y);
    g = march(g);
    g = tick(g, ends(g));
    expect(st(g).round).toBe(2);
    expect(st(g).teams[0].active).toEqual([x]);
    expect(energy(g, y)).toBeCloseTo(keptY);                                    // what they had is still theirs
    expect(energy(g, x)).toBeCloseTo(Math.min(DEFAULTS.rules.purse.max, keptX + incomeOf(st(g), 2)));
  });

  it('a pause holds everything: the plan clock, toys in the middle of a march, what is lined up', () => {
    let g = play(4);
    g = apply(g, act(g, on(g, 0), { kind: 'send', lane: 0, unit: 'tank' }, NOW + 500));
    g = march(g);
    const T = st(g).phaseAt;
    g = tick(g, T + 3000);
    g = apply(g, arenaPause(g, T + 3000));
    const held = projectGame(g, { kind: 'tv' }, T + 30_000).arenaView!;
    expect(towerView(held.view).field.units[0].pos).toBeCloseTo(P.units.tank.speed * 3, 1);
    g = apply(g, arenaResume(g, T + 63_000));
    expect(st(g).phaseEndsAt).toBe(T + DEFAULTS.marchMs + 60_000);
    expect(st(g).field.units[0].pos).toBeCloseTo(P.units.tank.speed * 3, 1);
  });

  it('the simulated hands play it to an end (towers go up, toys go out, they say when they are done), the same however the clock is stepped', () => {
    const run = (stepMs: number) => {
      let g = play(6);
      const ids = st(g).teams.flatMap((t) => t.members);
      for (let now = NOW + 300; now < NOW + 400_000 && !arenaOf(g)!.result; now += stepMs) {
        g = tick(g, now);
        if (arenaOf(g)!.result) break;
        if ((now - NOW - 300) % 600 !== 0) continue;                            // the hands look every 600 ms whatever the step
        for (const p of ids) {
          const o = handOffers({ ...ctx(now), instanceId: 'x', mode: 'captain_teams', participants: ids, duelists: ['p1', 'p2'], connected: () => true }, st(g), p);
          if (o.length) { const did = act(g, p, o[Math.floor(now / 7 + p.length) % o.length], now); if (!did.error) g = apply(g, did); }
        }
      }
      return g;
    };
    const a = run(100), b = run(300);
    expect(arenaOf(a)!.result).toBeTruthy();
    expect(arenaOf(a)!.result!.status).not.toBe('cancelled');
    expect(st(a).field.sides.some((x) => x.towers.length > 0)).toBe(true);
    expect(st(a).field.sides[0].sent + st(a).field.sides[1].sent).toBeGreaterThan(6);
    expect(arenaOf(b)!.result!.winners).toEqual(arenaOf(a)!.result!.winners);
    expect(st(b).field.sides.map((x) => x.hp)).toEqual(st(a).field.sides.map((x) => x.hp));
  });

  it('starts as the duel game with four at the table: captains and teams; refuses settings that make no battle', () => {
    const g = table(4, { duelPick: 'off', duelAppGames: ['elf_tower_wars'], voteWindow: 0 });
    g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Elf Tower Wars', kind: 'app', app: 'elf_tower_wars', via: 'picker' } };
    const started = apply(g, startDuelApp(g, ctx(NOW)));
    expect(arenaOf(started)).toMatchObject({ gameId: 'elf_tower_wars', mode: 'captain_teams', duelists: ['p2', 'p1'] });
    expect(st(started).teams.map((t) => t.captainId)).toEqual(['p2', 'p1']);
    expect(() => towers.validateConfig({ rounds: 0 })).toThrow();
    expect(() => towers.validateConfig({ minMarchMs: 99_999 })).toThrow();
    expect(() => towers.validateConfig({ rules: { lanes: 9 } })).toThrow();
    expect(towers.validateConfig({}).rules.purse.rate).toBe(0);                 // energy comes a round at a time, never by the clock
  });
});
