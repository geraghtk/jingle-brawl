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
import { DEFAULTS, handOffers, towers } from './towers';
import type { TowerState } from './towers';
import { energyOf, towerView } from './view';

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
const energy = (g: Game, who: string, now: number) => { const s = st(g); return energyOf(s.teams.find((t) => t.members.includes(who))!, who, now, s.cfg.rules, { rushAt: s.rushAt, rush: s.cfg.rush }); };

describe('Elf Tower Wars', () => {
  it('two teams under the captains; no jobs: everyone has their own energy, and a team earns the same whatever its size', () => {
    for (const n of [4, 6, 9, 14]) {
      const [a, b] = st(play(n)).teams;
      for (const t of [a, b]) expect(Object.keys(t.purses).sort()).toEqual([...t.members].sort());
      expect(a.rate * a.active.length).toBeCloseTo(b.rate * b.active.length);        // 4 against 5: the four earn more each
      expect(a.rate * a.active.length).toBeGreaterThanOrEqual(DEFAULTS.teamEnergy);
      expect(Math.min(a.rate, b.rate)).toBeGreaterThanOrEqual(DEFAULTS.minShare - 1e-9);   // a big table: nobody waits long for a go
    }
    const g = play(6);
    expect(st(g).teams.map((t) => t.captainId)).toEqual(['p1', 'p2']);
    const v = phone(g, on(g, 1), NOW + 1000);
    expect(v.myTeam).toBe('B');
    expect(energyOf(v.teams[1], on(g, 1), NOW + 1000, v.rules)).toBeCloseTo(5);       // a screen works its energy out from the view
    expect(tv(g, NOW).teams[0]).toMatchObject({ rate: 1, active: st(g).teams[0].members });
  });

  it('anyone may build, send or defend, paid from their own energy: nobody spends what they have not got, or a teammate\'s', () => {
    const g = play(6);
    const who = on(g, 0), mate = on(g, 0, 2);
    expect(act(g, who, { kind: 'patch' }, NOW + 100).error).toMatch(/Nothing to mend/);
    expect(act(g, who, { kind: 'build', lane: 0, slot: 0, tower: 'cannon' }, NOW + 100).error).toMatch(/Not enough energy/);   // 8, with 4 in hand
    expect(act(g, who, { kind: 'build', lane: 9, slot: 0, tower: 'snow' }, NOW + 100).error).toBeTruthy();
    expect(act(g, who, { kind: 'build', lane: 0, slot: 0, tower: 'laser' }, NOW + 100).error).toBeTruthy();
    let next = apply(g, act(g, who, { kind: 'build', lane: 0, slot: 0, tower: 'snow' }, NOW + 100));
    expect(st(next).field.sides[0].towers).toEqual([{ lane: 0, slot: 0, kind: 'snow' }]);
    expect(act(next, who, { kind: 'build', lane: 0, slot: 0, tower: 'snow' }, NOW + 4200).error).toBeUndefined();   // the same tower again: it may be made better
    expect(energy(next, who, NOW + 100)).toBeCloseTo(0.1);
    expect(energy(next, mate, NOW + 100)).toBeCloseTo(4.1);                              // a teammate's energy is untouched
    expect(act(next, who, { kind: 'send', lane: 2, unit: 'soldier' }, NOW + 200).error).toMatch(/Not enough energy/);   // the same purse: the tower took it
    next = apply(next, act(next, who, { kind: 'send', lane: 2, unit: 'soldier' }, NOW + 2100));          // and a toy, from the same hand, once it has refilled
    next = apply(next, act(next, mate, { kind: 'send', lane: 1, unit: 'soldier' }, NOW + 2200));
    expect(tv(next, NOW + 3200).field.units.map((u) => [u.side, u.lane])).toEqual([[0, 2], [0, 1]]);
    expect(arenaOf(next)!.deadlines.some((d) => d.id === 'tick')).toBe(true);              // the room keeps waking while a toy is out
    expect(act(next, 'nobody', { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 2300).error).toBeTruthy();
  });

  it('a workshop that falls loses: the other captain wins the duel, their team a Party point each', () => {
    let g = play(6, { rules: { ...P, baseHp: 2 } });
    const who = on(g, 0);
    g = apply(g, act(g, who, { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 100));
    g = apply(g, act(g, who, { kind: 'send', lane: 1, unit: 'soldier' }, NOW + 200));
    g = tick(g, NOW + 30_000);
    const r = arenaOf(g)!.result!, [a, b] = st(g).teams;
    expect(r).toMatchObject({ status: 'resolved', reason: 'the other workshop fell', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
    expect(st(g)).toMatchObject({ stage: 'done', winner: 'A' });
    expect(r.rewards!.map((x) => x.playerId).sort()).toEqual([...a.members].sort());
    expect(r.rewards!.every((x) => x.points === 1)).toBe(true);
    g.currentTurn = { ...g.currentTurn!, phase: 'minigame_arena' };
    const after = apply(g, handoff(g));
    expect(after.players![a.members[1]].partyPoints || 0).toBe(1);
    expect(after.players![b.members[1]].partyPoints || 0).toBe(0);
    expect(after.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p1' });
  });

  it('the bell: the stronger workshop wins; level: overtime with no more building; still level: the host decides, nobody is paid', () => {
    let hurt = play(6);
    hurt = apply(hurt, act(hurt, on(hurt, 1), { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 100));
    hurt = tick(hurt, NOW + DEFAULTS.roundMs);
    expect(arenaOf(hurt)!.result).toMatchObject({ status: 'resolved', reason: 'the stronger workshop at the bell', duel: { winningDuelistId: 'p2' } });
    let g = tick(play(6), NOW + DEFAULTS.roundMs);
    expect(arenaOf(g)!.result).toBeUndefined();
    expect(st(g)).toMatchObject({ stage: 'overtime', endsAt: NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs });
    expect(act(g, on(g, 0), { kind: 'build', lane: 0, slot: 0, tower: 'snow' }, NOW + DEFAULTS.roundMs + 500).error).toMatch(/overtime/);
    expect(act(g, on(g, 0), { kind: 'send', lane: 0, unit: 'soldier' }, NOW + DEFAULTS.roundMs + 500).error).toBeUndefined();
    g = tick(g, NOW + DEFAULTS.roundMs + DEFAULTS.overtimeMs);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'requires_tiebreaker', winners: [] });
    expect(arenaOf(g)!.result!.rewards).toBeUndefined();
  });

  it('toys sent later in the battle are sturdier, so it cannot end in a stand-off', () => {
    let g = play(6);
    g = apply(g, act(g, on(g, 0), { kind: 'send', lane: 0, unit: 'soldier' }, NOW));
    g = apply(g, act(g, on(g, 0), { kind: 'send', lane: 1, unit: 'soldier' }, NOW + 60_000));
    const units = st(g).field.units.filter((u) => u.lane === 1);
    expect(units[0].hp).toBeCloseTo(P.units.soldier.hp * (1 + DEFAULTS.toughen));
  });

  it('the finale: for the last half minute energy comes twice as fast, and every screen is told when', () => {
    const g = play(6), who = on(g, 0), rushAt = NOW + DEFAULTS.roundMs - DEFAULTS.rushMs;
    expect(st(g).rushAt).toBe(rushAt);
    expect(tv(g, NOW)).toMatchObject({ rushAt, rush: DEFAULTS.rush });
    let spent = apply(g, act(g, who, { kind: 'send', lane: 0, unit: 'tank' }, rushAt - 4000));     // 12 in hand, 6 left
    expect(energy(spent, who, rushAt - 4000)).toBeCloseTo(6);
    expect(energy(spent, who, rushAt)).toBeCloseTo(10);                                  // 1 a second up to the finale
    spent = apply(spent, act(spent, who, { kind: 'send', lane: 1, unit: 'tank' }, rushAt));
    expect(energy(spent, who, rushAt + 2000)).toBeCloseTo(8);                            // then 2 a second
    const v = phone(spent, who, rushAt + 2000);
    expect(energyOf(v.teams[0], who, rushAt + 2000, v.rules, v)).toBeCloseTo(8);         // a screen works out the same
    // straddling the start of it: half at one rate, half at the other
    const early = apply(g, act(g, who, { kind: 'send', lane: 0, unit: 'tank' }, rushAt - 1000));
    expect(energy(early, who, rushAt + 1000)).toBeCloseTo(6 + 1 + 2);
  });

  it('two of a team set off the avalanche between them, each paying; alone it only waits', () => {
    let g = play(6);
    const [a, b] = [on(g, 1, 0), on(g, 1, 1)], them = on(g, 0);
    for (const lane of [0, 0]) g = tick(apply(g, act(g, them, { kind: 'send', lane, unit: 'soldier' }, arenaOf(g)!.stateVersion ? NOW + 100 + st(g).field.units.length * 700 : NOW + 100)), NOW + 100 + st(g).field.units.length * 700);
    g = tick(g, NOW + 8000);
    expect(st(g).field.units).toHaveLength(2);
    g = apply(g, act(g, a, { kind: 'combo', lane: 0 }, NOW + 8000));
    expect(st(g).field.units).toHaveLength(2);
    expect(tv(g, NOW + 8100).field.sides[1]).toMatchObject({ armLane: 0, armBy: a });       // the TV can show it waiting
    expect(act(g, a, { kind: 'combo', lane: 0 }, NOW + 8200).error).toMatch(/second pair of hands/);
    g = apply(g, act(g, b, { kind: 'combo', lane: 0 }, NOW + 8500));
    expect(st(g).field.units).toHaveLength(0);
    expect(energy(g, a, NOW + 8500)).toBeCloseTo(12 - P.comboCost + 0.5);            // (paid when they armed it, half a second ago)
    expect(energy(g, b, NOW + 8500)).toBeCloseTo(12 - P.comboCost);
  });

  it("a phone that drops out earns nothing and its share goes to the teammates who are here; back, it earns again", () => {
    const whole = play(6);
    const [away, here, other] = [on(whole, 0, 2), on(whole, 0, 1), on(whole, 0, 0)];
    let g = play(6, {}, [away]);
    expect(st(g).teams[0]).toMatchObject({ active: [other, here], rate: 1.5 });
    expect(st(g).teams[1].rate).toBe(1);                                                // the other team is all here: the same 3 a second in all
    expect(energy(g, here, NOW + 2000)).toBeCloseTo(7);
    expect(energy(g, away, NOW + 2000)).toBeCloseTo(4);                                 // what it had, waiting for it
    // it comes back: the next thing that happens deals the rate out again, from what everyone has now
    g = { ...g, players: { ...g.players, [away]: { ...g.players![away], connected: true } } };
    g = apply(g, act(g, here, { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 2000));
    expect(st(g).teams[0]).toMatchObject({ rate: 1 });
    expect([...st(g).teams[0].active].sort()).toEqual([away, here, other].sort());
    expect(energy(g, other, NOW + 2000)).toBeCloseTo(7);                                // banked at the old rate
    expect(energy(g, other, NOW + 4000)).toBeCloseTo(9);                                // and on at the new one
    expect(energy(g, away, NOW + 4000)).toBeCloseTo(6);
    expect(energy(g, here, NOW + 4000)).toBeCloseTo(7);
  });

  it('a pause holds everything: toys stand where they were, energy does not refill, the bell waits', () => {
    let g = play(6);
    g = apply(g, act(g, on(g, 0), { kind: 'send', lane: 0, unit: 'soldier' }, NOW + 1000));
    g = tick(g, NOW + 3000);
    g = apply(g, arenaPause(g, NOW + 3000));
    const held = projectGame(g, { kind: 'tv' }, NOW + 50_000).arenaView!;
    const v = towerView(held.view);
    expect(held.pausedAt).toBe(NOW + 3000);
    expect(unitAt(v.field.units[0], v.field.asOf, held.pausedAt!)).toBeCloseTo(P.units.soldier.speed * 2, 1);
    g = apply(g, arenaResume(g, NOW + 63_000));
    expect(st(g).endsAt).toBe(NOW + DEFAULTS.roundMs + 60_000);
    expect(energy(g, on(g, 0, 2), NOW + 63_000)).toBeCloseTo(7);                          // three seconds' worth, not sixty-three
  });

  it('the simulated hands play it to an end (all-rounders: towers go up and toys go out), the same however the clock is stepped', () => {
    const mctx = (_g: Game, now: number) => ({ ...ctx(now), instanceId: 'x', mode: 'captain_teams' as const, participants: [], duelists: ['p1', 'p2'] as [string, string], connected: () => true });
    const run = (stepMs: number) => {
      let g = play(6);
      for (let now = NOW + 500; now < NOW + 200_000 && !arenaOf(g)!.result; now += 500) {
        for (let t = now - 500 + stepMs; t <= now; t += stepMs) g = tick(g, t);
        if (arenaOf(g)!.result) break;
        for (const p of Object.keys(g.players!)) {
          const o = handOffers(mctx(g, now), st(g), p)[0];
          if (o) g = apply(g, act(g, p, o, now));
        }
      }
      return g;
    };
    const a = run(500), b = run(100);
    expect(['resolved', 'requires_tiebreaker']).toContain(arenaOf(a)!.result!.status);   // (both sides playing the very same moves can end dead level)
    expect(st(a).field.sides.map((s) => s.towers.length)).toEqual([6, 6]);
    expect(st(a).field.sides[0].sent).toBeGreaterThan(10);
    expect(arenaOf(b)!.result).toEqual(arenaOf(a)!.result);
    // a hand with a habit only does that; the game's own offers are the all-rounder's
    const g = play(6);
    expect(handOffers(mctx(g, NOW + 100), st(g), on(g, 0), 'send').every((o) => o.kind === 'send')).toBe(true);
    expect(towers.offers(mctx(g, NOW + 100), st(g), on(g, 0))).toEqual(handOffers(mctx(g, NOW + 100), st(g), on(g, 0)));
  });

  it('starts as the duel game with four at the table: captains and teams; refuses settings that make no battle', () => {
    const g = table(4, { duelPick: 'off', duelAppGames: ['elf_tower_wars'], voteWindow: 0 });
    g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Elf Tower Wars', kind: 'app', app: 'elf_tower_wars', via: 'picker' } };
    const started = apply(g, startDuelApp(g, ctx(NOW)));
    expect(arenaOf(started)).toMatchObject({ gameId: 'elf_tower_wars', mode: 'captain_teams', duelists: ['p2', 'p1'] });
    expect(() => towers.validateConfig({ rules: { ...P, lanes: 0 } })).toThrow();
    expect(() => towers.validateConfig({ tickMs: 5 })).toThrow();
    expect(startArena(new Tx(table(2)), 'elf_tower_wars', { mode: 'direct_duel', participants: ['p1', 'p2'], duelists: ['p1', 'p2'] }, ctx(NOW)).error).toBeTruthy();
  });
});
