// A lane battle (theme-free: toys and towers, ants and anthills): two sides face each other across a
// few lanes. A side sends units down a lane towards the other side's base; the two armies pass each
// other, and only the other side's towers (and a thrown shot, and a raised shield) stop them. A unit
// that arrives hurts the base. Unlike the belt and the walker this cannot be settled from event to
// event in closed form (many units under fire at once), so it is a fixed-step simulation advanced
// lazily to `now`: the same steps in the same order whoever asks and however often, nothing sent per
// frame (a screen glides each unit from where it was at `asOf` at its speed). And a purse: energy that
// refills by the clock (closed form), one per player, so one teammate cannot spend the others' share.
// What stops what (the counters, as in every lane battler: each thing has an answer): a tower that hits one
// unit at a time is swamped by a pack; a tower with `splash` hurts everything near its target, so a pack
// dies to it, and a single sturdy unit shrugs it off. A tower built over again gets better (`level`). And
// a combo (`arm`): a strike down a lane that only goes off when a second player joins in within a moment.
// Versioned: a result says which rules made it.

export const LANES_VERSION = '1';

export interface UnitKind { cost: number; hp: number; speed: number; damage: number; /** Sent as a pack of this many, one behind the other (1). */ pack?: number }
export interface TowerKind { cost: number; dps: number; range: number; slow: number; /** Hurts every unit within this of its target too (0: one at a time). */ splash?: number }
export interface LaneParams {
  version: string;
  lanes: number;
  stepMs: number;
  baseHp: number;
  /** Where a side may build, as a distance from its own base (0 to 1), the same in every lane. */
  slots: number[];
  units: Record<string, UnitKind>;
  towers: Record<string, TowerKind>;
  sendMs: number;                  // between two units into the same lane
  zap: number; zapCost: number; zapMs: number; zapReach: number;       // a thrown shot at the lead unit on your half
  patch: number; patchCost: number; patchMs: number;                   // mending the base
  shieldCost: number; shieldMs: number; shieldEveryMs: number;         // a lane closed for a moment
  purse: { start: number; rate: number; max: number };
  /** A base defends itself a little: this much hurt a second to the unit nearest it within `guardReach` of it, in
   *  every lane (so a lone unit limps in, and a tower in front of the base stops it: attacks have to be massed or timed). */
  guard: number; guardReach: number;
  /** A tower built again on its own spot goes up a level, to `maxLevel`: each one this much more hurt (and a slowing tower that much more reach). */
  levelUp: number; maxLevel: number;
  /** The combo: one player arms it on a lane, a second must join within `comboMs`; then everything of theirs on your half
   *  of that lane takes `combo`. Each pays `comboCost`; once every `comboEveryMs`. Alone, it fizzles. */
  combo: number; comboCost: number; comboMs: number; comboEveryMs: number; comboReach: number;
}
export const DEFAULT_LANES: LaneParams = {
  version: LANES_VERSION, lanes: 3, stepMs: 100, baseHp: 20, slots: [0.2, 0.38],
  units: { soldier: { cost: 2, hp: 12, speed: 0.1, damage: 1 }, mouse: { cost: 4, hp: 5, speed: 0.13, damage: 1, pack: 3 }, tank: { cost: 6, hp: 40, speed: 0.055, damage: 3 } },
  towers: { snow: { cost: 4, dps: 5, range: 0.09, slow: 1 }, cannon: { cost: 7, dps: 6, range: 0.08, slow: 1, splash: 0.06 }, frost: { cost: 5, dps: 0, range: 0.14, slow: 0.5 } },
  sendMs: 500, zap: 4, zapCost: 2, zapMs: 500, zapReach: 0.55, patch: 3, patchCost: 4, patchMs: 3000, shieldCost: 5, shieldMs: 4000, shieldEveryMs: 12_000,
  purse: { start: 4, rate: 1, max: 12 },
  guard: 3, guardReach: 0.15,
  levelUp: 0.6, maxLevel: 3,
  combo: 30, comboCost: 3, comboMs: 2500, comboEveryMs: 20_000, comboReach: 0.6,
};

/** Energy that refills by the clock: `energy` as of `asOf`. */
export interface Purse { energy: number; asOf: number }
export const energyAt = (u: Purse, now: number, p: LaneParams['purse']) => Math.min(p.max, u.energy + (p.rate * Math.max(0, now - u.asOf)) / 1000);
/** Take `cost` out at `now`, or null if it is not there. */
export function spend(u: Purse, cost: number, now: number, p: LaneParams['purse']): Purse | null {
  const have = energyAt(u, now, p);
  return have + 1e-9 >= cost ? { energy: have - cost, asOf: now } : null;
}

export interface Unit { id: string; side: number; lane: number; kind: string; pos: number; hp: number; v: number }
export interface Tower { lane: number; slot: number; kind: string; /** 1 (absent) to `maxLevel`. */ level?: number }
/** How much stronger than its kind a tower is, for its level. */
export const powerOf = (t: Pick<Tower, 'level'>, p: LaneParams = DEFAULT_LANES) => 1 + p.levelUp * ((t.level || 1) - 1);
export interface Side {
  hp: number;
  towers: Tower[];
  /** A lane closed until a time (0: none), and when the next may be raised. */
  shieldLane: number; shieldUntil: number; shieldAt: number;
  patchAt: number; zapAt: number;
  /** The combo: when the next may go off, and the one waiting for a second player (`armLane` -1: none). */
  comboAt: number; armLane: number; armBy: string; armAt: number;
  sendAt: number[];
  sent: number; stopped: number;
}
export type LaneEventKind = 'sent' | 'built' | 'died' | 'arrived' | 'blocked' | 'zap' | 'patch' | 'shield' | 'armed' | 'combo';
export interface LaneEvent { n: number; at: number; kind: LaneEventKind; side: number; lane?: number; what?: string }
export interface Field {
  asOf: number;
  units: Unit[];
  sides: [Side, Side];
  events: LaneEvent[];
  n: number;
  ids: number;
}
const KEEP_EVENTS = 14;
/** How far apart the units of a pack set off. */
export const PACK_GAP = 0.022;

const newSide = (p: LaneParams): Side => ({ hp: p.baseHp, towers: [], shieldLane: -1, shieldUntil: 0, shieldAt: 0, patchAt: 0, zapAt: 0, comboAt: 0, armLane: -1, armBy: '', armAt: 0, sendAt: Array.from({ length: p.lanes }, () => 0), sent: 0, stopped: 0 });
export const newField = (now: number, p: LaneParams = DEFAULT_LANES): Field => ({ asOf: now, units: [], sides: [newSide(p), newSide(p)], events: [], n: 0, ids: 0 });

/** A stored field loses its empty lists: put them back. */
export function wholeField(f: Field, p: LaneParams = DEFAULT_LANES): Field {
  const side = (s: Side): Side => ({ ...newSide(p), ...s, towers: s?.towers || [], sendAt: Array.from({ length: p.lanes }, (_, i) => s?.sendAt?.[i] || 0) });
  return { ...f, units: f.units || [], sides: [side(f.sides?.[0]), side(f.sides?.[1])], events: f.events || [], n: f.n || 0, ids: f.ids || 0 };
}
/** Every time in the battle moved on by `ms` (the game was paused that long). */
export function shiftField(field: Field, ms: number, p: LaneParams = DEFAULT_LANES): Field {
  const f = wholeField(field, p), later = (t: number) => (t ? t + ms : t);
  return { ...f, asOf: f.asOf + ms, events: f.events.map((e) => ({ ...e, at: e.at + ms })),
    sides: f.sides.map((s) => ({ ...s, shieldUntil: later(s.shieldUntil), shieldAt: later(s.shieldAt), patchAt: later(s.patchAt), zapAt: later(s.zapAt), comboAt: later(s.comboAt), armAt: later(s.armAt), sendAt: s.sendAt.map(later) })) as [Side, Side] };
}

function say(f: Field, e: Omit<LaneEvent, 'n'>) { f.n += 1; f.events = [...f.events, { ...e, n: f.n }].slice(-KEEP_EVENTS); }
/** How far a unit is from the base it is walking at. */
const toGo = (u: Unit) => 1 - u.pos;
/** How far a tower reaches along its lane (a slowing tower reaches further for its level). */
export const reachOf = (t: Tower, p: LaneParams = DEFAULT_LANES) => { const k = p.towers[t.kind]; return k.range * (k.dps ? 1 : powerOf(t, p)); };
/** The unit a tower of `side` at `slot` in `lane` is aiming at: the one nearest its base, in reach. */
export function targetOf(f: Field, side: number, t: Tower, p: LaneParams = DEFAULT_LANES): Unit | null {
  const at = p.slots[t.slot];
  let best: Unit | null = null;
  for (const u of f.units) {
    if (u.side === side || u.lane !== t.lane || Math.abs(toGo(u) - at) > reachOf(t, p)) continue;
    if (!best || u.pos > best.pos || (u.pos === best.pos && u.id < best.id)) best = u;
  }
  return best;
}
/** The unit `side`'s own base is throwing at in `lane`: the one nearest it, within its reach. */
export function guardTarget(f: Field, side: number, lane: number, p: LaneParams = DEFAULT_LANES): Unit | null {
  if (!p.guard) return null;
  let best: Unit | null = null;
  for (const u of f.units) {
    if (u.side === side || u.lane !== lane || toGo(u) > p.guardReach) continue;
    if (!best || u.pos > best.pos || (u.pos === best.pos && u.id < best.id)) best = u;
  }
  return best;
}
/** Where a unit is at `now` (a screen's glide between the room's steps). */
export const unitAt = (u: Pick<Unit, 'pos' | 'v'>, asOf: number, now: number) => Math.min(1, u.pos + (u.v * Math.max(0, now - asOf)) / 1000);

/** One step of the battle, `dt` ms long, ending at `at`. */
function step(f: Field, dt: number, at: number, p: LaneParams) {
  // speeds first (a frost lamp in reach slows), then everyone walks
  for (const u of f.units) {
    const enemy = f.sides[1 - u.side];
    let slow = 1;
    for (const t of enemy.towers) { const k = p.towers[t.kind]; if (k.slow < 1 && t.lane === u.lane && Math.abs(toGo(u) - p.slots[t.slot]) <= reachOf(t, p)) slow = Math.min(slow, k.slow); }
    u.v = p.units[u.kind].speed * slow;
    u.pos = Math.min(1, u.pos + (u.v * dt) / 1000);
  }
  // every tower hurts the unit it is aiming at
  for (const side of [0, 1]) for (const t of f.sides[side].towers) {
    const k = p.towers[t.kind];
    if (!k.dps) continue;
    const u = targetOf(f, side, t, p);
    if (!u) continue;
    const hurt = (k.dps * powerOf(t, p) * dt) / 1000;
    if (k.splash) { for (const v of f.units) if (v.side !== side && v.lane === t.lane && Math.abs(v.pos - u.pos) <= k.splash) v.hp -= hurt; }
    else u.hp -= hurt;
  }
  // and each base, at whatever has got close
  if (p.guard) for (const side of [0, 1]) for (let lane = 0; lane < p.lanes; lane++) {
    const u = guardTarget(f, side, lane, p);
    if (u) u.hp -= (p.guard * dt) / 1000;
  }
  // the fallen, then the arrived (in the order they were sent)
  const left: Unit[] = [];
  for (const u of f.units) {
    const enemy = f.sides[1 - u.side];
    if (u.hp <= 0) { enemy.stopped += 1; say(f, { at, kind: 'died', side: u.side, lane: u.lane, what: u.kind }); continue; }
    if (u.pos < 1) { left.push(u); continue; }
    if (enemy.shieldLane === u.lane && at < enemy.shieldUntil) { enemy.stopped += 1; say(f, { at, kind: 'blocked', side: u.side, lane: u.lane, what: u.kind }); continue; }
    enemy.hp = Math.max(0, enemy.hp - p.units[u.kind].damage);
    say(f, { at, kind: 'arrived', side: u.side, lane: u.lane, what: u.kind });
  }
  f.units = left;
}

/** The battle brought up to `now`, in whole steps (what is left over waits for the next call). Stops the
 *  moment a base falls. */
export function advance(field: Field, now: number, p: LaneParams = DEFAULT_LANES): Field {
  const f = wholeField(field, p);
  f.units = f.units.map((u) => ({ ...u }));
  f.sides = f.sides.map((s) => ({ ...s, towers: [...s.towers], sendAt: [...s.sendAt] })) as [Side, Side];
  for (let turns = 0; turns < 6000 && f.asOf + p.stepMs <= now; turns++) {
    if (f.sides[0].hp <= 0 || f.sides[1].hp <= 0) break;
    f.asOf += p.stepMs;
    if (f.units.length) step(f, p.stepMs, f.asOf, p);
  }
  if (!f.units.length && f.asOf + p.stepMs <= now && f.sides[0].hp > 0 && f.sides[1].hp > 0) f.asOf += Math.floor((now - f.asOf) / p.stepMs) * p.stepMs;
  return f;
}
/** Is anything on the field moving by itself (the room needs to keep waking)? */
export const busy = (f: Field) => (f.units || []).length > 0;

export type Did = { field: Field; error?: undefined } | { field?: undefined; error: string };

/** Send a unit down a lane (the field is already advanced to `now`). `tough`: how much sturdier than its kind it is. */
export function send(field: Field, side: number, lane: number, kind: string, now: number, p: LaneParams = DEFAULT_LANES, tough = 1): Did {
  const f = wholeField(field, p);
  if (!p.units[kind]) return { error: 'No such toy' };
  if (!(lane >= 0 && lane < p.lanes)) return { error: 'No such lane' };
  if (now < f.sides[side].sendAt[lane]) return { error: 'That lane is busy' };
  const pack = Math.max(1, p.units[kind].pack || 1);
  f.units = [...f.units];
  for (let i = 0; i < pack; i++) { f.ids += 1; f.units.push({ id: `u${f.ids}`, side, lane, kind, pos: (pack - 1 - i) * PACK_GAP, hp: p.units[kind].hp * tough, v: p.units[kind].speed }); }
  f.sides = f.sides.map((s, i) => (i === side ? { ...s, sent: s.sent + 1, sendAt: s.sendAt.map((t, l) => (l === lane ? now + p.sendMs : t)) } : s)) as [Side, Side];
  say(f, { at: now, kind: 'sent', side, lane, what: kind });
  return { field: f };
}
/** Build a tower on one of your own spots: an empty one; the same kind again makes it better (a level); a dearer
 *  kind takes a cheaper one's place. */
export function build(field: Field, side: number, lane: number, slot: number, kind: string, now: number, p: LaneParams = DEFAULT_LANES): Did {
  const f = wholeField(field, p);
  if (!p.towers[kind]) return { error: 'No such tower' };
  if (!(lane >= 0 && lane < p.lanes && slot >= 0 && slot < p.slots.length)) return { error: 'No such spot' };
  const there = f.sides[side].towers.find((t) => t.lane === lane && t.slot === slot);
  const up = !!there && there.kind === kind;
  if (up && (there!.level || 1) >= p.maxLevel) return { error: 'That one is as good as it gets' };
  if (there && !up && p.towers[there.kind].cost >= p.towers[kind].cost) return { error: 'Something as good is built there' };
  const next: Tower = up ? { ...there!, level: (there!.level || 1) + 1 } : { lane, slot, kind };
  f.sides = f.sides.map((s, i) => (i === side ? { ...s, towers: [...s.towers.filter((t) => t !== there), next] } : s)) as [Side, Side];
  say(f, { at: now, kind: 'built', side, lane, what: `${kind}:${slot}:${next.level || 1}` });
  return { field: f };
}
/** Throw a shot at the lead enemy unit in a lane, if it is on your half. */
export function zap(field: Field, side: number, lane: number, now: number, p: LaneParams = DEFAULT_LANES): Did {
  const f = wholeField(field, p);
  if (now < f.sides[side].zapAt) return { error: 'Easy!' };
  const u = f.units.filter((x) => x.side !== side && x.lane === lane && toGo(x) <= p.zapReach).sort((a, b) => b.pos - a.pos)[0];
  if (!u) return { error: 'Nothing in reach in that lane' };
  f.units = f.units.map((x) => (x === u ? { ...x, hp: x.hp - p.zap } : x)).filter((x) => x.hp > 0);
  f.sides = f.sides.map((s, i) => (i === side ? { ...s, zapAt: now + p.zapMs, stopped: s.stopped + (u.hp - p.zap <= 0 ? 1 : 0) } : s)) as [Side, Side];
  say(f, { at: now, kind: 'zap', side, lane, what: u.hp - p.zap <= 0 ? 'down' : u.kind });
  return { field: f };
}
/** Mend your base a little. */
export function patch(field: Field, side: number, now: number, p: LaneParams = DEFAULT_LANES): Did {
  const f = wholeField(field, p), s = f.sides[side];
  if (now < s.patchAt) return { error: 'The glue is still wet' };
  if (s.hp >= p.baseHp) return { error: 'Nothing to mend' };
  f.sides = f.sides.map((x, i) => (i === side ? { ...x, hp: Math.min(p.baseHp, x.hp + p.patch), patchAt: now + p.patchMs } : x)) as [Side, Side];
  say(f, { at: now, kind: 'patch', side });
  return { field: f };
}
/** Close a lane at your base for a moment: whatever arrives is stopped. */
export function shield(field: Field, side: number, lane: number, now: number, p: LaneParams = DEFAULT_LANES): Did {
  const f = wholeField(field, p), s = f.sides[side];
  if (!(lane >= 0 && lane < p.lanes)) return { error: 'No such lane' };
  if (now < s.shieldAt) return { error: 'The shield is recharging' };
  f.sides = f.sides.map((x, i) => (i === side ? { ...x, shieldLane: lane, shieldUntil: now + p.shieldMs, shieldAt: now + p.shieldEveryMs } : x)) as [Side, Side];
  say(f, { at: now, kind: 'shield', side, lane });
  return { field: f };
}
/** Is `side`'s combo armed on a lane right now (a second player may still join)? */
export const armed = (s: Side, now: number, p: LaneParams = DEFAULT_LANES) => s.armLane >= 0 && now < s.armAt + p.comboMs;
/** The combo: the first player arms it on a lane; a second (a different one), on the same lane within the window,
 *  sets it off: every unit of theirs on your half of that lane is hurt. `alone`: nobody else is there to join, so
 *  one press is enough. Armed on one lane and pressed on another, it moves there (and waits again). */
export function arm(field: Field, side: number, lane: number, who: string, now: number, p: LaneParams = DEFAULT_LANES, alone = false): Did {
  const f = wholeField(field, p), s = f.sides[side];
  if (!(lane >= 0 && lane < p.lanes)) return { error: 'No such lane' };
  if (now < s.comboAt) return { error: 'Not ready yet' };
  const waiting = armed(s, now, p);
  if (waiting && s.armBy === who && s.armLane === lane) return { error: 'It needs a second pair of hands!' };
  if (!alone && !(waiting && s.armLane === lane && s.armBy !== who)) {
    f.sides = f.sides.map((x, i) => (i === side ? { ...x, armLane: lane, armBy: who, armAt: now } : x)) as [Side, Side];
    say(f, { at: now, kind: 'armed', side, lane });
    return { field: f };
  }
  let down = 0;
  f.units = f.units.map((u) => (u.side !== side && u.lane === lane && toGo(u) <= p.comboReach ? { ...u, hp: u.hp - p.combo } : u)).filter((u) => (u.hp > 0 ? true : (down += 1, false)));
  f.sides = f.sides.map((x, i) => (i === side ? { ...x, armLane: -1, armBy: '', armAt: 0, comboAt: now + p.comboEveryMs, stopped: x.stopped + down } : x)) as [Side, Side];
  say(f, { at: now, kind: 'combo', side, lane, what: String(down) });
  return { field: f };
}
