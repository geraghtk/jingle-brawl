// The world around the game, shared by the engine table (sim/table.ts) and the live-app driver
// (sim/live.ts): the people's skills and tastes, what happens when they use their hands, and a
// scripted host who presses the host's buttons as a patient host would.
//
// Skills are 0..1 (from `people[].skills`, 0.5 when not given):
//   duel   a table duel (thumb war, rock-paper-scissors…): the better duellist wins more often
//   whack  Whack-an-Elf: reaction time against the real pop-up sequence
//   sketch Sketch-Off: how good the drawing looks to the people voting on it
// A gift's value to a person is its quality plus their own taste (as in the economics sim).
import { whackPoints, whackSequence } from '../core/party';
import type { PartyGame } from '../core/party';
import { rng as makeRng } from '../core/rng';
import type { Rng } from '../core/rng';
import { HOST, offers } from '../core/table';
import type { Offer } from '../core/table';
import type { Game, PlayerId } from '../core/types';

export const GIFTS = [
  'Lava Lamp', 'Fancy Socks', 'Scented Candle', 'Rubber Duck', 'Ugly Sweater', 'Bluetooth Speaker',
  'Chia Pet', 'Hot Sauce Kit', 'Fondue Set', 'Snow Globe', 'Puzzle Box', 'Garden Gnome',
  'Coffee Sampler', 'Desk Cactus', 'Board Game', 'Bath Bombs', 'Novelty Mug', 'Beanie',
  'Plush Llama', 'Mini Drone', 'Waffle Maker', 'Tea Towel', 'Cookbook', 'Yoga Mat',
];

export interface Skills { duel: number; whack: number; sketch: number }
export interface Person { id: string; name: string; age?: number; skills?: Partial<Skills> }

export const hash = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0; };
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** People's skills and tastes, and what their hands do. Its own random stream. */
export class Physics {
  private quality: Record<string, number> = {};
  private taste: Record<string, number> = {};
  private physical: Record<string, PlayerId> = {};
  private r: Rng;
  constructor(private seed: number, private people: Person[]) { this.r = makeRng(hash(`${seed}|hands`)); }

  skill(id: string, k: keyof Skills): number {
    return clamp(this.people.find((p) => p.id === id)?.skills?.[k] ?? 0.5, 0, 1);
  }

  /** What a gift is worth to this person (0..12): its quality plus their own taste. Private. */
  value(id: string, giftName: string): number {
    // its own random stream per gift and per person: looking never changes what happens next
    const q = makeRng(hash(`${this.seed}|${giftName}`));
    this.quality[giftName] ??= 1 + Math.floor(q() * 10);
    const k = `${id}|${giftName}`;
    const tr = makeRng(hash(`${this.seed}|${k}`));
    this.taste[k] ??= (tr() + tr() + tr() - 1.5) * 2;
    return Math.round(clamp(this.quality[giftName] + this.taste[k], 0, 12) * 100) / 100;
  }

  /** Who wins the duel being fought at the table (decided once per duel). `person` maps a
   *  player id to whose skills they are (the live app's ids aren't the people's). */
  duelOutcome(game: Game, person = (id: PlayerId) => id): PlayerId | null {
    const t = game.currentTurn;
    if (t?.phase !== 'duel' || !t.challengerId || !t.defenderId) return null;
    const k = `${t.challengerId}|${t.defenderId}|${t.giftId}|${typeof game.lastDuel?.at === 'number' ? game.lastDuel.at : 0}`;
    if (!this.physical[k]) {
      const edge = 0.5 + 0.4 * (this.skill(person(t.challengerId), 'duel') - this.skill(person(t.defenderId), 'duel'));
      this.physical[k] = this.r() < edge ? t.challengerId : t.defenderId;
    }
    return this.physical[k];
  }

  /** Reaction time (ms) to an elf popping up, this time. */
  reaction(id: string): number {
    return (850 - 550 * this.skill(id, 'whack')) * (0.6 + this.r() * 0.8);
  }

  /** Taps the Grinch by mistake? (Better players hold back.) */
  slips(id: string): boolean {
    return this.r() < 0.25 * (1 - this.skill(id, 'whack'));
  }

  /** A whole game of Whack-an-Elf on the real pop-up sequence, at this person's reaction time. */
  whackScore(id: string, seed: number): number {
    let score = 0;
    for (const pop of whackSequence(seed)) {
      const react = this.reaction(id);
      if (react >= pop.upMs) continue;                         // gone before they got to it
      if (pop.kind === 'grinch' && !this.slips(id)) continue;  // they held back
      score += whackPoints(pop.kind, react / pop.upMs);
    }
    return Math.max(0, score);
  }

  /** How a drawing turns out (0..1): this person's skill, plus a bit of luck. */
  sketchQuality(id: string): number {
    return Math.round(clamp(this.skill(id, 'sketch') * 0.7 + this.r() * 0.3, 0, 1) * 100) / 100;
  }
}

/** How good each drawing on the voting screen looks (0..1), as a person sees it. */
export function drawingsOf(game: Game, known: Record<PlayerId, number> = {}): Record<PlayerId, number> {
  const mg = game.minigame as PartyGame | undefined;
  const out: Record<PlayerId, number> = {};
  for (const [id, d] of Object.entries(mg?.submissions || {})) out[id] = known[id] ?? Number(/q=([\d.]+)/.exec(d)?.[1] ?? 0.5);
  return out;
}

export interface HostOptions { patienceMs?: number; paceMs?: number; duelMs?: number }

/** A patient host (scripted): waits for the table, gives a duel time to be fought, keeps it moving. */
export class PatientHost {
  readonly o: Required<HostOptions>;
  private changedAt: number;
  private since: Record<string, number> = {};
  constructor(o: HostOptions, now: number) {
    this.o = { patienceMs: o.patienceMs ?? 90000, paceMs: o.paceMs ?? 12000, duelMs: o.duelMs ?? 40000 }; // a host who talks between steps
    this.changedAt = now;
  }
  private key(game: Game) {
    const t = game.currentTurn;
    return `${game.phase}|${t?.phase}|${t?.giftId}|${t?.challengerId}|${t?.defenderId}`;
  }
  /** Something happened at the table. */
  changed(now: number) { this.changedAt = now; }
  /** Pressing didn't work: wait a while before trying again. */
  waits(game: Game, now: number) { this.changedAt = now; this.since[this.key(game)] = now; }

  /** What the host presses next, and when. `winner`: the duel's real winner, as the host saw it. */
  due(game: Game, now: number, winner: () => PlayerId | null): { at: number; offer: Offer } | null {
    const offered = offers(game, HOST, now);
    const has = (type: Offer['type']) => offered.find((o) => o.type === type);
    const calm = this.changedAt + this.o.paceMs;
    const k = this.key(game);
    this.since[k] ??= now;
    const since = this.since[k];
    if (has('start')) return { at: calm, offer: { type: 'start' } };
    if (has('draw')) return { at: calm, offer: { type: 'draw' } };
    if (has('endgame')) return { at: calm, offer: { type: 'endgame' } };
    if (has('reveal') || has('reveal_auction')) {
      const waiting = Object.keys(game.players || {}).some((id) => offers(game, id, now).some((o) => o.type === 'bid'));
      return { at: waiting ? since + this.o.patienceMs : calm, offer: { type: has('reveal') ? 'reveal' : 'reveal_auction' } };
    }
    for (const type of ['start_duel', 'start_yield_duel', 'auction_duel', 'start_duel_app'] as const) if (has(type)) return { at: calm, offer: { type } };
    const w = has('duel_winner') ? winner() : null;
    if (w) return { at: Math.max(calm, since + this.o.duelMs), offer: { type: 'duel_winner', id: w } };
    // a guest's own move that isn't coming (stepped out, phone dead): the host covers for them
    const cover = offered.find((o) => ['keep', 'fight', 'skip_reprisal', 'path', 'nobids_defender'].includes(o.type));
    if (cover) return { at: since + this.o.patienceMs, offer: cover };
    if (has('close_voting')) return { at: since + this.o.patienceMs, offer: { type: 'close_voting' } };
    if (has('host_spin')) return { at: since + this.o.patienceMs, offer: { type: 'host_spin' } };
    if (offered.length === 1 && has('end')) return { at: calm, offer: { type: 'end' } };
    return null;
  }
}

export interface Situation {
  you: { chips: number; giftId: string | null } | null;
  focus: string | null;                                   // the gift the turn is about
  roles: { opener?: string; challenger?: string; defender?: string; active?: string };
  gifts: Record<string, { name: string; holder: string; value?: number }>; // value: to you (yours alone)
  values: Record<string, number>;
}

/** What a person knows of the game at a glance (the TV's holdings, their own phone and their own
 *  tastes), in the people's own ids. */
export function situation(game: Game, me: PlayerId | null, person: string | null, physics: Physics, personOf = (id: PlayerId) => id): Situation {
  const gifts: Situation['gifts'] = {};
  const holder = (owner: string) => (owner === 'misfit' ? 'the Misfit pile' : game.players?.[owner]?.name ?? 'someone');
  for (const [id, g] of Object.entries(game.gifts || {})) gifts[id] = { name: g.name, holder: holder(g.ownerId), ...(person ? { value: physics.value(person, g.name) } : {}) };
  const t = game.currentTurn;
  if (t?.misfitId && t.misfitName && !gifts[t.misfitId]) gifts[t.misfitId] = { name: t.misfitName, holder: 'the Misfit pile', ...(person ? { value: physics.value(person, t.misfitName) } : {}) };
  const p = me ? game.players?.[me] : undefined;
  const role = (id?: string) => (id ? personOf(id) : undefined);
  const roles = Object.fromEntries(Object.entries({ opener: role(t?.openerId), challenger: role(t?.challengerId), defender: role(t?.defenderId), active: role(t?.activePlayerId) })
    .filter(([, v]) => v)) as Situation['roles'];
  const values = Object.fromEntries(Object.entries(gifts).filter(([, g]) => g.value != null).map(([id, g]) => [id, g.value!]));
  return { you: p ? { chips: p.chips, giftId: p.giftId ?? null } : null, focus: t?.giftId ?? t?.misfitId ?? null, roles, gifts, values };
}
