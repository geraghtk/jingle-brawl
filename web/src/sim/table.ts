// A whole party of Yule Duel on a virtual clock, through the real engine: everyone at the table
// acts through the buttons on their own screen (core/table.ts), the host's automation runs on the
// same steps as the host page (core/hostauto.ts), and what each person sees is the TV plus their
// own phone (core/screens.ts). The people's skills, tastes and hands are sim/world.ts, shared with
// the live-app driver (sim/live.ts). Seeded: the same seed and the same presses give the same party.
import { hostAuto } from '../core/hostauto';
import type { AutoStep } from '../core/hostauto';
import { GAMES } from '../core/party';
import type { PartyGame } from '../core/party';
import * as A from '../core/actions';
import * as D from '../core/duel';
import { rng as makeRng } from '../core/rng';
import type { Rng } from '../core/rng';
import { phoneScreen, tvScreen } from '../core/screens';
import type { Screen } from '../core/screens';
import { HOST, isOffered, offers, perform } from '../core/table';
import type { Offer, Who } from '../core/table';
import { commitPlan } from '../core/commit';
import { dueStep } from '../core/scheduler';
import type { Game, Plan, Settings } from '../core/types';
import { avatarLabel, drawingsOf, GIFTS, PatientHost, Physics, pickAvatar, situation } from './world';
import type { AvatarPicks, HostOptions, Person, Situation } from './world';

export const START = 1_700_000_000_000;

export interface TableOptions {
  seed: number;
  people: Person[];
  settings?: Settings;          // as the host's setup screen makes them ({} = the house defaults)
  autoHost?: boolean | HostOptions;
}

export interface TableEvent { seq: number; t: number; text: string; who?: Who; kind: 'event' | 'act'; private?: boolean }

export interface Observation extends Situation {
  now: number;                  // ms since the party began
  tv: Screen;
  phone: Screen | null;         // null for the host
  offers: Offer[];
  drawings?: Record<string, number>; // Sketch-Off voting: how good each drawing looks
}

export class Table {
  game: Game;
  now = START;
  readonly people: Person[];
  readonly log: TableEvent[] = [];
  readonly did: Record<Who, Record<string, number>> = {};
  readonly physics: Physics;
  readonly avatars: AvatarPicks = {};
  private r: Rng;
  private fired = new Set<string>();
  private host: PatientHost | null;

  constructor(o: TableOptions) {
    this.r = makeRng(o.seed);
    this.people = o.people;
    this.physics = new Physics(o.seed, o.people);
    const ah = o.autoHost === true ? {} : o.autoHost || null;
    this.host = ah ? new PatientHost(ah, this.now) : null;
    let g = A.newGame({ hostId: HOST, hostUid: null, settings: { ...o.settings }, now: this.now });
    const r = makeRng(o.seed + 7919), taken = new Set<string>();                  // the avatar picker, as each joins
    for (const p of o.people) {
      const { press, avatar } = pickAvatar(p, taken, r);
      g = this.applyTo(g, A.joinPlayer(g, { playerId: p.id, name: p.name, uid: null, now: this.now, age: p.age ?? null, avatar: press }));
      const got = g.players?.[p.id]?.avatar || avatar;
      taken.add(got);
      this.avatars[p.id] = { avatar: got, label: avatarLabel(got), picked: !!press };
    }
    this.game = g;
  }

  // --- acting --------------------------------------------------------------------------------

  private applyTo(g: Game, plan: Plan): Game {
    return commitPlan(g, plan, this.now).game;
  }

  private note(text: string, who?: Who, kind: TableEvent['kind'] = 'event', priv = false) {
    this.log.push({ seq: this.log.length + 1, t: this.now - START, text, who, kind, ...(priv ? { private: true } : {}) });
  }

  private run(plan: Plan): boolean {
    if (plan.error) return false;
    const before = this.game;
    this.game = this.applyTo(this.game, plan);
    if (this.game === before) return false;              // a claim already taken
    this.host?.changed(this.now);
    for (const e of plan.events) this.note(e.text);
    if (plan.then) this.run(plan.then === 'drawEndgamePlayer' ? D.drawEndgamePlayer(this.game, this.r) : D.checkEndgameContinue(this.game, this.r));
    return true;
  }

  offers(who: Who): Offer[] {
    return offers(this.game, who, this.now);
  }

  /** `who` presses `offer`. Their hands fill in what they did when the offer leaves it open. */
  act(who: Who, offer: Offer): { ok: boolean; error?: string } {
    if (!isOffered(this.game, who, offer, this.now)) return { ok: false, error: `${offer.type} isn't on that screen now` };
    let o = offer;
    const mg = this.game.minigame as PartyGame | undefined;
    if (o.type === 'whack' && o.score == null && mg) o = { type: 'whack', score: this.physics.whackScore(who, mg.seed), final: true };
    if (o.type === 'sketch' && !o.drawing) o = { type: 'sketch', drawing: `data:,sketch;q=${this.physics.sketchQuality(who).toFixed(2)}` };
    const giftName = o.type === 'draw' && !o.giftName ? GIFTS[Object.keys(this.game.gifts || {}).length % GIFTS.length] : undefined;
    const plan = perform(this.game, who, o, { now: this.now, rng: this.r, giftName });
    if (plan.error) return { ok: false, error: plan.error };
    if (!this.run(plan)) return { ok: false, error: 'someone got there first' };
    (this.did[who] ||= {})[o.type] = (this.did[who][o.type] || 0) + 1;
    const sealed = o.type === 'bid';                      // a sealed bid is nobody else's business
    this.note(`${who === HOST ? 'The host' : this.game.players?.[who]?.name || who} ${describe(o, this.game)}`, who, 'act', sealed);
    return { ok: true };
  }

  // --- time ----------------------------------------------------------------------------------

  /** The host page's automatic step due soonest by `until` (each key once while it stands). */
  private dueStep(until: number): AutoStep | null {
    return dueStep(hostAuto(this.game), this.fired, this.now, until);
  }

  /** Let `ms` pass: the host page's timers fire, and the scripted host (if any) does its job. */
  advance(ms: number): TableEvent[] {
    const from = this.log.length;
    const until = this.now + Math.max(0, ms);
    for (let guard = 0; guard < 10000; guard++) {
      const step = this.dueStep(until);
      const hostAt = this.host ? this.host.due(this.game, this.now, () => this.physics.duelOutcome(this.game)) : null;
      const stepAt = step ? Math.max(this.now, step.at || this.now) : Infinity;
      const nextHost = hostAt ? Math.max(this.now, hostAt.at) : Infinity;
      const next = Math.min(stepAt, nextHost);
      if (next > until) break;
      this.now = next;
      if (step && stepAt <= nextHost) {
        this.fired.add(step.key);
        this.run(step.run(this.game, { now: this.now, rng: this.r }));
      } else if (hostAt && !this.act(HOST, hostAt.offer).ok) {
        this.host!.waits(this.game, this.now);             // try again after a pause
      }
    }
    this.now = until;
    return this.log.slice(from);
  }

  // --- what people see -----------------------------------------------------------------------

  observe(who: Who): Observation {
    const person = who === HOST ? null : who;
    const mg = this.game.minigame as PartyGame | undefined;
    return {
      now: this.now - START, tv: tvScreen(this.game, this.now), phone: who === HOST ? null : phoneScreen(this.game, who, this.now),
      offers: this.offers(who), ...situation(this.game, person, person, this.physics),
      ...(mg?.phase === 'voting' && mg.type in GAMES ? { drawings: drawingsOf(this.game) } : {}),
    };
  }

  get ended() { return this.game.phase === 'ended'; }

  result() {
    const seats = this.people.map((p) => {
      const pl = this.game.players![p.id];
      const gift = pl.giftId ? this.game.gifts?.[pl.giftId] : undefined;
      return { id: p.id, name: p.name, chips: pl.chips, gift: gift?.name ?? null, value: gift ? this.physics.value(p.id, gift.name) : 0,
        partyPoints: pl.partyPoints || 0, did: this.did[p.id] || {} };
    });
    return { ended: this.ended, minutes: Math.round((this.now - START) / 600) / 100, seats, host: this.did[HOST] || {} };
  }
}

/** A phone's buttons in words, for whoever watches. Offers name people by their own ids, which on the
 *  real app aren't its player ids: their names come from the people. */
export function buttons(offers: Offer[], game: Game, people: { id: string; name: string }[]): string[] {
  const named = { ...game, players: { ...game.players, ...Object.fromEntries(people.map((p) => [p.id, { name: p.name }])) } } as Game;
  return offers.map((o) => describe(o, named));
}

/** One line for the table's log. */
export function describe(o: Offer, game: Game): string {
  const nm = (id: string) => game.players?.[id]?.name || id;
  switch (o.type) {
    case 'bid': return `seals a bid of ${o.amount}`;
    case 'call_duel': return `backs ${nm(o.pick)}`;
    case 'vote_sketch': return `votes for ${nm(o.pick)}'s drawing`;
    case 'duel_winner': return `declares ${nm(o.id)} the winner`;
    case 'whack': return `whacks elves (${o.score} points)`;
    case 'sketch': return 'hands in a drawing';
    case 'gambit': case 'reprisal': case 'steal': return `${o.type}s ${game.gifts?.[o.giftId]?.name || 'a gift'}`;
    case 'pick_duel': return `picks ${o.name}`;
    case 'path': return o.path === 'steal' ? 'goes to steal a gift' : 'puts the Misfit up for auction';
    case 'nobids_defender': return `challenges ${nm(o.playerId)}`;
    default: return VERBS[o.type] || o.type.replace(/_/g, ' ');
  }
}

const VERBS: Partial<Record<Offer['type'], string>> = {
  keep: 'keeps their gift', fight: 'chooses to fight', yield: 'yields', spin: 'lets the wheel pick', skip_reprisal: 'lets it go',
  cancel_steal: 'goes back', start: 'starts the game', draw: "draws from Santa's sack", reveal: 'reveals the bids',
  start_duel: 'starts the duel', start_yield_duel: 'starts the yield duel', host_spin: 'spins the wheel', close_voting: 'closes the voting',
  start_duel_app: 'starts the duel game', skip_party: 'moves the party game on', endgame: 'starts the Misfit Lottery',
  reveal_auction: 'reveals the auction bids', auction_duel: 'calls the auction duel', end: 'ends the game',
  skip_bidding: 'closes bidding with no bids', skip_auction: 'closes the auction with no bids', complete_drawoff: 'moves on to the duel',
};
