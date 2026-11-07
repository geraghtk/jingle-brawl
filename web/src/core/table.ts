// The table as people meet it: what each person (a player, or the host) can do right now, and
// doing it. Each offer is a button on that person's own screen (the phone, or the host's
// console), offered only when the app shows it enabled; perform() builds the same plan the
// button's click does. Simulated guests (sim/table.ts) and the live-app driver
// (agents/live.js) both act through these, so they press exactly what a person could.
//
// Some offers carry what the person did with their hands, filled in by whoever plays them:
//   whack   {score, final}   their Whack-an-Elf score so far (final: time's up)
//   sketch  {drawing}        their Sketch-Off / Draw Off drawing (an image data URL)
//   draw    {giftName, seed} the gift being opened; the seed lets the console name who is drawn first
import * as A from './actions';
import * as D from './duel';
import * as DO from './drawoff';
import type { DrawOff } from './drawoff';
import { chooseDuelGame, duelAppWinner, duelGameStep, duelOptions, duelPicker, SPIN_MS, spinDuelGame, startDuelApp } from './duelgame';
import { GAMES } from './party';
import type { PartyGame } from './party';
import { advanceParty, reportWhack, submitSketch, voteSketch } from './partyflow';
import { rng as seeded } from './rng';
import type { Rng } from './rng';
import { gambitTargets, giftless, misfitPile, rankedBids, reprisalTargets, stealTargets, topTiedBidders, wrappedGifts } from './rules';
import { config, minCost } from './settings';
import { refuse } from './tx';
import type { Game, Plan, PlayerId } from './types';
import { closeVoting, voteStatus } from './voting';

export const HOST = 'host';
export type Who = PlayerId | typeof HOST;

export type Offer =
  // a phone
  | { type: 'bid'; amount: number }
  | { type: 'keep' }
  | { type: 'gambit'; giftId: string }
  | { type: 'fight' }
  | { type: 'yield' }
  | { type: 'pick_duel'; index: number; name: string }
  | { type: 'spin' }
  | { type: 'call_duel'; pick: PlayerId }
  | { type: 'reprisal'; giftId: string }
  | { type: 'skip_reprisal' }
  | { type: 'path'; path: 'steal' | 'auction' }
  | { type: 'steal'; giftId: string }
  | { type: 'cancel_steal' }
  | { type: 'nobids_defender'; playerId: PlayerId }
  | { type: 'whack'; score?: number; final?: boolean }
  | { type: 'sketch'; drawing?: string }
  | { type: 'vote_sketch'; pick: PlayerId }
  // the host's console
  | { type: 'start' }
  | { type: 'draw'; giftName?: string; seed?: number }
  | { type: 'reveal' }
  | { type: 'skip_bidding' }
  | { type: 'start_duel' }
  | { type: 'start_yield_duel' }
  | { type: 'host_spin' }
  | { type: 'close_voting' }
  | { type: 'start_duel_app' }
  | { type: 'duel_winner'; id: PlayerId }
  | { type: 'skip_party' }
  | { type: 'endgame' }
  | { type: 'reveal_auction' }
  | { type: 'skip_auction' }
  | { type: 'complete_drawoff' }
  | { type: 'auction_duel' }
  | { type: 'end' };

const affordable = (game: Game, pid: PlayerId, targets: [string, { naughtyLevel?: number }][]) => {
  const cfg = config(game.settings);
  const chips = game.players?.[pid]?.chips || 0;
  return targets.filter(([, g]) => minCost(cfg, g.naughtyLevel || 0) <= chips).map(([id]) => id);
};

/** The wheel is still turning on every screen (results stay hidden until it stops). */
const spinning = (game: Game, now: number) => {
  const at = game.currentTurn?.wheel?.spunAt;
  return typeof at === 'number' && now < at + SPIN_MS;
};

/** Everything `who` could press right now. */
export function offers(game: Game, who: Who, now: number): Offer[] {
  return who === HOST ? hostOffers(game, now) : phoneOffers(game, who, now);
}

function phoneOffers(game: Game, me: PlayerId, now: number): Offer[] {
  const t = game.currentTurn;
  const p = game.players?.[me];
  if (!p || !t) return [];
  const cfg = config(game.settings);
  const out: Offer[] = [];
  switch (t.phase) {
    case 'bidding': case 'endgame_bidding': {
      const owner = t.phase === 'bidding' ? t.openerId : t.activePlayerId;
      if (me === owner || t.bids?.[me]) return [];
      const max = Math.min(p.chips, cfg.maxBid);
      for (let i = 0; i <= max; i++) out.push({ type: 'bid', amount: i });
      return out;
    }
    case 'reveal': {
      if (t.openerId !== me) return [];
      const bids = rankedBids(t).filter(([, a]) => a > 0);
      if (bids.length === 0) {
        out.push({ type: 'keep' });
        for (const id of affordable(game, me, gambitTargets(game, t.giftId, me))) out.push({ type: 'gambit', giftId: id });
        return out;
      }
      if (bids.length === 1) return [];  // the host starts the duel
      return [{ type: 'fight' }, { type: 'yield' }];
    }
    case 'yield_choice':
      return t.openerId === me ? [{ type: 'fight' }, { type: 'yield' }] : [];
    case 'duel': {
      const step = duelGameStep(game);
      if (step === 'picker') {
        if (duelPicker(game) !== me) return [];
        duelOptions(game).forEach((o, i) => out.push({ type: 'pick_duel', index: i, name: o.name }));
        out.push({ type: 'spin' });
        return out;
      }
      if (step || spinning(game, now) || t.isTiebreak) return [];
      if (me === t.challengerId || me === t.defenderId || t.duelVotes?.[me]) return [];
      if (!voteStatus(game, now).accepting || duelAppWinner(t)) return [];
      return [{ type: 'call_duel', pick: t.challengerId! }, { type: 'call_duel', pick: t.defenderId! }];
    }
    case 'reprisal': {
      if (t.reprisalPlayerId !== me) return [];
      for (const id of affordable(game, me, reprisalTargets(game, t.excludedGiftId, me))) out.push({ type: 'reprisal', giftId: id });
      out.push({ type: 'skip_reprisal' });
      return out;
    }
    case 'endgame_choice':
      if (t.activePlayerId !== me) return [];
      if (stealTargets(game, me).length) out.push({ type: 'path', path: 'steal' });
      out.push({ type: 'path', path: 'auction' });
      return out;
    case 'endgame_steal_select':
      if (t.activePlayerId !== me) return [];
      for (const id of affordable(game, me, stealTargets(game, me))) out.push({ type: 'steal', giftId: id });
      out.push({ type: 'cancel_steal' });
      return out;
    case 'endgame_nobids':
      if (t.activePlayerId !== me) return [];
      return Object.keys(game.players || {}).filter((id) => id !== me).map((id) => ({ type: 'nobids_defender', playerId: id }));
    case 'minigame_party': {
      const mg = game.minigame as PartyGame | undefined;
      if (!mg || !(mg.type in GAMES) || !mg.participants?.includes(me)) return [];
      if (mg.phase === 'playing' && mg.type === 'whack_elf' && !mg.scores?.[me]?.final && now < mg.phaseEndsAt) return [{ type: 'whack' }];
      if (mg.phase === 'playing' && mg.type === 'sketch_off' && !mg.submissions?.[me]) return [{ type: 'sketch' }];
      if (mg.phase === 'voting' && !mg.votes?.[me]) {
        return Object.keys(mg.submissions || {}).filter((id) => id !== me).map((id) => ({ type: 'vote_sketch', pick: id }));
      }
      return [];
    }
    case 'minigame_draw_off': {
      const mg = game.minigame as DrawOff | undefined;
      return mg?.type === 'draw_off' && mg.phase === 'drawing' && mg.participants.includes(me) && !mg.submissions?.[me] ? [{ type: 'sketch' }] : [];
    }
    default:
      return [];
  }
}

function hostOffers(game: Game, now: number): Offer[] {
  const t = game.currentTurn;
  if (game.phase === 'lobby') return Object.keys(game.players || {}).length >= 2 ? [{ type: 'start' }] : [];
  if (game.phase === 'ended') return [];
  if (!t) {
    if (game.phase !== 'main') return [{ type: 'end' }];
    const out: Offer[] = [];
    const bag = giftless(game).length;
    if (wrappedGifts(game) > 0 && bag > 0) out.push({ type: 'draw' });
    else if (bag > 0 && misfitPile(game).length) out.push({ type: 'endgame' });
    out.push({ type: 'end' });
    return out;
  }
  switch (t.phase) {
    case 'bidding': return [{ type: 'reveal' }, { type: 'skip_bidding' }];
    // the console's Manual Override (for a player who can't use their phone): keep or a Gambit,
    // fight / yield, the Reprisal, the endgame path, the auction, a defender
    case 'reveal': {
      const n = rankedBids(t).filter(([, a]) => a > 0).length;
      if (n === 0) return [{ type: 'keep' }, ...affordable(game, t.openerId!, gambitTargets(game, t.giftId, t.openerId!)).map((id): Offer => ({ type: 'gambit', giftId: id }))];
      return n === 1 ? [{ type: 'start_duel' }] : [{ type: 'fight' }, { type: 'yield' }];
    }
    case 'yield_choice': return [{ type: 'fight' }, { type: 'yield' }];
    case 'reprisal': return [
      ...affordable(game, t.reprisalPlayerId!, reprisalTargets(game, t.excludedGiftId, t.reprisalPlayerId!)).map((id): Offer => ({ type: 'reprisal', giftId: id })),
      { type: 'skip_reprisal' },
    ];
    case 'endgame_choice': return [
      ...affordable(game, t.activePlayerId!, stealTargets(game, t.activePlayerId!)).map((id): Offer => ({ type: 'steal', giftId: id })),
      { type: 'path', path: 'auction' },
    ];
    case 'endgame_nobids': return Object.keys(game.players || {}).filter((id) => id !== t.activePlayerId).map((id) => ({ type: 'nobids_defender', playerId: id }));
    case 'duel_pending': return [{ type: 'start_duel' }];
    case 'yield_pending': return [{ type: 'start_yield_duel' }];
    case 'duel': {
      const step = duelGameStep(game);
      if (step === 'picker') return [{ type: 'host_spin' }];
      if (step || spinning(game, now) || duelAppWinner(t)) return [];
      const votes = voteStatus(game, now);
      const out: Offer[] = votes.accepting && votes.eligible.length ? [{ type: 'close_voting' }] : [];
      if (votes.blocking && !t.isTiebreak) return out;
      if (t.duelGame?.kind === 'app') return [...out, { type: 'start_duel_app' }];
      return [...out, { type: 'duel_winner', id: t.challengerId! }, { type: 'duel_winner', id: t.defenderId! }];
    }
    case 'minigame_party': return game.minigame && (game.minigame as PartyGame).type in GAMES ? [{ type: 'skip_party' }] : [];
    case 'endgame_bidding': return [{ type: 'reveal_auction' }, { type: 'skip_auction' }];
    case 'minigame_draw_off': {
      const mg = game.minigame as DrawOff | undefined;
      return mg?.type === 'draw_off' && mg.phase === 'results' && mg.winners?.length ? [{ type: 'complete_drawoff' }] : [];
    }
    case 'endgame_reveal': return rankedBids(t).length ? [{ type: 'auction_duel' }] : [];
    default: return [];
  }
}

/** Fields a person fills in (not compared with what's offered). */
export const HAND_FILLED = ['score', 'final', 'drawing', 'giftName', 'seed', 'name'];

/** Is this offer one of `who`'s right now? (Fields a player fills in aren't compared.) */
export function isOffered(game: Game, who: Who, offer: Offer, now: number): boolean {
  if (offer.type === 'bid' && who !== HOST && changesBid(game, who, offer.amount)) return true;
  const fixed = (o: Offer) => JSON.stringify(Object.entries(o).filter(([k]) => !HAND_FILLED.includes(k)).sort());
  return offers(game, who, now).some((o) => fixed(o) === fixed(offer));
}

/** A sealed bid can be changed until the reveal (the phone's Change Bid; offers() lists only first bids). */
function changesBid(game: Game, me: PlayerId, amount: number): boolean {
  const t = game.currentTurn;
  const p = game.players?.[me];
  if (!t || !p || !t.bids?.[me] || (t.phase !== 'bidding' && t.phase !== 'endgame_bidding')) return false;
  const owner = t.phase === 'bidding' ? t.openerId : t.activePlayerId;
  return me !== owner && Number.isInteger(amount) && amount >= 0 && amount <= Math.min(p.chips, config(game.settings).maxBid);
}

/** The plan that pressing `offer` produces (the same action the button's click runs). */
export function perform(game: Game, who: Who, offer: Offer, ctx: { now: number; rng: Rng; giftName?: string }): Plan {
  if (!isOffered(game, who, offer, ctx.now)) return refuse(`${offer.type} isn't on ${who === HOST ? 'the host' : 'their'} screen now`);
  const t = game.currentTurn!;
  switch (offer.type) {
    case 'bid': return A.submitBid(game, who, offer.amount, ctx.now);
    case 'keep': return A.keepGift(game);
    case 'gambit': return A.gambit(game, offer.giftId);
    case 'fight': case 'yield': {
      const fight = offer.type === 'fight';
      if (who === HOST) return hostChoice(game, fight, ctx);
      if (t.phase === 'yield_choice') return fight ? A.openerFight(game, t.selectedC1!, t.selectedC2!, true, ctx.rng) : A.openerYield(game, t.selectedC1!, t.selectedC2!, ctx.rng);
      const tied = topTiedBidders(t);
      const [[c1], [c2]] = rankedBids(t).filter(([, a]) => a > 0);
      if (tied.length >= 3) return fight ? A.openerFight(game, null, null, true, ctx.rng) : A.openerYield(game, null, null, ctx.rng);
      return fight ? A.openerFight(game, c1, c2, tied.length === 2, ctx.rng) : A.openerYield(game, c1, c2, ctx.rng);
    }
    case 'pick_duel': return chooseDuelGame(game, who, offer.index);
    case 'spin': return spinDuelGame(game, ctx, who);
    case 'call_duel': return A.voteOnDuel(game, who, offer.pick, ctx.now);
    case 'reprisal': return D.chooseReprisal(game, offer.giftId, ctx.now);   // the phone's, or the host's for them
    case 'skip_reprisal': return D.skipReprisal(game);
    case 'path': return offer.path === 'steal' ? D.chooseSteal(game) : D.startAuction(game);
    case 'steal': return D.executeSteal(game, offer.giftId);
    case 'cancel_steal': return D.cancelSteal(game);
    case 'nobids_defender': return D.noBidsDuel(game, offer.playerId);
    case 'whack': return reportWhack(game, who, Math.round(Number(offer.score) || 0), !!offer.final);
    case 'sketch': return submitSketch(game, who, offer.drawing || 'data:,');
    case 'vote_sketch': return voteSketch(game, who, offer.pick);
    case 'start': return A.startGame(game);
    case 'draw': return A.drawGift(game, offer.giftName || ctx.giftName || 'A gift', `gift_${ctx.now}`, offer.seed == null ? ctx : { ...ctx, rng: seeded(offer.seed) });
    case 'reveal': return A.revealBids(game);
    case 'skip_bidding': return A.skipBidding(game);
    case 'start_duel': {
      const c = t.phase === 'duel_pending' ? t.challengerId! : rankedBids(t).filter(([, a]) => a > 0)[0][0];
      return A.startDuel(game, c, t.openerId!);
    }
    case 'start_yield_duel': return A.startYieldDuel(game, t.yieldC1!, t.yieldC2!, false, ctx);
    case 'host_spin': return spinDuelGame(game, ctx);
    case 'close_voting': return closeVoting(game);
    case 'start_duel_app': return startDuelApp(game, ctx);
    case 'duel_winner': return t.isTiebreak ? A.recordTiebreakWinner(game, offer.id) : D.resolveDuel(game, offer.id, ctx.now);
    case 'skip_party': return advanceParty(game, (game.minigame as PartyGame).phase, ctx.now);
    case 'endgame': return D.startEndgame(game);
    case 'reveal_auction': return D.revealAuction(game);
    case 'skip_auction': return D.skipAuction(game);
    case 'complete_drawoff': return DO.completeDrawOff(game);
    case 'auction_duel': return D.auctionDuel(game, rankedBids(t)[0][0]);
    case 'end': return A.endGame(game);
  }
}

/** The host's override of the Opener's fight-or-yield (the console's buttons, not the phone's). */
function hostChoice(game: Game, fight: boolean, ctx: { now: number; rng: Rng }): Plan {
  const t = game.currentTurn!;
  if (t.phase === 'yield_choice') return fight ? A.startTiebreaker(game, t.selectedC1!, t.selectedC2!, ctx) : A.startYieldDuel(game, t.selectedC1!, t.selectedC2!, true, ctx);
  const tied = topTiedBidders(t);
  if (tied.length >= 2) return fight ? A.hostTieFight(game, ctx) : tied.length === 2 ? A.startYieldDuel(game, tied[0], tied[1], false, ctx) : A.hostTieYield(game, ctx);
  const [[c1], [c2]] = rankedBids(t).filter(([, a]) => a > 0);
  return fight ? A.startDuel(game, c1, t.openerId!) : A.startYieldDuel(game, c1, c2, false, ctx);
}
