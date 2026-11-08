// Every game action as a pure function: (game, …) → Plan (one atomic write + log lines).
// The room server runs these for every screen (core/intents.ts dispatch) and applies the plan.
// Ported rule-for-rule from the original host.html / player.html.
import { avatarOf, freeAvatar, isAvatar } from './avatars';
import { config, minCost, partyGameFor, variableStartingChips } from './settings';
import type { Config } from './settings';
import type { Rng } from './rng';
import { pick, shuffle } from './rng';
import { gambitTargets, potAndTax, rankedBids, topTiedBidders, yieldPayment } from './rules';
import { refuse, Tx } from './tx';
import type { Game, Plan, Player, PlayerId, Settings, Turn } from './types';
import { startPartyGame } from './partyflow';
import { voteStatus } from './voting';
import { duelGameStep } from './duelgame';

export interface Ctx { now: number; rng: Rng }

const cfgOf = (game: Game): Config => config(game.settings);
const name = (game: Game, id: PlayerId | undefined) => (id && game.players?.[id]?.name) || 'Someone';
const turnOf = (tx: Tx) => tx.get<Turn | null>('currentTurn');

// --- lobby -------------------------------------------------------------------------------

export function newGame(opts: { hostId: string; hostUid: string | null; settings: Settings; now: number }): Game {
  return {
    hostId: opts.hostId,
    ...(opts.hostUid ? { hostUid: opts.hostUid } : {}),
    settings: opts.settings,
    phase: 'lobby',
    createdAt: opts.now,
  };
}

/** Names are rendered in many places: strip tag brackets and cap the length (rules allow 60). */
export const cleanName = (raw: string) => raw.trim().replace(/[<>]/g, '').slice(0, 60);

/** The most people one game seats (a big office party; it also bounds a room's size on the server). */
export const MAX_PLAYERS = 50;

export function joinPlayer(game: Game, p: { playerId: PlayerId; name: string; uid: string | null; now: number; age?: number | null; avatar?: string | null }): Plan {
  const nm = cleanName(p.name);
  if (!nm) return refuse('Please enter your name');
  if (Object.keys(game.players || {}).length >= MAX_PLAYERS) return refuse(`This game is full (${MAX_PLAYERS} players).`);
  const cfg = cfgOf(game);
  const player: Player = {
    name: nm,
    // Variable chips are settled at Start (the table size isn't known yet)
    chips: cfg.startingChips ?? variableStartingChips(Object.keys(game.players || {}).length + 1),
    connected: true,
    joinedAt: p.now,
    ...(p.uid ? { ownerUid: p.uid } : {}),
    // kept only for under-16s: it decides who picks the duel (the Under-16 Exception)
    ...(typeof p.age === 'number' && p.age >= 1 && p.age < 16 ? { age: Math.floor(p.age) } : {}),
    // their pick, else the one their name gives (or the next one nobody at the table has)
    avatar: isAvatar(p.avatar) ? p.avatar : freeAvatar(nm, Object.values(game.players || {}).map((q) => avatarOf(q.name, q.avatar).id)),
  };
  return new Tx(game).set(`players/${p.playerId}`, player).plan();
}

export function startGame(game: Game): Plan {
  const ids = Object.keys(game.players || {});
  if (ids.length < 2) return refuse('Need at least 2 players to start!');
  const tx = new Tx(game).set('phase', 'main');
  // Rules: 10 chips for up to 10 players, 12 for larger groups — for EVERYONE (joiners
  // used to get 10 or 12 depending on how many had joined before them).
  const cfg = cfgOf(game);
  if (cfg.startingChips === null) {
    const chips = variableStartingChips(ids.length);
    for (const id of ids) tx.set(`players/${id}/chips`, chips);
  }
  return tx.event('🎄 Game started! Let the chaos begin!', 'gift').plan();
}

export const endGame = (game: Game): Plan => new Tx(game).set('phase', 'ended').plan();

// --- opening a gift + sealed bids ----------------------------------------------------------

export function drawGift(game: Game, giftName: string, giftId: string, ctx: Ctx): Plan {
  const bag = Object.entries(game.players || {}).filter(([, p]) => !p.giftId).map(([id]) => id);
  if (!bag.length) return refuse('No players left in the bag!');
  if (game.currentTurn) return refuse('Finish the current turn first');
  const nm = giftName.trim();
  if (!nm) return refuse('Name the gift');
  const opener = pick(bag, ctx.rng);
  return new Tx(game)
    .set(`gifts/${giftId}`, { name: nm, naughtyLevel: 0, ownerId: opener, createdAt: ctx.now })
    .set(`players/${opener}/giftId`, giftId)
    .set('currentTurn', { phase: 'bidding', activePlayerId: opener, openerId: opener, giftId })
    .event(`🎅 ${name(game, opener)} drawn from the bag!`, 'gift')
    .event(`📦 Opened "${nm}" - Bidding is open!`, 'gift')
    .plan();
}

/** A sealed bid (main game or Misfit auction); 0 = pass. Re-submitting replaces it. */
export function submitBid(game: Game, playerId: PlayerId, amount: number, now: number): Plan {
  const t = game.currentTurn;
  if (!t || (t.phase !== 'bidding' && t.phase !== 'endgame_bidding')) return refuse('Bidding is closed');
  if (playerId === (t.phase === 'bidding' ? t.openerId : t.activePlayerId)) return refuse("You can't bid on your own turn");
  const p = game.players?.[playerId];
  if (!p) return refuse('Not in this game');
  const max = Math.min(p.chips, cfgOf(game).maxBid);
  if (!Number.isInteger(amount) || amount < 0 || amount > max) return refuse(`Bid 0–${max}`);
  return new Tx(game).set(`currentTurn/bids/${playerId}`, { amount, submittedAt: now }).plan();
}

export const revealBids = (game: Game): Plan =>
  new Tx(game).set('currentTurn/phase', 'reveal').event('Bids revealed!', 'bid').plan();

export const skipBidding = (game: Game): Plan =>
  new Tx(game).set('currentTurn/phase', 'reveal').set('currentTurn/bids', null).plan();

/** Opener, nobody bid: keep the gift. */
export function keepGift(game: Game): Plan {
  const t = game.currentTurn;
  return new Tx(game).set('currentTurn', null).event(`${name(game, t?.openerId)} keeps their gift!`, 'gift').plan();
}

/** Opener, nobody bid: Grinch's Gambit — pay the Minimum Cost and duel for another gift. */
export function gambit(game: Game, targetGiftId: string): Plan {
  const t = game.currentTurn;
  if (!t?.openerId || t.phase !== 'reveal') return refuse('No Gambit now');
  const legal = gambitTargets(game, t.giftId, t.openerId).some(([id]) => id === targetGiftId);
  const target = game.gifts?.[targetGiftId];
  if (!legal || !target) return refuse("That gift can't be Gambited");
  const cost = minCost(cfgOf(game), target.naughtyLevel);
  const opener = game.players![t.openerId];
  if (opener.chips < cost) return refuse(`Not enough chips! Need ${cost}, have ${opener.chips}`);
  return new Tx(game)
    .add(`players/${t.openerId}/chips`, -cost)
    .set('currentTurn', {
      phase: 'duel', challengerId: t.openerId, defenderId: target.ownerId, giftId: targetGiftId,
      originalGiftId: t.giftId, pot: 0, isGambit: true,
    })
    .event(`😈 ${opener.name} uses Grinch's Gambit on "${target.name}"!`, 'duel')
    .plan();
}

// --- the Opener's choice when challenged (phone) --------------------------------------------

/** Two of the tied bidders at random (3+ way tie). */
export function randomTiedPair(game: Game, r: Rng): [PlayerId, PlayerId] {
  const [a, b] = shuffle(topTiedBidders(game.currentTurn), r);
  return [a, b];
}

/** Fight: duel the top bidder — or, on a tie, let the tied bidders settle who does. */
export function openerFight(game: Game, c1: PlayerId | null, c2: PlayerId | null, isTie: boolean, r: Rng): Plan {
  if (isTie && !c1) [c1, c2] = randomTiedPair(game, r);
  const tx = new Tx(game).set('currentTurn/openerChoice', 'fight');
  if (isTie) tx.set('currentTurn/phase', 'tiebreaker_pending').set('currentTurn/tiebreakerC1', c1).set('currentTurn/tiebreakerC2', c2);
  else tx.set('currentTurn/phase', 'duel_pending').set('currentTurn/challengerId', c1);
  return tx.plan();
}

/** Yield: step aside; two challengers duel each other (both pay half). */
export function openerYield(game: Game, c1: PlayerId | null, c2: PlayerId | null, r: Rng): Plan {
  if (!c1) [c1, c2] = randomTiedPair(game, r);
  return new Tx(game)
    .set('currentTurn/openerChoice', 'yield').set('currentTurn/phase', 'yield_pending')
    .set('currentTurn/yieldC1', c1).set('currentTurn/yieldC2', c2)
    .plan();
}

// --- host: starting duels -------------------------------------------------------------------

function startDuelTx(tx: Tx, challengerId: PlayerId, defenderId: PlayerId) {
  tx.claim('currentTurn/duelStartedBy', challengerId);
  const t = turnOf(tx)!;
  const bid = t.bids?.[challengerId]?.amount || 0;
  tx.add(`players/${challengerId}/chips`, -bid);
  const { pot, tax } = potAndTax(cfgOf(tx.game), bid);
  if (tax > 0) tx.event(`Santa Tax: ${tax} chip${tax > 1 ? 's' : ''} to the Bank`, 'bid');
  tx.set('currentTurn/phase', 'duel').set('currentTurn/challengerId', challengerId).set('currentTurn/defenderId', defenderId)
    .set('currentTurn/pot', pot).set('currentTurn/tax', tax).set('currentTurn/isYieldDuel', false).set('currentTurn/isTiebreak', false)
    .event(`⚔️ ${name(tx.game, challengerId)} vs ${name(tx.game, defenderId)} - Duel!`, 'duel');
}

/** Sealed-bid duel: the challenger pays their full bid into the pot. */
export function startDuel(game: Game, challengerId: PlayerId, defenderId: PlayerId): Plan {
  if (!game.currentTurn) return refuse('No current turn');
  const tx = new Tx(game);
  startDuelTx(tx, challengerId, defenderId);
  return tx.plan();
}

function startYieldDuelTx(tx: Tx, c1: PlayerId, c2: PlayerId) {
  tx.claim('currentTurn/duelStartedBy', c1);
  const t = turnOf(tx)!;
  const pay1 = yieldPayment(t.bids?.[c1]?.amount || 0);
  const pay2 = yieldPayment(t.bids?.[c2]?.amount || 0);
  tx.add(`players/${c1}/chips`, -pay1).add(`players/${c2}/chips`, -pay2)
    .event(`${name(tx.game, c1)} pays ${pay1}, ${name(tx.game, c2)} pays ${pay2} (half bids, rounded down)`, 'bid');
  const { pot, tax } = potAndTax(cfgOf(tx.game), pay1 + pay2);
  if (tax > 0) tx.event(`Santa Tax: ${tax} chip${tax > 1 ? 's' : ''} to the Bank`, 'bid');
  tx.set('currentTurn/phase', 'duel').set('currentTurn/challengerId', c1).set('currentTurn/defenderId', c2)
    .set('currentTurn/pot', pot).set('currentTurn/tax', tax).set('currentTurn/isYieldDuel', true)
    .set('currentTurn/yieldingOpener', t.openerId)
    .event(`🏃 ${name(tx.game, t.openerId)} yields! ${name(tx.game, c1)} vs ${name(tx.game, c2)}`, 'duel');
}

/**
 * Yield duel. `challengersChosen` = a mini-game or random draw already picked the two —
 * never route back into a mini-game (that used to loop forever).
 */
export function startYieldDuel(game: Game, c1: PlayerId, c2: PlayerId, challengersChosen: boolean, ctx: Ctx): Plan {
  const t = game.currentTurn;
  if (!t) return refuse('No current turn');
  const tied = topTiedBidders(t);
  const tx = new Tx(game);
  if (!challengersChosen && tied.length >= 3) {
    startMinigameTiebreakTx(tx, 'pick_two', tied, 'yield', ctx);
  } else {
    startYieldDuelTx(tx, c1, c2);
  }
  return tx.plan();
}

/** A tied top bid, Opener fights: the whole table plays a party game; the best-placed tied
 *  bidder then duels the Opener. (p1/p2 only matter when the tie wasn't recorded on the turn.) */
export function startTiebreaker(game: Game, p1: PlayerId, p2: PlayerId, ctx: Ctx): Plan {
  const t = game.currentTurn;
  if (!t) return refuse('No current turn');
  const tied = t.tiedBidders && t.tiedBidders.length >= 2 ? t.tiedBidders
    : topTiedBidders(t).length >= 2 ? topTiedBidders(t) : [p1, p2];
  const tx = new Tx(game);
  startMinigameTiebreakTx(tx, 'pick_one', tied, 'fight', ctx);
  return tx.plan();
}

/** Winner of a tie-break duel (games saved before ties became party games) → duels the Opener. */
export function recordTiebreakWinner(game: Game, winnerId: PlayerId): Plan {
  const t = game.currentTurn;
  if (!t?.openerId) return refuse('No opener');
  const tx = new Tx(game).event(`${name(game, winnerId)} wins the tiebreaker!`, 'duel');
  startDuelTx(tx, winnerId, t.openerId);
  return tx.plan();
}

/** Host override for a tied top bid: Opener fights (the party game picks 1). */
export function hostTieFight(game: Game, ctx: Ctx): Plan {
  const tied = topTiedBidders(game.currentTurn);
  if (tied.length < 2) return refuse('Not enough tied bidders');
  const tx = new Tx(game);
  startMinigameTiebreakTx(tx, 'pick_one', tied, 'fight', ctx);
  return tx.plan();
}

/** Host override for a 3+ way tie: Opener yields (the party game picks 2). */
export function hostTieYield(game: Game, ctx: Ctx): Plan {
  const tied = topTiedBidders(game.currentTurn);
  if (tied.length < 3) return refuse('Need 3+ tied bidders for this scenario');
  const tx = new Tx(game);
  startMinigameTiebreakTx(tx, 'pick_two', tied, 'yield', ctx);
  return tx.plan();
}

/** Random pick of two from a 3+ way tie, then the Opener chooses fight or yield. */
export function randomTiebreaker(game: Game, r: Rng): Plan {
  const [c1, c2] = randomTiedPair(game, r);
  if (!c2) return refuse('Not enough tied bidders');
  return new Tx(game).set('currentTurn/phase', 'yield_choice').set('currentTurn/selectedC1', c1).set('currentTurn/selectedC2', c2).plan();
}

/** Hand a tie to a whole-table party game (the house's pick, or one at random per tie). */
function startMinigameTiebreakTx(tx: Tx, goal: 'pick_one' | 'pick_two', tied: PlayerId[], choice: 'fight' | 'yield', ctx: Ctx) {
  tx.set('currentTurn/openerChoice', choice).set('currentTurn/minigameGoal', goal).set('currentTurn/tiedBidders', tied);
  startPartyGame(tx, partyGameFor(cfgOf(tx.game).tiebreakerMethod, ctx.rng), goal, tied, ctx);
}

/** Spectator vote on someone else's duel (stakes 1 chip when spectator bets are on). */
export function voteOnDuel(game: Game, voterId: PlayerId, votedFor: PlayerId, now: number): Plan {
  const t = game.currentTurn;
  if (!t || t.phase !== 'duel' || t.isTiebreak) return refuse('Cannot vote on this duel');
  if (votedFor !== t.challengerId && votedFor !== t.defenderId) return refuse('Invalid vote');
  if (voterId === t.challengerId || voterId === t.defenderId) return refuse("You can't vote on your own duel");
  if (t.duelVotes?.[voterId]) return refuse('You already voted!');
  if (duelGameStep(game)) return refuse('Wait until the duel is picked');
  if (!voteStatus(game, now).accepting) return refuse('Voting has closed');
  const chips = game.players?.[voterId]?.chips || 0;
  const stake = cfgOf(game).spectatorBets && chips >= 1 ? 1 : 0;
  return new Tx(game).set(`currentTurn/duelVotes/${voterId}`, { playerId: voterId, votedFor, stake, timestamp: now }).plan();
}

export { rankedBids };
