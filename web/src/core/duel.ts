// Settling duels, Reindeer Reprisals, and the Misfit Lottery (endgame).
import { config, minCost } from './settings';
import type { Rng } from './rng';
import { pick } from './rng';
import { dividendFor, giftless, misfitPile, naughtyAfterDuel, potAndTax, reprisalAllowed, reprisalTargets, stealTargets } from './rules';
import { betLine, settleBets } from './scoring';
import { refuse, SERVER_TIME, Tx } from './tx';
import type { Game, Gift, Plan, Player, PlayerId, Turn } from './types';
import { MISFIT } from './types';
import { voteStatus } from './voting';

const name = (game: Game, id: PlayerId | undefined | null) => (id && game.players?.[id]?.name) || 'Someone';

/** Send a gift to the Misfit pile (Naughty Level resets; FIFO order by time). */
function toMisfitPile(tx: Tx, giftId: string, now: number) {
  tx.set(`gifts/${giftId}/ownerId`, MISFIT).set(`gifts/${giftId}/naughtyLevel`, 0).set(`gifts/${giftId}/misfitOrder`, now);
}

function giveGift(tx: Tx, playerId: PlayerId, giftId: string | null) {
  tx.set(`players/${playerId}/giftId`, giftId);
  if (giftId) tx.set(`gifts/${giftId}/ownerId`, playerId);
}

/** The host clicked "X wins": settle chips, the gift, stats and votes, then offer a Reprisal. */
export function resolveDuel(game: Game, winnerId: PlayerId, now: number): Plan {
  const t = game.currentTurn;
  if (!t || t.phase !== 'duel') return refuse('No duel to resolve');
  if (!t.challengerId || !t.defenderId) return refuse('Invalid duel state');
  if (winnerId !== t.challengerId && winnerId !== t.defenderId) return refuse('Winner must be one of the duellists');
  const loserId = winnerId === t.challengerId ? t.defenderId : t.challengerId;
  const players = game.players || {};
  if (!players[winnerId] || !players[loserId]) return refuse('Player not found');
  const votes = voteStatus(game, now);
  if (votes.blocking && votes.endsAt != null) return refuse(`Waiting for ${votes.waiting.length} vote${votes.waiting.length > 1 ? 's' : ''}`);

  const cfg = config(game.settings);
  const tx = new Tx(game).claim('currentTurn/resolvedBy', winnerId)
    .set('undo', { players: game.players || {}, gifts: game.gifts || {} });
  // fighting a duel counts towards the Party Score, win or lose (duelists can't call their own duel)
  tx.add(`players/${winnerId}/duelsFought`, 1).add(`players/${loserId}/duelsFought`, 1);
  const pot = t.pot || 0;
  if (pot > 0) tx.add(`players/${winnerId}/chips`, pot).event(`${name(game, winnerId)} wins pot of ${pot} chips!`, 'bid');

  const dividend = dividendFor(cfg, t, loserId);
  if (dividend) tx.add(`players/${loserId}/chips`, dividend).event(`${name(game, loserId)} receives +1 Loser's Dividend`, 'bid');
  else if (!t.isYieldDuel && !t.isMisfitDuel && !t.isTiebreak && cfg.loserDividendRule === 'defenders') {
    tx.event(`${name(game, loserId)} attacked and lost - no dividend!`, 'bid');
  }

  // Published so phones show the right verdict
  const loserBid = t.bids?.[loserId]?.amount || 0;
  tx.set('lastDuel', {
    winnerId, loserId, pot, dividend,
    loserPaid: t.isYieldDuel ? Math.floor(loserBid / 2) : loserId === t.challengerId ? loserBid : 0,
    at: SERVER_TIME,
  });

  const gift = t.giftId ? game.gifts?.[t.giftId] : undefined;
  if (gift && t.giftId) {
    const level = naughtyAfterDuel(gift, t);
    if ((gift.naughtyLevel || 0) === 0 && t.bids && level > 0) tx.event(`🔥 Gift starts at Naughty ${level} (highest bid)`, 'bid');
    tx.set(`gifts/${t.giftId}/naughtyLevel`, level);
  }

  let redraw = false; // giftless auction winner: the active player goes back in the draw
  if (t.isNoBidsDuel && t.misfitId) {
    // The active player ALWAYS gets the Misfit; losing costs a 1-chip toll to the defender
    const active = t.challengerId;
    if (game.gifts?.[t.misfitId]) {
      giveGift(tx, active, t.misfitId);
      if (loserId === active) {
        if (tx.get<number>(`players/${active}/chips`) >= 1) {
          tx.add(`players/${active}/chips`, -1).add(`players/${t.defenderId}/chips`, 1)
            .event(`${name(game, active)} pays 1 chip Misfit Toll to ${name(game, t.defenderId)}`, 'bid');
        }
        tx.event(`${name(game, active)} takes misfit "${t.misfitName}" (lost duel)`, 'gift');
      } else {
        tx.event(`${name(game, active)} claims misfit "${t.misfitName}" for FREE!`, 'gift');
      }
    }
  } else if (t.isEndgameSteal && t.misfitId) {
    const active = t.challengerId;
    const hasMisfit = !!game.gifts?.[t.misfitId];
    if (winnerId === active) {
      giveGift(tx, active, t.giftId!); // takes the defender's gift…
      if (hasMisfit) { giveGift(tx, loserId, t.misfitId); tx.event(`${name(game, loserId)} takes misfit "${t.misfitName}"`, 'gift'); }
    } else if (hasMisfit) {
      giveGift(tx, active, t.misfitId);
      tx.event(`${name(game, active)} takes misfit "${t.misfitName}"`, 'gift');
    }
  } else if (t.isEndgameAuction && t.misfitId) {
    const active = t.challengerId;
    const bidder = t.defenderId;
    if (winnerId === active) {
      giveGift(tx, active, t.misfitId);
      tx.event(`${name(game, active)} wins the misfit!`, 'gift');
    } else {
      giveGift(tx, bidder, t.misfitId);
      if (t.bidderHadGift && t.bidderOldGift) {
        giveGift(tx, active, t.bidderOldGift);
        tx.event(`${name(game, active)} gets ${name(game, bidder)}'s old gift`, 'gift');
      } else {
        tx.event(`${name(game, bidder)} (was giftless) wins the misfit. ${name(game, active)} must try again!`, 'gift');
        redraw = true;
      }
    }
  } else if (t.giftId) {
    // Standard swap: winner takes the gift; their old gift goes to the loser; if the loser
    // still had their own gift too, the loser's ORIGINAL goes to the Misfit pile.
    const winnerOld = players[winnerId].giftId || null;
    const loserOld = players[loserId].giftId || null;
    giveGift(tx, winnerId, t.giftId);
    if (winnerOld && winnerOld !== t.giftId) {
      if (loserOld && loserOld !== t.giftId) {
        toMisfitPile(tx, loserOld, now);
        tx.event(`${game.gifts?.[loserOld]?.name} moved to Misfit Pile!`, 'gift');
      }
      giveGift(tx, loserId, winnerOld);
    } else if (loserOld === t.giftId) {
      tx.set(`players/${loserId}/giftId`, null).event(`${name(game, loserId)} is now giftless!`, 'gift');
    }
    if (t.isYieldDuel && t.yieldingOpener && tx.get(`players/${t.yieldingOpener}/giftId`) === t.giftId) {
      tx.set(`players/${t.yieldingOpener}/giftId`, null).event(`${name(game, t.yieldingOpener)} yielded and is now giftless!`, 'gift');
    }
  }

  tx.event(`🏆 ${name(game, winnerId)} wins the duel!`, 'duel');
  if (!t.isTiebreak) tx.add(`players/${winnerId}/duelWins`, 1).add(`players/${loserId}/duelLosses`, 1);

  settleVotes(tx, t, winnerId, cfg.spectatorBets > 0);

  if (redraw) { tx.set('currentTurn', null); return tx.plan('drawEndgamePlayer'); }

  if (reprisalAllowed(cfg, t)) {
    const after = tx.get<Game>('') as Game;
    const targets = reprisalTargets(after, t.giftId, loserId);
    const chips = tx.get<number>(`players/${loserId}/chips`) || 0;
    const affordable = targets.filter(([, g]) => chips >= minCost(cfg, g.naughtyLevel || 0));
    if (!targets.length) tx.event('No reprisal targets available - skipping', 'duel').set('currentTurn', null);
    else if (!affordable.length) tx.event(`${name(game, loserId)} can't afford any targets (${chips} chips) - skipping reprisal`, 'duel').set('currentTurn', null);
    else {
      tx.set('currentTurn', { phase: 'reprisal', reprisalPlayerId: loserId, excludedGiftId: t.giftId, reprisalDepth: (t.reprisalDepth || 0) + 1 })
        .event(`🦌 ${name(game, loserId)} can use Reindeer Reprisal!`, 'duel');
    }
    return tx.plan();
  }
  if ((t.reprisalDepth || 0) >= cfg.maxReprisalDepth) tx.event('Max reprisal chain reached!', 'duel');
  else if (t.isGambit) tx.event("No reprisal for Gambit loss - that's the risk!", 'duel');
  tx.set('currentTurn', null);
  return tx.plan(game.phase === 'endgame' ? 'checkEndgameContinue' : undefined);
}

/** Voting record for Party Score + (optionally) spectator bets. */
function settleVotes(tx: Tx, t: Turn, winnerId: PlayerId, bets: boolean) {
  const votes = t.duelVotes || {};
  for (const [voterId, vote] of Object.entries(votes)) {
    if (!tx.get(`players/${voterId}`)) continue;
    const right = vote.votedFor === winnerId;
    tx.add(`players/${voterId}/${right ? 'votingWins' : 'votingLosses'}`, 1);
    if (right) tx.event(`✅ ${name(tx.game, voterId)} correctly predicted the winner!`, 'bid');
  }
  if (!bets || !t.challengerId || !t.defenderId) return;
  const players = tx.game.players || {};
  const carry = Number(tx.game.betPool) || 0;
  const r = settleBets(votes, winnerId, { a: t.challengerId, line: betLine(players[t.challengerId], players[t.defenderId]), carry, tax: t.tax || 0 });
  for (const [voterId, change] of Object.entries(r.delta)) {
    if (change && tx.get(`players/${voterId}`)) tx.set(`players/${voterId}/chips`, Math.max(0, tx.get<number>(`players/${voterId}/chips`) + change));
  }
  if (r.carry !== carry) tx.set('betPool', r.carry || null);
  tx.set('lastDuel/bets', { pool: r.pool, each: r.each, toBank: r.toBank, winners: r.winners, carry: r.carry, ...(r.returned ? { returned: true } : {}) });
  if (r.returned) tx.event('🎲 Spectator bets: nobody backed the winner, so every stake goes back', 'bid');
  else if (r.pool) tx.event(`🎲 Spectator bets: a pool of ${r.pool} pays ${r.each} for 1 to ${r.winners} right caller${r.winners > 1 ? 's' : ''}${r.carry ? ` (${r.carry} left for the next pool)` : ''}`, 'bid');
}

// --- Reindeer Reprisal ------------------------------------------------------------------------

/** Pay the Minimum Cost and challenge a gift — or claim a Misfit outright (no one to duel). */
export function chooseReprisal(game: Game, targetGiftId: string, now: number): Plan {
  const t = game.currentTurn;
  if (!t || t.phase !== 'reprisal' || !t.reprisalPlayerId) return refuse('No reprisal now');
  const pid = t.reprisalPlayerId;
  if (!reprisalTargets(game, t.excludedGiftId, pid).some(([id]) => id === targetGiftId)) return refuse("That gift can't be targeted");
  const target = game.gifts![targetGiftId];
  const cost = minCost(config(game.settings), target.naughtyLevel);
  const me = game.players![pid];
  if (me.chips < cost) return refuse(`Not enough chips! Need ${cost}, have ${me.chips}`);
  const tx = new Tx(game).add(`players/${pid}/chips`, -cost);
  if (target.ownerId !== MISFIT) {
    return tx.set('currentTurn', {
      phase: 'duel', challengerId: pid, defenderId: target.ownerId, giftId: targetGiftId,
      pot: 0, isReprisal: true, reprisalDepth: t.reprisalDepth || 1,
    }).event(`🦌 ${me.name} uses Reprisal on "${target.name}"!`, 'duel').plan();
  }
  if (me.giftId && me.giftId !== targetGiftId) {
    toMisfitPile(tx, me.giftId, now);
    tx.event(`${game.gifts?.[me.giftId]?.name} moved to Misfit Pile!`, 'gift');
  }
  giveGift(tx, pid, targetGiftId);
  return tx.set('currentTurn', null).event(`${me.name} claims Misfit "${target.name}"!`, 'gift').plan();
}

export const skipReprisal = (game: Game): Plan => new Tx(game).set('currentTurn', null).event('Reprisal skipped', 'duel').plan();

// --- the Misfit Lottery -------------------------------------------------------------------------

export function startEndgame(game: Game): Plan {
  return new Tx(game).set('phase', 'endgame').event('🎰 Endgame: Misfit Lottery begins!', 'gift').plan('drawEndgamePlayer');
}

/** A giftless player is drawn and faces the oldest Misfit — or the game ends. */
export function drawEndgamePlayer(game: Game, r: Rng): Plan {
  const bag = giftless(game);
  const tx = new Tx(game);
  if (!bag.length) return tx.set('phase', 'ended').event('🎄 Everyone has a gift! Game complete!', 'gift').plan();
  const pile = misfitPile(game);
  if (!pile.length) return tx.event('⚠️ No misfits left but players still giftless!', 'gift').plan();
  const [misfitId, misfit] = pile[0];
  const active = pick(bag, r);
  return tx.set('currentTurn', { phase: 'endgame_choice', activePlayerId: active, misfitId, misfitName: misfit.name })
    .event(`🎅 ${name(game, active)} drawn! Target: "${misfit.name}"`, 'gift').plan();
}

/** After an endgame duel: next giftless player, or the end. */
export function checkEndgameContinue(game: Game, r: Rng): Plan {
  if (giftless(game).length && !misfitPile(game).length) {
    return new Tx(game).event('⚠️ No more misfits! Some players remain giftless.', 'gift').plan();
  }
  return drawEndgamePlayer(game, r);
}

export const chooseSteal = (game: Game): Plan => new Tx(game).set('currentTurn/phase', 'endgame_steal_select').plan();
export const cancelSteal = (game: Game): Plan => new Tx(game).set('currentTurn/phase', 'endgame_choice').plan();

/** Path A: pay the Minimum Cost to challenge a player for their gift; the loser takes the Misfit. */
export function executeSteal(game: Game, targetGiftId: string): Plan {
  const t = game.currentTurn;
  if (!t?.activePlayerId || !t.phase.startsWith('endgame')) return refuse('No steal now');
  if (!stealTargets(game, t.activePlayerId).some(([id]) => id === targetGiftId)) return refuse("That gift can't be stolen");
  const target = game.gifts![targetGiftId];
  const cost = minCost(config(game.settings), target.naughtyLevel);
  const me = game.players![t.activePlayerId];
  if (me.chips < cost) return refuse(`Not enough chips! Need ${cost}, have ${me.chips}`);
  return new Tx(game).add(`players/${t.activePlayerId}/chips`, -cost)
    .set('currentTurn', {
      phase: 'duel', challengerId: t.activePlayerId, defenderId: target.ownerId, giftId: targetGiftId,
      misfitId: t.misfitId, misfitName: t.misfitName, pot: 0, isEndgameSteal: true, isMisfitDuel: true,
    })
    .event(`🎯 ${me.name} challenges ${name(game, target.ownerId)} for "${target.name}"!`, 'duel').plan();
}

/** Path B: everyone (but the active player) bids on the Misfit. */
export function startAuction(game: Game): Plan {
  return new Tx(game).set('currentTurn/phase', 'endgame_bidding').set('currentTurn/bids', null)
    .event(`🎰 Auction for "${game.currentTurn?.misfitName}" - Everyone can bid!`, 'bid').plan();
}

export function skipAuction(game: Game): Plan {
  return new Tx(game).set('currentTurn/phase', 'endgame_nobids').set('currentTurn/bids', null)
    .event('No bids - Active player chooses defender', 'bid').plan();
}

/** Everyone passed → the active player picks a defender on their phone. */
export function revealAuction(game: Game): Plan {
  const any = Object.values(game.currentTurn?.bids || {}).some((b) => b.amount > 0);
  if (!any) return skipAuction(game);
  return new Tx(game).set('currentTurn/phase', 'endgame_reveal').event('Bids revealed!', 'bid').plan();
}

/** The highest bidder pays in and duels the active player for the Misfit. */
export function auctionDuel(game: Game, bidderId: PlayerId): Plan {
  const t = game.currentTurn;
  const bid = t?.bids?.[bidderId]?.amount || 0;
  if (!t?.activePlayerId || bid <= 0) return refuse('Invalid bid - cannot start duel with 0 chips');
  const bidder: Player = game.players![bidderId];
  const { pot, tax } = potAndTax(config(game.settings), bid);
  const tx = new Tx(game).claim('currentTurn/duelStartedBy', bidderId).add(`players/${bidderId}/chips`, -bid);
  if (tax > 0) tx.event(`Santa Tax: ${tax} chip${tax > 1 ? 's' : ''} to the Bank`, 'bid');
  return tx.set('currentTurn', {
    phase: 'duel', challengerId: t.activePlayerId, defenderId: bidderId, giftId: t.misfitId,
    misfitId: t.misfitId, misfitName: t.misfitName, pot, tax, isEndgameAuction: true, isMisfitDuel: true,
    bidderHadGift: !!bidder.giftId, bidderOldGift: bidder.giftId || null,
  }).event(`⚔️ ${name(game, t.activePlayerId)} vs ${bidder.name} for "${t.misfitName}"!`, 'duel').plan();
}

/** No bids on the Misfit: duel a chosen defender — win it free, or pay them a 1-chip toll. */
export function noBidsDuel(game: Game, defenderId: PlayerId): Plan {
  const t = game.currentTurn;
  if (!t?.activePlayerId || defenderId === t.activePlayerId || !game.players?.[defenderId]) return refuse('Pick another player');
  return new Tx(game).set('currentTurn', {
    phase: 'duel', challengerId: t.activePlayerId, defenderId, giftId: t.misfitId,
    misfitId: t.misfitId, misfitName: t.misfitName, pot: 0, isNoBidsDuel: true, isMisfitDuel: true,
  }).event(`⚔️ ${name(game, t.activePlayerId)} vs ${name(game, defenderId)} for the misfit!`, 'duel').plan();
}

export type { Gift };
