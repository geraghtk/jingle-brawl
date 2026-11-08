// Host admin: fix-ups when a phone dies or something goes wrong. Every edit keeps
// player.giftId and gift.ownerId consistent (a Misfit always has ownerId 'misfit').
import { refuse, Tx } from './tx';
import type { Game, Plan, PlayerId } from './types';
import { MISFIT } from './types';

const name = (game: Game, id: PlayerId | null | undefined) => (id && game.players?.[id]?.name) || 'Unknown';

function toMisfit(tx: Tx, giftId: string, now: number, naughtyLevel = 0) {
  tx.set(`gifts/${giftId}/ownerId`, MISFIT).set(`gifts/${giftId}/misfitOrder`, now).set(`gifts/${giftId}/naughtyLevel`, naughtyLevel);
}

export function setChips(game: Game, playerId: PlayerId, chips: number): Plan {
  if (!game.players?.[playerId]) return refuse('No such player');
  const n = Math.max(0, Math.min(9999, Math.floor(chips) || 0));
  return new Tx(game).set(`players/${playerId}/chips`, n).event(`⚙️ Admin: Set ${name(game, playerId)}'s chips to ${n}`, 'gift').plan();
}

export function removePlayerGift(game: Game, playerId: PlayerId, now: number): Plan {
  const giftId = game.players?.[playerId]?.giftId;
  if (!giftId) return refuse('Player has no gift');
  const tx = new Tx(game).set(`players/${playerId}/giftId`, null);
  toMisfit(tx, giftId, now);
  return tx.event(`⚙️ Admin: Moved ${name(game, playerId)}'s gift to Misfit pile`, 'gift').plan();
}

/** Rename / re-level / re-own a gift. newOwnerId null = Misfit pile. */
export function saveGift(game: Game, giftId: string, edit: { name?: string; naughtyLevel: number; ownerId: PlayerId | null }, now: number): Plan {
  const gift = game.gifts?.[giftId];
  if (!gift) return refuse('No such gift');
  const nm = edit.name?.trim() || gift.name;
  const tx = new Tx(game).set(`gifts/${giftId}/name`, nm).set(`gifts/${giftId}/naughtyLevel`, edit.naughtyLevel);
  const old = gift.ownerId;
  if (edit.ownerId) tx.set(`gifts/${giftId}/ownerId`, edit.ownerId);
  else if (old !== MISFIT) toMisfit(tx, giftId, now, edit.naughtyLevel);
  if (old && old !== MISFIT && old !== edit.ownerId) tx.set(`players/${old}/giftId`, null);
  if (edit.ownerId) {
    const theirs = game.players?.[edit.ownerId]?.giftId;
    if (theirs && theirs !== giftId) toMisfit(tx, theirs, now); // one gift each
    tx.set(`players/${edit.ownerId}/giftId`, giftId);
  }
  return tx.event(`⚙️ Admin: Updated "${nm}" (😈${edit.naughtyLevel})`, 'gift').plan();
}

export function moveGiftToMisfit(game: Game, giftId: string, now: number): Plan {
  const gift = game.gifts?.[giftId];
  if (!gift) return refuse('No such gift');
  const tx = new Tx(game);
  toMisfit(tx, giftId, now);
  if (gift.ownerId !== MISFIT) tx.set(`players/${gift.ownerId}/giftId`, null);
  return tx.event(`⚙️ Admin: Moved "${gift.name}" to Misfit pile`, 'gift').plan();
}

export function deleteGift(game: Game, giftId: string): Plan {
  const gift = game.gifts?.[giftId];
  if (!gift) return refuse('No such gift');
  const tx = new Tx(game).set(`gifts/${giftId}`, null);
  if (gift.ownerId !== MISFIT) tx.set(`players/${gift.ownerId}/giftId`, null);
  return tx.event(`⚙️ Admin: Deleted "${gift.name}"`, 'gift').plan();
}

/** Add a gift by hand — to the Misfit pile or straight to a player (who must not hold one). */
export function addGift(game: Game, giftId: string, giftName: string, ownerId: PlayerId | typeof MISFIT, now: number): Plan {
  const nm = giftName.trim();
  if (!nm) return refuse('Please enter a gift name');
  const tx = new Tx(game);
  if (ownerId === MISFIT) {
    tx.set(`gifts/${giftId}`, { name: nm, naughtyLevel: 0, ownerId: MISFIT, misfitOrder: now, createdAt: now });
  } else {
    if (!game.players?.[ownerId]) return refuse('No such player');
    const theirs = game.players[ownerId].giftId;
    if (theirs) toMisfit(tx, theirs, now);
    tx.set(`gifts/${giftId}`, { name: nm, naughtyLevel: 0, ownerId, createdAt: now }).set(`players/${ownerId}/giftId`, giftId);
  }
  return tx.event(`Added gift: "${nm}"`, 'gift').plan();
}

export const resetTurn = (game: Game): Plan => new Tx(game).set('currentTurn', null).set('minigame', null).event('⚙️ Admin: Reset turn', 'gift').plan();

/** Put players + gifts back as they were before the last duel was resolved. */
/** The host looks at the sealed bids before the reveal: their console shows the amounts from now
 *  on (projection.ts), and the table's log says so. Nobody else sees them. */
export function peekBids(game: Game): Plan {
  const t = game.currentTurn;
  if (!t || (t.phase !== 'bidding' && t.phase !== 'endgame_bidding')) return refuse('No sealed bids to look at');
  if (t.hostPeeked) return refuse('Already peeked');
  return new Tx(game).set('currentTurn/hostPeeked', true).event('👀 The host peeked at the sealed bids', 'bid').plan();
}

/** Put players and gifts back as they were before the last duel was resolved. */
export function undoDuel(game: Game): Plan {
  if (!game.undo) return refuse('No duel to undo');
  return new Tx(game).set('players', game.undo.players || {}).set('gifts', game.undo.gifts || {}).set('undo', null)
    .event('⚙️ Admin: Restored previous state', 'gift').plan();
}
