// What a duel is for, said before it is settled: an in-app game's ending announces the winner AND what
// they take, so nothing has to announce the duel a second time afterwards. It is worked out by running
// resolveDuel itself on the game as it stands and reading off what changed, so the promise cannot differ
// from what the hand-off then does. The room server works it out (it knows every chip) and sends it with
// the view (core/projection.ts): `arenaView.stakes`, or `minigame.stakes` for a party game / AI Draw Off
// that is deciding a duel.
import { resolveDuel } from './duel';
import { applyUpdates } from './tx';
import type { Game, LastDuel, PlayerId } from './types';
import { MISFIT } from './types';

export interface DuelStakes {
  winnerId: PlayerId;
  loserId: PlayerId;
  /** Chips: the pot the winner takes, Santa's cut already in the Fund, the loser's dividend, what the loser paid in. */
  pot: number;
  tax: number;
  dividend: number;
  loserPaid: number;
  /** A Misfit Toll the loser pays the winner. */
  toll: number;
  /** The gift the winner ends with because of this duel, and whether it was theirs already. */
  gift?: string;
  kept?: boolean;
  /** A gift the loser is handed (the winner's old one, the Misfit). */
  loserGift?: string;
  /** The loser had a gift and now has none. */
  loserGiftless?: boolean;
  /** Gifts that go to the Misfit pile. */
  toMisfit?: string[];
  /** The loser gets a Reindeer Reprisal. */
  reprisal?: boolean;
  /** The Misfit Lottery draws for the loser again (a giftless bidder won the Misfit). */
  again?: boolean;
  /** Who called it right, out of how many who called. */
  callers?: PlayerId[];
  voters?: number;
}

/** The game with this duel settled for `winnerId`: exactly resolveDuel's plan, applied. Null when there is no such duel. */
function settled(game: Game, winnerId: PlayerId, now: number): { before: Game; after: Game; then?: string } | null {
  const t = game.currentTurn;
  if (!t?.challengerId || !t.defenderId || t.lab) return null;
  // (an in-app game holds the turn in its own phase and has closed the vote; the hand-off puts it back to `duel`)
  const before: Game = { ...game, currentTurn: { ...t, phase: 'duel', voteClosed: true } };
  const plan = resolveDuel(before, winnerId, now);
  if (plan.error) return null;
  return { before, after: applyUpdates(before, plan.updates, now), then: plan.then };
}

/** What `winnerId` winning the duel on the table does. Null: no duel, or it cannot be settled that way. */
export function duelStakes(game: Game, winnerId: PlayerId, now: number): DuelStakes | null {
  const s = settled(game, winnerId, now);
  if (!s) return null;
  const { before, after } = s;
  const t = before.currentTurn!;
  const last = after.lastDuel as LastDuel | undefined;
  if (!last) return null;
  const loserId = last.loserId;
  const giftName = (id: string | null | undefined) => (id ? before.gifts?.[id]?.name : undefined);
  const had = (id: PlayerId) => before.players?.[id]?.giftId || null;
  const has = (id: PlayerId) => after.players?.[id]?.giftId || null;
  const chips = (g: Game, id: PlayerId) => Number(g.players?.[id]?.chips) || 0;

  const out: DuelStakes = {
    winnerId, loserId, pot: last.pot || 0, tax: t.tax || 0, dividend: last.dividend || 0, loserPaid: last.loserPaid || 0,
    toll: Math.max(0, chips(after, winnerId) - chips(before, winnerId) - (last.pot || 0)),
  };
  const wGift = has(winnerId);
  if (wGift && wGift !== had(winnerId)) out.gift = giftName(wGift);
  else if (wGift && wGift === t.giftId) { out.gift = giftName(wGift); out.kept = true; }
  const lGift = has(loserId);
  if (lGift && lGift !== had(loserId)) out.loserGift = giftName(lGift);
  else if (!lGift && had(loserId)) out.loserGiftless = true;
  const toMisfit = Object.entries(after.gifts || {})
    .filter(([id, g]) => g.ownerId === MISFIT && before.gifts?.[id]?.ownerId !== MISFIT).map(([, g]) => g.name);
  if (toMisfit.length) out.toMisfit = toMisfit;
  if (after.currentTurn?.phase === 'reprisal' && after.currentTurn.reprisalPlayerId === loserId) out.reprisal = true;
  if (s.then === 'drawEndgamePlayer') out.again = true;
  const votes = Object.entries(t.duelVotes || {});
  if (votes.length) { out.voters = votes.length; out.callers = votes.filter(([, v]) => v.votedFor === winnerId).map(([id]) => id); }
  return out;
}

/** A gift's name in a sentence: "the Fuzzy socks", but "A hand-knitted jumper" keeps the article it came with. */
export const theGift = (name: string) => (/^(a|an|the|some|my|our)\s/i.test(name.trim()) ? name.trim() : `the ${name.trim()}`);

const chipsWord = (n: number) => `${n} chip${n === 1 ? '' : 's'}`;

export interface StakesWords {
  /** What the winner takes: "Alice takes the Fuzzy socks and the pot of 3 chips". Empty: nothing but the win. */
  prize: string;
  /** What it means for the loser, as short notes: "paid 2 chips", "gets the Candle", "+1 chip Loser's Dividend", "a Reprisal chance". */
  loser: string[];
  /** The small print: a gift for the Misfit pile, Santa's cut, who called it. */
  foot: string[];
}

/** The stakes in words, the same on every screen (the wording follows the duel verdict's). */
export function stakesWords(s: DuelStakes, name: (id: PlayerId) => string): StakesWords {
  const takes: string[] = [];
  if (s.gift) takes.push(theGift(s.gift));
  if (s.pot > 0) takes.push(`the pot of ${chipsWord(s.pot)}`);
  if (s.toll > 0) takes.push(`a Misfit Toll of ${chipsWord(s.toll)}`);
  const verb = s.gift && s.kept ? 'keeps' : 'takes';
  const prize = takes.length ? `${name(s.winnerId)} ${verb} ${takes.join(' and ')}` : '';

  const loser: string[] = [];
  if (s.loserPaid > 0) loser.push(`paid ${chipsWord(s.loserPaid)}`);
  if (s.loserGift) loser.push(`gets ${theGift(s.loserGift)}`);
  else if (s.loserGiftless) loser.push('is left without a gift');
  if (s.dividend > 0) loser.push(`+${chipsWord(s.dividend)} Loser's Dividend`);
  if (s.reprisal) loser.push('a Reprisal chance');
  if (s.again) loser.push('is drawn again for the next Misfit');

  const foot: string[] = [];
  for (const g of s.toMisfit || []) foot.push(`${theGift(g)[0].toUpperCase()}${theGift(g).slice(1)} goes to the Misfit pile`);
  if (s.tax > 0) foot.push(`North Pole Fund +${s.tax}`);
  if (s.voters) foot.push(s.callers?.length ? `Called it: ${s.callers.map(name).join(', ')}` : 'Nobody saw that coming');
  return { prize, loser, foot };
}
