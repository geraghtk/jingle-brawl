// What the screens show, in words, for simulated guests: the TV everyone can see, and one
// person's own phone. Only what the app puts on that screen: sealed bids stay sealed until the
// reveal, a phone shows its own chips (everyone's only with public chips on), the tied names
// stay hidden until the party game settles them. (The live-app driver reads the real pages'
// text instead.)
import { duelGameStep, duelPicker, SPIN_MS } from './duelgame';
import { GAMES } from './party';
import type { PartyGame } from './party';
import { rankedBids, topTiedBidders } from './rules';
import { config } from './settings';
import type { Game, PlayerId } from './types';
import { voteStatus } from './voting';

export interface Screen { title: string; lines: string[] }

const nm = (game: Game, id?: PlayerId | null) => (id && game.players?.[id]?.name) || 'someone';
const giftName = (game: Game, id?: string | null) => (id && game.gifts?.[id]?.name) || 'a gift';

/** The TV: the whole table sees it. */
export function tvScreen(game: Game, now: number): Screen {
  const t = game.currentTurn;
  const cfg = config(game.settings);
  const lines: string[] = [];
  const holdings = Object.entries(game.players || {}).map(([, p]) =>
    `${p.name}: ${p.giftId ? `holds ${giftName(game, p.giftId)}` : 'still in the bag'}${cfg.publicChips ? `, ${p.chips} chips` : ''}`);
  if (game.phase === 'lobby') return { title: 'Waiting for players', lines: [`${Object.keys(game.players || {}).length} joined`] };
  if (game.phase === 'ended') return { title: 'Game over', lines: holdings };
  if (game.lastDuel && typeof game.lastDuel.at === 'number' && now - game.lastDuel.at < 6000) {
    lines.push(`${nm(game, game.lastDuel.winnerId)} beat ${nm(game, game.lastDuel.loserId)}${game.lastDuel.pot ? ` and took ${game.lastDuel.pot} chips` : ''}`);
  }
  if (!t) return { title: game.phase === 'endgame' ? 'Misfit Lottery' : 'Between turns', lines: [...lines, ...holdings] };
  const mg = game.minigame as PartyGame | undefined;
  switch (t.phase) {
    case 'bidding': {
      const bidders = Object.keys(game.players || {}).filter((id) => id !== t.openerId);
      const sealed = bidders.filter((id) => t.bids?.[id]);
      return { title: `${nm(game, t.openerId)} opened ${giftName(game, t.giftId)}`, lines: [...lines, `Sealed bids: ${sealed.length} of ${bidders.length} in`, ...holdings] };
    }
    case 'reveal': case 'yield_choice': case 'duel_pending': case 'yield_pending': case 'tiebreaker_pending': {
      const bids = rankedBids(t).filter(([, a]) => a > 0);
      const tied = topTiedBidders(t);
      lines.push(bids.length ? (tied.length >= 2 ? `A ${tied.length}-way tie at ${bids[0][1]} chips` : `Top bid: ${nm(game, bids[0][0])}, ${bids[0][1]} chips`) : 'No bids');
      if (t.phase === 'yield_pending') lines.push(`${nm(game, t.openerId)} yields: ${nm(game, t.yieldC1)} vs ${nm(game, t.yieldC2)}`);
      if (t.phase === 'duel_pending') lines.push(`${nm(game, t.openerId)} fights ${nm(game, t.challengerId)}`);
      return { title: `Bids revealed for ${giftName(game, t.giftId)}`, lines };
    }
    case 'duel': {
      const step = duelGameStep(game);
      const at = t.wheel?.spunAt;
      const turning = typeof at === 'number' && now < at + SPIN_MS;
      lines.push(`${nm(game, t.challengerId)} (red) vs ${nm(game, t.defenderId)} (green)${t.giftId ? ` for ${giftName(game, t.giftId)}` : ''}${t.pot ? `, ${t.pot} chips to the winner` : ''}`);
      if (step === 'picker') lines.push(`${nm(game, duelPicker(game))} is picking the duel`);
      else if (step === 'wheel' || turning) lines.push('The wheel is spinning');
      else if (t.duelGame) lines.push(`The duel: ${t.duelGame.name}`);
      const v = voteStatus(game, now);
      if (v.eligible.length && !turning && !step) lines.push(`${v.voted.length} of ${v.eligible.length} have called it${v.blocking ? ' (waiting for the rest)' : ''}`);
      return { title: 'Duel!', lines };
    }
    case 'reprisal':
      return { title: 'Reindeer Reprisal?', lines: [...lines, `${nm(game, t.reprisalPlayerId)} may strike back`] };
    case 'minigame_party': {
      if (!mg || !(mg.type in GAMES)) return { title: 'Party game', lines };
      const g = GAMES[mg.type];
      lines.push(mg.phase === 'intro' ? g.rules : mg.phase === 'playing' ? (mg.prompt ? `Draw: ${mg.prompt}` : 'Everyone is whacking elves')
        : mg.phase === 'voting' ? 'Vote for your favourite drawing' : '');
      if (mg.phase === 'results' && mg.results) {
        lines.push(`Winners: ${mg.results.winners.map((id) => nm(game, id)).join(' & ')}`);
        if (mg.results.bonus) lines.push(`Elf Bonus: ${nm(game, mg.results.bonus)} +1 chip`);
      }
      return { title: `${g.label} (${mg.phase})`, lines: lines.filter(Boolean) };
    }
    case 'endgame_choice': case 'endgame_steal_select': case 'endgame_nobids':
      return { title: 'Misfit Lottery', lines: [...lines, `${nm(game, t.activePlayerId)} draws the Misfit ${t.misfitName || ''}`.trim()] };
    case 'endgame_bidding': case 'endgame_reveal':
      return { title: `Auction for ${t.misfitName || 'the Misfit'}`, lines: [...lines, t.phase === 'endgame_reveal' ? `Top bid: ${nm(game, rankedBids(t)[0]?.[0])}` : 'Sealed bids going in'] };
    default:
      return { title: t.phase, lines };
  }
}

/** One person's phone. */
export function phoneScreen(game: Game, me: PlayerId, now: number): Screen {
  const p = game.players?.[me];
  if (!p) return { title: 'Not in this game', lines: [] };
  const t = game.currentTurn;
  const lines = [`You have ${p.chips} chips${p.giftId ? ` and hold ${giftName(game, p.giftId)}` : ' and no gift yet'}`];
  if (!t) return { title: game.phase === 'ended' ? 'Game over' : 'Watch the TV', lines };
  const mg = game.minigame as PartyGame | undefined;
  const bid = t.bids?.[me]?.amount;
  switch (t.phase) {
    case 'bidding': case 'endgame_bidding':
      if (me === (t.phase === 'bidding' ? t.openerId : t.activePlayerId)) return { title: 'Your gift is up for grabs', lines: [...lines, 'Waiting for the bids'] };
      return { title: bid == null ? `Bid on ${t.phase === 'bidding' ? giftName(game, t.giftId) : t.misfitName || 'the Misfit'}` : `Bid sealed: ${bid}`, lines };
    case 'reveal': case 'yield_choice': {
      const mine = t.openerId === me;
      const bids = rankedBids(t).filter(([, a]) => a > 0);
      if (mine && !bids.length) return { title: 'No challengers', lines: [...lines, "Keep your gift, or play Grinch's Gambit on someone else's"] };
      if (mine && bids.length >= 2) {
        const tied = topTiedBidders(t);
        return { title: 'Fight or yield?', lines: [...lines, tied.length >= 2 ? `${tied.length} tied at ${bids[0][1]}: fight and a party game picks who duels you; yield and they fight each other`
          : `${nm(game, bids[0][0])} bid ${bids[0][1]}; second ${nm(game, bids[1][0])}. Fight ${nm(game, bids[0][0])}, or yield and they fight each other`] };
      }
      return { title: 'Bids revealed', lines: [...lines, bid ? `Your bid: ${bid}` : 'You passed'] };
    }
    case 'duel': {
      const fighting = me === t.challengerId || me === t.defenderId;
      const step = duelGameStep(game);
      if (step === 'picker' && duelPicker(game) === me) return { title: 'Pick the duel', lines: [...lines, `You're up against ${nm(game, me === t.challengerId ? t.defenderId : t.challengerId)}`] };
      if (fighting) return { title: `Duel: you vs ${nm(game, me === t.challengerId ? t.defenderId : t.challengerId)}`, lines: [...lines, t.duelGame ? `The duel: ${t.duelGame.name}` : 'Choosing the duel'] };
      const mine = t.duelVotes?.[me]?.votedFor;
      return { title: 'Call it', lines: [...lines, `${nm(game, t.challengerId)} vs ${nm(game, t.defenderId)}`, mine ? `You backed ${nm(game, mine)}` : voteStatus(game, now).accepting ? 'Tap who you think will win' : 'Voting is closed'] };
    }
    case 'reprisal':
      return t.reprisalPlayerId === me ? { title: 'Reindeer Reprisal', lines: [...lines, 'Strike back at a gift, or skip'] } : { title: 'Reprisal', lines: [...lines, `${nm(game, t.reprisalPlayerId)} is deciding`] };
    case 'minigame_party': {
      if (!mg || !(mg.type in GAMES)) return { title: 'Party game', lines };
      const g = GAMES[mg.type];
      if (mg.phase === 'playing' && mg.type === 'sketch_off') lines.push(mg.submissions?.[me] ? 'Drawing handed in' : `Draw: ${mg.prompt}`);
      if (mg.phase === 'playing' && mg.type === 'whack_elf') lines.push(`Your score: ${mg.scores?.[me]?.score ?? 0}`);
      if (mg.phase === 'voting') lines.push(mg.votes?.[me] ? `You voted for ${nm(game, mg.votes[me])}'s drawing` : `Vote for a drawing: ${Object.keys(mg.submissions || {}).filter((id) => id !== me).map((id) => nm(game, id)).join(', ')}`);
      if (mg.phase === 'results' && mg.results) lines.push(`You placed ${mg.results.ranking.indexOf(me) + 1} of ${mg.results.ranking.length}`);
      return { title: `${g.label}: ${mg.phase}`, lines };
    }
    case 'endgame_choice': case 'endgame_steal_select': case 'endgame_nobids':
      return t.activePlayerId === me ? { title: 'Your Misfit Lottery turn', lines: [...lines, `The Misfit: ${t.misfitName || ''}`] } : { title: 'Misfit Lottery', lines: [...lines, `${nm(game, t.activePlayerId)} is choosing`] };
    default:
      return { title: 'Watch the TV', lines };
  }
}
