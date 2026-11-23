// Party mini-game flow: intro → playing → (voting) → results → the duel.
import { pacedMs } from './minigames/pace';
import type { Rng } from './rng';
import { GAMES, MAX_LIVE, RESULTS_MS, SKETCH_PROMPTS, results as partyResults, whackBounds } from './party';
import type { PartyGame, PartyType } from './party';
import { pick } from './rng';
import { config } from './settings';
import { potAndTax, topTiedBidders, yieldPayment } from './rules';
import { refuse, Tx } from './tx';
import type { Game, Plan, PlayerId } from './types';

const name = (game: Game, id: PlayerId) => game.players?.[id]?.name || 'Someone';

/** Once everyone is ready: 3, 2, 1 and the party game starts. */
export const PARTY_COUNT_MS = 3000;
/** While the intro waits for the table, its clock is set this far off (the host's Skip ahead still works). */
const WAIT_MS = 30 * 60_000;

/** Who the intro is still waiting for (a bot is always ready; a dropped phone is not waited for). */
export function partyAwaited(game: Game, mg: PartyGame): PlayerId[] {
  if (mg.phase !== 'intro' || !mg.awaiting) return [];
  return (mg.participants || []).filter((id) => { const p = game.players?.[id]; return !!p && !p.isBot && p.connected !== false && !mg.ready?.[id]; });
}

/** A player has read the rules: ready. The last one starts the 3, 2, 1. */
export function partyReady(game: Game, who: PlayerId, now: number): Plan {
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || !GAMES[mg.type] || mg.phase !== 'intro' || !mg.awaiting) return refuse('The game has started');
  if (!mg.participants?.includes(who)) return refuse("You're not in this game");
  if (mg.ready?.[who]) return refuse('Already ready');
  const tx = new Tx(game).set(`minigame/ready/${who}`, true);
  const left = partyAwaited(game, { ...mg, ready: { ...(mg.ready || {}), [who]: true } });
  if (!left.length) tx.set('minigame/awaiting', null).set('minigame/phaseEndsAt', now + PARTY_COUNT_MS);
  return tx.plan();
}

/** Nobody is left to wait for (the last one's phone dropped): the room starts the count itself. */
export function partyAllReady(game: Game, now: number): Plan {
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || !GAMES[mg.type] || mg.phase !== 'intro' || !mg.awaiting) return refuse('The game has started');
  if (partyAwaited(game, mg).length) return refuse('Still waiting for someone');
  return new Tx(game).set('minigame/awaiting', null).set('minigame/phaseEndsAt', now + PARTY_COUNT_MS).plan();
}

/** Start a party game for a tie (called inside a tie action's Tx). */
export function startPartyGame(tx: Tx, type: PartyType, goal: PartyGame['minigameGoal'], tied: PlayerId[], ctx: { now: number; rng: Rng }, players?: PlayerId[]) {
  const g = GAMES[type];
  const mg: PartyGame = {
    type, phase: 'intro', seed: Math.floor(ctx.rng() * 2147483646) + 1,
    participants: players ?? Object.keys(tx.game.players || {}), tiedBidders: tied, minigameGoal: goal,
    prize: goal === 'lab' ? 'none' : config(tx.game.settings).partyPrize,   // the Game Lab: practice, nothing is won
    startedAt: ctx.now, phaseEndsAt: ctx.now + g.introMs,
    ...(type === 'sketch_off' ? { prompt: pick(SKETCH_PROMPTS, ctx.rng) } : {}),
  };
  // the rules first, and nobody starts until every player has pressed Ready (then a short count)
  if (config(tx.game.settings).readyCheck && partyAwaited(tx.game, { ...mg, awaiting: true }).length) { mg.awaiting = true; mg.phaseEndsAt = ctx.now + WAIT_MS; }
  tx.set('minigame', mg).set('currentTurn/phase', 'minigame_party').set('currentTurn/minigameStatus', `${g.emoji} ${g.label}`)
    .event(goal === 'lab' ? `${g.emoji} ${g.label}: practice` : goal === 'duel'
      ? `${g.emoji} ${g.label} decides ${tied.map((id) => name(tx.game, id)).join(' vs ')}. Everyone plays!`
      : `${g.emoji} ${g.label}! Everyone plays. Tied: ${tied.map((id) => name(tx.game, id)).join(', ')}`, 'bid');
}

/**
 * Move a party game on from `fromPhase`. Claims minigame/claims/{fromPhase} so the phase
 * timer, an early finish and "Skip ahead" can race without double-paying rewards. The
 * new phase and its times land in one write (phones never see a half-set phase).
 */
export function advanceParty(game: Game, fromPhase: PartyGame['phase'], now: number): Plan {
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || !GAMES[mg.type] || mg.phase !== fromPhase) return refuse('That phase is over');
  const g = GAMES[mg.type];
  const tx = new Tx(game).claim(`minigame/claims/${fromPhase}`, true);
  if (fromPhase === 'intro') return tx.set('minigame/phase', 'playing').set('minigame/playStartAt', now).set('minigame/phaseEndsAt', now + (mg.type === 'sketch_off' ? pacedMs(game.settings, g.playMs) : g.playMs)).plan();
  if (fromPhase === 'playing' && mg.type === 'sketch_off') return tx.set('minigame/live', null).set('minigame/phase', 'voting').set('minigame/phaseEndsAt', now + pacedMs(game.settings, g.voteMs || 30000)).plan();
  if (fromPhase === 'results') return mg.minigameGoal === 'lab' ? refuse('Practice: play again or go back to the lobby') : completeTx(tx, mg).plan();

  // playing (Whack-an-Elf) or voting (Sketch-Off) → results + rewards
  const r = partyResults(mg);
  tx.set('minigame/phase', 'results').set('minigame/results', r).set('minigame/phaseEndsAt', now + RESULTS_MS);
  for (const [pid, pts] of Object.entries(r.points)) if (tx.get(`players/${pid}`)) tx.add(`players/${pid}/partyPoints`, pts);
  if (r.bonus && tx.get(`players/${r.bonus}`)) {
    tx.add(`players/${r.bonus}/chips`, 1).event(`🧝 Elf Bonus: ${name(game, r.bonus)} +1 chip from the North Pole Fund`, 'bid');
  }
  if (mg.minigameGoal === 'lab') return tx.event(`${g.emoji} Practice over: ${r.ranking[0] ? `${name(game, r.ranking[0])} came top` : 'nobody played'}`, 'bid').plan();
  return tx.event(`${g.emoji} ${r.winners.map((id) => name(game, id)).join(' & ')} win${r.winners.length > 1 ? '' : 's'} the ${mg.minigameGoal === 'duel' ? 'duel' : 'tie'}!`, 'bid').plan();
}

/** Results are in: the tie winner duels the Opener (or the two winners duel after a yield). */
function completeTx(tx: Tx, mg: PartyGame) {
  const winners = mg.results?.winners || partyResults(mg).winners;
  const t = tx.game.currentTurn!;
  tx.set('minigame', null);
  if (mg.minigameGoal === 'duel') {
    // Back to the duel with its winner; the host resolves it (pot, gift, votes) as usual
    return tx.set('currentTurn/phase', 'duel').set('currentTurn/duelGameWinner', winners[0]);
  }
  if (mg.minigameGoal === 'pick_two') {
    const [c1, c2] = winners;
    const pay1 = yieldPayment(t.bids?.[c1]?.amount || 0);
    const pay2 = yieldPayment(t.bids?.[c2]?.amount || 0);
    const { pot, tax } = potAndTax(config(tx.game.settings), pay1 + pay2);
    tx.add(`players/${c1}/chips`, -pay1).add(`players/${c2}/chips`, -pay2)
      .set('currentTurn/phase', 'duel').set('currentTurn/challengerId', c1).set('currentTurn/defenderId', c2)
      .set('currentTurn/pot', pot).set('currentTurn/tax', tax).set('currentTurn/isYieldDuel', true).set('currentTurn/yieldingOpener', t.openerId);
  } else {
    const c = winners[0];
    const bid = t.bids?.[c]?.amount || 0;
    const { pot, tax } = potAndTax(config(tx.game.settings), bid);
    tx.add(`players/${c}/chips`, -bid)
      .set('currentTurn/phase', 'duel').set('currentTurn/challengerId', c).set('currentTurn/defenderId', t.openerId)
      .set('currentTurn/pot', pot).set('currentTurn/tax', tax).set('currentTurn/isYieldDuel', false).set('currentTurn/isTiebreak', false);
  }
  return tx;
}

// --- what phones write ----------------------------------------------------------------------
export const submitSketch = (game: Game, pid: PlayerId, dataUrl: string): Plan => new Tx(game).set(`minigame/submissions/${pid}`, dataUrl).plan();
/** A phone's snapshot of its drawing so far, for the big screen's wall (shown there without a name). */
export function sketchLive(game: Game, pid: PlayerId, dataUrl: string): Plan {
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || mg.type !== 'sketch_off' || mg.phase !== 'playing' || !mg.participants?.includes(pid)) return refuse('Nobody is drawing now');
  if (mg.submissions?.[pid]) return refuse('Already handed in');
  if (dataUrl.length > MAX_LIVE) return refuse('That snapshot is too big');
  return new Tx(game).set(`minigame/live/${pid}`, dataUrl).plan();
}
export function voteSketch(game: Game, voter: PlayerId, target: PlayerId): Plan {
  if (voter === target) return refuse("You can't vote for your own drawing");
  return new Tx(game).set(`minigame/votes/${voter}`, target).plan();
}
/** A phone's Whack-an-Elf score so far. The phone counts its own taps, so the room holds what it says
 *  to what the pop-ups up to `now` could have scored (`whackBounds`; without `now`: the whole game's). */
export function reportWhack(game: Game, pid: PlayerId, score: number, final: boolean, now?: number): Plan {
  const mg = game.minigame as PartyGame | undefined;
  const elapsed = now != null && mg?.playStartAt != null ? Math.max(0, now - mg.playStartAt) : Infinity;
  const { min, max } = whackBounds(mg?.seed ?? 0, elapsed);
  const held = Math.max(min, Math.min(max, Math.round(Number(score) || 0)));
  return new Tx(game).set(`minigame/scores/${pid}`, { score: held, final }).plan();
}

export { topTiedBidders };
