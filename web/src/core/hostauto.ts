// What the host does by itself, as data: each step is due at a time (0 = now) and builds a
// plan. The host page arms a timer per step (useHostAutomation); the simulated table runs
// the same steps on its virtual clock, so both wait for the table by the same rules.
// A step's key changes when the state it waits on changes, so a timer is re-armed rather
// than fired twice; every plan is claimed, so a step racing a button can't apply twice.
import * as A from './actions';
import { duelAppWinner, duelGameStep, SPIN_MS, spinDuelGame } from './duelgame';
import { resolveDuel } from './duel';
import { everyoneDone, GAMES } from './party';
import type { PartyGame } from './party';
import { advanceParty } from './partyflow';
import type { Rng } from './rng';
import type { Game, Plan } from './types';
import { openVoting } from './voting';

export interface AutoStep {
  key: string;
  at: number;                 // when it's due (host clock); 0 = straight away
  quiet: boolean;             // a refusal isn't news (the state moved on)
  run: (game: Game, ctx: { now: number; rng: Rng }) => Plan;
}

/** Last Whack-an-Elf scores land a moment after the clock runs out. */
export const WHACK_GRACE_MS = 1500;
const GRACE_MS = 250;

export function hostAuto(game: Game): AutoStep[] {
  const out: AutoStep[] = [];
  const t = game.currentTurn;
  const mg = game.minigame as PartyGame | undefined;

  // party games: one timer per phase; move on early when everyone's done
  if (mg && mg.type in GAMES) {
    const from = mg.phase;
    const done = everyoneDone(mg);
    const grace = from === 'playing' && mg.type === 'whack_elf' ? WHACK_GRACE_MS : GRACE_MS;
    out.push({ key: `party|${mg.seed}|${from}|${done}`, at: done ? 0 : mg.phaseEndsAt + grace, quiet: true,
      run: (g, ctx) => advanceParty(g, from, ctx.now) });
  }

  // duel voting opens once the table knows which duel it is (after the wheel stops)
  const step = duelGameStep(game);
  const duelKey = t?.phase === 'duel' ? `${t.challengerId}|${t.defenderId}|${t.giftId}|${!!t.isTiebreak}` : null;
  const spunAt = typeof t?.wheel?.spunAt === 'number' ? t.wheel.spunAt : null;
  if (duelKey && !t!.isTiebreak && t!.voteEndsAt == null && !step) {
    out.push({ key: `vote|${duelKey}|${spunAt}`, at: spunAt == null ? 0 : spunAt + SPIN_MS, quiet: true,
      run: (g, ctx) => openVoting(g, ctx.now) });
  }

  // the wheel picks the duel when nobody is really defending (or the house always spins)
  if (duelKey && step === 'wheel') {
    out.push({ key: `spin|${duelKey}`, at: 0, quiet: true, run: (g, ctx) => spinDuelGame(g, ctx) });
  }

  // an in-app duel game decided it: settle the duel with that winner
  const appWinner = duelAppWinner(t);
  if (appWinner) {
    out.push({ key: `app|${appWinner}|${t!.duelAppStartedAt}`, at: 0, quiet: true, run: (g, ctx) => resolveDuel(g, appWinner, ctx.now) });
  }

  // ties start their party game themselves (once per turn)
  if (t?.phase === 'tiebreaker_pending') {
    const c1 = t.tiebreakerC1!, c2 = t.tiebreakerC2!;
    out.push({ key: `tie|${t.giftId}|${t.openerId}`, at: 0, quiet: false, run: (g, ctx) => A.startTiebreaker(g, c1, c2, ctx) });
  }
  return out;
}
