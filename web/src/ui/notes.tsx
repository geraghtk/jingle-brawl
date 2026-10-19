// Rule blurbs generated from the game's settings and the duel at hand — never hard-coded,
// because the printed rules differ from many house settings.
import type { ReactNode } from 'react';
import { reprisalAllowed } from '../core/rules';
import type { Config } from '../core/settings';
import type { Turn } from '../core/types';

export function dividendNote(cfg: Config, asDefender: boolean): ReactNode {
  const rule = cfg.loserDividendRule;
  const pays = rule === 'everyone' || (rule === 'defenders' && asDefender);
  return pays
    ? <>Get <strong>+{cfg.loserDividend} chip</strong> (Loser's Dividend)</>
    : <>No Loser's Dividend{rule === 'defenders' ? ' for attackers' : ''}</>;
}

export function taxNote(cfg: Config): string {
  if (cfg.taxThreshold >= 99 || cfg.maxWinnerPayout >= 99) return 'No Santa Tax this game';
  return `Santa Tax: pots of ${cfg.taxThreshold}+ pay the winner at most ${cfg.maxWinnerPayout} (rest → Bank)`;
}

export const reprisalNote = (cfg: Config, turn: Turn) => (reprisalAllowed(cfg, turn) ? ' + Reprisal chance!' : '');

/** What a spectator is told the loser of THIS duel gets. */
export function spectatorLoserNote(cfg: Config, turn: Turn): ReactNode {
  const noDividend = turn.isYieldDuel || turn.isMisfitDuel || turn.isTiebreak || cfg.loserDividendRule === 'nobody';
  const dividend = noDividend ? 'no dividend'
    : cfg.loserDividendRule === 'defenders' ? <><strong>+{cfg.loserDividend} chip</strong> if they were defending</>
    : <strong>+{cfg.loserDividend} chip</strong>;
  return <>Loser gets {dividend}{reprisalAllowed(cfg, turn) ? <> and a <strong>Reprisal</strong> chance</> : null}.</>;
}

/** What a vote on someone else's duel is worth. */
export function voteStakeNote(cfg: Config, chips: number): string {
  if (!cfg.spectatorBets) return 'A correct call adds 1 to your Party Score 🏅';
  if (chips < 1) return 'No chips to bet — your vote still counts for your Party Score 🏅';
  return "You bet 1 chip: right callers split the wrong callers' chips. Correct calls also add to your Party Score 🏅";
}
