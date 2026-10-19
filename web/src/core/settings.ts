import type { PartyType } from './party';
// House settings: defaults + reading what a game stored.
import type { DividendRule, PartyPrize, Settings, TiebreakerMethod } from './types';

export interface Config {
  publicChips: boolean;
  startingChips: number | null;   // null = variable by table size
  maxReprisalDepth: number;
  maxBid: number;
  maxWinnerPayout: number;
  loserDividend: number;
  loserDividendRule: DividendRule;
  taxThreshold: number;
  baseCost: number;
  tiebreakerMethod: Exclude<TiebreakerMethod, 'minigame'>;
  partyPrize: PartyPrize;
  spectatorBets: number;
  voteWindow: number;
  duelGames: string[];
  duelAppGames: PartyType[];
  duelPick: 'defender' | 'wheel' | 'off';
}

export const DEFAULT_DUEL_GAMES = ['Rock, Paper, Scissors', 'Thumb War', 'Staring Contest', 'Odds & Evens', 'Arm Wrestle', 'Rhyme Battle'];

export const DEFAULTS: Config = {
  publicChips: false,
  startingChips: null,
  maxReprisalDepth: 1,
  maxBid: 4,
  maxWinnerPayout: 2,
  loserDividend: 1,
  loserDividendRule: 'defenders',
  taxThreshold: 2,
  baseCost: 1,
  tiebreakerMethod: 'random',
  partyPrize: 'points_chips',
  spectatorBets: 0,
  voteWindow: 30,
  duelGames: DEFAULT_DUEL_GAMES,
  duelAppGames: [],
  duelPick: 'defender',
};

/** The settings in force for a game (`??`, never `||`: 0 is a real value, e.g. "0 + Naughty"). */
export function config(settings: Settings | undefined): Config {
  const s = settings || {};
  const method = s.tiebreakerMethod ?? DEFAULTS.tiebreakerMethod;
  return {
    publicChips: s.publicChips ?? DEFAULTS.publicChips,
    startingChips: typeof s.startingChips === 'number' ? s.startingChips : null,
    maxReprisalDepth: s.maxReprisalDepth ?? DEFAULTS.maxReprisalDepth,
    maxBid: s.maxBid ?? DEFAULTS.maxBid,
    maxWinnerPayout: s.maxWinnerPayout ?? DEFAULTS.maxWinnerPayout,
    loserDividend: DEFAULTS.loserDividend,
    loserDividendRule: s.loserDividendRule ?? DEFAULTS.loserDividendRule,
    taxThreshold: s.taxThreshold ?? DEFAULTS.taxThreshold,
    baseCost: s.baseCost ?? DEFAULTS.baseCost,
    // 'minigame' = the retired Counterfeit Canvas; old saved games run a Sketch-Off
    tiebreakerMethod: method === 'minigame' ? 'sketch_off' : method,
    partyPrize: s.partyPrize ?? DEFAULTS.partyPrize,
    spectatorBets: s.spectatorBets ?? DEFAULTS.spectatorBets,
    voteWindow: s.voteWindow ?? DEFAULTS.voteWindow,
    // Stored as text, one per line ("" = none); older games / RTDB may hand back an array or object
    duelGames: (typeof s.duelGames === 'string' ? s.duelGames.split('\n') : s.duelGames ? Object.values(s.duelGames) : DEFAULTS.duelGames)
      .map(String).map((x) => x.trim()).filter(Boolean),
    duelAppGames: s.duelAppGames ? Object.values(s.duelAppGames) : DEFAULTS.duelAppGames,
    duelPick: s.duelPick ?? DEFAULTS.duelPick,
  };
}

/** Rules: 10 chips for up to 10 players, 12 for larger groups. */
export const variableStartingChips = (playerCount: number) => (playerCount <= 10 ? 10 : 12);

/** Minimum Cost for Gambits, Reprisals and Endgame Steals. */
export const minCost = (cfg: Config, naughtyLevel: number) => cfg.baseCost + (naughtyLevel || 0);

export const isPartyMethod = (m: Config['tiebreakerMethod']) => m === 'sketch_off' || m === 'whack_elf';
export const isMinigameMethod = (m: Config['tiebreakerMethod']) => m === 'draw_off' || isPartyMethod(m);
