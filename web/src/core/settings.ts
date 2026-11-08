import type { DuelApp } from './duelgame';
// House settings: defaults + reading what a game stored.
import type { DividendRule, GremlinChallenges, PartyPrize, Settings } from './types';

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
  tiebreakerMethod: 'party_mix' | 'sketch_off' | 'whack_elf';
  partyPrize: PartyPrize;
  spectatorBets: number;
  voteWindow: number;
  duelGames: string[];
  duelAppGames: DuelApp[];
  duelPick: 'defender' | 'wheel' | 'off';
  gremlinChallenges: GremlinChallenges;
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
  tiebreakerMethod: 'party_mix',
  partyPrize: 'points_chips',
  spectatorBets: 0,
  voteWindow: 30,
  duelGames: DEFAULT_DUEL_GAMES,
  duelAppGames: [],
  duelPick: 'defender',
  gremlinChallenges: 'mixed',
};

/** The challenge kinds a Gremlin game uses. */
export const gremlinKinds = (c: GremlinChallenges): string[] => (c === 'mixed' ? ['quick_math', 'memory_flash', 'perfect_timing'] : [c]);

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
    // Ties are whole-table party games only. Older saved games: 'minigame' (the retired Counterfeit
    // Canvas) → Sketch-Off; 'random' / 'draw_off' (two-player tiebreaks) → the party mix
    tiebreakerMethod: method === 'minigame' ? 'sketch_off' : method === 'sketch_off' || method === 'whack_elf' ? method : 'party_mix',
    partyPrize: s.partyPrize ?? DEFAULTS.partyPrize,
    spectatorBets: s.spectatorBets ?? DEFAULTS.spectatorBets,
    voteWindow: s.voteWindow ?? DEFAULTS.voteWindow,
    // Stored as text, one per line ("" = none); older games / RTDB may hand back an array or object
    duelGames: (typeof s.duelGames === 'string' ? s.duelGames.split('\n') : s.duelGames ? Object.values(s.duelGames) : DEFAULTS.duelGames)
      .map(String).map((x) => x.trim()).filter(Boolean),
    duelAppGames: s.duelAppGames ? Object.values(s.duelAppGames) : DEFAULTS.duelAppGames,
    duelPick: s.duelPick ?? DEFAULTS.duelPick,
    gremlinChallenges: (['mixed', 'quick_math', 'memory_flash', 'perfect_timing'] as const).find((x) => x === s.gremlinChallenges) ?? DEFAULTS.gremlinChallenges,
  };
}

/** Rules: 10 chips for up to 10 players, 12 for larger groups. */
export const variableStartingChips = (playerCount: number) => (playerCount <= 10 ? 10 : 12);

/** Minimum Cost for Gambits, Reprisals and Endgame Steals. */
export const minCost = (cfg: Config, naughtyLevel: number) => cfg.baseCost + (naughtyLevel || 0);

/** Which party game settles this tie ('party_mix' picks one per tie). */
export const partyGameFor = (m: Config['tiebreakerMethod'], r: () => number): 'sketch_off' | 'whack_elf' =>
  m === 'party_mix' ? (r() < 0.5 ? 'whack_elf' : 'sketch_off') : m;
