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

/** How far each number may go (the setup screen offers far less; 99 is its "unlimited" / "never"). A
 *  setting from the network is clamped to these: an absurd one once made the room list a billion bids. */
export const LIMITS = {
  startingChips: [1, 99], maxReprisalDepth: [0, 10], maxBid: [0, 99], maxWinnerPayout: [0, 99], taxThreshold: [0, 99],
  baseCost: [0, 99], spectatorBets: [0, 9], voteWindow: [0, 600],
} as const;
type Numeric = keyof typeof LIMITS;
const within = (key: Numeric, v: unknown): number | undefined => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return undefined;
  const [lo, hi] = LIMITS[key];
  return Math.max(lo, Math.min(hi, Math.floor(v)));
};
const oneOf = <T extends string>(v: unknown, of: readonly T[]): T | undefined => of.find((x) => x === v);
const MAX_DUEL_GAMES = 30, MAX_GAME_NAME = 80, MAX_APPS = 20;
const APP_ID = /^[a-z][a-z0-9_]{0,39}$/;   // a game's id: starts with a letter (so never __proto__)

/** A room's settings as given by whoever created it (anyone can): only the settings there are, each of
 *  its own kind and within LIMITS; everything else is dropped. What it returns is what the room stores. */
export function sanitizeSettings(raw: unknown): Settings {
  const r = (raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {}) as Record<string, unknown>;
  const out: Settings = {};
  if (typeof r.publicChips === 'boolean') out.publicChips = r.publicChips;
  for (const k of Object.keys(LIMITS) as Numeric[]) {
    const n = within(k, r[k]);
    if (n !== undefined) out[k] = n;
  }
  const rule = oneOf(r.loserDividendRule, ['everyone', 'defenders', 'nobody'] as const);
  if (rule) out.loserDividendRule = rule;
  const method = oneOf(r.tiebreakerMethod, ['party_mix', 'sketch_off', 'whack_elf', 'random', 'draw_off', 'minigame'] as const);
  if (method) out.tiebreakerMethod = method;
  const prize = oneOf(r.partyPrize, ['points_chips', 'points', 'none'] as const);
  if (prize) out.partyPrize = prize;
  const pick = oneOf(r.duelPick, ['defender', 'wheel', 'off'] as const);
  if (pick) out.duelPick = pick;
  const grem = oneOf(r.gremlinChallenges, ['mixed', 'quick_math', 'memory_flash', 'perfect_timing'] as const);
  if (grem) out.gremlinChallenges = grem;
  const critic = oneOf(r.aiCriticBackend, ['openai', 'gemini', 'local', 'ollama'] as const);
  if (critic) out.aiCriticBackend = critic;
  // the host's own duels: text, one per line ("" = none)
  if (typeof r.duelGames === 'string' || Array.isArray(r.duelGames)) {
    const given: unknown[] = typeof r.duelGames === 'string' ? r.duelGames.slice(0, MAX_DUEL_GAMES * (MAX_GAME_NAME + 1)).split('\n') : r.duelGames;
    out.duelGames = given.filter((x): x is string => typeof x === 'string').map((x) => x.trim().slice(0, MAX_GAME_NAME)).filter(Boolean).slice(0, MAX_DUEL_GAMES).join('\n');
  }
  if (Array.isArray(r.duelAppGames)) {
    out.duelAppGames = [...new Set(r.duelAppGames.filter((x): x is string => typeof x === 'string' && APP_ID.test(x)))].slice(0, MAX_APPS);
  }
  return out;
}

/** The challenge kinds a Gremlin game uses. */
export const gremlinKinds = (c: GremlinChallenges): string[] => (c === 'mixed' ? ['quick_math', 'memory_flash', 'perfect_timing'] : [c]);

/** The settings in force for a game (`??`, never `||`: 0 is a real value, e.g. "0 + Naughty"). */
export function config(settings: Settings | undefined): Config {
  const s = settings || {};
  const method = s.tiebreakerMethod ?? DEFAULTS.tiebreakerMethod;
  return {
    publicChips: s.publicChips ?? DEFAULTS.publicChips,
    // numbers are read within LIMITS too (a room saved before settings were checked)
    startingChips: within('startingChips', s.startingChips) ?? null,
    maxReprisalDepth: within('maxReprisalDepth', s.maxReprisalDepth) ?? DEFAULTS.maxReprisalDepth,
    maxBid: within('maxBid', s.maxBid) ?? DEFAULTS.maxBid,
    maxWinnerPayout: within('maxWinnerPayout', s.maxWinnerPayout) ?? DEFAULTS.maxWinnerPayout,
    loserDividend: DEFAULTS.loserDividend,
    loserDividendRule: s.loserDividendRule ?? DEFAULTS.loserDividendRule,
    taxThreshold: within('taxThreshold', s.taxThreshold) ?? DEFAULTS.taxThreshold,
    baseCost: within('baseCost', s.baseCost) ?? DEFAULTS.baseCost,
    // Ties are whole-table party games only. Older saved games: 'minigame' (the retired Counterfeit
    // Canvas) → Sketch-Off; 'random' / 'draw_off' (two-player tiebreaks) → the party mix
    tiebreakerMethod: method === 'minigame' ? 'sketch_off' : method === 'sketch_off' || method === 'whack_elf' ? method : 'party_mix',
    partyPrize: s.partyPrize ?? DEFAULTS.partyPrize,
    spectatorBets: within('spectatorBets', s.spectatorBets) ?? DEFAULTS.spectatorBets,
    voteWindow: within('voteWindow', s.voteWindow) ?? DEFAULTS.voteWindow,
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
