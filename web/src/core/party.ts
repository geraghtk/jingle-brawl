// Party mini-games (Sketch-Off, Whack-an-Elf): the rules. The whole table plays; one
// ranking settles the tie (best-placed tied bidder, two after a yield) and pays party
// points 3/2/1 plus an Elf Bonus chip to the best finisher outside the tie.
import { rng } from './rng';
import type { PartyPrize, PlayerId } from './types';

export type PartyType = 'sketch_off' | 'whack_elf';
export type PartyPhase = 'intro' | 'playing' | 'voting' | 'results';

export interface PartyResults {
  ranking: PlayerId[];
  scoreOf: Record<PlayerId, number>;
  points: Record<PlayerId, number>;
  bonus: PlayerId | null;
  winners: PlayerId[];
}

export interface PartyGame {
  type: PartyType;
  phase: PartyPhase;
  seed: number;
  participants: PlayerId[];
  tiedBidders: PlayerId[];
  minigameGoal: 'pick_one' | 'pick_two' | 'duel'; // duel = an in-app duel game (duelgame.ts)
  prize: PartyPrize;
  startedAt: number;
  phaseEndsAt: number;
  playStartAt?: number;
  prompt?: string;
  scores?: Record<PlayerId, { score: number; final?: boolean }>;
  submissions?: Record<PlayerId, string>;
  votes?: Record<PlayerId, PlayerId>;
  claims?: Record<string, boolean>;
  results?: PartyResults;
}

export const GAMES: Record<PartyType, { label: string; emoji: string; introMs: number; playMs: number; voteMs?: number; rules: string }> = {
  sketch_off: {
    label: 'Sketch-Off', emoji: '✏️', introMs: 7000, playMs: 60000, voteMs: 30000,
    rules: 'Everyone draws the same prompt. Then everyone votes for their favourite (not your own!).',
  },
  whack_elf: {
    label: 'Whack-an-Elf', emoji: '🔨', introMs: 7000, playMs: 20000,
    rules: 'Tap the elves as they pop up. The faster you tap, the more it scores (up to 3). Golden elves score double. Tap the Grinch and lose 3!',
  },
};
export const METHODS = Object.keys(GAMES) as PartyType[];
export const RESULTS_MS = 9000;
const PLACE_POINTS = [3, 2, 1];

export const SKETCH_PROMPTS = [
  'A reindeer on a skateboard', 'Santa stuck in a chimney', 'A snowman at the beach',
  'An elf lifting weights', 'The Grinch doing yoga', 'A penguin opening a present',
  'A gingerbread house on fire', 'Rudolph with a cold', 'A Christmas tree in a tiny car',
  'Mrs Claus riding a motorbike', 'A polar bear ice skating', 'A sleigh with a flat tyre',
  'A cat attacking the wrapping paper', 'Santa on a treadmill', 'A reindeer at the dentist',
  'An elf asleep on the job', 'A snowman with a sunburn', 'The ugliest Christmas sweater ever',
  'A present that is obviously a bike', 'Grandma winning a snowball fight', 'A turkey hiding from the cook',
  'A gift wrapped by a toddler', 'A Christmas tree falling over', 'A penguin delivering presents',
  'A sleigh stuck in traffic', 'A dog wearing antlers and hating it', 'Mistletoe in an awkward place',
  'A stocking that is far too full', 'The Grinch at a gift exchange', 'A reindeer singing karaoke',
  'A fruitcake used as a doorstop', 'A gingerbread man running late', 'Carolers nobody asked for',
  'A candle that smells like socks', 'An elf on a coffee break', 'Santa squeezing into his suit',
];

export type WhackKind = 'elf' | 'gold' | 'grinch';
export interface WhackPop { at: number; cell: number; kind: WhackKind; upMs: number }

/** Points for a tap `frac` of the way through a pop-up (0 = just appeared): speed pays. */
export function whackPoints(kind: WhackKind, frac: number): number {
  if (kind === 'grinch') return -3;
  const speed = frac < 1 / 3 ? 3 : frac < 2 / 3 ? 2 : 1;
  return kind === 'gold' ? speed * 2 : speed;
}

/** Pop-ups for one Whack-an-Elf game (same seed → same sequence on every phone). */
export function whackSequence(seed: number, playMs = GAMES.whack_elf.playMs): WhackPop[] {
  const r = rng(seed);
  const items: WhackPop[] = [];
  const busyUntil = Array(9).fill(0) as number[];
  for (let at = 600; at < playMs - 700; at += 260 + r() * 260) {
    const progress = at / playMs;
    const upMs = Math.round(1000 - 450 * progress); // speeds up as the game goes on
    const free = busyUntil.map((t, cell) => (t <= at ? cell : -1)).filter((c) => c >= 0);
    if (!free.length) continue;
    const cell = free[Math.floor(r() * free.length)];
    const roll = r();
    const kind: WhackKind = roll < 0.1 ? 'gold' : roll < 0.26 ? 'grinch' : 'elf';
    busyUntil[cell] = at + upMs + 120;
    items.push({ at: Math.round(at), cell, kind, upMs });
  }
  return items;
}

/** A phone's clock may run this far ahead of the room's: a pop-up it already tapped counts. */
export const WHACK_CLOCK_SLACK_MS = 1500;

/** The most (and least) anyone can have scored `elapsedMs` into a Whack-an-Elf game: every elf and
 *  golden elf up so far tapped at its fastest, no Grinch (the least: every Grinch, nothing else). The
 *  room holds a phone's reported score to this: it can't claim a thousand. */
export function whackBounds(seed: number, elapsedMs = Infinity, playMs = GAMES.whack_elf.playMs): { min: number; max: number } {
  let min = 0, max = 0;
  for (const p of whackSequence(seed, playMs)) {
    if (p.at > elapsedMs + WHACK_CLOCK_SLACK_MS) break;
    if (p.kind === 'grinch') min += whackPoints('grinch', 0);
    else max += whackPoints(p.kind, 0);
  }
  return { min, max };
}

/** Sketch-Off score = votes received (no self-votes, only for real drawings). */
export function sketchScores(votes: Record<PlayerId, PlayerId> | undefined, submissions: Record<PlayerId, unknown> | undefined) {
  const scoreOf: Record<PlayerId, number> = {};
  for (const pid of Object.keys(submissions || {})) scoreOf[pid] = 0;
  for (const [voter, target] of Object.entries(votes || {})) {
    if (voter !== target && target in scoreOf) scoreOf[target]++;
  }
  return scoreOf;
}

/** Whack-an-Elf score = the latest score each phone reported. */
export function whackScores(scores: PartyGame['scores']) {
  const scoreOf: Record<PlayerId, number> = {};
  for (const [pid, s] of Object.entries(scores || {})) scoreOf[pid] = Number(s?.score) || 0;
  return scoreOf;
}

/** Everyone, best first. Equal scores are ordered by a seeded coin, not by seat. */
export function rank(participants: PlayerId[], scoreOf: Record<PlayerId, number>, seed: number): PlayerId[] {
  const r = rng(seed ^ 0x5eed);
  const coin = Object.fromEntries(participants.map((id) => [id, r()]));
  return [...participants].sort((a, b) =>
    (scoreOf[b] ?? -Infinity) - (scoreOf[a] ?? -Infinity) || coin[a] - coin[b]);
}

/** Tiebreak winners: tied bidders in ranking order — 1 to duel the Opener, 2 after a yield. */
export function tieWinners(ranking: PlayerId[], tiedBidders: PlayerId[], goal: PartyGame['minigameGoal']): PlayerId[] {
  const placed = ranking.filter((id) => tiedBidders.includes(id));
  const missing = tiedBidders.filter((id) => !placed.includes(id));
  return [...placed, ...missing].slice(0, goal === 'pick_two' ? 2 : 1);
}

/** 3/2/1 party points to scorers; the Elf Bonus to the best scorer outside the tie. */
export function rewards(ranking: PlayerId[], scoreOf: Record<PlayerId, number>, tiedBidders: PlayerId[], prize: PartyPrize = 'points_chips') {
  const points: Record<PlayerId, number> = {};
  let bonus: PlayerId | null = null;
  if (prize === 'none') return { points, bonus };
  const scorers = ranking.filter((id) => (scoreOf[id] ?? 0) > 0);
  scorers.slice(0, PLACE_POINTS.length).forEach((id, i) => { points[id] = PLACE_POINTS[i]; });
  if (prize === 'points_chips') bonus = scorers.find((id) => !tiedBidders.includes(id)) ?? null;
  return { points, bonus };
}

export function results(mg: PartyGame): PartyResults {
  const participants = mg.participants || [];
  const scoreOf = mg.type === 'sketch_off' ? sketchScores(mg.votes, mg.submissions) : whackScores(mg.scores);
  const ranking = rank(participants, scoreOf, mg.seed || 1);
  const tied = mg.tiedBidders || [];
  const { points, bonus } = rewards(ranking, scoreOf, tied, mg.prize);
  return { ranking, scoreOf, points, bonus, winners: tieWinners(ranking, tied, mg.minigameGoal) };
}

/** Sketch-Off moves on as soon as every drawing (or every possible vote) is in. */
export function everyoneDone(mg: PartyGame): boolean {
  if (mg.type !== 'sketch_off') return false;
  const n = (mg.participants || []).length;
  if (mg.phase === 'playing') return Object.keys(mg.submissions || {}).length >= n;
  if (mg.phase === 'voting') {
    const drawings = Object.keys(mg.submissions || {});
    const canVote = (mg.participants || []).filter((id) => drawings.some((d) => d !== id));
    return Object.keys(mg.votes || {}).length >= canVote.length;
  }
  return false;
}
