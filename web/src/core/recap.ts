// The end-of-night recap: who stood where, what everyone went home with, and a few light-hearted
// awards. Pure and read-only: it works from whatever view of the finished game a screen was sent
// (core/projection.ts), and every award is worked out from numbers the game already keeps on its
// players and gifts. Nothing here is recorded for the recap's sake.
import { partyStandings } from './scoring';
import type { Game, PlayerId } from './types';

export interface RecapGift { name: string; naughty: number }

/** One player's final line. `place` follows the standings (ties share a place: 1, 2, 2, 4); 0 = nobody scored, so no places. */
export interface RecapLine {
  id: PlayerId;
  name: string;
  avatar?: string;
  place: number;
  score: number;
  points: number;
  calls: number;
  duels: number;
  chips: number;
  gift: RecapGift | null;
}

/** A step of the podium: everyone who shares that place (the standings' own places, so 1, 2, 2 has no third step). */
export interface PodiumStep { place: 1 | 2 | 3; players: RecapLine[] }

export type AwardId = 'duel_champ' | 'crystal_ball' | 'party_star' | 'naughty_list' | 'brawler' | 'big_spender' | 'good_sport' | 'silent_night' | 'deep_pockets';

/** An award someone clearly earned. `icon` is a sticker's name (ui/icons.tsx). */
export interface Award { id: AwardId; title: string; icon: string; winner: PlayerId; name: string; detail: string }

/** A Sketch-Off drawing still in the view when the game ended (a data URL). */
export interface RecapDrawing { id: PlayerId; name: string; img: string }

export interface Recap {
  /** Somebody has a Party Score: there are places and a podium. */
  scored: boolean;
  lines: RecapLine[];
  podium: PodiumStep[];
  awards: Award[];
  drawings: RecapDrawing[];
  prompt?: string;
}

export const MIN_AWARDS = 3;
export const MAX_AWARDS = 5;

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The one line with the highest `value`, if it is alone at the top and at least `min`. */
function soleTop(lines: RecapLine[], value: (l: RecapLine) => number, min: number): RecapLine | null {
  let best: RecapLine | null = null, tied = false;
  for (const l of lines) {
    const v = value(l);
    if (!best || v > value(best)) { best = l; tied = false; } else if (v === value(best)) tied = true;
  }
  return best && !tied && value(best) >= min ? best : null;
}

/** The awards in the order they are considered: [id, title, icon, who earned it and what the card says]. */
const RULES: [AwardId, string, string, (lines: RecapLine[], game: Game) => { who: RecapLine; detail: string } | null][] = [
  // most duels won, alone at the top, at least one
  ['duel_champ', 'Duel Champion', 'duel', (lines, game) => {
    const wins = (l: RecapLine) => game.players?.[l.id]?.duelWins || 0;
    const who = soleTop(lines, wins, 1);
    return who && { who, detail: `Won ${plural(wins(who), 'duel')}` };
  }],
  // most duels called right, alone at the top, at least two
  ['crystal_ball', 'Crystal Ball', 'eye', (lines) => {
    const who = soleTop(lines, (l) => l.calls, 2);
    return who && { who, detail: `Called ${plural(who.calls, 'duel')} right` };
  }],
  // most Party points from the mini-games, alone at the top, at least one
  ['party_star', 'Life of the Party', 'party', (lines) => {
    const who = soleTop(lines, (l) => l.points, 1);
    return who && { who, detail: `${plural(who.points, 'point')} from the party games` };
  }],
  // holds the gift with the highest naughty level, alone at the top, level one or more
  ['naughty_list', 'Top of the Naughty List', 'naughty', (lines) => {
    const who = soleTop(lines, (l) => l.gift?.naughty || 0, 1);
    return who && { who, detail: 'Took home the most fought-over gift' };
  }],
  // most duels fought, alone at the top, at least two
  ['brawler', 'Yule Dueller', 'fire', (lines) => {
    const who = soleTop(lines, (l) => l.duels, 2);
    return who && { who, detail: `Fought ${plural(who.duels, 'duel')}` };
  }],
  // fewest chips left, alone at the bottom
  ['big_spender', 'Big Spender', 'chip', (lines) => {
    if (lines.length < 2) return null;
    const who = soleTop(lines, (l) => -l.chips, -Infinity);
    return who && { who, detail: who.chips > 0 ? `Only ${plural(who.chips, 'chip')} left` : 'Spent every last chip' };
  }],
  // most duels lost, alone at the top, at least two
  ['good_sport', 'Good Sport', 'shield', (lines, game) => {
    const losses = (l: RecapLine) => game.players?.[l.id]?.duelLosses || 0;
    const who = soleTop(lines, losses, 2);
    return who && { who, detail: `Lost ${plural(losses(who), 'duel')} and kept on smiling` };
  }],
  // the only one at a table of three or more who fought no duel
  ['silent_night', 'Silent Night', 'tree', (lines) => {
    const calm = lines.filter((l) => l.duels === 0);
    return lines.length >= 3 && calm.length === 1 ? { who: calm[0], detail: 'Not one duel all night' } : null;
  }],
  // most chips left, alone at the top, at least one
  ['deep_pockets', 'Deep Pockets', 'moneybag', (lines) => {
    const who = soleTop(lines, (l) => l.chips, 1);
    return who && { who, detail: `${plural(who.chips, 'chip')} still in the stocking` };
  }],
];

/** Up to MAX_AWARDS, one each while they last; a second for someone only to reach MIN_AWARDS. */
export function awardsOf(game: Game, lines: RecapLine[]): Award[] {
  const earned: Award[] = [];
  for (const [id, title, icon, rule] of RULES) {
    const hit = rule(lines, game);
    if (hit) earned.push({ id, title, icon, winner: hit.who.id, name: hit.who.name, detail: hit.detail });
  }
  const has = (given: Award[], who: PlayerId) => given.filter((a) => a.winner === who).length;
  const given: Award[] = [];
  for (const a of earned) if (given.length < MAX_AWARDS && has(given, a.winner) === 0) given.push(a);
  for (const a of earned) if (given.length < MIN_AWARDS && !given.includes(a) && has(given, a.winner) < 2) given.push(a);
  return earned.filter((a) => given.includes(a));            // in the order they are read out
}

/** Sketch-Off drawings that are still in this view (real pictures only: a hidden one is `data:,`). */
function drawingsOf(game: Game): { drawings: RecapDrawing[]; prompt?: string } {
  const mg = game.minigame as { type?: string; prompt?: string; submissions?: Record<PlayerId, unknown> } | null | undefined;
  if (!mg || mg.type !== 'sketch_off' || !mg.submissions) return { drawings: [] };
  const drawings = Object.entries(mg.submissions)
    .filter((e): e is [PlayerId, string] => typeof e[1] === 'string' && e[1].startsWith('data:image/'))
    .map(([id, img]) => ({ id, name: game.players?.[id]?.name || '?', img }));
  return { drawings, ...(typeof mg.prompt === 'string' && drawings.length ? { prompt: mg.prompt } : {}) };
}

export function recapOf(game: Game | null | undefined): Recap {
  const g: Game = game || { phase: 'ended' };
  const players = g.players || {};
  const table = partyStandings(players);
  const scored = table.some((r) => r.score > 0);
  const lines: RecapLine[] = table.map((r) => {
    const p = players[r.id] || { name: r.name, chips: 0 };
    const gift = p.giftId ? g.gifts?.[p.giftId] : undefined;
    return {
      id: r.id, name: r.name, ...(p.avatar ? { avatar: p.avatar } : {}), place: scored ? r.place : 0, score: r.score,
      points: r.points, calls: r.calls, duels: r.duels, chips: Number(p.chips) || 0,
      gift: p.giftId ? { name: gift?.name || 'Mystery gift', naughty: Number(gift?.naughtyLevel) || 0 } : null,
    };
  });
  const podium: PodiumStep[] = ([1, 2, 3] as const)
    .map((place) => ({ place, players: lines.filter((l) => l.place === place && l.score > 0) }))
    .filter((s) => s.players.length > 0);
  return { scored, lines, podium, awards: awardsOf(g, lines), ...drawingsOf(g) };
}

export const ordinal = (n: number) => { const s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
