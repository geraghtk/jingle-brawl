// Practice bots: stand-in players for the Game Lab, so one person can try a game that needs a table
// (or watch one on the TV with nobody at all). The host adds them in the lobby; the room server plays
// them (`botMove`, on a tick of its clock); they leave when the real game starts. A bot presses what
// the game itself says a simulated hand would (`offers()`), at a person's pace and not always right,
// so every game gets a bot without writing one. They never bid, vote in a duel or hold a gift: the
// real game has no bots in it.
import { joinPlayer, MAX_PLAYERS } from './actions';
import { doodle } from './doodle';
import { arenaAct, arenaOf, arenaOffers } from './minigames/runtime';
import { GAMES, whackBounds } from './party';
import type { PartyGame } from './party';
import { reportWhack, submitSketch, voteSketch } from './partyflow';
import type { Rng } from './rng';
import { refuse, Tx } from './tx';
import type { Game, Plan, Player, PlayerId } from './types';

/** How often the room server lets its bots look at the game. */
export const BOT_TICK_MS = 150;
export const MAX_BOTS = 8;
const NAMES = ['Tinsel', 'Pepper', 'Nutmeg', 'Cocoa', 'Mittens', 'Sprinkle', 'Figgy', 'Holly'];

/** The chance a bot acts on a tick when it has something to press (its reaction time), per game. */
const PACE: Record<string, number> = { nutcracker_knockout: 0.5, gift_conveyor: 0.3, mecha_santa: 0.4, elf_tower_wars: 0.25, reindeer_ring_toss: 0.12, pass_the_gremlin: 0.12 };
/** The Conveyor: how often a bot fits the wrong piece. */
const SLIP = 0.06;

export const isBot = (game: Game, id: PlayerId) => !!game.players?.[id]?.isBot;
export const botIds = (game: Game): PlayerId[] => Object.keys(game.players || {}).filter((id) => isBot(game, id));
export const humans = (game: Game): PlayerId[] => Object.keys(game.players || {}).filter((id) => !isBot(game, id));

/** A new bot in the lobby. */
export function addBot(game: Game, ctx: { now: number; rng: Rng }): Plan {
  if (game.phase !== 'lobby') return refuse('Practice bots are for the lobby only');
  if (game.currentTurn) return refuse('Finish the practice game first');
  const bots = botIds(game);
  if (bots.length >= MAX_BOTS) return refuse(`That is enough bots (${MAX_BOTS})`);
  if (Object.keys(game.players || {}).length >= MAX_PLAYERS) return refuse('The room is full');
  const taken = new Set(Object.values(game.players || {}).map((p) => p.name));
  const name = NAMES.map((n) => `Bot ${n}`).find((n) => !taken.has(n)) ?? `Bot ${bots.length + 1}`;
  let id = '';
  do id = `bot_${Math.floor(ctx.rng() * 36 ** 6).toString(36).padStart(6, '0')}`; while (game.players?.[id]);
  const joined = joinPlayer(game, { playerId: id, name, uid: null, now: ctx.now });
  if (joined.error) return joined;
  const player = { ...(joined.updates[`players/${id}`] as Player), isBot: true, connected: true };
  return new Tx(game).set(`players/${id}`, player).event(`🤖 ${name} joined to practise`, 'gift').plan();
}

/** The bots leave (the host's button). */
export function removeBots(game: Game): Plan {
  if (game.currentTurn) return refuse('Finish the practice game first');
  const bots = botIds(game);
  if (!bots.length) return refuse('No bots here');
  const tx = new Tx(game);
  for (const id of bots) tx.set(`players/${id}`, null);
  return tx.event('🤖 The practice bots left', 'gift').plan();
}

/** Is there anything on that a bot plays (the room server keeps its tick going while there is)? */
export function botsBusy(game: Game): boolean {
  const bots = botIds(game);
  if (!bots.length) return false;
  const a = arenaOf(game);
  if (a?.lifecycle === 'active' && bots.some((id) => a.participants.includes(id))) return true;
  const mg = game.minigame as PartyGame | undefined;
  return !!mg && mg.type in GAMES && (mg.phase === 'playing' || mg.phase === 'voting') && bots.some((id) => mg.participants?.includes(id));
}

/** How good this bot is at this game, 0 to 1 (the same all game long). */
function knack(id: PlayerId, seed: number): number {
  let h = seed | 0;
  for (const ch of id) h = (Math.imul(h, 31) + ch.charCodeAt(0)) | 0;
  return ((h >>> 0) % 1000) / 1000;
}

/** What one bot does on this tick, if anything. */
export function botMove(game: Game, id: PlayerId, ctx: { now: number; rng: Rng }): Plan | null {
  if (!isBot(game, id)) return null;
  const a = arenaOf(game);
  if (a?.lifecycle === 'active' && a.participants.includes(id)) {
    if (ctx.rng() >= (PACE[a.gameId] ?? 0.2)) return null;
    const can = arenaOffers(game, id, ctx.now, ctx.rng) as { kind?: string; variant?: number }[];
    if (!can.length) return null;
    let action = can[Math.floor(ctx.rng() * can.length)];
    if (action.kind === 'fit' && typeof action.variant === 'number' && ctx.rng() < SLIP) action = { ...action, variant: (action.variant + 1) % 3 };
    const plan = arenaAct(game, id, { action }, ctx);
    return plan.error ? null : plan;
  }
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || !(mg.type in GAMES) || !mg.participants?.includes(id)) return null;
  if (mg.type === 'whack_elf' && mg.phase === 'playing') {
    const mine = mg.scores?.[id];
    if (mine?.final) return null;
    const over = ctx.now >= mg.phaseEndsAt;
    if (!over && ctx.rng() >= 0.15) return null;
    // a bot gets a steady share of what the pop-ups so far could have scored
    const elapsed = Math.max(0, ctx.now - (mg.playStartAt ?? ctx.now));
    const score = Math.round(whackBounds(mg.seed, elapsed - 1500).max * (0.3 + 0.4 * knack(id, mg.seed)));
    if (!over && score === (mine?.score ?? 0)) return null;
    return reportWhack(game, id, Math.max(score, mine?.score ?? 0), over, ctx.now);
  }
  if (mg.type === 'sketch_off' && mg.phase === 'playing' && !mg.submissions?.[id]) {
    return ctx.now > (mg.playStartAt ?? 0) + 4000 && ctx.rng() < 0.03 ? submitSketch(game, id, doodle(ctx.rng)) : null;
  }
  if (mg.type === 'sketch_off' && mg.phase === 'voting' && !mg.votes?.[id]) {
    const others = Object.keys(mg.submissions || {}).filter((p) => p !== id);
    return others.length && ctx.rng() < 0.05 ? voteSketch(game, id, others[Math.floor(ctx.rng() * others.length)]) : null;
  }
  return null;
}
