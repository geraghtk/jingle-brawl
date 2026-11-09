// The Game Lab: try any registered mini-game from the lobby, on the real phones and TV, with
// nothing at stake. No gift, no bids, no duel, no chips or Party points: the host picks the game,
// who plays, the mode and (where the mode has them) the two duelists or captains. The game runs
// on the same runtime as in a duel; the result stays on every screen until the host plays again
// or goes back to the lobby. The real game can't start while a lab game is on.
import type { Rng } from '../rng';
import { config, gremlinKinds } from '../settings';
import { refuse, Tx } from '../tx';
import type { Game, Plan, PlayerId } from '../types';
import type { GameDefinition, MatchMode } from './contracts';
import { assignTeams } from './engines/teams';
import { getDef } from './registry';
import { arenaOf, startArena } from './runtime';

export interface LabSetup { gameId: string; players: PlayerId[]; mode?: MatchMode; duelists?: PlayerId[] }

/** Modes that name two duelists (or captains). */
export const NEEDS_DUELISTS: MatchMode[] = ['direct_duel', 'party_with_duelists', 'captain_teams'];

/** The modes a game can be tried in with `n` players. */
export function labModes(def: GameDefinition, n: number): MatchMode[] {
  return def.supportedModes.filter((m) => {
    if (m === 'direct_duel') return n === 2;
    if (m === 'party_with_duelists') return n >= 3;
    if (m === 'captain_teams') return n >= 2;
    return n >= 1;   // free_for_all, cooperative: one player can practise alone
  });
}

/** The mode a game is played in for real with `n` players (the lab's default). */
export function defaultLabMode(def: GameDefinition, n: number): MatchMode | undefined {
  const modes = labModes(def, n);
  for (const m of ['captain_teams', 'direct_duel', 'party_with_duelists', 'free_for_all', 'cooperative'] as MatchMode[]) if (modes.includes(m)) return m;
  return undefined;
}

export const inLab = (game: Game) => !!game.currentTurn?.lab;

/** A lab game is on (running or paused): its result isn't in yet. */
export function labRunning(game: Game): boolean {
  const a = arenaOf(game);
  return inLab(game) && !!a && (a.lifecycle === 'active' || a.lifecycle === 'paused');
}

/** Start (or start again) a lab game. Only in the lobby, and not over one that's still going. */
export function startLab(game: Game, setup: LabSetup, ctx: { now: number; rng: Rng }): Plan {
  if (game.phase !== 'lobby') return refuse('The Game Lab is open in the lobby only');
  if (game.currentTurn && !inLab(game)) return refuse('Something else is going on');
  if (labRunning(game)) return refuse('A game is still running: stop it first');
  const def = getDef(setup.gameId);
  if (!def) return refuse('No such game');
  const players = [...new Set(setup.players)].filter((id) => game.players?.[id]);
  if (players.length !== setup.players.length || !players.length) return refuse('Pick players who are in the room');
  const mode = setup.mode ?? defaultLabMode(def, players.length);
  if (!mode || !labModes(def, players.length).includes(mode)) return refuse(`${def.label} can't be played like that with ${players.length} ${players.length === 1 ? 'player' : 'players'}`);
  let duelists: [PlayerId, PlayerId] | undefined;
  if (NEEDS_DUELISTS.includes(mode)) {
    const d = setup.duelists?.length ? setup.duelists : players.slice(0, 2);
    if (d.length !== 2 || d[0] === d[1] || !d.every((id) => players.includes(id))) return refuse('Pick two different players to duel (or captain)');
    duelists = [d[0], d[1]];
  }
  const participants = mode === 'direct_duel' ? duelists! : players;
  let cfg: unknown;
  if (def.id === 'pass_the_gremlin') cfg = { challenges: gremlinKinds(config(game.settings).gremlinChallenges) };
  if (mode === 'captain_teams') cfg = { teams: assignTeams(duelists!, players, ctx.rng, { pairs: {} }) };   // the lab doesn't remember teammates

  const tx = new Tx(game).set('arena', null)
    .set('currentTurn', { phase: 'minigame_arena', lab: { gameId: def.id, players, mode, ...(duelists ? { duelists } : {}) } });
  let started: { error?: string };
  try { started = startArena(tx, def.id, { mode, participants, duelists, config: cfg }, ctx); }
  catch (e) { started = { error: `${def.label} couldn't start: ${(e as Error).message}` }; }
  if (started.error) return refuse(started.error);
  return tx.event(`🧪 Game Lab: practice, nothing is won or lost`, 'duel').plan();
}

/** Back to the lobby (stops a game that's still on). */
export function endLab(game: Game): Plan {
  if (!inLab(game)) return refuse('No Game Lab game');
  return new Tx(game).set('arena', null).set('currentTurn', null).event('🧪 Back to the lobby', 'duel').plan();
}
