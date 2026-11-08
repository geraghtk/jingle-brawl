// Which duel the two duellists actually play. The host lists the out-of-game duels at setup
// (rock-paper-scissors, thumb war…) and ticks which in-app mini-games may come up. Then, for
// each duel, either the defender picks one or the wheel picks — always the wheel when nobody
// is really defending (a yield duel between two challengers, a tiebreak). The V2 Under-16
// Exception overrides both: a duellist under 16 always picks (both under 16: the youngest; a
// tie: the defender). Players give an age only if they tick "under 16". Voting opens once
// the table knows what's being played. An in-app game settles the duel itself: everyone plays,
// the duellist who places higher wins, the rest earn party points as usual.
import { startDrawOff } from './drawoff';
import { getDef } from './minigames/registry';
import { startArena } from './minigames/runtime';
import { assignTeams, recordPairs } from './minigames/engines/teams';
import { GAMES } from './party';
import type { PartyType } from './party';
import { startPartyGame } from './partyflow';
import type { Rng } from './rng';
import { config, gremlinKinds } from './settings';
import { refuse, SERVER_TIME, Tx } from './tx';
import type { Game, Plan, PlayerId, Turn } from './types';
import { voteStatus } from './voting';

export { DEFAULT_DUEL_GAMES } from './settings';
export const SPIN_MS = 4800;

/** The in-app duel games: the party games (everyone plays), AI Draw Off (the two duelists draw), and
 *  any registered runtime game (core/minigames, by its id). */
export type DuelApp = PartyType | 'draw_off' | (string & {});
const BUILT_IN: Record<string, { label: string; emoji: string }> = { ...GAMES, draw_off: { label: 'AI Draw Off', emoji: '🎨' } };
export function duelAppInfo(app: DuelApp | undefined): { label: string; emoji: string } | undefined {
  if (!app) return undefined;
  return BUILT_IN[app] ?? getDef(app);
}

export interface DuelGameOption { name: string; kind: 'table' | 'app'; app?: DuelApp }
export interface DuelGameChoice extends DuelGameOption { via: 'picker' | 'wheel' }
export interface WheelSpin { options: DuelGameOption[]; index: number; spunAt: number | typeof SERVER_TIME }

/** Everything the duel could be: the host's list + the in-app games they ticked. */
export function duelOptions(game: Game): DuelGameOption[] {
  const cfg = config(game.settings);
  const table = cfg.duelGames.map((name) => ({ name, kind: 'table' as const }));
  // not for a tiebreak duel: its winner goes on to the real duel rather than being settled like one
  const apps = game.currentTurn?.isTiebreak ? []
    : cfg.duelAppGames.filter((a) => duelAppInfo(a)).map((app) => ({ name: duelAppInfo(app)!.label, kind: 'app' as const, app }));
  return [...table, ...apps];
}

export const UNDER = 16;
const youngAge = (game: Game, id?: PlayerId) => {
  const age = id ? game.players?.[id]?.age : undefined;
  return typeof age === 'number' && age < UNDER ? age : null;
};

/** Who picks this duel's game (null = the wheel). Under-16s first, then the house rule. */
export function duelPicker(game: Game): PlayerId | null {
  const t = game.currentTurn;
  if (!t?.challengerId || !t.defenderId) return null;
  const c = youngAge(game, t.challengerId), d = youngAge(game, t.defenderId);
  if (c != null && (d == null || c < d)) return t.challengerId;
  if (d != null) return t.defenderId;
  const cfg = config(game.settings);
  return cfg.duelPick === 'wheel' || t.isYieldDuel || t.isTiebreak ? null : t.defenderId;
}

/** Does this duel still need its game, and who decides it? null = nothing to decide. */
export function duelGameStep(game: Game): null | 'picker' | 'wheel' {
  const t = game.currentTurn;
  const cfg = config(game.settings);
  if (!t || t.phase !== 'duel' || t.duelGame || cfg.duelPick === 'off' || !duelOptions(game).length) return null;
  return duelPicker(game) ? 'picker' : 'wheel';
}

/** The picker (the defender, or an under-16 duellist) picks from the list — or asks for a spin. */
export function chooseDuelGame(game: Game, pid: PlayerId, index: number): Plan {
  if (duelGameStep(game) !== 'picker' || duelPicker(game) !== pid) return refuse('Not your pick');
  const option = duelOptions(game)[index];
  if (!option) return refuse('Pick one of the duels');
  const choice: DuelGameChoice = { ...option, via: 'picker' };
  return new Tx(game).claim('currentTurn/duelGame', choice)
    .event(`🎯 ${name(game, pid)} picks ${option.name}`, 'duel').plan();
}

/** Spin the wheel (the host when the wheel decides; or the picker's "let the wheel pick"). */
export function spinDuelGame(game: Game, ctx: { now: number; rng: Rng }, by?: PlayerId): Plan {
  const step = duelGameStep(game);
  if (!step) return refuse('Nothing to spin for');
  if (by && (step !== 'picker' || duelPicker(game) !== by)) return refuse('Not your pick');
  const options = duelOptions(game);
  const index = Math.floor(ctx.rng() * options.length);
  const choice: DuelGameChoice = { ...options[index], via: 'wheel' };
  const wheel: WheelSpin = { options, index, spunAt: ctx.now };
  return new Tx(game).claim('currentTurn/duelGame', choice).set('currentTurn/wheel', wheel)
    .event(`🎡 The wheel spins… ${options[index].name}!`, 'duel').plan();
}

/** An in-app duel game: the host starts it once the table has voted. */
export function startDuelApp(game: Game, ctx: { now: number; rng: Rng }): Plan {
  const t = game.currentTurn;
  if (!t || t.phase !== 'duel' || t.duelGame?.kind !== 'app' || !t.duelGame.app) return refuse('No mini-game to start');
  if (voteStatus(game, ctx.now).blocking && t.voteEndsAt != null) return refuse('Waiting for votes');
  const tx = new Tx(game).claim('currentTurn/duelAppStartedAt', ctx.now).set('currentTurn/voteClosed', true);
  const duelists: [PlayerId, PlayerId] = [t.challengerId!, t.defenderId!];
  if (t.duelGame.app === 'draw_off') startDrawOff(tx, duelists, ctx);
  else if (getDef(t.duelGame.app)) {
    // a runtime game: everyone plays if it can resolve the duelists that way (3+ at the table), else just the two
    const def = getDef(t.duelGame.app)!;
    const everyone = Object.keys(game.players || {}).filter((id) => game.players![id].connected !== false);
    const party = everyone.length >= 3 && def.supportedModes.includes('party_with_duelists');
    const arenaConfig = def.id === 'pass_the_gremlin' ? { challenges: gremlinKinds(config(game.settings).gremlinChallenges) } : undefined;
    let started: { error?: string };
    if (def.supportedModes.includes('captain_teams')) {
      // a team game: the duelists captain, the engine deals everyone else out (splitting past teammates)
      const table = [...new Set([...duelists, ...everyone])];
      const deal = assignTeams(duelists, table, ctx.rng, { pairs: game.teamPairs || {} });
      tx.set('teamPairs', recordPairs(deal.teams, game.teamPairs || {}));
      started = startArena(tx, def.id, { mode: 'captain_teams', participants: table, duelists, config: { teams: deal } }, ctx);
    } else {
      started = startArena(tx, def.id, { mode: party ? 'party_with_duelists' : 'direct_duel', participants: party ? everyone : duelists, duelists, config: arenaConfig }, ctx);
    }
    if (started.error) return refuse(started.error);
  } else startPartyGame(tx, t.duelGame.app as PartyType, 'duel', duelists, ctx);
  return tx.plan();
}

/** The in-app game says who won; the host resolves the duel with it. */
export const duelAppWinner = (t: Turn | null | undefined) => (t?.phase === 'duel' ? t.duelGameWinner ?? null : null);

const name = (game: Game, id: PlayerId) => game.players?.[id]?.name || 'Someone';
