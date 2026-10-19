// What every host (TV) component needs: the live game, the settings, act(), and the event log.
import { createContext, useCallback, useContext, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { checkEndgameContinue, drawEndgamePlayer } from '../core/duel';
import { rng } from '../core/rng';
import { config } from '../core/settings';
import type { Config } from '../core/settings';
import type { EventKind, Game, Plan } from '../core/types';
import { applyPlan } from '../firebase/game';

export interface LogEntry { text: string; kind: EventKind | ''; at: number }

export interface HostCtx {
  room: string;
  game: Game;
  cfg: Config;
  /** Build a plan from the CURRENT game and apply it (one atomic write). Returns false if refused / claim lost. */
  act: (build: (game: Game) => Plan, opts?: { quiet?: boolean }) => Promise<boolean>;
  log: LogEntry[];
  addEvent: (text: string, kind?: EventKind | '') => void;
  debug: boolean;
  setDebug: (on: boolean) => void;
  /** players + gifts before the last duel was resolved (Admin → Undo). */
  snapshot: Pick<Game, 'players' | 'gifts'> | null;
  takeSnapshot: () => void;
  clearSnapshot: () => void;
}

const Ctx = createContext<HostCtx | null>(null);

export function useHost(): HostCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('useHost outside <HostProvider>');
  return v;
}

const freshRng = () => rng(Date.now() ^ Math.floor(Math.random() * 1e9));

export function HostProvider({ room, game, children }: { room: string; game: Game; children: ReactNode }) {
  const latest = useRef(game);
  latest.current = game;
  const [log, setLog] = useState<LogEntry[]>([{ text: 'Game created! Share the room code with players.', kind: '', at: Date.now() }]);
  const [debug, setDebug] = useState(false);
  const [snapshot, setSnapshot] = useState<HostCtx['snapshot']>(null);
  const addEvent = useCallback((text: string, kind: EventKind | '' = '') => setLog((l) => [{ text, kind, at: Date.now() }, ...l].slice(0, 300)), []);

  const act = useCallback(async (build: (g: Game) => Plan, opts?: { quiet?: boolean }) => {
    const plan = build(latest.current);
    if (plan.error) { if (!opts?.quiet) addEvent(`❌ ${plan.error}`, 'bid'); return false; }
    const ok = await applyPlan(room, plan);
    if (!ok) return false;
    for (const e of plan.events) addEvent(e.text, e.kind);
    // Endgame follow-ups need the state the write produced: give the listener a moment
    if (plan.then) {
      setTimeout(() => {
        const next = plan.then === 'drawEndgamePlayer' ? drawEndgamePlayer : checkEndgameContinue;
        act((g) => next(g, freshRng()), opts);
      }, 500);
    }
    return true;
  }, [room, addEvent]);

  const takeSnapshot = useCallback(() => {
    const g = latest.current;
    setSnapshot(JSON.parse(JSON.stringify({ players: g.players || {}, gifts: g.gifts || {} })));
  }, []);

  return (
    <Ctx.Provider value={{
      room, game, cfg: config(game.settings), act, log, addEvent, debug, setDebug,
      snapshot, takeSnapshot, clearSnapshot: () => setSnapshot(null),
    }}>
      {children}
    </Ctx.Provider>
  );
}

export { freshRng };
