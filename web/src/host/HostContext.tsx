// What every host (TV) component needs: the live game, the settings, send(), and the event log.
// Buttons send intents (core/intents.ts); act() runs the host page's own automatic steps.
import { createContext, useCallback, useContext, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { checkEndgameContinue, drawEndgamePlayer } from '../core/duel';
import { dispatch } from '../core/intents';
import type { Intent } from '../core/intents';
import { rng } from '../core/rng';
import { config } from '../core/settings';
import { HOST } from '../core/table';
import type { Config } from '../core/settings';
import type { EventKind, Game, Plan } from '../core/types';
import { applyPlan, useServerOffset } from '../firebase/game';

export interface LogEntry { text: string; kind: EventKind | ''; at: number }

export interface HostCtx {
  room: string;
  game: Game;
  cfg: Config;
  /** A button: the console asks for something; refused unless it's on the console now. Returns false if refused / claim lost. */
  send: (intent: Intent, opts?: { quiet?: boolean }) => Promise<boolean>;
  /** Build a plan from the CURRENT game and apply it (one atomic write): the page's automatic steps only. */
  act: (build: (game: Game) => Plan, opts?: { quiet?: boolean }) => Promise<boolean>;
  log: LogEntry[];
  addEvent: (text: string, kind?: EventKind | '') => void;
  debug: boolean;
  setDebug: (on: boolean) => void;
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

  const offset = useRef(0);
  offset.current = useServerOffset();
  const send = useCallback((intent: Intent, opts?: { quiet?: boolean }) =>
    act((g) => dispatch(g, HOST, intent, { now: Date.now() + offset.current, rng: freshRng() }), opts), [act]);

  return (
    <Ctx.Provider value={{ room, game, cfg: config(game.settings), send, act, log, addEvent, debug, setDebug }}>
      {children}
    </Ctx.Provider>
  );
}

export { freshRng };
