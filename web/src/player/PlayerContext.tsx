// What every phone screen needs: the live game, who I am, the settings, and act().
import { createContext, useCallback, useContext, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { config } from '../core/settings';
import type { Config } from '../core/settings';
import type { Game, Plan, Player } from '../core/types';
import { applyPlan } from '../firebase/game';

export interface PlayerCtx {
  room: string;
  me: string;
  game: Game;
  cfg: Config;
  player: Player;
  /** Build a plan from the CURRENT game and apply it; refusals show as a toast. */
  act: (build: (game: Game) => Plan) => Promise<boolean>;
  toast: (msg: string) => void;
}

const Ctx = createContext<PlayerCtx | null>(null);

export function usePlayer(): PlayerCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('usePlayer outside <PlayerProvider>');
  return v;
}

export function PlayerProvider({ room, me, game, toast, children }: { room: string; me: string; game: Game; toast: (m: string) => void; children: ReactNode }) {
  const latest = useRef(game);
  latest.current = game;
  const act = useCallback(async (build: (g: Game) => Plan) => {
    const plan = build(latest.current);
    if (plan.error) { toast(plan.error); return false; }
    return applyPlan(room, plan);
  }, [room, toast]);
  const player = game.players?.[me];
  if (!player) return null;
  return <Ctx.Provider value={{ room, me, game, cfg: config(game.settings), player, act, toast }}>{children}</Ctx.Provider>;
}

/** Toast state for the phone. */
export function useToast(): [string, boolean, (msg: string) => void] {
  const [msg, setMsg] = useState('');
  const [show, setShow] = useState(false);
  const timer = useRef<number>();
  const toast = useCallback((m: string) => {
    setMsg(m); setShow(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setShow(false), 3000);
  }, []);
  return [msg, show, toast];
}
