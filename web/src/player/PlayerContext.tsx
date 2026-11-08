// What every phone screen needs: the live game, who I am, the settings, and send() (a button's intent).
import { createContext, useCallback, useContext, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import type { Intent } from '../core/intents';
import { config } from '../core/settings';
import type { Config } from '../core/settings';
import type { Game, Player } from '../core/types';
import type { RoomClient } from '../net/client';

export interface PlayerCtx {
  room: string;
  me: string;
  game: Game;
  cfg: Config;
  player: Player;
  /** A button: ask for something (refused unless it's on my screen now); refusals show as a toast. */
  send: (intent: Intent) => Promise<boolean>;
  toast: (msg: string) => void;
}

const Ctx = createContext<PlayerCtx | null>(null);

export function usePlayer(): PlayerCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('usePlayer outside <PlayerProvider>');
  return v;
}

export function PlayerProvider({ room, me, game, toast, client, children }: { room: string; me: string; game: Game; toast: (m: string) => void; client: RoomClient; children: ReactNode }) {
  const send = useCallback(async (intent: Intent) => {
    const ack = await client.send(intent);                 // the room decides
    if (!ack.ok) toast(ack.error || 'Not now');
    return ack.ok;
  }, [toast, client]);
  const player = game.players?.[me];
  if (!player) return null;
  return <Ctx.Provider value={{ room, me, game, cfg: config(game.settings), player, send, toast }}>{children}</Ctx.Provider>;
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
