// What every host (TV) component needs: the live game, the settings, send() (a button's intent,
// decided by the room server), and the event log the room keeps.
import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { Intent } from '../core/intents';
import { config } from '../core/settings';
import type { Config } from '../core/settings';
import type { EventKind, Game } from '../core/types';
import type { RoomClient } from '../net/client';

export interface LogEntry { text: string; kind: EventKind | ''; at: number }

export interface HostCtx {
  room: string;
  game: Game;
  cfg: Config;
  /** A button: the console asks for something; refused unless it's on the console now. Returns false if refused / claim lost. */
  send: (intent: Intent, opts?: { quiet?: boolean }) => Promise<boolean>;
  client: RoomClient;
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

export function HostProvider({ room, game, client, children }: { room: string; game: Game; client: RoomClient; children: ReactNode }) {
  const [log, setLog] = useState<LogEntry[]>([{ text: 'Game created! Share the room code with players.', kind: '', at: Date.now() }]);
  const [debug, setDebug] = useState(false);
  const addEvent = useCallback((text: string, kind: EventKind | '' = '') => setLog((l) => [{ text, kind, at: Date.now() }, ...l].slice(0, 300)), []);

  const send = useCallback(async (intent: Intent, opts?: { quiet?: boolean }) => {
    const ack = await client.send(intent);                 // the room decides; its log arrives below
    if (!ack.ok && !opts?.quiet) addEvent(`❌ ${ack.error}`, 'bid');
    return ack.ok;
  }, [client, addEvent]);
  // the room's event log (it keeps the last 300; a reconnect sends them again)
  useEffect(() => client.onLog((events) => setLog((l) => {
    const seen = new Set(l.map((e) => `${e.at}|${e.text}`));
    const fresh = events.filter((e) => !seen.has(`${e.at}|${e.text}`)).map((e) => ({ ...e, kind: e.kind as EventKind }));
    return [...fresh.reverse(), ...l].slice(0, 300);
  })), [client]);

  return (
    <Ctx.Provider value={{ room, game, cfg: config(game.settings), send, client, log, addEvent, debug, setDebug }}>
      {children}
    </Ctx.Provider>
  );
}
