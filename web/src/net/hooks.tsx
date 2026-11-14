// React glue for the room connection: useRoom (the live view of a room and its client),
// useServerOffset (the room server's clock) and useServerNow (its time, ticking).
import { createContext, useContext, useEffect, useState, useSyncExternalStore } from 'react';
import type { ReactNode } from 'react';
import type { Game } from '../core/types';
import { RoomClient } from './client';
import type { Status } from './client';

export interface RoomLink { client: RoomClient; game: Game | null; status: Status; offset: number; fatal: string | null; critic: RoomClient['critic'] }

/** Connect to a room for as long as the component lives (token: host or phone; none = the TV). */
export function useRoom(room: string, token?: string): RoomLink {
  const [client] = useState(() => new RoomClient(room, token));
  useEffect(() => { client.start(); return () => client.stop(); }, [client]);
  const snap = useSyncExternalStore(
    (fn) => client.subscribe(fn),
    () => `${client.version}|${client.status}|${client.offset}|${client.fatal}|${JSON.stringify(client.critic)}`,
  );
  void snap;
  return { client, game: client.game, status: client.status, offset: client.offset, fatal: client.fatal, critic: client.critic };
}

const OffsetCtx = createContext(0);

export function OffsetProvider({ offset, children }: { offset: number; children: ReactNode }) {
  return <OffsetCtx.Provider value={offset}>{children}</OffsetCtx.Provider>;
}

/** Server time minus this device's time (ms): every screen agrees when a party-game phase ends. */
export function useServerOffset(): number {
  return useContext(OffsetCtx);
}

/** The room's time (this device's clock + the offset), re-rendered every `ms`: a screen's countdown.
 *  Read afresh on every render, so a render for any other reason has the time as it is then. */
export function useServerNow(ms = 250): number {
  const offset = useServerOffset();
  const [, tick] = useState(0);
  useEffect(() => { const id = setInterval(() => tick((n) => n + 1), ms); return () => clearInterval(id); }, [ms]);
  return Date.now() + offset;
}
