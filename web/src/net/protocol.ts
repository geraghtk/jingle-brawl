// The wire between the room server and the screens (one WebSocket per screen). The server is the
// referee: screens send intents, the server sends each screen its own view of the game.
import type { Intent } from '../core/intents';
import type { GameEvent, Game, PlayerId } from '../core/types';

export const PROTOCOL = 2;
export type ViewerKind = 'host' | 'tv' | 'phone';

/** Screen → server. */
export type ClientMsg =
  | { t: 'hello'; v: number; room: string; token?: string }          // no token = the TV
  | { t: 'intent'; id: string; intent: Intent }                      // id: unique per press (resends reuse it)
  | { t: 'ping'; c: number };                                        // c: the screen's clock

/** Server → screen. */
export type ServerMsg =
  | { t: 'welcome'; viewer: ViewerKind; pid?: PlayerId }
  | { t: 'state'; version: number; game: Game; now: number }          // this screen's whole view (on connecting)
  | { t: 'patch'; base: number; version: number; updates: Record<string, unknown>; now: number } // what changed in it since `base`
  | { t: 'ack'; id: string; ok: boolean; error?: string }
  | { t: 'log'; events: (GameEvent & { at: number })[] }             // the host console's event log
  | { t: 'pong'; c: number; s: number }                              // s: the server's clock
  | { t: 'error'; error: string; fatal?: boolean };

/** HTTP: create a room (the host), check one, join one (a phone). */
export interface CreateRoomRes { code: string; token: string }
export interface LobbyRes { exists: boolean; phase?: Game['phase']; names?: string[]; avatars?: string[] }
export interface JoinReq { name: string; age?: number | null; avatar?: string | null }
export interface JoinRes { pid: PlayerId; token: string }
