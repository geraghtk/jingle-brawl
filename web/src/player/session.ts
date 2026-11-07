// The phone remembers which game it joined (per ?testId tab) so a refresh can rejoin.
import { storageKey } from '../net/storage';

/** token: the room server's session token (absent on Firebase). */
export interface Session { room: string; playerId: string; name: string; token?: string }

const K = { room: 'jingleBrawl_player_room', id: 'jingleBrawl_player_id', name: 'jingleBrawl_player_name', token: 'jingleBrawl_player_token' };

export function loadSession(): Session | null {
  const room = localStorage.getItem(storageKey(K.room));
  const playerId = localStorage.getItem(storageKey(K.id));
  const name = localStorage.getItem(storageKey(K.name));
  const token = localStorage.getItem(storageKey(K.token)) || undefined;
  return room && playerId && name ? { room, playerId, name, ...(token ? { token } : {}) } : null;
}

export function saveSession(s: Session) {
  localStorage.setItem(storageKey(K.room), s.room);
  localStorage.setItem(storageKey(K.id), s.playerId);
  localStorage.setItem(storageKey(K.name), s.name);
  if (s.token) localStorage.setItem(storageKey(K.token), s.token); else localStorage.removeItem(storageKey(K.token));
}

export function clearSession() {
  for (const k of Object.values(K)) localStorage.removeItem(storageKey(k));
}
