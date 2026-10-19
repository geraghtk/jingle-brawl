// The phone remembers which game it joined (per ?testId tab) so a refresh can rejoin.
import { storageKey } from '../firebase/app';

export interface Session { room: string; playerId: string; name: string }

const K = { room: 'jingleBrawl_player_room', id: 'jingleBrawl_player_id', name: 'jingleBrawl_player_name' };

export function loadSession(): Session | null {
  const room = localStorage.getItem(storageKey(K.room));
  const playerId = localStorage.getItem(storageKey(K.id));
  const name = localStorage.getItem(storageKey(K.name));
  return room && playerId && name ? { room, playerId, name } : null;
}

export function saveSession(s: Session) {
  localStorage.setItem(storageKey(K.room), s.room);
  localStorage.setItem(storageKey(K.id), s.playerId);
  localStorage.setItem(storageKey(K.name), s.name);
}

export function clearSession() {
  for (const k of Object.values(K)) localStorage.removeItem(storageKey(k));
}
