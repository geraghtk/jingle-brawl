// Which seats this device holds: the host's (room + host token) and, per ?testId tab, a player's
// (player/session.ts). The host plays too, from the same phone: each page links to the other.
import { loadSession } from '../player/session';

const K = { room: 'jingleBrawl_host_room', token: 'jingleBrawl_host_token' };

export function hostSeat(): { room: string; token: string } | null {
  const room = localStorage.getItem(K.room);
  const token = localStorage.getItem(K.token);
  return room && token ? { room, token } : null;
}
export function saveHostSeat(room: string, token: string) {
  localStorage.setItem(K.room, room);
  localStorage.setItem(K.token, token);
}
export function forgetHostSeat() {
  for (const k of Object.values(K)) localStorage.removeItem(k);
}
/** This device hosts that room. */
export const hosts = (room: string) => hostSeat()?.room === room;
/** This device (tab) plays in that room. */
export const playsIn = (room: string) => loadSession()?.room === room;
