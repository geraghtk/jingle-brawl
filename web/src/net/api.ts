// The room server's HTTP side: make a room (the host), look at one and join it (a phone).
import type { Settings } from '../core/types';
import type { CreateRoomRes, JoinReq, JoinRes, LobbyRes } from './protocol';

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json' } });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((body as { error?: string }).error || `HTTP ${res.status}`);
  return body as T;
}

export const createRoom = (settings: Settings) => call<CreateRoomRes>('/api/rooms', { method: 'POST', body: JSON.stringify({ settings }) });
export const lobby = (room: string) => call<LobbyRes>(`/api/rooms/${encodeURIComponent(room)}/lobby`);
export const joinRoom = (room: string, req: JoinReq) => call<JoinRes>(`/api/rooms/${encodeURIComponent(room)}/join`, { method: 'POST', body: JSON.stringify(req) });
