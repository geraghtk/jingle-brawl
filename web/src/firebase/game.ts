// Reading and writing games/{room}: one listener for the whole game (consistent snapshots),
// and plans applied as ONE atomic multi-path update after winning their claim.
import { get, onValue, ref, runTransaction, serverTimestamp, update } from 'firebase/database';
import { useEffect, useState } from 'react';
import type { Game, Plan } from '../core/types';
import { db } from './app';

export const gameRef = (room: string, path = '') => ref(db, `games/${room}${path ? `/${path}` : ''}`);

/** RTDB writes { '.sv': 'timestamp' } placeholders as-is; give it the SDK's sentinel. */
function withServerTime(v: unknown): unknown {
  if (v && typeof v === 'object' && !Array.isArray(v)) {
    const o = v as Record<string, unknown>;
    if (o['.sv'] === 'timestamp') return serverTimestamp();
    return Object.fromEntries(Object.entries(o).map(([k, x]) => [k, withServerTime(x)]));
  }
  return v;
}

/**
 * Apply a plan. Returns false if it was refused or lost its claim (double-tap, second tab,
 * a timer racing a button) — in which case nothing was written.
 */
export async function applyPlan(room: string, plan: Plan): Promise<boolean> {
  if (plan.error) return false;
  if (plan.claim) {
    const { committed } = await runTransaction(gameRef(room, plan.claim.path), (cur) => (cur == null ? plan.claim!.value : undefined));
    if (!committed) return false;
  }
  if (Object.keys(plan.updates).length) {
    await update(gameRef(room), withServerTime(plan.updates) as Record<string, unknown>);
  }
  return true;
}

export async function readGame(room: string): Promise<Game | null> {
  return (await get(gameRef(room))).val();
}

/** Live game state (null while loading or if the room doesn't exist). */
export function useGame(room: string | null): { game: Game | null; loading: boolean } {
  const [state, setState] = useState<{ game: Game | null; loading: boolean }>({ game: null, loading: !!room });
  useEffect(() => {
    if (!room) { setState({ game: null, loading: false }); return; }
    setState({ game: null, loading: true });
    return onValue(gameRef(room), (snap) => setState({ game: snap.val(), loading: false }));
  }, [room]);
  return state;
}

/** Server-clock offset (ms) so every device agrees when a party-game phase ends. */
export function useServerOffset(): number {
  const [offset, setOffset] = useState(0);
  useEffect(() => onValue(ref(db, '.info/serverTimeOffset'), (s) => setOffset(Number(s.val()) || 0)), []);
  return offset;
}
