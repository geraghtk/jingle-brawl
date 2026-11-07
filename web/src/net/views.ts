// A screen's views, rebuilt from what the room sent it (a whole view, then patches): what the
// screen had on show after each message. Tests use it to check every view a screen ever had.
import { applyUpdates } from '../core/tx';
import type { Game } from '../core/types';
import type { ServerMsg } from './protocol';

export function viewsFrom(msgs: ServerMsg[]): Game[] {
  const out: Game[] = [];
  let cur: Game | null = null;
  for (const m of msgs) {
    if (m.t === 'state') cur = m.game;
    else if (m.t === 'patch' && cur) cur = applyUpdates(cur, m.updates, m.now);
    else continue;
    out.push(cur);
  }
  return out;
}
