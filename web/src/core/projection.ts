// What each screen may know. The room server sends every viewer its own projection of the game,
// so a secret never reaches a screen that shouldn't show it (today any phone could read the
// whole game). A projection has the same shape as the game, so the screens don't change; what a
// viewer may not see is missing or sealed:
//   - sealed bids: until the reveal, others' bids are { amount: 0, sealed: true } (the host too:
//     the host often plays; after the console's Peek, logged for the table, the host sees them)
//   - chips: with publicChips off, phones and the TV see only their own until the game ends
//   - Sketch-Off and AI Draw Off drawings: nobody sees another's while drawing, the TV included
//     (the duellists can see the TV; who's done still shows)
//   - accounts and Undo: ownerUid / hostUid never leave the server; Undo is the host's
import { config } from './settings';
import type { Game, PlayerId } from './types';

export type Viewer = { kind: 'host' } | { kind: 'tv' } | { kind: 'phone'; pid: PlayerId };

const SEALED_PHASES = new Set(['bidding', 'endgame_bidding']);
export const HIDDEN_DRAWING = 'data:,';

export function projectGame(game: Game, viewer: Viewer): Game {
  const g: Game = JSON.parse(JSON.stringify(game));
  const me = viewer.kind === 'phone' ? viewer.pid : null;
  delete g.hostUid;
  if (viewer.kind !== 'host') delete g.undo;
  for (const p of Object.values(g.undo?.players || {})) delete p.ownerUid;

  const hideChips = !config(g.settings).publicChips && g.phase !== 'ended' && viewer.kind !== 'host';
  for (const [id, p] of Object.entries(g.players || {})) {
    delete p.ownerUid;
    if (hideChips && id !== me) delete (p as { chips?: number }).chips;
  }

  const t = g.currentTurn;
  if (t?.bids && SEALED_PHASES.has(t.phase) && !(viewer.kind === 'host' && t.hostPeeked)) {
    for (const [id, b] of Object.entries(t.bids)) {
      if (id !== me) t.bids[id] = { amount: 0, sealed: true, ...(b.submittedAt != null ? { submittedAt: b.submittedAt } : {}) };
    }
  }

  const mg = g.minigame;
  const drawing = mg && 'type' in mg && ((mg.type === 'sketch_off' && mg.phase === 'playing')
    || (mg.type === 'draw_off' && (mg.phase === 'prompt_reveal' || mg.phase === 'drawing')));
  if (drawing && mg.submissions) {
    for (const id of Object.keys(mg.submissions)) if (id !== me) mg.submissions[id] = HIDDEN_DRAWING;
  }
  return g;
}
