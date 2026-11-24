// What each screen may know. The room server sends every viewer its own projection of the game,
// so a secret never reaches a screen that shouldn't show it (today any phone could read the
// whole game). A projection has the same shape as the game, so the screens don't change; what a
// viewer may not see is missing or sealed:
//   - sealed bids: until the reveal, others' bids are { amount: 0, sealed: true } (the host too:
//     the host often plays; after the console's Peek, logged for the table, the host sees them)
//   - chips: with publicChips off, phones and the TV see only their own until the game ends
//   - AI Draw Off drawings: nobody sees the other's while drawing, the TV included (the duellists
//     can see the TV; who's done still shows). Sketch-Off: no phone sees another's; the TV and the
//     console show them growing on a wall, shuffled and nameless (`minigame.wall`)
//   - accounts and Undo: ownerUid / hostUid never leave the server; Undo is the host's
import { arenaView } from './minigames/runtime';
import { wallOrder, wallSlot } from './party';
import type { PartyGame } from './party';
import { config } from './settings';
import { duelStakes } from './stakes';
import type { Game, PlayerId } from './types';

export type Viewer = { kind: 'host' } | { kind: 'tv' } | { kind: 'phone'; pid: PlayerId };

const SEALED_PHASES = new Set(['bidding', 'endgame_bidding']);
export const HIDDEN_DRAWING = 'data:,';

/** The duelist an in-app game's ending is announcing as the duel's winner right now, if any. */
export function announcedWinner(game: Game): PlayerId | null {
  const t = game.currentTurn;
  if (!t || t.lab || !t.challengerId || !t.defenderId) return null;
  const duelist = (id: PlayerId | undefined | null) => (id && (id === t.challengerId || id === t.defenderId) ? id : null);
  const a = game.arena;
  if (a) {
    if (t.phase !== 'minigame_arena' || a.practice || a.endedAt == null || a.result?.status !== 'resolved') return null;
    const w = a.result.duel?.winningDuelistId;
    return w && a.duelists?.includes(w) ? duelist(w) : null;
  }
  const mg = game.minigame;
  if (!mg || !('type' in mg)) return null;
  if (mg.type === 'draw_off') return t.phase === 'minigame_draw_off' && mg.phase === 'results' ? duelist(mg.winners?.[0]) : null;
  const party = mg as PartyGame;
  return t.phase === 'minigame_party' && party.minigameGoal === 'duel' && party.phase === 'results' ? duelist(party.results?.winners?.[0]) : null;
}

export function projectGame(game: Game, viewer: Viewer, now = Date.now()): Game {
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

  if (g.arena) { g.arenaView = arenaView(g.arena, viewer, now); delete g.arena; }   // a game's secrets stay with it

  // an in-app game has named the duel's winner: its ending says what they take too (the room knows every
  // chip, so it works that out here, from the same resolveDuel that then settles it). Never for a practice
  // round or the Game Lab: they hand nothing off, so they promise nothing.
  const winner = announcedWinner(game);
  if (winner) {
    const stakes = duelStakes(game, winner, now);
    if (stakes && g.arenaView) g.arenaView.stakes = stakes;
    else if (stakes && g.minigame) g.minigame.stakes = stakes;
  }

  const mg = g.minigame;
  const drawing = mg && 'type' in mg && ((mg.type === 'sketch_off' && mg.phase === 'playing')
    || (mg.type === 'draw_off' && (mg.phase === 'prompt_reveal' || mg.phase === 'drawing')));
  // Sketch-Off: the TV and the console get the drawings as they grow, shuffled and without names
  // (Kevin, 2026-11-18: fun to watch); phones get nobody's but their own. AI Draw Off stays hidden.
  if (mg && 'type' in mg && mg.type === 'sketch_off') {
    const party = mg as PartyGame;
    if (party.phase === 'playing' && viewer.kind !== 'phone') {
      const wall: NonNullable<PartyGame['wall']> = {};
      wallOrder(party).forEach((id, i) => {
        const img = party.submissions?.[id] || party.live?.[id];
        if (img) wall[wallSlot(i)] = { img, ...(party.submissions?.[id] ? { done: true } : {}) };
      });
      party.wall = wall;
    }
    delete party.live;
  }
  if (drawing && mg.submissions) {
    for (const id of Object.keys(mg.submissions)) if (id !== me) mg.submissions[id] = HIDDEN_DRAWING;
  }
  return g;
}
