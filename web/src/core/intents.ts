// Intents: everything a person can ask the game to do, as data. A phone's buttons and the host
// console's buttons are table offers (core/table.ts); the host's Admin tab adds the admin_*
// intents. parseIntent checks an intent that arrived over the network (shape and bounds);
// dispatch turns it into the plan the click runs, refusing anything not on that person's screen.
import * as A from './actions';
import * as Admin from './admin';
import type { Rng } from './rng';
import { HOST, perform } from './table';
import type { Offer, Who } from './table';
import { refuse } from './tx';
import { MISFIT } from './types';
import type { Game, Plan } from './types';

export type AdminIntent =
  | { type: 'admin_set_chips'; pid: string; chips: number }
  | { type: 'admin_add_gift'; name: string; ownerId: string }          // a player id or 'misfit'
  | { type: 'admin_save_gift'; giftId: string; name?: string; naughtyLevel: number; ownerId: string | null }
  | { type: 'admin_move_misfit'; giftId: string }
  | { type: 'admin_delete_gift'; giftId: string }
  | { type: 'admin_remove_player_gift'; pid: string }
  | { type: 'admin_reset_turn' }
  | { type: 'admin_undo_duel' }
  | { type: 'admin_end_game' };

export type Intent = Offer | AdminIntent;

// --- parsing ------------------------------------------------------------------------------------

type Field = 'id' | 'id?' | 'idOrNull' | 'text' | 'text?' | 'int' | 'int?' | 'bool?' | 'drawing?' | 'path';
const INT: Record<string, [number, number]> = {
  amount: [0, 999], index: [0, 50], score: [-1000, 1000], chips: [0, 9999], naughtyLevel: [0, 20], seed: [-(2 ** 31), 2 ** 32],
};
const SHAPES: Record<Intent['type'], Record<string, Field>> = {
  bid: { amount: 'int' }, keep: {}, gambit: { giftId: 'id' }, fight: {}, yield: {},
  pick_duel: { index: 'int', name: 'text?' }, spin: {}, call_duel: { pick: 'id' },
  reprisal: { giftId: 'id' }, skip_reprisal: {}, path: { path: 'path' }, steal: { giftId: 'id' }, cancel_steal: {},
  nobids_defender: { playerId: 'id' }, whack: { score: 'int?', final: 'bool?' }, sketch: { drawing: 'drawing?' },
  vote_sketch: { pick: 'id' },
  start: {}, draw: { giftName: 'text?', seed: 'int?' }, reveal: {}, skip_bidding: {}, start_duel: {}, start_yield_duel: {},
  host_spin: {}, close_voting: {}, start_duel_app: {}, duel_winner: { id: 'id' }, skip_party: {}, endgame: {},
  reveal_auction: {}, skip_auction: {}, auction_duel: {}, complete_drawoff: {}, end: {},
  admin_set_chips: { pid: 'id', chips: 'int' }, admin_add_gift: { name: 'text', ownerId: 'id' },
  admin_save_gift: { giftId: 'id', name: 'text?', naughtyLevel: 'int', ownerId: 'idOrNull' },
  admin_move_misfit: { giftId: 'id' }, admin_delete_gift: { giftId: 'id' }, admin_remove_player_gift: { pid: 'id' },
  admin_reset_turn: {}, admin_undo_duel: {}, admin_end_game: {},
};
const ID = /^[A-Za-z0-9_-]{1,64}$/;
const MAX_TEXT = 80;
const MAX_DRAWING = 600_000; // a phone's JPEG/PNG data URL

function fieldOk(kind: Field, key: string, v: unknown): boolean {
  const optional = kind.endsWith('?');
  if (v === undefined) return optional;
  switch (kind.replace('?', '')) {
    case 'id': return typeof v === 'string' && ID.test(v);
    case 'idOrNull': return v === null || (typeof v === 'string' && ID.test(v));
    case 'text': return typeof v === 'string' && v.trim().length > 0 && v.length <= MAX_TEXT;
    case 'int': {
      const [lo, hi] = INT[key] || [-(2 ** 31), 2 ** 31];
      return Number.isInteger(v) && (v as number) >= lo && (v as number) <= hi;
    }
    case 'bool': return typeof v === 'boolean';
    case 'drawing': return typeof v === 'string' && v.length <= MAX_DRAWING && /^data:(image\/(png|jpeg|webp)(;base64)?)?,/.test(v);
    case 'path': return v === 'steal' || v === 'auction';
    default: return false;
  }
}

/** Check an intent that came over the wire: a known type, exactly its fields, each within bounds. */
export function parseIntent(x: unknown): Intent | { error: string } {
  if (!x || typeof x !== 'object' || Array.isArray(x)) return { error: 'not an intent' };
  const o = x as Record<string, unknown>;
  const shape = typeof o.type === 'string' ? SHAPES[o.type as Intent['type']] : undefined;
  if (!shape) return { error: `unknown intent ${String(o.type).slice(0, 40)}` };
  for (const k of Object.keys(o)) if (k !== 'type' && !(k in shape)) return { error: `${o.type}: unexpected ${k.slice(0, 40)}` };
  for (const [k, kind] of Object.entries(shape)) if (!fieldOk(kind, k, o[k])) return { error: `${o.type}: bad ${k}` };
  return o as unknown as Intent;
}

export const isAdmin = (i: Intent): i is AdminIntent => i.type.startsWith('admin_');

// --- dispatch -----------------------------------------------------------------------------------

/** The plan an intent from `who` produces: table offers through perform(), admin intents for the host only. */
export function dispatch(game: Game, who: Who, intent: Intent, ctx: { now: number; rng: Rng }): Plan {
  if (!isAdmin(intent)) return perform(game, who, intent, ctx);
  if (who !== HOST) return refuse('Only the host can do that');
  switch (intent.type) {
    case 'admin_set_chips': return Admin.setChips(game, intent.pid, intent.chips);
    case 'admin_add_gift': return Admin.addGift(game, `gift_${ctx.now}`, intent.name.trim(), intent.ownerId === MISFIT ? MISFIT : intent.ownerId, ctx.now);
    case 'admin_save_gift': return Admin.saveGift(game, intent.giftId, { name: intent.name, naughtyLevel: intent.naughtyLevel, ownerId: intent.ownerId }, ctx.now);
    case 'admin_move_misfit': return Admin.moveGiftToMisfit(game, intent.giftId, ctx.now);
    case 'admin_delete_gift': return Admin.deleteGift(game, intent.giftId);
    case 'admin_remove_player_gift': return Admin.removePlayerGift(game, intent.pid, ctx.now);
    case 'admin_reset_turn': return Admin.resetTurn(game);
    case 'admin_undo_duel': return Admin.undoDuel(game);
    case 'admin_end_game': return A.endGame(game);
  }
}
