// A phone opens the player page with a seat already saved (it reloaded, it woke up, the guest
// scanned the QR code again): what now? Straight back into the game, without a tap, unless there is
// a real question: the address names a different room than the saved seat.
// (A token the room refuses is found out when the phone connects: App.tsx's "Join a game".
//  Leave clears the saved seat, so there is nothing to resume.)
import type { Session } from './session';

export type Resume = 'enter' | 'ask' | 'forget';

export function resume(saved: Session, at: { room?: string | null; rejoin?: boolean; roomExists: boolean }): Resume {
  if (!at.roomExists || !saved.token) return 'forget';                  // the game is over and gone: a fresh join form
  if (at.rejoin) return 'enter';                                        // the host switching back from their console
  const asked = (at.room || '').trim().toUpperCase();
  if (asked && asked !== saved.room.toUpperCase()) return 'ask';        // a link to another party
  return 'enter';
}
