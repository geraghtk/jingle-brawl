// A countdown in whole seconds, as every screen shows it (the room's time: net/hooks useServerNow).

/** Seconds left until `endsAt`, rounded up (1 until the very end), never below 0. */
export const secsLeft = (endsAt: number, now: number): number => Math.max(0, Math.ceil((endsAt - now) / 1000));
