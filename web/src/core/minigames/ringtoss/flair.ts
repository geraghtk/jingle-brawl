// Ring Toss's show, worked out from what every screen already has (the throws, the scores, the
// clock): runs of ringers, who's ahead, the last seconds, the one target a set still needs. Nothing
// here scores or decides anything: the room's numbers are ringtoss.ts's.

export interface FlairThrow { n: number; pid: string; stage: string; pegId: string | null }

/** `pid`'s ringers in a row (scored rings only), counting back from their throw `upTo` (their latest
 *  if left out). A miss ends the run; practice rings don't count either way. */
export function streakAt(throws: FlairThrow[], pid: string, upTo = Infinity): number {
  let run = 0;
  for (let i = throws.length - 1; i >= 0; i--) {
    const t = throws[i];
    if (t.pid !== pid || t.n > upTo || t.stage === 'practice') continue;
    if (!t.pegId) break;
    run++;
  }
  return run;
}

/** A run worth shouting about starts at this many. */
export const STREAK_MIN = 3;

/** What the TV says about a run ('' below STREAK_MIN). */
export function streakCall(run: number): string {
  if (run < STREAK_MIN) return '';
  return run >= 5 ? `On fire! ${run} in a row!` : `${run} in a row!`;
}

/** A ringer's chime climbs with the run: a whole tone a ring, an octave and a bit at most. */
export function streakPitch(run: number): number {
  return 2 ** ((Math.min(Math.max(run, 1), 8) - 1) / 6);
}

/** The one player ahead of everyone (nobody while it's level, or before anyone scores). */
export function leaderOf(scores: Record<string, number>, ids: string[]): string | null {
  const best = Math.max(0, ...ids.map((id) => scores[id] || 0));
  const top = ids.filter((id) => (scores[id] || 0) === best);
  return best > 0 && top.length === 1 ? top[0] : null;
}

/** The scored round's last seconds, counted out loud: 5, 4, 3, 2, 1 (else null). */
export function finalCount(stage: string, endsAt: number, now: number): number | null {
  if (stage !== 'throwing') return null;
  const secs = Math.ceil((endsAt - now) / 1000);
  return secs >= 1 && secs <= 5 ? secs : null;
}

/** The targets in words. */
export const PEG_NAMES: Record<string, string> = {
  'left-outer': 'the far left tip', 'left-inner': 'the tall left tip', golden: 'the hat',
  'right-inner': 'the tall right tip', 'right-outer': 'the far right tip',
};

/** One target short of a full set: which one (else null). */
export function lastNeeded(pegs: { id: string }[], got: string[]): string | null {
  const need = pegs.filter((p) => !got.includes(p.id));
  return pegs.length > 1 && need.length === 1 ? need[0].id : null;
}
