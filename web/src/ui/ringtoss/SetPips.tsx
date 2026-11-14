// Ring Toss: a player's set so far, on the TV's board and on their phone.
import type { RingView } from '../../core/minigames/ringtoss/view';

/** A player's set so far: a pip per target (lit once ringed since their last set), and their full sets. */
export function SetPips({ v, pid, color }: { v: RingView; pid: string; color: string }) {
  const got = v.got[pid] || [], sets = v.sets[pid] || 0;
  return (
    <span className="rt-set" aria-label={`Set: ${got.length} of ${v.pegs.length}${sets ? `, ${sets} full ${sets === 1 ? 'set' : 'sets'}` : ''}`} data-testid={`ringtoss-set-${pid}`}>
      {v.pegs.map((p) => <i key={p.id} className={`${p.id === 'golden' ? 'hat' : ''}${got.includes(p.id) ? ' on' : ''}`} style={got.includes(p.id) ? { background: color, borderColor: color } : undefined} />)}
      {sets > 0 && <b className="rt-sets">×{sets}</b>}
    </span>
  );
}
