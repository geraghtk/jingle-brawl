// Nutcracker Knockout on a phone: the controller. For the two duelists the whole screen is three big
// pads (eyes on the TV, thumbs on the phone): punch left, punch right, hold to block. The phone only
// says what was pressed; the room times every punch and decides what it did. Everyone else is told to
// watch the TV.
import { useEffect, useRef } from 'react';
import { robotView } from '../core/minigames/robots/view';
import { useServerNow } from '../net/hooks';
import { BUSY_WORDS, PART_NAMES, ROBOT_COLORS } from '../ui/robots/look';
import { buzz, play } from '../ui/sound';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/robots.css';

export function RobotCard() {
  const { game, me, send } = usePlayer();
  const v = robotView(game.arenaView!.view);
  const now = useServerNow(100);
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const seat = v.mySeat ?? (v.order[0] === me ? 0 : v.order[1] === me ? 1 : null);
  const mine = seat != null ? v.fighters[seat] : null;
  const theirs = seat != null ? v.fighters[seat === 0 ? 1 : 0] : null;

  // my robot was hit (a part went): the phone thumps
  const parts = useRef(mine?.lost.length ?? 0);
  useEffect(() => {
    const n = mine?.lost.length ?? 0;
    if (n > parts.current) { buzz(n >= v.maxHealth ? [200, 80, 300] : [70, 30, 70]); play('thud'); }
    parts.current = n;
  }, [mine?.lost.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const press = (action: Record<string, unknown>) => { void send({ type: 'arena', action }); };   // a press the room refuses ("Not yet") just doesn't happen
  const guarding = useRef(false);
  const guard = (on: boolean) => {
    if (guarding.current === on) return;
    guarding.current = on;
    press({ kind: 'guard', on });
  };
  // a keyboard works too (the console's own seat, a laptop): A and D punch, Space blocks
  const fighting = seat != null && v.stage === 'fighting';
  useEffect(() => {
    if (!fighting) return;
    const down = (e: KeyboardEvent) => {
      if (e.repeat) return;
      const k = e.key.toLowerCase();
      if (k === 'a' || k === 'arrowleft') press({ kind: 'punch', hand: 'L' });
      else if (k === 'd' || k === 'arrowright') press({ kind: 'punch', hand: 'R' });
      else if (k === ' ') { e.preventDefault(); guard(true); }
    };
    const up = (e: KeyboardEvent) => { if (e.key === ' ') guard(false); };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); guard(false); };
  }, [fighting]); // eslint-disable-line react-hooks/exhaustive-deps

  if (seat == null || !mine || !theirs) {
    const [a, b] = v.order;
    return (
      <div className="card nk-watch" data-testid="robot-card">
        <p className="nk-watch-title">{name(a)} vs {name(b)}</p>
        <p>Watch the TV: their robots are boxing. First to {v.roundsToWin} rounds takes it.</p>
        <p className="nk-watch-score" data-testid="robot-score">{v.wins[a] || 0} : {v.wins[b] || 0}</p>
      </div>
    );
  }

  const rival = v.order[seat === 0 ? 1 : 0];
  const busy = mine.busy && mine.busyUntil > now ? mine.busy : undefined;
  const ready = fighting && !busy && !mine.punch;
  const guardLeft = mine.guard ? Math.max(0, 1 - (now - mine.guard.since) / v.guardMaxMs) : 1;
  const word = v.stage === 'intro' ? `Round ${v.round}: get ready`
    : v.stage === 'ko' ? (v.last?.winner === me ? (v.last.how === 'knockout' ? 'You took the head off!' : 'Your round, on time') : (v.last?.how === 'knockout' ? 'You lost your head' : 'Their round, on time'))
    : v.stage === 'done' ? ((v.wins[me] || 0) > (v.wins[rival] || 0) ? 'You win the fight!' : (v.wins[me] || 0) < (v.wins[rival] || 0) ? `${name(rival)} wins the fight` : 'Level: the host decides')
    : busy ? BUSY_WORDS[busy] : mine.guard ? 'Blocking' : theirs.punch ? 'Punch coming!' : v.sudden ? 'Sudden death: one hit' : '';
  const lastLost = mine.lost.at(-1);
  const pad = (hand: 'L' | 'R') => (
    <button type="button" className={`nk-pad punch ${hand === 'L' ? 'left' : 'right'}${mine.punch?.hand === hand ? ' swinging' : ''}`} disabled={!ready || !!mine.guard}
      aria-label={hand === 'L' ? 'Punch with the left' : 'Punch with the right'} data-testid={`robot-punch-${hand}`}
      onPointerDown={(e) => { e.preventDefault(); press({ kind: 'punch', hand }); }}>
      <span className="nk-pad-big">{hand === 'L' ? 'Left' : 'Right'}</span><span className="nk-pad-small">punch</span>
    </button>
  );
  return (
    <div className={`nk-controller${fighting ? '' : ' waiting'}`} style={{ ['--robot' as string]: ROBOT_COLORS[seat] }} data-testid="robot-card">
      <div className="nk-top">
        <span className="nk-me">You {v.wins[me] || 0} : {v.wins[rival] || 0} {name(rival)}</span>
        <span className="nk-mine" aria-label={`Your robot has ${mine.health} of ${v.maxHealth} parts left`} data-testid="robot-my-parts">
          {Array.from({ length: v.maxHealth }, (_, k) => <i key={k} className={k < mine.health ? '' : 'lost'} />)}
        </span>
      </div>
      <div className="nk-word" aria-live="polite" data-testid="robot-word">{word || (lastLost && fighting ? `Lost your ${PART_NAMES[lastLost]}` : 'Punch, or hold Block')}</div>
      <div className="nk-pads">
        {pad('L')}
        {pad('R')}
        <button type="button" className={`nk-pad block${mine.guard ? ' held' : ''}`} disabled={!fighting || busy === 'guard_break' || busy === 'stunned' || busy === 'flinch'}
          aria-label="Hold to block" aria-pressed={!!mine.guard} data-testid="robot-block"
          onPointerDown={(e) => { e.preventDefault(); try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* a pointer the browser no longer knows */ } guard(true); }}
          onPointerUp={() => guard(false)} onPointerCancel={() => guard(false)} onLostPointerCapture={() => guard(false)}>
          <span className="nk-guard-meter" style={{ transform: `scaleX(${guardLeft})` }} aria-hidden />
          <span className="nk-pad-big">Block</span><span className="nk-pad-small">hold, then let go to punch</span>
        </button>
      </div>
    </div>
  );
}
