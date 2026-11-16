// Nutcracker Knockout on a phone: the controller. For the two duelists the whole screen is four big
// pads (eyes on the TV, thumbs on the phone): a quick jab and a slow haymaker on top, hold to block and
// tap to dodge below. The phone only says what was pressed; the room times every punch and decides
// what it did. Everyone else is told to watch the TV.
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

  // my robot took a blow: the phone thumps (harder when a part goes, hardest for the head)
  const health = useRef(mine?.health ?? 0);
  const parts = useRef(mine?.lost.length ?? 0);
  useEffect(() => {
    const h = mine?.health ?? 0, n = mine?.lost.length ?? 0;
    if (h < health.current) { buzz(n >= v.parts ? [200, 80, 300] : n > parts.current ? [70, 30, 70] : 35); play('thud'); }
    health.current = h; parts.current = n;
  }, [mine?.health]); // eslint-disable-line react-hooks/exhaustive-deps

  const press = (action: Record<string, unknown>) => { void send({ type: 'arena', action }); };   // a press the room refuses ("Not yet") just doesn't happen
  const guarding = useRef(false);
  const guard = (on: boolean) => {
    if (guarding.current === on) return;
    guarding.current = on;
    press({ kind: 'guard', on });
  };
  // a keyboard works too (a laptop, the console's own seat): A jab, D haymaker, S dodge, Space block
  const fighting = seat != null && v.stage === 'fighting';
  useEffect(() => {
    if (!fighting) return;
    const down = (e: KeyboardEvent) => {
      if (e.repeat) return;
      const k = e.key.toLowerCase();
      if (k === 'a' || k === 'arrowleft') press({ kind: 'punch', punch: 'jab' });
      else if (k === 'd' || k === 'arrowright') press({ kind: 'punch', punch: 'heavy' });
      else if (k === 's' || k === 'arrowdown') press({ kind: 'dodge' });
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
  const left = v.parts - mine.lost.length;
  const word = v.stage === 'intro' ? `Round ${v.round}: get ready`
    : v.stage === 'ko' ? (v.last?.winner === me ? (v.last.how === 'knockout' ? 'You took the head off!' : 'Your round, on time') : (v.last?.how === 'knockout' ? 'You lost your head' : 'Their round, on time'))
    : v.stage === 'done' ? ((v.wins[me] || 0) > (v.wins[rival] || 0) ? 'You win the fight!' : (v.wins[me] || 0) < (v.wins[rival] || 0) ? `${name(rival)} wins the fight` : 'Level: the host decides')
    : busy && BUSY_WORDS[busy] ? BUSY_WORDS[busy] : mine.guard ? 'Blocking' : theirs.punch ? (theirs.punch.kind === 'heavy' ? 'Haymaker coming: dodge!' : 'Jab coming: block!')
    : v.sudden ? 'Sudden death: one hit' : mine.loose ? `Your ${PART_NAMES[mine.loose]} is loose` : '';
  const pad = (kind: 'jab' | 'heavy') => (
    <button type="button" className={`nk-pad punch ${kind}${mine.punch?.kind === kind ? ' swinging' : ''}`} disabled={!ready || !!mine.guard}
      aria-label={kind === 'jab' ? 'Jab: quick and light' : 'Haymaker: slow, hits hard, breaks a block'} data-testid={`robot-punch-${kind}`}
      onPointerDown={(e) => { e.preventDefault(); press({ kind: 'punch', punch: kind }); }}>
      <span className="nk-pad-big">{kind === 'jab' ? 'Jab' : 'Haymaker'}</span>
      <span className="nk-pad-small">{kind === 'jab' ? 'quick' : 'slow, breaks a block'}</span>
    </button>
  );
  return (
    <div className={`nk-controller${fighting ? '' : ' waiting'}`} style={{ ['--robot' as string]: ROBOT_COLORS[seat] }} data-testid="robot-card">
      <div className="nk-top">
        <span className="nk-me">You {v.wins[me] || 0} : {v.wins[rival] || 0} {name(rival)}</span>
        <span className="nk-mine" aria-label={`Your robot has ${left} of ${v.parts} parts left`} data-testid="robot-my-parts">
          {Array.from({ length: v.parts }, (_, k) => <i key={k} className={k < mine.lost.length ? 'lost' : mine.loose && k === mine.lost.length ? 'loose' : ''} />)}
        </span>
      </div>
      <div className="nk-word" aria-live="polite" data-testid="robot-word">{word || 'Jab, haymaker, block or dodge'}</div>
      <div className="nk-pads">
        {pad('jab')}
        {pad('heavy')}
        <button type="button" className={`nk-pad block${mine.guard ? ' held' : ''}`} disabled={!fighting || (!mine.guard && !ready)}
          aria-label="Hold to block" aria-pressed={!!mine.guard} data-testid="robot-block"
          onPointerDown={(e) => { e.preventDefault(); try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* a pointer the browser no longer knows */ } guard(true); }}
          onPointerUp={() => guard(false)} onPointerCancel={() => guard(false)} onLostPointerCapture={() => guard(false)}>
          <span className="nk-guard-meter" style={{ transform: `scaleX(${guardLeft})` }} aria-hidden />
          <span className="nk-pad-big">Block</span><span className="nk-pad-small">hold: stops a jab</span>
        </button>
        <button type="button" className={`nk-pad dodge${mine.dodge ? ' held' : ''}`} disabled={!ready || !!mine.guard}
          aria-label="Dodge: any punch misses" data-testid="robot-dodge"
          onPointerDown={(e) => { e.preventDefault(); press({ kind: 'dodge' }); }}>
          <span className="nk-pad-big">Dodge</span><span className="nk-pad-small">tap: it misses</span>
        </button>
      </div>
    </div>
  );
}
