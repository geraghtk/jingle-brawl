// Mecha Santa Mayhem on a phone: the buttons for your job on your team's robot, and nothing else (eyes
// on the TV, thumbs on the phone). A leg: STEP and JUMP. The balance: push left, push right. The arms:
// SMASH and GRAB. Every button works whenever it is pressed and nothing here says when to press it
// (Kevin, 2026-11-22: being told when was boring; the game is the team calling it out), so a button
// never lights up, changes or counts down: the phone only says what has just happened to the robot.
// A small team has two jobs each: both are on the screen. The phone only says what was pressed; the
// room decides what the robot did.
import { useEffect, useRef } from 'react';
import { posAt } from '../core/minigames/engines/walker';
import { JOBS, JOB_NAMES, mechaView } from '../core/minigames/mecha/view';
import { useServerNow } from '../net/hooks';
import { BUSY_WORDS, MECHA_COLORS } from '../ui/mecha/look';
import { buzz, play } from '../ui/sound';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/mecha.css';

export function MechaCard() {
  const { game, me, send } = usePlayer();
  const av = game.arenaView!;
  const v = mechaView(av.view);
  const ticking = useServerNow(80);
  const now = av.pausedAt ?? ticking;                    // paused: the race stands as it was
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const ti = v.teams.findIndex((t) => t.teamId === v.myTeam);
  const team = ti >= 0 ? v.teams[ti] : null;
  const other = ti >= 0 ? v.teams[ti === 0 ? 1 : 0] : null;
  const w = team?.walker;

  // my robot came to grief: the phone thumps
  const slips = useRef(w?.slips ?? 0);
  useEffect(() => {
    const n = w?.slips ?? 0;
    if (n > slips.current) { buzz(w?.busy === 'topple' ? [200, 80, 200] : [70, 40, 70]); play(w?.busy === 'topple' ? 'thud' : 'wrong'); }
    slips.current = n;
  }, [w?.slips]); // eslint-disable-line react-hooks/exhaustive-deps

  const press = (action: Record<string, unknown>) => { buzz(15); void send({ type: 'arena', action }, { quiet: true }); };

  if (!team || !w || !other) {
    return (
      <div className="card ms-watch" data-testid="mecha-card">
        <p className="ms-watch-title">Mecha Santa Mayhem</p>
        <p>Watch the TV: two teams, two giant wind-up Santas, one finish line.</p>
      </div>
    );
  }

  const running = v.stage !== 'done';
  const busy = w.busy && now < w.busyUntil ? w.busy : undefined;
  const mine = (v.myJobs || []).slice().sort((a, b) => a - b);
  const sharing = (k: number) => team.crew[k].filter((id) => id !== me).map(name).join(' and ');
  const ahead = posAt(w, now) - posAt(other.walker, now);
  const won = v.stage === 'done' ? (v.winner === team.teamId ? 'won' : v.winner ? 'lost' : 'level') : null;

  // what has just happened to the robot, never what to press next: that is for the team to call out
  const word = won === 'won' ? 'Your team won!' : won === 'lost' ? 'The other team won.' : won === 'level' ? 'Dead level: the host decides.' : busy ? BUSY_WORDS[busy] : '';

  const pad = (id: string, cls: string, big: string, label: string, action: Record<string, unknown>, job?: string) => (
    <button key={id} type="button" className={`ms-pad live ${cls}`} disabled={!running} aria-label={label} data-testid={id}
      onPointerDown={(e) => { e.preventDefault(); press(action); }}>
      {job && <span className="ms-pad-job">{job}</span>}
      <span className="ms-pad-big">{big}</span>
    </button>
  );
  const shared = (k: number) => (sharing(k) ? ` · with ${sharing(k)}` : '');
  const leg = (k: 0 | 1) => (
    <div key={k} className="ms-leg">
      {pad(`mecha-leg-${k}`, 'leg', 'STEP', `${JOB_NAMES[JOBS[k]]}: step`, { kind: 'step', leg: k }, `${JOB_NAMES[JOBS[k]]}${shared(k)}`)}
      {pad(`mecha-jump-${k}`, 'jump', 'JUMP', `${JOB_NAMES[JOBS[k]]}: jump`, { kind: 'jump', leg: k })}
    </div>
  );
  const balance = () => (
    <div key="balance" className="ms-balance" data-testid="mecha-balance">
      <span className="ms-pad-job">Balance{shared(2)}</span>
      <div className="ms-balance-pads">
        {pad('mecha-lean-left', 'lean', '◀', 'Push it left', { kind: 'lean', dir: -1 })}
        {pad('mecha-lean-right', 'lean', '▶', 'Push it right', { kind: 'lean', dir: 1 })}
      </div>
    </div>
  );
  const arms = () => (
    <div key="arms" className="ms-arms">
      {pad('mecha-smash', 'arm smash', 'SMASH', 'Arms: smash', { kind: 'smash' }, `Arms${shared(3)}`)}
      {pad('mecha-grab', 'arm grab', 'GRAB', 'Arms: grab', { kind: 'grab' })}
    </div>
  );

  return (
    <div className={`ms-controller jobs-${mine.length}${running ? '' : ' waiting'}`} style={{ ['--team' as string]: MECHA_COLORS[ti] }} data-testid="mecha-card">
      <div className="ms-top">
        <span className="ms-me" data-testid="mecha-my-team">Team {team.teamId} · {mine.map((k) => JOB_NAMES[JOBS[k]]).join(' + ')}</span>
        <span className={`ms-gap${ahead >= 0 ? ' ahead' : ''}`} data-testid="mecha-ahead">{Math.abs(ahead) < 0.5 ? 'Neck and neck' : ahead > 0 ? 'Ahead' : 'Behind'}</span>
      </div>
      <div className={`ms-phone-word${won ? ` game-over ${won === 'won' ? 'win' : won === 'lost' ? 'lose' : 'level'}` : ''}`} aria-live="polite" data-testid="mecha-word">{word || 'Watch the TV and call it out!'}</div>
      <div className="ms-pads">
        {mine.map((k) => (k <= 1 ? leg(k as 0 | 1) : k === 2 ? balance() : arms()))}
      </div>
    </div>
  );
}
