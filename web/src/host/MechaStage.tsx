// Mecha Santa Mayhem on the TV (and the host console): the two rooftop tracks in 3D (ui/mecha/scene3d,
// loaded when the game starts), and over them each team's crew (who has which job, whose go it is), how
// far each robot leans, the course with both robots on it, the clock, and the big words (Timber!,
// Jump!, Team A wins!). Sounds on the TV only.
import { useEffect, useRef, useState } from 'react';
import { leanAt, posAt } from '../core/minigames/engines/walker';
import { JOBS, mechaView } from '../core/minigames/mecha/view';
import type { MechaTeamView } from '../core/minigames/mecha/view';
import { useServerNow, useServerOffset } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { EVENT_WORDS, JOB_SHORT, MECHA_COLORS, OBSTACLE_NAMES, OBSTACLE_RECIPE } from '../ui/mecha/look';
import type { MechaScene } from '../ui/mecha/scene3d';
import { play } from '../ui/sound';
import '../ui/arena.css';
import '../ui/motion.css';
import '../ui/mecha.css';
import { useHost } from './HostContext';

const WORD_MS = 1300;

export function MechaStage() {
  const { game, tv } = useHost();
  const offset = useServerOffset();
  const ticking = useServerNow(100);
  const av = game.arenaView!;
  const now = av.pausedAt ?? ticking;                    // paused: the race stands as it was
  const v = mechaView(av.view);
  const canvas = useRef<HTMLCanvasElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const live = useRef({ v, offset, pausedAt: av.pausedAt });
  live.current = { v, offset, pausedAt: av.pausedAt };                 // the frame loop outlives renders: it reads this render's view
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let scene: MechaScene | null = null, raf = 0, gone = false;
    const onResize = () => scene?.resize();
    import('../ui/mecha/scene3d').then(({ createMechaScene }) => {
      if (gone || !canvas.current) return;
      scene = createMechaScene(canvas.current);
      window.addEventListener('resize', onResize);
      const loop = () => {
        const { v: view, offset: off, pausedAt } = live.current;
        scene!.resize();
        // (the scene says when it has split the picture in two: the strips get their labels, with no re-render per frame)
        const { split } = scene!.frame(view, pausedAt ?? Date.now() + off, MECHA_COLORS);
        stage.current?.classList.toggle('split', split);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }).catch(() => setFailed(true));             // no WebGL on this screen: the bars and words still tell the race
    return () => { gone = true; cancelAnimationFrame(raf); window.removeEventListener('resize', onResize); scene?.dispose(); };
  }, []);

  // out loud on the TV: each new thing a robot does, once; the start and the end
  const newest = v.teams.map((t) => t.walker.events.at(-1)?.n ?? 0);
  const heard = useRef({ n: newest, stage: v.stage as string, primed: false });
  useEffect(() => {
    const was = heard.current;
    heard.current = { n: newest.map((n, i) => Math.max(n, was.n[i] ?? 0)), stage: v.stage, primed: true };
    if (!tv) return;
    if (!was.primed) { play('go'); return; }
    v.teams.forEach((t, i) => {
      const pitch = i === 0 ? 1 : 0.84;                                 // two robots, two voices
      for (const e of t.walker.events) {
        if (e.n <= (was.n[i] ?? 0)) continue;
        if (e.kind === 'step' || e.kind === 'stride') play('tick', 0.45 * pitch);
        else if (e.kind === 'stumble' || e.kind === 'misjump') play('wrong', pitch);
        else if (e.kind === 'topple' || e.kind === 'fell') { play('clank', 0.6 * pitch); play('thud', 0.7 * pitch); }
        else if (e.kind === 'hit') play('thud', 1.2 * pitch);
        else if (e.kind === 'whiff') play('whoosh', 1.2 * pitch);
        else if (e.kind === 'smashed') { play('clank', 0.8 * pitch); play('pass', 0.7); }
        else if (e.kind === 'jump') play('jump', pitch);
        else if (e.kind === 'boost') play('whoosh', pitch);
        else if (e.kind === 'saved' || e.kind === 'topped') play('right', pitch);
        else if (e.kind === 'grabbed' || e.kind === 'rung') play('tick', 0.7 * pitch);
        else if (e.kind === 'slip' || e.kind === 'dropped') play('wrong', pitch);
        else if (e.kind === 'pump') play('whoosh', 0.8 * pitch);
      }
    });
    if (was.stage !== v.stage) { if (v.stage === 'done') play('fanfare'); else if (v.stage === 'overtime') play('bell'); }
  }, [newest.join(','), v.stage, tv]); // eslint-disable-line react-hooks/exhaustive-deps

  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const secs = secsLeft(v.endsAt, now);
  const winner = v.stage === 'done' ? v.teams.find((t) => t.teamId === v.winner) : null;
  const pct = (x: number) => `${Math.max(0, Math.min(100, (x / v.course.length) * 100))}%`;

  const panel = (t: MechaTeamView, i: number) => {
    const w = t.walker, lean = leanAt(w, now, t.walk);
    const e = w.events.at(-1);
    const word = e && now - e.at < WORD_MS ? EVENT_WORDS[e.kind] : '';
    const down = w.busy === 'topple' && now < w.busyUntil;
    return (
      <div className={`ms-team ${i === 0 ? 'left' : 'right'}${winner === t ? ' won' : winner ? ' lost' : ''}`} style={{ ['--team' as string]: MECHA_COLORS[i] }} data-testid={`mecha-team-${t.teamId}`}>
        <div className="ms-team-head">
          <span className="ms-team-name">Team {t.teamId}</span>
          <span className="ms-dist" data-testid={`mecha-dist-${t.teamId}`}>{Math.round((posAt(w, now) / v.course.length) * 100)}%</span>
        </div>
        <div className="ms-crew">
          {JOBS.map((job, k) => (
            <span key={job} className="ms-job" data-testid={`mecha-job-${t.teamId}-${k}`}>
              <i>{JOB_SHORT[job]}</i>
              {t.crew[k].map((id) => {
                const p = game.players?.[id];
                return <span key={id} className="ms-who on" title={name(id)}><Avatar name={p?.name || id} avatar={p?.avatar} size={30} /></span>;
              })}
            </span>
          ))}
        </div>
        <div className={`ms-lean${Math.abs(lean) > 0.6 ? ' danger' : ''}${down ? ' down' : ''}`} role="img" aria-label={down ? 'It has fallen over' : `Leaning ${Math.round(Math.abs(lean) * 100)}% ${lean < 0 ? 'back' : 'forward'}`} data-testid={`mecha-lean-${t.teamId}`}>
          <span className="ms-lean-mark" style={{ left: `${50 + lean * 50}%` }} />
        </div>
        {/* what this robot is coming up to, and what it takes (never when) */}
        {t.ahead && v.stage !== 'done' && <div key={t.ahead.id} className={`ms-next ${t.ahead.kind}`} data-testid={`mecha-next-${t.teamId}`}><b>{OBSTACLE_NAMES[t.ahead.kind]}</b><span>{OBSTACLE_RECIPE[t.ahead.kind]}</span></div>}
        {word && <div key={e!.n} className="ms-word" role="status" data-testid={`mecha-word-${t.teamId}`}>{word}</div>}
      </div>
    );
  };

  return (
    <div ref={stage} className={`ms-stage${v.stage === 'done' ? ' over' : ''}`} data-testid="mecha-stage">
      <canvas ref={canvas} className="ms-canvas" aria-label="Two giant wind-up Santas racing over the rooftops" />
      {/* too far apart for one picture: a strip each (the scene draws them; these are their labels) */}
      <div className="ms-split" aria-hidden data-testid="mecha-split">
        {v.teams.slice(0, 2).map((t, i) => <span key={t.teamId} className={`ms-strip s${i}`} style={{ ['--team' as string]: MECHA_COLORS[i] }}><b>Team {t.teamId}</b></span>)}
      </div>
      {failed && <p className="ms-nogl" role="status">This screen can't draw the rooftops in 3D: the race is in the bars.</p>}
      <div className="ms-hud">
        {v.teams[0] && panel(v.teams[0], 0)}
        <div className="ms-clock" data-testid="mecha-clock">
          <span className="ms-clock-label">{v.stage === 'done' ? 'Finish' : v.stage === 'overtime' ? 'Overtime' : 'To the line'}</span>
          {v.stage !== 'done' && <span className={`ms-secs${secs <= 10 ? ' hurry' : ''}`}>{secs}</span>}
        </div>
        {v.teams[1] && panel(v.teams[1], 1)}
      </div>
      {/* the course, start to finish: what is in the way, and where each robot is */}
      <div className="ms-course" data-testid="mecha-course">
        <div className="ms-course-line">
          {v.course.obstacles.map((o) => <span key={o.id} className={`ms-ob ${o.kind}`} style={{ left: pct(o.at), ...(o.width ? { width: pct(o.width) } : {}) }} />)}
          <span className="ms-flag" aria-hidden />
          {v.teams.map((t, i) => <span key={t.teamId} className="ms-runner" style={{ left: pct(posAt(t.walker, now)), ['--team' as string]: MECHA_COLORS[i] }} data-testid={`mecha-runner-${t.teamId}`}>{t.teamId}</span>)}
        </div>
      </div>
    </div>
  );
}
