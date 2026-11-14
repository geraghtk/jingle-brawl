// Nutcracker Knockout on the TV (and the host console): the ring in 3D (ui/robots/scene3d, loaded
// when the game starts), and over it who is fighting, the parts each robot has left, the rounds won,
// the clock and the big words (Round 2, Fight!, Knockout!). Sounds on the TV only.
import { useEffect, useRef, useState } from 'react';
import { PARTS } from '../core/minigames/robots/robots';
import { robotView } from '../core/minigames/robots/view';
import type { RobotView } from '../core/minigames/robots/view';
import { useServerNow, useServerOffset } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { ROBOT_COLORS, PART_NAMES } from '../ui/robots/look';
import type { RobotScene } from '../ui/robots/scene3d';
import { play } from '../ui/sound';
import '../ui/arena.css';
import '../ui/robots.css';
import { useHost } from './HostContext';

/** The big word across the ring right now, if any. */
export function splashFor(v: RobotView, now: number, name: (id: string) => string): string {
  if (v.stage === 'intro') return v.stageEndsAt - now < 900 ? 'Fight!' : `Round ${v.round}`;
  if (v.stage === 'ko' && v.last) return v.last.how === 'knockout' ? 'Knockout!' : 'Time!';
  if (v.stage === 'done') {
    const [a, b] = v.order;
    const wa = v.wins[a] || 0, wb = v.wins[b] || 0;
    return wa === wb ? 'Level!' : `${name(wa > wb ? a : b)} wins!`;
  }
  return v.sudden && v.stageEndsAt - now > 18_500 ? 'Sudden death!' : '';
}

export function RobotStage() {
  const { game, tv } = useHost();
  const offset = useServerOffset();
  const now = useServerNow(200);
  const av = game.arenaView!;
  const v = robotView(av.view);
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const canvas = useRef<HTMLCanvasElement>(null);
  const live = useRef({ v, offset });
  live.current = { v, offset };                 // the frame loop outlives renders: it reads this render's view
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let scene: RobotScene | null = null, raf = 0, gone = false;
    const onResize = () => scene?.resize();
    import('../ui/robots/scene3d').then(({ createRobotScene }) => {
      if (gone || !canvas.current) return;
      scene = createRobotScene(canvas.current);
      window.addEventListener('resize', onResize);
      const loop = () => {
        const { v: view, offset: off } = live.current;
        scene!.frame(view, Date.now() + off, ROBOT_COLORS);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }).catch(() => setFailed(true));             // no WebGL on this screen: the bars and words still tell the fight
    return () => { gone = true; cancelAnimationFrame(raf); window.removeEventListener('resize', onResize); scene?.dispose(); };
  }, []);

  // out loud on the TV: each new blow once, the bell, the knockout
  const heard = useRef({ n: v.events.at(-1)?.n ?? 0, stage: v.stage });
  const newest = v.events.at(-1)?.n ?? 0;
  useEffect(() => {
    const was = heard.current;
    heard.current = { n: Math.max(was.n, newest), stage: v.stage };
    if (!tv) return;
    for (const e of v.events) {
      if (e.n <= was.n) continue;
      if (e.kind === 'hit') play('thud', 1.4);
      else if (e.kind === 'counter') { play('thud', 0.9); play('wrong', 1.6); }
      else if (e.kind === 'blocked') play('tick', 0.5);
      else if (e.kind === 'parried') play('ding', 1.3);
      else if (e.kind === 'guard_break') play('pass', 0.7);
    }
    if (was.stage !== v.stage) {
      if (v.stage === 'fighting') play('go');
      else if (v.stage === 'ko') play(v.last?.how === 'knockout' ? 'doom' : 'horn');
      else if (v.stage === 'done') play('fanfare');
    }
  }, [newest, v.stage, tv]); // eslint-disable-line react-hooks/exhaustive-deps

  const splash = splashFor(v, now, name);
  const secs = secsLeft(v.stageEndsAt, now);
  const corner = (i: 0 | 1) => {
    const pid = v.order[i], p = game.players?.[pid], f = v.fighters[i];
    const wins = v.wins[pid] || 0;
    return (
      <div className={`nk-corner ${i === 0 ? 'left' : 'right'}`} style={{ ['--robot' as string]: ROBOT_COLORS[i] }} data-testid={`robot-corner-${pid}`}>
        <div className="nk-who">
          {p && <Avatar name={p.name} avatar={p.avatar} size={44} />}
          <span className="nk-name">{name(pid)}</span>
          <span className="nk-rounds" aria-label={`${wins} of ${v.roundsToWin} rounds`}>
            {Array.from({ length: v.roundsToWin }, (_, k) => <i key={k} className={k < wins ? 'won' : ''} />)}
          </span>
        </div>
        <div className="nk-parts" role="img" aria-label={`${f.health} of ${v.maxHealth} parts left`} data-testid={`robot-parts-${pid}`}>
          {PARTS.slice(PARTS.length - v.maxHealth).map((part) => (
            <span key={part} className={`nk-part${f.lost.includes(part) ? ' lost' : ''}${part === 'head' ? ' head' : ''}`} title={PART_NAMES[part]} />
          ))}
        </div>
      </div>
    );
  };
  return (
    <div className="nk-stage" data-testid="robot-stage">
      <canvas ref={canvas} className="nk-canvas" aria-label="Two wind-up robots boxing" />
      {failed && <p className="nk-nogl" role="status">This screen can't draw the ring in 3D: the fight is in the bars above.</p>}
      <div className="nk-hud">
        {corner(0)}
        <div className="nk-clock" data-testid="robot-clock">
          <span className="nk-round">{v.stage === 'done' ? 'Final bell' : v.sudden ? 'Sudden death' : `Round ${v.round}`}</span>
          {v.stage === 'fighting' && <span className={`nk-secs${secs <= 10 ? ' hurry' : ''}`}>{secs}</span>}
        </div>
        {corner(1)}
      </div>
      {splash && <div key={splash} className="nk-splash" role="status" data-testid="robot-splash">{splash}</div>}
    </div>
  );
}
