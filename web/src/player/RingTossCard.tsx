// Reindeer Ring Toss on a phone: the controller. While you can throw, the whole screen is the flick
// surface (eyes on the TV, not the phone): flick up anywhere, its direction aims and its speed is
// the power. The phone sends the flick as it happened (screen fractions and ms); the room works out
// where it lands. On it: your score, rings left, and how your last ring went. No flicking? "Use
// buttons": aim, then hold to charge and let go to throw.
import { useEffect, useRef, useState } from 'react';
import type { SwipePath } from '../core/minigames/engines/gesture';
import { miss } from '../core/minigames/engines/targets';
import type { RingView } from '../host/RingTossStage';
import { ringColor, ringView, visMs } from '../host/RingTossStage';
import { useServerOffset } from '../net/hooks';
import { buzz, play } from '../ui/sound';
import { usePlayer } from './PlayerContext';
import '../ui/ringtoss.css';

const CHARGE_PERIOD_MS = 1400;        // the hold-to-charge meter rises and falls in this long
const MAX_CHARGE = 1.3;

export function RingTossCard() {
  const { game, me, send } = usePlayer();
  const offset = useServerOffset();
  const av = game.arenaView!;
  const v = ringView(av.view);
  const [now, setNow] = useState(() => Date.now() + offset);
  const [buttons, setButtons] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { const id = setInterval(() => setNow(Date.now() + offset), 100); return () => clearInterval(id); }, [offset]);

  const mine = v.throws.filter((t) => t.pid === me);
  const last = mine.at(-1);
  const lastLanded = last && now >= last.at + visMs(last);   // with the TV: when the ring comes down there
  const seen = useRef(0);
  useEffect(() => {
    if (!last || !lastLanded || seen.current === last.n) return;
    seen.current = last.n;
    if (last.pegId) { play('right'); buzz(40); } else play('tick');
  }, [last?.n, lastLanded]); // eslint-disable-line react-hooks/exhaustive-deps

  const inPool = v.pool.includes(me);
  const left = v.myLeft ?? 0;
  const canThrow = inPool && left > 0 && v.stage !== 'done';
  const waiting = now < (v.nextAt || 0);
  const throwIt = async (action: Record<string, unknown>) => {
    setError('');
    play('pass');
    const ok = await send({ type: 'arena', action: { kind: 'throw', ...action } });
    if (!ok) setError('That one didn’t go: try again');
  };
  const secs = Math.max(0, Math.ceil((v.stageEndsAt - now) / 1000));
  const color = ringColor(av.participants, me);

  const top = (
    <div className="rt-top">
      <span className="rt-stage-pill" data-testid="ringtoss-stage-pill">
        {v.stage === 'practice' ? 'Practice' : v.stage === 'tiebreak' ? 'Tiebreak' : v.stage === 'done' ? 'Done' : 'Scored'} · {secs}s
      </span>
      <span className="rt-mine" aria-label={`Your score ${v.myScore ?? 0}`}>Score <b data-testid="ringtoss-my-score">{v.myScore ?? 0}</b></span>
    </div>
  );
  const rings = (
    <div className="rt-rings" aria-label={`${left} rings left`}>
      {Array.from({ length: left }, (_, i) => <span key={i} className="rt-ring-dot" style={{ borderColor: color }} />)}
    </div>
  );
  if (canThrow && !buttons) {
    return (
      <FlickSurface disabled={waiting} color={color} onSwipe={(path) => throwIt({ path })}>
        {top}
        {rings}
        <div className="rt-feedback big" aria-live="polite" data-testid="ringtoss-feedback">{feedback(v, last, !!lastLanded)}</div>
        <div className="grem-error" role="alert">{error}</div>
        <button type="button" className="rt-link rt-escape" onPointerDown={(e) => e.stopPropagation()} onClick={() => setButtons(true)} data-testid="ringtoss-controls-toggle">
          Use buttons instead of flicking
        </button>
      </FlickSurface>
    );
  }

  return (
    <div className="card rt-card" data-testid="ringtoss-card">
      {top}
      <div className="rt-feedback" aria-live="polite" data-testid="ringtoss-feedback">{feedback(v, last, !!lastLanded)}</div>
      {canThrow ? (
        <>
          {rings}
          <Controls disabled={waiting} color={color} onThrow={(aim, power) => throwIt({ aim, power })} />
          <button type="button" className="rt-link" onClick={() => setButtons(false)} data-testid="ringtoss-controls-toggle">Flick instead</button>
        </>
      ) : (
        <p className="rt-wait" data-testid="ringtoss-wait">
          {v.stage === 'done' ? 'All rings thrown!' : v.stage === 'tiebreak' && !inPool ? 'The duelists throw one more ring each…'
            : v.stage === 'practice' ? 'Practice done: the scored rings start soon' : 'All your rings are thrown: watch the TV!'}
        </p>
      )}
      <div className="grem-error" role="alert">{error}</div>
    </div>
  );
}

/** Your last ring in words you can act on. */
function feedback(v: RingView, last: RingView['throws'][number] | undefined, landed: boolean): string {
  if (v.stage === 'practice' && !last) return 'Watch the TV and flick up anywhere: the direction aims, faster goes further.';
  if (!last) return 'Flick up to throw!';
  if (!landed) return 'In the air…';
  if (last.pegId) {
    const p = v.pegs.find((x) => x.id === last.pegId)!;
    return last.stage === 'practice' ? `Ringer! (${p.points} in a scored round)` : `Ringer! +${last.points}`;
  }
  const m = miss(v.pegs, last.x, last.z);
  const depth = m.dz > 0.08 ? 'long' : m.dz < -0.08 ? 'short' : '';
  const side = m.dx > 0.05 ? 'right' : m.dx < -0.05 ? 'left' : '';
  return `Missed: a bit ${[depth, side].filter(Boolean).join(' and ') || 'off'}`;
}

/** The whole screen, while you can throw: flick up anywhere. The ring follows your finger and, when
 *  you let go, carries on up and off the screen along your flick (then it's on the TV). The path goes
 *  in screen heights (across too: a sideways millimetre is a millimetre) and ms since the touch began;
 *  the room turns it into the throw. */
function FlickSurface({ onSwipe, disabled, color, children }: { onSwipe: (path: SwipePath) => void; disabled: boolean; color: string; children: React.ReactNode }) {
  const path = useRef<SwipePath | null>(null);
  const t0 = useRef(0);
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const [gone, setGone] = useState<{ x: number; y: number } | null>(null);     // the ring leaving the screen
  useEffect(() => { if (!gone) return; const id = setTimeout(() => setGone(null), 420); return () => clearTimeout(id); }, [gone]);
  const pt = (e: React.PointerEvent): [number, number, number] => [
    e.clientX / window.innerHeight, e.clientY / window.innerHeight, Math.round(e.timeStamp - t0.current),
  ];
  const h = () => window.innerHeight;
  const ringAt = gone ? `translate(${gone.x}px, ${gone.y}px) scale(.7)` : drag ? `translate(${drag.x}px, ${Math.min(0, drag.y)}px)` : undefined;
  return (
    <div className={`rt-fullpad${disabled ? ' disabled' : ''}`} data-testid="ringtoss-card">
      <div className="rt-pad-surface" data-testid="ringtoss-pad"
        onPointerDown={(e) => {
          if (disabled) return;
          try { (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId); } catch { /* a pointer the browser no longer knows */ }
          t0.current = e.timeStamp;
          path.current = [pt(e)];
          setDrag({ x: 0, y: 0 });
        }}
        onPointerMove={(e) => {
          if (!path.current) return;
          const p = pt(e);
          path.current.push(p);
          if (path.current.length > 64) path.current.splice(1, 1);        // keep the start and the release
          setDrag({ x: (p[0] - path.current[0][0]) * h(), y: (p[1] - path.current[0][1]) * h() });
        }}
        onPointerUp={(e) => {
          const p = path.current;
          path.current = null;
          setDrag(null);
          if (!p) return;
          p.push(pt(e));
          const up = p[0][1] - p[p.length - 1][1];
          if (up > 0.03) {
            // off the top along the stroke: from where the finger let go, on past the top edge
            const dx = (p[p.length - 1][0] - p[0][0]) / up, last = drag ?? { x: 0, y: 0 };
            const rise = h() + 120;
            setGone({ x: last.x + dx * rise, y: last.y - rise });
            onSwipe(p.slice(-64));
          }
        }}
        onPointerCancel={() => { path.current = null; setDrag(null); }} />
      <div className="rt-fullpad-info">{children}</div>
      <span className={`rt-pad-ring${gone ? ' gone' : ''}`} style={{ borderColor: color, transform: ringAt }} />
      <span className="rt-pad-hint">{disabled ? 'Next ring…' : 'Flick up anywhere ↑'}</span>
    </div>
  );
}

/** The button way: aim with the slider, hold the button while the meter rises and falls, let go to throw. */
function Controls({ onThrow, disabled, color }: { onThrow: (aim: number, power: number) => void; disabled: boolean; color: string }) {
  const [aim, setAim] = useState(0);
  const [charge, setCharge] = useState<number | null>(null);
  const since = useRef(0);
  useEffect(() => {
    if (charge === null) return;
    let raf = 0;
    const tick = () => {
      const f = ((performance.now() - since.current) % CHARGE_PERIOD_MS) / CHARGE_PERIOD_MS;
      setCharge(MAX_CHARGE * (f < 0.5 ? f * 2 : 2 - f * 2));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [charge === null]); // eslint-disable-line react-hooks/exhaustive-deps
  const release = () => { if (charge !== null) { onThrow(aim, Math.max(0.1, charge)); setCharge(null); } };
  return (
    <div className="rt-controls" data-testid="ringtoss-controls">
      <label className="rt-aim">Aim
        <input type="range" min={-1} max={1} step={0.02} value={aim} onChange={(e) => setAim(Number(e.target.value))} aria-label="Aim left or right" data-testid="ringtoss-aim" />
      </label>
      <div className="rt-meter" aria-hidden><span style={{ width: `${((charge ?? 0) / MAX_CHARGE) * 100}%`, background: color }} /></div>
      <button type="button" className="rt-hold" disabled={disabled} data-testid="ringtoss-hold"
        onPointerDown={(e) => { e.preventDefault(); since.current = performance.now(); setCharge(0); }}
        onPointerUp={release} onPointerLeave={release}
        onKeyDown={(e) => { if ((e.key === ' ' || e.key === 'Enter') && charge === null) { since.current = performance.now(); setCharge(0); } }}
        onKeyUp={(e) => { if (e.key === ' ' || e.key === 'Enter') release(); }}>
        {charge === null ? 'Hold to charge, let go to throw' : 'Let go!'}
      </button>
    </div>
  );
}
