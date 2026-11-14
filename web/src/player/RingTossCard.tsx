// Reindeer Ring Toss on a phone: the controller. While you can throw, the whole screen is the flick
// surface (eyes on the TV, not the phone): flick up anywhere, its direction aims and its speed is
// the power. The phone sends the flick as it happened (screen fractions and ms); the room works out
// where it lands. On it: your score, the rings in your hand (the scored round: a batch, refilling after
// a short reload), your set so far, and how your last ring went. No flicking? "Use buttons": aim, then
// hold to charge and let go to throw.
import { useEffect, useRef, useState } from 'react';
import type { SwipePath } from '../core/minigames/engines/gesture';
import { miss } from '../core/minigames/engines/targets';
import { lastNeeded, PEG_NAMES, STREAK_MIN, streakAt, streakPitch } from '../core/minigames/ringtoss/flair';
import type { RingView } from '../host/RingTossStage';
import { ringColor, ringView, SetPips, visMs } from '../host/RingTossStage';
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
  const run = last && lastLanded ? streakAt(mine, me) : 0;      // my ringers in a row, once the last one is down
  useEffect(() => {
    if (!last || !lastLanded || seen.current === last.n) return;
    seen.current = last.n;
    if (last.bonus) { play('fanfare'); buzz([60, 40, 140]); }
    else if (last.pegId) { play('ding', streakPitch(run)); buzz(run >= STREAK_MIN ? [30, 30, 60] : 40); }
    else play('thud');
  }, [last?.n, lastLanded]); // eslint-disable-line react-hooks/exhaustive-deps
  // a ringer lights the pad in my colour
  const flash = last && lastLanded && last.pegId && now < last.at + visMs(last) + 1500 ? { n: last.n, set: !!last.bonus } : null;

  const inPool = v.pool.includes(me);
  // the scored round: a used-up batch comes back at reloadAt (this phone's clock shows it, no new view needed)
  const reloadAt = v.reloadAt[me];
  const empty = v.stage === 'throwing' && !((v.myLeft ?? 0) > 0) && reloadAt != null;
  const reloading = empty && now < reloadAt;
  const left = empty && !reloading ? v.batch : v.myLeft ?? 0;
  const canThrow = inPool && v.stage !== 'done' && (left > 0 || reloading);
  const waiting = reloading || now < (v.nextAt || 0);
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
      <span className={`rt-stage-pill${v.stage === 'throwing' && secs <= 10 ? ' hurry' : ''}`} data-testid="ringtoss-stage-pill">
        {v.stage === 'practice' ? 'Practice' : v.stage === 'tiebreak' ? 'Tiebreak' : v.stage === 'done' ? 'Done' : 'Scored'} · {secs}s
      </span>
      <span className="rt-mine" aria-label={`Your score ${v.myScore ?? 0}`}>Score <b data-testid="ringtoss-my-score">{v.myScore ?? 0}</b></span>
    </div>
  );
  // the scored round: the batch's slots, used ones hollow, refilling one by one while you reload; else the rings left
  const slots = v.stage === 'throwing' ? Math.max(v.batch || 0, left) : left;
  const refilled = reloading ? Math.floor((1 - (reloadAt - now) / Math.max(1, v.reloadMs)) * slots) : 0;
  const rings = (
    <div className={`rt-rings${reloading ? ' reloading' : ''}`} aria-label={reloading ? 'Reloading your rings' : `${left} rings in hand`} data-testid="ringtoss-rings">
      {Array.from({ length: slots }, (_, i) => {
        const full = reloading ? i < refilled : i < left;
        return <span key={i} className={`rt-ring-dot${full ? '' : ' used'}`} style={{ borderColor: color }} />;
      })}
    </div>
  );
  const need = lastNeeded(v.pegs, v.got[me] || []);
  const set = v.stage === 'throwing' || v.stage === 'tiebreak'
    ? <div className="rt-myset">Your set <SetPips v={v} pid={me} color={color} />{need && <span className="rt-need" data-testid="ringtoss-need">Just {PEG_NAMES[need] || 'one'} to go!</span>}</div> : null;
  const said = feedback(v, last, !!lastLanded, run);
  if (canThrow && !buttons) {
    return (
      <FlickSurface disabled={waiting} hint={reloading ? 'Reloading…' : undefined} color={color} flash={flash} onSwipe={(path) => throwIt({ path })}>
        {top}
        {rings}
        {set}
        <div key={flash?.n ?? 0} className={`rt-feedback big${flash ? ' hit' : ''}`} aria-live="polite" data-testid="ringtoss-feedback">{said}</div>
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
      <div className="rt-feedback" aria-live="polite" data-testid="ringtoss-feedback">{said}</div>
      {canThrow ? (
        <>
          {rings}
          {set}
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
function feedback(v: RingView, last: RingView['throws'][number] | undefined, landed: boolean, run = 0): string {
  if (v.stage === 'practice' && !last) return 'Watch the TV and flick up anywhere: the direction aims, faster goes further.';
  if (!last) return 'Flick up to throw!';
  if (!landed) return 'In the air…';
  if (last.pegId) {
    const p = v.pegs.find((x) => x.id === last.pegId)!;
    if (last.stage === 'practice') return `Ringer! (${p.points} in a scored round)`;
    return last.bonus ? `Full set! +${last.points + last.bonus}` : run >= STREAK_MIN ? `Ringer! +${last.points} · ${run} in a row!` : `Ringer! +${last.points}`;
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
function FlickSurface({ onSwipe, disabled, hint, color, flash, children }: {
  onSwipe: (path: SwipePath) => void; disabled: boolean; hint?: string; color: string;
  /** My last ring was a ringer (a full set: `set`): the pad lights up in my colour. */
  flash?: { n: number; set: boolean } | null; children: React.ReactNode;
}) {
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
      {flash && <i key={flash.n} className={`rt-pad-flash${flash.set ? ' set' : ''}`} style={{ background: color }} data-testid="ringtoss-flash" />}
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
      <span className="rt-pad-hint" data-testid="ringtoss-pad-hint">{hint ?? (disabled ? 'Next ring…' : 'Flick up anywhere ↑')}</span>
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
