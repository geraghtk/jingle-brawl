// Pass the Gremlin on a phone: holding it? solve the problem on the big keypad, then pass it on
// (pick someone or Random). Otherwise: who has it, how jumpy it is, and where it has been.
import { useEffect, useRef, useState } from 'react';
import type { Danger, Pass } from '../core/minigames/gremlin/gremlin';
import { useServerOffset } from '../net/hooks';
import { usePlayer } from './PlayerContext';
import '../ui/gremlin.css';

export interface GremlinView {
  stage: 'holding' | 'pass_ready' | 'sudden_death_intro' | 'doomed';
  round: 'main' | 'sudden_death';
  pool: string[]; holder: string; prevHolder: string | null;
  challengeKind: string; solved: boolean; danger: Danger; trail: Pass[];
  bystanderDoomed: string | null; doomed: string | null;
  prompt?: string; cooldownUntil?: number | null; canPassAt?: number; targets?: string[];
}

export function GremlinCard() {
  const { game, me, send } = usePlayer();
  const offset = useServerOffset();
  const av = game.arenaView;
  const v = av?.view as GremlinView | undefined;
  const name = (id?: string | null) => (id && game.players?.[id]?.name) || 'Someone';
  const [entry, setEntry] = useState('');
  const [error, setError] = useState('');
  const [, tick] = useState(0);
  const mine = v?.holder === me && v.stage !== 'doomed';
  const was = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (v?.holder === me && was.current !== me) navigator.vibrate?.([120, 60, 120]);   // it jumped to me
    was.current = v?.holder;
    setEntry('');
  }, [v?.holder, v?.prompt]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { const id = setInterval(() => tick((n) => n + 1), 250); return () => clearInterval(id); }, []);

  if (!av || !v) return null;
  const now = Date.now() + offset;
  const act = async (action: unknown) => {
    setError('');
    if (!(await send({ type: 'arena', action }))) setError('Not now');
  };
  const inPool = v.pool.includes(me);
  const cls = `card grem${mine ? ' mine' : ''} danger-${v.danger}`;

  if (v.stage === 'doomed' || av.result) {
    return (
      <div className={cls} data-testid="gremlin-card">
        <div className="grem-face">💥</div>
        <div className="grem-title">{v.doomed === me ? 'The Gremlin went berserk on you!' : `It went berserk on ${name(v.doomed)}!`}</div>
      </div>
    );
  }
  if (v.stage === 'sudden_death_intro') {
    return (
      <div className={cls} data-testid="gremlin-card">
        <span className="grem-banner">Sudden death</span>
        <div className="grem-face">👹</div>
        <div className="grem-title">It got {name(v.bystanderDoomed)}!</div>
        <p className="grem-sub">Now just {(av.duelists || []).map(name).join(' and ')}: whoever holds it next time loses the duel.</p>
      </div>
    );
  }
  return (
    <div className={cls} data-testid="gremlin-card">
      {v.round === 'sudden_death' && <span className="grem-banner">Sudden death</span>}
      <div className="grem-face" aria-hidden>👹</div>
      <div className="grem-title" data-testid="gremlin-holder">{mine ? 'You have the Gremlin!' : `${name(v.holder)} has the Gremlin`}</div>
      <div className={`grem-meter ${v.danger}`} aria-label={`danger: ${v.danger}`}><span /></div>
      {mine && v.stage === 'holding' && v.prompt && (
        <>
          <p className="grem-sub">Solve it to pass it on</p>
          <div className="grem-prompt" data-testid="gremlin-prompt">{v.prompt} = ?</div>
          <div className="grem-entry" data-testid="gremlin-entry">{entry || ' '}</div>
          <div className="grem-pad">
            {['1', '2', '3', '4', '5', '6', '7', '8', '9', '⌫', '0', 'OK'].map((k) => (
              <button key={k} className={k === 'OK' ? 'go' : ''} data-key={k} disabled={!!v.cooldownUntil && now < v.cooldownUntil}
                onClick={() => {
                  if (k === '⌫') return setEntry((e) => e.slice(0, -1));
                  if (k === 'OK') { if (entry) act({ kind: 'answer', value: Number(entry) }); return; }
                  setEntry((e) => (e + k).slice(0, 4));
                }}>{k}</button>
            ))}
          </div>
          {v.cooldownUntil && now < v.cooldownUntil ? <div className="grem-error">Wrong! A new one…</div> : <div className="grem-error">{error}</div>}
        </>
      )}
      {mine && v.stage === 'pass_ready' && (
        <>
          <p className="grem-sub">{now < (v.canPassAt || 0) ? 'Hold on…' : 'Quick, pass it on!'}</p>
          <div className="grem-targets">
            <button className="random" data-testid="gremlin-random" disabled={now < (v.canPassAt || 0)} onClick={() => act({ kind: 'pass_random' })}>🎲 Random</button>
            {(v.targets || []).map((id) => (
              <button key={id} data-testid={`gremlin-pass-${id}`} disabled={now < (v.canPassAt || 0)} onClick={() => act({ kind: 'pass', to: id })}>{name(id)}</button>
            ))}
          </div>
          <div className="grem-error">{error}</div>
        </>
      )}
      {!mine && <p className="grem-sub">{inPool ? 'Be ready: it could jump to you!' : "You're out of this round. Watch the TV!"}</p>}
      {!!v.trail.length && <div className="grem-trail">{v.trail.slice(-5).map((p, i) => <span key={i}>{name(p.from)} → {name(p.to)}{p.random ? ' 🎲' : ''}</span>)}</div>}
    </div>
  );
}
