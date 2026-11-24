// Pass the Gremlin on a phone: holding it? do the challenge (quick math on a keypad, a row of
// symbols to repeat, or a marker to stop in the zone), then pass it on (pick someone or Random).
// Otherwise: who has it, how jumpy it is, and where it has been. Sound and buzz when it jumps to you,
// on a right or wrong answer and at the doom (mute on the card, remembered on this phone).
import { useEffect, useRef, useState } from 'react';
import { SYMBOLS, markerAt } from '../core/minigames/gremlin/challenges';
import type { TimingParams } from '../core/minigames/gremlin/challenges';
import { outPoints } from '../core/minigames/gremlin/view';
import type { GremlinView } from '../core/minigames/gremlin/view';
import { useServerNow, useServerOffset } from '../net/hooks';
import { GremlinFace } from '../ui/gremlin';
import { glyphs } from '../ui/icons';
import { buzz, muted, play, setMuted } from '../ui/sound';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/arenaPhone.css';
import '../ui/gremlin.css';

type Act = (action: unknown) => Promise<boolean>;

export function GremlinCard() {
  const { game, me, send } = usePlayer();
  const offset = useServerOffset();
  const av = game.arenaView;
  const v = av?.view as GremlinView | undefined;
  const name = (id?: string | null) => (id && game.players?.[id]?.name) || 'Someone';
  const [error, setError] = useState('');
  const [quiet, setQuiet] = useState(muted);
  const now = useServerNow();
  const mine = v?.holder === me && v.stage !== 'doomed';
  const was = useRef<{ holder?: string; stage?: string; cooldown?: number | null }>({});
  useEffect(() => {
    const prev = was.current;
    if (v?.holder === me && prev.holder !== me && v.stage === 'holding') { buzz([120, 60, 120]); play('jump'); }   // it jumped to me
    else if (v && prev.holder === me && v.holder !== me) play('pass');
    if (mine && v?.stage === 'pass_ready' && prev.stage === 'holding') { buzz(40); play('right'); }
    if (mine && v?.cooldownUntil && v.cooldownUntil !== prev.cooldown) { buzz([60, 40, 60]); play('wrong'); }
    if (v?.stage === 'doomed' && prev.stage !== 'doomed') { buzz(v.doomed === me ? [300, 100, 300] : 80); play('doom'); }
    if (v && (v.stage === 'out_intro' || v.stage === 'sudden_death_intro') && prev.stage !== v.stage) { buzz(v.bystanderDoomed === me ? [300, 100, 300] : 80); play('doom'); }
    was.current = { holder: v?.holder, stage: v?.stage, cooldown: v?.cooldownUntil };
    setError('');
  }, [v?.holder, v?.stage, v?.issuedAt, v?.cooldownUntil]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!av || !v) return null;
  const act: Act = async (action) => {
    setError('');
    const ok = await send({ type: 'arena', action });
    if (!ok) setError('Not now');
    return !!ok;
  };
  const inPool = v.pool.includes(me);
  // the table playing along with a duel: a caught bystander is out; lasting earns Party points (not in practice)
  const duelists: string[] = av.duelists || [];
  const out = v.out || [];
  const party = av.mode === 'party_with_duelists' && duelists.length === 2;
  const watching = party && !duelists.includes(me);
  const pays = watching && !game.currentTurn?.lab;
  const survive = v.survive ?? 0;
  const points = (n: number) => `${n} Party point${n === 1 ? '' : 's'}`;
  const winner = v.doomed && duelists.includes(v.doomed) ? duelists.find((d) => d !== v.doomed) : null;
  const myOut = out.indexOf(me);
  const mineSoFar = myOut >= 0 ? outPoints(myOut, survive) : 0;
  const outLine = !pays ? "You're out. Watch the TV!" : mineSoFar ? `You're out, with ${points(mineSoFar)} for lasting. Watch the TV!` : "You're out: no points this time. Watch the TV!";
  const cls = `ap grem grem-phone${mine ? ' mine' : ''} danger-${v.danger}`;
  const muteBtn = (
    <button type="button" className="grem-mute" aria-pressed={quiet} aria-label={quiet ? 'Sound off' : 'Sound on'} data-testid="gremlin-mute"
      onClick={() => { setMuted(!quiet); setQuiet(!quiet); }}>{quiet ? '🔇' : '🔊'}</button>
  );
  // the top bar every game's phone has (ui/arenaPhone.css): the round on the left, the sound switch on the right
  const bar = (label: string, alarm = false) => (
    <div className="ap-bar">
      <span className={`ap-pill grem-round${alarm ? ' alarm' : ''}`} data-testid="gremlin-round">{label}</span>
      {muteBtn}
    </div>
  );

  if (v.stage === 'doomed' || av.result) {
    return (
      <div className={`${cls} doomed`} data-testid="gremlin-card" role="status">
        {bar('Pass the Gremlin')}
        <div className="grem-body">
        <div className="grem-face grem-boom" aria-hidden>💥</div>
        <div className="grem-title">{v.doomed === me ? 'The Gremlin went berserk on you!' : `It went berserk on ${name(v.doomed)}!`}</div>
        {winner && <p className={`grem-sub game-over ${winner === me ? 'win' : 'lose'}`} data-testid="gremlin-verdict">{av.practice ? 'Only practice: it does not count.' : winner === me ? 'You win the duel!' : `${name(winner)} wins the duel.`}</p>}
        {pays && myOut < 0 && survive > 0 && <p className="grem-sub grem-ok" data-testid="gremlin-earned">Never caught: {points(survive)} for you!</p>}
        {pays && myOut >= 0 && mineSoFar > 0 && <p className="grem-sub" data-testid="gremlin-earned">You lasted a while: {points(mineSoFar)}.</p>}
        </div>
      </div>
    );
  }
  if (v.stage === 'out_intro') {
    const meOut = v.bystanderDoomed === me;
    return (
      <div className={`${cls}${meOut ? ' doomed' : ''}`} data-testid="gremlin-card" role="status">
        {bar(`Round ${(v.roundNo ?? 1) + 1} next`, true)}
        <div className="grem-body">
        <div className="grem-face grem-boom" aria-hidden>💥</div>
        <div className="grem-title">{meOut ? 'It got you!' : `It got ${name(v.bystanderDoomed)}!`}</div>
        <p className="grem-sub" data-testid="gremlin-out-line">{meOut ? outLine : "They're out. Everyone else plays on, and the fuse is shorter this time!"}</p>
        </div>
      </div>
    );
  }
  if (v.stage === 'sudden_death_intro') {
    return (
      <div className={cls} data-testid="gremlin-card" role="status">
        {bar('Sudden death', true)}
        <div className="grem-body">
        <div className="grem-face" aria-hidden><GremlinFace danger="unstable" /></div>
        <div className="grem-title">It got {name(v.bystanderDoomed)}!</div>
        <p className="grem-sub">Now just {(av.duelists || []).map(name).join(' and ')}: whoever holds it next time loses the duel.</p>
        </div>
      </div>
    );
  }
  const cooling = !!v.cooldownUntil && now < v.cooldownUntil;
  return (
    <div className={cls} data-testid="gremlin-card">
      {v.round === 'sudden_death' ? bar('Sudden death', true) : party && (v.roundNo ?? 1) > 1 ? bar(`Round ${v.roundNo} · shorter fuse`, true) : bar('Pass the Gremlin')}
      <div className="grem-body">
      <div className="grem-face" aria-hidden><GremlinFace danger={v.danger} /></div>
      <div className="grem-title" data-testid="gremlin-holder" aria-live="assertive">{mine ? 'You have the Gremlin!' : `${name(v.holder)} has the Gremlin`}</div>
      <div className={`grem-meter ${v.danger}`} role="img" aria-label={`The Gremlin is ${v.danger}`}><span /></div>
      {mine && v.stage === 'holding' && (
        <>
          {v.challengeKind === 'memory_flash' && v.sequence ? <MemoryFlash key={v.issuedAt} v={v} act={act} disabled={cooling} />
            : v.challengeKind === 'perfect_timing' && v.startAt != null ? <PerfectTiming key={v.issuedAt} v={v} act={act} offset={offset} disabled={cooling} />
            : v.prompt ? <QuickMath key={v.issuedAt} prompt={v.prompt} act={act} disabled={cooling} /> : null}
          {cooling ? <div className="grem-error arena-error" role="alert">Wrong! A new one…</div> : <div className="grem-error arena-error" role="alert">{error}</div>}
        </>
      )}
      {mine && v.stage === 'pass_ready' && (
        <>
          <p className="grem-sub grem-ok">{now < (v.canPassAt || 0) ? 'Got it! Hold on…' : 'Got it! Quick, pass it on!'}</p>
          <div className="grem-targets">
            <button className="random" data-testid="gremlin-random" disabled={now < (v.canPassAt || 0)} onClick={() => act({ kind: 'pass_random' })}>{glyphs('🎲 Random')}</button>
            {(v.targets || []).map((id) => (
              <button key={id} data-testid={`gremlin-pass-${id}`} disabled={now < (v.canPassAt || 0)} onClick={() => act({ kind: 'pass', to: id })}>{name(id)}</button>
            ))}
          </div>
          <div className="grem-error arena-error" role="alert">{error}</div>
        </>
      )}
      {!mine && <p className="grem-sub" data-testid="gremlin-watch">{inPool ? `Be ready: it could jump to you! (${name(v.holder)} is doing ${v.challengeLabel || 'a challenge'})`
        : myOut >= 0 ? outLine
        : pays && survive ? `You were never caught: ${points(survive)}! Now it's just the duellists.`
        : "Just the duellists now. Watch the TV!"}</p>}
      {pays && inPool && survive > 0 && <p className="grem-sub grem-stake" data-testid="gremlin-stake">Caught and you're out. Last to the end for {points(survive)}.</p>}
      {!!v.trail.length && <div className="grem-trail">{v.trail.slice(-5).map((p, i) => <span key={i}>{name(p.from)} → {name(p.to)}{p.random ? ' 🎲' : ''}</span>)}</div>}
      </div>
    </div>
  );
}

function QuickMath({ prompt, act, disabled }: { prompt: string; act: Act; disabled: boolean }) {
  const [entry, setEntry] = useState('');
  return (
    <>
      <p className="grem-sub">Solve it to pass it on</p>
      <div className="grem-prompt" data-testid="gremlin-prompt">{prompt} = ?</div>
      <div className="grem-entry" data-testid="gremlin-entry" aria-live="polite">{entry || ' '}</div>
      <div className="grem-pad">
        {['1', '2', '3', '4', '5', '6', '7', '8', '9', '⌫', '0', 'OK'].map((k) => (
          <button key={k} className={k === 'OK' ? 'go' : ''} data-key={k} disabled={disabled} aria-label={k === '⌫' ? 'Delete' : k}
            onClick={() => {
              if (k === '⌫') return setEntry((e) => e.slice(0, -1));
              if (k === 'OK') { if (entry) void act({ kind: 'answer', value: Number(entry) }); return; }
              setEntry((e) => (e + k).slice(0, 4));
            }}>{k}</button>
        ))}
      </div>
    </>
  );
}

/** memory_flash: the row shows for a moment (counted from when this phone got it), then tap it back. */
function MemoryFlash({ v, act, disabled }: { v: GremlinView; act: Act; disabled: boolean }) {
  const [showing, setShowing] = useState(true);
  const [entry, setEntry] = useState<number[]>([]);
  const length = v.length || v.sequence!.length;
  useEffect(() => { const t = setTimeout(() => setShowing(false), v.showMs || 2500); return () => clearTimeout(t); }, [v.showMs]);
  const tap = (n: number) => {
    const next = [...entry, n];
    setEntry(next);
    if (next.length === length) { void act({ kind: 'answer', value: next }); setTimeout(() => setEntry([]), 300); }
  };
  if (showing) {
    return (
      <>
        <p className="grem-sub">Remember these, in order!</p>
        <div className="grem-row show" data-testid="gremlin-memory-show" aria-label={`Remember: ${v.sequence!.map((n) => SYMBOLS[n]).join(' ')}`}>
          {v.sequence!.map((n, i) => <span key={i} className="grem-sym" data-sym={n} style={{ animationDelay: `${i * 120}ms` }}>{glyphs(SYMBOLS[n])}</span>)}
        </div>
      </>
    );
  }
  return (
    <>
      <p className="grem-sub">Tap them back in the same order</p>
      <div className="grem-row" data-testid="gremlin-memory-entry" aria-live="polite">
        {Array.from({ length }, (_, i) => <span key={i} className={`grem-sym slot${entry[i] != null ? ' filled' : ''}`}>{entry[i] != null ? glyphs(SYMBOLS[entry[i]]) : ''}</span>)}
      </div>
      <div className="grem-pad syms" data-testid="gremlin-memory-pad">
        {SYMBOLS.map((s, n) => <button key={n} data-sym={n} disabled={disabled || entry.length >= length} aria-label={s} onClick={() => tap(n)}>{glyphs(s)}</button>)}
      </div>
      <button type="button" className="grem-link" disabled={!entry.length} onClick={() => setEntry([])}>Start over</button>
    </>
  );
}

/** perfect_timing: the marker swings on the room's clock; STOP sends when you tapped, in room time. */
function PerfectTiming({ v, act, offset, disabled }: { v: GremlinView; act: Act; offset: number; disabled: boolean }) {
  const p: TimingParams = { startAt: v.startAt!, periodMs: v.periodMs!, band: v.band! };
  const marker = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const x = markerAt(p, Date.now() + offset) ?? 0;
      if (marker.current) marker.current.style.left = `${x * 100}%`;
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [p.startAt, p.periodMs, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      <p className="grem-sub">Stop it in the green zone!</p>
      <div className="grem-track" data-testid="gremlin-timing" data-start={p.startAt} data-period={p.periodMs} data-band={p.band.join(',')} data-offset={offset}>
        <span className="zone" style={{ left: `${p.band[0] * 100}%`, width: `${(p.band[1] - p.band[0]) * 100}%` }} />
        <span className="marker" ref={marker} />
      </div>
      <button type="button" className="grem-stop" data-testid="gremlin-stop" disabled={disabled}
        onPointerDown={(e) => { e.preventDefault(); void act({ kind: 'answer', value: { at: Date.now() + offset } }); }}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); void act({ kind: 'answer', value: { at: Date.now() + offset } }); } }}>
        STOP
      </button>
    </>
  );
}
