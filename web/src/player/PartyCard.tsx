// Party mini-games on the phone: intro → sketch pad / Whack-an-Elf → voting → results.
import { useEffect, useMemo, useRef, useState } from 'react';
import { GAMES, rank, WHACK_ROLL, whackScores, whackSequence, whackTap } from '../core/party';
import type { PartyGame } from '../core/party';
import { useServerOffset } from '../net/hooks';
import { PartyResults, useSecondsLeft } from '../ui/PartyResults';
import '../ui/party.css';
import { usePlayer } from './PlayerContext';
import { Glyphs, Icon } from '../ui/icons';
import { buzz, play } from '../ui/sound';
import { WhackHole, WhackLegend, preloadWhackArt } from '../ui/whack';
import type { HoleState } from '../ui/whack';
import type { WhackKind } from '../core/party';

export function PartyCard() {
  const { game, me } = usePlayer();
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || !GAMES[mg.type] || !(mg.participants || []).includes(me)) return null;
  return (
    <div className={`card pg${mg.type === 'whack_elf' && mg.phase === 'playing' ? ' whack-full' : ''}`} id="party-card" data-testid="party-card">
      <Screen key={`${mg.seed}|${mg.phase}`} mg={mg} />
    </div>
  );
}

function Screen({ mg }: { mg: PartyGame }) {
  const { game, me, cfg } = usePlayer();
  const offset = useServerOffset();
  const secs = useSecondsLeft(mg.phaseEndsAt, offset);
  const g = GAMES[mg.type];
  if (mg.phase === 'intro') {
    const tied = (mg.tiedBidders || []).includes(me);
    return (
      <>
        <h3><Glyphs>{g.emoji}</Glyphs> {g.label}!</h3>
        <p className="pg-sub">{g.rules}</p>
        {mg.type === 'whack_elf' && <WhackLegend />}
        {mg.prompt && <div className="pg-prompt">“{mg.prompt}”</div>}
        <p className="pg-sub">
          {mg.minigameGoal === 'lab' ? 'Practice: nothing is won or lost.' : tied ? <><b>You're in the tie!</b> Beat the other tied bidders to win it.</>
            : `Not in the tie? Play for party points${cfg.partyPrize === 'points_chips' ? ' and the Elf Bonus chip!' : '!'}`}
        </p>
        {mg.type === 'whack_elf' && secs <= 3 ? <StartCount secs={secs} /> : <><div className="pg-timer">{secs}s</div><p className="pg-sub">Get ready…</p></>}
      </>
    );
  }
  if (mg.phase === 'playing' && mg.type === 'sketch_off') return <SketchPad mg={mg} secs={secs} offset={offset} />;
  if (mg.phase === 'playing') return <WhackBoard mg={mg} secs={secs} offset={offset} />;
  if (mg.phase === 'voting') return <Voting mg={mg} secs={secs} />;
  if (mg.phase === 'results' && mg.results) return <PartyResults mg={mg} game={game} me={me} />;
  return null;
}

const COLORS = ['#111', '#d7263d', '#1f9d55', '#1e6fd9', '#f2a900', '#8b5a2b'];
type Stroke = { color: string; pts: [number, number][] };

function SketchPad({ mg, secs, offset }: { mg: PartyGame; secs: number; offset: number }) {
  const { me, send } = usePlayer();
  const canvas = useRef<HTMLCanvasElement>(null);
  const strokes = useRef<Stroke[]>([]);
  const current = useRef<Stroke | null>(null);
  const [color, setColor] = useState(COLORS[0]);
  const [submitted, setSubmitted] = useState(!!mg.submissions?.[me]);
  const submittedRef = useRef(submitted);
  submittedRef.current = submitted;

  const paint = () => {
    const c2d = canvas.current?.getContext('2d');
    if (!c2d) return;
    c2d.fillStyle = '#fff'; c2d.fillRect(0, 0, 640, 640);
    c2d.lineCap = 'round'; c2d.lineJoin = 'round'; c2d.lineWidth = 12;
    for (const s of strokes.current) {
      c2d.strokeStyle = s.color; c2d.beginPath();
      s.pts.forEach(([x, y], i) => (i ? c2d.lineTo(x, y) : c2d.moveTo(x, y)));
      if (s.pts.length === 1) c2d.lineTo(s.pts[0][0] + 0.1, s.pts[0][1]);
      c2d.stroke();
    }
  };
  const at = (e: React.PointerEvent): [number, number] => {
    const r = canvas.current!.getBoundingClientRect();
    return [((e.clientX - r.left) * 640) / r.width, ((e.clientY - r.top) * 640) / r.height];
  };
  const submit = () => {
    if (submittedRef.current || !strokes.current.length || !canvas.current) return;
    setSubmitted(true);
    const out = document.createElement('canvas'); out.width = out.height = 320;
    out.getContext('2d')!.drawImage(canvas.current, 0, 0, 320, 320);
    send({ type: 'sketch', drawing: out.toDataURL('image/jpeg', 0.7) });
  };
  useEffect(() => {
    paint();
    // Time's up: hand in whatever is on the canvas
    const t = setTimeout(submit, Math.max(0, (mg.phaseEndsAt || 0) - (Date.now() + offset) - 400));
    return () => clearTimeout(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <h3><Icon name="pencil" /> Draw: “{mg.prompt}”</h3>
      <div className="pg-timer">{secs}s</div>
      <canvas ref={canvas} className="pg-pad" width={640} height={640}
        onPointerDown={(e) => { if (submitted) return; canvas.current!.setPointerCapture?.(e.pointerId); current.current = { color, pts: [at(e)] }; strokes.current.push(current.current); paint(); }}
        onPointerMove={(e) => { if (current.current) { current.current.pts.push(at(e)); paint(); } }}
        onPointerUp={() => { current.current = null; }} onPointerCancel={() => { current.current = null; }} />
      <div className="pg-colors">
        {COLORS.map((c) => <button key={c} style={{ background: c }} className={c === color ? 'on' : ''} aria-label="colour" onClick={() => setColor(c)} />)}
      </div>
      <div className="pg-btns">
        <button className="pg-btn" data-act="undo" onClick={() => { if (!submitted) { strokes.current.pop(); paint(); } }}>↶ Undo</button>
        <button className="pg-btn" data-act="clear" onClick={() => { if (!submitted) { strokes.current = []; paint(); } }}>Clear</button>
        <button className="pg-btn primary" data-act="done" data-testid="sketch-done" onClick={submit}>{submitted ? '✓ Submitted' : 'Done!'}</button>
      </div>
    </>
  );
}

/** The last three seconds before the elves come: 3, 2, 1, each with a tick. */
function StartCount({ secs }: { secs: number }) {
  useEffect(() => { play('tick', secs <= 1 ? 1.5 : 1); }, [secs]);
  return <div className="wh-count" key={secs} role="status" data-testid="whack-count">{Math.max(1, secs)}</div>;
}

const ordinal = (n: number) => { const s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };

/** Whack-an-Elf on a phone: the whole screen is the board. Every phone gets the same pop-ups (the seed);
 *  the phone counts its own taps and reports the score (the room holds it to what was possible). */
function WhackBoard({ mg, secs, offset }: { mg: PartyGame; secs: number; offset: number }) {
  const { send, game, me } = usePlayer();
  const items = useMemo(() => whackSequence(mg.seed), [mg.seed]);
  const [now, setNow] = useState(() => Date.now() + offset);
  const [score, setScore] = useState(0);
  const hit = useRef(new Set<number>());
  const [floats, setFloats] = useState<{ cell: number; v: number; id: number; roll: boolean }[]>([]);
  const scoreRef = useRef(0);
  const sent = useRef<number | null>(null);
  const t0 = mg.playStartAt || 0;
  // hits in a row; an elf that gets away (or the Grump) ends it
  const streak = useRef(0);
  const [roll, setRoll] = useState(0);
  const settled = useRef(0);                    // pop-ups before this one are over (hit or gone)
  // what each hole showed last, so a pop-up slides back down (or stays bonked) after its time is up
  const last = useRef<({ kind: WhackKind; hit: boolean } | undefined)[]>([]);

  useEffect(() => {
    preloadWhackArt();
    play('go');
    const frame = setInterval(() => {
      const at = Date.now() + offset;
      setNow(at);
      // an elf that went back down untapped ends the roll
      while (settled.current < items.length && items[settled.current].at + items[settled.current].upMs < at - t0) {
        const i = settled.current++;
        if (!hit.current.has(i) && items[i].kind !== 'grump' && streak.current) { streak.current = 0; setRoll(0); }
      }
    }, 40);
    const report = (final: boolean) => {
      if (final || sent.current !== scoreRef.current) { sent.current = scoreRef.current; send({ type: 'whack', score: scoreRef.current, final }); }
    };
    const live = setInterval(() => report(false), 700);
    const end = setTimeout(() => { clearInterval(frame); clearInterval(live); report(true); }, Math.max(0, mg.phaseEndsAt - (Date.now() + offset) - 150));
    return () => { clearInterval(frame); clearInterval(live); clearTimeout(end); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // the last five seconds tick
  useEffect(() => { if (secs <= 5 && secs >= 1) play('tick', secs === 1 ? 1.5 : 1); }, [secs]);

  const t = now - t0;
  const showing = Array.from({ length: 9 }, (_, c) => items.findIndex((it) => it.cell === c && t >= it.at && t < it.at + it.upMs));
  const tap = (c: number) => {
    const idx = showing[c];
    if (idx < 0 || hit.current.has(idx)) return;
    hit.current.add(idx);
    const it = items[idx];
    const r = whackTap(it.kind, (Date.now() + offset - t0 - it.at) / it.upMs, streak.current);
    streak.current = r.streak;
    setRoll(r.streak);
    scoreRef.current += r.points;
    setScore(scoreRef.current);
    // a bonk you can hear and feel: the Grump groans, gold rings, a roll climbs
    if (it.kind === 'grump') { play('wrong'); buzz([90, 40, 120]); }
    else if (it.kind === 'gold') { play('ding', 1.2); buzz([20, 20, 40]); }
    else { play(r.roll ? 'ding' : 'thud', r.roll ? 1 + Math.min(r.streak, 10) * 0.07 : 1.4); buzz(25); }
    const id = Math.random();
    setFloats((f) => [...f, { cell: c, v: r.points, id, roll: r.roll }]);
    setTimeout(() => setFloats((f) => f.filter((x) => x.id !== id)), 750);
  };
  // where I stand right now (the others' scores as they last reported)
  const others = whackScores((game.minigame as PartyGame | undefined)?.scores);
  const ids = mg.participants || [];
  const place = 1 + ids.filter((id) => id !== me && (others[id] || 0) > score).length;

  return (
    <>
      <div className="whack-top">
        <span className={`whack-clock${secs <= 5 ? ' hurry' : ''}`} aria-label={`${secs} seconds left`}>{secs}s</span>
        <span className="whack-score" key={score} aria-label={`Your score ${score}`} data-testid="whack-score">{score}</span>
        <span className="whack-place" data-testid="whack-place">{ordinal(place)}<small> of {ids.length}</small></span>
      </div>
      <div className={`whack-roll${roll >= WHACK_ROLL ? ' on' : ''}`} aria-live="polite" data-testid="whack-roll">
        {roll >= WHACK_ROLL ? `On a roll! ${roll} in a row: +1 each` : t < 900 ? 'Go!' : roll > 0 ? `${roll} in a row` : 'Tap the elves, not the Grump'}
      </div>
      <div className="whack-grid">
        {showing.map((idx, c) => {
          const it = idx >= 0 ? items[idx] : undefined;
          const wasHit = idx >= 0 && hit.current.has(idx);
          const up = !!it && !wasHit;
          if (it) last.current[c] = { kind: it.kind, hit: wasHit };
          const shown = last.current[c];
          const state: HoleState = up ? 'up' : shown?.hit ? 'hit' : 'down';
          return (
            <button key={c} className={`whack-cell wh-cell ${state}${shown ? ` ${shown.kind}` : ''}`} data-cell={c} data-kind={it ? it.kind : ''}
              aria-label={up ? (it!.kind === 'grump' ? 'The Grump: leave it' : it!.kind === 'gold' ? 'A golden elf' : 'An elf') : 'An empty hole'}
              onPointerDown={() => tap(c)}>
              <WhackHole kind={shown?.kind} state={state} />
              <span className="wh-hit" />
              {floats.filter((f) => f.cell === c).map((f) => (
                <div key={f.id} className={`whack-float${f.v < 0 ? ' bad' : f.roll ? ' roll' : ''}`}>{f.v > 0 ? `+${f.v}` : f.v}</div>
              ))}
            </button>
          );
        })}
      </div>
    </>
  );
}

function Voting({ mg, secs }: { mg: PartyGame; secs: number }) {
  const { me, send } = usePlayer();
  const myVote = mg.votes?.[me];
  const order = rank(Object.keys(mg.submissions || {}), {}, mg.seed); // stable shuffle
  return (
    <>
      <h3><Icon name="vote" /> Vote for the best drawing</h3>
      <p className="pg-sub">“{mg.prompt}” (not your own!)</p>
      <div className="pg-timer">{secs}s</div>
      <div className="pg-gallery">
        {order.length ? order.map((pid, i) => (
          <figure key={pid} className={`${pid === me ? 'mine' : ''} ${myVote === pid ? 'voted' : ''}`}>
            {pid === me ? (
              <><img src={mg.submissions![pid]} alt="your drawing" /><figcaption>Yours</figcaption></>
            ) : (
              <button data-vote={pid} onClick={() => send({ type: 'vote_sketch', pick: pid })}>
                <img src={mg.submissions![pid]} alt={`drawing ${i + 1}`} /><figcaption>#{i + 1}{myVote === pid ? ' ✓' : ''}</figcaption>
              </button>
            )}
          </figure>
        )) : <p className="pg-sub">Nobody finished a drawing!</p>}
      </div>
    </>
  );
}
