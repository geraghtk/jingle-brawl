// Party mini-games on the phone: intro → sketch pad / Whack-an-Elf → voting → results.
import { useEffect, useMemo, useRef, useState } from 'react';
import { GAMES, rank, whackPoints, whackSequence } from '../core/party';
import type { PartyGame } from '../core/party';
import { reportWhack, submitSketch, voteSketch } from '../core/partyflow';
import { useServerOffset } from '../firebase/game';
import { GRINCH_SVG } from '../ui/grinch';
import { svg as mascotSvg } from '../ui/mascots';
import { PartyResults, useSecondsLeft } from '../ui/PartyResults';
import '../ui/party.css';
import { usePlayer } from './PlayerContext';
import { Glyphs, Icon } from '../ui/icons';

export function PartyCard() {
  const { game, me } = usePlayer();
  const mg = game.minigame as PartyGame | undefined;
  if (!mg || !GAMES[mg.type] || !(mg.participants || []).includes(me)) return null;
  return (
    <div className="card pg" id="party-card" data-testid="party-card">
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
        {mg.prompt && <div className="pg-prompt">“{mg.prompt}”</div>}
        <p className="pg-sub">
          {tied ? <><b>You're in the tie!</b> Beat the other tied bidders to win it.</>
            : `Not in the tie? Play for party points${cfg.partyPrize === 'points_chips' ? ' and the Elf Bonus chip!' : '!'}`}
        </p>
        <div className="pg-timer">{secs}s</div>
        <p className="pg-sub">Get ready…</p>
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
  const { me, act } = usePlayer();
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
    act((g) => submitSketch(g, me, out.toDataURL('image/jpeg', 0.7)));
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

function WhackBoard({ mg, secs, offset }: { mg: PartyGame; secs: number; offset: number }) {
  const { me, act } = usePlayer();
  const items = useMemo(() => whackSequence(mg.seed), [mg.seed]);
  const [now, setNow] = useState(() => Date.now() + offset);
  const [score, setScore] = useState(0);
  const hit = useRef(new Set<number>());
  const [floats, setFloats] = useState<{ cell: number; v: number; id: number }[]>([]);
  const scoreRef = useRef(0);
  const sent = useRef<number | null>(null);
  const t0 = mg.playStartAt || 0;

  useEffect(() => {
    const frame = setInterval(() => setNow(Date.now() + offset), 40);
    const report = (final: boolean) => {
      if (final || sent.current !== scoreRef.current) { sent.current = scoreRef.current; act((g) => reportWhack(g, me, scoreRef.current, final)); }
    };
    const live = setInterval(() => report(false), 700);
    const end = setTimeout(() => { clearInterval(frame); clearInterval(live); report(true); }, Math.max(0, mg.phaseEndsAt - (Date.now() + offset) - 150));
    return () => { clearInterval(frame); clearInterval(live); clearTimeout(end); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const t = now - t0;
  const showing = Array.from({ length: 9 }, (_, c) => items.findIndex((it) => it.cell === c && t >= it.at && t < it.at + it.upMs));
  const tap = (c: number) => {
    const idx = showing[c];
    if (idx < 0 || hit.current.has(idx)) return;
    hit.current.add(idx);
    const it = items[idx];
    const v = whackPoints(it.kind, (Date.now() + offset - t0 - it.at) / it.upMs);
    scoreRef.current += v;
    setScore(scoreRef.current);
    const id = Math.random();
    setFloats((f) => [...f, { cell: c, v, id }]);
    setTimeout(() => setFloats((f) => f.filter((x) => x.id !== id)), 650);
  };

  return (
    <>
      <h3><Icon name="mallet" /> Whack-an-Elf!</h3>
      <div className="pg-timer">{secs}s</div>
      <div className="whack-score">{score}</div>
      <div className="whack-grid">
        {showing.map((idx, c) => {
          const it = idx >= 0 ? items[idx] : undefined;
          const up = !!it && !hit.current.has(idx);
          return (
            <button key={c} className={`whack-cell${up ? ' up' : ''}${it?.kind === 'gold' ? ' gold' : ''}`} data-cell={c} data-kind={it ? it.kind : ''}
              onPointerDown={() => tap(c)}>
              <div className="pop" dangerouslySetInnerHTML={{ __html: it ? (it.kind === 'grinch' ? GRINCH_SVG : mascotSvg.elf) : '' }} />
              {floats.filter((f) => f.cell === c).map((f) => (
                <div key={f.id} className="whack-float" style={{ color: f.v > 0 ? '#1f9d55' : '#d7263d' }}>{f.v > 0 ? `+${f.v}` : f.v}</div>
              ))}
            </button>
          );
        })}
      </div>
    </>
  );
}

function Voting({ mg, secs }: { mg: PartyGame; secs: number }) {
  const { me, act } = usePlayer();
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
              <button data-vote={pid} onClick={() => act((g) => voteSketch(g, me, pid))}>
                <img src={mg.submissions![pid]} alt={`drawing ${i + 1}`} /><figcaption>#{i + 1}{myVote === pid ? ' ✓' : ''}</figcaption>
              </button>
            )}
          </figure>
        )) : <p className="pg-sub">Nobody finished a drawing!</p>}
      </div>
    </>
  );
}
