// AI Draw Off on the phone: the two duellists read the prompt, draw against the clock and see how
// the critic rated them; everyone else watches the TV.
import { useEffect, useRef, useState } from 'react';
import type { DrawOff } from '../core/drawoff';
import { useServerOffset } from '../net/hooks';
import { useSecondsLeft } from '../ui/PartyResults';
import { usePlayer } from './PlayerContext';
import { Icon } from '../ui/icons';

type Seg = { color: string; size: number; from: [number, number]; to: [number, number] };

export function DrawOffCard() {
  const { game, me, send, toast } = usePlayer();
  const mg = game.minigame as DrawOff | undefined;
  const offset = useServerOffset();
  const secs = useSecondsLeft(mg?.drawingEndTime, offset);
  const canvas = useRef<HTMLCanvasElement>(null);
  const strokes = useRef<Seg[]>([]);
  const [count, setCount] = useState(0);
  const [color, setColor] = useState('#000000');
  const [size, setSize] = useState(5);
  const last = useRef<[number, number] | null>(null);
  const submitted = !!mg?.submissions?.[me];
  const participant = (mg?.participants || []).includes(me);
  const drawing = mg?.type === 'draw_off' && mg.phase === 'drawing' && participant && !submitted;

  const paint = () => {
    const c = canvas.current;
    const c2d = c?.getContext('2d');
    if (!c || !c2d) return;
    // Opaque white: a transparent PNG reads as black to vision models (hiding black strokes)
    c2d.fillStyle = '#ffffff'; c2d.fillRect(0, 0, c.width, c.height);
    c2d.lineCap = 'round'; c2d.lineJoin = 'round';
    for (const s of strokes.current) {
      c2d.strokeStyle = s.color; c2d.lineWidth = s.size;
      c2d.beginPath(); c2d.moveTo(s.from[0] * c.width, s.from[1] * c.height); c2d.lineTo(s.to[0] * c.width, s.to[1] * c.height); c2d.stroke();
    }
  };
  // A fresh canvas once per Draw Off — never mid-drawing when a rival submits
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const r = c.parentElement!.getBoundingClientRect();
    c.width = Math.max(200, Math.round(r.width)); c.height = Math.round(c.width * 0.75);
    strokes.current = []; setCount(0); paint();
  }, [mg?.startedAt, drawing]); // eslint-disable-line react-hooks/exhaustive-deps

  const submit = async () => {
    if (!canvas.current || submitted || mg?.phase !== 'drawing') return;
    // JPEG on the opaque white page: a fraction of a PNG's size, and the critic reads it the same
    if (await send({ type: 'sketch', drawing: canvas.current!.toDataURL('image/jpeg', 0.8) })) toast('Drawing submitted! ✅');
  };
  useEffect(() => { if (drawing && secs === 0) submit(); }, [secs]); // eslint-disable-line react-hooks/exhaustive-deps

  const pos = (e: React.PointerEvent): [number, number] => {
    const r = canvas.current!.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };

  if (!mg || mg.type !== 'draw_off') {
    return <div className="card minigame-card" id="minigame-card"><div id="minigame-status" style={{ textAlign: 'center' }}><Icon name="wait" /> Initializing AI Draw Off...</div></div>;
  }
  const myScore = mg.scores?.[me];
  const names = mg.participants.map((p) => game.players?.[p]?.name || 'Someone');
  const winner = mg.winners?.[0];
  return (
    <div className="card minigame-card" id="minigame-card" data-testid="drawoff-card">
      <div className="card-title" id="minigame-card-title"><Icon name="palette" /> AI Draw Off</div>
      <div id="minigame-status" style={{ textAlign: 'center', marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        {mg.phase === 'prompt_reveal' && <>
          <div style={{ marginBottom: 6 }}><span className="f6-chip critic-chip"><Icon name="critic" /> AI critic</span></div>
          <div className="drawoff-prompt f6-disp">“{mg.prompt}”</div>
          <div style={{ color: 'var(--muted)' }}>{participant ? 'Drawing starts in a moment. Get ready!' : `${names.join(' and ')} are about to draw. Watch the TV!`}</div>
        </>}
        {mg.phase === 'drawing' && !participant && <div>{names.join(' and ')} are drawing. Watch the TV!</div>}
        {mg.phase === 'drawing' && participant && submitted && <>
          <div style={{ color: 'var(--accent-green)', fontWeight: 700 }}><Icon name="check" /> Submitted!</div>
          <div>Waiting for your opponent... {secs}s left</div>
        </>}
        {drawing && <>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 6 }}>
            <span className="f6-chip critic-chip"><Icon name="critic" /> AI critic</span>
            <span className="drawoff-timer f6-disp f6-num" id="drawoff-timer">{secs}s</span>
          </div>
          <div className="drawoff-prompt f6-disp">“{mg.prompt}”</div>
        </>}
        {mg.phase === 'judging' && <>
          <div style={{ fontSize: '1.1rem', color: 'var(--accent-gold)' }}>Draw: "{mg.prompt}"</div>
          <div style={{ marginTop: '0.5rem' }}><Icon name="critic" /> AI is judging all drawings...</div>
        </>}
        {mg.phase === 'results' && <>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-gold)', marginBottom: '0.5rem' }}>Draw: "{mg.prompt}"</div>
          {mg.wittyVerdict && <div style={{ background: 'var(--bg-card-hover)', borderLeft: '4px solid var(--accent-gold)', padding: '0.75rem', borderRadius: 8, marginBottom: '0.75rem', fontStyle: 'italic' }}><Icon name="masks" /> {mg.wittyVerdict}</div>}
          <div style={{ fontWeight: 700, marginBottom: '0.5rem' }} data-testid="drawoff-result">
            {winner ? (winner === me ? 'You win the Draw Off!' : `${game.players?.[winner]?.name || 'Your rival'} wins the Draw Off`) : "The critic couldn't decide: the host picks the winner"}
          </div>
          {participant && myScore && <div style={{ marginBottom: '0.25rem' }}>Your score: {myScore.rating}/100</div>}
          {participant && myScore?.critique && <div style={{ fontStyle: 'italic' }}>{myScore.critique}</div>}
        </>}
      </div>
      <div id="canvas-container" style={{ position: 'relative', background: 'white', borderRadius: 12, overflow: 'hidden', marginBottom: '1rem', minHeight: 200, display: drawing ? 'block' : 'none' }}>
        <canvas ref={canvas} id="drawing-canvas" data-strokes={count} style={{ display: 'block', width: '100%', height: 'auto', touchAction: 'none', cursor: 'crosshair' }}
          onPointerDown={(e) => { if (!drawing) return; canvas.current!.setPointerCapture?.(e.pointerId); last.current = pos(e); }}
          onPointerMove={(e) => {
            if (!last.current) return;
            const p = pos(e);
            strokes.current.push({ color, size, from: last.current, to: p });
            last.current = p; setCount(strokes.current.length); paint();
          }}
          onPointerUp={() => { last.current = null; }} onPointerCancel={() => { last.current = null; }} />
      </div>
      {drawing && <>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
          <input type="color" id="drawing-color" value={color} onChange={(e) => setColor(e.target.value)} style={{ width: 50, height: 40, border: '2px solid var(--border)', borderRadius: 8, cursor: 'pointer' }} />
          <input type="range" id="drawing-size" min={2} max={20} value={size} onChange={(e) => setSize(parseInt(e.target.value))} style={{ flex: 1, minWidth: 100 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: 60 }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Size:</span>
            <span id="drawing-size-display" style={{ fontWeight: 700, color: 'var(--accent-gold)' }}>{size}</span>
          </div>
        </div>
        <div id="drawing-controls-container" style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
          <button className="btn btn-secondary" onClick={() => { strokes.current.pop(); setCount(strokes.current.length); paint(); }} style={{ flex: 1 }}>↶ Undo</button>
          <button className="btn btn-primary" data-testid="drawoff-submit" onClick={submit} style={{ flex: 1 }}><Icon name="check" /> Submit Drawing</button>
        </div>
      </>}
    </div>
  );
}
