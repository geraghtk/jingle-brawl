// Sketch-Off on the TV (and the host console), full screen like the other games: the prompt and a
// countdown; then a wall of easels, one per artist, each drawing growing as its phone draws it (shuffled,
// no names: the room server sends them that way, core/projection.ts); then the gallery to vote on, as
// big as the screen allows; then whose was whose, with the votes.
import { useEffect, useRef, useState } from 'react';
import { GAMES, rank, sketchScores, wallSlot } from '../core/party';
import type { PartyGame } from '../core/party';
import { useServerOffset } from '../net/hooks';
import { Glyphs, Icon } from '../ui/icons';
import { PartyCeremony } from '../ui/Ceremony';
import { useSecondsLeft } from '../ui/PartyResults';
import { sketchGrid, WALL } from '../ui/sketchGrid';
import { PartyReady } from '../ui/PartyReady';
import '../ui/party.css';
import '../ui/sketch.css';
import { useHost } from './HostContext';

/** The wall's area, measured, so the drawings are as big as they can be in it. */
function useAspect(): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [aspect, setAspect] = useState(2);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => { if (el.clientHeight > 0) setAspect(el.clientWidth / el.clientHeight); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, aspect];
}

function Wall({ n, children, testid }: { n: number; children: React.ReactNode; testid: string }) {
  const [ref, aspect] = useAspect();
  const g = sketchGrid(n, aspect);
  return (
    <div className="sk-wall" ref={ref} data-testid={testid}>
      <div className="sk-grid" style={{ ['--cols' as string]: g.cols, ['--rows' as string]: g.rows, ['--gap' as string]: `${WALL.gap}cqh`, ['--pad' as string]: `${WALL.pad}cqh`, ['--cap' as string]: `${WALL.cap}cqh` }}>{children}</div>
    </div>
  );
}

export function SketchStage({ mg }: { mg: PartyGame }) {
  const { game, cfg } = useHost();
  const offset = useServerOffset();
  const secs = useSecondsLeft(mg.phaseEndsAt, offset);
  const g = GAMES[mg.type];
  const name = (id: string) => game.players?.[id]?.name || '?';
  const ids = mg.participants || [];
  const clock = <span className={`sk-clock${secs <= 5 ? ' hurry' : ''}`} aria-label={`${secs} seconds left`}>{secs}s</span>;

  if (mg.phase === 'intro') {
    return (
      <div className="sk-stage intro pg" data-testid="sketch-stage">
        <p className="sk-kicker"><Glyphs>{g.emoji}</Glyphs> Everyone draws</p>
        <div className="sk-prompt big">“{mg.prompt}”</div>
        <p className="sk-rules">{g.rules}</p>
        <p className="sk-rules">{mg.minigameGoal === 'lab' ? 'Practice: nothing is won or lost.' : <>Tied: <b>{(mg.tiedBidders || []).map(name).join(', ')}</b>. Everyone else plays for party points{cfg.partyPrize === 'points_chips' ? ' and the Elf Bonus chip' : ''}.</>}</p>
        {mg.awaiting ? <PartyReady mg={mg} game={game} tv /> : <div className="sk-count" key={secs}>{secs}</div>}
      </div>
    );
  }

  if (mg.phase === 'playing') {
    const wall = mg.wall || {};
    const done = Object.keys(mg.submissions || {}).length;
    return (
      <div className="sk-stage pg" data-testid="sketch-stage">
        <header className="sk-head">
          {clock}
          <div className="sk-prompt"><Icon name="pencil" /> Draw: “{mg.prompt}”</div>
          <span className="sk-tally" data-testid="sketch-tally">{done}/{ids.length} in</span>
        </header>
        <Wall n={ids.length} testid="sketch-wall">
          {ids.map((_, i) => {
            const w = wall[wallSlot(i)];
            return (
              <figure key={i} className={`sk-easel${w?.done ? ' done' : ''}${w?.img ? '' : ' blank'}`} data-testid={`sketch-easel-${i}`}>
                <div className="sk-paper">
                  {w?.img && w.img.length > 6 ? <img src={w.img} alt="a drawing in progress" draggable={false} /> : <span className="sk-wait"><Icon name="pencil" /></span>}
                </div>
                <figcaption>{w?.done ? '✓ Done' : w?.img ? 'Drawing…' : 'Thinking…'}</figcaption>
              </figure>
            );
          })}
        </Wall>
        <p className="sk-foot">Whose is whose? All is revealed after the vote.</p>
      </div>
    );
  }

  if (mg.phase === 'voting') {
    const order = rank(Object.keys(mg.submissions || {}), {}, mg.seed);
    return (
      <div className="sk-stage pg" data-testid="sketch-stage">
        <header className="sk-head">
          {clock}
          <div className="sk-prompt"><Icon name="vote" /> Vote on your phones: “{mg.prompt}”</div>
          <span className="sk-tally">{Object.keys(mg.votes || {}).length}/{ids.length} voted</span>
        </header>
        {order.length ? (
          <Wall n={order.length} testid="sketch-gallery">
            {order.map((pid, i) => (
              <figure key={pid} className="sk-easel show">
                <div className="sk-paper"><img src={mg.submissions![pid]} alt={`drawing ${i + 1}`} draggable={false} /><span className="sk-num">{i + 1}</span></div>
              </figure>
            ))}
          </Wall>
        ) : <p className="sk-foot">Nobody finished a drawing!</p>}
      </div>
    );
  }

  if (mg.phase === 'results' && mg.results) {
    const r = mg.results;
    const votes = sketchScores(mg.votes, mg.submissions);
    const shown = r.ranking.filter((id) => (mg.submissions?.[id] || '').length > 6);
    return (
      <div className="sk-stage results pg" data-testid="sketch-stage">
        {!!shown.length && (
          <Wall n={shown.length} testid="sketch-reveal">
            {shown.map((pid) => (
              <figure key={pid} className={`sk-easel show${r.winners.includes(pid) ? ' win' : ''}`}>
                <div className="sk-paper"><img src={mg.submissions![pid]} alt={`${name(pid)}'s drawing`} draggable={false} /></div>
                <figcaption><b>{name(pid)}</b> · {votes[pid] || 0} {votes[pid] === 1 ? 'vote' : 'votes'}</figcaption>
              </figure>
            ))}
          </Wall>
        )}
        {/* (the wall beside it shows every drawing with its name: the card needs no picture, unless there is no wall) */}
        <div className="sk-ceremony"><PartyCeremony mg={mg} game={game} where="tv" inline picture={false} /></div>
      </div>
    );
  }
  return null;
}
