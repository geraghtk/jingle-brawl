// Whack-an-Elf on the TV (and the host console), full screen like the other games: the rules and a
// 3, 2, 1 before it starts; then the same board every phone has, popping away for the room to watch,
// beside a live race of everyone's score; then the results. The TV plays the start, the last five
// seconds and the finish out loud (each phone plays its own bonks).
import { useEffect, useMemo, useRef, useState } from 'react';
import { GAMES, rank, whackScores, whackSequence } from '../core/party';
import type { PartyGame } from '../core/party';
import { useServerNow, useServerOffset } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { Glyphs } from '../ui/icons';
import { PartyReady } from '../ui/PartyReady';
import { PartyResults, useSecondsLeft } from '../ui/PartyResults';
import { play } from '../ui/sound';
import { WhackHole, WhackLegend, preloadWhackArt } from '../ui/whack';
import type { HoleState } from '../ui/whack';
import '../ui/arena.css';
import '../ui/party.css';
import { FullCtx } from './ArenaFull';
import { useHost } from './HostContext';
import { SketchStage } from './SketchStage';
import { PanelButtons } from './TurnPanel';
import { useContext } from 'react';

/** Is a party game on right now (Whack-an-Elf or Sketch-Off, any of its phases)? Both fill the screen. */
export function partyGameOn(game: { currentTurn?: { phase?: string } | null; minigame?: unknown }): boolean {
  const type = (game.minigame as PartyGame | undefined)?.type;
  return game.currentTurn?.phase === 'minigame_party' && !!type && type in GAMES;
}

/** The full-screen frame (the same as a runtime mini-game's: host/ArenaFull.tsx), around the stage. */
export function PartyFull() {
  const { game, tv } = useHost();
  const { setFull } = useContext(FullCtx);
  const mg = game.minigame as PartyGame | undefined;
  const [screenFull, setScreenFull] = useState(() => typeof document !== 'undefined' && !!document.fullscreenElement);
  useEffect(() => {
    const on = () => setScreenFull(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', on);
    return () => document.removeEventListener('fullscreenchange', on);
  }, []);
  if (!mg) return null;
  const g = GAMES[mg.type];
  return (
    <div className="arena-full" data-testid="party-full">
      <header className="arena-full-head">
        <span className="arena-full-title"><Glyphs>{`${g.emoji} ${g.label}`}</Glyphs></span>
        <span className="arena-full-tools">
          {!screenFull && document.fullscreenEnabled && (
            <button type="button" className="arena-full-btn" onClick={() => document.documentElement.requestFullscreen?.().catch(() => {})}>⛶ Fill the screen</button>
          )}
          {!tv && <span className="arena-full-host"><PanelButtons /></span>}
          {!tv && <button type="button" className="arena-full-btn" data-testid="arena-full-exit" onClick={() => setFull(false)}>✕ Console view</button>}
        </span>
      </header>
      <div className="arena-full-body">{mg.type === 'sketch_off' ? <SketchStage mg={mg} /> : <WhackStage mg={mg} />}</div>
    </div>
  );
}

export function WhackStage({ mg }: { mg: PartyGame }) {
  const { game, tv, cfg } = useHost();
  const offset = useServerOffset();
  const now = useServerNow(60);
  const secs = useSecondsLeft(mg.phaseEndsAt, offset);
  const g = GAMES[mg.type];
  const name = (id: string) => game.players?.[id]?.name || '?';
  const ids = mg.participants || [];
  const tied = mg.tiedBidders || [];
  const items = useMemo(() => whackSequence(mg.seed), [mg.seed]);
  useEffect(() => { preloadWhackArt(); }, []);

  // out loud on the TV: the countdown in, the start, the last five seconds, the finish
  const heard = useRef({ phase: mg.phase, secs });
  useEffect(() => {
    const was = heard.current;
    heard.current = { phase: mg.phase, secs };
    if (!tv) return;
    if (was.phase !== mg.phase) { if (mg.phase === 'playing') play('go'); else if (mg.phase === 'results') play('horn'); return; }
    if (secs !== was.secs && secs >= 1 && ((mg.phase === 'intro' && secs <= 3) || (mg.phase === 'playing' && secs <= 5))) play('tick', secs === 1 ? 1.5 : 1);
  }, [mg.phase, secs, tv]);

  if (mg.phase === 'results' && mg.results) return <div className="wh-stage pg results" data-testid="whack-stage"><PartyResults mg={mg} game={game} /></div>;

  if (mg.phase === 'intro') {
    return (
      <div className="wh-stage pg intro" data-testid="whack-stage">
        <h3 className="wh-title"><Glyphs>{g.emoji}</Glyphs> {g.label}!</h3>
        <p className="wh-rules">{g.rules}</p>
        <WhackLegend />
        <p className="wh-who">
          {mg.minigameGoal === 'lab' ? <>Practice: nothing is won or lost. </> : mg.minigameGoal === 'duel' ? <>The duel: <b>{tied.map(name).join(' vs ')}</b>. </> : <>Tied: <b>{tied.map(name).join(', ')}</b>. </>}
          Everyone plays, on your phones{mg.minigameGoal === 'lab' ? '' : cfg.partyPrize !== 'none' ? ': the rest for party points' : ''}{mg.minigameGoal !== 'lab' && cfg.partyPrize === 'points_chips' ? ' and the Elf Bonus chip' : ''}.
        </p>
        {mg.awaiting ? <PartyReady mg={mg} game={game} tv />
          : secs <= 3 ? <div className="wh-count" key={secs} role="status" data-testid="whack-count">{Math.max(1, secs)}</div> : <div className="wh-getready">Get ready… {secs}</div>}
      </div>
    );
  }

  // playing: the board (the same pop-ups as on every phone, for the room to watch) and the race
  const t = now - (mg.playStartAt || 0);
  const showing = Array.from({ length: 9 }, (_, c) => items.find((it) => it.cell === c && t >= it.at && t < it.at + it.upMs));
  const scores = whackScores(mg.scores);
  const order = rank(ids, scores, mg.seed);
  const top = Math.max(5, ...Object.values(scores));
  const lead = order[0] && (scores[order[0]] || 0) > (scores[order[1]] ?? -Infinity) && (scores[order[0]] || 0) > 0 ? order[0] : null;
  return (
    <div className="wh-stage pg playing" data-testid="whack-stage">
      <div className="wh-board">
        <div className="whack-grid" aria-hidden>
          <BoardCells showing={showing} />
        </div>
        {t < 900 && <div className="wh-go" role="status">Go!</div>}
        {secs <= 5 && secs >= 1 && <div className="wh-final" key={secs} aria-hidden>{secs}</div>}
      </div>
      <div className="wh-side">
        <div className="wh-side-head">
          <div className={`wh-clock${secs <= 5 ? ' hurry' : ''}`} data-testid="whack-clock">{secs}</div>
          <p className="wh-side-word">Tap the elves,<br />not the Grump!</p>
        </div>
      <ol className="wh-race pg-bars" aria-label="Scores so far">
        {order.map((id, i) => {
          const p = game.players?.[id];
          const s = scores[id] || 0;
          return (
            <li key={id} className={`pg-bar wh-racer${id === lead ? ' lead' : ''}${tied.includes(id) ? ' tied' : ''}`} data-testid={`whack-racer-${id}`}>
              <span className="wh-pos">{i + 1}</span>
              {p && <Avatar name={p.name} avatar={p.avatar} size={40} />}
              <span className="nm">{name(id)}{tied.includes(id) && <small> {mg.minigameGoal === 'duel' ? 'duelling' : 'in the tie'}</small>}</span>
              <span className="track"><span className="fill" style={{ width: `${Math.max(0, (s / top) * 100)}%` }} /></span>
              <span className="v" key={s}>{s}</span>
            </li>
          );
        })}
      </ol>
      </div>
    </div>
  );
}

/** The nine holes as they are right now. A pop-up that has just gone slides back down (it keeps its picture). */
function BoardCells({ showing }: { showing: ({ kind: 'elf' | 'gold' | 'grump' } | undefined)[] }) {
  const last = useRef<({ kind: 'elf' | 'gold' | 'grump' } | undefined)[]>([]);
  return (
    <>
      {showing.map((it, c) => {
        if (it) last.current[c] = { kind: it.kind };
        const shown = last.current[c];
        const state: HoleState = it ? 'up' : 'down';
        return <div key={c} className={`wh-cell ${state}${shown ? ` ${shown.kind}` : ''}`}><WhackHole kind={shown?.kind} state={state} /></div>;
      })}
    </>
  );
}
