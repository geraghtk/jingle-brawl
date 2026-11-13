// A mini-game full screen, by default: on the TV always while one is on (the room's big screen is the
// arena), and on the console unless the host goes back to the console view (their PC may be the big
// screen); the console's buttons for the game are in its header. One stage at a time: the
// turn panel leaves its own copy out meanwhile, so sounds and animations aren't doubled.
import { createContext, useContext, useEffect, useState } from 'react';
import { duelAppInfo } from '../core/duelgame';
import { Glyphs } from '../ui/icons';
import { ArenaStage } from './ArenaStage';
import { useHost } from './HostContext';
import { PanelButtons } from './TurnPanel';

export const FullCtx = createContext<{ full: boolean; setFull: (on: boolean) => void }>({ full: false, setFull: () => {} });
export const useArenaFull = () => useContext(FullCtx);

/** Is a mini-game on screen now (running, or its result)? */
export function useArenaOn(): boolean {
  const { game } = useHost();
  return game.currentTurn?.phase === 'minigame_arena' && !!game.arenaView;
}

export function ArenaFull() {
  const { game, tv } = useHost();
  const { setFull } = useArenaFull();
  const info = duelAppInfo(game.arenaView?.gameId);
  const [screenFull, setScreenFull] = useState(() => typeof document !== 'undefined' && !!document.fullscreenElement);
  useEffect(() => {
    const on = () => setScreenFull(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', on);
    return () => document.removeEventListener('fullscreenchange', on);
  }, []);
  const lab = !!game.currentTurn?.lab;
  return (
    <div className="arena-full" data-testid="arena-full">
      <header className="arena-full-head">
        <span className="arena-full-title"><Glyphs>{`${info?.emoji ?? '🎮'} ${info?.label ?? 'Mini-game'}`}</Glyphs>{lab && <small> · practice</small>}</span>
        <span className="arena-full-tools">
          {!screenFull && document.fullscreenEnabled && (
            <button type="button" className="arena-full-btn" onClick={() => document.documentElement.requestFullscreen?.().catch(() => {})}>⛶ Fill the screen</button>
          )}
          {!tv && <span className="arena-full-host"><PanelButtons /></span>}
          {!tv && <button type="button" className="arena-full-btn" data-testid="arena-full-exit" onClick={() => setFull(false)}>✕ Console view</button>}
        </span>
      </header>
      <div className="arena-full-body"><ArenaStage /></div>
    </div>
  );
}
