// The end-of-night extra prize: the Party MVP (best Party Score = party-game points + correct
// duel calls). The TV rolls through the contenders like a slot machine, lands on the winner and
// rains chips; the MVP's own phone gets the same moment (and a buzz), everyone else is told who.
import { useEffect, useMemo, useState } from 'react';
import { partyMvps, partyScore } from '../core/scoring';
import type { Game, PlayerId } from '../core/types';
import { Avatar } from './avatar';
import './mvp.css';

const ROLL_MS = 2600;

export function useMvps(game: Game) {
  return useMemo(() => partyMvps(game.players), [game.players]);
}

export function MvpReveal({ game, me, variant, onClose }: { game: Game; me?: PlayerId; variant: 'tv' | 'phone'; onClose: () => void }) {
  const mvps = useMvps(game);
  const players = game.players || {};
  const names = mvps.map((id) => players[id]?.name || '?');
  const winnerLabel = names.join(' & ');
  const contenders = useMemo(() => {
    const scored = Object.values(players).filter((p) => partyScore(p) > 0).map((p) => p.name);
    return scored.length > 1 ? scored : Object.values(players).map((p) => p.name);
  }, [players]);
  const [rolling, setRolling] = useState(true);
  const [shown, setShown] = useState(contenders[0] || winnerLabel);

  useEffect(() => {
    // Slot-machine roll: fast, then slower, then the winner
    let i = 0;
    let delay = 70;
    let elapsed = 0;
    let id: number;
    const tick = () => {
      elapsed += delay;
      if (elapsed >= ROLL_MS) { setShown(winnerLabel); setRolling(false); return; }
      i = (i + 1) % contenders.length;
      setShown(contenders[i]);
      delay = Math.min(320, delay * 1.12);
      id = window.setTimeout(tick, delay);
    };
    id = window.setTimeout(tick, delay);
    return () => clearTimeout(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const iWon = !!me && mvps.includes(me);
  useEffect(() => {
    if (!rolling && iWon && variant === 'phone') navigator.vibrate?.([120, 60, 120, 60, 300]);
  }, [rolling]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!mvps.length) return null;
  const top = players[mvps[0]];
  const heading = variant === 'phone' && iWon ? (mvps.length > 1 ? "You're a joint Party MVP" : "You're the Party MVP") : mvps.length > 1 ? 'Joint Party MVPs' : 'Party MVP';
  return (
    <div className={`mvp-stage ${variant}${rolling ? ' rolling' : ' landed'}${iWon ? ' mine' : ''}`} id="mvp-reveal" role="dialog" aria-label={heading} onClick={rolling ? undefined : onClose}>
      {!rolling && <ChipRain count={variant === 'tv' ? 26 : 14} />}
      <div className="mvp-body">
        <span className="fz-sticker yel mvp-sticker">The extra prize</span>
        <div className="mvp-heading f6-disp">{heading}</div>
        <div className="mvp-roll" aria-live="polite">
          {!rolling && mvps.length === 1 && <Avatar name={top?.name || '?'} avatar={top?.avatar} size={variant === 'tv' ? 96 : 64} />}
          <span className="mvp-name f6-disp fz-riso" data-testid="mvp-name">{rolling ? shown : winnerLabel}</span>
        </div>
        {!rolling && top && (
          <div className="mvp-score">
            <span className="f6-num"><strong>{top.partyPoints || 0}</strong> party-game points</span>
            <span aria-hidden>+</span>
            <span className="f6-num"><strong>{top.votingWins || 0}</strong> correct duel calls</span>
            <span aria-hidden>+</span>
            <span className="f6-num"><strong>{top.duelsFought || 0}</strong> duels fought</span>
            <span aria-hidden>=</span>
            <span className="mvp-total f6-num">{partyScore(top)}</span>
          </div>
        )}
        {!rolling && <div className="mvp-hint">{variant === 'phone' && !iWon ? `Cheer for ${winnerLabel}!` : 'Tap to see the final standings'}</div>}
      </div>
    </div>
  );
}

function ChipRain({ count }: { count: number }) {
  const drops = useMemo(() => Array.from({ length: count }, (_, i) => ({
    left: Math.random() * 100, delay: Math.random() * 1.6, dur: 2.4 + Math.random() * 1.8, size: 22 + Math.random() * 26, key: i,
  })), [count]);
  return (
    <div className="mvp-rain" aria-hidden>
      {drops.map((d) => (
        <span key={d.key} className="jb-chip mvp-drop" style={{ left: `${d.left}%`, animationDelay: `${d.delay}s`, animationDuration: `${d.dur}s`, ['--size' as string]: `${d.size}px` }} />
      ))}
    </div>
  );
}
