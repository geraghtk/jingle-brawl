// "Spoils!" / "Ouch." — shown to the two duellists when the host publishes a result.
// Chips always fly loser → winner: into my count from "from X", or out to "to X";
// a Loser's Dividend then flies in from the Bank.
import { useEffect, useRef, useState } from 'react';
import { flyChips } from '../ui/chips';
import { usePlayer } from './PlayerContext';

interface Verdict { won: boolean; amount: number; chips: number; dividend: number; opponent?: string }

export function VerdictModal() {
  const { game, me, player } = usePlayer();
  const d = game.lastDuel;
  const key = d ? `${d.at}|${d.winnerId}` : null;
  const seen = useRef<string | null | undefined>(undefined); // first snapshot = a duel from before I (re)joined
  const [v, setV] = useState<Verdict | null>(null);
  const chipsRef = useRef<HTMLDivElement>(null);
  const partyRef = useRef<HTMLSpanElement>(null);
  const dividendRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (seen.current === undefined || key === seen.current) { seen.current = key; return; }
    seen.current = key;
    if (!d || (d.winnerId !== me && d.loserId !== me)) return;
    const won = d.winnerId === me;
    setV({
      won, amount: won ? d.pot || 0 : d.loserPaid || 0, chips: player.chips, dividend: won ? 0 : d.dividend || 0,
      opponent: game.players?.[won ? d.loserId : d.winnerId]?.name,
    });
    navigator.vibrate?.(100);
    const close = setTimeout(() => setV(null), 3000);
    return () => clearTimeout(close);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fly the chips once the modal is on screen
  useEffect(() => {
    if (!v) return;
    const counter = chipsRef.current;
    const tick = (start: number, n: number) => { if (counter) counter.textContent = String(start + n); };
    const finish = () => { if (counter) counter.textContent = String(v.chips); };
    const t = setTimeout(() => {
      if (v.won) {
        if (v.amount && partyRef.current) flyChips({ from: partyRef.current, to: counter, count: v.amount, size: 30, onLand: (n) => tick(v.chips - v.amount, n), onDone: finish });
        else finish();
        return;
      }
      const dividend = () => (v.dividend && dividendRef.current
        ? flyChips({ from: dividendRef.current, to: counter, count: v.dividend, size: 30, onLand: (n) => tick(v.chips - v.dividend, n), onDone: finish })
        : finish());
      if (v.amount && partyRef.current) flyChips({ from: counter, to: partyRef.current, count: v.amount, size: 30, onDone: dividend });
      else dividend();
    }, 350);
    return () => clearTimeout(t);
  }, [v]);

  return (
    <div className={`duel-result-modal${v ? ' active' : ''}`} id="duel-result-modal">
      <div className={`duel-result-content ${v?.won ? 'winner' : 'loser'}`} id="duel-result-content">
        {v && (
          <>
            <div className="fz-halftone verdict-halftone" />
            <div className="f6-cap" style={{ color: 'rgba(255,255,255,0.75)' }}>Verdict</div>
            <h2 className="verdict-title f6-disp fz-riso">{v.won ? 'Spoils!' : 'Ouch.'}</h2>
            <span className="f6-chip verdict-chip">{v.won ? 'Cry about it.' : 'Lose with grace? Never. Kick back.'}</span>
            {(v.won || v.amount > 0) && (
              <div className="duel-result-amount f6-disp f6-num">{v.won ? '+' : '−'}{v.amount} <span className="f6-coin" /></div>
            )}
            {v.opponent && v.amount > 0 && (
              <span className="verdict-party" id="verdict-party" ref={partyRef}>{v.won ? 'from' : 'to'} {v.opponent}</span>
            )}
            <div className="verdict-stats">
              <div className="vstat"><div className="v f6-disp f6-num" id="verdict-chips" ref={chipsRef}>{v.won ? v.chips - v.amount : v.chips - v.dividend}</div><div className="k">chips now</div></div>
              {v.won
                ? <div className="vstat"><div className="v f6-disp f6-num">+{v.amount}</div><div className="k">pot won</div></div>
                : v.dividend > 0 && <div className="vstat" id="verdict-dividend" ref={dividendRef}><div className="v f6-disp f6-num">+{v.dividend}</div><div className="k">dividend</div></div>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
