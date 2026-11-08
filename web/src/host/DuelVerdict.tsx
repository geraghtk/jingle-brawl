// The TV's big moment after every duel: the winner's corner colour floods the screen, the pot
// arcs from the loser to the winner (their count ticks up as each chip lands), Santa's cut drops
// into the North Pole Fund, and the people who called it get named.
import { useEffect, useRef, useState } from 'react';
import type { LastDuel, Turn } from '../core/types';
import { Avatar } from '../ui/avatar';
import { flyChips } from '../ui/chips';
import { useHost } from './HostContext';

const SHOW_MS = 6500;

/** chipsBefore: null when this screen may not know the winner's chips (the TV, public chips off): it counts the win instead. */
interface Verdict { duel: LastDuel; turn: Turn; chipsBefore: number | null }

export function DuelVerdict() {
  const { game } = useHost();
  const t = game.currentTurn;
  const lastTurn = useRef<Turn | null>(null);
  if (t?.phase === 'duel' && !t.isTiebreak) lastTurn.current = t;

  const [v, setV] = useState<Verdict | null>(null);
  const seen = useRef<number | null | undefined>(undefined);
  const at = typeof game.lastDuel?.at === 'number' ? game.lastDuel.at : null;
  useEffect(() => {
    if (seen.current === undefined || at === seen.current) { seen.current = at; return; } // not on (re)load
    seen.current = at;
    const duel = game.lastDuel;
    const turn = lastTurn.current;
    if (!duel || !turn) return;
    const now = game.players?.[duel.winnerId]?.chips;
    setV({ duel, turn, chipsBefore: typeof now === 'number' ? Math.max(0, now - duel.pot) : null });
  }, [at]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!v) return;
    const id = setTimeout(() => setV(null), SHOW_MS);
    return () => clearTimeout(id);
  }, [v]);

  if (!v) return null;
  return <VerdictStage key={String(v.duel.at)} v={v} onClose={() => setV(null)} />;
}

function VerdictStage({ v, onClose }: { v: Verdict; onClose: () => void }) {
  const { game } = useHost();
  const { duel, turn } = v;
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const winnerIsChallenger = duel.winnerId === turn.challengerId;
  const gift = turn.giftId ? game.gifts?.[turn.giftId]?.name : undefined;
  const tax = turn.tax || 0;
  const callers = Object.entries(turn.duelVotes || {}).filter(([, x]) => x.votedFor === duel.winnerId).map(([id]) => name(id));
  const voters = Object.keys(turn.duelVotes || {}).length;

  const base = v.chipsBefore ?? 0;
  const [chips, setChips] = useState(base);
  const [bank, setBank] = useState(0);
  const loser = useRef<HTMLDivElement>(null);
  const winner = useRef<HTMLDivElement>(null);
  const fund = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const ids: number[] = [];
    ids.push(window.setTimeout(() => {
      if (duel.pot > 0) {
        flyChips({
          from: loser.current, to: winner.current, count: duel.pot, size: 46, stagger: 120, duration: 1000,
          onLand: (n) => setChips(base + Math.min(n, duel.pot)),
          onDone: () => setChips(base + duel.pot), // more than 12 chips: only 12 are drawn
        });
      }
    }, 700));
    if (tax > 0) {
      ids.push(window.setTimeout(() => flyChips({
        from: loser.current, to: fund.current, count: tax, size: 30, onLand: (n) => setBank(n), onDone: () => setBank(tax),
      }), 700 + 120 * Math.min(duel.pot, 12) + 500));
    }
    return () => ids.forEach(clearTimeout);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const outcome = gift
    ? (duel.winnerId === turn.defenderId ? `keeps the ${gift}` : `takes the ${gift}`)
    : 'wins the duel';
  return (
    <div className={`verdict-stage ${winnerIsChallenger ? 'red' : 'green'}`} id="duel-verdict" role="status" onClick={onClose}>
      <div className="verdict-halftone fz-halftone" />
      <div className="verdict-body">
        <span className="fz-sticker ink verdict-sticker">{winnerIsChallenger ? 'Red corner wins' : 'Green corner wins'}</span>
        <h1 className="verdict-name f6-disp fz-riso">{name(duel.winnerId)}</h1>
        <p className="verdict-outcome">{outcome}</p>

        <div className="verdict-purse">
          <div className="purse-tile loser" ref={loser} data-duel-corner={duel.loserId}>
            <Avatar name={name(duel.loserId)} avatar={game.players?.[duel.loserId]?.avatar} size={56} />
            <div className="purse-name">{name(duel.loserId)}</div>
            <div className="purse-note">
              {duel.loserPaid > 0 ? `paid ${duel.loserPaid}` : 'defended for free'}
              {duel.dividend > 0 && <> · +{duel.dividend} back</>}
            </div>
          </div>
          <div className="purse-arc" aria-hidden />
          <div className="purse-tile winner" ref={winner} data-duel-corner={duel.winnerId}>
            <Avatar name={name(duel.winnerId)} avatar={game.players?.[duel.winnerId]?.avatar} size={56} />
            <div className="purse-name">{name(duel.winnerId)}</div>
            <div className="purse-chips f6-num" data-testid="verdict-winner-chips"><span className="jb-chip" />{v.chipsBefore == null ? `+${chips}` : chips}</div>
          </div>
        </div>

        <div className="verdict-foot">
          {voters > 0 && (
            <div className="verdict-callers" data-testid="verdict-callers">
              {callers.length ? <>Called it: <strong>{callers.join(', ')}</strong></> : 'Nobody saw that coming'}
            </div>
          )}
          {tax > 0 && (
            <div className="verdict-fund" ref={fund}>
              <span className="jb-chip" /> North Pole Fund <strong className="f6-num">+{bank}</strong>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
