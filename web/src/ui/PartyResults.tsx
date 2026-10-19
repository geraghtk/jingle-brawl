// Party game results list — shared by the phone and the TV.
import { useEffect, useState } from 'react';
import { GAMES } from '../core/party';
import type { PartyGame } from '../core/party';
import type { Game } from '../core/types';

export function PartyResults({ mg, game, me }: { mg: PartyGame; game: Game; me?: string }) {
  const r = mg.results!;
  const tied = mg.tiedBidders || [];
  const name = (id: string) => game.players?.[id]?.name || '?';
  const unit = mg.type === 'sketch_off' ? 'vote' : 'pt';
  return (
    <>
      <h3>{GAMES[mg.type].emoji} Results</h3>
      <p className="pg-sub">
        {r.winners.map(name).join(' & ')} {r.winners.length > 1 ? 'win' : 'wins'} the tie
        {mg.minigameGoal === 'pick_two' ? ' and duel for the gift' : ' and duel the Opener'}.
      </p>
      <ol className="pg-rank">
        {r.ranking.map((id, i) => {
          const s = r.scoreOf?.[id];
          return (
            <li key={id} className={`${tied.includes(id) ? 'tied' : ''} ${r.winners.includes(id) ? 'win' : ''}`}>
              <span className="pos">{i + 1}</span>
              <span className="nm">{name(id)}{id === me ? ' (you)' : ''}</span>
              <span>{s == null ? '—' : `${s} ${unit}${Math.abs(s) === 1 ? '' : 's'}`}</span>
              {r.winners.includes(id) ? <span className="tag">wins the tie</span> : tied.includes(id) ? <span className="tag">tied</span> : null}
              {!!r.points?.[id] && <span className="tag">+{r.points[id]} party pts</span>}
              {r.bonus === id && <span className="tag">+1 chip Elf Bonus</span>}
            </li>
          );
        })}
      </ol>
    </>
  );
}

/** Seconds left in the current phase, ticking. */
export function useSecondsLeft(endsAt: number | undefined, offset: number): number {
  const calc = () => Math.max(0, Math.ceil(((endsAt || 0) - (Date.now() + offset)) / 1000));
  const [s, setS] = useState(calc);
  useEffect(() => {
    setS(calc());
    const id = setInterval(() => setS(calc()), 250);
    return () => clearInterval(id);
  }, [endsAt, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  return s;
}
