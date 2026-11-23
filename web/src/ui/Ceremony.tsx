// How every mini-game ends, on every screen: the winners come in, paper falls, the Party points they
// earned count up. It comes up a moment after the game's own ending (which stays behind it) and stays
// until the room moves on. What it says is core/minigames/ceremony.ts; this only draws it.
import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import { CEREMONY_DELAY_MS, ceremonyFor, myEnding } from '../core/minigames/ceremony';
import type { CeremonyRow } from '../core/minigames/ceremony';
import type { ArenaView } from '../core/minigames/runtime';
import type { Game } from '../core/types';
import { Avatar } from './avatar';
import { Confetti, CountUp } from './motion';
import { buzz, play } from './sound';
import './arena.css';

const MINE: Record<string, string> = { won: 'You won!', lost: 'Not this time.', safe: 'You made it!', level: 'Too close to call!', watched: '' };

export function Ceremony({ game, av, now, where, me }: { game: Game; av: ArenaView; now: number; where: 'tv' | 'console' | 'phone'; me?: string }) {
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const practice = !!game.currentTurn?.lab;
  const c = av.practice ? null : ceremonyFor(av.result, name, practice);   // a practice round decided nothing: no announcement
  const up = !!c && av.endedAt != null && now >= av.endedAt + CEREMONY_DELAY_MS;
  const mine = me ? myEnding(av.result, me, av.participants) : 'watched';
  const myPoints = me && c ? [...c.stars, ...c.also].find((r) => r.id === me)?.points || 0 : 0;

  // once, as it comes up: the points land with a chime on the TV, a phone that won buzzes
  const rung = useRef(false);
  useEffect(() => {
    if (!up || rung.current) return;
    rung.current = true;
    if (where === 'tv' && c && [...c.stars, ...c.also].some((r) => r.points > 0)) play('ding');
    if (where === 'phone' && mine === 'won') buzz([40, 60, 120]);
  }, [up, where, c, mine]);

  if (!c || !up) return null;
  const size = where === 'tv' ? 96 : 56;
  const person = (r: CeremonyRow, i: number, cls: string) => (
    <li key={r.id} className={`cer-person ${cls} mo-pop`} style={{ ['--i']: i + 2 } as CSSProperties} data-testid={`ceremony-player-${r.id}`}>
      <Avatar name={name(r.id)} avatar={game.players?.[r.id]?.avatar} size={size} />
      <span className="cer-name">{name(r.id)}</span>
      {r.points > 0 && <span className="cer-points"><CountUp value={r.points} from={0} prefix="+" delay={500 + i * 70} /> <i aria-hidden>★</i></span>}
    </li>
  );
  const win = c.tone === 'win' && (where !== 'phone' || mine === 'won' || mine === 'watched');
  return (
    <div className={`cer cer-${where} cer-${c.tone}`} role="status" aria-label="How the game ended" data-testid="arena-ceremony">
      <div className="cer-card">
        {win && <Confetti n={where === 'phone' ? 40 : 90} seed={av.stateVersion || 1} forever={practice} />}
        {where === 'phone' && MINE[mine] && <div className={`cer-mine cer-mine-${mine} mo-slam`} data-testid="ceremony-mine">{MINE[mine]}</div>}
        <h2 className={`cer-headline ${where === 'phone' && MINE[mine] ? 'mo-up' : 'mo-slam'}`} data-testid="ceremony-headline">{c.headline}</h2>
        {c.sub && <p className="cer-sub mo-up" style={{ ['--i']: 1 } as CSSProperties}>{c.sub}</p>}
        {c.stars.length > 0 && <ul className="cer-stars" aria-label="The winners">{c.stars.map((r, i) => person(r, i, 'star'))}</ul>}
        {c.scores.length > 0 && (
          <ol className="cer-scores" aria-label="The scores">
            {c.scores.map((r, i) => (
              <li key={r.id} className="mo-up" style={{ ['--i']: i + 3 } as CSSProperties}>
                <span className="cer-rank">{r.rank ?? i + 1}</span>
                <Avatar name={name(r.id)} avatar={game.players?.[r.id]?.avatar} size={where === 'tv' ? 48 : 30} />
                <span className="cer-name">{name(r.id)}</span>
                <b className="cer-score"><CountUp value={r.score || 0} from={0} delay={300 + i * 70} /></b>
              </li>
            ))}
          </ol>
        )}
        {c.also.length > 0 && (
          <>
            <p className="cer-also-k mo-fade" style={{ ['--i']: 5 } as CSSProperties}>Party points too</p>
            <ul className="cer-also" aria-label="Also earning Party points">{c.also.map((r, i) => person(r, i + 3, 'also'))}</ul>
          </>
        )}
        {where === 'phone' && myPoints > 0 && <p className="cer-my-points mo-pop" style={{ ['--i']: 6 } as CSSProperties} data-testid="ceremony-my-points">+{myPoints} Party {myPoints === 1 ? 'point' : 'points'} for you</p>}
        {practice && <p className="cer-practice mo-fade" style={{ ['--i']: 6 } as CSSProperties}>A practice round: nothing won or lost.</p>}
      </div>
    </div>
  );
}
