// The ending every game shares, drawn: the headline lands, the winners come in, paper falls, Party points
// count up, the scores line up. ui/Ceremony.tsx gives it a runtime mini-game's result, PartyCeremony (there
// too) a party game's. When the game decided a duel it also says what the duel was for (`stakes`: the gift,
// the pot, what the loser gets), so nothing announces that duel a second time afterwards.
import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import { flyChips } from './chips';
import type { Ceremony as CeremonyWords, CeremonyRow } from '../core/minigames/ceremony';
import type { DuelStakes } from '../core/stakes';
import type { Game } from '../core/types';
import { Avatar } from './avatar';
import { FitName } from './FittedName';
import { Confetti, CountUp } from './motion';
import { StakesLines } from './Stakes';
import './arena.css';
import './ceremony.css';

export type Where = 'tv' | 'console' | 'phone';
export type Mine = 'won' | 'lost' | 'safe' | 'level' | 'watched';
const MINE: Record<Mine, string> = { won: 'You won!', lost: 'Not this time.', safe: 'You made it!', level: 'Too close to call!', watched: '' };

/** How many score rows fit: the TV's card must not grow off the screen; a phone's scrolls. */
export const SCORE_ROWS: Record<Where, number> = { tv: 8, console: 10, phone: 30 };

export interface CeremonyCardProps {
  c: CeremonyWords;
  game: Game;
  where: Where;
  /** How it went for the person holding this phone. */
  mine?: Mine;
  me?: string;
  /** The Game Lab: nothing won or lost (the paper keeps falling, a line says so). */
  practice?: boolean;
  seed?: number;
  /** The duel this game decided: what the winner takes. */
  stakes?: DuelStakes | null;
  /** A picture for the card (Sketch-Off's top drawing). */
  picture?: { img: string; caption: string } | null;
  /** In the stage's own space instead of over the whole screen (a party game's stage has nothing else to show). */
  inline?: boolean;
  testid?: string;
}

/** The card's own pieces have come in by now. */
const CHIPS_AFTER_MS = 1100;
/** How many chips the winner is paid by this duel (the pot and a Misfit Toll): the ones that fly. */
export const chipsWon = (s: Pick<DuelStakes, 'pot' | 'toll'>) => Math.max(0, (s.pot || 0) + (s.toll || 0));

export function CeremonyCard({ c, game, where, mine = 'watched', me, practice = false, seed = 1, stakes, picture, inline, testid = 'arena-ceremony' }: CeremonyCardProps) {
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const size = where === 'tv' ? 96 : 56;
  // the chips the duel was for fly from the words that name them to the winner's picture (Kevin, 2026-11-23: the old
  // verdict's flying chips went with it, and he missed them), once the card has landed
  const card = useRef<HTMLDivElement>(null);
  const chips = stakes ? chipsWon(stakes) : 0, chipsTo = stakes?.winnerId;
  useEffect(() => {
    if (!chips || !chipsTo) return;
    const t = setTimeout(() => {
      const el = card.current; if (!el) return;
      const to = el.querySelector(`[data-testid="ceremony-player-${chipsTo}"] .f6-avatar, [data-testid="ceremony-player-${chipsTo}"]`) || el.querySelector('.cer-person');
      flyChips({ from: el.querySelector('[data-testid="ceremony-prize"]'), to, count: chips, size: where === 'tv' ? 54 : where === 'console' ? 36 : 28, stagger: 110 });
    }, CHIPS_AFTER_MS);
    return () => clearTimeout(t);
  }, [chips, chipsTo, where]);
  const shownElsewhere = new Set([...c.stars, ...c.also].map((r) => r.id));
  const myPoints = me ? [...c.stars, ...c.also, ...c.scores].find((r) => r.id === me)?.points || 0 : 0;
  const person = (r: CeremonyRow, i: number, cls: string) => (
    <li key={r.id} className={`cer-person ${cls} mo-pop`} style={{ ['--i']: i + 2 } as CSSProperties} data-testid={`ceremony-player-${r.id}`}>
      <Avatar name={name(r.id)} avatar={game.players?.[r.id]?.avatar} size={size} />
      <span className="cer-name"><FitName name={name(r.id)} chars={13} lines={2} min={0.6} /></span>
      {r.points > 0 && <span className="cer-points"><CountUp value={r.points} from={0} prefix="+" delay={500 + i * 70} /> <i aria-hidden>★</i></span>}
    </li>
  );
  const win = c.tone === 'win' && (where !== 'phone' || mine === 'won' || mine === 'watched');
  // the scores that fit, and always the row of the person holding this phone
  // (on the big screen the stakes take the room of some rows: the card must stay on the screen)
  let rows = c.scores.slice(0, where === 'tv' && stakes ? Math.min(4, SCORE_ROWS.tv) : SCORE_ROWS[where]);
  if (me && !rows.some((r) => r.id === me)) { const mineRow = c.scores.find((r) => r.id === me); if (mineRow) rows = [...rows.slice(0, -1), mineRow]; }
  const unit = (n: number) => (c.unit ? ` ${c.unit[Math.abs(n) === 1 ? 0 : 1]}` : '');
  return (
    <div className={`cer cer-${where} cer-${c.tone}${inline ? ' cer-inline' : ''}`} role="status" aria-label="How the game ended" data-testid={testid}>
      <div ref={card} className={`cer-card${rows.length > 5 ? ' cer-many' : ''}`}>
        {win && <Confetti n={where === 'phone' ? 40 : 90} seed={seed} forever={practice} />}
        {where === 'phone' && MINE[mine] && <div className={`cer-mine cer-mine-${mine} mo-slam`} data-testid="ceremony-mine">{MINE[mine]}</div>}
        {/* (a phone that won alone has just been told so: not twice) */}
        {!(where === 'phone' && mine === 'won' && c.stars.length === 1) && <h2 className={`cer-headline ${where === 'phone' && MINE[mine] ? 'mo-up' : 'mo-slam'}`} data-testid="ceremony-headline">{c.headline}</h2>}
        {c.sub && <p className="cer-sub mo-up" style={{ ['--i']: 1 } as CSSProperties} data-testid="ceremony-sub">{c.sub}</p>}
        {c.stars.length > 0 && <ul className="cer-stars" aria-label="The winners">{c.stars.map((r, i) => person(r, i, 'star'))}</ul>}
        <StakesLines stakes={stakes} game={game} style={{ ['--i']: 3 } as CSSProperties} />
        {picture && picture.img.length > 6 && (
          <figure className="cer-picture mo-pop" style={{ ['--i']: 3 } as CSSProperties} data-testid="ceremony-picture">
            <img src={picture.img} alt={picture.caption} draggable={false} />
            <figcaption>{picture.caption}</figcaption>
          </figure>
        )}
        {rows.length > 0 && (
          <ol className="cer-scores" aria-label="The table's scores" style={{ ['--rows']: Math.ceil(rows.length / 2) } as CSSProperties}>
            {rows.map((r, i) => (
              <li key={r.id} className={`mo-up${c.stars.some((s) => s.id === r.id) ? ' win' : ''}${r.id === me ? ' me' : ''}${r.score == null ? ' out' : ''}`} style={{ ['--i']: i + 3 } as CSSProperties} data-testid={`ceremony-score-${r.id}`}>
                <span className="cer-rank">{r.rank ?? i + 1}</span>
                <Avatar name={name(r.id)} avatar={game.players?.[r.id]?.avatar} size={where === 'tv' ? 48 : 30} />
                <span className="cer-name">{name(r.id)}{r.tag && <small className="cer-tag">{r.tag}</small>}</span>
                {r.points > 0 && !shownElsewhere.has(r.id) && <span className="cer-points"><CountUp value={r.points} from={0} prefix="+" delay={500 + i * 70} /> <i aria-hidden>★</i></span>}
                {r.score == null ? <span className="cer-noscore">didn't play</span>
                  : <b className="cer-score"><CountUp value={r.score} from={0} delay={300 + i * 70} />{c.unit && <small>{unit(r.score)}</small>}</b>}
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
        {practice && <p className="cer-practice mo-fade" style={{ ['--i']: 6 } as CSSProperties}>Practice: nothing won, nothing lost.</p>}
      </div>
    </div>
  );
}
