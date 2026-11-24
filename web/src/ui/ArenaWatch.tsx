// A phone that is not in the mini-game (the table is watching a duel): the same full-screen frame the players'
// phones have (ui/arenaPhone.css), saying what is going on and to look up. It used to be the app's header over a
// small white card: a different app from the one the people playing were holding.
import type { ReactNode } from 'react';
import { Icon } from './icons';
import './arenaPhone.css';

export function ArenaWatch({ title, line, score, testid }: { title: string; line: string; score?: ReactNode; testid: string }) {
  return (
    <div className="ap ap-watch" data-testid={testid}>
      <div className="ap-bar"><span className="ap-pill">Watching</span><span className="ap-mid" /><span className="ap-clock">On the TV</span></div>
      <div className="ap-watch-body">
        <span className="ap-watch-eye mo-pop" aria-hidden><Icon name="eye" /></span>
        <h2 className="ap-watch-title mo-up">{title}</h2>
        {score != null && <p className="ap-watch-score" data-testid={`${testid.replace(/-card$/, '')}-score`}>{score}</p>}
        <p className="ap-watch-line mo-up">{line}</p>
      </div>
    </div>
  );
}
