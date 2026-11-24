// The sealed bids, opened, on the phone: everyone's bid highest first, the top bid(s) marked, my own
// row marked, a tie said plainly. The TV shows the same reveal; a guest looking at their phone no
// longer misses it. What to list (and that nothing is listed before the reveal) is core/reveal.ts.
import type { CSSProperties, ReactNode } from 'react';
import { revealedBids, revealSummary } from '../core/reveal';
import { Avatar } from '../ui/avatar';
import { FitName } from '../ui/FittedName';
import { Icon } from '../ui/icons';
import { usePlayer } from './PlayerContext';
import '../ui/motion.css';
import './revealedBids.css';

/** `next`: what the table is waiting for now, under the list. */
export function RevealedBids({ what, next, id }: { what?: string; next?: ReactNode; id?: string }) {
  const { game, me } = usePlayer();
  const r = revealedBids(game);
  if (!r) return null;
  const max = Math.max(1, r.top);
  const summary = revealSummary(game, r, me);
  return (
    <div className="card rb" id={id} data-testid="revealed-bids" data-tie={r.tie ? r.leaders.length : undefined}>
      <div className="f6-cap rb-cap"><Icon name="envelope" /> The bids are in{what ? <> for <span className="rb-what">{what}</span></> : null}</div>
      <div className={`rb-summary f6-disp${r.tie ? ' tie' : ''}${summary.length > 44 ? ' long' : ''}`} data-testid="bids-summary">{summary}</div>
      <ol className="rb-rows">
        {r.rows.map((row, i) => {
          const p = game.players?.[row.id];
          const mine = row.id === me;
          return (
            <li key={row.id} className={`rb-row mo-up${row.top ? ' top' : ''}${mine ? ' mine' : ''}${row.amount > 0 ? '' : ' pass'}`}
              style={{ ['--i']: i, ['--fill']: `${Math.round((row.amount / max) * 100)}%` } as CSSProperties}
              data-testid={`bid-row-${row.id}`} data-top={row.top || undefined} data-mine={mine || undefined}>
              <span className="rb-place f6-num" aria-hidden>{row.top ? <Icon name="crown" /> : row.place ?? ''}</span>
              <Avatar name={p?.name || '?'} avatar={p?.avatar} size={30} />
              <span className="rb-name">
                <FitName name={p?.name || 'Someone'} chars={14} lines={2} min={0.6} inline />
                {mine && <span className="rb-you">You</span>}
                {row.top && <span className="sr-only">{r.tie ? ', tied for the top bid' : ', the top bid'}</span>}
              </span>
              <span className="rb-amount f6-disp f6-num" data-testid={`bid-amount-${row.id}`}>
                {row.amount > 0 ? <>{row.amount} <span className="f6-coin" /></> : row.missing ? 'no bid' : 'pass'}
              </span>
            </li>
          );
        })}
      </ol>
      {next && <div className="rb-next" data-testid="bids-next">{next}</div>}
    </div>
  );
}
