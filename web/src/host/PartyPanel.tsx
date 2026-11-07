// Party mini-games on the TV: rules + who's tied, live Whack-an-Elf leaderboard,
// the Sketch-Off gallery while phones vote, then the results.
import { GAMES, rank, whackScores } from '../core/party';
import type { PartyGame } from '../core/party';
import { useServerOffset } from '../net/hooks';
import { PartyResults, useSecondsLeft } from '../ui/PartyResults';
import '../ui/party.css';
import { useHost } from './HostContext';
import { Glyphs, Icon } from '../ui/icons';
import { WhackLegend } from '../ui/whack';

export function PartyPanel({ mg }: { mg: PartyGame }) {
  const { game, cfg } = useHost();
  const offset = useServerOffset();
  const secs = useSecondsLeft(mg.phaseEndsAt, offset);
  const g = GAMES[mg.type];
  const name = (id: string) => game.players?.[id]?.name || '?';
  const ids = mg.participants || [];
  let body;
  if (mg.phase === 'intro') {
    body = <>
      <h3><Glyphs>{g.emoji}</Glyphs> {g.label}!</h3>
      <p className="pg-sub">{g.rules}</p>
      {mg.type === 'whack_elf' && <WhackLegend />}
      {mg.prompt && <div className="pg-prompt">“{mg.prompt}”</div>}
      <p className="pg-sub">Tied: <b>{(mg.tiedBidders || []).map(name).join(', ')}</b>. Everyone else plays for party points{cfg.partyPrize === 'points_chips' ? ' and the Elf Bonus chip' : ''}.</p>
      <div className="pg-timer">{secs}s</div>
    </>;
  } else if (mg.phase === 'playing' && mg.type === 'sketch_off') {
    body = <><h3><Icon name="pencil" /> Drawing “{mg.prompt}”</h3><div className="pg-timer">{secs}s</div><p className="pg-sub">{Object.keys(mg.submissions || {}).length}/{ids.length} drawings in</p></>;
  } else if (mg.phase === 'playing') {
    const scores = whackScores(mg.scores);
    const top = Math.max(5, ...Object.values(scores));
    body = <>
      <h3><Icon name="mallet" /> Whack-an-Elf!</h3><div className="pg-timer">{secs}s</div>
      <div className="pg-bars">{rank(ids, scores, mg.seed).map((id) => (
        <div className="pg-bar" key={id}>
          <span className="nm">{name(id)}</span>
          <span className="track"><span className="fill" style={{ width: `${Math.max(0, ((scores[id] || 0) / top) * 100)}%` }} /></span>
          <span className="v">{scores[id] || 0}</span>
        </div>
      ))}</div>
    </>;
  } else if (mg.phase === 'voting') {
    const order = rank(Object.keys(mg.submissions || {}), {}, mg.seed);
    body = <>
      <h3><Icon name="vote" /> Vote on your phones!</h3>
      <p className="pg-sub">“{mg.prompt}” · {Object.keys(mg.votes || {}).length}/{ids.length} voted</p>
      <div className="pg-timer">{secs}s</div>
      <div className="pg-gallery">{order.map((pid, i) => <figure key={pid}><img src={mg.submissions![pid]} alt={`drawing ${i + 1}`} /><figcaption>#{i + 1}</figcaption></figure>)}</div>
    </>;
  } else if (mg.phase === 'results' && mg.results) {
    body = <PartyResults mg={mg} game={game} />;
  }
  return <div className="party-host pg" style={{ gridColumn: '1 / -1' }}>{body}</div>;
}
