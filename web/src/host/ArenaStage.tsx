// A runtime mini-game on the TV and the host console (core/minigames). Pass the Gremlin: who has
// it, big; how jumpy it is; where it has been. Never the answer or the deadline (the room never
// sends them).
import type { GremlinView } from '../player/GremlinCard';
import { Avatar } from '../ui/avatar';
import '../ui/gremlin.css';
import { useHost } from './HostContext';

export function ArenaStage() {
  const { game } = useHost();
  const av = game.arenaView;
  if (!av) return null;
  if (av.gameId !== 'pass_the_gremlin') return <div>{av.gameId}</div>;
  const v = av.view as GremlinView;
  const name = (id?: string | null) => (id && game.players?.[id]?.name) || 'Someone';
  const holder = game.players?.[v.holder];
  return (
    <div className={`grem tv danger-${v.danger}`} data-testid="gremlin-stage">
      {(v.round === 'sudden_death' || v.stage === 'sudden_death_intro') && <span className="grem-banner">Sudden death: just the duellists</span>}
      {av.lifecycle === 'paused' && <span className="grem-banner">Paused</span>}
      <div className="grem-face" aria-hidden>{v.stage === 'doomed' ? '💥' : '👹'}</div>
      {v.stage === 'doomed' ? <div className="grem-holder">It went berserk on {name(v.doomed)}!</div>
        : v.stage === 'sudden_death_intro' ? <div className="grem-holder">It got {name(v.bystanderDoomed)}!</div>
        : <div className="grem-holder" data-testid="gremlin-tv-holder">{holder && <Avatar name={holder.name} avatar={holder.avatar} size={56} />} {name(v.holder)} has it</div>}
      <p className="grem-sub">{v.stage === 'pass_ready' ? 'Solved! Passing it on…' : v.stage === 'holding' ? 'Solving a quick math problem to pass it on' : ''}</p>
      <div className={`grem-meter ${v.danger}`} style={{ maxWidth: 520 }}><span /></div>
      {!!v.trail.length && <div className="grem-trail" style={{ fontSize: 16 }}>{v.trail.slice(-8).map((p, i) => <span key={i}>{name(p.from)} → {name(p.to)}{p.random ? ' 🎲' : ''}</span>)}</div>}
    </div>
  );
}
