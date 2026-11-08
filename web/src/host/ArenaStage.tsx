// A runtime mini-game on the TV and the host console (core/minigames). Pass the Gremlin: who has
// it, big; what kind of challenge they are doing (never the challenge itself); how jumpy it is; where
// it has been. Never the answer or the deadline (the room never sends them). The TV plays the jumps
// and the doom out loud; the console stays quiet (the host's phone has its own sounds).
import { useEffect, useRef, useState } from 'react';
import { GremlinFace } from '../player/GremlinCard';
import type { GremlinView } from '../player/GremlinCard';
import { Avatar } from '../ui/avatar';
import { play } from '../ui/sound';
import '../ui/gremlin.css';
import { useHost } from './HostContext';
import { RingTossStage } from './RingTossStage';

export function ArenaStage() {
  const { game } = useHost();
  if (game.arenaView?.gameId === 'reindeer_ring_toss') return <RingTossStage />;
  return <GremlinStage />;
}

function GremlinStage() {
  const { game, tv } = useHost();
  const av = game.arenaView;
  const v = av?.gameId === 'pass_the_gremlin' ? (av.view as GremlinView) : null;
  const [jumped, setJumped] = useState(false);
  const was = useRef<{ holder?: string; stage?: string }>({});
  useEffect(() => {
    if (!v) return;
    const prev = was.current;
    if (prev.holder && prev.holder !== v.holder) {
      setJumped(true);
      if (tv) play('pass');
      const t = setTimeout(() => setJumped(false), 500);
      was.current = { holder: v.holder, stage: v.stage };
      return () => clearTimeout(t);
    }
    if (tv && v.stage === 'doomed' && prev.stage !== 'doomed') play('doom');
    was.current = { holder: v.holder, stage: v.stage };
  }, [v?.holder, v?.stage]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!av) return null;
  if (!v) return <div>{av.gameId}</div>;
  const name = (id?: string | null) => (id && game.players?.[id]?.name) || 'Someone';
  const holder = game.players?.[v.holder];
  return (
    <div className={`grem tv danger-${v.danger}${jumped ? ' jumped' : ''}${v.stage === 'doomed' ? ' doomed' : ''}`} data-testid="gremlin-stage">
      {(v.round === 'sudden_death' || v.stage === 'sudden_death_intro') && <span className="grem-banner">Sudden death: just the duellists</span>}
      {av.lifecycle === 'paused' && <span className="grem-banner">Paused</span>}
      <div className={`grem-face${v.stage === 'doomed' ? ' grem-boom' : ''}`} aria-hidden>{v.stage === 'doomed' ? '💥' : <GremlinFace danger={v.danger} />}</div>
      {v.stage === 'doomed' ? <div className="grem-holder" role="status">It went berserk on {name(v.doomed)}!</div>
        : v.stage === 'sudden_death_intro' ? <div className="grem-holder" role="status">It got {name(v.bystanderDoomed)}!</div>
        : <div className="grem-holder" data-testid="gremlin-tv-holder" aria-live="polite">{holder && <Avatar name={holder.name} avatar={holder.avatar} size={56} />} {name(v.holder)} has it</div>}
      <p className="grem-sub grem-kind" data-testid="gremlin-tv-kind">
        {v.stage === 'pass_ready' ? 'Done! Passing it on…' : v.stage === 'holding' ? `Doing ${v.challengeLabel || 'a challenge'} to pass it on` : ''}
      </p>
      <div className={`grem-meter ${v.danger}`} style={{ maxWidth: 520 }} role="img" aria-label={`The Gremlin is ${v.danger}`}><span /></div>
      {!!v.trail.length && <div className="grem-trail" style={{ fontSize: 16 }}>{v.trail.slice(-8).map((p, i) => <span key={i}>{name(p.from)} → {name(p.to)}{p.random ? ' 🎲' : ''}</span>)}</div>}
    </div>
  );
}
