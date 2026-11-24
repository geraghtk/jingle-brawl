// Pass the Gremlin on the TV and the host console: the table. Everyone who can hold it sits round an
// oval; the Gremlin perches on whoever has it and hops across when it is passed (the last few passes
// stay drawn as fading dotted lines round the table, never through the words in its middle); the middle says who has it, what kind of challenge they are doing (never
// the challenge itself), for how long, a wrong answer as it happens, and how jumpy it is. Never the
// answer or the deadline (the room never sends them). The TV plays the jumps and the doom out loud; the
// console stays quiet (the host's phone has its own sounds).
import { useEffect, useRef, useState } from 'react';
import { avatarOf } from '../core/avatars';
import { outPoints } from '../core/minigames/gremlin/view';
import type { GremlinView } from '../core/minigames/gremlin/view';
import { GremlinFace } from '../ui/gremlin';
import { passPath, seats, seatSize } from '../ui/gremlinSeats';
import { play } from '../ui/sound';
import { useServerNow } from '../net/hooks';
import '../ui/gremlin.css';
import '../ui/gremlinTable.css';
import { useHost } from './HostContext';

const MOOD: Record<string, string> = { calm: 'Calm, for now', uneasy: 'Getting twitchy', unstable: 'About to blow!' };

export function GremlinStage() {
  const { game, tv } = useHost();
  const av = game.arenaView;
  const v = av?.gameId === 'pass_the_gremlin' ? (av.view as GremlinView) : null;
  const live = useServerNow(250);
  const [jumped, setJumped] = useState(false);
  const [wrong, setWrong] = useState(0);
  const was = useRef<{ holder?: string; stage?: string; misses?: number }>({});
  useEffect(() => {
    if (!v) return;
    const prev = was.current;
    const misses = v.misses ?? 0;
    // a wrong answer by the same holder: say so for a moment
    if (prev.holder === v.holder && prev.misses != null && misses > prev.misses) setWrong(Date.now());
    was.current = { holder: v.holder, stage: v.stage, misses };
    if (prev.holder && prev.holder !== v.holder) {
      setJumped(true);
      setWrong(0);
      if (tv) play('pass');
      const t = setTimeout(() => setJumped(false), 600);
      return () => clearTimeout(t);
    }
    if (tv && prev.stage !== v.stage && (v.stage === 'doomed' || v.stage === 'out_intro' || v.stage === 'sudden_death_intro')) play('doom');
  }, [v?.holder, v?.stage, v?.misses]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!wrong) return;
    const t = setTimeout(() => setWrong(0), 1400);
    return () => clearTimeout(t);
  }, [wrong]);
  if (!av || !v) return null;

  const now = av.pausedAt ?? live;
  const name = (id?: string | null) => (id && game.players?.[id]?.name) || 'Someone';
  // the table: everyone in the game, in the order they sat down (it doesn't change when the pool shrinks)
  const table = av.participants?.length ? av.participants : v.pool;
  const spots = seats(table.length);
  const spotOf = (id?: string | null) => { const i = id ? table.indexOf(id) : -1; return i < 0 ? null : spots[i]; };
  const size = seatSize(table.length);
  const duelists: string[] = av.duelists || [];
  const sudden = v.round === 'sudden_death' || v.stage === 'sudden_death_intro';
  const over = v.stage === 'doomed';
  // a bystander has just been caught: a beat before the next round (or the sudden death)
  const caught = v.stage === 'out_intro' || v.stage === 'sudden_death_intro';
  const victim = over ? v.doomed : caught ? v.bystanderDoomed : null;
  // the whole table is playing along with a duel: caught bystanders are out, and earn Party points for lasting
  const out = v.out || [];
  const party = av.mode === 'party_with_duelists' && duelists.length === 2;
  const pays = party && !game.currentTurn?.lab;
  const survive = v.survive ?? 0;
  const lasted = table.filter((id) => !duelists.includes(id) && !out.includes(id));
  const toGo = Math.max(0, (v.maxOut ?? 0) - out.length);
  // a duelist caught settles the duel (in a practice game too: say so, or it looks as if it just stopped)
  const winner = over && v.doomed && duelists.includes(v.doomed) ? duelists.find((d) => d !== v.doomed) : null;
  const perch = spotOf(victim ?? v.holder) ?? { x: 50, y: 50 };
  const held = v.held || {};
  const heldFor = v.holdSince != null ? Math.max(0, Math.floor((now - v.holdSince) / 1000)) : null;
  const trail = v.trail.slice(-3);

  return (
    <div className={`grem tv grem-table danger-${v.danger}${jumped ? ' jumped' : ''}${over ? ' doomed' : ''}${sudden ? ' sudden' : ''}`}
      style={{ ['--seat' as string]: `${size}cqh` }} data-testid="gremlin-stage">
      <div className="gt-felt" aria-hidden />
      {/* where it has just been: the last few passes, the newest strongest */}
      <svg className="gt-passes" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        {trail.map((p, i) => {
          const a = spotOf(p.from), b = spotOf(p.to);
          return a && b ? <path key={p.at} d={passPath(a, b, (trail.length - 1 - i) * 0.035)} className={`gt-pass age-${trail.length - 1 - i}`} data-testid="gremlin-pass" vectorEffect="non-scaling-stroke" /> : null;
        })}
      </svg>

      {table.map((id, i) => {
        const p = game.players?.[id];
        const a = avatarOf(p?.name || id, p?.avatar);
        const gone = out.indexOf(id);
        const away = !v.pool.includes(id) || gone >= 0;
        const cls = ['gt-seat', id === v.holder && !over && !caught ? 'has' : '', id === victim ? 'boom' : '', away && id !== victim ? 'out' : ''].filter(Boolean).join(' ');
        const pts = pays && gone >= 0 ? outPoints(gone, survive) : 0;
        return (
          <div key={id} className={cls} style={{ left: `${spots[i].x}%`, top: `${spots[i].y}%` }} data-testid={`gremlin-seat-${id}`}>
            <span className="gt-pic" style={{ background: a.color }}>
              <img src={`art/avatars/${a.id}.webp`} alt="" draggable={false} />
              {duelists.includes(id) && <i className="gt-duel" data-testid={`gremlin-duelist-${id}`}>Duel</i>}
            </span>
            <span className="gt-name">{name(id)}</span>
            {!!held[id] && gone < 0 && <span className="gt-held" aria-label={`held it ${held[id]} times`}>×{held[id]}</span>}
            {gone >= 0 && <span className="gt-outtag" data-testid={`gremlin-out-${id}`}>Out{pts ? ` · +${pts}` : ''}</span>}
          </div>
        );
      })}

      {/* the Gremlin: perched on whoever has it, hopping over when it is passed */}
      <div className="gt-gremlin" style={{ left: `${perch.x}%`, top: `${perch.y}%` }} aria-hidden>
        <div className={`gt-hop${over || caught ? ' grem-boom' : ''}`} key={v.holder + v.stage}>
          {victim ? <span className="gt-bang">💥</span> : <span className="grem-face"><GremlinFace danger={v.danger} /></span>}
        </div>
      </div>

      <div className="gt-mid">
        {sudden ? <span className="grem-banner">Sudden death: just the duellists</span>
          : party && (v.roundNo ?? 1) > 1 && !over ? <span className="grem-banner" data-testid="gremlin-tv-round">Round {v.roundNo} · shorter fuse</span> : null}
        {av.lifecycle === 'paused' && <span className="grem-banner">Paused</span>}
        {over ? <div className="grem-holder" role="status">It went berserk on {name(v.doomed)}!</div>
          : caught ? <div className="grem-holder" role="status" data-testid="gremlin-tv-caught">It got {name(v.bystanderDoomed)}!</div>
          : <div className="grem-holder" data-testid="gremlin-tv-holder" aria-live="polite">{name(v.holder)} has it</div>}
        <p className="grem-sub grem-kind" data-testid="gremlin-tv-kind">
          {wrong && v.stage === 'holding' ? <b className="gt-wrong" key={wrong} data-testid="gremlin-tv-wrong">Wrong! Another one…</b>
            : v.stage === 'pass_ready' ? 'Done! Passing it on…'
            : v.stage === 'holding' ? <>Doing {v.challengeLabel || 'a challenge'} to pass it on{heldFor != null && heldFor >= 2 ? <span className="gt-secs"> · {heldFor}s</span> : null}</>
            : v.stage === 'out_intro' ? "They're out. Everyone else plays on!"
            : v.stage === 'sudden_death_intro' ? "They're out. Now it's just the duellists."
            : over ? <>
              {winner && <b data-testid="gremlin-tv-verdict">{av?.practice ? 'Only practice: nobody loses a thing' : `${name(winner)} wins the duel!`}</b>}
              {pays && lasted.length > 0 && survive > 0 && <span className="gt-lasted" data-testid="gremlin-tv-lasted">Never caught: {lasted.map(name).join(', ')} · +{survive} Party point{survive === 1 ? '' : 's'} each</span>}
            </>
            : ''}
        </p>
        {!over && !caught && (
          <div className="gt-mood">
            <div className={`grem-meter ${v.danger}`} role="img" aria-label={`The Gremlin is ${v.danger}`}><span /></div>
            <span className="gt-mood-word" data-testid="gremlin-tv-mood">{MOOD[v.danger] || ''}</span>
          </div>
        )}
        {!over && <p className="gt-rule" data-testid="gremlin-tv-rule">{!party ? "Don't be holding it when it goes berserk."
          : sudden ? 'Whoever holds it when it goes berserk loses the duel.'
          : toGo === 1 && out.length ? 'One more caught, then the duellists finish it alone.'
          : 'Caught? You are out. A duellist caught loses the duel.'}</p>}
      </div>
    </div>
  );
}
