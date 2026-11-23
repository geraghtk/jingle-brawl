// The Great Gift Conveyor on a phone: your station on your team's belt, close up. Toys roll through
// it with their order on a tag; when one is in your station, tap the piece it asks for before it has
// gone by. A raccoon instead of a toy: shoo it. Small teams work two stations each (one under the
// other). The phone only says which piece: the room decides whether the toy was still there.
import { useEffect, useRef, useState } from 'react';
import { posAt } from '../core/minigames/engines/belt';
import { conveyorView } from '../core/minigames/conveyor/view';
import type { ConveyorTeamView, ConveyorView } from '../core/minigames/conveyor/view';
import { useFrameNow } from '../net/hooks';
import { secsLeft } from '../ui/clock';
import { PART_NAMES, TEAM_COLORS, VARIANT_NAMES } from '../ui/conveyor';
import { Belt, Swatch } from '../ui/conveyorBelt';
import { buzz, play } from '../ui/sound';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/conveyor.css';

const NONE = -1;
/** How much belt a phone shows before and after its station (you see a toy coming). */
const LEAD = 0.2, TAIL = 0.05;

export function ConveyorCard() {
  const { game, send } = usePlayer();
  const v = conveyorView(game.arenaView!.view);
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const live = useFrameNow();
  const now = game.arenaView!.pausedAt ?? live;          // paused: the belt stands where it was
  const mine = v.teams.find((t) => t.teamId === v.myTeam);
  const theirs = v.teams.find((t) => t.teamId !== v.myTeam);
  // my team's sleigh filled up one more, or lost one to a raccoon
  const was = useRef({ delivered: mine?.delivered ?? 0, stolen: mine?.stolen ?? 0 });
  useEffect(() => {
    if (!mine) return;
    if (mine.stolen > was.current.stolen) { play('wrong'); buzz([80, 40, 80]); } else if (mine.delivered > was.current.delivered) play('ding');
    was.current = { delivered: mine.delivered, stolen: mine.stolen };
  }, [mine?.delivered, mine?.stolen]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!mine || !theirs) return <div className="card" data-testid="conveyor-card"><p>Watch the TV: the teams are racing!</p></div>;

  const secs = secsLeft(v.endsAt, now);
  const stations = v.myStations || [];
  return (
    <div className={`card cv-card${stations.length > 1 ? ' many' : ''}`} data-testid="conveyor-card" style={{ ['--team' as string]: TEAM_COLORS[mine.teamId] }}>
      <div className="cv-top">
        <span className="cv-team" data-testid="conveyor-my-team">Team {mine.teamId} · captain {name(mine.captainId)}</span>
        <span className="cv-clock">{v.stage === 'overtime' ? 'Overtime ' : ''}{v.stage === 'done' ? '' : `${secs}s`}</span>
      </div>
      <div className="cv-race" aria-label={`Your team ${mine.delivered} of ${v.quota}, the other team ${theirs.delivered}`}>
        <span>Us <b data-testid="conveyor-ours">{mine.delivered}</b></span>
        <span className="cv-quota">first to {v.quota} · speed {mine.level + 1}</span>
        <span>Them <b>{theirs.delivered}</b></span>
      </div>
      {v.stage === 'done' ? (
        <p className={`cv-wait game-over ${v.winner ? (v.winner === mine.teamId ? 'win' : 'lose') : 'level'}`} role="status">{v.winner ? (v.winner === mine.teamId ? 'Your team won!' : 'The other team won.') : 'Level: the host decides.'}</p>
      ) : stations.map((k) => <Station key={k} v={v} team={mine} k={k} now={now} send={send} />)}
    </div>
  );
}

/** One of my stations: the stretch of belt around it, what the toy in it asks for, and the pieces to tap. */
function Station({ v, team, k, now, send }: { v: ConveyorView; team: ConveyorTeamView; k: number; now: number; send: ReturnType<typeof usePlayer>['send'] }) {
  const part = v.parts[k], zone = v.zones[k];
  const [said, setSaid] = useState<{ text: string; good: boolean; at: number } | null>(null);
  const here = (p: number) => p >= zone.from && p <= zone.to;
  const riders = team.toys.map((toy) => ({ toy, p: posAt(team, toy.pos, now) }));
  // the one to act on: in my station now; else the next one coming (so its order can be read early)
  const inZone = riders.find((r) => here(r.p));
  const coming = riders.filter((r) => r.p < zone.from).sort((a, b) => b.p - a.p)[0];
  const focus = inZone || coming;
  const done = !!inZone && !inZone.toy.stowaway && (inZone.toy.fitted[k] ?? NONE) !== NONE;
  const fit = async (variant: number) => {
    if (!inZone) return;
    const right = inZone.toy.want[k] === variant;
    buzz(right ? 30 : [60, 40, 60]);
    const ok = await send({ type: 'arena', action: { kind: 'fit', toy: inZone.toy.id, variant } });
    setSaid(ok ? { text: right ? 'On!' : 'Wrong piece: that one is for the bin', good: right, at: Date.now() } : { text: 'Too late: it has gone by', good: false, at: Date.now() });
    if (ok) play(right ? 'right' : 'wrong');
  };
  const shoo = async () => {
    if (!inZone) return;
    buzz(50);
    if (await send({ type: 'arena', action: { kind: 'shoo', toy: inZone.toy.id } })) { play('pass'); setSaid({ text: 'Shooed!', good: true, at: Date.now() }); }
  };
  const fresh = said && Date.now() - said.at < 1400 ? said : null;
  const raccoon = !!focus?.toy.stowaway;
  return (
    <section className={`cv-station${inZone && !done ? ' live' : ''}`} data-testid={`conveyor-station-${k}`}>
      <header className="cv-station-head">
        <span className="cv-station-name">Your station: {PART_NAMES[part]}</span>
        <span className={`cv-said${fresh?.good ? ' good' : ''}`} aria-live="polite" data-testid={`conveyor-said-${k}`}>{fresh?.text}</span>
      </header>
      <Belt v={v} team={team} now={now} from={zone.from - LEAD} to={zone.to + TAIL} tags mine={[k]} />
      <div className="cv-ask" data-testid={`conveyor-ask-${k}`}>
        {!focus ? 'Waiting for the next toy…'
          : raccoon ? (inZone ? 'A raccoon! Shoo it off the belt!' : 'A raccoon is coming…')
          : done ? 'Fitted. Next one coming…'
          : <>{inZone ? 'It wants' : 'Next one wants'} <Swatch part={part} variant={focus.toy.want[k]} /> <strong>{VARIANT_NAMES[part][focus.toy.want[k]]}</strong></>}
      </div>
      {raccoon ? (
        <button type="button" className="cv-shoo" disabled={!inZone} onPointerDown={(e) => { e.preventDefault(); void shoo(); }} data-testid={`conveyor-shoo-${k}`}>Shoo!</button>
      ) : (
        <div className="cv-choices" role="group" aria-label={`${PART_NAMES[part]} to fit`}>
          {Array.from({ length: v.variants }, (_, variant) => (
            <button key={variant} type="button" className="cv-choice" disabled={!inZone || done} data-testid={`conveyor-fit-${k}-${variant}`}
              aria-label={`${VARIANT_NAMES[part][variant]} ${PART_NAMES[part].toLowerCase()}`}
              onPointerDown={(e) => { e.preventDefault(); void fit(variant); }}>
              <Swatch part={part} variant={variant} big /><span>{VARIANT_NAMES[part][variant]}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
