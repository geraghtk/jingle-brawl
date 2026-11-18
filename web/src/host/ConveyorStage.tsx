// The Great Gift Conveyor on the TV (and the host console): two factories racing. Each lane is a
// team's line (ui/conveyorFactory.tsx: the workshop, a machine per station with who works it, the toys
// rolling along, built up piece by piece), its sleigh filling up and its reject bin. The belt's speed
// is on show: it goes up as the team delivers. Sounds on the TV only.
import { useEffect, useRef } from 'react';
import { conveyorView } from '../core/minigames/conveyor/view';
import { useFrameNow } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { TEAM_COLORS } from '../ui/conveyor';
import { Bin, FactoryLine, Sleigh, useEndFx } from '../ui/conveyorFactory';
import type { ConveyorTeamView, ConveyorView } from '../core/minigames/conveyor/view';
import { play } from '../ui/sound';
import '../ui/arena.css';
import '../ui/ringtoss.css';   // its clock is Ring Toss's .rt-banner (and the full-screen rules for it)
import '../ui/conveyor.css';
import '../ui/conveyorFactory.css';
import { useHost } from './HostContext';

export function ConveyorStage() {
  const { game, tv } = useHost();
  const v = conveyorView(game.arenaView!.view);
  const live = useFrameNow();
  const now = game.arenaView!.pausedAt ?? live;          // paused: the belts stand where they were
  // out loud: a toy on the sleigh, one in the bin, a raccoon away with one, a belt speeding up
  const heard = useRef(v.teams.map((t) => ({ d: t.delivered, r: t.rejected, s: t.stolen, l: t.level })));
  const key = v.teams.map((t) => `${t.delivered}.${t.rejected}.${t.stolen}.${t.level}`).join('|');
  useEffect(() => {
    const was = heard.current;
    heard.current = v.teams.map((t) => ({ d: t.delivered, r: t.rejected, s: t.stolen, l: t.level }));
    if (!tv) return;
    v.teams.forEach((t, i) => {
      const w = was[i];
      if (!w) return;
      if (t.stolen > w.s) play('wrong');
      else if (t.delivered > w.d) play('ding', 1 + Math.min(t.delivered, 8) * 0.06);
      if (t.rejected > w.r) play('thud');
      if (t.level > w.l) play('jump');
    });
  }, [key, tv]); // eslint-disable-line react-hooks/exhaustive-deps
  const secs = secsLeft(v.endsAt, now);
  return (
    <div className="cv-stage" data-testid="conveyor-stage">
      <div className={`rt-banner${v.stage !== 'done' && secs <= 10 ? ' hurry' : ''}`} data-testid="conveyor-clock">
        {v.stage === 'done' ? (v.winner ? `Team ${v.winner} wins!` : 'Level: the host decides') : `${v.stage === 'overtime' ? 'Overtime: first team ahead wins! ' : ''}${secs}s · first to ${v.quota} toys`}
      </div>
      {v.teams.map((t) => <Lane key={t.teamId} v={v} t={t} now={now} />)}
    </div>
  );
}

/** One team's factory: who is on it, how fast its belt runs, the line, the sleigh and the bin. */
function Lane({ v, t, now }: { v: ConveyorView; t: ConveyorTeamView; now: number }) {
  const { game } = useHost();
  const who = (id: string) => ({ name: game.players?.[id]?.name || 'Someone', avatar: game.players?.[id]?.avatar });
  const end = useEndFx(t);
  const lost = v.stage === 'done' && !!v.winner && v.winner !== t.teamId;
  return (
    <section className={`cv-lane${v.winner === t.teamId ? ' won' : ''}${lost ? ' lost' : ''}`} style={{ ['--team' as string]: TEAM_COLORS[t.teamId] }} data-testid={`conveyor-lane-${t.teamId}`}>
      <header className="cv-lane-head">
        <span className="cv-lane-name">Team {t.teamId}</span>
        <span className="cv-roster">
          {t.members.map((id) => <span key={id} className={`cv-mate${id === t.captainId ? ' captain' : ''}`}><Avatar name={who(id).name} avatar={who(id).avatar} size={30} />{who(id).name}{id === t.captainId && <small>captain</small>}</span>)}
        </span>
        <span className={`cv-speed${t.level >= 5 ? ' hot' : ''}`} aria-label={`Belt speed ${t.level + 1}`} data-testid={`conveyor-speed-${t.teamId}`}>
          Belt speed {Array.from({ length: 8 }, (_, k) => <i key={k} className={k <= t.level ? 'on' : ''} />)}
        </span>
      </header>
      <div className="cv-floor">
        <FactoryLine v={v} team={t} now={now} who={who} />
        <div className="cv-end">
          <Sleigh count={t.delivered} quota={v.quota} color={TEAM_COLORS[t.teamId]} teamId={t.teamId} bump={end.delivered} />
          <Bin count={t.rejected + t.stolen} teamId={t.teamId} bump={end.rejected} />
        </div>
        {v.winner === t.teamId && <div className="cv-won-flag" role="status">Winners!</div>}
      </div>
    </section>
  );
}
