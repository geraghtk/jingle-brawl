// The Great Gift Conveyor on the TV (and the host console): two factories racing. Each lane is a
// team's belt with its stations (who works each one) and the toys rolling along it, built up piece by
// piece; at the end the sleigh and its count, and the reject bin. The belt's speed is on show: it
// goes up as the team delivers. Sounds on the TV only.
import { useEffect, useRef } from 'react';
import { conveyorView } from '../core/minigames/conveyor/view';
import { useFrameNow } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { TEAM_COLORS } from '../ui/conveyor';
import { Belt } from '../ui/conveyorBelt';
import { play } from '../ui/sound';
import '../ui/arena.css';
import '../ui/ringtoss.css';   // its clock is Ring Toss's .rt-banner (and the full-screen rules for it)
import '../ui/conveyor.css';
import { useHost } from './HostContext';

export function ConveyorStage() {
  const { game, tv } = useHost();
  const v = conveyorView(game.arenaView!.view);
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const now = useFrameNow();
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
      {v.teams.map((t) => (
        <section key={t.teamId} className={`cv-lane${v.winner === t.teamId ? ' won' : ''}`} style={{ ['--team' as string]: TEAM_COLORS[t.teamId] }} data-testid={`conveyor-lane-${t.teamId}`}>
          <header className="cv-lane-head">
            <span className="cv-lane-name">Team {t.teamId}</span>
            {game.players?.[t.captainId] && <Avatar name={name(t.captainId)} avatar={game.players[t.captainId].avatar} size={30} />}
            <span className="cv-captain">captain {name(t.captainId)}</span>
            <span className="cv-speed" aria-label={`Belt speed ${t.level + 1}`} data-testid={`conveyor-speed-${t.teamId}`}>
              Speed {Array.from({ length: 8 }, (_, k) => <i key={k} className={k <= t.level ? 'on' : ''} />)}
            </span>
          </header>
          <div className="cv-floor">
            <div className="cv-chute" aria-hidden>IN</div>
            <Belt v={v} team={t} now={now} booth={(k) => (
              <span className="cv-crew">{t.crew[k].map((p) => <span key={p} className="cv-worker">{name(p)}</span>)}</span>
            )} />
            <div className="cv-end">
              <div className="cv-sleigh" data-testid={`conveyor-sleigh-${t.teamId}`}>
                <span className="cv-sleigh-count">{t.delivered}</span><span className="cv-sleigh-of">of {v.quota}</span>
              </div>
              <div className="cv-bin" aria-label={`${t.rejected} rejects`} data-testid={`conveyor-bin-${t.teamId}`}>bin {t.rejected}</div>
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}
