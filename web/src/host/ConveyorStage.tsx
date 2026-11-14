// The Great Gift Conveyor on the TV (and the host console): two factories racing. Each lane is the
// team's stations in a row (who works each one), the gifts on the tables and in the trays, and the
// sleigh at the end with the count. A gift that moves on slides to its next station.
import { useEffect, useRef } from 'react';
import { STOWAWAYS } from '../core/minigames/conveyor/conveyor';
import type { ConveyorView } from '../core/minigames/conveyor/view';
import { useServerNow } from '../net/hooks';
import { Avatar } from '../ui/avatar';
import { secsLeft } from '../ui/clock';
import { STEP_NAMES, TEAM_COLORS } from '../ui/conveyor';
import { play } from '../ui/sound';
import '../ui/ringtoss.css';   // its clock is Ring Toss's .rt-banner (and the full-screen rules for it)
import '../ui/conveyor.css';
import { useHost } from './HostContext';

export function ConveyorStage() {
  const { game, tv } = useHost();
  const v = game.arenaView!.view as ConveyorView;
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const now = useServerNow();
  const total = v.teams.reduce((n, t) => n + t.delivered, 0);
  const seen = useRef(total);
  useEffect(() => { if (tv && total > seen.current) play('right'); seen.current = total; }, [total, tv]);
  const secs = secsLeft(v.endsAt, now);
  return (
    <div className="cv-stage" data-testid="conveyor-stage">
      <div className="rt-banner">
        {v.stage === 'done' ? (v.winner ? `Team ${v.winner} wins!` : 'Level: the host decides') : `${v.stage === 'overtime' ? 'Overtime! ' : ''}${secs}s · first to ${v.quota} gifts`}
      </div>
      {v.teams.map((t) => (
        <section key={t.teamId} className={`cv-lane${v.winner === t.teamId ? ' won' : ''}`} style={{ ['--team' as string]: TEAM_COLORS[t.teamId] }} data-testid={`conveyor-lane-${t.teamId}`}>
          <header className="cv-lane-head">
            <span className="cv-lane-name">Team {t.teamId}</span>
            {game.players?.[t.captainId] && <Avatar name={name(t.captainId)} avatar={game.players[t.captainId].avatar} size={30} />}
            <span className="cv-captain">captain {name(t.captainId)}</span>
            <span className="cv-members">{t.members.filter((m) => m !== t.captainId).map(name).join(', ')}</span>
          </header>
          <div className="cv-belt">
            {t.stations.map((s, i) => {
              const here = t.gifts.filter((g) => g.station === i);
              return (
                <div key={i} className="cv-st">
                  <div className="cv-st-name">{s.steps.map((k) => STEP_NAMES[k]).join(' + ')}</div>
                  <div className="cv-st-who">{s.workers.map(name).join(' / ')}</div>
                  <div className="cv-table" aria-label={`${here.filter((g) => g.working).length} being worked, ${here.filter((g) => !g.working).length} waiting`}>
                    {here.filter((g) => g.working).map((g) => <span key={g.id} className="cv-box working">🎁{g.stowaway != null && <i className="cv-stow">{STOWAWAYS[g.stowaway]}</i>}</span>)}
                    {here.filter((g) => !g.working).map((g) => <span key={g.id} className="cv-box waiting">🎁</span>)}
                  </div>
                </div>
              );
            })}
            <div className="cv-sleigh" data-testid={`conveyor-delivered-${t.teamId}`}>
              <span className="cv-sleigh-icon" aria-hidden>🛷</span>
              <b>{t.delivered}</b><span>/ {v.quota}</span>
              {t.wrong > 0 && <small className="cv-wrong">{t.wrong} wrong</small>}
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}
