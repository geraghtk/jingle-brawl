// The Great Gift Conveyor on a phone: your team, the race, and your workstation. When a gift lands on
// your table you see its order and do your step (pick the toy, wrap it, label it, or check it and load
// it on the sleigh; shoo out any stowaway first). Then it moves on to the next phone.
import { useEffect, useRef, useState } from 'react';
import { PAPERS, STOWAWAYS, TAGS, TOYS } from '../core/minigames/conveyor/conveyor';
import type { StepKind } from '../core/minigames/conveyor/conveyor';
import type { ConveyorView } from '../core/minigames/conveyor/view';
import { useServerNow } from '../net/hooks';
import { secsLeft } from '../ui/clock';
import { buzz, play } from '../ui/sound';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/conveyor.css';

export const PAPER_COLORS: Record<string, string> = { red: '#e23b3b', green: '#1fa36b', gold: '#f2b705', blue: '#2f6fde', purple: '#8a4bd6' };
export const TEAM_COLORS: Record<string, string> = { A: '#ff3366', B: '#18b88a' };

const STEP_NAMES: Record<StepKind, string> = { pick: 'Pick', wrap: 'Wrap', label: 'Label', check: 'Check & load' };
export const stowawayName = (n: number) => (STOWAWAYS[n] === '🦝' ? 'raccoon' : 'penguin');

export function ConveyorCard() {
  const { game, send } = usePlayer();
  const v = game.arenaView!.view as ConveyorView;
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const now = useServerNow();
  const [error, setError] = useState('');
  const mine = v.teams.find((t) => t.teamId === v.myTeam);
  const theirs = v.teams.find((t) => t.teamId !== v.myTeam);
  const job = v.job;
  const was = useRef<{ item?: string; delivered?: number }>({});
  useEffect(() => {
    if (job && job.item !== was.current.item) { play('jump'); buzz(60); }
    if (mine && was.current.delivered !== undefined && mine.delivered > was.current.delivered) play('right');
    was.current = { item: job?.item, delivered: mine?.delivered };
  }, [job?.item, job?.step, mine?.delivered]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!mine || !theirs) return <div className="card" data-testid="conveyor-card"><p>Watch the TV: the teams are racing!</p></div>;

  const act = async (action: Record<string, unknown>) => {
    setError('');
    const ok = await send({ type: 'arena', action: { item: job!.item, ...action } });
    if (!ok) setError('Too late: that gift moved on');
  };
  const secs = secsLeft(v.endsAt, now);
  const station = v.myStation != null && v.myStation >= 0 ? mine.stations[v.myStation] : null;

  return (
    <div className="card cv-card" data-testid="conveyor-card" style={{ ['--team' as string]: TEAM_COLORS[mine.teamId] }}>
      <div className="cv-top">
        <span className="cv-team" data-testid="conveyor-my-team">Team {mine.teamId} · captain {name(mine.captainId)}</span>
        <span className="cv-clock">{v.stage === 'overtime' ? 'Overtime ' : ''}{secs}s</span>
      </div>
      <div className="cv-race" aria-label={`Your team ${mine.delivered} of ${v.quota}, the other team ${theirs.delivered}`}>
        <span>Us <b data-testid="conveyor-ours">{mine.delivered}</b></span>
        <span className="cv-quota">first to {v.quota}</span>
        <span>Them <b>{theirs.delivered}</b></span>
      </div>
      {station && <div className="cv-station">Your station: {station.steps.map((s) => STEP_NAMES[s]).join(' + ')}</div>}
      {v.stage === 'done' ? (
        <p className="cv-wait" role="status">{v.winner ? (v.winner === mine.teamId ? 'Your team won!' : 'The other team won.') : 'Level: the host decides.'}</p>
      ) : job ? <Job job={job} act={act} /> : (
        <p className="cv-wait" data-testid="conveyor-waiting">Waiting for the next gift…</p>
      )}
      <div className="arena-error" role="alert">{error}</div>
    </div>
  );
}

function OrderSlip({ order }: { order: ConveyorView['job'] extends infer J ? J extends { order: infer O } ? O : never : never }) {
  return (
    <div className="cv-order" data-testid="conveyor-order" aria-label={`Order: ${TOYS[order.toy]} in ${PAPERS[order.paper]} paper for ${TAGS[order.tag]}`}>
      <span className="cv-k">Order</span>
      <span className="cv-toy">{TOYS[order.toy]}</span>
      <span className="cv-paper" style={{ background: PAPER_COLORS[PAPERS[order.paper]] }}>{PAPERS[order.paper]}</span>
      <span className="cv-tag">for {TAGS[order.tag]}</span>
    </div>
  );
}

function Job({ job, act }: { job: NonNullable<ConveyorView['job']>; act: (a: Record<string, unknown>) => void }) {
  const g = job.gift;
  return (
    <div className="cv-job" data-testid="conveyor-job" data-step={job.step}>
      <OrderSlip order={job.order} />
      {job.step === 'pick' && (<><p className="cv-ask">Pick the toy</p>
        <div className="cv-choices">{job.options.toy.map((t) => <button key={t} className="cv-choice big" data-choice={t} aria-label={TOYS[t]} onClick={() => act({ kind: 'pick', choice: t })}>{TOYS[t]}</button>)}</div></>)}
      {job.step === 'wrap' && (<><p className="cv-ask">Wrap it in {PAPERS[job.order.paper]}</p>
        <div className="cv-choices">{job.options.paper.map((p) => <button key={p} className="cv-choice" data-choice={p} style={{ background: PAPER_COLORS[PAPERS[p]], color: '#fff' }} onClick={() => act({ kind: 'wrap', choice: p })}>{PAPERS[p]}</button>)}</div></>)}
      {job.step === 'label' && (<><p className="cv-ask">Label it</p>
        <div className="cv-choices">{job.options.tag.map((t) => <button key={t} className="cv-choice tag" data-choice={t} onClick={() => act({ kind: 'label', choice: t })}>{TAGS[t]}</button>)}</div></>)}
      {job.step === 'check' && (
        <>
          <p className="cv-ask">Does it match the order?</p>
          <div className="cv-gift" data-testid="conveyor-gift" aria-label={`The gift: ${g.toy != null ? TOYS[g.toy] : 'no toy'} in ${g.paper != null ? PAPERS[g.paper] : 'no'} paper for ${g.tag != null ? TAGS[g.tag] : 'nobody'}`}>
            <span className="cv-toy">{g.toy != null ? TOYS[g.toy] : '?'}</span>
            <span className="cv-paper" style={{ background: g.paper != null ? PAPER_COLORS[PAPERS[g.paper]] : '#ccc' }}>{g.paper != null ? PAPERS[g.paper] : '?'}</span>
            <span className="cv-tag">for {g.tag != null ? TAGS[g.tag] : '?'}</span>
          </div>
          {g.stowaway != null ? (   /* != : a view drops nulls, so "none" arrives as undefined */
            <button className="cv-shoo" data-testid="conveyor-shoo" onClick={() => act({ kind: 'shoo' })}>
              The box is wobbling: a {stowawayName(g.stowaway)} got in! <span className="cv-stow-icon">{STOWAWAYS[g.stowaway]}</span> Shoo it out
            </button>
          ) : (
            <div className="cv-check">
              <button className="cv-load" data-testid="conveyor-load" onClick={() => act({ kind: 'load' })}>🛷 Load it on the sleigh</button>
              <button className="cv-back" data-testid="conveyor-back" onClick={() => act({ kind: 'back' })}>↩ Wrong: send it back</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
