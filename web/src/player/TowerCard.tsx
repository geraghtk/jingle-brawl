// Elf Tower Wars on a phone. It is played in rounds of two halves (Kevin, 2026-11-23: with everything live
// at once "I'm always looking at my phone screen"):
//   PLAN    the phone is busy: six cards (three toys, three towers) and a map. Pick a card, then tap where it
//           goes: a toy lines up at the door of a path, a tower goes on one of the team's spots (the same
//           tower again makes it better). "Done" when you are; when everyone is, the march starts at once.
//   MARCH   the phone is quiet: "Watch the TV!", and three reactions for the thumb (a snowball, a shield,
//           the avalanche that takes two of you): pick one, tap a path.
// The map is the TV's battlefield the same way round (your workshop on your team's side). The phone only
// says what was pressed; the room decides what happened.
import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { armed, unitAt } from '../core/minigames/engines/lanes';
import { energyOf, fieldClock, towerView } from '../core/minigames/towers/view';
import { useServerNow } from '../net/hooks';
import { secsLeft } from '../ui/clock';
import { buzz, play } from '../ui/sound';
import { CARD_HINTS, CARD_NAMES, CARD_TIPS, TOWER_COLORS, TOWER_NAMES, TowerArt, UnitArt } from '../ui/towers/art';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/motion.css';
import '../ui/towers.css';

const LANE_NAMES = ['Top', 'Middle', 'Bottom', 'Lane 4', 'Lane 5'];
const INK = '#1d2a3a';
const HELP_ART: Record<string, ReactNode> = {
  zap: <svg viewBox="0 0 40 40" aria-hidden><path d="M4,30 L15,23 M3,22 L12,19" stroke={INK} strokeWidth="3" strokeLinecap="round" opacity=".45" /><circle cx="24" cy="18" r="12" fill="#fff" stroke={INK} strokeWidth="3" /><path d="M18,13 Q22,9 27,10" fill="none" stroke="#b9d6f2" strokeWidth="3" strokeLinecap="round" /></svg>,
  shield: <svg viewBox="0 0 40 40" aria-hidden><path d="M20,4 L34,9 V19 Q34,30 20,36 Q6,30 6,19 V9 Z" fill="#7cc4ff" stroke={INK} strokeWidth="3" strokeLinejoin="round" /><path d="M20,9 V31 M11,18 H29" stroke="#fff" strokeWidth="3.5" strokeLinecap="round" /></svg>,
  combo: <svg viewBox="0 0 40 40" aria-hidden><path d="M3,34 L16,10 L22,20 L27,13 L37,34 Z" fill="#fff" stroke={INK} strokeWidth="3" strokeLinejoin="round" /><path d="M13,16 L16,10 L19,15 M25,16 L27,13 L29,17" fill="none" stroke="#b9d6f2" strokeWidth="2.5" strokeLinecap="round" /></svg>,
};

interface Card { id: string; cost: number; where: 'lane' | 'spot'; pic: ReactNode }

export function TowerCard() {
  const { game, me, send } = usePlayer();
  const av = game.arenaView!;
  const v = towerView(av.view);
  const ticking = useServerNow(150);
  const now = av.pausedAt ?? ticking;                    // paused: the battle stands as it was
  const ti = v.teams.findIndex((t) => t.teamId === v.myTeam);
  const team = ti >= 0 ? v.teams[ti] : null;
  const [planCard, setPlanCard] = useState('soldier');
  const [reaction, setReaction] = useState('zap');
  const [said, setSaid] = useState('');
  const f = v.field, p = v.rules;
  const mySide = ti >= 0 ? f.sides[ti] : null;

  // my workshop was hit: the phone thumps
  const hp = useRef(mySide?.hp ?? 0);
  useEffect(() => {
    const h = mySide?.hp ?? 0;
    if (h < hp.current) { buzz([60, 40, 60]); play('thud'); }
    hp.current = h;
  }, [mySide?.hp]); // eslint-disable-line react-hooks/exhaustive-deps
  // the round turns over: a buzz, so a thumb knows without the eyes leaving the TV
  const half = `${v.round}|${v.phase}`;
  const lastHalf = useRef(half);
  useEffect(() => { if (lastHalf.current !== half) { lastHalf.current = half; buzz(v.phase === 'plan' ? [30, 40, 30] : 60); } }, [half, v.phase]);
  useEffect(() => { if (!said) return; const t = setTimeout(() => setSaid(''), 1400); return () => clearTimeout(t); }, [said]);

  if (!team || !mySide || !me) {
    return (
      <div className="card tw-watch" data-testid="tower-card">
        <p className="tw-watch-title">Elf Tower Wars</p>
        <p>Watch the TV: two workshops, three lanes, and a lot of toys.</p>
      </div>
    );
  }

  const running = v.stage !== 'done';
  const planning = running && v.phase === 'plan';
  const marching = running && v.phase === 'march';
  const color = TOWER_COLORS[ti];
  const press = async (action: Record<string, unknown>) => {
    buzz(15);
    const ok = await send({ type: 'arena', action }, { quiet: true });
    if (!ok) setSaid('Not now');
  };
  const have = energyOf(team, me);
  const fnow = fieldClock(v, now);
  const secs = secsLeft(v.phaseEndsAt, now);
  const lanes = Array.from({ length: p.lanes }, (_, l) => l);
  const theirs = f.sides[ti === 0 ? 1 : 0];
  const coming = (lane: number) => f.units.filter((u) => u.side !== ti && u.lane === lane);
  const inReach = (lane: number) => coming(lane).some((u) => 1 - unitAt(u, f.asOf, fnow) <= p.zapReach);
  const lined = (lane: number, side = ti) => v.queue.filter((q) => q.side === side && q.lane === lane).length;
  const iAmDone = team.done.includes(me);
  const waitingFor = v.teams.flatMap((t) => t.active.filter((id) => !t.done.includes(id))).length;
  const won = v.stage === 'done' ? (v.winner === team.teamId ? 'Your team won!' : v.winner ? 'The other team won.' : 'Dead level: the host decides.') : '';

  const cards: Card[] = [
    ...Object.entries(p.units).map(([id, k]): Card => ({ id, cost: k.cost, where: 'lane', pic: <UnitArt kind={id} color={color} style={{ width: '100%', height: '100%' }} /> })),
    ...Object.entries(p.towers).map(([id, k]): Card => ({ id, cost: k.cost, where: 'spot', pic: <TowerArt kind={id} color={color} style={{ width: '100%', height: '100%' }} /> })),
  ];
  const reactions: Card[] = [
    { id: 'zap', cost: p.zapCost, where: 'lane', pic: HELP_ART.zap },
    { id: 'shield', cost: p.shieldCost, where: 'lane', pic: HELP_ART.shield },
    { id: 'combo', cost: p.comboCost, where: 'lane', pic: HELP_ART.combo },
  ];
  const card = planning ? cards.find((c) => c.id === planCard) || cards[0] : reactions.find((c) => c.id === reaction) || reactions[0];
  const isUnit = !!p.units[card.id], isTower = !!p.towers[card.id];
  const rich = have + 1e-9 >= card.cost;

  /** May the picked card go down this path now? */
  const laneOpen = (lane: number) => rich && (
    planning ? isUnit && !iAmDone
      : marching && (card.id === 'zap' ? now >= mySide.zapAt && inReach(lane) : card.id === 'shield' ? now >= mySide.shieldAt
        : now >= mySide.comboAt && !(armed(mySide, now, p) && mySide.armBy === me && mySide.armLane === lane)));
  const towerAt = (lane: number, slot: number) => mySide.towers.find((x) => x.lane === lane && x.slot === slot);
  /** May the picked tower go on this spot: an empty one, the same tower again (it gets better), or over a cheaper one? */
  const spotOpen = (lane: number, slot: number) => { const t = towerAt(lane, slot); return planning && !iAmDone && isTower && rich && (!t || (t.kind === card.id ? (t.level || 1) < p.maxLevel : p.towers[t.kind].cost < card.cost)); };
  const tapLane = (lane: number) => {
    if (!running) return;
    if (planning) { if (isUnit) void press({ kind: 'send', lane, unit: card.id }); else setSaid('Towers go on your spots'); }
    else void press({ kind: card.id, lane });
  };
  const tapSpot = (lane: number, slot: number) => {
    if (!planning) { setSaid('Towers are built while you plan'); return; }
    if (isTower) void press({ kind: 'build', lane, slot, tower: card.id });
    else setSaid('Pick a tower card first');
  };

  const verb = planning ? (isUnit ? 'Line up' : '') : card.id === 'zap' ? 'Throw' : card.id === 'shield' ? 'Shield' : 'Go!';
  const short = Math.ceil(card.cost - have);
  const hint = !running ? ''
    : planning && iAmDone ? (waitingFor ? `Done. Waiting for ${waitingFor} more…` : 'Here they go!')
    : !rich ? `${CARD_NAMES[card.id]} costs ${card.cost}: ${short} more energy`
    : marching && card.id === 'shield' && now < mySide.shieldAt ? `The shield is back in ${Math.ceil((mySide.shieldAt - now) / 1000)}s`
    : marching && card.id === 'combo' && now < mySide.comboAt ? `The avalanche is back in ${Math.ceil((mySide.comboAt - now) / 1000)}s`
    : isTower && mySide.towers.some((t) => t.kind === card.id && (t.level || 1) < p.maxLevel) ? `${CARD_TIPS[card.id]} Tap yours again to make it better.`
    : CARD_TIPS[card.id];

  const cardButton = (c: Card, pick: (id: string) => void, off: boolean) => (
    <button key={c.id} type="button" className={`tw-card${c.id === card.id ? ' picked' : ''}${have + 1e-9 >= c.cost ? '' : ' poor'}`} disabled={off}
      aria-pressed={c.id === card.id} aria-label={`${CARD_NAMES[c.id]}: ${CARD_HINTS[c.id]}, ${c.cost} energy`} data-testid={`tower-card-${c.id}`}
      onPointerDown={(e) => { e.preventDefault(); buzz(8); pick(c.id); }}>
      <span className="tw-card-pic">{c.pic}</span>
      <b>{CARD_NAMES[c.id]}</b>
      <small>{CARD_HINTS[c.id]}</small>
      <span className="tw-cost">{c.cost}</span>
    </button>
  );

  return (
    <div className={`tw-controller${running ? '' : ' waiting'}${ti === 1 ? ' flip' : ''} ${planning ? 'plan' : marching ? 'march' : ''}`} style={{ ['--team' as string]: color }} data-testid="tower-card">
      <div className="tw-top">
        <span className="tw-me" data-testid="tower-my-team">Team {team.teamId}</span>
        {running && <span className="tw-round" data-testid="tower-phase">Round {v.round} of {v.rounds} · {planning ? `Plan ${secs}` : 'March'}</span>}
        <span className="tw-score" aria-label={`Your workshop ${Math.ceil(mySide.hp)}, theirs ${Math.ceil(theirs.hp)}`} data-testid="tower-score"><b style={{ color: color }}>{Math.ceil(mySide.hp)}</b> : <b style={{ color: TOWER_COLORS[ti === 0 ? 1 : 0] }}>{Math.ceil(theirs.hp)}</b></span>
      </div>
      <div className={`tw-phone-word${won ? ` game-over ${v.winner === team.teamId ? 'win' : v.winner ? 'lose' : 'level'}` : ''}${marching && !said ? ' watch' : ''}`} aria-live="polite" data-testid="tower-word">
        {won || said || (marching ? 'Watch the TV!' : v.stage === 'overtime' ? 'Extra round: plan your march' : 'Plan your march')}
      </div>

      <div className="tw-map" data-testid="tower-map">
        <span className="tw-map-home" aria-hidden>Your workshop</span>
        {lanes.map((lane) => {
          const n = coming(lane).length, open = laneOpen(lane), mine = lined(lane), their = lined(lane, ti === 0 ? 1 : 0);
          const shielded = mySide.shieldLane === lane && now < mySide.shieldUntil;
          return (
            <div key={lane} className="tw-lane-row">
              {p.slots.map((_q, slot) => {
                const t = towerAt(lane, slot), ok = spotOpen(lane, slot);
                return (
                  <button key={slot} type="button" className={`tw-spot-btn${t ? ' built' : ''}${ok ? ' open' : ''}`} disabled={!running}
                    aria-label={`${LANE_NAMES[lane]} path, spot ${slot + 1}${t ? `: ${TOWER_NAMES[t.kind]}` : ''}`} data-testid={`tower-spot-btn-${lane}-${slot}`}
                    onPointerDown={(e) => { e.preventDefault(); tapSpot(lane, slot); }}>
                    {t ? <TowerArt kind={t.kind} level={t.level || 1} color={color} flip={ti === 1} style={{ width: '78%', height: '78%' }} /> : <i aria-hidden />}
                    {t && (t.level || 1) > 1 && <span className="tw-level" aria-label={`level ${t.level}`}>{'★'.repeat((t.level || 1) - 1)}</span>}
                  </button>
                );
              })}
              <button type="button" className={`tw-path${open ? ' open' : ''}${n ? ' alarm' : ''}${shielded ? ' shielded' : ''}`} disabled={!running}
                aria-label={`${LANE_NAMES[lane]} path`} data-testid={`tower-lane-${lane}`} onPointerDown={(e) => { e.preventDefault(); tapLane(lane); }}>
                <b>{LANE_NAMES[lane]}</b>
                <small className="tw-path-info" data-testid={`tower-lane-info-${lane}`}>
                  {shielded ? 'Shield up' : planning ? [mine ? `${mine} lined up` : '', their ? `they have ${their}` : '', n ? `${n} of theirs here` : ''].filter(Boolean).join(' · ') || 'empty' : n ? `${n} coming!` : 'quiet'}
                </small>
                {open && verb && <span className="tw-path-go">{verb}</span>}
              </button>
            </div>
          );
        })}
      </div>

      <p className="tw-hint" data-testid="tower-hint">{hint}</p>
      <div className="tw-purse" role="img" aria-label={`${Math.floor(have)} energy`} data-testid="tower-energy">
        <span className="tw-purse-bar"><i style={{ width: `${Math.min(100, (have / p.purse.max) * 100)}%` }} /></span><b>{Math.floor(have)}</b>
      </div>
      {marching ? (
        <div className="tw-cards react" data-testid="tower-reactions">{reactions.map((c) => cardButton(c, setReaction, false))}</div>
      ) : (
        <>
          <div className="tw-cards" data-testid="tower-hand">{cards.map((c) => cardButton(c, setPlanCard, !planning || iAmDone))}</div>
          <button type="button" className={`tw-done${iAmDone ? ' is' : ''}`} disabled={!planning || iAmDone} data-testid="tower-done"
            onPointerDown={(e) => { e.preventDefault(); if (planning && !iAmDone) void press({ kind: 'done' }); }}>
            {iAmDone ? 'Done ✓' : 'Done: start the march'}
          </button>
        </>
      )}
    </div>
  );
}
