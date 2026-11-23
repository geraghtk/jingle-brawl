// Elf Tower Wars on a phone (eyes on the TV, thumbs on the phone). Everyone has the same hand of cards
// and their own energy: pick a card, then tap where it goes. A toy goes down a path; a tower on one of
// the team's own spots; a snowball or a shield on a path; glue straight on the workshop. The picked
// card stays picked, so the next tap sends another. The map is the TV's battlefield the same way
// round (your workshop on your team's side). The phone only says what was pressed; the room decides
// what happened.
import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { armed, unitAt } from '../core/minigames/engines/lanes';
import { energyOf, towerView } from '../core/minigames/towers/view';
import { useServerNow } from '../net/hooks';
import { buzz, play } from '../ui/sound';
import { CARD_HINTS, CARD_NAMES, CARD_TIPS, TOWER_COLORS, TOWER_NAMES, TowerArt, UnitArt } from '../ui/towers/art';
import { usePlayer } from './PlayerContext';
import '../ui/arena.css';
import '../ui/towers.css';

const LANE_NAMES = ['Top', 'Middle', 'Bottom', 'Lane 4', 'Lane 5'];
const INK = '#1d2a3a';
const HELP_ART: Record<string, ReactNode> = {
  zap: <svg viewBox="0 0 40 40" aria-hidden><path d="M4,30 L15,23 M3,22 L12,19" stroke={INK} strokeWidth="3" strokeLinecap="round" opacity=".45" /><circle cx="24" cy="18" r="12" fill="#fff" stroke={INK} strokeWidth="3" /><path d="M18,13 Q22,9 27,10" fill="none" stroke="#b9d6f2" strokeWidth="3" strokeLinecap="round" /></svg>,
  shield: <svg viewBox="0 0 40 40" aria-hidden><path d="M20,4 L34,9 V19 Q34,30 20,36 Q6,30 6,19 V9 Z" fill="#7cc4ff" stroke={INK} strokeWidth="3" strokeLinejoin="round" /><path d="M20,9 V31 M11,18 H29" stroke="#fff" strokeWidth="3.5" strokeLinecap="round" /></svg>,
  combo: <svg viewBox="0 0 40 40" aria-hidden><path d="M3,34 L16,10 L22,20 L27,13 L37,34 Z" fill="#fff" stroke={INK} strokeWidth="3" strokeLinejoin="round" /><path d="M13,16 L16,10 L19,15 M25,16 L27,13 L29,17" fill="none" stroke="#b9d6f2" strokeWidth="2.5" strokeLinecap="round" /><circle cx="11" cy="31" r="4" fill="#dbe9f7" stroke={INK} strokeWidth="2.5" /><circle cx="22" cy="32" r="5" fill="#dbe9f7" stroke={INK} strokeWidth="2.5" /></svg>,
  patch: <svg viewBox="0 0 40 40" aria-hidden><path d="M17,3 H23 L24,11 H16 Z" fill="#ffc21a" stroke={INK} strokeWidth="3" strokeLinejoin="round" /><rect x="10" y="11" width="20" height="25" rx="5" fill="#fff" stroke={INK} strokeWidth="3" /><path d="M20,17 V29 M14,23 H26" stroke="#e0143c" strokeWidth="4" strokeLinecap="round" /></svg>,
};

interface Card { id: string; cost: number; where: 'lane' | 'spot' | 'now'; pic: ReactNode }

export function TowerCard() {
  const { game, me, send } = usePlayer();
  const av = game.arenaView!;
  const v = towerView(av.view);
  const ticking = useServerNow(150);
  const now = av.pausedAt ?? ticking;                    // paused: the battle stands as it was
  const ti = v.teams.findIndex((t) => t.teamId === v.myTeam);
  const team = ti >= 0 ? v.teams[ti] : null;
  const [picked, setPicked] = useState('soldier');
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
  const color = TOWER_COLORS[ti];
  const press = async (action: Record<string, unknown>) => {
    buzz(15);
    const ok = await send({ type: 'arena', action }, { quiet: true });
    if (!ok) setSaid('Not yet');
  };
  const have = energyOf(team, me, now, p, v);
  const finale = running && now >= v.rushAt && v.rush > 1;
  const lanes = Array.from({ length: p.lanes }, (_, l) => l);
  const theirs = f.sides[ti === 0 ? 1 : 0];
  const coming = (lane: number) => f.units.filter((u) => u.side !== ti && u.lane === lane);
  const inReach = (lane: number) => coming(lane).some((u) => 1 - unitAt(u, f.asOf, now) <= p.zapReach);
  const out = (lane: number) => f.units.filter((u) => u.side === ti && u.lane === lane).length;
  const won = v.stage === 'done' ? (v.winner === team.teamId ? 'Your team won!' : v.winner ? 'The other team won.' : 'Dead level: the host decides.') : '';

  const cards: Card[] = [
    ...Object.entries(p.units).map(([id, k]): Card => ({ id, cost: k.cost, where: 'lane', pic: <UnitArt kind={id} color={color} style={{ width: '100%', height: '100%' }} /> })),
    { id: 'zap', cost: p.zapCost, where: 'lane', pic: HELP_ART.zap },
    { id: 'shield', cost: p.shieldCost, where: 'lane', pic: HELP_ART.shield },
    { id: 'combo', cost: p.comboCost, where: 'lane', pic: HELP_ART.combo },
    ...Object.entries(p.towers).map(([id, k]): Card => ({ id, cost: k.cost, where: 'spot', pic: <TowerArt kind={id} color={color} style={{ width: '100%', height: '100%' }} /> })),
    { id: 'patch', cost: p.patchCost, where: 'now', pic: HELP_ART.patch },
  ];
  const card = cards.find((c) => c.id === picked) || cards[0];
  const isUnit = !!p.units[card.id], isTower = !!p.towers[card.id];
  const rich = running && have + 1e-9 >= card.cost;
  const building = isTower && v.stage !== 'overtime';

  /** May the picked card go down this path now? */
  const laneOpen = (lane: number) => rich && (isUnit ? now >= mySide.sendAt[lane] : card.id === 'zap' ? now >= mySide.zapAt && inReach(lane) : card.id === 'shield' ? now >= mySide.shieldAt : card.id === 'combo' ? now >= mySide.comboAt && !(armed(mySide, now, p) && mySide.armBy === me && mySide.armLane === lane) : false);
  const towerAt = (lane: number, slot: number) => mySide.towers.find((x) => x.lane === lane && x.slot === slot);
  /** May the picked tower go on this spot: an empty one, the same tower again (it gets better), or over a cheaper one? */
  const spotOpen = (lane: number, slot: number) => { const t = towerAt(lane, slot); return rich && building && (!t || (t.kind === card.id ? (t.level || 1) < p.maxLevel : p.towers[t.kind].cost < card.cost)); };
  const tapLane = (lane: number) => {
    if (isUnit) void press({ kind: 'send', lane, unit: card.id });
    else if (card.id === 'zap' || card.id === 'shield' || card.id === 'combo') void press({ kind: card.id, lane });
    else setSaid('Towers go on your spots');
  };
  const tapSpot = (lane: number, slot: number) => {
    if (isTower) void press({ kind: 'build', lane, slot, tower: card.id });
    else setSaid('Pick a tower card first');
  };
  const tapCard = (c: Card) => {
    if (c.where === 'now') { void press({ kind: 'patch' }); return; }
    buzz(8);
    setPicked(c.id);
  };
  const glueReady = running && have >= p.patchCost && mySide.hp < p.baseHp && now >= mySide.patchAt;

  const verb = isUnit ? 'Send' : card.id === 'zap' ? 'Throw' : card.id === 'shield' ? 'Shield' : card.id === 'combo' ? 'Go!' : '';
  const short = Math.ceil(card.cost - have);
  const hint = !running ? ''
    : isTower && v.stage === 'overtime' ? 'Overtime: no more building. Send toys!'
    : !rich ? `${CARD_NAMES[card.id]} costs ${card.cost}: ${short} more energy`
    : card.id === 'shield' && now < mySide.shieldAt ? `The shield is back in ${Math.ceil((mySide.shieldAt - now) / 1000)}s`
    : card.id === 'combo' && now < mySide.comboAt ? `The avalanche is back in ${Math.ceil((mySide.comboAt - now) / 1000)}s`
    : isTower && mySide.towers.some((t) => t.kind === card.id && (t.level || 1) < p.maxLevel) ? `${CARD_TIPS[card.id]} Tap yours again to make it better.`
    : CARD_TIPS[card.id];

  return (
    <div className={`tw-controller${running ? '' : ' waiting'}${ti === 1 ? ' flip' : ''}`} style={{ ['--team' as string]: color }} data-testid="tower-card">
      <div className="tw-top">
        <span className="tw-me" data-testid="tower-my-team">Team {team.teamId}</span>
        <span className="tw-score" data-testid="tower-score">Us {Math.ceil(mySide.hp)} : {Math.ceil(theirs.hp)} Them</span>
      </div>
      <div className={`tw-phone-word${won ? ` game-over ${v.winner === team.teamId ? 'win' : v.winner ? 'lose' : 'level'}` : ''}`} aria-live="polite" data-testid="tower-word">{won || said || (v.stage === 'overtime' ? 'Overtime!' : finale ? 'Double energy!' : '')}</div>

      <div className="tw-map" data-testid="tower-map">
        <span className="tw-map-home" aria-hidden>Your workshop</span>
        {lanes.map((lane) => {
          const n = coming(lane).length, open = laneOpen(lane);
          const shielded = mySide.shieldLane === lane && now < mySide.shieldUntil;
          return (
            <div key={lane} className="tw-lane-row">
              {p.slots.map((_q, slot) => {
                const t = towerAt(lane, slot), ok = spotOpen(lane, slot);
                return (
                  <button key={slot} type="button" className={`tw-spot-btn${t ? ' built' : ''}${ok ? ' open' : ''}`} disabled={!running}
                    aria-label={`${LANE_NAMES[lane]} path, spot ${slot + 1}${t ? `: ${TOWER_NAMES[t.kind]}` : ''}`} data-testid={`tower-spot-btn-${lane}-${slot}`}
                    onPointerDown={(e) => { e.preventDefault(); tapSpot(lane, slot); }}>
                    {t ? <TowerArt kind={t.kind} color={color} flip={ti === 1} style={{ width: '78%', height: '78%' }} /> : <i aria-hidden />}
                    {t && (t.level || 1) > 1 && <span className="tw-level" aria-label={`level ${t.level}`}>{'★'.repeat((t.level || 1) - 1)}</span>}
                  </button>
                );
              })}
              <button type="button" className={`tw-path${open ? ' open' : ''}${n ? ' alarm' : ''}${shielded ? ' shielded' : ''}`} disabled={!running}
                aria-label={`${LANE_NAMES[lane]} path`} data-testid={`tower-lane-${lane}`} onPointerDown={(e) => { e.preventDefault(); tapLane(lane); }}>
                <b>{LANE_NAMES[lane]}</b>
                <small className="tw-path-info">{shielded ? 'Shield up' : n ? `${n} coming!` : out(lane) ? `${out(lane)} of yours out` : 'quiet'}</small>
                {open && verb && <span className="tw-path-go">{verb}</span>}
              </button>
            </div>
          );
        })}
      </div>

      <p className="tw-hint" data-testid="tower-hint">{hint}</p>
      <div className={`tw-purse${finale ? ' rush' : ''}`} role="img" aria-label={`${Math.floor(have)} energy`} data-testid="tower-energy">
        <span className="tw-purse-bar"><i style={{ width: `${(have / p.purse.max) * 100}%` }} /></span><b>{Math.floor(have)}</b>
      </div>
      <div className="tw-cards">
        {cards.map((c) => {
          const can = running && have + 1e-9 >= c.cost;
          const off = c.where === 'now' ? !glueReady : !running || (!!p.towers[c.id] && v.stage === 'overtime');
          return (
            <button key={c.id} type="button" className={`tw-card${c.id === card.id && c.where !== 'now' ? ' picked' : ''}${can ? '' : ' poor'}${c.where === 'now' && glueReady ? ' go' : ''}`} disabled={off}
              aria-pressed={c.id === card.id} aria-label={`${CARD_NAMES[c.id]}: ${CARD_HINTS[c.id]}, ${c.cost} energy`} data-testid={`tower-card-${c.id}`}
              onPointerDown={(e) => { e.preventDefault(); tapCard(c); }}>
              <span className="tw-card-pic">{c.pic}</span>
              <b>{CARD_NAMES[c.id]}</b>
              <small>{CARD_HINTS[c.id]}</small>
              <span className="tw-cost">{c.cost}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
