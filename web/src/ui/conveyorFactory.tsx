// The Great Gift Conveyor's factory floor, as the TV draws one team's line: a workshop wall strung
// with lights, a machine over each station (its sign, who works it, a lamp that says a toy is waiting,
// a tool that comes down when a piece goes on), the belt with its rollers, the toys riding out of the
// hopper and into the checker at the far end; and beside it the sleigh filling up and the reject bin.
// Like the belt itself (ui/conveyorBelt.tsx) it decides nothing: a toy is drawn where it is NOW from
// the view; the sparks, stamps and "+1"s are read off what changed between two views (so are a toy's
// little squash as a piece snaps on, the raccoon turning up and being shooed off, and the toy that
// tumbles into the bin).
import { useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import { posAt, zoneOf } from '../core/minigames/engines/belt';
import type { PartKind } from '../core/minigames/conveyor/conveyor';
import type { ConveyorTeamView, ConveyorToyView, ConveyorView } from '../core/minigames/conveyor/view';
import { Avatar } from './avatar';
import { PART_NAMES } from './conveyor';
import { ToyCar } from './conveyorBelt';

const NONE = -1;
const INK = '#1d2a3a';
export interface Worker { name: string; avatar?: string | null }

/** `k`: the station it happened at; `toy`: the toy it happened to; `ghost` and `p`: something that has
    left the belt (a shooed raccoon) and where it was. */
type Fx = { id: number; kind: 'good' | 'bad' | 'delivered' | 'rejected' | 'stolen' | 'faster' | 'raccoon' | 'shooed'; k?: number; toy?: string; ghost?: ConveyorToyView; p?: number };
const FX_MS = 1100;

/** Do something FX_MS from now, whatever else happens meanwhile (only leaving the screen calls it off).
    (An effect that cleared its own timer when the next change came left its sparks and notes up for good.) */
function useAfter(): (fn: () => void) => void {
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());
  useEffect(() => { const all = timers.current; return () => { for (const t of all) clearTimeout(t); all.clear(); }; }, []);
  return (fn) => { const t = setTimeout(() => { timers.current.delete(t); fn(); }, FX_MS); timers.current.add(t); };
}

/** What changed on a line between two views: pieces fitted (right or wrong, on which toy, at which
    station), a raccoon that has turned up, one that was shooed off (it is gone and it took nothing). */
export function lineChanges(was: { toys: ConveyorToyView[]; stolen: number }, team: Pick<ConveyorTeamView, 'toys' | 'stolen'>): Omit<Fx, 'id'>[] {
  const before = new Map(was.toys.map((t) => [t.id, t])), now = new Set(team.toys.map((t) => t.id));
  const out: Omit<Fx, 'id'>[] = [];
  for (const t of team.toys) {
    const b = before.get(t.id);
    if (!b) { if (t.stowaway) out.push({ kind: 'raccoon', toy: t.id }); continue; }
    t.fitted.forEach((f, k) => { if (f !== NONE && (b.fitted[k] ?? NONE) === NONE) out.push({ kind: f === t.want[k] ? 'good' : 'bad', k, toy: t.id }); });
  }
  if (team.stolen === was.stolen) for (const b of was.toys) if (b.stowaway && !now.has(b.id)) out.push({ kind: 'shooed', toy: b.id, ghost: b });
  return out;
}

/** What just happened on this line, from one view to the next (for about a second each). `where`: how
    far along each toy was when it was last drawn. */
function useLineFx(team: ConveyorTeamView, where: MutableRefObject<Map<string, number>>): { fx: Fx[]; stamps: number[] } {
  const [fx, setFx] = useState<Fx[]>([]);
  const [stamps, setStamps] = useState<number[]>([]);
  const seen = useRef<{ toys: ConveyorToyView[]; d: number; r: number; s: number; l: number } | null>(null);
  const n = useRef(0);
  const after = useAfter();
  const sig = `${team.delivered}.${team.rejected}.${team.stolen}.${team.level}|${team.toys.map((t) => `${t.id}:${t.fitted.join('')}`).join(',')}`;
  useEffect(() => {
    const was = seen.current;
    seen.current = { toys: team.toys, d: team.delivered, r: team.rejected, s: team.stolen, l: team.level };
    if (!was) return;
    const add: Fx[] = lineChanges({ toys: was.toys, stolen: was.s }, team).map((c) => ({ ...c, id: ++n.current, ...(c.ghost ? { p: where.current.get(c.ghost.id) } : {}) }));
    for (const id of [...where.current.keys()]) if (!team.toys.some((t) => t.id === id)) where.current.delete(id);
    if (team.delivered > was.d) add.push({ id: ++n.current, kind: 'delivered' });
    if (team.rejected > was.r) add.push({ id: ++n.current, kind: 'rejected' });
    if (team.stolen > was.s) add.push({ id: ++n.current, kind: 'stolen' });
    if (team.level > was.l) add.push({ id: ++n.current, kind: 'faster' });
    if (!add.length) return;
    setFx((cur) => [...cur, ...add]);
    setStamps((cur) => { const next = [...cur]; for (const a of add) if (a.k != null) next[a.k] = (next[a.k] || 0) + 1; return next; });
    const ids = new Set(add.map((a) => a.id));
    after(() => setFx((cur) => cur.filter((x) => !ids.has(x.id))));
  }, [sig]); // eslint-disable-line react-hooks/exhaustive-deps
  return { fx, stamps };
}

/** The tool that hangs over a station. */
function Tool({ part }: { part: PartKind }) {
  const common = { stroke: INK, strokeWidth: 3, strokeLinejoin: 'round' as const, strokeLinecap: 'round' as const };
  return (
    <svg viewBox="0 0 48 48" className="cvf-tool" aria-hidden>
      <rect x="20" y="0" width="8" height="12" fill="#9aa7b8" {...common} />
      {part === 'wheels' && <g><circle cx="24" cy="29" r="15" fill={INK} {...common} /><circle cx="24" cy="29" r="6.5" fill="#fff" {...common} /></g>}
      {part === 'lights' && <g><path d="M24,12 C12,12 8,22 13,30 C16,34 17,36 17,40 L31,40 C31,36 32,34 35,30 C40,22 36,12 24,12 Z" fill="#ffd84a" {...common} /><path d="M18,44 L30,44" {...common} /></g>}
      {part === 'paint' && <g><path d="M12,12 L36,12 L32,28 L16,28 Z" fill="#fff" {...common} /><path d="M19,28 L29,28 L27,36 L21,36 Z" fill="#9aa7b8" {...common} /><path d="M16,42 L16,45 M24,40 L24,46 M32,42 L32,45" {...common} stroke="#2f6fde" /></g>}
      {part === 'bow' && <g><path d="M24,28 L8,16 L8,40 Z M24,28 L40,16 L40,40 Z" fill="#ff7ab6" {...common} /><circle cx="24" cy="28" r="6" fill="#ff7ab6" {...common} /></g>}
    </svg>
  );
}

/** One team's line. `who(id)`: the name and picture of a worker. */
export function FactoryLine({ v, team, now, who }: { v: ConveyorView; team: ConveyorTeamView; now: number; who: (id: string) => Worker }) {
  const pct = (p: number) => `${p * 100}%`;
  // the tread scrolls with the belt: how far it has run, in belt widths
  const run = (team.speed * Math.max(0, now - team.asOf)) / 1000 * 100;
  const here = team.toys.map((t) => ({ t, p: posAt(team, t.pos, now) }));
  const where = useRef(new Map<string, number>());
  for (const { t, p } of here) where.current.set(t.id, p);
  const { fx, stamps } = useLineFx(team, where);
  /** What has just been done to a toy: a piece snapped on (`snap`), a wrong one (`jolt`). */
  const done = (id: string) => { const x = fx.find((y) => y.toy === id && (y.kind === 'good' || y.kind === 'bad')); return x ? (x.kind === 'good' ? ' snap' : ' jolt') : ''; };
  // a raccoon still at the mouth of the hopper: the hopper sounds the alarm
  const sneaking = here.some(({ t, p }) => t.stowaway && p < 0.24);
  return (
    <div className={`cvf${fx.some((x) => x.kind === 'faster') ? ' faster' : ''}`} data-testid={`conveyor-belt-${team.teamId}`}>
      <div className="cvf-wall" aria-hidden />
      <div className="cvf-garland" aria-hidden>
        {Array.from({ length: 28 }, (_, i) => (
          <svg key={i} viewBox="0 0 60 26" className="cvf-garland-bit">
            <path d="M0,3 Q30,22 60,3" fill="none" stroke={INK} strokeWidth="2" />
            {[[12, 9.5], [30, 12.5], [48, 9.5]].map(([x, y], j) => (
              <ellipse key={j} cx={x} cy={y + 5} rx="4.2" ry="5.6" fill={['#e23b3b', '#f2b705', '#1fa36b', '#38b6e6'][(i * 3 + j) % 4]} stroke={INK} strokeWidth="1.6" className={`cvf-bulb b${(i + j) % 3}`} />
            ))}
          </svg>
        ))}
      </div>
      <div className="cvf-floor" aria-hidden />

      {v.zones.map((z, k) => {
        const waiting = here.some(({ t, p }) => zoneOf(v.zones, p) === k && (t.stowaway || (t.fitted[k] ?? NONE) === NONE));
        const alarm = here.some(({ t, p }) => t.stowaway && zoneOf(v.zones, p) === k);
        const crew = team.crew[k] || [];
        const burst = fx.filter((x) => x.k === k);
        return (
          <div key={k} className={`cvf-station${waiting ? ' waiting' : ''}${alarm ? ' alarm' : ''}`} style={{ left: pct(z.from), width: pct(z.to - z.from) }} data-testid={`conveyor-zone-${team.teamId}-${k}`}>
            <span className="cvf-post l" aria-hidden /><span className="cvf-post r" aria-hidden />
            <div className="cvf-head">
              <div className={`cvf-sign${burst.some((b) => b.kind === 'good') ? ' good' : ''}`}><i className="cvf-lamp" aria-hidden /><span>{PART_NAMES[v.parts[k]]}</span></div>
              <div className={`cvf-crew${crew.length > 2 ? ' crowd' : ''}`}>
                {crew.map((id) => { const w = who(id); return <span key={id} className="cvf-worker"><Avatar name={w.name} avatar={w.avatar} size={40} /><b>{w.name}</b></span>; })}
                {!crew.length && <span className="cvf-worker nobody"><b>nobody here</b></span>}
              </div>
              {/* the machine lights up as a piece goes on (red for a wrong one) */}
              {burst.map((b) => (b.kind === 'good' || b.kind === 'bad' ? <i key={b.id} className={`cvf-flash ${b.kind}`} aria-hidden /> : null))}
            </div>
            <div className="cvf-arm" key={stamps[k] || 0} data-stamped={stamps[k] ? '' : undefined} aria-hidden><Tool part={v.parts[k]} /></div>
            {burst.map((b) => (
              <span key={b.id} className={`cvf-burst ${b.kind}`} aria-hidden>
                {b.kind === 'good' ? Array.from({ length: 8 }, (_, i) => <i key={i} style={{ ['--a' as string]: `${i * 45}deg` }} />) : <b>✕</b>}
              </span>
            ))}
          </div>
        );
      })}

      <div className="cvf-belt" aria-hidden>
        <div className="cvf-tread" style={{ backgroundPositionX: `${run}cqw` }} />
        <div className="cvf-rail" />
        <div className="cvf-rollers" style={{ backgroundPositionX: `${run}cqw` }} />
      </div>
      <div className="cvf-legs" aria-hidden />

      {here.map(({ t, p }) => (p < -0.08 || p > 1.08 ? null : (
        <div key={t.id} className={`cvf-rider${t.stowaway ? ' stowaway' : ''}${done(t.id)}`} style={{ left: pct(p) }} data-testid={`conveyor-toy-${t.id}`}>
          <ToyCar toy={t} parts={v.parts} />
          {t.stowaway && <span className="cvf-alert" aria-hidden>!</span>}
        </div>
      )))}
      {/* a raccoon shooed off the belt: it jumps clear, from where it was */}
      {fx.map((x) => (x.kind === 'shooed' && x.ghost && x.p != null ? (
        <div key={x.id} className="cvf-shooed" style={{ left: pct(Math.max(0.13, x.p)) }} aria-hidden>
          <span className="cvf-rider stowaway"><ToyCar toy={x.ghost} parts={v.parts} /></span>
          <i /><i /><i />
          <b>Shoo!</b>
        </div>
      ) : null))}

      <div className={`cvf-hopper${sneaking ? ' alarm' : ''}`} aria-hidden>
        <span className="cvf-plate">IN</span>
        <span className="cvf-dots"><i /><i /><i /></span>
        <span className="cvf-mouth" />
      </div>
      <div className={`cvf-checker${fx.some((x) => x.kind === 'delivered') ? ' ok' : ''}${fx.some((x) => x.kind === 'rejected' || x.kind === 'stolen') ? ' no' : ''}`} aria-hidden>
        <span className="cvf-plate">CHECK</span>
        <span className="cvf-eye" />
        <span className="cvf-mouth" />
      </div>

      {fx.some((x) => x.kind === 'faster') && <div className="cvf-faster" role="status">Faster!</div>}
      {fx.filter((x) => x.kind === 'stolen').map((x) => <div key={x.id} className="cvf-note" role="status">The raccoon got one!</div>)}
      {fx.filter((x) => x.kind === 'raccoon').map((x) => <div key={x.id} className="cvf-note warn" aria-hidden>Raccoon on the belt!</div>)}
    </div>
  );
}

const GIFT_COLORS = ['#e23b3b', '#1fa36b', '#2f6fde', '#f2b705', '#ff7ab6', '#8a7bff'];
/** Where the first gifts of the pile sit in the sleigh (x, y of each box's bottom centre; then they just stack higher). */
const PILE: [number, number][] = [[38, 47], [62, 47], [86, 47], [106, 47], [50, 28], [74, 28], [97, 28], [62, 9], [86, 9], [74, -10]];

/** The sleigh at the end of a line, with a gift on it for every toy delivered. */
export function Sleigh({ count, quota, color, teamId, bump }: { count: number; quota: number; color: string; teamId: string; bump: boolean }) {
  return (
    <div className={`cvf-sleigh${bump ? ' bump' : ''}${count >= quota ? ' full' : ''}`} data-testid={`conveyor-sleigh-${teamId}`}>
      <svg viewBox="0 -38 130 130" className="cvf-sleigh-art" aria-hidden>
        {PILE.slice(0, Math.min(count, PILE.length)).map(([x, y], i) => (
          <g key={i} className={bump && i === count - 1 ? 'cvf-gift new' : 'cvf-gift'}>
            <rect x={x - 12} y={y - 19} width="24" height="19" rx="3" fill={GIFT_COLORS[i % GIFT_COLORS.length]} stroke={INK} strokeWidth="2.5" />
            <path d={`M${x},${y - 19} L${x},${y} M${x - 12},${y - 10} L${x + 12},${y - 10}`} stroke="#fff" strokeWidth="3.5" />
            <path d={`M${x},${y - 19} q-6,-7 -8,-1 M${x},${y - 19} q6,-7 8,-1`} fill="none" stroke={INK} strokeWidth="2" strokeLinecap="round" />
          </g>
        ))}
        {/* the sleigh: a curled front, the tub, the runner */}
        <path d="M22,44 Q8,44 8,32 Q8,22 18,24" fill="none" stroke={INK} strokeWidth="5" strokeLinecap="round" />
        <path d="M22,42 L112,42 Q120,42 118,52 L114,70 Q112,76 104,76 L40,76 Q26,76 22,60 Z" fill={color} stroke={INK} strokeWidth="3.5" strokeLinejoin="round" />
        <path d="M30,52 L108,52" stroke="#fff" strokeWidth="4" strokeLinecap="round" opacity=".55" />
        <path d="M44,76 L44,86 M96,76 L96,86" stroke={INK} strokeWidth="4" strokeLinecap="round" />
        <path d="M24,88 L112,88 Q124,88 124,78" fill="none" stroke={INK} strokeWidth="5" strokeLinecap="round" />
      </svg>
      <div className="cvf-tally"><b key={count}>{count}</b><span>of {quota}</span></div>
      {bump && <span className="cvf-plus" aria-hidden>+1</span>}
    </div>
  );
}

/** The reject bin under it. `lost`: the toy that has just been thrown out (it tumbles in). */
export function Bin({ count, teamId, bump, lost, parts }: { count: number; teamId: string; bump: boolean; lost?: ConveyorToyView | null; parts?: readonly PartKind[] }) {
  return (
    <div className={`cvf-bin${bump ? ' bump' : ''}`} aria-label={`${count} lost`} data-testid={`conveyor-bin-${teamId}`}>
      {bump && lost && parts && <span key={lost.id} className="cvf-tumble" aria-hidden><ToyCar toy={lost} parts={parts} /></span>}
      <svg viewBox="0 0 40 40" aria-hidden>
        <path d="M6,12 L34,12 L31,36 Q30.5,38 28,38 L12,38 Q9.5,38 9,36 Z" fill="#c9d2df" stroke={INK} strokeWidth="3" strokeLinejoin="round" />
        <path d="M3,12 L37,12 M15,6 L25,6 M14,19 L15,32 M20,19 L20,32 M26,19 L25,32" stroke={INK} strokeWidth="3" strokeLinecap="round" />
      </svg>
      <span>lost <b key={count}>{count}</b></span>
    </div>
  );
}

/** The toy that went into the bin between two views: it was on the belt, now it is not (the one furthest along, if several). */
export function binned(was: ConveyorToyView[], toys: ConveyorToyView[]): ConveyorToyView | null {
  const still = new Set(toys.map((t) => t.id));
  return was.filter((t) => !t.stowaway && !still.has(t.id)).sort((a, b) => b.pos - a.pos)[0] || null;
}

/** What changed at the end of the line just now (the stage uses it for the sleigh's and the bin's bump).
    `lost`: the toy that was thrown out, as it was when it left the belt. */
export function useEndFx(team: ConveyorTeamView): { delivered: boolean; rejected: boolean; lost: ConveyorToyView | null } {
  const [on, setOn] = useState<{ delivered: boolean; rejected: boolean; lost: ConveyorToyView | null }>({ delivered: false, rejected: false, lost: null });
  const was = useRef({ d: team.delivered, r: team.rejected + team.stolen, thrown: team.rejected, toys: team.toys });
  const turn = useRef(0);
  const after = useAfter();
  const sig = `${team.delivered}.${team.rejected}.${team.stolen}|${team.toys.map((t) => `${t.id}:${t.fitted.join('')}`).join(',')}`;
  useEffect(() => {
    const before = was.current;
    const d = team.delivered > before.d, r = team.rejected + team.stolen > before.r;
    was.current = { d: team.delivered, r: team.rejected + team.stolen, thrown: team.rejected, toys: team.toys };
    if (!d && !r) return;
    setOn({ delivered: d, rejected: r, lost: team.rejected > before.thrown ? binned(before.toys, team.toys) : null });
    const mine = ++turn.current;                                        // (a later bump keeps the lamp on for its own second)
    after(() => { if (turn.current === mine) setOn({ delivered: false, rejected: false, lost: null }); });
  }, [sig]); // eslint-disable-line react-hooks/exhaustive-deps
  return on;
}
