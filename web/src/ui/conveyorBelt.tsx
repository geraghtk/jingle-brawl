// The Great Gift Conveyor's belt, as both the TV and the phones draw it: a strip with its stations,
// and the toys riding it, each drawn as far as it has been built. A toy is placed where it is NOW
// (engines/belt posAt: where it was, how fast the belt runs), so it glides with no message per frame.
// The TV shows the whole belt; a phone shows a close-up of one station (`from`..`to`).
import type { ReactNode } from 'react';
import { posAt } from '../core/minigames/engines/belt';
import type { PartKind } from '../core/minigames/conveyor/conveyor';
import type { ConveyorTeamView, ConveyorToyView, ConveyorView } from '../core/minigames/conveyor/view';
import { PART_NAMES, PRIMER, VARIANT_COLORS, VARIANT_MARKS, VARIANT_NAMES } from './conveyor';

const NONE = -1;
const INK = '#1d2a3a';
/** An empty wheel arch or lamp socket. */
const WELL = '#46536a';
const BODY = 'M10,58 L10,44 Q10,38 18,37 L30,35 L38,24 Q40,21 45,21 L66,21 Q71,21 74,25 L80,36 L88,39 Q93,41 93,47 L93,58 Z';

/** A little toy car, as far as it has been built: bare riveted tin until painted, up on stands until its
 *  wheels are on, an empty lamp socket, no bow until fitted. A wrong piece is on it all the same, with a cross. `tag`: what its order asks for. */
export function ToyCar({ toy, parts, tag = false }: { toy: ConveyorToyView; parts: readonly PartKind[]; tag?: boolean }) {
  const at = (p: PartKind) => parts.indexOf(p);
  const fit = (p: PartKind) => (at(p) >= 0 ? toy.fitted[at(p)] ?? NONE : NONE);
  const col = (p: PartKind) => (fit(p) === NONE ? null : VARIANT_COLORS[p][fit(p)]);
  const wrong = parts.some((_p, k) => (toy.fitted[k] ?? NONE) !== NONE && toy.fitted[k] !== toy.want[k]);
  if (toy.stowaway) {
    return (
      <svg viewBox="0 0 100 80" className="cv-toy stowaway" role="img" aria-label="A raccoon">
        <ellipse cx="50" cy="52" rx="30" ry="22" fill="#8b8f99" stroke="#1d2a3a" strokeWidth="3" />
        <circle cx="50" cy="30" r="20" fill="#a7abb5" stroke="#1d2a3a" strokeWidth="3" />
        <path d="M34,14 L30,2 L42,10 Z M66,14 L70,2 L58,10 Z" fill="#8b8f99" stroke="#1d2a3a" strokeWidth="3" strokeLinejoin="round" />
        <path d="M30,28 Q50,18 70,28 Q62,40 50,34 Q38,40 30,28 Z" fill="#1d2a3a" />
        <circle cx="42" cy="29" r="3.5" fill="#fff" /><circle cx="58" cy="29" r="3.5" fill="#fff" />
        <ellipse cx="50" cy="38" rx="3.5" ry="2.5" fill="#1d2a3a" />
        <path d="M78,58 Q96,50 90,34" fill="none" stroke="#1d2a3a" strokeWidth="9" strokeLinecap="round" />
        <path d="M78,58 Q96,50 90,34" fill="none" stroke="#a7abb5" strokeWidth="5" strokeLinecap="round" strokeDasharray="6 6" />
      </svg>
    );
  }
  const painted = col('paint') != null;
  const body = col('paint') || PRIMER;
  return (
    <svg viewBox="0 0 100 80" className={`cv-toy${wrong ? ' wrong' : ''}`} role="img"
      aria-label={`A toy: ${parts.map((p, k) => `${PART_NAMES[p]} ${(toy.fitted[k] ?? NONE) === NONE ? 'missing' : VARIANT_NAMES[p][toy.fitted[k]]}`).join(', ')}`}>
      {tag && (
        <g className="cv-tag">
          <rect x="8" y="0" width="84" height="17" rx="5" fill="#fff" stroke="#1d2a3a" strokeWidth="2" />
          {parts.map((p, k) => <circle key={p} cx={22 + k * 19} cy="8.5" r="5.5" fill={VARIANT_COLORS[p][toy.want[k]]} stroke="#1d2a3a" strokeWidth="1.5" />)}
        </g>
      )}
      <ellipse cx="51" cy="74" rx="42" ry="4" fill="rgba(20, 30, 50, .22)" />
      {/* the bow sits on the roof, its tails down the back */}
      {col('bow') && (
        <g stroke={INK} strokeWidth="2.5" strokeLinejoin="round" fill={col('bow')!}>
          <path d="M47,22 L40,31 L45,31 Z M53,22 L60,31 L55,31 Z" />
          <path d="M50,21 Q40,8 34,13 Q31,20 50,21 Z M50,21 Q60,8 66,13 Q69,20 50,21 Z" />
          <circle cx="50" cy="20.5" r="4.2" />
          <path d="M39,14 Q42,13 45,16 M61,14 Q58,13 55,16" fill="none" stroke="rgba(255,255,255,.7)" strokeWidth="1.6" strokeLinecap="round" />
        </g>
      )}
      {/* bumpers, behind the body */}
      <rect x="5" y="50" width="10" height="7" rx="3.5" fill="#eef2f7" stroke={INK} strokeWidth="2.5" />
      <rect x="88" y="50" width="9" height="7" rx="3.5" fill="#eef2f7" stroke={INK} strokeWidth="2.5" />
      {/* the body: bare tin until it is painted */}
      <path d={BODY} fill={body} stroke={INK} strokeWidth="3" strokeLinejoin="round" />
      <path d="M11.5,51 L91.5,51 L91.5,56.5 L11.5,56.5 Z" fill="rgba(20, 30, 50, .16)" />
      <path d="M16,41.5 Q22,39.5 31,38.5" fill="none" stroke="rgba(255,255,255,.6)" strokeWidth="2.4" strokeLinecap="round" />
      {!painted && (
        <g fill="#8d97a6">
          {[[17, 46], [44, 43], [61, 43], [84, 44], [52, 53]].map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="1.5" />)}
        </g>
      )}
      {/* the windows, a door */}
      <path d="M42,26 L62,26 Q66,26 68,29 L72,36 L36,36 Z" fill="#dff0ff" stroke={INK} strokeWidth="2.5" strokeLinejoin="round" />
      <path d="M53,26 L53,36" stroke={INK} strokeWidth="2.5" />
      <path d="M42.5,33.5 L46,28.5 M58,33.5 L61.5,28.5" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
      <path d="M53,36 L53,50" stroke="rgba(20, 30, 50, .3)" strokeWidth="1.8" strokeLinecap="round" />
      <rect x="56" y="40" width="7" height="2.6" rx="1.3" fill="rgba(20, 30, 50, .45)" />
      {/* the headlight: an empty socket until it is fitted */}
      {col('lights')
        ? (
          <g>
            <circle cx="88" cy="46" r="10" fill={col('lights')!} opacity=".3" />
            <circle cx="87" cy="46" r="6" fill={col('lights')!} stroke={INK} strokeWidth="2.5" />
            <circle cx="85.2" cy="44.2" r="1.9" fill="#fff" />
          </g>
        )
        : <circle cx="87" cy="46" r="5" fill={WELL} stroke={INK} strokeWidth="2.5" />}
      {/* the wheels: empty arches and a bare axle until they are fitted */}
      {[28, 72].map((x) => (
        <g key={x}>
          <path d={`M${x - 13.5},58.5 A13.5,13.5 0 0 1 ${x + 13.5},58.5 Z`} fill={WELL} stroke={INK} strokeWidth="2.5" strokeLinejoin="round" />
          {col('wheels')
            ? (
              <g>
                <circle cx={x} cy="62" r="11.5" fill={col('wheels')!} stroke={INK} strokeWidth="3" />
                <circle cx={x} cy="62" r="7.4" fill="none" stroke="rgba(20, 30, 50, .28)" strokeWidth="2.2" />
                <circle cx={x} cy="62" r="3.8" fill="#fff" stroke={INK} strokeWidth="2" />
                <path d={`M${x - 8},${56} Q${x - 4},${52.5} ${x + 1},${52.6}`} fill="none" stroke="rgba(255,255,255,.65)" strokeWidth="1.8" strokeLinecap="round" />
              </g>
            )
            : (
              <g>
                <rect x={x - 2} y="57" width="4" height="13" rx="1.5" fill="#8d97a6" stroke={INK} strokeWidth="2" />
                <rect x={x - 7} y="68" width="14" height="5" rx="2" fill="#8d97a6" stroke={INK} strokeWidth="2" />
              </g>
            )}
        </g>
      ))}
      {wrong && <path d="M70,4 L92,26 M92,4 L70,26" stroke="#e0143c" strokeWidth="6" strokeLinecap="round" />}
    </svg>
  );
}

/** A colour and its mark, for a button or a sign (the mark is the clue that isn't colour). */
export function Swatch({ part, variant, big = false }: { part: PartKind; variant: number; big?: boolean }) {
  return (
    <span className={`cv-swatch${big ? ' big' : ''}`} style={{ background: VARIANT_COLORS[part][variant] }} aria-hidden>
      <span className="cv-mark">{VARIANT_MARKS[variant]}</span>
    </span>
  );
}

/** One team's belt between `from` and `to` (the whole belt by default), with its stations and its toys
 *  where they are at `now`. `booth(k)`: what stands at station k (who works it, on the TV). */
export function Belt({ v, team, now, from = 0, to = 1, tags = false, booth, mine }: {
  v: ConveyorView; team: ConveyorTeamView; now: number; from?: number; to?: number;
  /** Show each toy's order above it (the phone: you need to read it; the TV would be a wall of dots). */
  tags?: boolean;
  booth?: (k: number) => ReactNode;
  /** Stations to light up as "yours". */
  mine?: number[];
}) {
  const span = to - from;
  const pct = (p: number) => `${((p - from) / span) * 100}%`;
  const w = (d: number) => `${(d / span) * 100}%`;
  // the tread scrolls with the belt: how far it has run, in tread marks
  const run = ((team.speed * Math.max(0, now - team.asOf)) / 1000 / span) * 100;
  return (
    <div className="cv-belt" data-testid={`conveyor-belt-${team.teamId}`}>
      <div className="cv-tread" style={{ backgroundPositionX: `${run}cqw` }} aria-hidden />
      {v.zones.map((z, k) => (z.to < from || z.from > to ? null : (
        <div key={k} className={`cv-zone${mine?.includes(k) ? ' mine' : ''}`} style={{ left: pct(z.from), width: w(z.to - z.from) }} data-testid={`conveyor-zone-${team.teamId}-${k}`}>
          <span className="cv-zone-name">{PART_NAMES[v.parts[k]]}</span>
          {booth?.(k)}
        </div>
      )))}
      {team.toys.map((toy) => {
        const p = posAt(team, toy.pos, now);
        if (p < from - 0.08 || p > to + 0.08) return null;
        return (
          <div key={toy.id} className="cv-rider" style={{ left: pct(p) }} data-testid={`conveyor-toy-${toy.id}`}>
            <ToyCar toy={toy} parts={v.parts} tag={tags} />
          </div>
        );
      })}
    </div>
  );
}
