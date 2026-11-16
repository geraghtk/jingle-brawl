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

/** A little toy car, as far as it has been built: grey until painted, no wheels, lights or bow until
 *  fitted. A wrong piece is on it all the same, with a cross. `tag`: what its order asks for. */
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
      {/* the bow sits on the roof */}
      {col('bow') && <g><path d="M50,24 L36,14 L36,32 Z M50,24 L64,14 L64,32 Z" fill={col('bow')!} stroke="#1d2a3a" strokeWidth="2.5" strokeLinejoin="round" /><circle cx="50" cy="24" r="4.5" fill={col('bow')!} stroke="#1d2a3a" strokeWidth="2.5" /></g>}
      <path d="M10,58 L10,44 Q10,38 18,37 L30,35 L38,24 Q40,21 45,21 L66,21 Q71,21 74,25 L80,36 L88,39 Q93,41 93,47 L93,58 Z"
        fill={body} stroke="#1d2a3a" strokeWidth="3" strokeLinejoin="round" strokeDasharray={col('paint') ? undefined : '5 4'} />
      <path d="M42,26 L62,26 Q66,26 68,29 L72,36 L36,36 Z" fill="#e8f3ff" stroke="#1d2a3a" strokeWidth="2.5" strokeLinejoin="round" />
      {/* a headlight at the front */}
      <circle cx="87" cy="47" r="6" fill={col('lights') || 'none'} stroke="#1d2a3a" strokeWidth="2.5" strokeDasharray={col('lights') ? undefined : '3 3'} />
      {[28, 72].map((x) => (
        <g key={x}>
          <circle cx={x} cy="62" r="12" fill={col('wheels') || 'none'} stroke="#1d2a3a" strokeWidth="3" strokeDasharray={col('wheels') ? undefined : '4 4'} />
          {col('wheels') && <circle cx={x} cy="62" r="4" fill="#fff" stroke="#1d2a3a" strokeWidth="2" />}
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
