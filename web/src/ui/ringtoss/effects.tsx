// Ring Toss's effects on the TV scene (SVG, animated by ui/ringtoss.css): the snow, a ring lying where
// it landed, a ringer's burst of stars, a miss's puff and shards, a full set's confetti. They play
// whatever the OS motion setting is: they are the game's feedback.
import { useRef } from 'react';
import { RING_COLORS } from './colors';
import { ringR, W } from './scene';

/** A small seeded random stream (mulberry32): the same snow, confetti and shards on every render. Kept
 *  here, not core/rng.ts: that one's arithmetic differs a little, and the pictures would change. */
function seeded(seed: number): () => number {
  let s = seed | 0;
  return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Snow drifting down the scene, always: small slow flakes far off (behind the reindeer), a few big
 *  quick ones close to. Each falls on its own time and sways its own way (CSS). */
export function Snow({ n, seed, far = false }: { n: number; seed: number; far?: boolean }) {
  const flakes = useRef<{ x: number; r: number; dur: number; delay: number; sway: number }[]>();
  if (!flakes.current) {
    const rnd = seeded(seed);
    flakes.current = Array.from({ length: n }, () => ({
      x: rnd() * W, r: far ? 2 + rnd() * 2.5 : 4.5 + rnd() * 4, dur: (far ? 11 : 6.5) + rnd() * 6, delay: -rnd() * 17, sway: (rnd() - 0.5) * (far ? 90 : 170),
    }));
  }
  return (
    <g className={`rt-snowfall${far ? ' far' : ''}`} aria-hidden>
      {flakes.current.map((f, i) => (
        <circle key={i} className="rt-flake" cx={f.x.toFixed(1)} cy={-12} r={f.r.toFixed(1)} fill="#fff"
          style={{ ['--dur' as string]: `${f.dur.toFixed(2)}s`, ['--delay' as string]: `${f.delay.toFixed(2)}s`, ['--sway' as string]: `${f.sway.toFixed(0)}px` }} />
      ))}
    </g>
  );
}

/** A full set: confetti in every ring colour falls the height of the scene. */
export function Confetti({ seed }: { seed: number }) {
  const rnd = seeded(seed * 977 + 5);
  return (
    <g className="rt-confetti" aria-hidden>
      {Array.from({ length: 46 }, (_, i) => {
        const w = 9 + rnd() * 9;
        return (
          <g key={i} className="rt-conf-fall" transform={`translate(${(rnd() * W).toFixed(0)} -30)`}
            style={{ ['--dur' as string]: `${(1.5 + rnd() * 0.9).toFixed(2)}s`, ['--delay' as string]: `${(rnd() * 0.35).toFixed(2)}s`, ['--sway' as string]: `${((rnd() - 0.5) * 220).toFixed(0)}px` }}>
            <rect className="rt-conf" x={-w / 2} y={-w / 4} width={w} height={w / 2} rx={1.5} fill={RING_COLORS[i % RING_COLORS.length]}
              style={{ ['--rot' as string]: `${Math.round((rnd() - 0.5) * 1400)}deg` }} />
          </g>
        );
      })}
    </g>
  );
}

/** A colour lighter (amt > 0) or darker (amt < 0): the ring's shading. */
export function shade(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const ch = (v: number) => Math.round(amt > 0 ? v + (255 - v) * amt : v * (1 + amt));
  return `#${[n >> 16, (n >> 8) & 255, n & 255].map((v) => ch(v).toString(16).padStart(2, '0')).join('')}`;
}

/** A ring with some body: shaded top to bottom, a darker inner edge, a shine along the top; on the
 *  ground, a soft shadow under it. (A ringer is drawn whole behind the antler and again, cut along its
 *  middle, over it: so it hangs round the prong with no seams.) */
export function Ring({ at, color, practice = false, ground = false, ringer = false, hidePuff = false }: {
  at: { sx: number; sy: number; k: number }; color: number; practice?: boolean; ground?: boolean; ringer?: boolean;
  /** Its smoke is drawn elsewhere (a ring behind the reindeer: the smoke rises in front, where it shows). */
  hidePuff?: boolean;
}) {
  const r = ringR(at.k), ry = r * 0.42, sw = r * 0.24;
  const { sx, sy } = at;
  const top = (rx: number, ryy: number) => `M${sx - rx},${sy} A${rx},${ryy} 0 0 1 ${sx + rx},${sy}`;
  return (
    <g opacity={practice ? 0.55 : 1}>
      <g className={`rt-ring${ground ? ' miss' : ringer ? ' ringer' : ''}`}>
        {ground && <ellipse cx={sx} cy={sy + sw * 0.55} rx={r * 1.04} ry={ry * 1.1} fill="none" stroke="#1d2a3a" strokeOpacity={0.22} strokeWidth={sw * 1.5} />}
        <ellipse cx={sx} cy={sy} rx={r} ry={ry} fill="none" stroke={`url(#rt-rg-${color})`} strokeWidth={sw} />
        <path d={top(r - sw * 0.38, ry - sw * 0.3)} fill="none" stroke="#000" strokeOpacity={0.22} strokeWidth={sw * 0.22} />
        <path d={`M${sx - r * 0.55},${sy - ry * 0.92} A${r},${ry} 0 0 1 ${sx + r * 0.25},${sy - ry * 1.02}`} fill="none" stroke="#fff" strokeOpacity={0.6} strokeWidth={sw * 0.25} strokeLinecap="round" />
      </g>
      {ground && !hidePuff && <Puff sx={sx} sy={sy} r={r} color={RING_COLORS[color]} />}
    </g>
  );
}

/** A ringer's moment, right as it lands: a flash, then stars burst outward in its thrower's colour, each
 *  spinning as it flies, with gold twinkles; its points rise. (Positions are offsets the CSS animates to.) */
export function Burst({ at, color, points, tag = '' }: { at: { sx: number; sy: number; k: number }; color: string; points: number; tag?: string }) {
  const r = Math.max(ringR(at.k), 26) * 1.6, n = 12;   // a good burst even far away
  const star = (R: number) => {
    const pts = Array.from({ length: 10 }, (_, i) => {
      const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? R * 0.48 : R;
      return `${(Math.cos(a) * rr).toFixed(1)},${(Math.sin(a) * rr).toFixed(1)}`;
    });
    return `M${pts.join('L')}Z`;
  };
  const twinkle = (R: number) => `M0,${-R} Q0,0 ${R},0 Q0,0 0,${R} Q0,0 ${-R},0 Q0,0 0,${-R}Z`;
  return (
    <g className="rt-burst" aria-hidden transform={`translate(${at.sx.toFixed(1)} ${at.sy.toFixed(1)})`}>
      <circle className="rt-flash" r={r * 1.1} fill="url(#rt-flash)" />
      {Array.from({ length: n }, (_, i) => {
        const a = (i / n) * Math.PI * 2 + (i % 2) * 0.2, d = r * (2.2 + (i % 3) * 0.5);
        return (
          <g key={i} className="rt-fly" style={{ ['--dx' as string]: `${(Math.cos(a) * d).toFixed(1)}px`, ['--dy' as string]: `${(Math.sin(a) * d * 0.8 - r * 0.4).toFixed(1)}px`, animationDelay: `${(i % 3) * 25}ms` }}>
            <path className="rt-spin" d={star(r * (0.3 + (i % 3) * 0.08))} fill={color} stroke="#fff" strokeWidth={2} strokeLinejoin="round"
              style={{ animationDirection: i % 2 ? 'reverse' : 'normal' }} />
          </g>
        );
      })}
      {Array.from({ length: 6 }, (_, i) => {
        const a = (i / 6) * Math.PI * 2 + 0.5, d = r * (1.3 + (i % 2) * 0.9);
        return <path key={`t${i}`} className="rt-twinkle" d={twinkle(r * 0.22)} fill="#ffe27a"
          transform={`translate(${(Math.cos(a) * d).toFixed(1)} ${(Math.sin(a) * d * 0.8 - r * 0.3).toFixed(1)})`} style={{ animationDelay: `${80 + i * 45}ms` }} />;
      })}
      {points > 0 && <text y={-r * 0.7} textAnchor="middle" className="rt-plus" fill={color}>+{points}</text>}
      {tag && <text y={-r * 0.7 - 34} textAnchor="middle" className="rt-plus rt-who" fill={color} data-testid="ringtoss-who">{tag}</text>}
    </g>
  );
}

/** A missed ring, right as it lands: it crumples into bits in its thrower's colour that scatter and fall,
 *  under a soft puff of snow dust (shaded grey-blue, no outlines: it lands on snow). */
export function Puff({ sx, sy, r: ring, color }: { sx: number; sy: number; r: number; color: string }) {
  const r = Math.max(ring, 34) * 1.6;   // a good cloud even for a ring far away
  const blobs = [[-0.55, 0.05, 0.62], [0.5, 0.0, 0.66], [0, -0.2, 0.78], [-0.2, 0.25, 0.5], [0.3, 0.28, 0.52]];
  return (
    <g className="rt-puff" aria-hidden transform={`translate(${sx.toFixed(1)} ${sy.toFixed(1)})`}>
      {blobs.map(([x, y, s], i) => (
        <circle key={i} className="rt-smoke" cx={x * r} cy={y * r * 0.5} r={s * r} fill="url(#rt-smoke)"
          style={{ ['--rise' as string]: `${(-r * (0.5 + i * 0.12)).toFixed(1)}px`, ['--drift' as string]: `${(x * r * 0.6).toFixed(1)}px`, animationDelay: `${i * 30}ms` }} />
      ))}
      {bits(sx, sy, ring, color)}
    </g>
  );
}

/** A missed ring shatters: its band snaps into 4 to 6 chunks (together the whole ring), each a curved slab
 *  with jagged broken ends, that burst outward from where they were on the ring, up and then falling
 *  (gravity: the CSS eases each one up, then down), spinning; and a few small splinters. Seeded by where it
 *  landed, so every ring breaks differently but the same way on each render. */
function bits(sx: number, sy: number, ring: number, color: string) {
  const rnd = seeded((Math.round(sx * 7) * 73856093) ^ (Math.round(sy * 7) * 19349663));
  const r = Math.max(ring, 22), w = Math.max(5, r * 0.24), ro = r + w / 2, R = Math.max(ring, 32) * 1.4;   // R: how far they fly
  const pt = (rad: number, a: number, cx: number) => `${(rad * Math.cos(a) - cx).toFixed(1)},${(rad * Math.sin(a)).toFixed(1)}`;
  /** A chunk of the band from angle -span/2 to span/2, centred on itself, its two ends broken jaggedly. */
  const slab = (span: number) => {
    const a0 = -span / 2, a1 = span / 2, cx = ro - w / 2, jag = (w / ro) * 0.9, pts: string[] = [];
    for (let k = 0; k <= 6; k++) pts.push(pt(ro, a0 + (span * k) / 6, cx));                       // the outer edge
    for (const f of [0.3, 0.55, 0.8]) pts.push(pt(ro - w * f, a1 + (rnd() - 0.5) * 2 * jag, cx));   // a broken end
    for (let k = 6; k >= 0; k--) pts.push(pt(ro - w, a0 + (span * k) / 6, cx));                   // the inner edge
    for (const f of [0.8, 0.55, 0.3]) pts.push(pt(ro - w * f, a0 + (rnd() - 0.5) * 2 * jag, cx));   // the other broken end
    return pts.join(' ');
  };
  /** A splinter: a thin jagged sliver. */
  const splinter = (len: number) => {
    const t = len * (0.25 + rnd() * 0.2);
    return [[-len / 2, 0], [-len * 0.1, -t], [len * 0.15, -t * 0.4], [len / 2, 0], [len * 0.05, t * 0.7]].map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  };
  const fly = (a: number, hard: number) => {
    const v = R * hard;
    return {
      ['--dx' as string]: `${(Math.cos(a) * v * 1.2).toFixed(1)}px`,
      ['--up' as string]: `${Math.min(-R * 0.3, (Math.sin(a) * 0.5 - 0.6) * v).toFixed(1)}px`,   // outward, and up
      ['--down' as string]: `${(r * (0.3 + rnd() * 0.8)).toFixed(1)}px`,
      ['--rot' as string]: `${Math.round((rnd() - 0.5) * 900)}deg`, ['--dur' as string]: `${Math.round(700 + rnd() * 500)}ms`,
      ['--delay' as string]: `${Math.round(260 + rnd() * 60)}ms`,
    };
  };
  // the chunks: random shares of the whole circle, each where it was on the ring (squashed like the ring)
  const k = 4 + Math.floor(rnd() * 3), cuts = Array.from({ length: k }, () => 0.6 + rnd()), total = cuts.reduce((a, b) => a + b, 0);
  let at = rnd() * Math.PI * 2;
  const chunks = cuts.map((c, i) => {
    const span = (c / total) * Math.PI * 2 - 0.12, mid = at + span / 2 + 0.06;
    at += (c / total) * Math.PI * 2;
    const ox = Math.cos(mid) * (ro - w / 2), oy = Math.sin(mid) * (ro - w / 2) * 0.42;
    return (
      <g key={`c${i}`} transform={`translate(${ox.toFixed(1)} ${oy.toFixed(1)})`}>
        <g className="rt-bit-x" style={fly(Math.atan2(oy, ox), 0.5 + rnd() * 0.9)}>
          <g className="rt-bit-y">
            <polygon className="rt-bit" points={slab(span)} transform={`rotate(${((mid * 180) / Math.PI).toFixed(0)}) scale(1 ${(0.5 + Math.abs(Math.cos(mid)) * 0.5).toFixed(2)})`}
              fill={shade(color, (rnd() - 0.5) * 0.4)} stroke={shade(color, -0.45)} strokeWidth={1.2} strokeLinejoin="miter" />
          </g>
        </g>
      </g>
    );
  });
  const splinters = Array.from({ length: 4 + Math.floor(rnd() * 4) }, (_, i) => {
    const a = rnd() * Math.PI * 2, ox = Math.cos(a) * ro * 0.9, oy = Math.sin(a) * ro * 0.38;
    return (
      <g key={`s${i}`} transform={`translate(${ox.toFixed(1)} ${oy.toFixed(1)})`}>
        <g className="rt-bit-x" style={fly(a, 0.8 + rnd() * 1.1)}>
          <g className="rt-bit-y">
            <polygon className="rt-bit" points={splinter(w * (0.8 + rnd() * 1.2))} transform={`rotate(${Math.round(rnd() * 360)})`}
              fill={shade(color, (rnd() - 0.5) * 0.6)} stroke={shade(color, -0.45)} strokeWidth={0.8} />
          </g>
        </g>
      </g>
    );
  });
  return [...chunks, ...splinters];
}
