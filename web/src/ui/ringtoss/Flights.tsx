// Ring Toss's rings in the air on the TV scene: moved every frame outside React (the frame loop reads
// live.current, never a closure from an old render). How a flight looks is engines/projectile
// (arcFor / arcAt) plus a spin (scene.ts tumble); where it lands is the room's.
import { useEffect, useRef } from 'react';
import { arcAt, arcFor } from '../../core/minigames/engines/projectile';
import type { RingThrow } from '../../core/minigames/ringtoss/view';
import { colorIndex, RING_COLORS } from './colors';
import { DEER_Z, HIGH, ringR, toScreen, tumble, visMs } from './scene';

/** A ring's trail: how faint each copy is, and how far apart in time. */
const TRAIL = [0.34, 0.22, 0.13, 0.06];
const TRAIL_GAP_MS = 34;

export type FlightNodes = Map<number, { ring: SVGEllipseElement | null; shadow: SVGEllipseElement | null; back?: SVGEllipseElement | null }>;

/** The copies of the rings in the air that show once they're behind the reindeer (drawn before it). */
export function FlightsBehind({ throws, nodes, participants }: { throws: RingThrow[]; nodes: { current: FlightNodes }; participants: string[] }) {
  return (
    <g className="rt-flights behind">
      {throws.map((t) => (
        <ellipse key={t.n} ref={(e) => { const n = nodes.current.get(t.n); nodes.current.set(t.n, { ring: n?.ring ?? null, shadow: n?.shadow ?? null, back: e }); }}
          className="rt-ring flying" fill="none" stroke={`url(#rt-rg-${colorIndex(participants, t.pid)})`} style={{ visibility: 'hidden' }} />
      ))}
    </g>
  );
}

export function Flights({ throws, startOf, offset, participants, ends, nodes }: {
  throws: RingThrow[]; startOf: (t: RingThrow) => number; offset: number; participants: string[];
  /** A ringer's flight ends on its peg's tip: where (x, z) and how high. */
  ends: Map<number, { x: number; z: number; h: number }>;
  /** Shared with FlightsBehind (its back copies). */
  nodes: { current: FlightNodes };
}) {
  const live = useRef({ throws, startOf, offset, ends });
  live.current = { throws, startOf, offset, ends };
  /** Each ring's trail: fainter copies of it where it was a moment ago (worked out, not remembered). */
  const trails = useRef(new Map<number, (SVGEllipseElement | null)[]>());
  const place = (t: RingThrow, at: number) => {
    const n = nodes.current.get(t.n);
    if (!n?.ring || !n.shadow) return;
    // live.current: the frame loop outlives renders, so it must read this render's starts (a stale
    // closure timed a replay's ring #1 from the last game's ring #1: it was "already down")
    const dur = visMs(t), since = at - live.current.startOf(t);
    const u = Math.min(1, Math.max(0, since / dur));
    // real gravity (engines/projectile arcAt): quick out of the hand, top late, a steep drop; a ringer drops onto its tip
    const end = live.current.ends.get(t.n) ?? { x: t.x, z: t.z, h: 0 };
    const f = arcAt(arcFor(Math.hypot(end.x, end.z), end.h), u);
    const x = end.x * f.along, z = end.z * f.along, hh = f.h;
    const c = toScreen(x, z, hh), g = toScreen(x, z, 0);
    const tb = tumble(t.n, Math.max(0, since), t.angle);
    const spin = `rotate(${tb.deg.toFixed(1)} ${c.sx.toFixed(1)} ${c.sy.toFixed(1)})`;
    // past the reindeer: behind it (the copy in the back layer shows, the front one hides)
    const behind = z > DEER_Z + 0.01;
    n.ring.style.visibility = n.shadow.style.visibility = behind ? 'hidden' : '';
    if (n.back) {
      n.back.style.visibility = behind ? '' : 'hidden';
      n.back.setAttribute('cx', String(c.sx)); n.back.setAttribute('cy', String(c.sy));
      n.back.setAttribute('rx', String(ringR(c.k))); n.back.setAttribute('ry', String(ringR(c.k) * tb.open)); n.back.setAttribute('stroke-width', String(ringR(c.k) * 0.22));
      n.back.setAttribute('transform', spin);
    }
    const r = ringR(c.k);
    n.ring.setAttribute('cx', String(c.sx)); n.ring.setAttribute('cy', String(c.sy));
    n.ring.setAttribute('rx', String(r)); n.ring.setAttribute('ry', String(r * tb.open)); n.ring.setAttribute('stroke-width', String(r * 0.22));
    n.ring.setAttribute('transform', spin);
    n.shadow.setAttribute('cx', String(g.sx)); n.shadow.setAttribute('cy', String(g.sy));
    const lift = Math.min(1, hh / HIGH);
    n.shadow.setAttribute('rx', String(ringR(g.k) * 0.9 * (1 - 0.6 * lift))); n.shadow.setAttribute('ry', String(ringR(g.k) * 0.3 * (1 - 0.6 * lift)));
    n.shadow.setAttribute('opacity', String(0.28 * (1 - 0.8 * lift)));
    (trails.current.get(t.n) || []).forEach((e, i) => {
      if (!e) return;
      const ago = since - (i + 1) * TRAIL_GAP_MS;
      const pf = arcAt(arcFor(Math.hypot(end.x, end.z), end.h), Math.max(0, ago) / dur);
      const pz = end.z * pf.along;
      if (ago <= 0 || u >= 1 || pz > DEER_Z + 0.01) { e.style.visibility = 'hidden'; return; }
      const pc = toScreen(end.x * pf.along, pz, pf.h), pt = tumble(t.n, ago, t.angle), pr = ringR(pc.k);
      e.style.visibility = '';
      e.setAttribute('cx', String(pc.sx)); e.setAttribute('cy', String(pc.sy));
      e.setAttribute('rx', String(pr)); e.setAttribute('ry', String(pr * pt.open)); e.setAttribute('stroke-width', String(pr * 0.2));
      e.setAttribute('transform', `rotate(${pt.deg.toFixed(1)} ${pc.sx.toFixed(1)} ${pc.sy.toFixed(1)})`);
    });
  };
  useEffect(() => {
    let raf = 0;
    const frame = () => {
      const { throws: ts, offset: off } = live.current;
      const at = Date.now() + off;
      for (const t of ts) place(t, at);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <g className="rt-flights">
      {throws.map((t) => (
        <g key={t.n}>
          {TRAIL.map((o, i) => (
            <ellipse key={i} ref={(e) => { const a = trails.current.get(t.n) || []; a[i] = e; trails.current.set(t.n, a); }}
              className="rt-trail" fill="none" stroke={RING_COLORS[colorIndex(participants, t.pid)]} opacity={o} style={{ visibility: 'hidden' }} />
          ))}
          <ellipse ref={(e) => { nodes.current.set(t.n, { ...nodes.current.get(t.n), shadow: e, ring: nodes.current.get(t.n)?.ring ?? null }); if (e) place(t, Date.now() + offset); }} fill="#1d2a3a" opacity={0} />
          <ellipse ref={(e) => { nodes.current.set(t.n, { ...nodes.current.get(t.n), ring: e, shadow: nodes.current.get(t.n)?.shadow ?? null }); if (e) place(t, Date.now() + offset); }}
            className="rt-ring flying" fill="none" stroke={`url(#rt-rg-${colorIndex(participants, t.pid)})`} />
        </g>
      ))}
    </g>
  );
}
