// What stands round Nutcracker Knockout's ring: an arena at dusk, kept quiet so the robots are the picture. Tiers of
// toy spectators (dim shapes, still until a blow lands), a Christmas tree each side, a string of warm lights, a little snow.
// (Kevin, 2026-11-23: the ring used to stand in front of a plain sky.) It decides nothing and reads nothing
// but the clock and `cheer` (0 to 1: how excited the crowd is, set by the scene when a punch lands).
import * as THREE from 'three';
import { at, ball, cone, cyl, GOLD, INK } from '../toon3d';

// (Kevin, the same day: the first version was "a bit too busy", a lot of vibrant colours and movement. So: the crowd is
// dim shapes in the colours of the dusk, with no outlines, that keep still until a blow lands; the lights are one warm
// colour; the trees are plain; nothing but the robots is bright.)
const COATS = [0x4b5a9c, 0x5a4f9a, 0x44689e, 0x6a5a9e, 0x3f4f8a];
const HEAD = 0xb9a6c4, WARM = 0xffdf8a;
const flat = (geo: THREE.BufferGeometry, color: number) => new THREE.Mesh(geo, new THREE.MeshToonMaterial({ color }));

/** The first row's distance from the middle of the ring (well back: the crowd is scenery, the robots are the picture), the rows' spacing, the first seat's height. */
export const ROW0 = 6.1, ROW_GAP = 0.95, SEAT_Y = -0.3;
/** Where the spectators sit: rows of an arc behind the ring (never between the camera and the fight). */
export function crowdSeats(rows = 3, perRow = 15): { x: number; y: number; z: number; row: number }[] {
  const seats: { x: number; y: number; z: number; row: number }[] = [];
  for (let row = 0; row < rows; row++) {
    const r = ROW0 + row * ROW_GAP, n = perRow + row * 2;
    for (let i = 0; i < n; i++) {
      const a = Math.PI * (1.13 + (0.74 * (i + (row % 2) * 0.5)) / (n - 0.5));       // the far side of the ring only
      seats.push({ x: Math.cos(a) * r, y: SEAT_Y + row * 0.5, z: Math.sin(a) * r, row });
    }
  }
  return seats;
}

export interface Arena { tick(t: number, dt: number, cheer: number): void }

export function buildArena(scene: THREE.Scene, rnd: () => number = Math.random): Arena {
  const far = new THREE.Group();                         // (nothing out here casts a shadow: it is scenery)
  scene.add(far);
  // the ground the ring stands on, and the steps the crowd sits on
  far.add(at(cyl(16, 16, 0.1, 0xcfdcf0, 40), 0, -0.47, 0));
  scene.fog = new THREE.Fog(0x4a4a98, 7.5, 19);        // the far side of the arena sinks into the dusk
  for (let row = 0; row < 3; row++) {
    const r = ROW0 + row * ROW_GAP;
    const step = new THREE.Mesh(new THREE.CylinderGeometry(r + 0.45, r + 0.45, 0.55, 40, 1, true, Math.PI * 0.56, Math.PI * 0.88), new THREE.MeshToonMaterial({ color: row % 2 ? 0x2b3a63 : 0x34467a, side: THREE.DoubleSide }));
    const top = new THREE.Mesh(new THREE.RingGeometry(r - 0.42, r + 0.45, 40, 1, Math.PI * 1.06, Math.PI * 0.88), new THREE.MeshToonMaterial({ color: 0x3d5190, side: THREE.DoubleSide }));
    top.rotation.x = -Math.PI / 2;
    far.add(at(step, 0, SEAT_Y - 0.27 + row * 0.5, 0), at(top, 0, SEAT_Y + row * 0.5, 0));
  }
  // the crowd: a toy each, a body and a head, some in a hat
  const crowd = crowdSeats().map((s) => {
    const g = new THREE.Group(), coat = COATS[Math.floor(rnd() * COATS.length)];
    g.add(at(flat(new THREE.CylinderGeometry(0.13, 0.2, 0.42, 10), coat), 0, 0.21, 0), at(flat(new THREE.SphereGeometry(0.15, 12, 10), HEAD), 0, 0.55, 0));
    g.scale.setScalar(0.9);
    if (rnd() < 0.5) g.add(at(flat(new THREE.ConeGeometry(0.13, 0.26, 10), COATS[Math.floor(rnd() * COATS.length)]), 0, 0.78, 0));
    g.position.set(s.x, s.y, s.z);
    g.lookAt(0, s.y, 0);
    far.add(g);
    return { g, y: s.y, phase: rnd() * Math.PI * 2, keen: 0.6 + rnd() * 0.8 };
  });
  // a tree and a pile of presents each side of the ring
  for (const side of [-1, 1]) {
    const tree = new THREE.Group();
    tree.add(at(cyl(0.16, 0.2, 0.5, 0x7a4a2a, 10), 0, 0.25, 0));
    [[1.15, 1.3, 0.95], [0.9, 1.15, 1.75], [0.62, 1.0, 2.5]].forEach(([r, h, y]) => tree.add(at(cone(r, h, 0x1c7a5c, 16), 0, y, 0)));
    tree.add(at(ball(0.16, GOLD), 0, 3.12, 0));
    for (let k = 0; k < 6; k++) {
      const a = k * 2.4 + 0.6, y = 0.8 + (k / 6) * 1.8, r = 1.0 - (k / 6) * 0.6;
      tree.add(at(flat(new THREE.SphereGeometry(0.07, 10, 8), WARM), Math.cos(a) * r, y, Math.sin(a) * r));
    }
    far.add(at(tree, side * 4.25, -0.42, -3.3));
  }
  // strings of lights overhead, post to post
  const bulbs: { m: THREE.Mesh; phase: number }[] = [];
  const posts: [number, number, number][] = [[-8.6, 4.3, -6.5], [-3, 3.9, -8.2], [3, 3.9, -8.2], [8.6, 4.3, -6.5]];   // (the two in the middle hang from above: no pole across the picture)
  posts.filter(([x]) => Math.abs(x) > 5).forEach(([x, y, z]) => far.add(at(cyl(0.07, 0.09, y + 0.5, INK, 8), x, (y + 0.5) / 2 - 0.5, z), at(ball(0.16, GOLD), x, y + 0.05, z)));
  for (let s = 0; s < posts.length - 1; s++) {
    const a = new THREE.Vector3(...posts[s]), b = new THREE.Vector3(...posts[s + 1]);
    const pts: THREE.Vector3[] = [];
    for (let k = 0; k <= 12; k++) {
      const u = k / 12, p = a.clone().lerp(b, u);
      p.y -= Math.sin(Math.PI * u) * 0.75;                                              // it sags
      pts.push(p);
      if (k > 0 && k < 12) {
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.075, 10, 8), new THREE.MeshBasicMaterial({ color: WARM, fog: false }));
        m.position.copy(p).y -= 0.1;
        far.add(m);
        bulbs.push({ m, phase: rnd() * 6.28 });
      }
    }
    far.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: INK })));
  }
  far.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.castShadow = false; m.receiveShadow = false; } });
  // snow, drifting down behind and beside the ring
  const N = 110, where = new Float32Array(N * 3), fall = new Float32Array(N);
  for (let i = 0; i < N; i++) { where[i * 3] = (rnd() - 0.5) * 22; where[i * 3 + 1] = rnd() * 8; where[i * 3 + 2] = -10 + rnd() * 10.5; fall[i] = 0.25 + rnd() * 0.3; }
  const flakes = new THREE.BufferGeometry();
  flakes.setAttribute('position', new THREE.BufferAttribute(where, 3));
  scene.add(new THREE.Points(flakes, new THREE.PointsMaterial({ color: 0xffffff, size: 0.06, transparent: true, opacity: 0.6, depthWrite: false })));

  return {
    tick(t, dt, cheer) {
      for (const c of crowd) {
        // still while they watch; a small hop when a blow lands (the keener ones higher)
        c.g.position.y = c.y + Math.abs(Math.sin(t * 8 + c.phase)) * 0.14 * cheer * c.keen;
      }
      for (const b of bulbs) b.m.scale.setScalar(0.92 + 0.1 * Math.sin(t * 1.2 + b.phase));
      for (let i = 0; i < N; i++) {
        where[i * 3 + 1] -= fall[i] * dt;
        where[i * 3] += Math.sin(t * 0.7 + i) * dt * 0.12;
        if (where[i * 3 + 1] < -0.4) where[i * 3 + 1] = 8;
      }
      flakes.attributes.position.needsUpdate = true;
    },
  };
}
