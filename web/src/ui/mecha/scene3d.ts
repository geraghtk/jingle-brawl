// Mecha Santa Mayhem's rooftops, in 3D (three.js, loaded only when this game is on a screen). Two giant
// wind-up Santas, one per team, walk two snowy rooftop tracks side by side, seen from the side; the
// camera goes along with them and pulls back when one gets ahead. Everything they do is read from the
// room's view each frame (a step from its start and landing times, the lean from its growth, a fall,
// a jump): the scene decides nothing.
import * as THREE from 'three';
import { leanAt, posAt } from '../../core/minigames/engines/walker';
import type { Walker } from '../../core/minigames/engines/walker';
import type { MechaView } from '../../core/minigames/mecha/view';
import { at, ball, box, cone, cyl, disposeScene, ease, GLOVE, GOLD, INK, piece, SNOW, STEEL } from '../toon3d';

/** World units per course unit, and where the two tracks lie (team A at the back, B at the front). */
const S = 0.62;
const TRACK_Z = [-1.25, 1.25];
const ROOF_Y = 0;
/** How far apart (world units) the two may be and still both be in the picture. */
const MAX_SPREAD = 15;
const WHITE = 0xffffff;

interface Robot {
  root: THREE.Group;        // stands on the roof at its place along the track, faces +x
  tilt: THREE.Group;        // everything: leans forward and back from the feet, lies down when it falls
  body: THREE.Group;        // above the hips (bobs, crouches)
  legs: [THREE.Group, THREE.Group];
  arms: [THREE.Group, THREE.Group];
  key: THREE.Object3D;
  flame: THREE.Object3D;
  lean: number; crouch: number; swing: number; y: number;
}

/** A part that glows (a lamp, an eye, a dial): its own light, no shading, no outline. */
const glow = (geo: THREE.BufferGeometry, color: number) => new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color }));
const dome = (r: number, c: number) => piece(new THREE.SphereGeometry(r, 22, 12, 0, Math.PI * 2, 0, Math.PI / 2), c);
const HIP = 0.95;

/** One giant wind-up Santa: a riveted clockwork robot in a Santa suit, about 3 units tall, in its team's
    coat. Built facing +x; the camera sees it from the side, so what says "robot" (the piston legs, the
    shoulder plates, the dial in its side, the bolt in its ear, the chimney and the key in its back) is
    on its sides and its back. */
function buildSanta(color: number): Robot {
  const root = new THREE.Group(), tilt = new THREE.Group();
  tilt.scale.setScalar(0.9);
  root.add(tilt);
  const leg = (z: number) => {
    const hip = at(new THREE.Group(), 0, HIP, z);
    hip.add(at(ball(0.18, STEEL), 0, 0, 0));                                              // hip joint
    hip.add(at(cyl(0.1, 0.1, 0.36, STEEL, 12), 0, -0.2, 0));                              // piston
    hip.add(at(ball(0.15, GOLD), 0, -0.4, 0));                                            // knee
    hip.add(at(cyl(0.17, 0.22, 0.3, color, 14), 0, -0.56, 0));                            // shin plate
    hip.add(at(cyl(0.24, 0.24, 0.1, WHITE, 14), 0, -0.7, 0));                             // fur cuff
    hip.add(at(box(0.66, 0.22, 0.44, INK), 0.1, -0.84, 0));                               // boot
    hip.add(at(box(0.2, 0.18, 0.46, STEEL), 0.37, -0.86, 0));                             // steel toe cap
    tilt.add(hip);
    return hip;
  };
  const legs: [THREE.Group, THREE.Group] = [leg(-0.34), leg(0.34)];
  const body = at(new THREE.Group(), 0, HIP, 0);
  tilt.add(body);
  body.add(at(piece(new THREE.CylinderGeometry(0.6, 0.78, 1.0, 26), color), 0, 0.6, 0));  // the boiler of a coat
  body.add(at(dome(0.6, color), 0, 1.1, 0));
  body.add(at(cyl(0.8, 0.82, 0.14, WHITE, 28), 0, 0.1, 0));                               // coat hem
  body.add(at(cyl(0.76, 0.77, 0.17, INK, 28), 0, 0.38, 0));                               // belt
  body.add(at(box(0.1, 0.28, 0.36, GOLD), 0.76, 0.38, 0));                                // buckle
  body.add(at(box(0.08, 0.62, 0.2, WHITE), 0.66, 0.8, 0));                                // the fur down the front
  for (const z of [-1, 1]) {
    for (const x of [-0.42, 0, 0.42]) body.add(at(ball(0.055, GOLD), x, 0.38, z * Math.sqrt(0.78 * 0.78 - x * x)));   // belt studs
    // a pressure dial in each side
    const ring = piece(new THREE.TorusGeometry(0.2, 0.05, 10, 22), GOLD, 0.025);
    body.add(at(ring, 0.06, 0.8, z * 0.67));
    const glass = glow(new THREE.CylinderGeometry(0.18, 0.18, 0.04, 20), 0xbdeeff);
    glass.rotation.x = Math.PI / 2;
    body.add(at(glass, 0.06, 0.8, z * 0.67));
    const needle = at(box(0.03, 0.16, 0.03, INK), 0.09, 0.84, z * 0.7);
    needle.rotation.z = -0.6;
    body.add(needle);
    // shoulder plates
    body.add(at(dome(0.3, color), 0, 1.08, z * 0.8));
    body.add(at(cyl(0.31, 0.31, 0.07, WHITE, 18), 0, 1.06, z * 0.8));
  }
  // in its back: the wind-up key, a little chimney, the rocket's nozzle
  const key = at(new THREE.Group(), -0.72, 0.72, 0);
  key.add(at(cyl(0.05, 0.05, 0.34, STEEL, 10), -0.14, 0, 0).rotateZ(Math.PI / 2));
  const bow = piece(new THREE.TorusGeometry(0.24, 0.07, 10, 22), GOLD, 0.025);
  bow.rotation.y = Math.PI / 2;
  key.add(at(bow, -0.42, 0, 0));
  body.add(key);
  body.add(at(cyl(0.15, 0.17, 0.56, 0x8c5a3c, 12), -0.42, 1.42, 0.28));
  body.add(at(cyl(0.2, 0.2, 0.09, WHITE, 14), -0.42, 1.72, 0.28));
  body.add(at(cyl(0.13, 0.2, 0.22, STEEL, 14), -0.82, 0.3, 0).rotateZ(-Math.PI / 2));
  // head: a steel helmet with a visor, lamp eyes, a lamp nose, a plated beard, a bolt in each ear, the hat
  const head = at(new THREE.Group(), 0.06, 1.52, 0);
  head.add(at(cyl(0.2, 0.26, 0.16, STEEL, 16), 0, -0.04, 0));
  head.add(at(ball(0.44, 0xffdcc0), 0, 0.3, 0));                                         // a painted tin face
  head.add(at(box(0.5, 0.15, 0.62, INK), 0.2, 0.42, 0));                                  // visor
  for (const z of [-0.17, 0.17]) head.add(at(glow(new THREE.SphereGeometry(0.075, 14, 10), 0xffe27a), 0.45, 0.4, z));
  for (const z of [-0.34, 0.34]) head.add(at(glow(new THREE.SphereGeometry(0.06, 12, 8), 0xffe27a), 0.24, 0.4, z));     // (seen from the side)
  head.add(at(glow(new THREE.SphereGeometry(0.1, 16, 12), 0xff5a5a), 0.52, 0.26, 0));     // nose
  for (const [r, x, y] of [[0.4, 0.15, 0.0], [0.31, 0.24, -0.26], [0.2, 0.3, -0.5]] as const) {
    const plate = piece(new THREE.SphereGeometry(r, 20, 14), WHITE);
    plate.scale.set(0.85, 1, 1.08);
    head.add(at(plate, x, y, 0));
  }
  for (const z of [-0.15, 0.15]) { const m = piece(new THREE.SphereGeometry(0.12, 14, 10), WHITE); m.scale.set(1, 0.7, 1.7); head.add(at(m, 0.44, 0.17, z)); }
  for (const z of [-1, 1]) {
    head.add(at(cyl(0.13, 0.13, 0.12, GOLD, 14), 0, 0.3, z * 0.44).rotateX(Math.PI / 2));
    head.add(at(box(0.16, 0.035, 0.03, INK), 0, 0.3, z * 0.51));
  }
  head.add(at(cyl(0.47, 0.47, 0.16, WHITE, 24), 0, 0.64, 0));                             // hat brim
  const hat = cone(0.42, 0.8, color, 20);
  hat.rotation.z = 0.5;
  head.add(at(hat, -0.18, 1.02, 0));
  head.add(at(ball(0.15, WHITE), -0.48, 1.3, 0));
  body.add(head);
  const arm = (z: number) => {
    const shoulder = at(new THREE.Group(), 0, 1.04, z);
    shoulder.add(at(ball(0.15, STEEL), 0, -0.04, 0));
    shoulder.add(at(cyl(0.09, 0.09, 0.3, STEEL, 12), 0, -0.22, 0));
    shoulder.add(at(ball(0.12, GOLD), 0, -0.4, 0));                                       // elbow
    shoulder.add(at(cyl(0.14, 0.19, 0.34, color, 14), 0, -0.6, 0));                       // sleeve plate
    shoulder.add(at(cyl(0.21, 0.21, 0.1, WHITE, 14), 0, -0.78, 0));
    shoulder.add(at(ball(0.24, GLOVE), 0, -0.96, 0));                                     // mitten
    shoulder.add(at(ball(0.1, GLOVE), 0.19, -0.9, 0));
    body.add(shoulder);
    return shoulder;
  };
  const arms: [THREE.Group, THREE.Group] = [arm(-0.9), arm(0.9)];
  // the rocket's flame, for a boost
  const flame = at(cone(0.24, 1.0, 0xffa726, 12), -1.42, 0.3, 0);
  flame.rotation.z = Math.PI / 2;
  flame.visible = false;
  body.add(flame);
  root.traverse((o) => { const m = (o as THREE.Mesh).material as THREE.MeshToonMaterial | undefined; if (m?.type === 'MeshToonMaterial' && m.color.getHex() === color) o.userData.coat = true; });
  return { root, tilt, body, legs, arms, key, flame, lean: 0, crouch: 0, swing: 0, y: 0 };
}

interface Spark { obj: THREE.Object3D; born: number; life: number; v: THREE.Vector3 }
interface WallBuilt { id: string; bricks: THREE.Object3D[][] }   // per track

export interface MechaScene {
  /** Draw the race as the room's view has it at `now` (the room's clock). `colors`: team A's and B's. */
  frame(view: MechaView, now: number, colors: [string, string]): void;
  resize(): void;
  dispose(): void;
}

export function createMechaScene(canvas: HTMLCanvasElement): MechaScene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.1, 200);

  scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x6f7fa8, 1.45));
  const moon = new THREE.DirectionalLight(0xffffff, 1.7);
  moon.castShadow = true;
  moon.shadow.mapSize.set(1024, 1024);
  Object.assign(moon.shadow.camera, { left: -14, right: 14, top: 10, bottom: -6, near: 1, far: 40 });
  scene.add(moon, moon.target);

  const robots: [Robot, Robot] = [buildSanta(0xff3366), buildSanta(0x18b88a)];
  robots.forEach((r, i) => { r.root.position.z = TRACK_Z[i]; scene.add(r.root); });
  let tinted: [string, string] = ['', ''];
  function tint(r: Robot, hex: string) {
    const c = new THREE.Color(hex);
    r.root.traverse((o) => { if (o.userData.coat) ((o as THREE.Mesh).material as THREE.MeshToonMaterial).color.copy(c); });
  }

  // the course is built from the first view (its length and what is on it never change)
  let built = false;
  const walls: WallBuilt[] = [];
  const boosts: { id: string; objs: THREE.Object3D[] }[] = [];
  const course = new THREE.Group();
  scene.add(course);
  const noShadow = (o: THREE.Object3D) => { o.traverse((m) => { (m as THREE.Mesh).castShadow = false; (m as THREE.Mesh).receiveShadow = true; }); return o; };
  function build(view: MechaView) {
    built = true;
    const len = view.course.length, gaps = view.course.obstacles.filter((o) => o.kind === 'gap');
    // the rooftops: one long snowy roof per track, cut where the gaps are
    const cuts: [number, number][] = [];
    let from = -6;
    for (const g of gaps) { cuts.push([from, g.at]); from = g.at + (g.width || 0); }
    cuts.push([from, len + 8]);
    TRACK_Z.forEach((z) => {
      for (const [a, b] of cuts) {
        const w = (b - a) * S;
        course.add(noShadow(at(box(w, 0.5, 1.9, SNOW), (a + b) / 2 * S, ROOF_Y - 0.25, z)));
        course.add(noShadow(at(box(w, 16, 1.7, 0x5a4a78), (a + b) / 2 * S, ROOF_Y - 8.5, z)));           // the house under it
        for (let x = a + 2; x < b - 1; x += 5) for (const y of [-1.4, -3.2, -5]) course.add(noShadow(at(box(0.5, 0.6, 0.06, 0xffe08a), x * S, ROOF_Y + y, z + 0.86)));   // lit windows
      }
    });
    // the start line and the finish arch
    course.add(at(box(0.12, 0.02, 4.6, GLOVE), 0, ROOF_Y + 0.011, 0));
    const fx = len * S;
    for (const z of [-2.45, 2.45]) {
      const post = new THREE.Group();
      for (let k = 0; k < 9; k++) post.add(at(cyl(0.11, 0.11, 0.34, k % 2 ? 0xffffff : GLOVE, 12), 0, 0.17 + k * 0.34, 0));
      course.add(at(post, fx, ROOF_Y, z));
    }
    course.add(at(box(0.16, 0.5, 5.1, GOLD), fx, ROOF_Y + 3.3, 0));
    for (let k = 0; k < 8; k++) course.add(noShadow(at(box(0.14, 0.02, 0.58, k % 2 ? INK : 0xffffff), fx, ROOF_Y + 0.012, -2.05 + k * 0.585)));
    // what is in the way, on both tracks
    for (const o of view.course.obstacles) {
      if (o.kind === 'wall') {
        const w: WallBuilt = { id: o.id, bricks: [] };
        TRACK_Z.forEach((z) => {
          const bricks: THREE.Object3D[] = [];
          for (let row = 0; row < 4; row++) for (let k = 0; k < 3; k++) {
            const b = at(box(0.42, 0.42, 0.56, row % 2 ? 0xe6eef7 : 0xffffff), o.at * S + 0.62 + (row % 2 ? 0.05 : -0.05), ROOF_Y + 0.21 + row * 0.42, z - 0.58 + k * 0.58);
            course.add(b);
            bricks.push(b);
          }
          w.bricks.push(bricks);
        });
        walls.push(w);
      } else if (o.kind === 'boost') {
        const objs: THREE.Object3D[] = [];
        TRACK_Z.forEach((z) => {
          const g = new THREE.Group();
          for (let k = 0; k < 5; k++) g.add(at(cyl(0.13, 0.13, 0.26, k % 2 ? 0xffffff : GLOVE, 12), 0, 0.13 + k * 0.26, 0));
          g.add(at(cone(0.16, 0.34, 0xffa726, 12), 0, 1.5, 0));
          g.add(at(ball(0.1, 0xfff3b0), 0, 1.38, 0));
          course.add(at(g, o.at * S, ROOF_Y, z - 0.82));
          objs.push(g);
        });
        boosts.push({ id: o.id, objs });
      }
    }
    // the town behind: rows of dark houses, chimneys, a few trees
    for (let x = -12; x < len * S + 14; x += 2.6) {
      const h = 2.2 + ((Math.sin(x * 12.9898) * 43758.5453) % 1 + 1) % 1 * 2.4;
      const house = noShadow(at(box(2.1, h, 1.4, 0x3a4a78), x, ROOF_Y - 3 + h / 2, -5.2));
      course.add(house, noShadow(at(box(2.3, 0.3, 1.6, 0xdfe8f5), x, ROOF_Y - 3 + h + 0.15, -5.2)));
      if (Math.round(x * 10) % 3 === 0) course.add(noShadow(at(box(0.4, 0.8, 0.4, 0x6b3d2a), x + 0.5, ROOF_Y - 3 + h + 0.7, -5.2)));
      if (Math.round(x * 10) % 4 === 0) { course.add(noShadow(at(cone(0.7, 1.8, 0x1f7a5c, 8), x + 1.2, ROOF_Y - 1.1, -3.4))); }
    }
  }

  const sparks: Spark[] = [];
  const starGeo = new THREE.OctahedronGeometry(0.1);
  function burst(x: number, y: number, z: number, color: number, n: number, speed: number) {
    for (let i = 0; i < n; i++) {
      const s = new THREE.Mesh(starGeo, new THREE.MeshBasicMaterial({ color, transparent: true }));
      s.position.set(x, y, z);
      const a = Math.random() * Math.PI * 2;
      scene.add(s);
      sparks.push({ obj: s, born: performance.now(), life: 500 + Math.random() * 300, v: new THREE.Vector3(Math.cos(a) * speed, Math.abs(Math.sin(a)) * speed + 1.2, (Math.random() - 0.5) * speed) });
    }
  }

  const seen = [-1, -1];
  let last = performance.now(), shake = 0, primed = false;
  const camAt = new THREE.Vector3(0, 2.2, 14), lookAt = new THREE.Vector3(0, 1.4, 0);

  function pose(r: Robot, w: Walker, view: MechaView, i: number, now: number, dt: number, t: number) {
    const p = view.teams[i].walk;
    const x = posAt(w, now);
    const m = w.move && now < w.move.lands ? w.move : null;
    const f = m ? Math.max(0, Math.min(1, (now - m.at) / Math.max(1, m.lands - m.at))) : 0;
    const busy = w.busy && now < w.busyUntil ? w.busy : null;
    const won = view.stage === 'done' && view.winner === view.teams[i].teamId;
    // up and down: a hop with each step, a leap over a gap, down in the gap after stepping into it
    let y = 0;
    if (m) y = Math.sin(f * Math.PI) * (m.kind === 'jump' ? 1.7 : m.kind === 'boost' ? 0.5 : 0.12);
    if (busy === 'climb') { const left = (w.busyUntil - now) / p.climbMs; y = -1.7 * Math.min(1, left * 1.6); }
    if (won) y = Math.abs(Math.sin(t * 5)) * 0.35;
    r.y = busy === 'climb' ? ease(r.y, y, dt, 8) : y;
    r.root.position.x = x * S + (busy === 'climb' ? 0.55 : 0);
    r.root.position.y = ROOF_Y + r.y;
    // the lean: forward (screen right) and back; flat on the roof when it has fallen, up again at the end
    let lean = leanAt(w, now, p) * 0.62;
    if (busy === 'topple') { const left = w.busyUntil - now; lean = (w.fell || 1) * (left > 700 ? 1.5 : 1.5 * (left / 700)); }
    if (busy === 'stumble') lean += Math.sin(t * 22) * 0.16;
    if (busy === 'misjump') lean += Math.sin(t * 18) * 0.08;
    r.lean = ease(r.lean, lean, dt, busy === 'topple' ? 9 : 16);
    r.tilt.rotation.z = -r.lean;
    // legs: the stepping one swings through, the other pushes back; both tuck for a jump
    const stepLeg = m && m.kind === 'step' ? m.leg : -1;
    r.legs.forEach((leg, k) => {
      let a = 0;
      if (stepLeg === k) a = Math.sin(f * Math.PI) * 0.8 - 0.15 * f;
      else if (stepLeg >= 0) a = -Math.sin(f * Math.PI) * 0.35;
      if (m && m.kind === 'jump') a = 0.9 * Math.sin(f * Math.PI) * (k === 0 ? 1 : -0.6);
      if (m && m.kind === 'boost') a = -0.5;
      if (busy === 'stumble') a = Math.sin(t * 20 + k * 2) * 0.5;
      leg.rotation.z = ease(leg.rotation.z, a, dt, 24);
    });
    // crouch for a jump; bob with the clockwork
    r.crouch = ease(r.crouch, w.crouch ? 1 : 0, dt, 14);
    r.body.position.y = HIP - r.crouch * 0.24 + Math.sin(t * 6 + i * 1.3) * 0.015;
    r.key.rotation.x = t * (busy ? 0.6 : 4);
    // arms: swing for a hit on a wall, up for a boost or a win, out for balance otherwise
    const hit = w.events.length && ['hit', 'smashed', 'whiff'].includes(w.events[w.events.length - 1].kind) ? now - w.events[w.events.length - 1].at : 9999;
    r.swing = hit < 220 ? Math.sin((hit / 220) * Math.PI) : ease(r.swing, 0, dt, 12);
    r.arms.forEach((arm, k) => {
      let a = 0.25 + Math.abs(r.lean) * 0.9 + r.swing * 1.9, out = (k === 0 ? 1 : -1) * (0.25 + Math.abs(r.lean) * 0.5);
      if (m && m.kind === 'boost') a = 3.0;
      if (m && m.kind === 'jump') a = 2.4;
      if (won) { a = 2.9 + Math.sin(t * 9 + k) * 0.15; out = (k === 0 ? 1 : -1) * 0.5; }
      if (busy === 'topple') a = 1.4;
      arm.rotation.z = ease(arm.rotation.z, a, dt, 20);
      arm.rotation.x = out;
    });
    r.flame.visible = !!m && m.kind === 'boost';
    if (r.flame.visible) r.flame.scale.setScalar(0.8 + Math.random() * 0.5);
  }

  function frame(view: MechaView, now: number, colors: [string, string]) {
    const stamp = performance.now(), dt = Math.min(0.05, (stamp - last) / 1000), t = stamp / 1000;
    last = stamp;
    if (!built && view.course?.length) build(view);
    if (colors[0] !== tinted[0] || colors[1] !== tinted[1]) { tint(robots[0], colors[0]); tint(robots[1], colors[1]); tinted = [colors[0], colors[1]]; }
    view.teams.slice(0, 2).forEach((team, i) => {
      const w = team.walker, r = robots[i];
      pose(r, w, view, i, now, dt, t);
      // walls come down brick by brick; a boost that is taken is gone
      for (const wall of walls) {
        const hits = w.hits[wall.id] || 0, bricks = wall.bricks[i], left = Math.ceil(bricks.length * Math.max(0, 1 - hits / team.walk.wallHits));
        bricks.forEach((b, k) => { b.visible = k < left; });
      }
      for (const b of boosts) b.objs[i].visible = !w.taken.includes(b.id);
      // what just happened to this robot
      const newest = w.events.length ? w.events[w.events.length - 1].n : seen[i];
      if (primed) for (const e of w.events) {
        if (e.n <= seen[i]) continue;
        const x = r.root.position.x, z = TRACK_Z[i];
        if (e.kind === 'topple') { burst(x, 0.4, z, 0xffffff, 18, 3.2); shake = Math.max(shake, 0.35); }
        else if (e.kind === 'stumble' || e.kind === 'misjump') burst(x, 0.3, z, 0xdfe8f2, 6, 1.8);
        else if (e.kind === 'hit') { burst(x + 0.9, 1.2, z, 0xffffff, 6, 2.6); shake = Math.max(shake, 0.08); }
        else if (e.kind === 'smashed') { burst(x + 0.9, 1.0, z, 0xffffff, 22, 4.2); burst(x + 0.9, 1.2, z, 0xffe27a, 8, 3); shake = Math.max(shake, 0.3); }
        else if (e.kind === 'boost') burst(x - 0.6, 1.4, z, 0xffa726, 14, 3);
        else if (e.kind === 'fell') { burst(x + 0.6, 0.3, z, 0xffffff, 12, 2.4); shake = Math.max(shake, 0.2); }
        else if (e.kind === 'finish') burst(x, 2.6, z, 0xffe27a, 40, 5);
        else if (e.kind === 'step' || e.kind === 'stride') burst(x - 0.2, 0.08, z, 0xffffff, 2, 0.9);
      }
      seen[i] = Math.max(seen[i], newest);
    });
    primed = true;

    for (let k = sparks.length - 1; k >= 0; k--) {
      const s = sparks[k], age = (stamp - s.born) / s.life;
      if (age >= 1) { scene.remove(s.obj); sparks.splice(k, 1); continue; }
      s.v.y -= 7 * dt;
      s.obj.position.addScaledVector(s.v, dt);
      s.obj.rotation.x += dt * 9; s.obj.rotation.y += dt * 7;
      s.obj.scale.setScalar(1 - age * 0.6);
      ((s.obj as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = 1 - age * age;
    }

    // the camera: between the two, far enough back to hold both, a little ahead of them
    const xa = robots[0].root.position.x, xb = robots[1].root.position.x;
    // (one far behind is left off the edge rather than shrinking both to specks: the bar below still shows it)
    const spread = Math.min(Math.abs(xa - xb), MAX_SPREAD), mid = Math.max(xa, xb) - spread / 2 + 2.2;
    const half = Math.max(9.5, spread / 2 + 5);
    const hfov = 2 * Math.atan(Math.tan((camera.fov * Math.PI) / 360) * camera.aspect);
    const dist = half / Math.tan(hfov / 2);
    shake = Math.max(0, shake - dt * 1.4);
    camAt.set(ease(camAt.x, mid, dt, 4), ease(camAt.y, 2.6 + dist * 0.2, dt, 4), ease(camAt.z, dist, dt, 3));
    lookAt.set(ease(lookAt.x, mid, dt, 4), 2.5, 0);
    camera.position.set(camAt.x + (Math.random() - 0.5) * shake * 0.4, camAt.y + (Math.random() - 0.5) * shake * 0.4, camAt.z);
    camera.lookAt(lookAt);
    moon.position.set(camAt.x - 4, 12, 9);
    moon.target.position.set(camAt.x, 0, 0);
    renderer.render(scene, camera);
  }

  let sized = '';
  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    if (sized === `${w}x${h}`) return;                                                 // (called every frame: the stage has no resize of its own to hear)
    sized = `${w}x${h}`;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();

  return { frame, resize, dispose() { disposeScene(scene, renderer); } };
}
