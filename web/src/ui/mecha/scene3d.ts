// Mecha Santa Mayhem's rooftops, in 3D (three.js, loaded only when this game is on a screen). Two giant
// wind-up Santas, one per team, walk two snowy rooftop tracks side by side, seen from the side; the
// camera goes along with them and pulls back when one gets ahead. Everything they do is read from the
// room's view each frame (a step from its start and landing times, the lean from its growth, a fall,
// a jump): the scene decides nothing.
import * as THREE from 'three';
import { leanAt, posAt, rungsOf, swingAt } from '../../core/minigames/engines/walker';
import type { Walker } from '../../core/minigames/engines/walker';
import type { MechaView } from '../../core/minigames/mecha/view';
import { at, ball, box, cone, cyl, disposeScene, ease, GLOVE, GOLD, INK, piece, SNOW, STEEL } from '../toon3d';
import { hangAt, PUNCH_MS, punchAt, ROPE, splitFor, STRIPS, touchdown, VAULT_CLEAR, vaultAt } from './fx';
import type { Flight } from './fx';

/** World units per course unit, and where the two tracks lie (team A at the back, B at the front). */
const S = 0.62;
const TRACK_Z = [-1.25, 1.25];
const ROOF_Y = 0;
/** How far apart (world units) the two may be and still both be in the picture (further: it splits, fx.ts). */
const MAX_SPREAD = 15;
/** How tall a climb stands, and how thick it is. */
const STACK_H = 3.1, STACK_W = 0.55, STACK_X = 0.72;
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
  /** Where it was when it last hung on a rope (world x, height): a let-go starts from there, not from the edge it grabbed at. */
  hung: { x: number; y: number } | null;
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
  return { root, tilt, body, legs, arms, key, flame, lean: 0, crouch: 0, swing: 0, y: 0, hung: null };
}

/** A bit of an effect: a star that flies and falls, or (`puff`) a soft ball of snow or smoke that swells, slows and fades. */
interface Spark { obj: THREE.Object3D; born: number; life: number; v: THREE.Vector3; puff?: { from: number; to: number; lift: number } }
interface WallBuilt { id: string; bricks: THREE.Object3D[][] }   // per track

export interface MechaScene {
  /** Draw the race as the room's view has it at `now` (the room's clock). `colors`: team A's and B's.
   *  Says whether the picture is split in two (one strip per robot) so the stage can label the strips. */
  frame(view: MechaView, now: number, colors: [string, string]): { split: boolean };
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
  interface Rope { id: string; pivot: THREE.Group; px: number }
  const ropeSets: Rope[][] = [];                                          // per rope: its two (one per track)
  const course = new THREE.Group();
  scene.add(course);
  const noShadow = (o: THREE.Object3D) => { o.traverse((m) => { (m as THREE.Mesh).castShadow = false; (m as THREE.Mesh).receiveShadow = true; }); return o; };
  function build(view: MechaView) {
    built = true;
    const len = view.course.length, gaps = view.course.obstacles.filter((o) => o.kind === 'gap' || o.kind === 'rope').sort((a, b) => a.at - b.at);
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
      } else if (o.kind === 'climb') {
        // a tall chimney stack across the track, rungs up the side it is climbed from
        const n = rungsOf(o, view.teams[0]?.walk);
        TRACK_Z.forEach((z) => {
          const x = o.at * S + STACK_X + STACK_W / 2;
          course.add(at(box(STACK_W, STACK_H, 1.7, 0x9a4b3c), x, ROOF_Y + STACK_H / 2, z));
          course.add(at(box(STACK_W + 0.16, 0.22, 1.86, 0x6b3d2a), x, ROOF_Y + STACK_H + 0.11, z));
          course.add(noShadow(at(box(STACK_W + 0.2, 0.14, 1.9, SNOW), x, ROOF_Y + STACK_H + 0.29, z)));
          for (let k = 0; k < n; k++) course.add(at(box(0.1, 0.07, 0.9, GOLD), x - STACK_W / 2 - 0.05, ROOF_Y + ((k + 1) / (n + 0.4)) * STACK_H, z));
          for (let row = 0; row < 6; row++) course.add(noShadow(at(box(STACK_W + 0.02, 0.03, 1.72, 0x7d3a2e), x, ROOF_Y + 0.5 + row * 0.5, z)));
        });
      } else if (o.kind === 'rope') {
        // a gantry over the gap with a rope hanging from it, one per track
        const px = (o.at + (o.width || 0) / 2) * S, ropes: Rope[] = [];
        TRACK_Z.forEach((z) => {
          course.add(at(cyl(0.09, 0.11, ROPE.pivotY + 0.6, STEEL, 10), px, ROOF_Y + (ROPE.pivotY + 0.6) / 2 - 3.2, z - 1.05));
          course.add(at(box(0.14, 0.14, 1.3, STEEL), px, ROOF_Y + ROPE.pivotY + 0.05, z - 0.45));
          course.add(at(ball(0.13, GOLD), px, ROOF_Y + ROPE.pivotY, z));
          const pivot = at(new THREE.Group(), px, ROOF_Y + ROPE.pivotY, z);
          pivot.add(at(cyl(0.035, 0.035, ROPE.length, 0xd9b26a, 6), 0, -ROPE.length / 2, 0));
          pivot.add(at(ball(0.11, 0xd9b26a), 0, -ROPE.length, 0));
          course.add(pivot);
          ropes.push({ id: o.id, pivot, px });
        });
        ropeSets.push(ropes);
      } else if (o.kind === 'ice') {
        // a stretch of bare ice: pale blue and shiny on top of the snow
        TRACK_Z.forEach((z) => {
          const w = (o.width || 0) * S, cx = (o.at + (o.width || 0) / 2) * S;
          course.add(noShadow(at(box(w, 0.03, 1.74, 0x9fdcff), cx, ROOF_Y + 0.016, z)));
          for (let k = 0; k < Math.floor(w / 0.9); k++) course.add(noShadow(at(box(0.5, 0.005, 0.06, 0xffffff), o.at * S + 0.5 + k * 0.9, ROOF_Y + 0.034, z - 0.5 + (k % 3) * 0.45)));
        });
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

  // soft balls: the snow a foot kicks up, the dust of a landing, the smoke and fire behind a rocket candle
  const puffGeo = new THREE.SphereGeometry(0.5, 10, 8);
  function puff(x: number, y: number, z: number, color: number, size: number, life: number, vx: number, vy: number, vz: number, lift = 0) {
    const s = new THREE.Mesh(puffGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false }));
    s.position.set(x, y, z);
    s.scale.setScalar(size * 0.35);
    scene.add(s);
    sparks.push({ obj: s, born: performance.now(), life, v: new THREE.Vector3(vx, vy, vz), puff: { from: size * 0.35, to: size, lift } });
  }
  /** Snow thrown up where something comes down: a ring of puffs low along the roof. */
  function dust(x: number, z: number, n: number, size: number, speed: number) {
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2 + Math.random() * 0.6, v = speed * (0.6 + Math.random() * 0.6);
      // (white on a white roof would not show: every other puff is in the snow's own shadow colour, and they rise clear of it)
      puff(x + Math.cos(a) * 0.12, ROOF_Y + 0.12, z + Math.sin(a) * 0.12, k % 2 ? 0xffffff : 0xb9cbe6, size * (0.8 + Math.random() * 0.5), 440 + Math.random() * 220, Math.cos(a) * v, 1.1 + Math.random() * 0.9, Math.sin(a) * v * 0.6);
    }
  }
  /** The camera punch: one thump at a time; the bigger wins while the last one is still fresh. */
  let thump = 0, thumpAt = -1e9;
  function punch(amp: number) {
    const stamp = performance.now();
    if (stamp - thumpAt < PUNCH_MS / 2 && amp < thump) return;
    thump = amp; thumpAt = stamp;
  }

  const seen = [-1, -1];
  const air: [Flight | null, Flight | null] = [null, null];           // the move each robot was last seen making
  const trailAt = [0, 0];
  const nozzle = new THREE.Vector3();
  let last = performance.now(), lastNow = 0, primed = false;
  const camAt = new THREE.Vector3(0, 2.2, 14), lookAt = new THREE.Vector3(0, 1.4, 0);
  let split = false;
  const strips = [0, 1].map(() => ({ at: camAt.clone(), look: lookAt.clone() }));

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
    // on a climb: up the stack rung by rung, then over the top and down the far side
    const grip = w.grip;
    // (each rung is a heave: up quickly, then a moment hanging on, so it reads as climbing and not as walking up the wall)
    if (grip?.kind === 'climb') {
      const n = rungsOf(view.course.obstacles.find((o) => o.id === grip.id), p);
      const heave = m && m.kind === 'rung' ? 1 - (1 - f) * (1 - f) * (1 - f) : 0;
      y = Math.min(1, (grip.rung + heave) / n) * (STACK_H + VAULT_CLEAR - 0.9) + (m && m.kind === 'rung' ? 0 : Math.sin(t * 7) * 0.02);
    }
    let wx = x * S + (busy === 'climb' ? 0.55 : 0);
    if (grip?.kind === 'climb') wx += 0.16;                                  // pressed up against it
    if (m && m.kind === 'vault') {
      // over the top, clear of it, then down the far side (never through it)
      const v = vaultAt(f, STACK_H);
      y = v.y;
      wx = (m.from + (m.to - m.from) * v.over) * S;
    }
    // letting go of a rope: the leap starts from where it was hanging (not back at the edge it grabbed from)
    if (m && m.kind === 'jump' && r.hung) {
      wx = r.hung.x + (m.to * S - r.hung.x) * f;
      y = r.hung.y * (1 - f) + Math.sin(f * Math.PI) * 1.1;
    }
    r.y = busy === 'climb' || grip?.kind === 'climb' ? ease(r.y, y, dt, grip ? 18 : 8) : y;
    // on a rope: hanging from it, swinging with the clock (pulled in to it first)
    const rope = grip?.kind === 'rope' ? ropeSets.find((set) => set[0].id === grip.id)?.[i] : undefined;
    if (rope && grip) {
      const hang = hangAt(swingAt(grip, now, p));
      const k = m && m.kind === 'pull' ? f : 1;
      wx = wx + (rope.px + hang.dx - wx) * k;
      r.y = r.y + (hang.y - r.y) * k;
      r.hung = { x: wx, y: r.y };
    } else if (!(m && m.kind === 'jump') && !(busy === 'climb')) r.hung = null;
    // dropped off the rope into the gap: it falls from where it hung, it does not jump back to the edge
    if (busy === 'climb' && r.hung) { r.hung.x = ease(r.hung.x, wx, dt, 5); wx = r.hung.x; }
    r.root.position.x = wx;
    r.root.position.y = ROOF_Y + r.y;
    // the lean: forward (screen right) and back; flat on the roof when it has fallen, up again at the end
    // (on a climb it leans in to the wall, chest to the bricks, and sways a little with each heave)
    let lean = grip?.kind === 'climb' ? 0.3 + leanAt(w, now, p) * 0.12 + (m && m.kind === 'rung' ? Math.sin(f * Math.PI) * 0.06 : 0) : leanAt(w, now, p) * 0.62;
    if (m && m.kind === 'vault') lean = 0.5 * Math.sin(Math.min(1, f / 0.6) * Math.PI);
    if (rope && grip) lean = hangAt(swingAt(grip, now, p)).angle * 0.5;
    if (busy === 'topple') { const left = w.busyUntil - now; lean = (w.fell || 1) * (left > 700 ? 1.5 : 1.5 * (left / 700)); }
    if (busy === 'stumble') lean += Math.sin(t * 22) * 0.16;
    if (busy === 'misjump') lean += Math.sin(t * 18) * 0.08;
    r.lean = ease(r.lean, lean, dt, busy === 'topple' ? 9 : 16);
    r.tilt.rotation.z = -r.lean;
    // legs: the stepping one swings through, the other pushes back; both tuck for a jump
    const stepLeg = m && m.kind === 'step' ? m.leg : -1;
    const rungLeg = m && m.kind === 'rung' ? m.leg : -1;
    r.legs.forEach((leg, k) => {
      let a = 0;
      if (stepLeg === k) a = Math.sin(f * Math.PI) * 0.8 - 0.15 * f;
      else if (stepLeg >= 0) a = -Math.sin(f * Math.PI) * 0.35;
      if (m && m.kind === 'jump') a = 0.9 * Math.sin(f * Math.PI) * (k === 0 ? 1 : -0.6);
      if (m && m.kind === 'boost') a = -0.5;
      if (m && m.kind === 'vault') a = f < 0.6 ? 1.0 * (k === 0 ? 1 : 0.4) : 0.15;
      // climbing: the knee comes up to the next rung and pushes down; the other leg hangs straight under it
      if (grip?.kind === 'climb') a = rungLeg === k ? 1.25 * Math.sin(Math.min(1, f * 1.6) * Math.PI * 0.5) * (1 - Math.max(0, f - 0.6) / 0.4) : (w.last === k ? 0.55 : -0.12);
      if (rope) a = Math.sin(t * 3 + k * 1.7) * 0.3 - 0.2;                 // dangling
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
      // on a rope both hands are up on it; on a climb they go hand over hand, the hand on the side of the leg that is
      // stepping reaching for the next hold as the other pulls down
      if (grip?.kind === 'rope') { a = 3.05; out = 0; }
      if (grip?.kind === 'climb') { const reach = rungLeg === k ? Math.sin(f * Math.PI) : 0, high = (grip.rung + k) % 2 === 0; a = (high ? 2.75 : 2.05) + reach * 0.5 - (rungLeg >= 0 && rungLeg !== k ? Math.sin(f * Math.PI) * 0.35 : 0); out = (k === 0 ? 1 : -1) * 0.12; }
      if (m && m.kind === 'vault') { a = f < 0.6 ? 2.2 : 2.9; out = (k === 0 ? 1 : -1) * 0.4; }
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
        if (now - e.at > 1500) continue;                                  // (old news: this screen was away)
        if (e.kind === 'topple') { burst(x, 0.4, z, 0xffffff, 18, 3.2); dust(x + (w.fell || 1) * 1.3, z, 9, 0.95, 2.4); punch(0.42); }
        else if (e.kind === 'stumble' || e.kind === 'misjump') { burst(x, 0.3, z, 0xdfe8f2, 6, 1.8); dust(x, z, 4, 0.5, 1.2); }
        else if (e.kind === 'hit') { burst(x + 0.9, 1.2, z, 0xffffff, 6, 2.6); punch(0.1); }
        else if (e.kind === 'smashed') { burst(x + 0.9, 1.0, z, 0xffffff, 22, 4.2); burst(x + 0.9, 1.2, z, 0xffe27a, 8, 3); dust(x + 1.0, z, 8, 0.9, 2.6); punch(0.36); }
        else if (e.kind === 'boost') burst(x - 0.6, 1.4, z, 0xffa726, 14, 3);
        else if (e.kind === 'fell') { burst(x + 0.6, 0.3, z, 0xffffff, 12, 2.4); punch(0.28); }
        else if (e.kind === 'finish') burst(x, 2.6, z, 0xffe27a, 40, 5);
        else if (e.kind === 'grabbed') burst(x + 0.5, 2.4, z, 0xffe27a, 6, 2);
        else if (e.kind === 'slip' || e.kind === 'dropped') { burst(x, 1.2, z, 0xdfe8f2, 6, 1.8); dust(x, z, 4, 0.5, 1.2); }
        else if (e.kind === 'topped') burst(x + 0.6, STACK_H + 1, z, 0xffe27a, 16, 3.4);
        else if (e.kind === 'pump') burst(x, 2.6, z, 0xffffff, 5, 1.6);
      }
      seen[i] = Math.max(seen[i], newest);
      // a foot comes down: the move this screen was watching has landed (a puff of snow under that boot;
      // more snow and a thump after a jump)
      const td = touchdown(air[i], w.move, now, w.busy && now < w.busyUntil ? w.busy : null);
      air[i] = td.air;
      if (primed && td.landed) {
        const x = r.root.position.x, z = TRACK_Z[i];
        if (td.landed.kind === 'step') dust(x + 0.25, z + (td.landed.leg === 0 ? -0.3 : 0.3), 5, 0.5, 1.5);
        else if (td.landed.kind === 'jump' || td.landed.kind === 'vault') { dust(x + 0.1, z, 10, 0.8, 2.3); punch(0.24); }
        else if (td.landed.kind === 'rung' || td.landed.kind === 'pull') { /* a hand on a rung: no snow */ }
        else dust(x + 0.1, z, 6, 0.6, 1.6);
      }
      // a rocket candle burning: it leaves fire and smoke behind it (only while the clock runs)
      const boosting = !!w.move && w.move.kind === 'boost' && now < w.move.lands;
      if (primed && boosting && now !== lastNow && stamp - trailAt[i] >= 26) {
        trailAt[i] = stamp;
        r.flame.getWorldPosition(nozzle);
        const j = () => (Math.random() - 0.5) * 0.16;
        puff(nozzle.x + j(), nozzle.y + j(), nozzle.z + j(), Math.random() < 0.5 ? 0xffa726 : 0xffd84a, 0.34, 300, -1.6, j() * 3, j() * 3);
        puff(nozzle.x - 0.15 + j(), nozzle.y + j(), nozzle.z + j(), 0xf4f8fc, 0.6, 850, -0.5, 0.25, j() * 2, 0.9);
      }
    });
    primed = true;
    lastNow = now;

    for (let k = sparks.length - 1; k >= 0; k--) {
      const s = sparks[k], age = (stamp - s.born) / s.life;
      const mat = (s.obj as THREE.Mesh).material as THREE.MeshBasicMaterial;
      if (age >= 1) { scene.remove(s.obj); mat.dispose(); sparks.splice(k, 1); continue; }
      if (s.puff) {
        // a puff swells fast then holds, slows in the air, and thins out
        s.v.multiplyScalar(Math.max(0, 1 - 3.2 * dt));
        s.v.y += s.puff.lift * dt;
        s.obj.position.addScaledVector(s.v, dt);
        s.obj.scale.setScalar(s.puff.from + (s.puff.to - s.puff.from) * (1 - (1 - age) * (1 - age)));
        mat.opacity = 0.9 * (1 - age * age);
        continue;
      }
      s.v.y -= 7 * dt;
      s.obj.position.addScaledVector(s.v, dt);
      s.obj.rotation.x += dt * 9; s.obj.rotation.y += dt * 7;
      s.obj.scale.setScalar(1 - age * 0.6);
      mat.opacity = 1 - age * age;
    }

    // each rope hangs with whoever is on it (else it sways a little by itself)
    for (const set of ropeSets) set.forEach((rope, i) => {
      const g = view.teams[i]?.walker.grip;
      const held = g?.kind === 'rope' && g.id === rope.id ? g : null;
      const want = held ? hangAt(swingAt(held, now, view.teams[i].walk)).angle : Math.sin(t * 1.3 + i) * 0.06;
      rope.pivot.rotation.z = ease(rope.pivot.rotation.z, want, dt, 12);
    });

    // the camera: between the two, far enough back to hold both, a little ahead of them. When one is too far
    // behind for that (Kevin, 2026-11-23: it used to walk off the screen), the picture splits into two strips,
    // one camera each, and joins again when they are close.
    const xa = robots[0].root.position.x, xb = robots[1].root.position.x;
    split = splitFor(xa - xb, split);
    const age = stamp - thumpAt, py = punchAt(thump, age), px = punchAt(thump * 0.35, age + 22);
    const W = canvas.clientWidth || 1, H = canvas.clientHeight || 1;
    const shoot = (c: { at: THREE.Vector3; look: THREE.Vector3 }, aspect: number) => {
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      camera.position.set(c.at.x + px, c.at.y + py, c.at.z);
      camera.lookAt(c.look.x + px, c.look.y + py, c.look.z);
      moon.position.set(c.at.x - 4, 12, 9);
      moon.target.position.set(c.at.x, 0, 0);
      renderer.render(scene, camera);
    };
    if (!split) {
      const spread = Math.min(Math.abs(xa - xb), MAX_SPREAD), mid = Math.max(xa, xb) - spread / 2 + 2.2;
      const half = Math.max(9.5, spread / 2 + 5);
      const hfov = 2 * Math.atan(Math.tan((camera.fov * Math.PI) / 360) * (W / H));
      const dist = half / Math.tan(hfov / 2);
      camAt.set(ease(camAt.x, mid, dt, 4), ease(camAt.y, 2.6 + dist * 0.2, dt, 4), ease(camAt.z, dist, dt, 3));
      lookAt.set(ease(lookAt.x, mid, dt, 4), 2.5, 0);
      // (the strip cameras wait where the one camera is, so a split starts from this picture)
      strips.forEach((c) => { c.at.copy(camAt); c.look.copy(lookAt); });
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, W, H);
      shoot({ at: camAt, look: lookAt }, W / H);
      return { split: false };
    }
    // split: each strip follows its own robot, a little ahead of it, near enough to read
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, W, H);
    renderer.clear();
    renderer.setScissorTest(true);
    [xa, xb].forEach((x, i) => {
      const c = strips[i], strip = STRIPS[i];
      c.at.set(ease(c.at.x, x + 2.6, dt, 5), ease(c.at.y, 3.4, dt, 5), ease(c.at.z, 13.5, dt, 4));
      c.look.set(ease(c.look.x, x + 2.6, dt, 5), ease(c.look.y, 2.3, dt, 5), TRACK_Z[i] * 0.4);
      const y = Math.round(H * (1 - strip.top - strip.height)), h = Math.round(H * strip.height);
      renderer.setViewport(0, y, W, h);
      renderer.setScissor(0, y, W, h);
      shoot(c, W / h);
    });
    renderer.setScissorTest(false);
    // (the one camera follows the leader meanwhile, so joining again starts from somewhere sensible)
    camAt.set(Math.max(xa, xb) - MAX_SPREAD / 2 + 2.2, camAt.y, camAt.z);
    lookAt.set(camAt.x, 2.5, 0);
    return { split: true };
  }

  let sized = '';
  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    if (sized === `${w}x${h}`) return;                                                 // (called every frame: the stage has no resize of its own to hear)
    sized = `${w}x${h}`;
    renderer.setSize(w, h, false);
  }
  resize();

  return { frame, resize, dispose() { disposeScene(scene, renderer); } };
}
