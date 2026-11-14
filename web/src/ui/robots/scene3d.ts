// Nutcracker Knockout's ring, in 3D (three.js, loaded only when this game is on a screen). Two wind-up
// toy robots built from simple shapes with flat toon shading and dark outlines, to sit beside the
// game's sticker art. Everything they do is read from the room's view each frame (a punch from its
// start and landing times, a guard, a flinch): the scene decides nothing. A part the view says is
// lost is knocked off and flies; the head goes last and furthest.
import * as THREE from 'three';
import type { Part } from '../../core/minigames/robots/robots';
import type { RobotFighterView, RobotView } from '../../core/minigames/robots/view';

const INK = 0x1d2a3a;
const GOLD = 0xffc21a;
const STEEL = 0xc9d2de;
const GLOVE = 0xe63946;
const SNOW = 0xf4f8fc;

/** A shape in the house look: flat toon colour and an ink outline (the same shape, a little bigger, seen from inside). */
function piece(geo: THREE.BufferGeometry, color: number, outline = 0.035): THREE.Group {
  const g = new THREE.Group();
  const mesh = new THREE.Mesh(geo, new THREE.MeshToonMaterial({ color }));
  mesh.castShadow = true;
  const line = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: INK, side: THREE.BackSide }));
  geo.computeBoundingSphere();
  const r = geo.boundingSphere?.radius || 1;
  line.scale.setScalar(1 + outline / r);
  g.add(line, mesh);
  return g;
}
const box = (w: number, h: number, d: number, c: number) => piece(new THREE.BoxGeometry(w, h, d, 1, 1, 1), c);
const cyl = (rt: number, rb: number, h: number, c: number, seg = 20) => piece(new THREE.CylinderGeometry(rt, rb, h, seg), c);
const ball = (r: number, c: number) => piece(new THREE.SphereGeometry(r, 20, 14), c);
const at = <T extends THREE.Object3D>(o: T, x: number, y: number, z: number): T => { o.position.set(x, y, z); return o; };

interface Arm { shoulder: THREE.Group; elbow: THREE.Group }
interface Robot {
  root: THREE.Group;        // stands on the floor, faces +x
  body: THREE.Group;        // everything above the hips (leans, flinches)
  head: THREE.Group;
  arms: { L: Arm; R: Arm };
  key: THREE.Group;
  parts: Record<Part, THREE.Object3D>;
  /** What the view last said was lost (to see a part go). */
  gone: Set<Part>;
  /** Smoothed poses. */
  lean: number; guardAmt: number; reach: { L: number; R: number };
}

/** One robot, about 1.9 units tall, in its duelist's colour. Built facing +x. */
function buildRobot(color: number): Robot {
  const root = new THREE.Group();
  // legs and hips stay put; the rest leans from the hips
  root.add(at(box(0.22, 0.5, 0.26, INK), 0, 0.25, -0.17), at(box(0.22, 0.5, 0.26, INK), 0, 0.25, 0.17));
  root.add(at(box(0.3, 0.1, 0.3, STEEL), 0.04, 0.05, -0.17), at(box(0.3, 0.1, 0.3, STEEL), 0.04, 0.05, 0.17));
  const body = at(new THREE.Group(), 0, 0.52, 0);
  root.add(body);
  body.add(at(box(0.46, 0.2, 0.62, STEEL), 0, 0.08, 0));                                  // hips
  body.add(at(box(0.5, 0.62, 0.7, color), 0, 0.5, 0));                                    // chest
  body.add(at(box(0.52, 0.09, 0.72, GOLD), 0, 0.2, 0));                                   // belt
  const chestDoor = at(box(0.06, 0.34, 0.4, GOLD), 0.27, 0.52, 0);
  chestDoor.add(at(ball(0.035, INK), 0.04, 0, 0.13));
  body.add(chestDoor);
  // the wind-up key in its back
  const key = at(new THREE.Group(), -0.27, 0.55, 0);
  key.add(at(cyl(0.035, 0.035, 0.22, STEEL, 10), -0.1, 0, 0).rotateZ(Math.PI / 2));
  const bow = piece(new THREE.TorusGeometry(0.12, 0.04, 10, 20), GOLD, 0.02);
  bow.rotation.y = Math.PI / 2;
  key.add(at(bow, -0.26, 0, 0));
  body.add(key);
  // neck spring, head, hat
  body.add(at(cyl(0.09, 0.11, 0.12, STEEL, 12), 0, 0.87, 0));
  const head = at(new THREE.Group(), 0, 0.93, 0);
  head.add(at(box(0.44, 0.4, 0.46, 0xffe3c2), 0, 0.2, 0));
  head.add(at(box(0.1, 0.16, 0.36, 0xffffff), 0.2, 0.1, 0));                              // the nutcracker's teeth
  for (const z of [-0.11, 0.11]) {
    head.add(at(ball(0.06, 0xffffff), 0.21, 0.27, z));
    head.add(at(ball(0.032, INK), 0.26, 0.27, z));
    head.add(at(box(0.04, 0.035, 0.15, INK), 0.23, 0.36, z).rotateX(z < 0 ? 0.45 : -0.45));   // furrowed brows
  }
  head.add(at(box(0.1, 0.07, 0.3, INK), 0.21, 0.19, 0));                                  // moustache
  const hat = at(new THREE.Group(), 0, 0.4, 0);
  hat.add(at(cyl(0.2, 0.22, 0.34, color), 0, 0.17, 0));
  hat.add(at(cyl(0.225, 0.225, 0.07, GOLD), 0, 0.05, 0));
  hat.add(at(cyl(0.26, 0.26, 0.04, INK), 0.05, 0.01, 0));
  hat.add(at(ball(0.07, GOLD), 0, 0.38, 0));
  head.add(hat);
  body.add(head);
  // arms: a shoulder that swings, an elbow that straightens, a big glove
  const arm = (z: number): { arm: Arm; plate: THREE.Object3D } => {
    const shoulder = at(new THREE.Group(), 0, 0.72, z);
    shoulder.add(at(cyl(0.075, 0.075, 0.3, STEEL, 10), 0, -0.15, 0));
    const elbow = at(new THREE.Group(), 0, -0.3, 0);
    elbow.add(at(cyl(0.065, 0.065, 0.26, STEEL, 10), 0, -0.13, 0));
    elbow.add(at(ball(0.17, GLOVE), 0, -0.34, 0));
    elbow.add(at(cyl(0.12, 0.12, 0.08, 0xffffff, 14), 0, -0.2, 0));
    shoulder.add(elbow);
    body.add(shoulder);
    const plate = at(ball(0.16, GOLD), 0, 0.76, z + Math.sign(z) * 0.03);
    plate.scale.set(1, 0.7, 1);
    body.add(plate);
    return { arm: { shoulder, elbow }, plate };
  };
  const L = arm(-0.45), R = arm(0.45);
  return {
    root, body, head, key, arms: { L: L.arm, R: R.arm },
    parts: { hat, left_shoulder: L.plate, right_shoulder: R.plate, chest_door: chestDoor, key, head },
    gone: new Set(), lean: 0, guardAmt: 0, reach: { L: 0, R: 0 },
  };
}

interface Loose { obj: THREE.Object3D; v: THREE.Vector3; spin: THREE.Vector3; born: number; rest: boolean }
interface Spark { obj: THREE.Object3D; born: number; life: number; v: THREE.Vector3 }

export interface RobotScene {
  /** Draw the fight as the room's view has it at `now` (the room's clock). `colors`: fighter 0's and 1's. */
  frame(view: RobotView, now: number, colors: [string, string]): void;
  resize(): void;
  dispose(): void;
}

const ease = (cur: number, to: number, dt: number, rate: number) => cur + (to - cur) * Math.min(1, dt * rate);

export function createRobotScene(canvas: HTMLCanvasElement): RobotScene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 16 / 9, 0.1, 60);
  const CAM = new THREE.Vector3(0, 1.6, 5.9), LOOK = new THREE.Vector3(0, 1.08, 0);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb9c8dc, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 1.9);
  sun.position.set(2.5, 6, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -2, near: 1, far: 16 });
  scene.add(sun);

  // the ring: a round snowy stage with a gold rim, candy-cane posts, a rope
  const stage = cyl(3.1, 3.3, 0.36, SNOW, 48);
  stage.position.y = -0.18;
  stage.children.forEach((m) => { (m as THREE.Mesh).receiveShadow = true; (m as THREE.Mesh).castShadow = false; });
  scene.add(stage, at(cyl(3.32, 3.32, 0.12, GOLD, 48), 0, -0.36, 0));
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2 + Math.PI / 6, x = Math.cos(a) * 2.85, z = Math.sin(a) * 2.85;
    if (z > 1.2) continue;                                                              // none between the camera and the fight
    const post = new THREE.Group();
    for (let k = 0; k < 6; k++) post.add(at(cyl(0.07, 0.07, 0.17, k % 2 ? 0xffffff : GLOVE, 12), 0, 0.085 + k * 0.17, 0));
    post.add(at(ball(0.11, GOLD), 0, 1.1, 0));
    scene.add(at(post, x, 0, z));
  }
  const rope = piece(new THREE.TorusGeometry(2.85, 0.03, 8, 64, Math.PI * 1.34), 0xffffff, 0.012);
  rope.rotation.set(Math.PI / 2, 0, Math.PI * 0.83);
  scene.add(at(rope, 0, 0.8, 0));

  const robots: [Robot, Robot] = [buildRobot(0xff3366), buildRobot(0x18b88a)];
  const GAP = 0.8;                                                                      // apart at rest; a punch steps in (LUNGE) to reach the other's chin
  const LUNGE = 0.42;
  /** Each robot is turned a little to the camera, so the faces show (a pure side view hides them). */
  const TURN = 0.28;
  robots[0].root.position.x = -GAP;
  robots[1].root.position.x = GAP;
  robots[0].root.rotation.y = -TURN;
  robots[1].root.rotation.y = Math.PI + TURN;
  scene.add(robots[0].root, robots[1].root);
  let tinted: [string, string] = ['', ''];

  const loose: Loose[] = [];
  const sparks: Spark[] = [];
  const starGeo = new THREE.OctahedronGeometry(0.09);
  let seenEvent = -1, last = performance.now(), shake = 0, primed = false, lastRound = -1;

  /** Colour a robot's coat and hat (the first build uses placeholders). */
  function tint(r: Robot, hex: string) {
    const c = new THREE.Color(hex);
    r.root.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.MeshToonMaterial | undefined;
      if (m && (m as THREE.Material).type === 'MeshToonMaterial' && o.userData.coat) m.color.copy(c);
    });
  }
  // mark the coat-coloured pieces once (chest and hat body: the two placeholders above)
  for (const r of robots) {
    const want = r === robots[0] ? 0xff3366 : 0x18b88a;
    r.root.traverse((o) => { const m = (o as THREE.Mesh).material as THREE.MeshToonMaterial | undefined; if (m?.color && m.type === 'MeshToonMaterial' && m.color.getHex() === want) o.userData.coat = true; });
  }

  /** Knock a part off: the piece on the robot hides, a copy flies away from the punch. */
  function fling(r: Robot, part: Part, side: 0 | 1) {
    const src = r.parts[part];
    const copy = src.clone(true);
    src.updateWorldMatrix(true, false);
    src.matrixWorld.decompose(copy.position, copy.quaternion, copy.scale);
    src.visible = false;
    scene.add(copy);
    const away = side === 0 ? -1 : 1, big = part === 'head';
    loose.push({
      obj: copy, born: performance.now(), rest: false,
      v: new THREE.Vector3(away * (big ? 2.6 : 1.6 + Math.random()), big ? 6.2 : 3 + Math.random() * 1.5, (Math.random() - 0.3) * (big ? 1.2 : 2.4)),
      spin: new THREE.Vector3(Math.random() * 8 - 4, Math.random() * 8 - 4, (big ? 9 : 6) * -away),
    });
    if (big) shake = 0.5;
  }

  function burst(x: number, y: number, color: number, n: number, speed: number) {
    for (let i = 0; i < n; i++) {
      const s = new THREE.Mesh(starGeo, new THREE.MeshBasicMaterial({ color, transparent: true }));
      s.position.set(x, y, 0.35);
      const a = Math.random() * Math.PI * 2;
      scene.add(s);
      sparks.push({ obj: s, born: performance.now(), life: 420 + Math.random() * 260, v: new THREE.Vector3(Math.cos(a) * speed, Math.sin(a) * speed + 1, (Math.random() - 0.5) * speed) });
    }
  }

  /** How far a punching arm is thrown (0 at rest, negative drawn back, 1 at full reach). */
  function reachOf(f: RobotFighterView, hand: 'L' | 'R', now: number): number {
    const p = f.punch;
    if (!p || p.hand !== hand) return 0;
    const u = Math.max(0, Math.min(1, (now - p.at) / Math.max(1, p.lands - p.at)));
    return u < 0.55 ? -0.5 * Math.sin((u / 0.55) * Math.PI / 2) : -0.5 + 1.5 * ((u - 0.55) / 0.45) ** 2;
  }

  function pose(r: Robot, f: RobotFighterView, i: 0 | 1, now: number, dt: number, t: number, stage: RobotView['stage'], won: boolean) {
    const busy = f.busy && f.busyUntil > now ? f.busy : undefined;
    const down = f.health <= 0;
    // lean: back when hit, further when stunned, forward into a punch, slumped when the head is gone
    const wantLean = down ? -1.15 : busy === 'flinch' ? -0.38 : busy === 'stunned' ? -0.22 + Math.sin(t * 14) * 0.1 : busy === 'guard_break' ? 0.3 : f.punch ? 0.16 : f.guard ? -0.08 : 0;
    r.lean = ease(r.lean, wantLean, dt, busy === 'flinch' ? 26 : 12);
    r.body.rotation.z = -r.lean;
    r.body.position.y = 0.52 + (down ? -0.02 : Math.sin(t * 3.2 + i * 1.7) * 0.018);
    const step = Math.max(0, r.reach.L, r.reach.R) * LUNGE - Math.max(0, -r.lean) * 0.3;        // in with a punch, back when hit
    r.root.position.x = (i === 0 ? -GAP : GAP) + (i === 0 ? 1 : -1) * step;
    r.head.rotation.z = busy === 'stunned' ? Math.sin(t * 9) * 0.25 : 0;
    r.head.rotation.y = busy === 'stunned' ? Math.sin(t * 6) * 0.4 : 0;
    r.key.rotation.x = t * (down ? 0 : busy === 'stunned' ? 0.6 : 3);
    r.guardAmt = ease(r.guardAmt, f.guard ? 1 : 0, dt, 22);
    const cheer = won && (stage === 'ko' || stage === 'done') ? 1 : 0;
    for (const hand of ['L', 'R'] as const) {
      const a = r.arms[hand];
      const target = reachOf(f, hand, now);
      // snap out on the way in, ease back afterwards
      r.reach[hand] = target > r.reach[hand] ? ease(r.reach[hand], target, dt, 40) : ease(r.reach[hand], target, dt, f.punch ? 30 : 9);
      const reach = r.reach[hand];
      const drop = busy === 'guard_break' || down ? 1 : 0;
      // shoulder swings forward (towards +x is a rotation about z); the elbow straightens as the punch goes out
      const ready = 0.55;                                                               // fists up, the fighting stance
      let sh = ready + reach * 1.42, el = 1.55 - Math.max(0, reach) * 1.5 + Math.max(0, -reach) * 0.5;
      sh = sh * (1 - r.guardAmt) + 1.35 * r.guardAmt;
      el = el * (1 - r.guardAmt) + 2.25 * r.guardAmt;
      sh = sh * (1 - drop) + 0.05 * drop; el = el * (1 - drop) + 0.15 * drop;
      if (cheer) { sh = 2.9 + Math.sin(t * 8 + (hand === 'L' ? 0 : 1.6)) * 0.12; el = 0.3; }
      a.shoulder.rotation.z = sh;
      a.elbow.rotation.z = el;
      // gloves come together in front of the face for a guard; thrown wide open for a win (clear of the face)
      a.shoulder.rotation.x = (hand === 'L' ? 1 : -1) * (cheer ? -0.75 : 0.12 + r.guardAmt * 0.28);
    }
  }

  function frame(view: RobotView, now: number, colors: [string, string]) {
    const stamp = performance.now(), dt = Math.min(0.05, (stamp - last) / 1000), t = stamp / 1000;
    last = stamp;
    if (colors[0] !== tinted[0] || colors[1] !== tinted[1]) { tint(robots[0], colors[0]); tint(robots[1], colors[1]); tinted = [colors[0], colors[1]]; }
    // a new round: everything back on, the floor swept
    if (view.round !== lastRound && view.stage === 'intro') {
      for (const r of robots) { r.gone.clear(); for (const p of Object.values(r.parts)) p.visible = true; }
      for (const l of loose.splice(0)) scene.remove(l.obj);
    }
    lastRound = view.stage === 'intro' ? view.round : lastRound;
    ([0, 1] as const).forEach((i) => {
      const r = robots[i], f = view.fighters[i];
      for (const part of f.lost) {
        if (r.gone.has(part)) continue;
        r.gone.add(part);
        if (primed) fling(r, part, i); else r.parts[part].visible = false;               // already gone when this screen opened: just not there
      }
      const won = !!view.last && view.last.winner === view.order[i];
      pose(r, f, i, now, dt, t, view.stage, won);
    });
    // what just happened: stars for a hit, a ring of sparks off a block
    const newest = view.events.length ? view.events[view.events.length - 1].n : seenEvent;
    if (primed) for (const e of view.events) {
      if (e.n <= seenEvent) continue;
      const hitSide = e.by === 0 ? 1 : 0, x = (hitSide === 0 ? -1 : 1) * (GAP - 0.42);
      if (e.kind === 'hit') { burst(x, 1.75, 0xffe27a, 9, 3); shake = Math.max(shake, 0.12); }
      else if (e.kind === 'counter') { burst(x, 1.75, 0xff5a4a, 16, 4.2); shake = Math.max(shake, 0.25); }
      else if (e.kind === 'blocked') burst(x * 0.5, 1.6, 0xc9d2de, 6, 2);
      else if (e.kind === 'parried') { burst(x * 0.5, 1.6, 0x8a7bff, 14, 3.4); shake = Math.max(shake, 0.1); }
      else if (e.kind === 'guard_break') burst((e.by === 0 ? -1 : 1) * GAP * 0.8, 1.5, 0xffffff, 8, 2.4);
    }
    seenEvent = Math.max(seenEvent, newest);
    primed = true;

    for (let k = loose.length - 1; k >= 0; k--) {
      const l = loose[k];
      if (!l.rest) {
        l.v.y -= 13 * dt;
        l.obj.position.addScaledVector(l.v, dt);
        l.obj.rotation.x += l.spin.x * dt; l.obj.rotation.y += l.spin.y * dt; l.obj.rotation.z += l.spin.z * dt;
        const floor = Math.hypot(l.obj.position.x, l.obj.position.z) < 3.1 ? 0.14 : -2.5;
        if (l.obj.position.y < floor && l.v.y < 0) {
          l.obj.position.y = floor;
          if (Math.abs(l.v.y) < 1.4) l.rest = true;
          l.v.y *= -0.42; l.v.x *= 0.6; l.v.z *= 0.6; l.spin.multiplyScalar(0.5);
        }
      }
      if (stamp - l.born > 14000) { scene.remove(l.obj); loose.splice(k, 1); }
    }
    for (let k = sparks.length - 1; k >= 0; k--) {
      const s = sparks[k], age = (stamp - s.born) / s.life;
      if (age >= 1) { scene.remove(s.obj); sparks.splice(k, 1); continue; }
      s.v.y -= 6 * dt;
      s.obj.position.addScaledVector(s.v, dt);
      s.obj.rotation.x += dt * 9; s.obj.rotation.y += dt * 7;
      s.obj.scale.setScalar(1 - age * 0.6);
      ((s.obj as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = 1 - age * age;
    }
    shake = Math.max(0, shake - dt * 1.4);
    camera.position.set(CAM.x + Math.sin(t * 0.35) * 0.18 + (Math.random() - 0.5) * shake * 0.35, CAM.y + (Math.random() - 0.5) * shake * 0.35, CAM.z);
    camera.lookAt(LOOK);
    renderer.render(scene, camera);
  }

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // a tall screen (the console in a narrow window): step back so both robots stay in the picture
    camera.fov = camera.aspect < 1.5 ? 32 * Math.min(2.2, 1.5 / camera.aspect) : 32;
    camera.updateProjectionMatrix();
  }
  resize();

  return {
    frame, resize,
    dispose() {
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => x.dispose());
      });
      renderer.dispose();
    },
  };
}
