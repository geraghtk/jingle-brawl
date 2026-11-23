// The house look in 3D (three.js): simple shapes with flat toon colour and a dark ink outline, to sit
// beside the game's sticker art. Shared by the games that draw in 3D (Nutcracker Knockout's ring,
// Mecha Santa's rooftops); only ever loaded with one of their scenes, never by a phone.
import * as THREE from 'three';

export const INK = 0x1d2a3a;
export const GOLD = 0xffc21a;
export const STEEL = 0xc9d2de;
export const GLOVE = 0xe63946;
export const SNOW = 0xf4f8fc;

/** A shape in the house look: flat toon colour and an ink outline (the same shape, a little bigger, seen from inside). */
export function piece(geo: THREE.BufferGeometry, color: number, outline = 0.035): THREE.Group {
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
export const box = (w: number, h: number, d: number, c: number) => piece(new THREE.BoxGeometry(w, h, d, 1, 1, 1), c);
export const cyl = (rt: number, rb: number, h: number, c: number, seg = 20) => piece(new THREE.CylinderGeometry(rt, rb, h, seg), c);
export const ball = (r: number, c: number) => piece(new THREE.SphereGeometry(r, 20, 14), c);
export const cone = (r: number, h: number, c: number, seg = 18) => piece(new THREE.ConeGeometry(r, h, seg), c);
/** A turned-wood barrel: `r` at its middle, narrower (by `taper`) at its flat ends. Reads as round from any side. */
export function barrel(r: number, h: number, c: number, taper = 0.2, seg = 24): THREE.Group {
  const pts = [new THREE.Vector2(0, -h / 2)];
  for (let i = 0; i <= 12; i++) { const t = i / 12; pts.push(new THREE.Vector2(r * (1 - taper + taper * Math.sin(Math.PI * t) ** 0.7), (t - 0.5) * h)); }
  pts.push(new THREE.Vector2(0, h / 2));
  return piece(new THREE.LatheGeometry(pts, seg), c);
}
export const at = <T extends THREE.Object3D>(o: T, x: number, y: number, z: number): T => { o.position.set(x, y, z); return o; };
export const ease = (cur: number, to: number, dt: number, rate: number) => cur + (to - cur) * Math.min(1, dt * rate);

/** Free everything a scene made (geometries, materials). */
export function disposeScene(scene: THREE.Scene, renderer: THREE.WebGLRenderer) {
  scene.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose?.();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => x.dispose());
  });
  renderer.dispose();
}
