// A scribble as a PNG data URL, made without a canvas (the room server has none): what a practice bot
// hands in for Sketch-Off (core/bots.ts). Greyscale, stored uncompressed: a few kilobytes.
import type { Rng } from './rng';

const W = 120, H = 90;

const CRC = (() => {
  const t: number[] = [];
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t.push(c >>> 0); }
  return t;
})();
function crc(bytes: number[]): number {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
const u32 = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const chunk = (type: string, data: number[]) => { const body = [...type].map((ch) => ch.charCodeAt(0)).concat(data); return [...u32(data.length), ...body, ...u32(crc(body))]; };

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function base64(bytes: number[]): string {
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const a = bytes[i], b = bytes[i + 1], c = bytes[i + 2];
    out += B64[a >> 2] + B64[((a & 3) << 4) | ((b ?? 0) >> 4)] + (b == null ? '=' : B64[((b & 15) << 2) | ((c ?? 0) >> 6)]) + (c == null ? '=' : B64[c & 63]);
  }
  return out;
}

/** A PNG (greyscale, 8 bits) of `pixels` (rows of W bytes). */
function png(pixels: Uint8Array): string {
  const raw: number[] = [];
  for (let y = 0; y < H; y++) { raw.push(0); for (let x = 0; x < W; x++) raw.push(pixels[y * W + x]); }
  let a = 1, b = 0;
  for (const v of raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
  const z = [0x78, 0x01];
  for (let i = 0; i < raw.length; i += 65535) {
    const part = raw.slice(i, i + 65535), last = i + 65535 >= raw.length;
    z.push(last ? 1 : 0, part.length & 255, part.length >> 8, ~part.length & 255, (~part.length >> 8) & 255, ...part);
  }
  z.push(...u32(((b << 16) | a) >>> 0));
  const bytes = [137, 80, 78, 71, 13, 10, 26, 10, ...chunk('IHDR', [...u32(W), ...u32(H), 8, 0, 0, 0, 0]), ...chunk('IDAT', z), ...chunk('IEND', [])];
  return `data:image/png;base64,${base64(bytes)}`;
}

/** A few wobbly loops and lines on white. */
export function doodle(rng: Rng): string {
  const px = new Uint8Array(W * H).fill(255);
  const dot = (x: number, y: number) => {
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = Math.round(x) + dx, Y = Math.round(y) + dy;
      if (X >= 0 && X < W && Y >= 0 && Y < H) px[Y * W + X] = 30;
    }
  };
  const strokes = 3 + Math.floor(rng() * 3);
  for (let s = 0; s < strokes; s++) {
    const cx = 20 + rng() * (W - 40), cy = 18 + rng() * (H - 36), r = 8 + rng() * 22, squash = 0.5 + rng() * 0.8;
    const turns = 0.6 + rng() * 1.2, start = rng() * Math.PI * 2, drift = (rng() - 0.5) * 30;
    for (let i = 0; i <= 160; i++) {
      const f = i / 160, ang = start + f * turns * Math.PI * 2;
      dot(cx + Math.cos(ang) * r + drift * f, cy + Math.sin(ang) * r * squash);
    }
  }
  return png(px);
}
