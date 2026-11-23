// The recap as a picture to keep or share (about 1080 x 1350, a portrait post). `layoutPicture` is
// the maths (pure: where every box goes and what its text is once it fits), `drawPicture` paints
// those boxes on a canvas, `sharePicture` hands the PNG to the phone's share sheet or downloads it.
// Only this page's own files are drawn (avatars and icons from art/), so the canvas is never tainted.
import { avatarOf } from '../core/avatars';
import { ordinal } from '../core/recap';
import type { Recap, RecapLine } from '../core/recap';

export const PIC_W = 1080;
export const PIC_H = 1350;
const M = 60;                                    // side margin
const INNER = PIC_W - 2 * M;

export type PicFont = 'disp' | 'body';
export type Measure = (text: string, size: number, weight: number, font: PicFont) => number;

interface Box { x: number; y: number; w: number; h: number }
export interface PicText extends Box { kind: 'text'; text: string; full: string; size: number; weight: number; font: PicFont; color: string; align: 'left' | 'center' | 'right' }
export interface PicRect extends Box { kind: 'rect'; r: number; fill: string; stroke?: string; shadow?: string }
export interface PicAvatar extends Box { kind: 'avatar'; avatar: string; color: string }
export interface PicIcon extends Box { kind: 'icon'; icon: string }
export type PicItem = PicText | PicRect | PicAvatar | PicIcon;

const GOLD = '#ffc53d', WHITE = '#ffffff', SOFT = 'rgba(255, 255, 255, 0.72)', INK = '#0d1018';
const STEP_FILL: Record<number, string> = { 1: GOLD, 2: '#dfe6ee', 3: '#f0a26b' };
const STEP_H: Record<number, number> = { 1: 150, 2: 112, 3: 96 };

/** A rough width for text when there is no canvas to ask (tests, and a canvas that will not measure). */
export const roughMeasure: Measure = (text, size, weight) => [...text].length * size * (weight >= 700 ? 0.62 : 0.56);

/** `text`, or as much of it as fits in `maxW` with an ellipsis. */
export function fitText(text: string, maxW: number, size: number, weight: number, font: PicFont, measure: Measure): string {
  if (measure(text, size, weight, font) <= maxW) return text;
  const chars = [...text];
  let lo = 0, hi = chars.length;
  while (lo < hi) {                               // the longest prefix that fits with the ellipsis
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(`${chars.slice(0, mid).join('').trimEnd()}…`, size, weight, font) <= maxW) lo = mid; else hi = mid - 1;
  }
  return `${chars.slice(0, lo).join('').trimEnd()}…`;
}

/** The names on one podium step: "Ann", "Ann & Bo", "Ann, Bo +2". */
export function stepNames(players: { name: string }[]): string {
  const n = players.map((p) => p.name);
  return n.length <= 2 ? n.join(' & ') : `${n[0]}, ${n[1]} +${n.length - 2}`;
}

export interface PictureText { brand: string; title: string; date: string }

/** Every box of the picture, back to front. Text boxes never overlap; `text` is what fits, `full` what was meant. */
export function layoutPicture(recap: Recap, words: PictureText, measure: Measure = roughMeasure): PicItem[] {
  const items: PicItem[] = [];
  const text = (b: Box, full: string, size: number, weight: number, font: PicFont, color: string, align: PicText['align'] = 'left') => {
    items.push({ kind: 'text', ...b, full, text: fitText(full, b.w, size, weight, font, measure), size, weight, font, color, align });
  };

  // what each section needs
  const steps = recap.podium;
  const podiumH = steps.length ? 330 : 0;
  const awards = recap.awards;
  const awardRows = Math.ceil(awards.length / 2);
  const AWARD_H = 74, AWARD_GAP = 12, LABEL_H = 34;
  const awardsH = awards.length ? LABEL_H + awardRows * AWARD_H + (awardRows - 1) * AWARD_GAP : 0;
  const cols = recap.lines.length > 6 ? 2 : 1;
  const listRows = Math.ceil(recap.lines.length / cols);
  const HEADER_H = 170, TOP = 44, BOTTOM = 46, ROW_GAP = 8;
  const sections = [podiumH, awardsH, listRows ? 1 : 0].filter(Boolean).length;
  const room = PIC_H - TOP - BOTTOM - HEADER_H - podiumH - awardsH - (listRows ? LABEL_H : 0);
  const MIN_GAP = 22;
  // the list takes what is left, up to a comfortable row; the rest opens the gaps and centres the page
  const rowH = listRows ? Math.max(40, Math.min(78, Math.floor((room - sections * MIN_GAP - (listRows - 1) * ROW_GAP) / listRows))) : 0;
  const listH = listRows ? LABEL_H + listRows * rowH + (listRows - 1) * ROW_GAP : 0;
  const slack = Math.max(0, room - sections * MIN_GAP - (listRows ? listRows * rowH + (listRows - 1) * ROW_GAP : 0));
  const gap = MIN_GAP + Math.min(34, Math.floor(slack / Math.max(1, sections)));
  let y = TOP + Math.floor(Math.max(0, slack - (gap - MIN_GAP) * sections) / 2);

  // header
  text({ x: M, y, w: INNER, h: 36 }, words.brand.toUpperCase(), 28, 700, 'disp', GOLD, 'center');
  text({ x: M, y: y + 42, w: INNER, h: 78 }, words.title, 64, 700, 'disp', WHITE, 'center');
  text({ x: M, y: y + 128, w: INNER, h: 34 }, words.date, 24, 600, 'body', SOFT, 'center');
  y += HEADER_H;

  // podium: second, first, third, on blocks of their own height
  if (steps.length) {
    y += gap;
    const order = [2, 1, 3].map((place) => steps.find((s) => s.place === place)).filter((s): s is NonNullable<typeof s> => !!s);
    const COL = 300, COL_GAP = 18;
    const x0 = (PIC_W - (order.length * COL + (order.length - 1) * COL_GAP)) / 2;
    const floor = y + podiumH;
    order.forEach((s, i) => {
      const x = x0 + i * (COL + COL_GAP), h = STEP_H[s.place], top = floor - h;
      items.push({ kind: 'rect', x, y: top, w: COL, h, r: 22, fill: STEP_FILL[s.place], shadow: 'rgba(0, 0, 0, 0.35)' });
      const ph = Math.min(54, h - 50);
      text({ x: x + 16, y: top + 10 + (h - 50 - ph) / 2, w: COL - 32, h: ph }, ordinal(s.place), s.place === 1 ? 50 : 40, 700, 'disp', INK, 'center');
      text({ x: x + 16, y: top + h - 38, w: COL - 32, h: 28 }, `${s.players[0].score} Party Score`, 21, 700, 'body', INK, 'center');
      text({ x, y: top - 46, w: COL, h: 38 }, stepNames(s.players), 30, 700, 'disp', WHITE, 'center');
      const shown = s.players.slice(0, 3);
      const size = s.place === 1 ? 108 : 90, step = shown.length > 1 ? Math.min(size + 8, (COL - size) / (shown.length - 1)) : 0;
      const ax = x + (COL - (size + step * (shown.length - 1))) / 2;
      shown.forEach((p, k) => {
        const a = avatarOf(p.name, p.avatar);
        items.push({ kind: 'avatar', x: ax + k * step, y: top - 54 - size, w: size, h: size, avatar: a.id, color: a.color });
      });
    });
    y = floor;
  }

  // awards: two to a row, an odd last one across both
  if (awards.length) {
    y += gap;
    text({ x: M, y, w: INNER, h: 26 }, 'AWARDS', 20, 800, 'body', GOLD);
    y += LABEL_H;
    const W2 = (INNER - 16) / 2;
    awards.forEach((a, i) => {
      const wide = awards.length % 2 === 1 && i === awards.length - 1;
      const b = { x: M + (i % 2) * (W2 + 16), y: y + Math.floor(i / 2) * (AWARD_H + AWARD_GAP), w: wide ? INNER : W2, h: AWARD_H };
      items.push({ kind: 'rect', ...b, r: 18, fill: 'rgba(255, 255, 255, 0.09)', stroke: 'rgba(255, 255, 255, 0.16)' });
      items.push({ kind: 'icon', x: b.x + 14, y: b.y + 12, w: 50, h: 50, icon: a.icon });
      text({ x: b.x + 76, y: b.y + 9, w: b.w - 92, h: 28 }, a.title, 23, 700, 'disp', GOLD);
      text({ x: b.x + 76, y: b.y + 39, w: b.w - 92, h: 26 }, `${a.name}: ${a.detail}`, 19, 600, 'body', WHITE);
    });
    y += awardsH - LABEL_H;
  }

  // everyone, and the gift they went home with
  if (listRows) {
    y += gap;
    text({ x: M, y, w: INNER, h: 26 }, 'WHO GOT WHAT', 20, 800, 'body', GOLD);
    y += LABEL_H;
    const colW = cols === 2 ? (INNER - 16) / 2 : INNER;
    const two = rowH >= 52;                       // room for the gift under the name; else beside it
    recap.lines.forEach((l: RecapLine, i) => {
      const c = Math.floor(i / listRows), r = i % listRows;
      const b = { x: M + c * (colW + 16), y: y + r * (rowH + ROW_GAP), w: colW, h: rowH };
      const top = l.place > 0 && l.place <= 3 && l.score > 0;
      items.push({ kind: 'rect', ...b, r: 16, fill: top ? 'rgba(255, 197, 61, 0.16)' : 'rgba(255, 255, 255, 0.07)' });
      const placeW = 62, av = Math.min(rowH - 12, 54), scoreW = 64;
      if (l.place > 0) text({ x: b.x + 10, y: b.y + (rowH - 30) / 2, w: placeW - 12, h: 30 }, ordinal(l.place), 22, 700, 'disp', top ? GOLD : SOFT, 'center');
      const a = avatarOf(l.name, l.avatar);
      items.push({ kind: 'avatar', x: b.x + placeW, y: b.y + (rowH - av) / 2, w: av, h: av, avatar: a.id, color: a.color });
      const tx = b.x + placeW + av + 14, tw = b.x + b.w - scoreW - 14 - tx;
      const gift = l.gift ? l.gift.name : 'No gift this year';
      if (two) {
        const lh = Math.min(30, (rowH - 8) / 2);
        text({ x: tx, y: b.y + rowH / 2 - lh, w: tw, h: lh }, l.name, 23, 700, 'disp', WHITE);
        text({ x: tx, y: b.y + rowH / 2, w: tw, h: lh }, gift, 18, 600, 'body', SOFT);
      } else {
        const nw = Math.floor(tw * 0.44);
        text({ x: tx, y: b.y + (rowH - 28) / 2, w: nw, h: 28 }, l.name, 21, 700, 'disp', WHITE);
        text({ x: tx + nw + 10, y: b.y + (rowH - 26) / 2, w: tw - nw - 10, h: 26 }, gift, 17, 600, 'body', SOFT);
      }
      text({ x: b.x + b.w - scoreW - 8, y: b.y + (rowH - 32) / 2, w: scoreW, h: 32 }, String(l.score), 26, 700, 'disp', top ? GOLD : WHITE, 'right');
    });
    y += listH - LABEL_H;
  }
  return items;
}

// --- painting ------------------------------------------------------------------------------------

const FAMILY: Record<PicFont, string> = { disp: '"Space Grotesk", system-ui, sans-serif', body: '"Manrope", system-ui, sans-serif' };
const fontOf = (size: number, weight: number, font: PicFont) => `${weight} ${size}px ${FAMILY[font]}`;

function rounded(ctx: CanvasRenderingContext2D, b: Box, r: number) {
  const k = Math.min(r, b.w / 2, b.h / 2);
  ctx.beginPath();
  ctx.moveTo(b.x + k, b.y);
  ctx.arcTo(b.x + b.w, b.y, b.x + b.w, b.y + b.h, k);
  ctx.arcTo(b.x + b.w, b.y + b.h, b.x, b.y + b.h, k);
  ctx.arcTo(b.x, b.y + b.h, b.x, b.y, k);
  ctx.arcTo(b.x, b.y, b.x + b.w, b.y, k);
  ctx.closePath();
}

/** The night sky behind everything: ink, a glow, a little paper in the air. Fixed, so two saves look the same. */
function backdrop(ctx: CanvasRenderingContext2D, keepClear: Box[]) {
  const g = ctx.createLinearGradient(0, 0, 0, PIC_H);
  g.addColorStop(0, '#1b1436'); g.addColorStop(0.55, '#10142a'); g.addColorStop(1, INK);
  ctx.fillStyle = g; ctx.fillRect(0, 0, PIC_W, PIC_H);
  const glow = ctx.createRadialGradient(PIC_W / 2, 330, 40, PIC_W / 2, 330, 640);
  glow.addColorStop(0, 'rgba(255, 51, 102, 0.34)'); glow.addColorStop(1, 'rgba(255, 51, 102, 0)');
  ctx.fillStyle = glow; ctx.fillRect(0, 0, PIC_W, PIC_H);
  const paper = ['#ff3366', '#ffc21a', '#18b88a', '#3b9cf5', '#ffffff', '#b565f0'];
  let s = 20261225;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = 0; i < 110; i++) {
    const x = rnd() * PIC_W, y = rnd() * rnd() * PIC_H * 0.8, w = 10 + rnd() * 12, edge = Math.abs(x - PIC_W / 2) / (PIC_W / 2);
    const spin = rnd() * Math.PI;
    if (keepClear.some((b) => x > b.x - 14 && x < b.x + b.w + 14 && y > b.y - 14 && y < b.y + b.h + 14)) continue;   // never behind words
    ctx.save();
    ctx.globalAlpha = 0.3 + 0.6 * edge;
    ctx.translate(x, y); ctx.rotate(spin);
    ctx.fillStyle = paper[i % paper.length];
    ctx.fillRect(-w / 2, -w * 0.22, w, w * 0.45);
    ctx.restore();
  }
}

export type PicImages = Record<string, CanvasImageSource | undefined>;
export const avatarSrc = (id: string) => `art/avatars/${id}.webp`;
export const iconSrc = (name: string) => `art/icons/${name}.webp`;

export function drawPicture(ctx: CanvasRenderingContext2D, items: PicItem[], images: PicImages = {}) {
  backdrop(ctx, items.filter((it) => it.kind !== 'avatar' && it.kind !== 'icon'));
  for (const it of items) {
    if (it.kind === 'rect') {
      if (it.shadow) { ctx.fillStyle = it.shadow; rounded(ctx, { ...it, y: it.y + 8 }, it.r); ctx.fill(); }
      ctx.fillStyle = it.fill; rounded(ctx, it, it.r); ctx.fill();
      if (it.stroke) { ctx.strokeStyle = it.stroke; ctx.lineWidth = 2; ctx.stroke(); }
    } else if (it.kind === 'avatar') {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'; rounded(ctx, { ...it, y: it.y + 4 }, it.w * 0.32); ctx.fill();
      ctx.fillStyle = it.color; rounded(ctx, it, it.w * 0.32); ctx.fill();
      ctx.strokeStyle = WHITE; ctx.lineWidth = Math.max(2, it.w * 0.04); ctx.stroke();
      const img = images[avatarSrc(it.avatar)];
      if (img) { const p = it.w * 0.06; ctx.drawImage(img, it.x + p, it.y + p, it.w - 2 * p, it.h - 2 * p); }
    } else if (it.kind === 'icon') {
      const img = images[iconSrc(it.icon)];
      if (img) ctx.drawImage(img, it.x, it.y, it.w, it.h);
      else { ctx.fillStyle = GOLD; ctx.beginPath(); ctx.arc(it.x + it.w / 2, it.y + it.h / 2, it.w * 0.3, 0, Math.PI * 2); ctx.fill(); }
    } else {
      ctx.font = fontOf(it.size, it.weight, it.font);
      ctx.fillStyle = it.color; ctx.textBaseline = 'middle'; ctx.textAlign = it.align;
      const x = it.align === 'center' ? it.x + it.w / 2 : it.align === 'right' ? it.x + it.w : it.x;
      ctx.fillText(it.text, x, it.y + it.h / 2 + it.size * 0.04, it.w);
    }
  }
}

/** One of this page's own pictures, or nothing if it will not load in time (the tile is drawn without it). */
function load(src: string, ms = 2500): Promise<HTMLImageElement | undefined> {
  return new Promise((done) => {
    const img = new Image();
    const t = setTimeout(() => done(undefined), ms);
    img.onload = () => { clearTimeout(t); done(img); };
    img.onerror = () => { clearTimeout(t); done(undefined); };
    img.src = src;                                // relative: the same origin as the page, so the canvas stays clean
  });
}

/** The recap as a PNG. */
export async function renderPicture(recap: Recap, words: PictureText): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = PIC_W; canvas.height = PIC_H;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('no canvas');
  try { await Promise.all([document.fonts.load(fontOf(40, 700, 'disp')), document.fonts.load(fontOf(20, 600, 'body')), document.fonts.load(fontOf(20, 800, 'body'))]); } catch { /* the system font will do */ }
  const measure: Measure = (t, size, weight, font) => { ctx.font = fontOf(size, weight, font); return ctx.measureText(t).width; };
  const items = layoutPicture(recap, words, measure);
  const srcs = [...new Set(items.flatMap((it) => (it.kind === 'avatar' ? [avatarSrc(it.avatar)] : it.kind === 'icon' ? [iconSrc(it.icon)] : [])))];
  const loaded = await Promise.all(srcs.map((s) => load(s)));
  const images: PicImages = Object.fromEntries(srcs.map((s, i) => [s, loaded[i]]));
  const blob = (withImages: boolean) => new Promise<Blob>((ok, no) => {
    drawPicture(ctx, items, withImages ? images : {});
    try { canvas.toBlob((b) => (b ? ok(b) : no(new Error('no picture'))), 'image/png'); } catch (e) { no(e); }
  });
  // a picture that will not come out with its stickers (a browser being strict) still comes out without them
  return blob(true).catch(() => blob(false));
}

export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.rel = 'noopener';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

/** Offer the picture to the device's share sheet; where there is none (or it refuses), download it. */
export async function sharePicture(blob: Blob, filename: string, title: string): Promise<'shared' | 'saved' | 'cancelled'> {
  try {
    const file = new File([blob], filename, { type: 'image/png' });
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({ files: [file], title });
      return 'shared';
    }
  } catch (e) {
    if ((e as { name?: string })?.name === 'AbortError') return 'cancelled';   // they closed the sheet: that is an answer
  }
  download(blob, filename);
  return 'saved';
}
