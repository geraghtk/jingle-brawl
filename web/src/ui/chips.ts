// Jingle Chips: the casino-style chip (.jb-chip / .f6-coin) and flyChips(), which arcs chips
// from one element to another (loser → winner, pot → Bank …) and flashes the target on landing.
const css = `
  .jb-chip, .f6-coin {
    --size: 34px;
    display: inline-block; position: relative; flex-shrink: 0; vertical-align: -0.18em;
    width: var(--size); height: var(--size); border-radius: 50%;
    background:
      radial-gradient(circle at 34% 28%, rgba(255,255,255,0.7) 0 5%, transparent 24%),
      radial-gradient(circle, #ffd45e 0 45%, #b8770f 46% 49%, transparent 50%),
      repeating-conic-gradient(from 11.25deg, #d9334a 0 22.5deg, #fff6e8 22.5deg 45deg);
    box-shadow:
      inset 0 0 0 calc(var(--size) * 0.05) #8c1c2b,
      inset 0 calc(var(--size) * -0.06) calc(var(--size) * 0.1) rgba(0,0,0,0.22),
      0 calc(var(--size) * 0.08) calc(var(--size) * 0.22) rgba(30,15,0,0.3);
  }
  .jb-chip::after, .f6-coin::after {
    content: '★'; position: absolute; inset: 0; display: grid; place-items: center;
    font: 700 calc(var(--size) * 0.4)/1 system-ui, sans-serif; color: #a5670b;
    text-shadow: 0 1px 0 rgba(255,236,170,0.9), 0 -1px 0 rgba(110,55,0,0.35);
  }
  .f6-coin { --size: 1.05em; }
  .jb-chip-fly { position: fixed; left: 0; top: 0; z-index: 2200; pointer-events: none; opacity: 0; will-change: transform; }
  .jb-chip-land { animation: jbChipLand 0.3s ease-out; }
  @keyframes jbChipLand { 35% { filter: brightness(1.35) saturate(1.2); } }
`;
if (typeof document !== 'undefined' && !document.getElementById('jb-chip-style')) {
  const style = document.createElement('style');
  style.id = 'jb-chip-style';
  style.textContent = css;
  document.head.appendChild(style);
}

type Target = Element | { x: number; y: number } | null | undefined;

function centerOf(target: Target) {
  if (!target) return null;
  if (!(target instanceof Element)) return target;
  const r = target.getBoundingClientRect();
  if (!r.width && !r.height) return null;
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

function flash(target: Target) {
  if (!(target instanceof Element)) return;
  target.classList.remove('jb-chip-land');
  void (target as HTMLElement).offsetWidth; // restart the animation
  target.classList.add('jb-chip-land');
}

export interface FlyOptions {
  from: Target; to: Target; count: number;
  size?: number; stagger?: number; duration?: number;
  onLand?: (n: number) => void; onDone?: () => void;
}

/** Fly `count` chips (max 12 drawn) along an arc; skipped (onDone at once) if either end is off-screen. */
export function flyChips({ from, to, count, size = 34, stagger = 90, duration = 950, onLand, onDone }: FlyOptions) {
  // Announce the flight (tests / the agent harness listen for it)
  const label = (t: Target) => (t instanceof HTMLElement ? t.dataset.duelCorner || t.dataset.playerId || t.id || null : null);
  const a = centerOf(from);
  const b = centerOf(to);
  const n = Math.max(0, Math.min(count | 0, 12));
  // drawn = chips actually animated (0 when an end is off-screen)
  document.dispatchEvent(new CustomEvent('jb:chips', { detail: { from: label(from), to: label(to), count, drawn: !a || !b ? 0 : n } }));
  if (!a || !b || !n) { if (onDone) setTimeout(onDone, 0); return; }
  const dx = b.x - a.x;
  const lift = Math.min(240, 50 + Math.hypot(dx, b.y - a.y) * 0.35);
  let landed = 0;
  for (let i = 0; i < n; i++) {
    const chip = document.createElement('div');
    chip.className = 'jb-chip jb-chip-fly';
    chip.style.setProperty('--size', `${size}px`);
    document.body.appendChild(chip);
    const jitter = (Math.random() - 0.5) * size;
    const cx = a.x + dx / 2 + jitter;
    const cy = Math.min(a.y, b.y) - lift + jitter / 2;
    const frames: Keyframe[] = [];
    for (let s = 0; s <= 18; s++) {
      const t = s / 18;
      const u = 1 - t;
      const x = u * u * a.x + 2 * u * t * cx + t * t * b.x;
      const y = u * u * a.y + 2 * u * t * cy + t * t * b.y;
      const flip = Math.max(0.12, Math.abs(Math.cos(t * Math.PI * 3)));
      const grow = 1 + 0.3 * Math.sin(t * Math.PI);
      frames.push({
        offset: t, opacity: t < 0.04 ? t / 0.04 : 1,
        transform: `translate(${(x - size / 2).toFixed(1)}px, ${(y - size / 2).toFixed(1)}px) scale(${(grow * flip).toFixed(3)}, ${grow.toFixed(3)})`,
      });
    }
    const anim = chip.animate(frames, { duration, delay: i * stagger, easing: 'cubic-bezier(0.4, 0, 0.6, 1)', fill: 'forwards' });
    anim.onfinish = () => {
      chip.remove();
      landed++;
      flash(to);
      onLand?.(landed);
      if (landed === n) onDone?.();
    };
  }
}
