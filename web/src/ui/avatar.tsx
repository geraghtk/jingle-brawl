// Frost avatar tiles: the player's sticker (core/avatars.ts) on its colour.
import { avatarOf } from '../core/avatars';

// Gift tiles take a colour from their name (the palette the avatars had before stickers)
const COLORS = ['#18b88a', '#ff7a59', '#8a7bff', '#38b6e6', '#ffc53d', '#ff3d8b', '#c08457', '#ff3366'];

export function avatarColor(name: string) {
  let h = 0;
  for (const c of String(name || '')) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return COLORS[h % COLORS.length];
}

export function Avatar({ name, avatar, size = 30 }: { name: string; avatar?: string | null; size?: number }) {
  const a = avatarOf(name, avatar);
  return (
    <span className="f6-avatar" data-avatar={a.id} style={{ width: size, height: size, background: a.color }}>
      <img src={`art/avatars/${a.id}.webp`} alt="" draggable={false} style={{ width: '88%', height: '88%', objectFit: 'contain' }} />
    </span>
  );
}
