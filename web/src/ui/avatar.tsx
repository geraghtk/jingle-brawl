// Frost avatar tiles: emoji + colour picked deterministically from the player's name.
const KINDS: [string, string][] = [
  ['🧝', '#18b88a'], ['🦌', '#ff7a59'], ['😈', '#8a7bff'], ['☃️', '#38b6e6'],
  ['⭐', '#ffc53d'], ['🐧', '#ff3d8b'], ['☕', '#c08457'], ['🎁', '#ff3366'],
];

export function avatarKind(name: string): [string, string] {
  let h = 0;
  for (const c of String(name || '')) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return KINDS[h % KINDS.length];
}

export const avatarColor = (name: string) => avatarKind(name)[1];

export function Avatar({ name, size = 30 }: { name: string; size?: number }) {
  const [emoji, bg] = avatarKind(name);
  return (
    <span className="f6-avatar" style={{ width: size, height: size, background: bg, fontSize: Math.round(size * 0.5) }}>
      {emoji}
    </span>
  );
}
