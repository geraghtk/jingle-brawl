// The avatars a player can pick when they join (players/{id}/avatar). Their pictures are stickers in
// web/public/art/avatars/<id>.webp (made with tools/art). Without a pick, one follows from the name.
export interface AvatarKind { id: string; label: string; color: string }

export const AVATARS: AvatarKind[] = [
  { id: 'elf', label: 'Elf', color: '#18b88a' },
  { id: 'reindeer', label: 'Reindeer', color: '#ff7a59' },
  { id: 'imp', label: 'Imp', color: '#8a7bff' },
  { id: 'snowman', label: 'Snowman', color: '#38b6e6' },
  { id: 'star', label: 'Star', color: '#ffc53d' },
  { id: 'penguin', label: 'Penguin', color: '#ff3d8b' },
  { id: 'cocoa', label: 'Cocoa', color: '#c08457' },
  { id: 'gift', label: 'Gift', color: '#ff3366' },
  { id: 'gingerbread', label: 'Gingerbread', color: '#e0913a' },
  { id: 'polarbear', label: 'Polar bear', color: '#5aa9e6' },
  { id: 'owl', label: 'Owl', color: '#b06ab3' },
  { id: 'fox', label: 'Fox', color: '#ff8a3d' },
  { id: 'nutcracker', label: 'Nutcracker', color: '#d7263d' },
  { id: 'cat', label: 'Cat', color: '#f2a541' },
  { id: 'dog', label: 'Puppy', color: '#e76f51' },
  { id: 'yeti', label: 'Yeti', color: '#7fb7be' },
  { id: 'gnome', label: 'Gnome', color: '#2a9d8f' },
  { id: 'llama', label: 'Llama', color: '#e9a6c4' },
  { id: 'bauble', label: 'Bauble', color: '#4cc9f0' },
  { id: 'sloth', label: 'Sloth', color: '#9c7a5b' },
  { id: 'hedgehog', label: 'Hedgehog', color: '#6d9f71' },
  { id: 'sprout', label: 'Sprout', color: '#7cb518' },
  { id: 'pudding', label: 'Pudding', color: '#a0522d' },
  { id: 'mrsclaus', label: 'Mrs Claus', color: '#e5383b' },
];

const BY_ID = new Map(AVATARS.map((a) => [a.id, a]));

export const isAvatar = (id: unknown): id is string => typeof id === 'string' && BY_ID.has(id);

const hash = (name: string) => {
  let h = 0;
  for (const c of String(name || '')) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return h;
};

/** A player's avatar: the one they picked, else one that follows from their name. */
export function avatarOf(name: string, picked?: string | null): AvatarKind {
  if (picked && BY_ID.has(picked)) return BY_ID.get(picked)!;
  return AVATARS[hash(name) % AVATARS.length];
}

/** The avatar for someone joining without a pick: the one their name gives, or the next one nobody has. */
export function freeAvatar(name: string, taken: Iterable<string | undefined>): string {
  const used = new Set(taken);
  const start = hash(name) % AVATARS.length;
  for (let i = 0; i < AVATARS.length; i++) {
    const a = AVATARS[(start + i) % AVATARS.length];
    if (!used.has(a.id)) return a.id;
  }
  return AVATARS[start].id;                       // a table of more than 24: someone shares
}
