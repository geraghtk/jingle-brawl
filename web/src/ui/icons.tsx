// The game's icons: sticker pictures (web/public/art/icons/<name>.webp, made with tools/art) in place of
// emoji. <Icon name> draws one; <Glyphs> draws text (a string or a node) with the emoji it knows as
// icons. Each picture's alt text is its emoji, so button names, screen readers and tests read the same.
import { Children } from 'react';
import type { ReactNode } from 'react';

export const ICONS = {
  '⚔️': 'duel', '⚔': 'duel', '🎰': 'bid', '🎯': 'target', '🏃': 'run', '😈': 'naughty', '⏳': 'wait', '🤖': 'critic',
  '✅': 'check', '🎁': 'gift', '🏆': 'trophy', '🦌': 'reprisal', '🎅': 'santa', '✏️': 'pencil', '🏅': 'medal',
  '🎡': 'wheel', '🗳️': 'vote', '🎄': 'tree', '🎨': 'palette', '🎲': 'dice', '⚖️': 'scales', '🔨': 'mallet',
  '👁️': 'eye', '🎉': 'party', '🎭': 'masks', '👑': 'crown', '🪙': 'chip', '💰': 'moneybag', '💡': 'idea',
  '🧸': 'teddy', '🔒': 'lock', '✉️': 'envelope', '✉': 'envelope', '🛡️': 'shield', '⚡': 'zap', '😢': 'sad',
  '🔥': 'fire', '🎪': 'tent', '🔄': 'swap',
} as const;

// Grinch's Gambit has its own picture (a hand taking a gift); 😈 in text is the naughty imp
const EXTRA = { gambit: '😈' } as const;

export type IconName = (typeof ICONS)[keyof typeof ICONS] | keyof typeof EXTRA;

// alt text: an icon's first emoji (the full form, e.g. ⚔️ rather than the bare ⚔)
const EMOJI_OF = { ...Object.fromEntries(Object.entries(ICONS).reverse().map(([e, n]) => [n, e])), ...EXTRA } as Record<IconName, string>;
const KNOWN = new RegExp(`(${Object.keys(ICONS).sort((a, b) => b.length - a.length).join('|')})`, 'u');

export function Icon({ name, className }: { name: IconName; className?: string }) {
  return <img className={`yd-icon${className ? ` ${className}` : ''}`} src={`art/icons/${name}.webp`} alt={EMOJI_OF[name]} draggable={false} />;
}

/** Text with the emoji it knows drawn as icons (anything else passes through untouched). */
export function glyphs(text: string): ReactNode {
  const parts = text.split(KNOWN);
  if (parts.length === 1) return text;
  return parts.map((p, i) => (i % 2 ? <Icon key={i} name={ICONS[p as keyof typeof ICONS]} /> : p));
}

export function Glyphs({ children }: { children?: ReactNode }) {
  return <>{Children.map(children, (c) => (typeof c === 'string' ? glyphs(c) : c))}</>;
}
