// Elf Tower Wars: the pictures the TV and the phones share, and the names and colours of things. The
// pictures are the game's flat cartoon art (web/public/art/towers, made by tools/art/towers.py): every
// piece a team owns exists twice, in team A's red and team B's green.
import type { CSSProperties } from 'react';

/** Team A's and team B's colours (the same two as every team game). */
export const TOWER_COLORS: [string, string] = ['#ff3366', '#18b88a'];
export const UNIT_NAMES: Record<string, string> = { soldier: 'Toy soldier', mouse: 'Wind-up mice', tank: 'Teddy tank' };
export const UNIT_HINTS: Record<string, string> = { soldier: 'quick, light', tank: 'slow, tough, hits hard' };
export const TOWER_NAMES: Record<string, string> = { snow: 'Snowball tower', cannon: 'Candy cannon', frost: 'Frost lamp' };
export const TOWER_HINTS: Record<string, string> = { snow: 'steady', cannon: 'hits hard', frost: 'slows them' };
/** The phone's cards: short names and what each is for. */
export const CARD_NAMES: Record<string, string> = { soldier: 'Soldier', mouse: 'Mice', tank: 'Teddy', zap: 'Snowball', shield: 'Shield', combo: 'Avalanche', snow: 'Snow', cannon: 'Cannon', frost: 'Frost', patch: 'Glue' };
export const CARD_HINTS: Record<string, string> = { soldier: 'steady', mouse: 'a pack', tank: 'tough', zap: 'hit a toy', shield: 'block', combo: 'takes two', snow: 'one toy', cannon: 'crowds', frost: 'slows', patch: 'mend' };
/** What each card is for and what answers it (the phone's hint line when it is picked). */
export const CARD_TIPS: Record<string, string> = {
  soldier: 'Toy soldier: cheap and steady. A snow tower stops it.',
  mouse: 'Three wind-up mice: they swamp a snow tower. A cannon stops them all.',
  tank: 'Teddy tank: slow and tough. It shrugs off a cannon; frost and snow towers wear it down.',
  zap: 'Snowball: hits the nearest toy coming at you.',
  shield: 'Shield: nothing gets into your workshop down that path for a moment.',
  combo: 'Avalanche: two of you must tap the same path within a moment. Call it out!',
  snow: 'Snow tower: hits one toy hard. Good against soldiers and tanks.',
  cannon: 'Candy cannon: hits a whole crowd. Good against mice.',
  frost: 'Frost lamp: slows everything near it, so your towers get longer at them.',
  patch: 'Glue: mends your workshop a little.',
};

/** Every picture the game draws (a test checks each file is there). */
export const TOWER_ART = [
  ...['soldier', 'tank', 'mouse', 'snow', 'cannon', 'frost', 'workshop'].flatMap((n) => [`${n}-a`, `${n}-b`]),
  // a tower built again on its spot has a picture of its own for each level (Kevin, 2026-11-23: an upgrade should show)
  ...['snow', 'cannon', 'frost'].flatMap((n) => [2, 3].flatMap((level) => [`${n}${level}-a`, `${n}${level}-b`])),
  'pine', 'snowman', 'gifts', 'cane', 'field',
];
const KNOWN = new Set(TOWER_ART);
/** A piece's picture in a team's colour (an unknown kind falls back to the nearest thing we have). */
export const artSrc = (name: string, color?: string) => {
  const file = color ? `${name}-${color === TOWER_COLORS[1] ? 'b' : 'a'}` : name;
  return `art/towers/${KNOWN.has(file) ? file : color ? `soldier-${color === TOWER_COLORS[1] ? 'b' : 'a'}` : 'pine'}.webp`;
};
const fit: CSSProperties = { objectFit: 'contain', objectPosition: 'center bottom', display: 'block' };

/** A toy on the march, in its team's colour, facing the way it walks (the art walks to the right). */
export function UnitArt({ kind, color, flip, style }: { kind: string; color: string; flip?: boolean; style?: CSSProperties }) {
  return <img src={artSrc(kind, color)} alt="" draggable={false} style={{ ...fit, ...style, transform: `${style?.transform || ''}${flip ? ' scaleX(-1)' : ''}` }} />;
}

/** The picture of a tower at a level (level 1, or a level with no picture of its own: the plain one). */
export const towerPiece = (kind: string, level = 1) => (level > 1 && KNOWN.has(`${kind}${Math.min(3, level)}-a`) ? `${kind}${Math.min(3, level)}` : kind);

/** A tower on its spot (it aims to the right: flipped for the team on the right), as grand as its level. */
export function TowerArt({ kind, color, flip, style, level }: { kind: string; color: string; flip?: boolean; style?: CSSProperties; level?: number }) {
  return <img src={artSrc(towerPiece(kind, level), color)} alt="" draggable={false} style={{ ...fit, ...style, transform: flip ? 'scaleX(-1)' : undefined }} />;
}

/** A team's workshop: `health` 0 to 1 (it leans, darkens and smokes as it goes: the smoke is the stage's). */
export function WorkshopArt({ color, health, flip, style }: { color: string; health: number; flip?: boolean; style?: CSSProperties }) {
  const hurt = 1 - Math.max(0, Math.min(1, health));
  return (
    <img src={artSrc('workshop', color)} alt="" draggable={false}
      style={{ ...fit, ...style, transformOrigin: '50% 100%', transform: `${flip ? 'scaleX(-1) ' : ''}rotate(${hurt * -4}deg)`, filter: `brightness(${1 - hurt * 0.3}) saturate(${1 - hurt * 0.45})` }} />
  );
}
