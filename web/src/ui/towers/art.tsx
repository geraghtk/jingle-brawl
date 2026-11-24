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

/** A lane's gate at a workshop: where that team's toys come out onto a path that does not start at the
 *  workshop's own door (the top and the bottom one: toys used to appear out of the snow there). Drawn, not
 *  painted, in the art's flat look: stone posts, a doorway, a roof in the team's colour under snow. */
export function LaneGate({ color, style }: { color: string; style?: CSSProperties }) {
  const ink = '#1d2a3a';
  return (
    <svg viewBox="0 0 100 112" style={{ display: 'block', overflow: 'visible', ...style }} aria-hidden>
      <ellipse cx="50" cy="104" rx="50" ry="8" fill="rgba(40, 70, 110, .18)" />
      {/* the doorway, dark inside, and its two doors standing open */}
      <path d="M24,104 L24,58 Q50,30 76,58 L76,104 Z" fill="#2a2140" stroke={ink} strokeWidth="4" strokeLinejoin="round" />
      <path d="M24,104 L24,60 L33,66 L33,100 Z M76,104 L76,60 L67,66 L67,100 Z" fill="#a8683a" stroke={ink} strokeWidth="3" strokeLinejoin="round" />
      {/* the posts: stone, a course or two marked */}
      {[8, 74].map((x) => (
        <g key={x}>
          <rect x={x} y="46" width="18" height="60" rx="3" fill="#cfd4de" stroke={ink} strokeWidth="4" />
          <path d={`M${x},66 h18 M${x},86 h18 M${x + 9},46 v20 M${x + 9},86 v20`} stroke="#9aa3b5" strokeWidth="2.5" fill="none" />
        </g>
      ))}
      {/* the roof, in the team's colour, snow along its ridge */}
      <path d="M-2,50 L50,10 L102,50 L92,56 L50,26 L8,56 Z" fill={color} stroke={ink} strokeWidth="4" strokeLinejoin="round" />
      <path d="M14,36 Q22,26 32,24 Q40,10 50,8 Q60,10 68,24 Q78,26 86,36 Q78,34 70,30 Q60,20 50,18 Q40,20 30,30 Q22,34 14,36 Z" fill="#fff" stroke={ink} strokeWidth="3" strokeLinejoin="round" />
      {/* a pennant on top */}
      <path d="M50,8 L50,-10" stroke={ink} strokeWidth="3.5" strokeLinecap="round" />
      <path d="M50,-10 L68,-5 L50,1 Z" fill={color} stroke={ink} strokeWidth="3" strokeLinejoin="round" />
    </svg>
  );
}

/** Does this lane start at the workshop's own door (the middle one), or does it need a gate of its own? */
export const laneHasGate = (lane: number, lanes: number) => lanes > 1 && lane !== (lanes - 1) / 2;
