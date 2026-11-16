// Whack-an-Elf art (web/public/art/whack): a snowy hole in two layers so a pop-up sinks behind the front
// lip (rendered in Blender by tools/art/whack.py), a candy-cane mallet that swings on a hit, and the
// pop-ups in the game's flat 2D look (tools/art/whack2d.py + whack2d_place.py: Flux, then placed on the
// hole's canvas): the elf, the golden elf (the same elf turned to gold) and the Grump, each with a
// bonked twin (the same picture knocked askew, stars round its head). Every picture shares the hole's
// 384 px canvas.
import type { WhackKind } from '../core/party';

export type HoleState = 'down' | 'up' | 'hit';
const KINDS: WhackKind[] = ['elf', 'gold', 'grump'];
const ALT: Record<WhackKind, string> = { elf: 'elf', gold: 'golden elf', grump: 'Grump' };
export const WHACK_ART = ['hole-back', 'hole-front', 'mallet', ...KINDS, ...KINDS.map((k) => `${k}-bonk`)];
export const whackArt = (name: string) => `art/whack/${name}.webp`;

/** Fetch the pictures before the first pop-up so nothing flickers in. */
export function preloadWhackArt() {
  if (typeof Image === 'undefined') return;
  for (const n of WHACK_ART) new Image().src = whackArt(n);
}

/** One hole. `kind` stays set while a pop-up slides back down, so it sinks instead of vanishing. */
export function WhackHole({ kind, state }: { kind?: WhackKind; state: HoleState }) {
  return (
    <>
      <img className="wh-back" src={whackArt('hole-back')} alt="" draggable={false} />
      <div className="wh-pit">
        {kind && <img className="wh-pop" src={whackArt(state === 'hit' ? `${kind}-bonk` : kind)} alt={ALT[kind]} draggable={false} />}
      </div>
      <img className="wh-front" src={whackArt('hole-front')} alt="" draggable={false} />
      {state === 'hit' && <img className="wh-mallet" src={whackArt('mallet')} alt="" draggable={false} />}
    </>
  );
}

/** What each pop-up is worth, for the intro screens. */
export function WhackLegend() {
  const rows: [WhackKind, string][] = [['elf', '+1 to +3'], ['gold', 'Double!'], ['grump', '-3']];
  return (
    <div className="wh-legend">
      {rows.map(([k, v]) => (
        <figure key={k}>
          <div className={`wh-cell up ${k}`}><WhackHole kind={k} state="up" /></div>
          <figcaption>{v}</figcaption>
        </figure>
      ))}
    </div>
  );
}
