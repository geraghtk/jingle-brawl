// Whack-an-Elf art (web/public/art/whack, rendered in Blender by tools/art/whack.py): a snowy hole in
// two layers so a pop-up sinks behind the front lip, the elf / golden elf / Grinch and their bonked
// faces, and a candy-cane mallet that swings on a hit. Every picture shares the hole's camera.
import type { WhackKind } from '../core/party';

export type HoleState = 'down' | 'up' | 'hit';
const KINDS: WhackKind[] = ['elf', 'gold', 'grinch'];
const ALT: Record<WhackKind, string> = { elf: 'elf', gold: 'golden elf', grinch: 'Grinch' };
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
  const rows: [WhackKind, string][] = [['elf', '+1 to +3'], ['gold', 'Double!'], ['grinch', '-3']];
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
