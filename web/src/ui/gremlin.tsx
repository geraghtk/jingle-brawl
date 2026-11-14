// Pass the Gremlin: what its phone card and its TV stage both draw.
import { useState } from 'react';
import type { Danger } from '../core/minigames/gremlin/gremlin';

/** The Gremlin itself: its sticker, or the emoji while the art isn't there. */
export function GremlinFace({ danger }: { danger: Danger }) {
  const [art, setArt] = useState(true);
  return art
    ? <img className="grem-art" src={`art/gremlin/${danger}.webp`} alt="👹" draggable={false} onError={() => setArt(false)} />
    : <span>👹</span>;
}
