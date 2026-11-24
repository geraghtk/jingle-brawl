// The pieces the TV's main-game scenes are built from (host/TvStage.tsx, host/TvScenes.tsx): a name set
// as big as fits, an avatar and a gift box sized in the picture's own unit, the players' rail.
import type { CSSProperties, ReactNode } from 'react';
import type { Game, Gift, PlayerId } from '../core/types';
import { Avatar, avatarColor } from '../ui/avatar';
import { fitText, railShape } from '../ui/tvLayout';
import { useHost } from './HostContext';

/** n hundredths of the picture's height, as a CSS length. */
export const u = (n: number) => `calc(var(--tvu) * ${n})`;
export const vars = (o: Record<string, string | number>) => o as CSSProperties;

export const nameOf = (game: Game, id?: PlayerId | null) => (id && game.players?.[id]?.name) || 'Someone';
export const chipWord = (n: number) => `${n} chip${n === 1 ? '' : 's'}`;

/** Text set as big as fits its box: it shrinks, then wraps between words; it never breaks a word or leaves the box. */
export function Fit({ text, width, max, min, lines = 1, upper = false, className = '', testid, id }: {
  text: string; width: number; max: number; min: number; lines?: number; upper?: boolean; className?: string; testid?: string; id?: string;
}) {
  const shown = upper ? text.toUpperCase() : text;
  const f = fitText(shown, { width, max, min, maxLines: lines, upper });
  return (
    <div id={id} data-testid={testid} className={`tvs-fit${f.tight ? ' tight' : ''} ${className}`}
      style={{ fontSize: u(f.size), maxWidth: u(width), WebkitLineClamp: lines }}>{shown}</div>
  );
}

export function Av({ id, size, className = '' }: { id?: PlayerId | null; size: number; className?: string }) {
  const { game } = useHost();
  const p = id ? game.players?.[id] : undefined;
  return <span className={`tvs-av ${className}`} style={vars({ '--av': u(size) })}><Avatar name={p?.name || '?'} avatar={p?.avatar} /></span>;
}

export function GiftBox({ name, size, color }: { name?: string; size: number; color?: string }) {
  return (
    <div className="f6-gift tvs-gift" style={vars({ '--gift': u(size), '--gift-color': color ?? avatarColor(name || 'gift') })}>
      <span className="ribbon-v" /><span className="ribbon-h" /><span className="bow" />
    </div>
  );
}

export const Naughty = ({ level }: { level?: number }) => {
  const n = Math.max(0, Math.min(5, level ?? 0));
  return <span className="tvs-naughty" aria-label={`Naughty level ${n}`}>NAUGHTY <i>{'●'.repeat(n)}{'○'.repeat(5 - n)}</i></span>;
};

/** A line that says who the table is waiting for. */
export const Wait = ({ children }: { children: ReactNode }) => <div className="tvs-wait mo-up"><span className="tvs-wait-dot" />{children}</div>;

/** A few gifts as tiles (who holds each): the targets of a reprisal or a steal, the Misfit pile. */
export function GiftTiles({ gifts, max = 6 }: { gifts: [string, Gift][]; max?: number }) {
  const { game } = useHost();
  if (!gifts.length) return null;
  const shown = gifts.slice(0, gifts.length > max ? max - 1 : max);
  return (
    <div className="tvs-gifts">
      {shown.map(([id, g], i) => (
        <div key={id} className="tvs-gifttile mo-pop" style={vars({ '--i': i })}>
          <GiftBox name={g.name} size={6} />
          <div className="tvs-gifttile-txt">
            <Fit text={g.name} width={36} max={2.7} min={1.7} lines={2} className="tvs-gifttile-name" />
            <div className="tvs-gifttile-owner">{g.ownerId === 'misfit' ? 'Misfit pile' : nameOf(game, g.ownerId)}</div>
          </div>
        </div>
      ))}
      {gifts.length > shown.length && <div className="tvs-gifttile more">+{gifts.length - shown.length} more</div>}
    </div>
  );
}

/** Everyone at the table along the bottom: who holds what, their chips when chips are public, who the turn is about. */
export function Rail({ focus }: { focus: Set<PlayerId> }) {
  const { game, cfg } = useHost();
  const players = Object.entries(game.players || {});
  if (!players.length) return null;
  const shape = railShape(players.length);
  const av = shape.compact ? 5 : Math.min(8, shape.tileW * 0.32);
  const chips = !!cfg.publicChips;
  const textW = shape.tileW - av - (shape.compact ? 3.4 : 4.4);
  return (
    <footer className={`tvs-rail${shape.compact ? ' compact' : ''}`} id="players-grid" style={vars({ '--cols': shape.cols, '--tile': u(shape.tileW) })}>
      {players.map(([id, p]) => {
        const gift = p.giftId ? game.gifts?.[p.giftId]?.name || 'A gift' : null;
        return (
          <div key={id} className={`tvs-seat${focus.has(id) ? ' on' : ''}${gift ? ' has' : ''}`} data-player-id={id}>
            <Av id={id} size={av} />
            <div className="tvs-seat-txt">
              <Fit text={p.name} width={textW} max={shape.compact ? 2.2 : 2.9} min={1.3} lines={2} className="tvs-seat-name" />
              <div className="tvs-seat-sub">
                {chips && typeof p.chips === 'number' && <span className="tvs-seat-chips f6-num"><span className="f6-coin" />{p.chips}</span>}
                <span className="tvs-seat-gift">{gift ?? (chips ? 'In bag' : 'In the bag')}</span>
              </div>
            </div>
          </div>
        );
      })}
    </footer>
  );
}
