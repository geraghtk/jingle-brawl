// A party game's intro while it waits for the table: who is ready, and (on a phone in the game) the
// Ready button. Shared by the TV's intro screens and the phone's (the look is the Briefing's: arena.css).
import type { PartyGame } from '../core/party';
import type { Game } from '../core/types';
import { Avatar } from './avatar';
import './arena.css';

export function PartyReady({ mg, game, me, onReady, tv }: { mg: PartyGame; game: Game; me?: string; onReady?: () => void; tv?: boolean }) {
  const people = (mg.participants || []).map((id) => ({ id, p: game.players?.[id] })).filter((x) => !!x.p);
  const isReady = (id: string) => !!mg.ready?.[id] || !!game.players?.[id]?.isBot;
  const waiting = people.filter((x) => !isReady(x.id) && x.p!.connected !== false);
  const mine = !!me && (mg.participants || []).includes(me);
  return (
    <div className={`brief party-ready${tv ? ' brief-tv-inline' : ''}`} data-testid="party-ready">
      <ul className="brief-roster" aria-label="Who is ready">
        {people.map(({ id, p }) => (
          <li key={id} className={isReady(id) ? 'ready' : p!.connected === false ? 'away' : ''} data-testid={`party-ready-${id}`}>
            <Avatar name={p!.name || id} avatar={p!.avatar} size={44} />
            <span className="brief-name">{p!.name || 'Someone'}</span>
            <span className="brief-mark" aria-label={isReady(id) ? 'ready' : 'not ready yet'}>{isReady(id) ? '✓' : '…'}</span>
          </li>
        ))}
      </ul>
      {mine && !isReady(me!) && onReady
        ? <button type="button" className="brief-ready" data-testid="party-ready-btn" onClick={onReady}>I am ready!</button>
        : <p className="brief-wait" role="status">{waiting.length ? `${me ? 'Ready! ' : ''}Waiting for ${waiting.map((x) => x.p!.name || 'someone').join(', ')}${me ? '' : ': press Ready on your phone'}` : 'Everyone is ready…'}</p>}
    </div>
  );
}
