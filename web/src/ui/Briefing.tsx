// Before a mini-game starts: how to play it, who is ready, and (once everyone is) the 3, 2, 1. The same
// panel on the TV, the host console and every phone; a phone in the game also has the Ready button.
// The room decides when the game starts (core/minigames/runtime.ts: briefing); this only shows it.
import { useEffect, useRef } from 'react';
import type { ArenaView } from '../core/minigames/runtime';
import type { Game } from '../core/types';
import { Avatar } from './avatar';
import { Glyphs } from './icons';
import { play } from './sound';
import './arena.css';

/** How to play each game, in a few short lines (a game not listed here shows its one-paragraph rules). */
export const HOW_TO: Record<string, string[]> = {
  pass_the_gremlin: [
    'The Gremlin is passed round the table, phone to phone.',
    'Holding it? Do the task on your phone, then pass it on.',
    'It goes berserk at a secret moment: do not be the one holding it!',
  ],
  reindeer_ring_toss: [
    'Flick up on your phone to throw a ring at the reindeer on the TV.',
    'Each antler tip and the hat tip scores 1. Ring all five for a full set: 5 more.',
    'A few practice throws first, then throw as many as you can before time runs out.',
  ],
  gift_conveyor: [
    'Two teams, two belts on the TV.',
    'Toys roll past your station: tap the piece each toy asks for before it has gone by.',
    'Finished toys land on the sleigh. The first team to fill it wins.',
    'Shoo off the raccoon before it takes a toy!',
  ],
  nutcracker_knockout: [
    'Two wind-up robots, one phone each: Jab, Haymaker, Block, Dodge.',
    'A block stops a jab. A haymaker smashes through a block. A dodge makes anything miss.',
    'Knock the parts off one by one. Best of three rounds.',
  ],
  mecha_santa: [
    'Each team drives one giant wind-up Santa along the rooftops.',
    'Your phone is only your buttons: a leg, the balance or the arms. Nothing tells you when to press, so watch the TV and call it out!',
    'Legs: step in turn, left, right, left, right. The same leg twice, or a step before the last foot is down, and it trips.',
    'A gap: both legs JUMP together. A wall: the arms SMASH it. A rocket candle: the arms GRAB it. Balance: push it back before it tips over.',
    'The first robot over the line wins.',
  ],
  elf_tower_wars: [
    'Two workshops, three paths between them.',
    'On your phone: pick a card, then tap where it goes. Everything costs energy, and yours refills by itself.',
    'Send toys down a path: every one that reaches the other workshop hurts it.',
    'Build towers on your spots to stop theirs: a snow tower for soldiers, a cannon for a pack of mice. Build one again to make it better.',
    'The avalanche needs two of you on the same path at once. Energy doubles for the last half minute.',
    'Knock theirs down, or have the stronger one at the bell.',
  ],
};

export interface BriefingProps {
  game: Game;
  av: ArenaView;
  /** What the game is called and its picture (an emoji, drawn as an icon). */
  label: string; emoji: string; rules: string;
  now: number;
  /** A phone: who is holding it, and its Ready button. */
  me?: string; onReady?: () => void;
  /** The host console: start without waiting for the rest. */
  onStartNow?: () => void;
  /** `tv`: over the whole screen; `console`: a card above the stage; `phone`: the phone's whole screen. */
  where: 'tv' | 'console' | 'phone';
}

export function Briefing({ game, av, label, emoji, rules, now, me, onReady, onStartNow, where }: BriefingProps) {
  const b = av.briefing || { ready: [] };
  const lines = HOW_TO[av.gameId] || rules.split(/(?<=[.!?])\s+/).filter(Boolean);
  const people = av.participants.map((id) => ({ id, p: game.players?.[id] })).filter((x) => !!x.p);
  const isReady = (id: string) => b.ready.includes(id) || !!game.players?.[id]?.isBot;
  const waiting = people.filter((x) => !isReady(x.id) && x.p!.connected !== false);
  const count = b.goAt ? Math.max(1, Math.ceil((b.goAt - now) / 1000)) : 0;
  const mine = !!me && av.participants.includes(me);
  const iAmReady = !!me && isReady(me);

  // the TV counts the last three seconds out loud
  const said = useRef(0);
  useEffect(() => {
    if (where !== 'tv' || !count || said.current === count) return;
    said.current = count;
    play('tick', count === 1 ? 1.5 : 1);
  }, [count, where]);

  return (
    <div className={`brief brief-${where}`} role="dialog" aria-label={`How to play ${label}`} data-testid="arena-briefing">
      <div className="brief-card">
        <h2 className="brief-title"><Glyphs>{emoji}</Glyphs> {label}</h2>
        <ol className="brief-how" data-testid="briefing-how">
          {lines.map((line, i) => <li key={i}><span className="brief-n">{i + 1}</span><span>{line}</span></li>)}
        </ol>
        <ul className="brief-roster" aria-label="Who is ready">
          {people.map(({ id, p }) => (
            <li key={id} className={`${isReady(id) ? 'ready' : p!.connected === false ? 'away' : ''}`} data-testid={`briefing-player-${id}`}>
              <Avatar name={p!.name || id} avatar={p!.avatar} size={44} />
              <span className="brief-name">{p!.name || 'Someone'}</span>
              <span className="brief-mark" aria-label={isReady(id) ? 'ready' : 'not ready yet'}>{isReady(id) ? '✓' : '…'}</span>
            </li>
          ))}
        </ul>
        {count > 0 ? (
          <div className="brief-count" key={count} role="status" data-testid="briefing-count">{count}</div>
        ) : where === 'phone' && mine && !iAmReady ? (
          <button type="button" className="brief-ready" data-testid="briefing-ready" onPointerDown={(e) => { e.preventDefault(); onReady?.(); }} onClick={() => onReady?.()}>I am ready!</button>
        ) : (
          <p className="brief-wait" role="status" data-testid="briefing-wait">
            {where === 'phone' && !mine ? 'You are watching this one: eyes on the TV.'
              : waiting.length ? `${where === 'phone' ? 'Ready! ' : ''}Waiting for ${waiting.map((x) => x.p!.name || 'someone').join(', ')}${where === 'phone' ? '' : ': press Ready on your phone'}`
              : 'Everyone is ready…'}
          </p>
        )}
        {where === 'console' && !count && onStartNow && (
          <button type="button" className="btn btn-secondary brief-start" data-testid="arena-start-now" onClick={onStartNow}>Start without waiting</button>
        )}
      </div>
    </div>
  );
}
