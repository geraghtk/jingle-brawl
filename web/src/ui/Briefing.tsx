// Before a mini-game starts: how to play it, who is ready, and (once everyone is) the 3, 2, 1. The same
// panel on the TV, the host console and every phone; a phone in the game also has the Ready button.
// The room decides when the game starts (core/minigames/runtime.ts: briefing); this only shows it.
import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { ArenaView } from '../core/minigames/runtime';
import type { Game } from '../core/types';
import { Avatar } from './avatar';
import { Glyphs } from './icons';
import { play } from './sound';
import './motion.css';
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
    'A chimney stack: the arms GRAB, then the legs climb. A rope: the arms GRAB, then both legs JUMP to let go. The course is new every race.',
    'The first robot over the line wins.',
  ],
  elf_tower_wars: [
    'Two workshops, three paths between them. Rounds of two halves, until a workshop falls.',
    'PLAN: everyone gets energy. On your phone pick a card, then tap where it goes: toys line up at a path, towers go on your spots. Talk it over, then press Done.',
    'MARCH: both armies walk out together. Watch the TV! Every toy that reaches the other workshop hurts it.',
    'Towers stop theirs: a snow tower for soldiers, a cannon for a pack of mice. Build one again to make it better.',
    'While they march you have a snowball, a shield, and an avalanche that needs two of you on the same path.',
    'Knock theirs down, or have the stronger one after the last round.',
  ],
};

/** True for a moment after the instructions give way to the game: "GO!". */
export function useGo(lifecycle: string | undefined, instanceId: string | undefined): boolean {
  const was = useRef<string | undefined>(undefined);
  const [go, setGo] = useState('');
  useEffect(() => {
    const before = was.current;
    was.current = lifecycle;
    if (before !== 'briefing' || lifecycle !== 'active' || !instanceId) return;
    setGo(instanceId);
    const t = setTimeout(() => setGo(''), 900);
    return () => clearTimeout(t);
  }, [lifecycle, instanceId]);
  return !!go && go === instanceId;
}

/** Games that open with a word of their own ("Round 1", practice throws): no "GO!" laid over it. */
export const OWN_OPENING = new Set(['nutcracker_knockout', 'reindeer_ring_toss']);

/** "GO!" across the screen for a moment (it takes no taps: the game under it is already on). */
export function GoFlash() {
  return <div className="arena-go" aria-hidden data-testid="arena-go"><b>GO!</b></div>;
}

/** A practice round is on: a ribbon across the top of the game saying so, and how long is left. */
export function PracticeRibbon({ endsAt, now }: { endsAt?: number; now: number }) {
  const left = endsAt != null ? Math.max(0, Math.ceil((endsAt - now) / 1000)) : null;
  return (
    <div className="practice-ribbon mo-down" role="status" data-testid="practice-ribbon">
      <b>Practice</b><span>This one does not count{left != null ? `: ${left}` : ''}</span>
    </div>
  );
}

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

  // after a practice round: no instructions again, just "now for real" and the count
  if (b.real) {
    return (
      <div className={`brief brief-${where} brief-real`} role="dialog" aria-label={`${label}: now for real`} data-testid="arena-briefing">
        <div className="brief-card">
          <p className="brief-kicker mo-down">That was practice</p>
          <h2 className="brief-title"><Glyphs>{emoji}</Glyphs> Now for real!</h2>
          {count > 0 && <div className="brief-count" key={count} role="status" data-testid="briefing-count">{count}</div>}
        </div>
      </div>
    );
  }

  return (
    <div className={`brief brief-${where}${lines.length >= 5 ? " brief-long" : ""}`} role="dialog" aria-label={`How to play ${label}`} data-testid="arena-briefing">
      <div className="brief-card">
        <h2 className="brief-title"><Glyphs>{emoji}</Glyphs> {label}</h2>
        <ol className="brief-how" data-testid="briefing-how">
          {lines.map((line, i) => <li key={i} style={{ ['--i']: i } as CSSProperties}><span className="brief-n">{i + 1}</span><span>{line}</span></li>)}
        </ol>
        {av.practice && <p className="brief-practice" data-testid="briefing-practice">First a short practice round: it does not count.</p>}
        <ul className="brief-roster" aria-label="Who is ready">
          {people.map(({ id, p }, i) => (
            <li key={id} style={{ ['--i']: i } as CSSProperties} className={`${isReady(id) ? 'ready' : p!.connected === false ? 'away' : ''}`} data-testid={`briefing-player-${id}`}>
              <Avatar name={p!.name || id} avatar={p!.avatar} size={44} />
              <span className="brief-name">{p!.name || 'Someone'}</span>
              <span className="brief-mark" aria-label={isReady(id) ? 'ready' : 'not ready yet'}>{isReady(id) ? '✓' : '…'}</span>
            </li>
          ))}
        </ul>
        {count > 0 && <div className="brief-count" key={count} role="status" data-testid="briefing-count">{count}</div>}
        {where === 'phone' && mine && !iAmReady && !count ? (
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
