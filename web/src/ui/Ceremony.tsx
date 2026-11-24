// How every mini-game ends, on every screen: the winners come in, paper falls, the Party points they
// earned count up. It comes up a moment after the game's own ending (which stays behind it) and stays
// until the room moves on. What it says is core/minigames/ceremony.ts; this only draws it.
import { useEffect, useRef } from 'react';
import { CEREMONY_DELAY_MS, ceremonyFor, myEnding, myPartyEnding, partyCeremony } from '../core/minigames/ceremony';
import type { ArenaView } from '../core/minigames/runtime';
import type { PartyGame } from '../core/party';
import { theGift } from '../core/stakes';
import type { Game } from '../core/types';
import { CeremonyCard } from './CeremonyCard';
import type { Where } from './CeremonyCard';
import { buzz, play } from './sound';
import './arena.css';

export function Ceremony({ game, av, now, where, me }: { game: Game; av: ArenaView; now: number; where: 'tv' | 'console' | 'phone'; me?: string }) {
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const practice = !!game.currentTurn?.lab;
  const c = av.practice ? null : ceremonyFor(av.result, name, practice);   // a practice round decided nothing: no announcement
  const up = !!c && av.endedAt != null && now >= av.endedAt + CEREMONY_DELAY_MS;
  const mine = me ? myEnding(av.result, me, av.participants) : 'watched';

  // once, as it comes up: the points land with a chime on the TV, a phone that won buzzes
  const rung = useRef(false);
  useEffect(() => {
    if (!up || rung.current) return;
    rung.current = true;
    if (where === 'tv' && c && [...c.stars, ...c.also].some((r) => r.points > 0)) play('ding');
    if (where === 'phone' && mine === 'won') buzz([40, 60, 120]);
  }, [up, where, c, mine]);

  if (!c || !up) return null;
  // (the stakes only when the room sent them: a practice round or the Game Lab promises nothing)
  return <CeremonyCard c={c} game={game} where={where} mine={mine} me={me} practice={practice} seed={av.stateVersion || 1} stakes={practice ? null : av.stakes} />;
}

/** A party game's results (Sketch-Off, Whack-an-Elf): the same ending. With `inline` the card sits in the
 *  stage (the TV's and the console's party stage has nothing else to show); a phone gets its whole screen. */
export function PartyCeremony({ mg, game, where, me, inline, picture = true }: { mg: PartyGame; game: Game; where: Where; me?: string; inline?: boolean; picture?: boolean }) {
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const t = game.currentTurn;
  const lab = mg.minigameGoal === 'lab';
  const gift = t?.giftId ? game.gifts?.[t.giftId]?.name : undefined;
  const c = partyCeremony(mg, name, { opener: t?.openerId ? name(t.openerId) : undefined, gift: gift ? theGift(gift) : undefined });
  const mine = me ? myPartyEnding(mg, me) : 'watched';
  const rung = useRef(false);
  useEffect(() => {
    if (!c || rung.current) return;
    rung.current = true;
    if (where === 'phone' && mine === 'won') buzz([40, 60, 120]);
  }, [c, where, mine]);
  if (!c) return null;
  // Sketch-Off: the drawing the table liked best, on the card
  const top = mg.type === 'sketch_off' ? (mg.results?.ranking || []).find((id) => (mg.submissions?.[id] || '').length > 6 && (mg.results?.scoreOf?.[id] ?? 0) > 0) : undefined;
  const pic = picture && top ? { img: mg.submissions![top], caption: `The table's favourite: ${name(top)}` } : null;
  return <CeremonyCard c={c} game={game} where={where} mine={mine} me={me} practice={lab} seed={mg.seed || 1} stakes={lab ? null : mg.stakes} picture={pic} inline={inline} testid="party-ceremony" />;
}
