// The host's automatic steps, as the host page and the simulated table both run them.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import { hostAuto, WHACK_GRACE_MS } from './hostauto';
import type { PartyGame } from './party';
import { SPIN_MS } from './duelgame';
import { apply, ctx, NOW, table } from './testkit';
import type { Game } from './types';

function tied(settings = {}): Game {
  let g = table(4, settings);
  g = apply(g, A.drawGift(g, 'Gift', 'g1', ctx(1)));
  const opener = g.currentTurn!.openerId!;
  for (const id of ['p1', 'p2', 'p3', 'p4'].filter((x) => x !== opener)) g = apply(g, A.submitBid(g, id, 2, NOW));
  g = apply(g, A.revealBids(g));
  return apply(g, A.openerFight(g, null, null, true, ctx(2).rng));
}

describe('hostAuto', () => {
  it('a tie starts its party game straight away; the party then runs on its phase clock', () => {
    let g = tied({ tiebreakerMethod: 'whack_elf' });
    const tie = hostAuto(g).find((s) => s.key.startsWith('tie|'));
    expect(tie?.at).toBe(0);
    g = apply(g, tie!.run(g, ctx(3)));
    const mg = g.minigame as PartyGame;
    expect(g.currentTurn!.phase).toBe('minigame_party');
    expect(hostAuto(g).map((s) => [s.key.split('|')[0], s.at])).toEqual([['party', mg.phaseEndsAt + 250]]);
    g = apply(g, hostAuto(g)[0].run(g, { now: mg.phaseEndsAt, rng: ctx().rng }));      // intro → playing
    const playing = g.minigame as PartyGame;
    expect(hostAuto(g)[0].at).toBe(playing.phaseEndsAt + WHACK_GRACE_MS);               // last scores land
  });

  it('Sketch-Off moves on as soon as every drawing is in (a new key, due now)', () => {
    let g = tied({ tiebreakerMethod: 'sketch_off' });
    g = apply(g, hostAuto(g)[0].run(g, ctx(3)));
    g = apply(g, hostAuto(g)[0].run(g, ctx(3)));                                         // → playing
    const before = hostAuto(g)[0];
    for (const id of (g.minigame as PartyGame).participants) g = apply(g, { updates: { [`minigame/submissions/${id}`]: 'data:,' }, events: [] });
    const after = hostAuto(g)[0];
    expect(after.key).not.toBe(before.key);
    expect(after.at).toBe(0);
  });

  it('the duel vote opens once the wheel has stopped, not while it spins', () => {
    let g = table(3, { duelPick: 'wheel', voteWindow: 20, duelGames: 'Thumb War\nArm Wrestle' });
    g = apply(g, A.drawGift(g, 'Gift', 'g1', ctx(1)));
    const opener = g.currentTurn!.openerId!;
    const challenger = ['p1', 'p2', 'p3'].find((x) => x !== opener)!;
    g = apply(g, A.submitBid(g, challenger, 1, NOW));
    g = apply(g, A.revealBids(g));
    g = apply(g, A.startDuel(g, challenger, opener));
    const spin = hostAuto(g).find((s) => s.key.startsWith('spin|'))!;
    expect(hostAuto(g).some((s) => s.key.startsWith('vote|'))).toBe(false);
    g = apply(g, spin.run(g, ctx(4)));
    const vote = hostAuto(g).find((s) => s.key.startsWith('vote|'))!;
    expect(vote.at).toBe(NOW + SPIN_MS);
  });

  describe('the Misfit Lottery', () => {
    const lottery = (): Game => ({
      ...table(3), phase: 'endgame',
      players: { p1: { name: 'P1', chips: 10, giftId: 'g1' }, p2: { name: 'P2', chips: 10, giftId: 'g2' }, p3: { name: 'P3', chips: 10 } },
      gifts: { g1: { name: 'Socks', ownerId: 'p1', naughtyLevel: 0 }, g2: { name: 'Mug', ownerId: 'p2', naughtyLevel: 0 }, g3: { name: 'Candle', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1 } },
    });
    const step = (g: Game) => hostAuto(g).find((s) => s.key.startsWith('lottery|'));

    it('between two endgame turns the next giftless player is drawn straight away, from the state alone', () => {
      let g = lottery();
      expect(step(g)?.at).toBe(0);
      g = apply(g, step(g)!.run(g, ctx(1)));
      expect(g.currentTurn).toMatchObject({ phase: 'endgame_choice', activePlayerId: 'p3', misfitId: 'g3' });
      expect(step(g)).toBeUndefined();                                    // a turn is on: nothing to draw
    });

    it('everyone holds a gift: the step ends the game', () => {
      let g = lottery();
      g = { ...g, players: { ...g.players, p3: { name: 'P3', chips: 10, giftId: 'g3' } }, gifts: { ...g.gifts, g3: { name: 'Candle', ownerId: 'p3', naughtyLevel: 0 } } };
      g = apply(g, step(g)!.run(g, ctx(1)));
      expect(g.phase).toBe('ended');
      expect(hostAuto(g)).toEqual([]);
    });

    it('no step outside the endgame, or when the giftless have no Misfit to face (the host ends that game)', () => {
      expect(step({ ...lottery(), phase: 'main' })).toBeUndefined();
      const g = lottery();
      expect(step({ ...g, gifts: { g1: g.gifts!.g1, g2: g.gifts!.g2 } })).toBeUndefined();
    });

    it('each draw is a new step: the key follows who is still giftless and what is on the pile', () => {
      const g = lottery();
      const more: Game = { ...g, gifts: { ...g.gifts, g4: { name: 'Hat', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 2 } } };
      expect(step(more)!.key).not.toBe(step(g)!.key);
    });
  });
});
