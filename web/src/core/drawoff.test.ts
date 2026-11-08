// AI Draw Off, the duel game: the two duellists draw, the critic's higher score wins the duel;
// when the critic can't decide, nobody wins by default and the host picks.
import { describe, expect, it } from 'vitest';
import * as DO from './drawoff';
import * as G from './duelgame';
import { hostAuto } from './hostauto';
import { projectGame, HIDDEN_DRAWING } from './projection';
import { rng } from './rng';
import { HOST, offers, perform } from './table';
import { apply, NOW, table } from './testkit';
import type { Game } from './types';

const ctx = (now = NOW, seed = 3) => ({ now, rng: rng(seed) });

function drawOffDuel(): Game {
  const g = table(4, { duelPick: 'wheel', duelGames: [], duelAppGames: ['draw_off'], voteWindow: 0, aiCriticBackend: 'openai' });
  g.gifts = { gA: { name: 'Gift A', ownerId: 'p1', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'gA';
  g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', giftId: 'gA', pot: 2 };
  let x = apply(g, G.spinDuelGame(g, ctx()));
  expect(x.currentTurn?.duelGame?.app).toBe('draw_off');
  x = apply(x, G.startDuelApp(x, ctx()));
  return x;
}
/** Run the automatic step due now (as the server would). */
const step = (g: Game, now: number) => { const s = hostAuto(g).find((x) => (x.at || now) <= now); return s ? apply(g, s.run(g, ctx(now))) : g; };
const mgOf = (g: Game) => g.minigame as DO.DrawOff;

function drawn(): Game {
  let g = drawOffDuel();
  expect(g.currentTurn?.phase).toBe('minigame_draw_off');
  expect(mgOf(g).participants).toEqual(['p2', 'p1']);
  g = step(g, mgOf(g).promptRevealEndTime);                      // prompt → drawing
  expect(mgOf(g).phase).toBe('drawing');
  g = apply(g, perform(g, 'p2', { type: 'sketch', drawing: 'data:image/jpeg;base64,AAAA' }, ctx()));
  expect(offers(g, 'p3', NOW)).toEqual([]);                      // only the duellists draw
  g = apply(g, perform(g, 'p1', { type: 'sketch', drawing: 'data:image/jpeg;base64,BBBB' }, ctx()));
  g = step(g, NOW);                                              // both drew: judging straight away
  expect(mgOf(g).phase).toBe('judging');
  return g;
}

describe('AI Draw Off as a duel game', () => {
  it('the higher score wins the duel, and it resolves like any duel game', () => {
    let g = drawn();
    g = apply(g, DO.judged(g, { scores: { p2: { rating: 81 }, p1: { rating: 64 } }, verdict: 'P2 by a reindeer nose.' }, NOW));
    expect(mgOf(g)).toMatchObject({ phase: 'results', winners: ['p2'] });
    g = step(g, mgOf(g).resultsEndTime!);                          // back to the duel with the winner
    expect(g.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p2' });
    g = step(g, NOW);                                              // the duel resolves with them
    expect(g.lastDuel).toMatchObject({ winnerId: 'p2', loserId: 'p1' });
  });

  it("a tie or a failed critic: nobody wins by default; the host picks, and it's marked as theirs", () => {
    for (const result of [{ scores: { p2: { rating: 70 }, p1: { rating: 70 } } }, { failed: 'no AI critic key' }]) {
      let g = drawn();
      g = apply(g, DO.judged(g, result, NOW));
      expect(mgOf(g).winners).toEqual([]);
      g = apply(g, DO.completeDrawOff(g));
      expect(g.currentTurn?.duelGameWinner).toBeUndefined();
      expect(g.currentTurn?.duelGameOutcome?.status).toBe('no_contest');
      expect(hostAuto(g).some((s) => s.key.startsWith('app|'))).toBe(false);   // nothing resolves it by itself
      const later = NOW + 5 * 60_000;                             // the wheel stopped long ago
      expect(offers(g, HOST, later).map((o) => o.type)).toEqual(['duel_winner', 'duel_winner']);
      const plan = perform(g, HOST, { type: 'duel_winner', id: 'p1' }, ctx(later));
      expect(plan.events[0].text).toMatch(/Host decision/);
      expect(apply(g, plan).lastDuel).toMatchObject({ winnerId: 'p1' });
    }
  });

  it('nobody sees a drawing until judging: not the rival, not the TV', () => {
    let g = drawOffDuel();
    g = step(g, mgOf(g).promptRevealEndTime);
    g = apply(g, perform(g, 'p2', { type: 'sketch', drawing: 'data:image/jpeg;base64,SECRET' }, ctx()));
    const subs = (v: Parameters<typeof projectGame>[1]) => (projectGame(g, v).minigame as DO.DrawOff).submissions;
    expect(subs({ kind: 'phone', pid: 'p1' })).toEqual({ p2: HIDDEN_DRAWING });
    expect(subs({ kind: 'tv' })).toEqual({ p2: HIDDEN_DRAWING });
    expect(subs({ kind: 'phone', pid: 'p2' })).toEqual({ p2: 'data:image/jpeg;base64,SECRET' });
  });

  it("the drawing clock: time's up moves it to judging even if someone didn't draw", () => {
    let g = drawOffDuel();
    g = step(g, mgOf(g).promptRevealEndTime);
    g = step(g, mgOf(g).drawingEndTime! + DO.DRAW_GRACE_MS);
    expect(mgOf(g).phase).toBe('judging');
  });
});
