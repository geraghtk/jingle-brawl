// Intents: every button is data the game can check. Parsing (an intent from the network),
// who may send what, and parity: a button's intent produces the plan its old click built.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import { dispatch, parseIntent } from './intents';
import type { Intent } from './intents';
import { pick, rng } from './rng';
import { giftless, rankedBids, topTiedBidders } from './rules';
import { HOST, offers } from './table';
import { apply, ctx, NOW, table } from './testkit';
import type { Game } from './types';

function opened(g: Game): Game {
  for (let seed = 1; seed < 200; seed++) {
    const plan = A.drawGift(g, 'Gift A', 'gA', ctx(seed));
    if (plan.updates['players/p1/giftId'] === 'gA') return apply(g, plan);
  }
  throw new Error('no seed draws p1');
}
const bid = (g: Game, bids: Record<string, number>) => Object.entries(bids).reduce((acc, [id, n]) => apply(acc, A.submitBid(acc, id, n, NOW)), g);
const revealed = (n: number, bids: Record<string, number>) => { const g = bid(opened(table(n)), bids); return apply(g, A.revealBids(g)); };

describe('parseIntent: what may arrive over the network', () => {
  const ok: Intent[] = [
    { type: 'bid', amount: 3 }, { type: 'keep' }, { type: 'gambit', giftId: 'gift_1' }, { type: 'pick_duel', index: 2, name: 'Rock Paper Scissors' },
    { type: 'whack', score: 12, final: true }, { type: 'sketch', drawing: 'data:image/jpeg;base64,AAAA' }, { type: 'draw', giftName: 'Socks', seed: 42 },
    { type: 'admin_set_chips', pid: 'p1', chips: 9 }, { type: 'admin_save_gift', giftId: 'g', naughtyLevel: 2, ownerId: null },
  ];
  it('accepts every well-formed intent as is', () => {
    for (const i of ok) expect(parseIntent(i)).toEqual(i);
  });
  it('refuses junk: unknown types, extra or missing fields, out-of-range numbers, odd ids, huge or non-image drawings', () => {
    const bad: unknown[] = [
      null, 'bid', [], { type: 'steal_everything' }, { type: 'bid' }, { type: 'bid', amount: 2.5 }, { type: 'bid', amount: -1 },
      { type: 'bid', amount: 3, playerId: 'p2' },            // who you are comes from your session, not the message
      { type: 'gambit', giftId: '../players' }, { type: 'whack', score: 1e9 }, { type: 'whack', final: 'yes' },
      { type: 'sketch', drawing: 'javascript:alert(1)' }, { type: 'sketch', drawing: `data:image/png;base64,${'A'.repeat(700_000)}` },
      { type: 'draw', giftName: 'x'.repeat(81) }, { type: 'admin_set_chips', pid: 'p1', chips: 1e6 }, { type: 'admin_add_gift', name: '   ', ownerId: 'p1' },
    ];
    for (const b of bad) expect(parseIntent(b), JSON.stringify(b)?.slice(0, 80)).toHaveProperty('error');
  });
});

describe('dispatch: who may send what', () => {
  it('admin intents are the host console only', () => {
    const g = table(3);
    expect(dispatch(g, 'p1', { type: 'admin_set_chips', pid: 'p1', chips: 99 }, ctx()).error).toMatch(/host/);
    expect(apply(g, dispatch(g, HOST, { type: 'admin_set_chips', pid: 'p1', chips: 7 }, ctx())).players!.p1.chips).toBe(7);
  });
  it("a button that isn't on that person's screen is refused (a phone can't reveal the bids or bid on its own gift)", () => {
    const g = bid(opened(table(3)), {});
    expect(dispatch(g, 'p2', { type: 'reveal' }, ctx()).error).toBeTruthy();
    expect(dispatch(g, 'p1', { type: 'bid', amount: 1 }, ctx()).error).toBeTruthy();
    expect(dispatch(g, HOST, { type: 'reveal' }, ctx()).error).toBeUndefined();
  });
  it('a sealed bid can be changed until the reveal, within your chips and the max bid', () => {
    let g = bid(opened(table(3)), { p2: 1 });
    g = apply(g, dispatch(g, 'p2', { type: 'bid', amount: 3 }, ctx()));
    expect(g.currentTurn!.bids!.p2.amount).toBe(3);
    expect(dispatch(g, 'p2', { type: 'bid', amount: 99 }, ctx()).error).toBeTruthy();
    g = apply(g, A.revealBids(g));
    expect(dispatch(g, 'p2', { type: 'bid', amount: 1 }, ctx()).error).toBeTruthy();
  });
  it("the console's draw names who opens before asking the gift (its seed draws the person it previewed)", () => {
    const g = table(4);
    const seed = 1234;
    const previewed = pick(giftless(g), rng(seed));
    const after = apply(g, dispatch(g, HOST, { type: 'draw', giftName: 'Socks', seed }, ctx()));
    expect(after.currentTurn?.openerId).toBe(previewed);
  });
});

describe("the console's Manual Override (now offers, like the phone's buttons)", () => {
  it('no bids: the host can keep or play a Gambit for the Opener; bidding can be skipped', () => {
    const g = revealed(3, {});
    expect(offers(g, HOST, NOW).map((o) => o.type)).toContain('keep');
    expect(offers(bid(opened(table(3)), {}), HOST, NOW).map((o) => o.type)).toContain('skip_bidding');
  });
});

describe('parity: each button sends an intent that builds the plan its click used to build', () => {
  const same = (g: Game, who: string, intent: Intent, old: (r: ReturnType<typeof rng>) => unknown) =>
    expect(dispatch(g, who, intent, { now: NOW, rng: rng(5) })).toEqual(old(rng(5)));

  it("the Opener's fight / yield (one top bid, a 2-way tie, a 3-way tie)", () => {
    let g = revealed(4, { p2: 3, p3: 1, p4: 0 });
    same(g, 'p1', { type: 'fight' }, (r) => A.openerFight(g, 'p2', 'p3', false, r));
    same(g, 'p1', { type: 'yield' }, (r) => A.openerYield(g, 'p2', 'p3', r));
    g = revealed(4, { p2: 2, p3: 2, p4: 0 });
    const [[c1], [c2]] = rankedBids(g.currentTurn!);   // the phone named the two tied bidders in this order
    same(g, 'p1', { type: 'fight' }, (r) => A.openerFight(g, c1, c2, true, r));
    same(g, 'p1', { type: 'yield' }, (r) => A.openerYield(g, c1, c2, r));
    g = revealed(4, { p2: 2, p3: 2, p4: 2 });
    same(g, 'p1', { type: 'fight' }, (r) => A.openerFight(g, null, null, true, r));
    same(g, 'p1', { type: 'yield' }, (r) => A.openerYield(g, null, null, r));
  });
  it("the console's override of the Opener's choice", () => {
    let g = revealed(4, { p2: 3, p3: 1, p4: 0 });
    same(g, HOST, { type: 'fight' }, () => A.startDuel(g, 'p2', 'p1'));
    same(g, HOST, { type: 'yield' }, (r) => A.startYieldDuel(g, 'p2', 'p3', false, { now: NOW, rng: r }));
    g = revealed(4, { p2: 2, p3: 2, p4: 0 });
    const tied = topTiedBidders(g.currentTurn);
    same(g, HOST, { type: 'yield' }, (r) => A.startYieldDuel(g, tied[0], tied[1], false, { now: NOW, rng: r }));
    g = revealed(4, { p2: 2, p3: 2, p4: 2 });
    same(g, HOST, { type: 'fight' }, (r) => A.hostTieFight(g, { now: NOW, rng: r }));
    same(g, HOST, { type: 'yield' }, (r) => A.hostTieYield(g, { now: NOW, rng: r }));
  });
});
