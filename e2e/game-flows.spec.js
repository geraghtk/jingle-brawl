// @ts-check
// Permutation tests for the main game flow.
//
// Goal: exercise every decision fork in a real host+players game and catch bugs
// via INVARIANTS (chips never negative, gifts never orphaned/duplicated,
// player.giftId <-> gift.ownerId consistent, the turn always resolves, no page
// errors). Ambiguous chip splits (Santa Tax) are printed, not asserted.
//
// Each test is one fresh emulator room with 3 players.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

async function setup(browser, settings = {}) {
  const { host, hostCtx, roomCode, errors: hostErrors } = await H.createGame(browser, settings);
  const { pages, contexts, errors: playerErrors } = await H.joinPlayers(browser, roomCode, NAMES);
  await H.startGame(host, roomCode);

  const cleanup = async () => {
    await hostCtx.close();
    for (const c of contexts) await c.close();
  };
  const assertNoPageErrors = () => {
    const all = [...hostErrors, ...playerErrors];
    expect(all, `page errors:\n${all.join('\n')}`).toEqual([]);
  };
  return { host, roomCode, pages, cleanup, assertNoPageErrors };
}

/** name -> phone page, keyed also by player id for reprisal skipping. */
async function pagesById(roomCode, pages) {
  const byName = await H.playersByName(roomCode);
  const map = {};
  for (const [name, page] of Object.entries(pages)) {
    if (byName[name]) map[byName[name].id] = page;
  }
  return map;
}

/** Print the chip line so the ambiguous Santa-Tax split is visible, not asserted. */
async function reportChips(roomCode, tag) {
  const p = await H.playersByName(roomCode);
  const line = NAMES.map((n) => `${n}=${p[n].chips}`).join('  ');
  console.log(`  [chips ${tag}] ${line}`);
}

test.describe('Game flow permutations', () => {
  test('no bids: opener keeps the gift', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Scented Candle');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));

      await H.allBid(pages, opener, {}); // everyone passes
      await H.revealBids(host);
      await H.waitForTurnPhase(roomCode, 'reveal');
      await H.openerKeeps(pages[opener]);
      await H.waitForTurnPhase(roomCode, null);

      const game = await H.assertInvariants(roomCode, 'no-bids keep');
      const openerId = (await H.playersByName(roomCode))[opener].id;
      const gift = Object.values(game.gifts).find((g) => g.name === 'Scented Candle');
      expect(gift.ownerId).toBe(openerId);
      for (const n of NAMES) expect((await H.playersByName(roomCode))[n].chips).toBe(10);
      await reportChips(roomCode, 'after keep');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('single bidder: challenger wins the gift', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Rubber Duck');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);

      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder);
      await H.resolveDuel(host, roomCode, await pagesById(roomCode, pages));

      const game = await H.assertInvariants(roomCode, 'single bidder wins');
      const p = await H.playersByName(roomCode);
      const gift = Object.values(game.gifts).find((g) => g.name === 'Rubber Duck');
      expect(gift.ownerId).toBe(p[bidder].id);
      expect(gift.ownerId).not.toBe(p[opener].id);
      await reportChips(roomCode, 'challenger won');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('single bidder: defender (opener) wins, keeps the gift', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Mystery Box');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);

      await H.allBid(pages, opener, { [bidder]: 3 }); // 3 => triggers Santa Tax
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, opener);
      await H.resolveDuel(host, roomCode, await pagesById(roomCode, pages));

      const game = await H.assertInvariants(roomCode, 'defender wins');
      const p = await H.playersByName(roomCode);
      const gift = Object.values(game.gifts).find((g) => g.name === 'Mystery Box');
      expect(gift.ownerId).toBe(p[opener].id);
      // Challenger paid their bid and lost it into the pot -> fewer chips than start
      expect(p[bidder].chips).toBeLessThan(10);
      await reportChips(roomCode, 'defender won (bid 3, tax split)');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('two unequal bids, opener fights: top bidder duels the opener', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Fancy Socks');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [high, low] = NAMES.filter((n) => n !== opener);

      await H.allBid(pages, opener, { [high]: 2, [low]: 1 });
      await H.revealBids(host);
      await H.openerFights(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'duel_pending');
      await H.clickStartDuel(host);
      const duel = await H.waitForTurnPhase(roomCode, 'duel');

      const p0 = await H.playersByName(roomCode);
      expect(duel.challengerId).toBe(p0[high].id);
      expect(duel.defenderId).toBe(p0[opener].id);

      await H.declareWinner(host, high);
      await H.resolveDuel(host, roomCode, await pagesById(roomCode, pages));

      const game = await H.assertInvariants(roomCode, 'two-bid fight');
      const p = await H.playersByName(roomCode);
      const gift = Object.values(game.gifts).find((g) => g.name === 'Fancy Socks');
      expect(gift.ownerId).toBe(p[high].id);
      await reportChips(roomCode, 'two-bid fight, high wins');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('two unequal bids, opener yields: the two bidders duel each other', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Yield Mug');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [high, low] = NAMES.filter((n) => n !== opener);

      await H.allBid(pages, opener, { [high]: 2, [low]: 1 });
      await H.revealBids(host);
      await H.openerYields(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'yield_pending');
      await H.clickStartYieldDuel(host);
      const duel = await H.waitForTurnPhase(roomCode, 'duel');

      const p0 = await H.playersByName(roomCode);
      expect([duel.challengerId, duel.defenderId].sort())
        .toEqual([p0[high].id, p0[low].id].sort());

      await H.declareWinner(host, high);
      await H.resolveDuel(host, roomCode, await pagesById(roomCode, pages));

      const game = await H.assertInvariants(roomCode, 'yield duel');
      const p = await H.playersByName(roomCode);
      const gift = Object.values(game.gifts).find((g) => g.name === 'Yield Mug');
      expect(gift.ownerId).toBe(p[high].id); // duel winner takes it
      expect(p[opener].giftId ?? null).toBeNull(); // opener yielded -> giftless
      await reportChips(roomCode, 'yield duel, high wins');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('tied bids, opener yields: challengers duel for the gift', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Ugly Sweater');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [c1, c2] = NAMES.filter((n) => n !== opener);

      await H.allBid(pages, opener, { [c1]: 2, [c2]: 2 });
      await H.revealBids(host);
      await H.openerYields(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'yield_pending');
      await H.clickStartYieldDuel(host);
      const duel = await H.waitForTurnPhase(roomCode, 'duel');

      const p0 = await H.playersByName(roomCode);
      const winnerName = duel.challengerId === p0[c1].id ? c1 : c2;
      await H.declareWinner(host, winnerName);
      await H.resolveDuel(host, roomCode, await pagesById(roomCode, pages));

      const game = await H.assertInvariants(roomCode, 'tied yield');
      const p = await H.playersByName(roomCode);
      const gift = Object.values(game.gifts).find((g) => g.name === 'Ugly Sweater');
      expect(gift.ownerId).toBe(p[winnerName].id);
      await reportChips(roomCode, 'tied yield');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('tied bids, opener fights: the whole table plays a party game, the better tied bidder duels the opener', async ({ browser }) => {
    test.setTimeout(90000); // a whole party game is played
    // Sketch-Off: the outcome follows the votes the test casts (Whack-an-Elf phones report live scores)
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'tiebreaker-method': 'sketch_off' });
    try {
      const turn = await H.drawGift(host, roomCode, 'Gag Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [c1, c2] = NAMES.filter((n) => n !== opener);

      await H.allBid(pages, opener, { [c1]: 2, [c2]: 2 });
      await H.revealBids(host);
      await H.openerFights(pages[opener]);
      // a tie is a party game for everyone (no two-player tiebreak duel any more)
      const p0 = await H.playersByName(roomCode);
      const tbWinnerName = c2; // c2 places above c1 (and above the opener)
      const mainDuel = await H.settlePartyTie(host, roomCode, [p0[c2].id, p0[opener].id, p0[c1].id]);
      expect(mainDuel.isTiebreak).toBeFalsy();
      const p1 = await H.playersByName(roomCode);
      expect(mainDuel.challengerId).toBe(p1[tbWinnerName].id);
      expect(mainDuel.defenderId).toBe(p1[opener].id);

      await H.declareWinner(host, opener); // opener defends successfully
      await H.resolveDuel(host, roomCode, await pagesById(roomCode, pages));

      const game = await H.assertInvariants(roomCode, 'tied fight');
      const p = await H.playersByName(roomCode);
      const gift = Object.values(game.gifts).find((g) => g.name === 'Gag Gift');
      expect(gift.ownerId).toBe(p[opener].id); // opener kept it
      await reportChips(roomCode, 'tied fight, opener defends');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('reprisal is offered to a duel loser when another gift exists', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      // Round 1: open a gift and keep it, so a second gift exists as a reprisal target
      const t1 = await H.drawGift(host, roomCode, 'First Gift');
      const opener1 = H.openerNameOf(t1, await H.playersByName(roomCode));
      await H.allBid(pages, opener1, {});
      await H.revealBids(host);
      await H.waitForTurnPhase(roomCode, 'reveal');
      await H.openerKeeps(pages[opener1]);
      await H.waitForTurnPhase(roomCode, null);
      await H.assertInvariants(roomCode, 'after round 1 keep');

      // Round 2: someone opens, a challenger bids and wins -> the loser (opener2)
      // should now be offered a reprisal because 'First Gift' is a legal target.
      const t2 = await H.drawGift(host, roomCode, 'Second Gift');
      const opener2 = H.openerNameOf(t2, await H.playersByName(roomCode));
      const bidder = NAMES.find((n) => n !== opener2 && n !== opener1) || NAMES.find((n) => n !== opener2);

      await H.allBid(pages, opener2, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder);

      const reprisal = await H.waitForTurnPhase(roomCode, 'reprisal');
      const p = await H.playersByName(roomCode);
      expect(reprisal.reprisalPlayerId).toBe(p[opener2].id);

      // Loser skips -> turn resolves cleanly
      await H.skipReprisal(pages[opener2]);
      await H.waitForTurnPhase(roomCode, null);
      await H.assertInvariants(roomCode, 'after reprisal skip');
      await reportChips(roomCode, 'reprisal offered then skipped');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('full game of keeps reaches the final tally with every gift owned', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      for (let round = 0; round < NAMES.length; round++) {
        const turn = await H.drawGift(host, roomCode, `Gift ${round + 1}`);
        const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
        await H.allBid(pages, opener, {});
        await H.revealBids(host);
        await H.waitForTurnPhase(roomCode, 'reveal');
        await H.openerKeeps(pages[opener]);
        await H.waitForTurnPhase(roomCode, null);
        await H.assertInvariants(roomCode, `after round ${round + 1}`);
      }

      host.once('dialog', (d) => d.accept());
      await host.getByRole('button', { name: /End Game for All/i }).click();
      await H.waitForGame(roomCode, (g) => g.phase === 'ended', { label: 'ended phase' });

      const game = await H.assertInvariants(roomCode, 'final tally');
      await expect(host.locator('#turn-panel')).toContainText(/Who got/i);
      const owners = Object.values(game.gifts).map((g) => g.ownerId).sort();
      const ids = Object.keys(game.players).sort();
      expect(owners).toEqual(ids); // every player owns exactly one gift
      await reportChips(roomCode, 'final');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });
});
