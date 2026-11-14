// @ts-check
// Regression tests for bugs found in the 2026-10 full code review.
//
// Each test drives a real host + phones on the room server and asserts the
// behaviour the rules (Jingle Brawl V2.md) or the UI promise. They were written
// to FAIL against the pre-fix code, then the fixes were made.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

// The mascots pick a line at random from a pool per moment (web/src/ui/mascots.ts): match any
// line of the pool.
const MASCOT_LINES = (() => {
  const src = require('fs').readFileSync(require('path').join(__dirname, '..', 'web', 'src', 'ui', 'mascots.ts'), 'utf8');
  const { code } = require('esbuild').transformSync(src, { loader: 'ts', format: 'cjs' });
  const module = { exports: {} };
  new Function('module', 'exports', 'location', code)(module, module.exports, { search: '' });
  return module.exports.lines;
})();
const saysOneOf = (moment, x) => new RegExp(MASCOT_LINES[moment]
  .map((l) => (typeof l === 'function' ? l(x) : l).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'));

async function setup(browser, settings = {}, names = NAMES) {
  const { host, hostCtx, roomCode, errors: hostErrors } = await H.createGame(browser, settings);
  const { pages, contexts, errors: playerErrors } = await H.joinPlayers(browser, roomCode, names);
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

/** Round where the opener keeps (no bids). Returns the opener's name. */
async function keepRound(host, roomCode, pages, giftName) {
  const turn = await H.drawGift(host, roomCode, giftName);
  const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
  await H.allBid(pages, opener, {});
  await H.revealBids(host);
  await H.waitForTurnPhase(roomCode, 'reveal');
  await H.openerKeeps(pages[opener]);
  await H.waitForTurnPhase(roomCode, null);
  return opener;
}

const giftByName = (game, name) => {
  const entry = Object.entries(game.gifts || {}).find(([, g]) => g.name === name);
  return entry ? { id: entry[0], ...entry[1] } : null;
};

test.describe('Regressions (2026-10 review)', () => {
  test('custom Starting Chips setting reaches players who join', async ({ browser }) => {
    const { host, hostCtx, roomCode } = await H.createGame(browser, { 'starting-chips': '15' });
    const { contexts } = await H.joinPlayers(browser, roomCode, ['Alice', 'Bob']);
    try {
      const p = await H.playersByName(roomCode);
      expect(p.Alice.chips).toBe(15);
      expect(p.Bob.chips).toBe(15);
      expect((await H.dbGet(roomCode, 'settings')).startingChips).toBe(15);
    } finally {
      await hostCtx.close();
      for (const c of contexts) await c.close();
    }
  });

  test('a spectator can vote on a duel without an error toast', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Vote Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder, spectator] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');

      const phone = pages[spectator];
      await phone.getByRole('button', { name: new RegExp(`🏆 ${bidder}`) }).click();
      await expect(phone.locator('#toast')).toContainText(/Voted for/);
      await expect(phone.locator('#status-card')).toContainText(/You voted for/);
      const p = await H.playersByName(roomCode);
      const votes = await H.dbGet(roomCode, 'currentTurn/duelVotes');
      expect(votes[p[spectator].id].votedFor).toBe(p[bidder].id);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("cancelling Grinch's Gambit returns the opener to Keep/Gambit", async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      await keepRound(host, roomCode, pages, 'First Gift');
      const turn = await H.drawGift(host, roomCode, 'Second Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      await H.allBid(pages, opener, {});
      await H.revealBids(host);
      await H.waitForTurnPhase(roomCode, 'reveal');

      const phone = pages[opener];
      await phone.getByRole('button', { name: /Grinch's Gambit/ }).click();
      await expect(phone.locator('#status-card')).toContainText(/Choose a gift to challenge/);
      await phone.getByRole('button', { name: /Cancel/ }).click();
      await expect(phone.getByRole('button', { name: /Keep My Gift/ })).toBeVisible();
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("duel help text matches the duel: a Gambit promises no Reprisal, and only defenders get the dividend", async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const keeper = await keepRound(host, roomCode, pages, 'Owned Gift');
      const turn = await H.drawGift(host, roomCode, 'Gambit Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const spectator = NAMES.find((n) => n !== opener && n !== keeper);
      await H.allBid(pages, opener, {});
      await H.revealBids(host);
      await H.waitForTurnPhase(roomCode, 'reveal');
      await pages[opener].getByRole('button', { name: /Grinch's Gambit/ }).click();
      await pages[opener].locator('#status-card').getByText('Owned Gift').click();
      await H.waitForGame(roomCode, (g) => (g.currentTurn?.isGambit ? g : false), { label: 'gambit duel' });

      for (const name of [opener, keeper, spectator]) {
        const card = pages[name].locator('#status-card');
        await expect(card).toBeVisible();
        await expect(card, `${name}'s phone`).not.toContainText(/Reprisal (chance|opportunity)/i);
      }
      // default dividend rule is "defenders": the spectator is told only a losing defender gets +1
      await expect(pages[spectator].locator('#status-card')).toContainText(/\+1 chip if they were defending/i);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("Grinch's Gambit never offers a Misfit-pile gift (there is no owner to duel)", async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      await keepRound(host, roomCode, pages, 'Owned Gift');
      await H.dbPut(roomCode, 'gifts/gift_misfit_test', {
        name: 'Old Fruitcake', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1, createdAt: 1,
      });
      // 3 players, 2 gifts (one kept + this misfit) -> 1 wrapped gift left to draw
      const turn = await H.drawGift(host, roomCode, 'Gambit Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      await H.allBid(pages, opener, {});
      await H.revealBids(host);
      await H.waitForTurnPhase(roomCode, 'reveal');

      const phone = pages[opener];
      await phone.getByRole('button', { name: /Grinch's Gambit/ }).click();
      const card = phone.locator('#status-card');
      await expect(card).toContainText('Owned Gift');
      await expect(card).not.toContainText('Old Fruitcake');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('a phone-initiated reprisal starts a clean duel (no stale votes, tax or bids)', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const keeper = await keepRound(host, roomCode, pages, 'Kept Gift');
      const turn = await H.drawGift(host, roomCode, 'Contested Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const bidder = NAMES.find((n) => n !== opener && n !== keeper);
      expect(keeper).not.toBe(opener);

      await H.allBid(pages, opener, { [bidder]: 3 }); // bid 3 -> Santa Tax on the pot
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      // The keeper (not in the duel) votes, leaving duelVotes on this turn
      await pages[keeper].getByRole('button', { name: new RegExp(`🏆 ${bidder}`) }).click();
      await H.waitForGame(roomCode, (g) => g.currentTurn?.duelVotes, { label: 'vote recorded' });
      await H.declareWinner(host, bidder);

      const reprisal = await H.waitForTurnPhase(roomCode, 'reprisal');
      const p = await H.playersByName(roomCode);
      expect(reprisal.reprisalPlayerId).toBe(p[opener].id);

      await pages[opener].locator('.action-option', { hasText: 'Kept Gift' }).click();
      const duel = await H.waitForGame(
        roomCode, (g) => (g.currentTurn?.phase === 'duel' && g.currentTurn.isReprisal ? g.currentTurn : false),
        { label: 'reprisal duel' });
      expect(duel.challengerId).toBe(p[opener].id);
      expect(duel.defenderId).toBe(p[keeper].id);
      expect(duel.duelVotes ?? null).toBeNull();
      expect(duel.tax ?? 0).toBe(0);
      expect(duel.bids ?? null).toBeNull();
      expect(duel.reprisalDepth).toBe(1);

      await H.declareWinner(host, keeper);
      await H.waitForTurnPhase(roomCode, null);
      const game = await H.assertInvariants(roomCode, 'after reprisal duel');
      // Rules: naughty +1 per duel. The reprisal had no auction, so 0 -> 1.
      expect(giftByName(game, 'Kept Gift').naughtyLevel).toBe(1);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  // Found by the agent harness: spectators' phones stayed on "X is choosing a target…"
  // for the whole reprisal duel, so nobody could follow or vote on it.
  test('spectators see (and can vote on) a reprisal duel', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const keeper = await keepRound(host, roomCode, pages, 'Kept Gift');
      const turn = await H.drawGift(host, roomCode, 'Contested Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const bidder = NAMES.find((n) => n !== opener && n !== keeper);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder);
      await H.waitForTurnPhase(roomCode, 'reprisal');
      await expect(pages[bidder].locator('#status-card')).toContainText(/choosing a target/i);

      await pages[opener].locator('.action-option', { hasText: 'Kept Gift' }).click();
      await H.waitForGame(roomCode, (g) => (g.currentTurn?.isReprisal ? g : false), { label: 'reprisal duel' });

      const spectator = pages[bidder];
      await expect(spectator.locator('#status-card')).toContainText(/Duel in Progress/i);
      await spectator.getByRole('button', { name: new RegExp(`🏆 ${keeper}`) }).click();
      await H.waitForGame(roomCode, (g) => (g.currentTurn?.duelVotes ? g : false), { label: 'spectator vote' });
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("host 'Choose Target for Them' reprisal closes the modal and keeps chain depth", async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const keeper = await keepRound(host, roomCode, pages, 'Kept Gift');
      const turn = await H.drawGift(host, roomCode, 'Contested Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const bidder = NAMES.find((n) => n !== opener && n !== keeper);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder);
      await H.waitForTurnPhase(roomCode, 'reprisal');

      await host.locator('#action-buttons summary').click();
      await host.getByRole('button', { name: /Choose Target for Them/ }).click();
      await host.locator('#modal-content .option-btn', { hasText: 'Kept Gift' }).click();
      const duel = await H.waitForGame(
        roomCode, (g) => (g.currentTurn?.phase === 'duel' && g.currentTurn.isReprisal ? g.currentTurn : false),
        { label: 'reprisal duel' });
      await expect(host.locator('#modal-overlay')).not.toHaveClass(/active/);
      expect(duel.reprisalDepth).toBe(1);
      // Naughty must only rise when the duel resolves, not when it starts
      expect(giftByName(await H.dbGet(roomCode), 'Kept Gift').naughtyLevel).toBe(0);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('a reprisal never targets the loser\'s own gift', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const keeper = await keepRound(host, roomCode, pages, 'Keeper Gift');
      const turn = await H.drawGift(host, roomCode, 'Swap Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      // The keeper (already holding a gift) wins the new one -> their old gift
      // goes to the opener. The only non-excluded gift is now the opener's own.
      await H.allBid(pages, opener, { [keeper]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, keeper);

      const state = await H.waitForGame(
        roomCode,
        (g) => (g.currentTurn == null ? { over: true } : g.currentTurn.phase === 'reprisal' ? g.currentTurn : false),
        { label: 'turn resolved or reprisal' });
      expect(state.over, 'no legal reprisal target exists, so none should be offered').toBe(true);
      await H.assertInvariants(roomCode, 'swap');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('yield duel: both challengers pay half their bid, rounded DOWN', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Yield Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [high, low] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [high]: 3, [low]: 1 });
      await H.revealBids(host);
      await H.openerYields(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'yield_pending');
      await H.clickStartYieldDuel(host);
      const duel = await H.waitForTurnPhase(roomCode, 'duel');
      const p = await H.playersByName(roomCode);
      expect(p[high].chips).toBe(10 - 1); // floor(3/2)
      expect(p[low].chips).toBe(10 - 0); // floor(1/2)
      expect(duel.pot + (duel.tax || 0)).toBe(1);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('a duel loser never sees the winner verdict (pot 1, defender dividend +1)', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Verdict Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 1 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder); // opener (defender) loses, gets +1 dividend
      const loserModal = pages[opener].locator('#duel-result-content');
      await expect(loserModal).toContainText('Ouch.', { timeout: 10000 });
      await expect(pages[bidder].locator('#duel-result-content')).toContainText('Spoils!', { timeout: 10000 });
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('admin "Move to Misfit" puts the gift where the Misfit Lottery looks', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const keeper = await keepRound(host, roomCode, pages, 'Admin Gift');
      await host.getByRole('button', { name: 'Admin', exact: true }).click();
      await host.locator('#admin-panel .card-title').click();
      const gift = giftByName(await H.dbGet(roomCode), 'Admin Gift');
      await host.locator('#admin-gift-select').selectOption(gift.id);
      await host.getByRole('button', { name: /Misfit Pile/ }).click();
      const after = await H.waitForGame(
        roomCode, (g) => (g.gifts[gift.id].ownerId !== (gift.ownerId) ? g : false), { label: 'gift moved' });
      expect(after.gifts[gift.id].ownerId).toBe('misfit');
      expect(after.players[(await H.playersByName(roomCode))[keeper].id].giftId ?? null).toBeNull();
      await H.assertInvariants(roomCode, 'admin misfit');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('a game saved with the retired Counterfeit Canvas setting runs a Sketch-Off (exactly once)', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      await H.dbPatch(roomCode, { 'settings/tiebreakerMethod': 'minigame' });
      const turn = await H.drawGift(host, roomCode, 'Legacy Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [c1, c2] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [c1]: 2, [c2]: 2 });
      await H.revealBids(host);
      await H.openerFights(pages[opener]);
      const mg = await H.waitForGame(roomCode, (g) => (g.minigame?.type ? g.minigame : false), { label: 'party game' });
      expect(mg.type).toBe('sketch_off');
      await host.waitForTimeout(1500);
      const log = await host.locator('#event-log').innerText();
      expect(log.match(/Sketch-Off! Everyone plays/g)?.length, 'starts exactly once').toBe(1);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('3-way tie, opener yields from the phone: the duelists are two of the TIED bidders', async ({ browser }) => {
    test.setTimeout(90000); // a whole party game is played
    const FOUR = ['Alice', 'Bob', 'Charlie', 'Dave'];
    // Sketch-Off: the outcome follows the votes the test casts (Whack-an-Elf phones report live scores)
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'tiebreaker-method': 'sketch_off' }, FOUR);
    try {
      const turn = await H.drawGift(host, roomCode, 'Triple Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const tied = FOUR.filter((n) => n !== opener);
      await H.allBid(pages, opener, Object.fromEntries(tied.map((n) => [n, 2])));
      await H.revealBids(host);
      await expect(pages[opener].locator('#status-card')).toContainText('3-WAY TIE');
      await H.openerYields(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'yield_pending');
      await H.clickStartYieldDuel(host); // 3+ tied: the whole table's party game picks the two
      const p = await H.playersByName(roomCode);
      const tiedIds = tied.map((n) => p[n].id);
      // the best two TIED bidders in the party game are the two who duel (votes: tiedIds[2] most, then tiedIds[0])
      const duel = await H.settlePartyTie(host, roomCode, [tiedIds[2], tiedIds[0], p[opener].id, tiedIds[1]]);
      expect(duel.isYieldDuel).toBe(true);
      expect([duel.challengerId, duel.defenderId].sort()).toEqual([tiedIds[2], tiedIds[0]].sort());
      const winner = Object.keys(p).find((n) => p[n].id === duel.challengerId);
      await H.declareWinner(host, winner);
      await H.waitForTurnPhase(roomCode, null);
      await H.assertInvariants(roomCode, '3-way yield');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('endgame auction: a rival bid keeps your slider; resolving the last misfit ends the game', async ({ browser }) => {
    test.setTimeout(60000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      // Craft the endgame: Alice & Bob hold gifts, one Misfit, Charlie giftless, nothing wrapped
      const p = await H.playersByName(roomCode);
      await H.dbPatch(roomCode, {
        'gifts/g_a': { name: 'Alice Gift', ownerId: p.Alice.id, naughtyLevel: 0, createdAt: 1 },
        'gifts/g_b': { name: 'Bob Gift', ownerId: p.Bob.id, naughtyLevel: 0, createdAt: 2 },
        'gifts/g_m': { name: 'Lonely Misfit', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1, createdAt: 3 },
        [`players/${p.Alice.id}/giftId`]: 'g_a',
        [`players/${p.Bob.id}/giftId`]: 'g_b',
      });
      // injected state can lag on the host under suite load — wait until it shows
      await expect(host.locator('#stat-wrapped')).toHaveText('0', { timeout: 15000 });
      await host.locator('#btn-endgame').click();
      const choice = await H.waitForTurnPhase(roomCode, 'endgame_choice');
      expect(choice.activePlayerId).toBe(p.Charlie.id);

      await pages.Charlie.getByRole('button', { name: /Path B/ }).click();
      await H.waitForTurnPhase(roomCode, 'endgame_bidding');
      const alicePanel = pages.Alice.locator('#bid-panel');
      await alicePanel.getByRole('button', { name: '2', exact: true }).click();
      await H.submitBid(pages.Bob, 1);
      await H.waitForGame(roomCode, (g) => g.currentTurn?.bids, { label: 'Bob bid' });
      await expect(pages.Alice.locator('#bid-amount-display')).toHaveText('2');
      await expect(host.locator('.sealed-count')).toContainText('of 2'); // Charlie doesn't bid
      await alicePanel.getByRole('button', { name: /Seal it/ }).click();

      await host.getByRole('button', { name: /Reveal Bids/ }).click();
      await host.getByRole('button', { name: /Duel Highest Bidder/ }).click();
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, 'Charlie');
      const ended = await H.waitForGame(roomCode, (g) => (g.phase === 'ended' ? g : false), { label: 'game ended' });
      expect(ended.gifts.g_m.ownerId).toBe(p.Charlie.id);
      await H.assertInvariants(roomCode, 'endgame done');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  // Found by the agent harness: revealing an all-pass Misfit auction gave the HOST a
  // "Choose Defender" modal — the active player never got to pick on their phone.
  test('endgame auction where everyone passes: the active player picks the defender on their phone', async ({ browser }) => {
    test.setTimeout(60000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const p = await H.playersByName(roomCode);
      await H.dbPatch(roomCode, {
        'gifts/g_a': { name: 'Alice Gift', ownerId: p.Alice.id, naughtyLevel: 0, createdAt: 1 },
        'gifts/g_b': { name: 'Bob Gift', ownerId: p.Bob.id, naughtyLevel: 0, createdAt: 2 },
        'gifts/g_m': { name: 'Lonely Misfit', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1, createdAt: 3 },
        [`players/${p.Alice.id}/giftId`]: 'g_a',
        [`players/${p.Bob.id}/giftId`]: 'g_b',
      });
      await expect(host.locator('#stat-wrapped')).toHaveText('0', { timeout: 15000 });
      await host.locator('#btn-endgame').click();
      await H.waitForTurnPhase(roomCode, 'endgame_choice');
      await pages.Charlie.getByRole('button', { name: /Path B/ }).click();
      await H.waitForTurnPhase(roomCode, 'endgame_bidding');
      await H.submitBid(pages.Alice, 0);
      await H.submitBid(pages.Bob, 0);
      await H.waitForGame(roomCode, (g) => Object.keys(g.currentTurn?.bids || {}).length === 2, { label: 'both passed' });

      await host.getByRole('button', { name: /Reveal Bids/ }).click();
      await H.waitForTurnPhase(roomCode, 'endgame_nobids');
      await expect(pages.Charlie.locator(`[data-testid="nobids-defender-${p.Alice.id}"], [onclick="executeNoBidsDuel('${p.Alice.id}')"]`)).toBeVisible();
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  // Found by the agent harness: getMinCost used `baseCost || 1`, so "0 + Naughty" cost 1 + Naughty.
  test('Base Challenge Cost "0 + Naughty" makes a Naughty-2 gift cost 2 on host and phones', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'base-cost': '0' });
    try {
      // Bob and Charlie hold gifts, so the draw lands on Alice; nobody bids → Alice Gambits
      const p = await H.playersByName(roomCode);
      await H.dbPatch(roomCode, {
        'gifts/gB': { name: 'Naughty Two', ownerId: p.Bob.id, naughtyLevel: 2, createdAt: 1 },
        'gifts/gC': { name: 'Plain One', ownerId: p.Charlie.id, naughtyLevel: 0, createdAt: 2 },
        [`players/${p.Bob.id}/giftId`]: 'gB', [`players/${p.Charlie.id}/giftId`]: 'gC',
      });
      const turn = await H.drawGift(host, roomCode, 'Third Gift');
      expect(turn.openerId).toBe(p.Alice.id);
      await H.allBid(pages, 'Alice', {});
      await H.revealBids(host);
      await pages.Alice.getByRole('button', { name: /Grinch's Gambit/ }).click();
      const card = pages.Alice.locator('#status-card');
      await expect(card.locator('.action-option', { hasText: 'Naughty Two' })).toContainText('2 chips');
      await expect(card.locator('.action-option', { hasText: 'Plain One' })).toContainText('0 chip');
      if (await host.evaluate(() => typeof GAME_CONFIG !== 'undefined')) { // the original host page
        expect(await host.evaluate(() => GAME_CONFIG.getMinCost(2))).toBe(2);
      }
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("the phone's bidding guide states this game's tax and dividend settings", async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, {
      'max-payout': '3', 'tax-threshold': '4', 'loser-dividend': 'defenders',
    });
    try {
      const turn = await H.drawGift(host, roomCode, 'Guide Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const bidder = NAMES.find((n) => n !== opener);
      const guide = pages[bidder].locator('#bid-help-text');
      await expect(guide).toContainText('pots of 4+ pay the winner at most 3');
      await expect(guide).toContainText("No Loser's Dividend for attackers");
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  // Found by the agent harness (agents/): a double-tap on "X wins" ran recordDuelWinner
  // twice — the winner was paid the pot twice.
  test('double-tapping the duel winner button resolves the duel once', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Double Tap');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');

      const btn = host.locator('#duel-modal-content').getByRole('button', { name: new RegExp(`${bidder} wins`, 'i') });
      await btn.waitFor({ state: 'visible' });
      await btn.dblclick();
      await H.waitForGame(roomCode, (g) => (g.currentTurn?.phase !== 'duel' ? g : false), { label: 'duel resolved' });
      await host.waitForTimeout(1500); // let any second resolution land

      const p = await H.playersByName(roomCode);
      expect(p[bidder].chips, 'winner: paid 2, won the 2-chip pot once').toBe(10);
      expect(p[opener].chips, 'defender: one +1 dividend').toBe(11);
      expect(p[bidder].duelWins).toBe(1);
      expect(p[opener].duelLosses).toBe(1);
      expect(giftByName(await H.dbGet(roomCode), 'Double Tap').naughtyLevel, 'first duel sets naughty to the top bid, once').toBe(2);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('double-tapping Start Duel charges the challenger once', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Double Start');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 3 });
      await H.revealBids(host);
      await host.getByRole('button', { name: /Start Duel$|Start Duel!/i }).first().dblclick();
      await H.waitForTurnPhase(roomCode, 'duel');
      await host.waitForTimeout(1500);
      expect((await H.playersByName(roomCode))[bidder].chips, 'bid of 3 paid once').toBe(7);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('double-tapping Start Yield Duel charges each challenger once', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Double Yield');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [high, low] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [high]: 4, [low]: 2 });
      await H.revealBids(host);
      await H.openerYields(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'yield_pending');
      await host.getByRole('button', { name: /Start Yield Duel/i }).dblclick();
      await H.waitForTurnPhase(roomCode, 'duel');
      await host.waitForTimeout(1500);
      const p = await H.playersByName(roomCode);
      expect(p[high].chips, 'half of 4 paid once').toBe(8);
      expect(p[low].chips, 'half of 2 paid once').toBe(9);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('won chips fly from the loser to the winner (host) and the verdicts say who from/to (phones)', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      await host.evaluate(() => {
        window.__flights = [];
        document.addEventListener('jb:chips', (e) => window.__flights.push(e.detail));
      });
      const turn = await H.drawGift(host, roomCode, 'Flight Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, opener, { keepVerdict: true }); // defender wins the 2-chip pot

      const p = await H.playersByName(roomCode);
      // drawn: the chips really animate on screen (they used to fly between small tiles below the
      // fold — and the old check passed even when nothing was drawn)
      const flights = (x) => x.filter((f) => f.from === p[bidder].id && f.to === p[opener].id && f.count === 2);
      await expect.poll(async () => flights(await host.evaluate(() => window.__flights)).map((f) => f.drawn)).toEqual([2]);
      // …as part of the TV verdict: the winner, what they won, their chips counting up
      const verdict = host.locator('#duel-verdict');
      await expect(verdict).toBeVisible();
      await expect(verdict.locator('.verdict-name')).toHaveText(opener);
      await expect(verdict.locator('.verdict-outcome')).toHaveText('keeps the Flight Gift');
      await expect(verdict.getByTestId('verdict-winner-chips')).toHaveText('12');
      await expect(pages[opener].locator('#verdict-party')).toHaveText(`from ${bidder}`);
      await expect(pages[bidder].locator('#verdict-party')).toHaveText(`to ${opener}`);
      await expect(pages[opener].locator('#verdict-chips')).toHaveText('12'); // 10 + pot 2, after the chips land
      await expect(verdict).toBeHidden({ timeout: 10000 }); // and it clears itself
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  // Found by play-testing as host + one phone: the host's "X wins" was live the moment the duel
  // started, so the duel was over before the table could vote.
  test('duel voting: the host cannot call the duel until every spectator has voted', async ({ browser }) => {
    const names = ['Alice', 'Bob', 'Charlie', 'Dana'];
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'vote-window': '60' }, names);
    try {
      const turn = await H.drawGift(host, roomCode, 'Vote Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder, ...spectators] = names.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 1 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      const p = await H.playersByName(roomCode);

      const gate = host.getByTestId('vote-gate');
      const winBtn = host.getByTestId(`modal-winner-${p[opener].id}`);
      await expect(gate).toContainText('0/2');
      await expect(winBtn).toBeDisabled();

      await pages[spectators[0]].getByTestId(`vote-${p[opener].id}`).click();
      await expect(gate).toContainText('1/2');
      await expect(winBtn).toBeDisabled(); // one still deciding

      await pages[spectators[1]].getByTestId(`vote-${p[bidder].id}`).click();
      await expect(gate).toContainText('Votes are in');
      await expect(winBtn).toBeEnabled();
      await H.declareWinner(host, opener, { keepVerdict: true });
      await expect(host.getByTestId('verdict-callers')).toHaveText(`Called it: ${spectators[0]}`);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('duel voting: the host can close voting early; the phone then stops taking votes', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'vote-window': '60' });
    try {
      const turn = await H.drawGift(host, roomCode, 'Close Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder, spectator] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 1 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      const p = await H.playersByName(roomCode);
      await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toBeDisabled();
      await host.getByTestId('close-voting').click();
      await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toBeEnabled();
      await expect(pages[spectator].getByTestId(`vote-${p[opener].id}`)).toHaveCount(0);
      await expect(pages[spectator].locator('#status-card')).toContainText('Voting closed');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('mascots: Santa opens the game, greets the opener, the reindeer calls a Reprisal — and none block clicks', async ({ browser }) => {
    test.setTimeout(60000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const santa = host.locator('[data-mascot="santa"]');
      await expect(santa).toContainText(saysOneOf('start'));
      expect(await santa.evaluate((el) => getComputedStyle(el).pointerEvents)).toBe('none');

      const keeper = await keepRound(host, roomCode, pages, 'Kept Gift'); // clicks work with mascots up
      const turn = await H.drawGift(host, roomCode, 'Contested Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      await expect(pages[opener].locator('[data-mascot="santa"]')).toContainText(saysOneOf('phoneOpener'));
      const bidder = NAMES.find((n) => n !== opener && n !== keeper);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder);
      await H.waitForTurnPhase(roomCode, 'reprisal');
      await Promise.all([ // each is only up for a few seconds
        expect(host.locator('[data-mascot="reindeer"]')).toContainText(saysOneOf('reprisal', opener), { timeout: 15000 }),
        expect(pages[opener].locator('[data-mascot="reindeer"]')).toContainText(saysOneOf('phoneReprisal'), { timeout: 15000 }),
      ]);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('?mascots=0 keeps the mascots away', async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const host = await ctx.newPage();
    await host.goto(`${H.BASE}/host.html?mascots=0`);
    await host.getByRole('button', { name: /Create New Game/i }).click();
    await host.waitForSelector('#game-screen', { state: 'visible' });
    const roomCode = (await host.locator('#room-code-display').textContent()).trim();
    const { contexts } = await H.joinPlayers(browser, roomCode, ['Alice', 'Bob']);
    try {
      await H.startGame(host, roomCode); // Santa would say "Ho ho ho" here
      await host.waitForTimeout(2500);
      await expect(host.locator('[data-mascot]')).toHaveCount(0);
    } finally {
      await ctx.close();
      for (const c of contexts) await c.close();
    }
  });

  // --- Party mini-games: the whole table plays; ranking settles the tie + prizes --------

  /** Two tied bidders, opener fights → the party game starts. Returns names. */
  async function tieIntoPartyGame(host, roomCode, pages, giftName) {
    const turn = await H.drawGift(host, roomCode, giftName);
    const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
    const [t1, t2] = NAMES.filter((n) => n !== opener);
    await H.allBid(pages, opener, { [t1]: 2, [t2]: 2 });
    await H.revealBids(host);
    await H.openerFights(pages[opener]);
    await H.waitForGame(roomCode, (g) => (g.currentTurn?.phase === 'minigame_party' && g.minigame?.phase === 'intro' ? g : false), { label: 'party intro' });
    return { opener, t1, t2 };
  }
  const skipAhead = (host) => host.locator('[data-testid="skip-party-phase"], #action-buttons button[onclick="skipPartyPhase()"]').click();
  const partyPhase = (roomCode, phase) =>
    H.waitForGame(roomCode, (g) => (g.minigame?.phase === phase ? g.minigame : false), { label: `party ${phase}`, timeout: 20000 });

  async function whack(page, ms, { avoidGrinch = true } = {}) {
    const end = Date.now() + ms;
    while (Date.now() < end) {
      const sel = avoidGrinch ? '#party-card .whack-cell.up:not([data-kind="grinch"])' : '#party-card .whack-cell.up';
      const cell = page.locator(sel).first();
      if (await cell.count()) await cell.dispatchEvent('pointerdown').catch(() => {});
      await page.waitForTimeout(60);
    }
  }

  test('Whack-an-Elf: everyone plays; the better tied bidder wins the tie, the best other player gets the Elf Bonus', async ({ browser }) => {
    test.setTimeout(90000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'tiebreaker-method': 'whack_elf' });
    try {
      const { opener, t1, t2 } = await tieIntoPartyGame(host, roomCode, pages, 'Whack Gift');
      for (const n of NAMES) await expect(pages[n].locator('#party-card')).toContainText('Whack-an-Elf');
      await skipAhead(host);
      await partyPhase(roomCode, 'playing');
      await expect(pages[t1].locator('#party-card .whack-grid')).toBeVisible();
      await Promise.all([whack(pages[t1], 6000), whack(pages[opener], 2500)]); // t2 never taps
      await host.waitForTimeout(1200); // live scores land
      const scores = (await H.dbGet(roomCode, 'minigame/scores')) || {};
      const p = await H.playersByName(roomCode);
      expect(scores[p[t1].id]?.score).toBeGreaterThan(scores[p[opener].id]?.score ?? 0);
      expect(scores[p[opener].id]?.score).toBeGreaterThan(0);

      await skipAhead(host);
      const mg = await partyPhase(roomCode, 'results');
      expect(mg.results.winners).toEqual([p[t1].id]);
      expect(mg.results.bonus).toBe(p[opener].id);
      await expect(pages[t2].locator('#party-card')).toContainText('wins the tie');
      await expect(host.locator('#bids-grid')).toContainText('Elf Bonus');

      await skipAhead(host); // continue to the duel
      const duel = await H.waitForTurnPhase(roomCode, 'duel');
      expect(duel.challengerId).toBe(p[t1].id);
      expect(duel.defenderId).toBe(p[opener].id);
      const after = await H.playersByName(roomCode);
      expect(after[opener].chips, 'Elf Bonus +1 chip').toBe(11);
      expect(after[t1].chips, 'tie winner paid the bid of 2').toBe(8);
      expect(after[t1].partyPoints).toBe(3);
      expect(after[opener].partyPoints).toBe(2);
      // the TV's overall leaderboard has it live, before the game ends
      await expect(host.locator('#card-party-score .f6-row').first()).toContainText(t1);
      await expect(host.locator(`#card-party-score [data-player-id="${p[opener].id}"]`)).toContainText('2 party pts');
      expect(after[t2].partyPoints ?? 0).toBe(0);
      expect(await H.dbGet(roomCode, 'minigame')).toBeNull();
      await expect(pages[t1].locator('#party-card')).toBeHidden();
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  async function scribble(page, dx) {
    const pad = page.locator('#party-card canvas.pg-pad');
    await pad.waitFor({ state: 'visible' });
    const b = await pad.boundingBox();
    await page.mouse.move(b.x + 30, b.y + 30);
    await page.mouse.down();
    for (let i = 1; i <= 8; i++) await page.mouse.move(b.x + 30 + i * dx, b.y + 30 + i * 12);
    await page.mouse.up();
    await page.locator('#party-card [data-act="done"]').click();
  }

  test('Sketch-Off: everyone draws and votes (never for themselves); most votes among the tied wins', async ({ browser }) => {
    test.setTimeout(90000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'tiebreaker-method': 'sketch_off' });
    try {
      const { opener, t1, t2 } = await tieIntoPartyGame(host, roomCode, pages, 'Sketch Gift');
      const mg0 = await H.dbGet(roomCode, 'minigame');
      await expect(pages[opener].locator('#party-card')).toContainText(mg0.prompt);
      await skipAhead(host);
      await partyPhase(roomCode, 'playing');
      for (const [i, n] of NAMES.entries()) await scribble(pages[n], 10 + i * 5);
      await partyPhase(roomCode, 'voting'); // everyone handed in → moves on by itself

      const p = await H.playersByName(roomCode);
      const vote = async (voter, target) => {
        const card = pages[voter].locator('#party-card');
        await expect(card.locator('figure.mine')).toHaveCount(1); // own drawing isn't votable
        await card.locator(`[data-vote="${p[target].id}"]`).click();
      };
      await vote(opener, t2);
      await vote(t1, t2);
      await vote(t2, t1);
      const mg = await partyPhase(roomCode, 'results'); // everyone voted → moves on by itself
      expect(mg.results.scoreOf).toEqual({ [p[t2].id]: 2, [p[t1].id]: 1, [p[opener].id]: 0 });
      expect(mg.results.winners).toEqual([p[t2].id]);
      expect(mg.results.bonus ?? null, 'the opener got no votes → no Elf Bonus').toBeNull();

      await skipAhead(host);
      const duel = await H.waitForTurnPhase(roomCode, 'duel');
      expect(duel.challengerId).toBe(p[t2].id);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('the final tally shows each Party Score (party points + correct calls + duels fought), in that order, and crowns the Party MVP', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const p = await H.playersByName(roomCode);
      await H.dbPatch(roomCode, {
        [`players/${p.Bob.id}/partyPoints`]: 3, [`players/${p.Bob.id}/votingWins`]: 1,
        [`players/${p.Alice.id}/votingWins`]: 2, [`players/${p.Alice.id}/votingLosses`]: 4,
        phase: 'ended',
      });
      await expect(host.locator('.tally-grid .f6-row', { hasText: 'Bob' })).toContainText('4 party score');
      await expect(host.locator('.tally-grid .f6-row', { hasText: 'Alice' })).toContainText('2 party score');
      await expect(host.locator('.tally-grid .f6-row', { hasText: 'Charlie' })).not.toContainText('party score');
      await expect(host.locator('#bids-grid')).toContainText('Party MVP: Bob');
      // the standings are by Party Score, not chips: a player who kept every chip and did nothing is last
      await H.dbPatch(roomCode, { [`players/${p.Charlie.id}/chips`]: 50, [`players/${p.Alice.id}/duelsFought`]: 3 });
      await expect(host.locator('.tally-grid .f6-row').first()).toContainText('Alice');          // 2 calls + 3 duels fought
      await expect(host.locator('.tally-grid .f6-row', { hasText: 'Alice' })).toContainText('5 party score');
      await expect(host.locator('.tally-grid .f6-row').last()).toContainText('Charlie');
      await expect(host.locator('.tally-grid .f6-row').last()).toContainText('50 chips left');
      await expect(host.locator('#bids-grid')).toContainText('Party MVP: Alice');
      // each phone's place is its Party Score place too
      await expect(pages.Alice.getByTestId('final-place')).toHaveText('1st place.', { timeout: 10000 });
      await expect(pages.Charlie.getByTestId('final-place')).toHaveText('3rd place.');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('the extra prize: the TV reveals the Party MVP, the MVP\'s phone celebrates, the others are told', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const p = await H.playersByName(roomCode);
      await H.dbPatch(roomCode, { [`players/${p.Bob.id}/partyPoints`]: 3, [`players/${p.Bob.id}/votingWins`]: 1, phase: 'ended' });
      const tv = host.locator('#mvp-reveal');
      await expect(tv).toBeVisible();
      await expect(tv.getByTestId('mvp-name')).toHaveText('Bob', { timeout: 6000 }); // after the roll lands
      await expect(tv).toContainText('3 party-game points');
      await expect(tv).toContainText('1 correct duel calls');
      await expect(pages.Bob.locator('#mvp-reveal')).toContainText("You're the Party MVP", { timeout: 6000 });
      await expect(pages.Alice.locator('#mvp-reveal')).toContainText('Cheer for Bob!', { timeout: 6000 });
      await tv.click(); // on to the standings
      await expect(tv).toBeHidden();
      await expect(host.locator('.tally-grid')).toBeVisible();
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('the host shows the betting leaderboard once a duel vote is settled', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      await expect(host.locator('#card-betting')).toBeHidden(); // no votes yet
      const turn = await H.drawGift(host, roomCode, 'Vote Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder, spectator] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 1 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await pages[spectator].getByRole('button', { name: new RegExp(`🏆 ${bidder}`) }).click();
      await H.waitForGame(roomCode, (g) => g.currentTurn?.duelVotes, { label: 'vote recorded' });
      await H.declareWinner(host, bidder);

      const board = host.locator('#card-betting');
      await expect(board).toBeVisible();
      await expect(board).toContainText(spectator);
      expect((await H.playersByName(roomCode))[spectator].votingWins).toBe(1);
      expect((await H.playersByName(roomCode))[spectator].chips, 'bets are off by default').toBe(10);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("spectator bets (experiment): a right call wins the wrong callers' chips", async ({ browser }) => {
    const names = ['Alice', 'Bob', 'Charlie', 'Dana'];
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'spectator-bets': '1' }, names);
    try {
      const turn = await H.drawGift(host, roomCode, 'Bet Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder, right, wrong] = names.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 1 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await expect(pages[right].locator('#status-card')).toContainText(/bet 1 chip/i);
      await pages[right].getByRole('button', { name: new RegExp(`🏆 ${bidder}`) }).click();
      await pages[wrong].getByRole('button', { name: new RegExp(`🏆 ${opener}`) }).click();
      await H.waitForGame(roomCode, (g) => Object.keys(g.currentTurn?.duelVotes || {}).length === 2, { label: 'both voted' });
      await H.declareWinner(host, bidder);
      await H.waitForGame(roomCode, (g) => (g.lastDuel?.bets ? g : false), { label: 'bets settled' });

      const p = await H.playersByName(roomCode);
      expect(p[right].chips).toBe(11);
      expect(p[wrong].chips).toBe(9);
      expect(await H.dbGet(roomCode, 'lastDuel/bets')).toEqual({ pool: 1, toBank: 0, winners: 1 });
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('a dropped first connection is retried, so Join still works instead of hanging', async ({ browser }) => {
    const { hostCtx, roomCode } = await H.createGame(browser);
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const phone = await ctx.newPage();
      const errors = [];
      phone.on('pageerror', (e) => errors.push(e.message));
      // A busy server / flaky Wi-Fi: the phone's first connection to the room drops at once
      let tries = 0;
      await phone.routeWebSocket(/\/ws$/, (ws) => { if (tries++ === 0) ws.close(); else ws.connectToServer(); });
      await phone.goto(`${H.PLAYER_BASE}/player.html?room=${roomCode}&testId=7&name=Dave`);
      await phone.getByRole('button', { name: /Join Game/i }).click();
      await expect(phone.locator('#game-screen')).toBeVisible({ timeout: 10000 });
      await expect(phone.locator('#display-name')).toHaveText('Dave');
      expect(tries).toBeGreaterThan(1);
      expect(errors).toEqual([]);
    } finally {
      await hostCtx.close();
      await ctx.close();
    }
  });

  test("a player named O'Brien can rejoin after a refresh", async ({ browser }) => {
    const { hostCtx, roomCode } = await H.createGame(browser);
    const { pages, contexts, errors } = await H.joinPlayers(browser, roomCode, ["O'Brien"]);
    try {
      const phone = pages["O'Brien"];
      await phone.reload();
      await phone.getByRole('button', { name: /Rejoin Game/ }).click();
      await expect(phone.locator('#game-screen')).toBeVisible();
      await expect(phone.locator('#display-name')).toHaveText("O'Brien");
      expect(errors).toEqual([]);
    } finally {
      await hostCtx.close();
      for (const c of contexts) await c.close();
    }
  });
});
