// @ts-check
// Regression tests for bugs found in the 2026-10 full code review.
//
// Each test drives a real host + phones on the RTDB emulator and asserts the
// behaviour the rules (Jingle Brawl V2.md) or the UI promise. They were written
// to FAIL against the pre-fix code, then the fixes were made.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

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

  test('Counterfeit Canvas: "Done Drawing" passes the turn to the next artist', async ({ browser }) => {
    test.setTimeout(90000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, {
      'tiebreaker-method': 'minigame',
      'minigame-timer': 'unlimited',
    });
    try {
      const turn = await H.drawGift(host, roomCode, 'Canvas Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [c1, c2] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [c1]: 2, [c2]: 2 });
      await H.revealBids(host);
      await H.openerFights(pages[opener]);

      const mg = await H.waitForGame(roomCode, (g) => (g.minigame?.phase === 'drawing' && g.minigame.currentPlayerId ? g.minigame : false),
        { label: 'drawing phase', timeout: 35000 });
      const p = await H.playersByName(roomCode);
      const nameById = Object.fromEntries(Object.entries(p).map(([n, v]) => [v.id, n]));
      const first = nameById[mg.currentPlayerId];
      const watcher = pages[NAMES.find((n) => n !== first)];

      // The artist draws; another phone mirrors it — and mirrors an Undo too
      const artist = pages[first];
      const box = await artist.locator('#drawing-canvas').boundingBox();
      await artist.mouse.move(box.x + 30, box.y + 30);
      await artist.mouse.down();
      for (let i = 1; i <= 5; i++) await artist.mouse.move(box.x + 30 + i * 12, box.y + 30);
      await artist.mouse.up();
      await expect.poll(() => watcher.evaluate('drawingStrokes.length')).toBeGreaterThan(0);
      const seen = await watcher.evaluate('drawingStrokes.length');
      await artist.getByRole('button', { name: /Undo Last Stroke/ }).click();
      await expect.poll(() => watcher.evaluate('drawingStrokes.length')).toBe(seen - 1);

      await artist.locator('#drawing-done-btn').click();

      const next = await H.waitForGame(roomCode, (g) => (g.minigame?.currentTurnIndex === 1 && g.minigame.currentPlayerId !== mg.currentPlayerId ? g.minigame : false),
        { label: 'turn passed to next artist', timeout: 8000 });
      expect(next.currentPlayerId).toBe(mg.playerOrder[1]);

      const log = await host.locator('#event-log').innerText();
      expect(log.match(/Starting The Counterfeit Canvas/g)?.length, 'minigame must start exactly once').toBe(1);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test("Draw Off: one artist submitting doesn't wipe another artist's canvas", async ({ browser }) => {
    test.setTimeout(90000);
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, {
      'tiebreaker-method': 'draw_off',
    });
    try {
      const turn = await H.drawGift(host, roomCode, 'Draw Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [c1, c2] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [c1]: 2, [c2]: 2 });
      await H.revealBids(host);
      await H.openerFights(pages[opener]);
      await H.waitForGame(roomCode, (g) => (g.minigame?.phase === 'drawing' ? g : false),
        { label: 'draw off drawing', timeout: 30000 });

      const artist = pages[c1];
      const canvas = artist.locator('#drawing-canvas');
      await expect(canvas).toBeVisible();
      const box = await canvas.boundingBox();
      await artist.mouse.move(box.x + 20, box.y + 20);
      await artist.mouse.down();
      for (let i = 1; i <= 10; i++) await artist.mouse.move(box.x + 20 + i * 10, box.y + 20 + i * 5);
      await artist.mouse.up();
      const strokesBefore = await artist.evaluate('drawingStrokes.length');
      expect(strokesBefore).toBeGreaterThan(0);

      await pages[c2].getByRole('button', { name: /Submit Drawing/ }).click();
      await H.waitForGame(roomCode, (g) => (g.minigame?.submissions && Object.keys(g.minigame.submissions).length === 1 ? g : false),
        { label: 'one submission' });
      await artist.waitForTimeout(1000);
      expect(await artist.evaluate('drawingStrokes.length')).toBe(strokesBefore);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('3-way tie, opener yields from the phone: the duelists are two of the TIED bidders', async ({ browser }) => {
    const FOUR = ['Alice', 'Bob', 'Charlie', 'Dave'];
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, {}, FOUR);
    try {
      const turn = await H.drawGift(host, roomCode, 'Triple Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const tied = FOUR.filter((n) => n !== opener);
      await H.allBid(pages, opener, Object.fromEntries(tied.map((n) => [n, 2])));
      await H.revealBids(host);
      await expect(pages[opener].locator('#status-card')).toContainText('3-WAY TIE');
      await H.openerYields(pages[opener]);
      const pending = await H.waitForTurnPhase(roomCode, 'yield_pending');
      const p = await H.playersByName(roomCode);
      const tiedIds = tied.map((n) => p[n].id);
      expect(tiedIds).toContain(pending.yieldC1);
      expect(tiedIds).toContain(pending.yieldC2);
      expect(pending.yieldC1).not.toBe(pending.yieldC2);

      await H.clickStartYieldDuel(host);
      const duel = await H.waitForTurnPhase(roomCode, 'duel');
      const winner = Object.keys(p).find((n) => p[n].id === duel.challengerId);
      await H.declareWinner(host, winner);
      await H.waitForTurnPhase(roomCode, null);
      await H.assertInvariants(roomCode, '3-way yield');
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('Draw Off picking 2 for a yield ends in the yield duel (no endless re-run) and hides the card', async ({ browser }) => {
    test.setTimeout(60000);
    const FOUR = ['Alice', 'Bob', 'Charlie', 'Dave'];
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'tiebreaker-method': 'draw_off' }, FOUR);
    try {
      const turn = await H.drawGift(host, roomCode, 'Loop Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const tied = FOUR.filter((n) => n !== opener);
      await H.allBid(pages, opener, Object.fromEntries(tied.map((n) => [n, 2])));
      await H.revealBids(host);
      await H.openerYields(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'yield_pending');
      await H.clickStartYieldDuel(host);
      const mg = await H.waitForGame(roomCode, (g) => (g.minigame?.type === 'draw_off' ? g.minigame : false), { label: 'draw off started' });
      expect(mg.minigameGoal).toBe('pick_two');
      expect(mg.participants.length).toBe(3);
      await expect(pages[tied[0]].locator('#minigame-card')).toBeVisible();

      // Skip the 3-minute drawing + AI judging: jump straight to results
      const p = await H.playersByName(roomCode);
      const [w1, w2] = [p[tied[0]].id, p[tied[1]].id];
      await H.dbPatch(roomCode, { 'minigame/phase': 'results', 'minigame/winners': [w1, w2], 'minigame/rankings': [w1, w2, p[tied[2]].id] });
      await host.getByRole('button', { name: /Continue to Duel/ }).click();

      const duel = await H.waitForTurnPhase(roomCode, 'duel');
      expect(duel.isYieldDuel).toBe(true);
      expect([duel.challengerId, duel.defenderId].sort()).toEqual([w1, w2].sort());
      await host.waitForTimeout(1500);
      expect(await H.dbGet(roomCode, 'minigame'), 'a second Draw Off must not start').toBeNull();
      await expect(pages[tied[0]].locator('#minigame-card')).toBeHidden();
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
      await expect(pages.Charlie.locator(`[onclick="executeNoBidsDuel('${p.Alice.id}')"]`)).toBeVisible();
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  // Found by the agent harness: getMinCost used `baseCost || 1`, so "0 + Naughty" cost 1 + Naughty.
  test('Base Challenge Cost "0 + Naughty" makes a Naughty-2 gift cost 2 on host and phones', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser, { 'base-cost': '0' });
    try {
      await expect.poll(() => host.evaluate(() => GAME_CONFIG.getMinCost(2))).toBe(2);
      await expect.poll(() => pages.Alice.evaluate(() => GAME_CONFIG.getMinCost(2))).toBe(2);
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

  test('double-tapping the tiebreak winner starts the main duel once', async ({ browser }) => {
    const { host, roomCode, pages, cleanup, assertNoPageErrors } = await setup(browser);
    try {
      const turn = await H.drawGift(host, roomCode, 'Double Tiebreak');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [c1, c2] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [c1]: 2, [c2]: 2 });
      await H.revealBids(host);
      await H.openerFights(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'tiebreaker_pending');
      await H.clickStartTiebreakerDuel(host);
      const tb = await H.waitForGame(roomCode, (g) => (g.currentTurn?.isTiebreak ? g.currentTurn : false), { label: 'tiebreak duel' });
      const p0 = await H.playersByName(roomCode);
      const winner = tb.challengerId === p0[c1].id ? c1 : c2;
      await host.locator('#action-buttons').getByRole('button', { name: new RegExp(`${winner} Wins`, 'i') }).dblclick();
      await H.waitForGame(roomCode, (g) => (g.currentTurn?.phase === 'duel' && !g.currentTurn.isTiebreak ? g : false), { label: 'main duel' });
      await host.waitForTimeout(1500);
      expect((await H.playersByName(roomCode))[winner].chips, 'tiebreak winner pays their bid of 2 once').toBe(8);
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
        const real = window.flyChips;
        window.flyChips = (o) => {
          const who = (el) => el?.dataset?.duelCorner || el?.dataset?.playerId || null;
          window.__flights.push({ from: who(o.from), to: who(o.to), count: o.count });
          return real(o);
        };
      });
      const turn = await H.drawGift(host, roomCode, 'Flight Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, opener); // defender wins the 2-chip pot

      const p = await H.playersByName(roomCode);
      await expect.poll(() => host.evaluate(() => window.__flights)).toContainEqual({ from: p[bidder].id, to: p[opener].id, count: 2 });
      await expect(pages[opener].locator('#verdict-party')).toHaveText(`from ${bidder}`);
      await expect(pages[bidder].locator('#verdict-party')).toHaveText(`to ${opener}`);
      await expect(pages[opener].locator('#verdict-chips')).toHaveText('12'); // 10 + pot 2, after the chips land
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
      await expect(santa).toContainText(/Ho ho ho/);
      expect(await santa.evaluate((el) => getComputedStyle(el).pointerEvents)).toBe('none');

      const keeper = await keepRound(host, roomCode, pages, 'Kept Gift'); // clicks work with mascots up
      const turn = await H.drawGift(host, roomCode, 'Contested Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      await expect(pages[opener].locator('[data-mascot="santa"]')).toContainText(/You're up/);
      const bidder = NAMES.find((n) => n !== opener && n !== keeper);
      await H.allBid(pages, opener, { [bidder]: 2 });
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await H.declareWinner(host, bidder);
      await H.waitForTurnPhase(roomCode, 'reprisal');
      await Promise.all([ // each is only up for a few seconds
        expect(host.locator('[data-mascot="reindeer"]')).toContainText(/Reindeer kick back/, { timeout: 15000 }),
        expect(pages[opener].locator('[data-mascot="reindeer"]')).toContainText(/Kick back/i, { timeout: 15000 }),
      ]);
      assertNoPageErrors();
    } finally {
      await cleanup();
    }
  });

  test('?mascots=0 keeps the mascots away', async ({ page }) => {
    await page.goto(`${H.BASE}/rules.html?mascots=0`);
    await page.addScriptTag({ url: `${H.BASE}/mascots.js` });
    await page.evaluate(() => Mascots.say('santa', 'hello'));
    await page.waitForTimeout(800);
    await expect(page.locator('[data-mascot]')).toHaveCount(0);
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
