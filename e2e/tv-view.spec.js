// @ts-check
// The shared screen (tv.html?room=CODE) on its own device: it follows the game, shows the join QR,
// and has no controls and no secrets; the host runs the game from their own device.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

test('the TV follows the game with no controls: lobby QR, sealed bids as "in", the duel with no winner buttons', async ({ browser }) => {
  const { host, hostCtx, roomCode, errors } = await H.createGame(browser);
  const tvCtx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const tvErrors = [];
  try {
    const tv = await tvCtx.newPage();
    tv.on('pageerror', (e) => tvErrors.push(e.message));
    // the host console links the TV screen
    await expect(host.getByTestId('tv-link')).toContainText(`tv.html?room=${roomCode}`);
    await tv.goto(`${H.BASE}/tv.html?room=${roomCode}`);
    await expect(tv.locator('#room-code-display')).toHaveText(roomCode);
    await expect(tv.locator('#qr-code svg')).toBeVisible();
    await expect(tv.locator('#btn-start-game')).toHaveCount(0);       // the host starts it from their device
    await expect(tv.locator('.sidebar')).toHaveCount(0);

    const { pages, contexts } = await H.joinPlayers(browser, roomCode, NAMES);
    try {
      await expect(tv.locator('#roster-count')).toHaveText('3 joined');
      await H.startGame(host, roomCode);
      const turn = await H.drawGift(host, roomCode, 'TV Gift');
      const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
      const [bidder] = NAMES.filter((n) => n !== opener);
      await H.allBid(pages, opener, { [bidder]: 3 });
      await expect(tv.locator('#turn-panel')).toContainText('TV Gift');
      await expect(tv.locator('#action-buttons')).toHaveCount(0);
      await expect(tv.locator('#turn-panel button')).toHaveCount(0);
      await H.revealBids(host);
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await expect(tv.locator('#duel-modal-overlay.active')).toBeVisible();
      await expect(tv.locator('#duel-modal-overlay button')).toHaveCount(0);   // no "X wins" on the TV
      await H.declareWinner(host, bidder);
      await expect(tv.locator('#duel-verdict')).toBeVisible();
      expect([...errors, ...tvErrors]).toEqual([]);
    } finally {
      for (const c of contexts) await c.close();
    }
  } finally {
    await tvCtx.close();
    await hostCtx.close();
  }
});

test('the TV without a room code asks for one', async ({ page }) => {
  await page.goto(`${H.BASE}/tv.html`);
  await expect(page.getByTestId('tv-room-form')).toBeVisible();
});
