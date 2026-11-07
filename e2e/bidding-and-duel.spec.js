// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Full turn: Host draws player, players bid, host reveals, duel.
 * Uses ?testId= for isolated player sessions.
 */
test.describe('Bidding and Duel', () => {
  const BASE = require('./helpers').BASE;
  const PLAYER_BASE = process.env.PLAYER_BASE || BASE;
  const PLAYER_NAMES = ['Alice', 'Bob', 'Charlie'];
  const GIFT_NAME = 'Test Gift';

  test('full bidding flow: draw, bid, reveal, duel', async ({ browser }) => {
    // --- Host: Create game ---
    const hostContext = await browser.newContext();
    const hostPage = await hostContext.newPage();

    await hostPage.goto(`${BASE}/host.html`);
    await hostPage.getByRole('button', { name: /Create New Game/i }).click();

    await expect(hostPage.locator('#game-screen')).toBeVisible();
    const roomCode = (await hostPage.locator('#room-code-display').textContent()).trim();
    expect(roomCode).toMatch(/^[A-Z0-9]{4}$/);

    // --- Players: Join ---
    const playerPages = await Promise.all(
      PLAYER_NAMES.map(async (name, i) => {
        const ctx = await browser.newContext();
        const page = await ctx.newPage();
        await page.goto(
          `${PLAYER_BASE}/player.html?room=${roomCode}&testId=${i}&name=${name}`
        );
        await page.getByRole('button', { name: /Join Game/i }).click();
        await expect(page.locator('#game-screen')).toBeVisible();
        return { page, ctx, name };
      })
    );

    // Host starts game
    await hostPage.getByRole('button', { name: /Start Game/i }).click();

    // Host: Draw next player (will prompt for gift name)
    hostPage.once('dialog', (dialog) => dialog.accept(GIFT_NAME));
    await hostPage.getByRole('button', { name: /Draw.*Santa|Santa.*Sack/i }).click();

    // Wait for bidding phase
    await expect(hostPage.locator('#turn-panel')).toContainText(/BIDDING|Bidding/i, {
      timeout: 5000,
    });

    // Players see bid panel - Alice (opener) might not bid, Bob and Charlie bid
    // We don't know who the opener is from the test - check bid panel visibility
    const alicePage = playerPages[0].page;
    const bobPage = playerPages[1].page;
    const charliePage = playerPages[2].page;

    // Wait for the bidding phase to reach each player: either the bid panel
    // (can bid) or the status card (opener / in-the-bag) becomes visible. Both
    // elements are always in the DOM, so filter to the shown one to avoid a
    // strict-mode violation.
    await expect(alicePage.locator('#bid-panel:visible, #status-card:visible').first()).toBeVisible({ timeout: 5000 });
    await expect(bobPage.locator('#bid-panel:visible, #status-card:visible').first()).toBeVisible({ timeout: 5000 });
    await expect(charliePage.locator('#bid-panel:visible, #status-card:visible').first()).toBeVisible({ timeout: 5000 });

    // Bob bids 2 chips (click the "2" quick button if visible)
    const bobBidPanel = bobPage.locator('#bid-panel');
    if (await bobBidPanel.isVisible()) {
      await bobBidPanel.getByRole('button', { name: '2' }).click();
      await bobBidPanel.getByRole('button', { name: /Seal it|Lock In Bid|Lock/i }).click();
    }

    // Charlie passes (or bids 0)
    const charlieBidPanel = charliePage.locator('#bid-panel');
    if (await charlieBidPanel.isVisible()) {
      await charlieBidPanel.getByRole('button', { name: 'Pass' }).click();
      await charlieBidPanel.getByRole('button', { name: /Seal it|Lock In Bid|Lock/i }).click();
    }

    // If Alice has bid panel, she can pass too
    const aliceBidPanel = alicePage.locator('#bid-panel');
    if (await aliceBidPanel.isVisible()) {
      await aliceBidPanel.getByRole('button', { name: 'Pass' }).click();
      await aliceBidPanel.getByRole('button', { name: /Seal it|Lock In Bid|Lock/i }).click();
    }

    // Host reveals bids
    await hostPage.getByRole('button', { name: /Reveal All Bids/i }).click();

    // Wait for reveal/duel phase
    await expect(hostPage.locator('#turn-panel')).toContainText(
      /Reveal|Duel|Bids|TIE|challenger/i,
      { timeout: 5000 }
    );

    // If there was a bidder, we might need opener to choose Fight/Yield or duel to run
    // The duel can be triggered automatically or by host - depending on tiebreaker
    // For random tiebreaker with single high bidder, duel starts automatically
    const turnPanelText = await hostPage.locator('#turn-panel').textContent();
    if (turnPanelText.includes('Waiting') || turnPanelText.includes('opener')) {
      // Opener needs to act - use manual override if available
      const keepBtn = hostPage.getByRole('button', { name: /Opener Keeps|Keep/i });
      const gambitBtn = hostPage.getByRole('button', { name: /Gambit/i });
      if (await keepBtn.isVisible()) {
        await keepBtn.click();
      } else if (await gambitBtn.isVisible()) {
        // Skip gambit for test simplicity - would need more flow
      }
    }

    await hostContext.close();
    for (const { ctx } of playerPages) await ctx.close();
  });
});
