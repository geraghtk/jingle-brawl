// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Multi-user flow: Host creates game, players join, host starts.
 * Each player runs in an isolated browser context (separate storage).
 */
test.describe('Lobby and Join', () => {
  const BASE = process.env.BASE_URL || 'http://localhost:8080';
  const PLAYER_NAMES = ['Alice', 'Bob', 'Charlie'];

  test('host creates game, 3 players join, host starts game', async ({ browser }) => {
    // --- Host: Create game ---
    const hostContext = await browser.newContext();
    const hostPage = await hostContext.newPage();

    await hostPage.goto(`${BASE}/host.html`);
    await expect(hostPage.locator('#setup-screen')).toBeVisible();

    // Click Create New Game (defaults: random tiebreaker, no AI critic)
    await hostPage.getByRole('button', { name: /Create New Game/i }).click();

    // Wait for game screen and room code
    await expect(hostPage.locator('#game-screen')).toBeVisible();
    const roomCode = (await hostPage.locator('#room-code-display').textContent()).trim();
    expect(roomCode).toMatch(/^[A-Z0-9]{4}$/);

    // --- Players: Join in parallel ---
    const playerPages = await Promise.all(
      PLAYER_NAMES.map(async (name, i) => {
        const ctx = await browser.newContext();
        const page = await ctx.newPage();
        await page.goto(
          `${BASE}/player.html?room=${roomCode}&testId=${i}&name=${name}`
        );
        return { page, ctx, name };
      })
    );

    // Each player: room & name should be pre-filled, click Join
    for (const { page, name } of playerPages) {
      await expect(page.locator('#room-code-input')).toHaveValue(roomCode);
      await expect(page.locator('#player-name-input')).toHaveValue(name);
      await page.getByRole('button', { name: /Join Game/i }).click();
    }

    // All players should see game screen
    for (const { page } of playerPages) {
      await expect(page.locator('#game-screen')).toBeVisible();
      await expect(page.locator('#status-card')).toContainText(/Waiting|ready/i);
    }

    // Host should see 3 players, Start Game enabled
    await expect(hostPage.locator('#connected-count')).toHaveText('3');
    await expect(hostPage.getByRole('button', { name: /Start Game/i })).toBeEnabled();

    // Host starts game
    await hostPage.getByRole('button', { name: /Start Game/i }).click();

    // Players should see updated status (main game - "in the bag" or "drawn")
    for (const { page } of playerPages) {
      await expect(page.locator('#status-card')).toContainText(/bag|drawn|gift|Draw|Welcome/i, {
        timeout: 8000,
      });
    }

    // Host should see draw controls (Draw from Santa's Sack or similar)
    await expect(hostPage.getByRole('button', { name: /Draw.*Santa|Santa.*Sack/i })).toBeVisible({
      timeout: 5000,
    });

    await hostContext.close();
    for (const { ctx } of playerPages) await ctx.close();
  });
});
