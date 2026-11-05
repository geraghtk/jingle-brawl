// @ts-check
// Avatars: a player picks one when they join (web/src/core/avatars.ts); the TV shows it. Without a
// pick they get one nobody at the table has; the picker greys out the taken ones. Every tile is a
// sticker picture (web/public/art/avatars).
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

test.skip(/:8080/.test(H.BASE), 'React app only (not the original pages)');

test('a player picks an avatar when they join and the TV shows it; no pick gets a free one; taken ones are greyed out', async ({ browser }) => {
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  try {
    const phone = await ctx.newPage();
    await phone.goto(`${H.PLAYER_BASE}/player.html?room=${roomCode}&testId=1&name=Alice`);
    await phone.getByRole('button', { name: 'Sloth' }).click();
    await expect(phone.getByRole('button', { name: 'Sloth' })).toHaveAttribute('aria-pressed', 'true');
    await phone.getByRole('button', { name: /Join Game/i }).click();
    await phone.waitForSelector('#game-screen', { state: 'visible' });
    const { contexts } = await H.joinPlayers(browser, roomCode, ['Bob']);
    try {
      const p = await H.playersByName(roomCode);
      expect(p.Alice.avatar).toBe('sloth');
      expect(p.Bob.avatar).toBeTruthy();
      expect(p.Bob.avatar).not.toBe('sloth');
      const alice = host.locator('.roster-row', { hasText: 'Alice' }).locator('.f6-avatar');
      await expect(alice).toHaveAttribute('data-avatar', 'sloth');
      const bob = host.locator('.roster-row', { hasText: 'Bob' }).locator('.f6-avatar img');
      for (const img of [alice.locator('img'), bob]) {
        await expect.poll(() => img.evaluate((e) => /** @type {HTMLImageElement} */ (e).naturalWidth)).toBeGreaterThan(0);
      }
      // a third phone: Alice's and Bob's avatars are taken
      const third = await ctx.newPage();
      await third.goto(`${H.PLAYER_BASE}/player.html?room=${roomCode}&testId=3&name=Cleo`);
      await expect(third.locator('[data-avatar="sloth"]')).toBeDisabled();
      await expect(third.locator(`[data-avatar="${p.Bob.avatar}"]`)).toBeDisabled();
      await expect(third.locator('.avatar-opt:not([disabled])')).toHaveCount(22);
    } finally {
      for (const c of contexts) await c.close();
    }
  } finally {
    await ctx.close();
    await hostCtx.close();
  }
});
