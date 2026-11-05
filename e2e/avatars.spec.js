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

test('the VS screens show both duellists by their avatars: the TV and every phone', async ({ browser }) => {
  const names = ['Alice', 'Bob', 'Cleo'];
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, names);
  try {
    await H.startGame(host, roomCode);
    const turn = await H.drawGift(host, roomCode, 'VS Gift');
    const p = await H.playersByName(roomCode);
    const opener = H.openerNameOf(turn, p);
    const [bidder, spectator] = names.filter((n) => n !== opener);
    const av = (n) => p[n].avatar;
    await H.allBid(pages, opener, { [bidder]: 2, [spectator]: 1 });
    await H.revealBids(host);
    await H.openerFights(pages[opener]);
    await H.waitForTurnPhase(roomCode, 'duel_pending');
    const pending = host.locator('.duel-pending-row .f6-avatar');
    await expect(pending).toHaveCount(2);
    await expect(pending.first()).toHaveAttribute('data-avatar', av(bidder));
    await expect(pending.last()).toHaveAttribute('data-avatar', av(opener));
    await H.clickStartDuel(host);
    await H.waitForTurnPhase(roomCode, 'duel');
    await expect(host.locator(`.duel-corner.red .f6-avatar[data-avatar="${av(bidder)}"]`)).toBeVisible();
    await expect(host.locator(`.duel-corner.green .f6-avatar[data-avatar="${av(opener)}"]`)).toBeVisible();
    // the duellists' phones: YOU (your avatar) vs THEM (theirs)
    const mine = pages[bidder].locator('.duel-you-row .f6-avatar');
    await expect(mine).toHaveCount(2);
    await expect(mine.first()).toHaveAttribute('data-avatar', av(bidder));
    await expect(mine.last()).toHaveAttribute('data-avatar', av(opener));
    // a spectator calling it sees both corners by their avatars
    await expect(pages[spectator].locator(`.callit-corner.red .f6-avatar[data-avatar="${av(bidder)}"]`)).toBeVisible();
    await expect(pages[spectator].locator(`.callit-corner.green .f6-avatar[data-avatar="${av(opener)}"]`)).toBeVisible();
  } finally {
    await hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
