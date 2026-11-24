// @ts-check
// The host console asks in its own dialogs (web/src/host/Dialogs.tsx), never a native prompt / confirm /
// alert box, and the setup screen keeps its settings in groups with Create always in reach.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

/** Any native dialog on this page fails the test (and is dismissed, so the page is not left hanging). */
function noNativeDialogs(page, who) {
  const seen = [];
  page.on('dialog', (d) => { seen.push(`${who}: ${d.type()} "${d.message()}"`); d.dismiss().catch(() => {}); });
  return seen;
}

test.describe('The console has its own dialogs', () => {
  test('a whole turn opens no native dialog; the gift is named with Enter, Cancel draws nobody, End Game is a danger confirm', async ({ browser }) => {
    const { host, hostCtx, roomCode, errors } = await H.createGame(browser);
    const native = noNativeDialogs(host, 'HOST');
    const { pages, contexts } = await H.joinPlayers(browser, roomCode, NAMES);
    try {
      await H.startGame(host, roomCode);
      const draw = host.getByTestId('draw');
      const dialog = host.getByTestId('gift-dialog');
      const box = host.getByTestId('gift-name');

      // the dialog: who was drawn (with their avatar), the box has the focus, an empty name cannot be sent
      await draw.click();
      await expect(dialog).toBeVisible();
      await expect(host.getByTestId('gift-drawn')).toContainText(/(Alice|Bob|Charlie) was drawn/);
      await expect(dialog.locator('.f6-avatar')).toBeVisible();
      await expect(box).toBeFocused();
      await expect(host.getByTestId('gift-submit')).toBeDisabled();
      await box.press('Enter');
      await box.fill('   ');
      await expect(host.getByTestId('gift-submit')).toBeDisabled();
      await box.press('Enter');
      await expect(dialog).toBeVisible();
      expect((await H.dbGet(roomCode)).currentTurn ?? null).toBeNull();
      expect(Number(await box.getAttribute('maxlength'))).toBeLessThanOrEqual(60);

      // Cancel, and Escape: nobody is drawn, no gift is made
      await host.getByTestId('gift-cancel').click();
      await expect(dialog).toHaveCount(0);
      await draw.click();
      await expect(box).toBeFocused();
      await expect(box).toHaveValue('');                              // nothing left over from the last time
      await box.fill('Never opened');
      await host.keyboard.press('Escape');
      await expect(dialog).toHaveCount(0);
      const before = await H.dbGet(roomCode);
      expect(before.currentTurn ?? null).toBeNull();
      expect(Object.keys(before.gifts || {})).toHaveLength(0);

      // typed, then Enter: the turn starts with the player the dialog named
      await draw.click();
      const named = (await host.getByTestId('gift-drawn').textContent()).replace(/ was drawn.*/, '').trim();
      await box.fill('Fuzzy socks');
      await box.press('Enter');
      await expect(dialog).toHaveCount(0);
      const turn = await H.waitForTurnPhase(roomCode, 'bidding');
      const players = await H.playersByName(roomCode);
      expect(H.openerNameOf(turn, players)).toBe(named);
      expect(Object.values((await H.dbGet(roomCode)).gifts).map((g) => g.name)).toEqual(['Fuzzy socks']);

      // the rest of the turn, with a Peek on the way (a plain confirm)
      await H.allBid(pages, named, {});
      await host.getByTestId('peek-bids').click();
      await expect(host.getByTestId('dialog')).toContainText(/Peek at the sealed bids/);
      await host.getByTestId('dialog-confirm').click();
      await expect(host.getByTestId('peek-bids')).toHaveCount(0);
      await H.revealBids(host);
      await H.waitForTurnPhase(roomCode, 'reveal');
      await H.openerKeeps(pages[named]);
      await H.waitForTurnPhase(roomCode, null);

      // a danger confirm: marked as one, Cancel has the focus, and cancelling changes nothing
      await host.getByTestId('end-game').click();
      const confirm = host.getByTestId('dialog');
      await expect(confirm).toHaveClass(/dlg-danger/);
      await expect(host.getByTestId('dialog-confirm')).toHaveClass(/btn-danger/);
      await expect(host.getByTestId('dialog-cancel')).toBeFocused();
      await host.getByTestId('dialog-cancel').click();
      await expect(confirm).toHaveCount(0);
      expect((await H.dbGet(roomCode)).phase).toBe('main');
      await host.getByTestId('end-game').click();
      await host.keyboard.press('Escape');
      await expect(confirm).toHaveCount(0);
      expect((await H.dbGet(roomCode)).phase).toBe('main');
      await host.getByTestId('end-game').click();
      await host.getByTestId('dialog-confirm').click();
      await H.waitForGame(roomCode, (g) => g.phase === 'ended', { label: 'ended' });

      expect(native, `native dialogs:\n${native.join('\n')}`).toEqual([]);
      expect(errors, `page errors:\n${errors.join('\n')}`).toEqual([]);
    } finally {
      await hostCtx.close();
      for (const c of contexts) await c.close();
    }
  });

  test('Host Admin: Undo with nothing to undo says so in a notice, Reset Turn is a danger confirm', async ({ browser }) => {
    const { host, hostCtx, roomCode } = await H.createGame(browser);
    const native = noNativeDialogs(host, 'HOST');
    const { contexts } = await H.joinPlayers(browser, roomCode, NAMES);
    try {
      await H.startGame(host, roomCode);
      await host.locator('.sidebar-tab-btn[data-tab="admin"]').click();
      await host.locator('#admin-panel .card-title').click();
      await host.getByTestId('admin-undo').click();
      await expect(host.getByTestId('dialog')).toContainText(/Nothing to undo/);
      await host.getByTestId('dialog-ok').click();
      await expect(host.getByTestId('dialog')).toHaveCount(0);

      await host.locator('.sidebar-tab-btn[data-tab="game"]').click();
      await H.drawGift(host, roomCode, 'Mystery mug');
      await host.locator('.sidebar-tab-btn[data-tab="admin"]').click();
      await host.getByTestId('admin-reset-turn').click();
      await expect(host.getByTestId('dialog')).toHaveClass(/dlg-danger/);
      await host.getByTestId('dialog-cancel').click();
      expect((await H.dbGet(roomCode)).currentTurn?.phase).toBe('bidding');
      await host.getByTestId('admin-reset-turn').click();
      await host.getByTestId('dialog-confirm').click();
      await H.waitForTurnPhase(roomCode, null);
      expect(native, `native dialogs:\n${native.join('\n')}`).toEqual([]);
    } finally {
      await hostCtx.close();
      for (const c of contexts) await c.close();
    }
  });

  test('in the Host tab of the host\'s phone the gift dialog fits the screen', async ({ browser }) => {
    const { host, hostCtx, roomCode } = await H.createGame(browser);
    const { contexts } = await H.joinPlayers(browser, roomCode, NAMES);
    try {
      await H.startGame(host, roomCode);
      await host.setViewportSize({ width: 390, height: 844 });
      await host.getByTestId('draw').click();
      const dialog = host.getByTestId('gift-dialog');
      await expect(dialog).toBeVisible();
      await expect(host.getByTestId('gift-name')).toBeFocused();
      const box = await dialog.boundingBox();
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(390);
      expect(box.y + box.height).toBeLessThan(844 / 2);              // in the top half: a keyboard does not cover it
      for (const id of ['gift-cancel', 'gift-submit']) await expect(host.getByTestId(id)).toBeInViewport();
    } finally {
      await hostCtx.close();
      for (const c of contexts) await c.close();
    }
  });
});

test.describe('Setup: the settings in groups', () => {
  for (const size of [{ width: 1366, height: 768 }, { width: 390, height: 844 }]) {
    test(`at ${size.width}x${size.height} the groups are closed with a summary each, and Create is in reach with any of them open`, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: size });
      const host = await ctx.newPage();
      try {
        await host.goto(`${H.BASE}/host.html`);
        const create = host.getByTestId('create-game');
        const groups = ['chips', 'duels', 'winnings', 'minigames', 'ties'];
        await expect(create).toBeInViewport({ ratio: 1 });
        for (const g of groups) {
          await expect(host.getByTestId(`setup-group-${g}`)).toHaveAttribute('aria-expanded', 'false');
          await expect(host.getByTestId(`setup-summary-${g}`)).not.toBeEmpty();
        }
        await expect(host.locator('#setting-max-bid')).toHaveCount(0);
        // closed, the whole screen fits a laptop without scrolling
        if (size.width > 1000) expect(await host.evaluate(() => document.documentElement.scrollHeight <= window.innerHeight)).toBe(true);
        expect(await host.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

        // a summary follows the choice
        await host.getByTestId('setup-group-chips').click();
        await host.locator('#setting-max-bid').selectOption('5');
        await host.locator('#setting-starting-chips').selectOption('15');
        await host.getByTestId('setup-group-chips').click();
        await expect(host.getByTestId('setup-summary-chips')).toContainText('15 chips each');
        await expect(host.getByTestId('setup-summary-chips')).toContainText('bids up to 5');

        // opening one closes the other; Create stays on the screen wherever the page is scrolled
        for (const g of groups) {
          await host.getByTestId(`setup-group-${g}`).click();
          await expect(host.getByTestId(`setup-group-${g}`)).toHaveAttribute('aria-expanded', 'true');
          await expect(host.locator('.sg.open')).toHaveCount(1);
          await host.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
          await expect(create).toBeInViewport({ ratio: 1 });
          await host.evaluate(() => window.scrollTo(0, 0));
          await expect(create).toBeInViewport({ ratio: 1 });
          expect(await host.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
        }

        // every setting is still there
        await host.getByTestId('setup-open-all').click();
        for (const id of ['starting-chips', 'max-bid', 'base-cost', 'public-chips', 'duel-pick', 'vote-window', 'reprisal-depth', 'spectator-bets', 'duel-games',
          'max-payout', 'tax-threshold', 'loser-dividend', 'duel-table-games', 'ready-check', 'practice-round', 'pace', 'tiebreaker-method', 'party-prize',
          'duel-app-whack_elf', 'duel-app-sketch_off', 'duel-app-draw_off', 'duel-app-pass_the_gremlin', 'duel-app-reindeer_ring_toss', 'duel-app-gift_conveyor',
          'duel-app-nutcracker_knockout', 'duel-app-mecha_santa', 'duel-app-elf_tower_wars']) {
          await expect(host.locator(`#setting-${id}`), id).toBeVisible();
        }
        await host.locator('#setting-duel-app-pass_the_gremlin').check();
        await expect(host.locator('#setting-gremlin-challenges')).toBeVisible();
        await host.locator('#setting-duel-app-draw_off').check();
        await expect(host.locator('#setting-ai-critic-backend')).toBeVisible();
        await expect(host.locator('#setting-ai-critic-key')).toBeVisible();
        // one line per game
        const rows = host.locator('.sg-game');
        for (let i = 0; i < await rows.count(); i++) expect((await rows.nth(i).boundingBox()).height).toBeLessThanOrEqual(size.width > 600 ? 46 : 76);

        await create.click();
        await host.waitForSelector('#game-screen', { state: 'visible' });
        const room = (await host.locator('#room-code-display').textContent()).trim();
        const settings = await H.dbGet(room, 'settings');
        expect(settings.maxBid).toBe(5);
        expect(settings.startingChips).toBe(15);
        expect(settings.duelAppGames).toContain('draw_off');
      } finally {
        await ctx.close();
      }
    });
  }

  test('untouched, Create makes the same game as before the groups (the defaults)', async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1366, height: 768 } });
    const host = await ctx.newPage();
    try {
      await host.goto(`${H.BASE}/host.html`);
      await host.getByTestId('create-game').click();
      await host.waitForSelector('#game-screen', { state: 'visible' });
      const room = (await host.locator('#room-code-display').textContent()).trim();
      const s = await H.dbGet(room, 'settings');
      expect(s).toMatchObject({
        maxReprisalDepth: 1, maxBid: 4, maxWinnerPayout: 2, loserDividendRule: 'defenders', spectatorBets: 0, voteWindow: 30,
        readyCheck: true, practiceRound: true, pace: 'normal', duelPick: 'defender', duelTableGames: 'teams', taxThreshold: 2, baseCost: 1,
        tiebreakerMethod: 'party_mix', partyPrize: 'points_chips',
      });
      expect(s.startingChips ?? null).toBeNull();
      expect(!!s.publicChips).toBe(false);
    } finally {
      await ctx.close();
    }
  });
});
