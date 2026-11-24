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
      await expect(tv.locator('.turn-panel, .host-strip')).toHaveCount(0);   // its own stage, not the console's cards
      await expect(tv.locator('#turn-panel button')).toHaveCount(0);
      // the host's audited Peek: amounts on the console only, and the log says so
      const ids = await H.playersByName(roomCode);
      await expect(host.getByTestId(`sealed-${ids[bidder].id}`)).toHaveText('✓ Sealed');
      await host.getByTestId('peek-bids').click();
      await host.getByTestId('dialog-confirm').click();
      await expect(host.getByTestId(`sealed-${ids[bidder].id}`)).toHaveText('👀 3');
      await expect(tv.getByTestId(`sealed-${ids[bidder].id}`)).toHaveText('✓ Sealed');
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

// The TV's own stage for the main game (host/TvStage.tsx): one picture that never scrolls, with nothing
// leaving it, in every phase, on a 720p TV with a long name and a long gift name at the table.
test('the TV is one picture that never scrolls: lobby, sealed bids, reveal, duel and endgame at 1280x720', async ({ browser }) => {
  test.setTimeout(90000);
  const LONG = 'Bartholomew-Alexander';
  const names = ['Alice', 'Bob', LONG, 'Charlotte'];
  const { host, hostCtx, roomCode, errors } = await H.createGame(browser);
  const tvCtx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const tvErrors = [];
  try {
    const tv = await tvCtx.newPage();
    tv.on('pageerror', (e) => tvErrors.push(e.message));
    await tv.goto(`${H.BASE}/tv.html?room=${roomCode}`);
    /** The page does not scroll, the stage's picture is the whole screen, and nothing in it (or in the duel over it) leaves the screen. */
    const fits = async (label) => {
      await tv.waitForTimeout(700);                                   // the scene has arrived
      const m = await tv.evaluate(() => {
        const out = [];
        const off = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && (r.right > innerWidth + 1 || r.bottom > innerHeight + 1 || r.left < -1 || r.top < -1); };
        document.querySelectorAll('#tv-stage .tvs-pic *, #duel-modal-overlay.active *').forEach((e) => { if (off(e)) out.push(String(e.className)); });
        // a name set to fit never spills out of its own box sideways
        document.querySelectorAll('#tv-stage .tvs-fit').forEach((e) => { if (e.scrollWidth > e.clientWidth + 2) out.push(`fit: ${e.textContent}`); });
        const pic = document.querySelector('#tv-stage .tvs-pic').getBoundingClientRect();
        return { scrollH: document.documentElement.scrollHeight, scrollW: document.documentElement.scrollWidth, h: innerHeight, w: innerWidth, pic: [Math.round(pic.width), Math.round(pic.height)], out };
      });
      expect(m.scrollH, `${label}: no vertical scroll`).toBeLessThanOrEqual(m.h);
      expect(m.scrollW, `${label}: no horizontal scroll`).toBeLessThanOrEqual(m.w);
      expect(m.pic, `${label}: the picture is the screen`).toEqual([1280, 720]);
      expect(m.out, `${label}: nothing leaves the picture`).toEqual([]);
      await expect(tv.locator('#tv-stage button:not([data-testid=music-toggle])'), `${label}: no controls`).toHaveCount(0);
    };

    await expect(tv.getByTestId('tv-stage')).toHaveAttribute('data-scene', 'lobby');
    const { pages, contexts } = await H.joinPlayers(browser, roomCode, names);
    try {
      await expect(tv.locator('#lobby-roster-list .roster-row')).toHaveCount(4);
      await expect(tv.locator('#lobby-roster-list')).toContainText(LONG);
      await fits('lobby');

      await H.startGame(host, roomCode);
      await expect(tv.getByTestId('tv-stage')).toHaveAttribute('data-scene', 'between');
      await expect(tv.locator('#players-grid [data-player-id]')).toHaveCount(4);   // everyone on the rail
      await fits('between turns');

      const gift = 'An Enormous Inflatable Reindeer Lawn';
      const turn = await H.drawGift(host, roomCode, gift);
      const ids = await H.playersByName(roomCode);
      const opener = H.openerNameOf(turn, ids);
      const bidders = names.filter((n) => n !== opener);
      await expect(tv.getByTestId('tv-stage')).toHaveAttribute('data-scene', 'bidding');
      await expect(tv.getByTestId('tv-gift-name')).toHaveText(gift);
      await expect(tv.getByTestId('tv-sealed-count')).toHaveText('0of 3');
      await H.submitBid(pages[bidders[0]], 3);
      await expect(tv.getByTestId('tv-sealed-count')).toHaveText('1of 3');
      await expect(tv.getByTestId(`sealed-${ids[bidders[0]].id}`)).toHaveText('✓ Sealed');   // in, never how much
      await expect(tv.getByTestId(`sealed-${ids[bidders[1]].id}`)).toHaveCount(0);
      await fits('sealed bids');
      await H.submitBid(pages[bidders[1]], 2);
      await H.submitBid(pages[bidders[2]], 0);

      await H.revealBids(host);
      await expect(tv.getByTestId('tv-stage')).toHaveAttribute('data-scene', 'reveal');
      await expect(tv.locator('.tvs-col.shown')).toHaveCount(2, { timeout: 8000 });   // the bars rise one at a time; the pass has none
      await expect(tv.locator('.tvs-outcome')).toContainText(`Highest bid: ${bidders[0]}`);
      await fits('reveal');

      await H.openerFights(pages[opener]);
      await H.waitForTurnPhase(roomCode, 'duel_pending');
      await expect(tv.getByTestId('tv-stage')).toHaveAttribute('data-scene', 'duel_pending');
      await fits('duel pending');
      await H.clickStartDuel(host);
      await H.waitForTurnPhase(roomCode, 'duel');
      await expect(tv.locator('#duel-modal-overlay.active')).toBeVisible();
      // the duel is the whole picture (once it has opened: it comes in a touch smaller)
      await expect.poll(async () => { const box = await tv.locator('#duel-modal-content').boundingBox(); return [Math.round(box.width), Math.round(box.height)]; }).toEqual([1280, 720]);
      await expect(tv.locator('#duel-modal-overlay button')).toHaveCount(0);
      await fits('duel');
      await H.declareWinner(host, bidders[0], { keepVerdict: true });
      await expect(tv.locator('#duel-verdict')).toBeVisible();
      const overflow = await tv.locator('#duel-verdict .verdict-name').evaluate((e) => e.scrollWidth > e.clientWidth + 2 || e.getBoundingClientRect().right > innerWidth);
      expect(overflow).toBe(false);                                   // the winner fits the verdict, however long the name
      await tv.locator('#duel-verdict').click();

      // the endgame: a Misfit Lottery turn (written into the room: getting there by play takes the whole night)
      await H.dbPatch(roomCode, { phase: 'endgame', lastDuel: null, 'gifts/mf1': { name: 'A Suspiciously Heavy Fruitcake', ownerId: 'misfit', naughtyLevel: 1, misfitOrder: 1 },
        currentTurn: { phase: 'endgame_choice', activePlayerId: ids[LONG].id, misfitId: 'mf1', misfitName: 'A Suspiciously Heavy Fruitcake' } });
      await expect(tv.getByTestId('tv-stage')).toHaveAttribute('data-scene', 'endgame_choice');
      await expect(tv.locator('#turn-panel')).toContainText('A Suspiciously Heavy Fruitcake');
      await fits('endgame');
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

test('the host plays too, from one phone: their game and the console are two tabs', async ({ browser }) => {
  test.setTimeout(90000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const contexts = [];
  try {
    const phone = await ctx.newPage();
    const errors = [];
    phone.on('pageerror', (e) => errors.push(e.message));
    await phone.goto(`${H.BASE}/player.html`);
    await phone.getByTestId('host-a-game').click();                  // from the join screen: host a game
    await phone.getByRole('button', { name: /Create New Game/i }).click();
    await phone.waitForSelector('#game-screen');
    const room = (await phone.locator('#strip-room').textContent()).trim();
    await phone.getByTestId('join-as-player').click();               // …and take a seat
    await expect(phone.locator('#room-code-input')).toHaveValue(room);
    await phone.locator('#player-name-input').fill('Hostie');
    await phone.getByRole('button', { name: /Join Game/i }).click();
    await expect(phone.locator('#display-name')).toHaveText('Hostie');

    // two tabs: my game (open) and the host's console, already running behind it
    await expect(phone.getByTestId('tab-game')).toHaveAttribute('aria-selected', 'true');
    await expect(phone.getByTestId('host-waiting')).toHaveCount(0);  // nothing to do with one player
    const console_ = phone.frameLocator('[data-testid=host-frame]');
    await expect(console_.locator('#strip-room')).toHaveText(room);
    await expect(console_.getByTestId('to-player')).toHaveCount(0);  // no "My player screen" link inside the tab

    // a second player arrives: the game waits on the host (Start), and the Host tab says so
    const other = await browser.newContext({ viewport: { width: 390, height: 844 } });
    contexts.push(other);
    const bo = await other.newPage();
    await bo.goto(`${H.BASE}/player.html?room=${room}&testId=7&name=Bo`);
    await bo.getByRole('button', { name: /Join Game/i }).click();
    await expect(phone.getByTestId('host-waiting')).toBeVisible();

    // the Host tab: the console, right here; Start from it
    await phone.getByTestId('tab-host').click();
    await expect(phone.getByTestId('tab-host')).toHaveAttribute('aria-selected', 'true');
    await expect(phone.locator('#display-name')).toBeHidden();
    await console_.getByRole('button', { name: /Start Game/i }).click();
    await H.waitForGame(room, (g) => g.phase === 'main', { label: 'started from the Host tab' });

    // back to my game: the same page, no reload, the game moved on
    await phone.getByTestId('tab-game').click();
    await expect(phone.locator('#display-name')).toHaveText('Hostie');
    await expect(phone.locator('#players-card')).toBeVisible();      // the table row shows once the game is on
    await expect(phone.getByTestId('host-waiting')).toBeVisible();   // next: the host draws a gift

    // a reload keeps the tab you were on
    await phone.getByTestId('tab-host').click();
    await phone.goto(`${H.BASE}/player.html?rejoin=1`);
    await expect(phone.getByTestId('tab-host')).toHaveAttribute('aria-selected', 'true');

    // the console on its own still links back to the player screen
    const solo = await ctx.newPage();
    await solo.goto(`${H.BASE}/host.html?autoLoad=1`);
    await solo.getByTestId('to-player').click();
    await expect(solo.getByTestId('tab-game')).toHaveAttribute('aria-selected', 'true');

    const players = await H.dbGet(room, 'players');
    expect(Object.values(players).map((p) => p.name).sort()).toEqual(['Bo', 'Hostie']);
    expect(errors).toEqual([]);
  } finally {
    await ctx.close();
    for (const c of contexts) await c.close();
  }
});
