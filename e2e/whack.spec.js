// @ts-check
// Whack-an-Elf, the show: the TV and the console go full screen with the board and a live race; the
// phones count in 3, 2, 1; a phone's board fills its screen and shows the score, the place and the
// roll (three hits in a row: a point more each, until an elf gets away or the Grump is tapped).
const { test, expect } = require('@playwright/test');
const H = require('./helpers');
const NAMES = ['Alice', 'Bob', 'Charlie', 'Dana'];
const shot = async (page, name) => { if (process.env.WH_SHOTS) await page.screenshot({ path: `${process.env.WH_SHOTS}/${name}.png` }); };

/** Tap every pop-up that is up on this phone (`grump`: the Grump too), as thumbs would. */
const tapUp = (page, grump = false) => page.evaluate((g) => {
  let n = 0;
  for (const c of document.querySelectorAll(g ? '#party-card .whack-cell.up' : '#party-card .whack-cell.up:not([data-kind="grump"])')) { c.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })); n++; }
  return n;
}, grump);

test('Whack-an-Elf: full screen on the TV with a live race; phones count in, fill the screen, show score, place and the roll', async ({ browser }) => {
  test.setTimeout(150000);
  const r = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-whack_elf': 'on' });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    const tvErrors = [];
    tv.on('pageerror', (e) => tvErrors.push(String(e)));
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    await H.startGame(r.host, r.roomCode);
    const turn = await H.drawGift(r.host, r.roomCode, 'Whack Gift');
    const p = await H.playersByName(r.roomCode);
    const opener = H.openerNameOf(turn, p);
    const [bidder, other] = NAMES.filter((n) => n !== opener);
    await H.allBid(pages, opener, { [bidder]: 1 });
    await H.revealBids(r.host);
    await H.clickStartDuel(r.host);
    await expect(r.host.getByTestId('duel-game-name')).toContainText('Whack', { timeout: 10000 });
    await r.host.getByTestId('start-duel-app').click();
    await H.waitForTurnPhase(r.roomCode, 'minigame_party');

    // the intro: full screen on the TV and (by default) the console, with who is duelling; then 3, 2, 1 everywhere
    await expect(tv.getByTestId('party-full')).toBeVisible();
    await expect(r.host.getByTestId('party-full')).toBeVisible();
    await expect(tv.getByTestId('whack-stage')).toContainText(/The duel: \w+ vs \w+/);
    await expect(tv.getByTestId('whack-stage')).toContainText('Tap the Grump and lose 3!');
    await shot(tv, 'tv-intro');
    await expect(tv.getByTestId('whack-count')).toHaveText(/^[123]$/, { timeout: 9000 });
    await expect(pages[bidder].getByTestId('whack-count')).toHaveText(/^[123]$/);
    await shot(tv, 'tv-count');

    await H.waitForGame(r.roomCode, (g) => (g.minigame?.phase === 'playing' ? g : false), { label: 'playing', timeout: 15000 });
    const phone = pages[bidder];
    // the phone's board is the whole screen
    await expect(phone.locator('#party-card.whack-full .whack-grid')).toBeVisible();
    const box = await phone.locator('#party-card').boundingBox();
    const view = phone.viewportSize();
    expect(box.width).toBeGreaterThanOrEqual(view.width - 1);
    expect(box.height).toBeGreaterThan(view.height * 0.85);
    // a good player hits every elf (never the Grump); another taps everything
    for (let i = 0; i < 32; i++) { await phone.waitForTimeout(150); await tapUp(phone); if (i % 3 === 0) await tapUp(pages[other], true); }
    const score = Number(await phone.getByTestId('whack-score').textContent());
    expect(score).toBeGreaterThan(6);
    await expect(phone.getByTestId('whack-roll')).toContainText(/On a roll! \d+ in a row/);
    await expect(phone.getByTestId('whack-place')).toContainText('1st');
    await expect(phone.getByTestId('whack-place')).toContainText('of 4');
    await shot(phone, 'phone-play');

    // the TV: the board is popping, and the race has the leader on top with a full bar
    await expect(tv.locator('[data-testid=whack-stage] .whack-grid .wh-cell')).toHaveCount(9);
    const first = tv.locator('[data-testid=whack-stage] .wh-racer').first();
    await expect(first).toContainText(bidder, { timeout: 5000 });
    await expect(first).toContainText('duelling');
    expect((await first.locator('.fill').boundingBox()).width).toBeGreaterThan(40);
    await expect(tv.getByTestId(`whack-racer-${p[opener].id}`)).toContainText('duelling');
    await shot(tv, 'tv-play');
    await shot(r.host, 'console-play');

    // the room still holds the reported score to what was possible (a roll counted in)
    const mg = await H.dbGet(r.roomCode, 'minigame');
    expect(mg.scores[p[bidder].id].score).toBeGreaterThan(6);

    // the console can go back to its own view, where the game has a Full screen button
    await r.host.getByTestId('arena-full-exit').click();
    await expect(r.host.getByTestId('party-full')).toBeHidden();
    await expect(r.host.getByTestId('arena-full-open')).toBeVisible();
    await r.host.getByTestId('arena-full-open').click();
    await expect(r.host.getByTestId('party-full')).toBeVisible();

    // the results, full screen too
    await H.waitForGame(r.roomCode, (g) => (g.minigame?.phase === 'results' ? g : false), { label: 'results', timeout: 40000 });
    // (the same ending as every other game: ui/Ceremony.tsx PartyCeremony, set into the stage and sized for the room)
    await expect(tv.getByTestId('whack-stage').getByTestId('party-ceremony')).toBeVisible();
    await expect(tv.getByTestId('ceremony-headline')).toContainText(/wins the tie!|wins the duel!|comes top!|win the tie!/);
    const card = await tv.getByTestId('party-ceremony').locator('.cer-card').boundingBox();
    const vp = tv.viewportSize();
    expect(card.width, 'the results card is not lost in the screen').toBeGreaterThan(vp.width * 0.4);
    expect(card.y + card.height, 'and it is all on the screen').toBeLessThanOrEqual(vp.height);
    await expect(phone.getByTestId('party-ceremony')).toBeVisible();
    await expect(phone.locator('#party-card.whack-full')).toHaveCount(0);            // the phone's card is a card again
    await shot(tv, 'tv-results');
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
