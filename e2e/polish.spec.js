// @ts-check
// What every mini-game shares, on real pages: the practice round the first time a game comes up in a
// night (marked on every screen, deciding nothing, then the same game for real), the results ceremony
// (the winner announced on the TV and on each phone before the duel is settled), and at the end of the
// night "Play again with the same people".
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob'];
const shot = async (page, name) => { if (process.env.POLISH_SHOTS) await page.screenshot({ path: `${process.env.POLISH_SHOTS}/${name}.png` }); };
/** The room's pending deadlines fall due now (the Gremlin's intro, then its secret moment). */
const dueNow = async (room) => {
  const a = await H.dbGet(room, 'arena');
  if ((a?.deadlines || []).length) await H.dbPut(room, 'arena/deadlines', a.deadlines.map((d) => ({ ...d, at: Date.now() })));
};
/** Hurry the running game to its end (whoever holds the Gremlin is caught). */
const finish = async (room) => {
  await expect.poll(async () => { await dueNow(room); return !!(await H.dbGet(room, 'arena'))?.result; }, { timeout: 20000, intervals: [400] }).toBe(true);
};

/** Two at the table, a duel whose game is Pass the Gremlin, started; the TV watching. */
async function toArena(browser, settings = {}) {
  const r = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-pass_the_gremlin': 'on', ...settings });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const tv = await tvCtx.newPage();
  const tvErrors = [];
  tv.on('pageerror', (e) => tvErrors.push(String(e)));
  await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
  await H.startGame(r.host, r.roomCode);
  const turn = await H.drawGift(r.host, r.roomCode, 'Mystery Gift');
  const p = await H.playersByName(r.roomCode);
  const opener = H.openerNameOf(turn, p);
  const [bidder] = NAMES.filter((n) => n !== opener);
  await H.allBid(pages, opener, { [bidder]: 1 });
  await H.revealBids(r.host);
  await H.clickStartDuel(r.host);
  await expect(r.host.getByTestId('duel-game-name')).toContainText('Pass the Gremlin', { timeout: 10000 });
  await r.host.getByTestId('start-duel-app').click();
  await H.waitForTurnPhase(r.roomCode, 'minigame_arena');
  const close = async () => { await tvCtx.close(); await r.hostCtx.close(); for (const c of contexts) await c.close(); };
  return { ...r, pages, phoneErrors, tvErrors, tv, p, close };
}

test('the first time a game comes up: a practice round on every screen that decides nothing, then the same game for real, announced when it ends', async ({ browser }) => {
  test.setTimeout(180000);
  const r = await toArena(browser, { 'practice-round': 'on' });
  try {
    const { tv, pages, roomCode } = r;
    // practice: marked on the TV and on both phones, and the room knows it is one
    await expect(tv.getByTestId('practice-ribbon')).toBeVisible();
    await expect(tv.getByTestId('practice-ribbon')).toContainText('This one does not count');
    for (const n of NAMES) await expect(pages[n].getByTestId('practice-ribbon')).toBeVisible();
    await shot(tv, 'practice-tv'); await shot(pages.Alice, 'practice-phone');
    const first = await H.dbGet(roomCode, 'arena');
    expect(first.practice.on).toBe(true);
    expect(first.practice.endsAt).toBeGreaterThan(Date.now());

    // somebody is caught in practice: nobody is announced, the duel is not settled, and the game is set up again for real
    await tv.getByTestId('gremlin-stage').evaluate((el) => { el.dataset.old = '1'; });
    await finish(roomCode);
    await expect(tv.getByTestId('arena-briefing')).toContainText('Now for real', { timeout: 8000 });
    await shot(tv, 'real-tv'); await shot(pages.Alice, 'real-phone');
    await expect(tv.getByTestId('arena-ceremony')).toHaveCount(0);
    const real = await H.dbGet(roomCode, 'arena');
    expect(real.instanceId).not.toBe(first.instanceId);
    // a new game is drawn on a new stage, never on what the last one built (Mecha Santa's replay kept the old course in 3D)
    await expect(tv.getByTestId('gremlin-stage')).not.toHaveAttribute('data-old', '1');
    expect(real.practice ?? null).toBeNull();
    expect(real.result ?? null).toBeNull();
    expect((await H.dbGet(roomCode, 'lastDuel')) ?? null).toBeNull();
    await expect(pages.Alice.getByTestId('arena-briefing')).toContainText('Now for real');
    await expect(pages.Alice.getByTestId('briefing-ready')).toHaveCount(0);      // nobody presses Ready a second time

    // the count ends and the real game runs, with no ribbon
    await H.waitForGame(roomCode, (g) => (g.arena?.lifecycle === 'active' ? g : false), { label: 'the real game started', timeout: 12000 });
    await expect(tv.getByTestId('practice-ribbon')).toHaveCount(0);
    await expect(pages.Bob.getByTestId('practice-ribbon')).toHaveCount(0);

    // this one counts: when it ends the winner is announced on the TV and each phone says how it went for them
    for (const page of [tv, ...Object.values(pages)]) await page.evaluate(() => { window.__chips = []; document.addEventListener('jb:chips', (e) => window.__chips.push(e.detail)); });
    await finish(roomCode);
    const result = (await H.dbGet(roomCode, 'arena')).result;
    const winner = NAMES.find((n) => r.p[n].id === result.duel.winningDuelistId);
    const loser = NAMES.find((n) => n !== winner);
    await expect(tv.getByTestId('arena-ceremony')).toBeVisible({ timeout: 8000 });
    await expect(tv.getByTestId('ceremony-headline')).toHaveText(`${winner} wins!`);
    await expect(tv.getByTestId(`ceremony-player-${r.p[winner].id}`)).toBeVisible();
    await expect(tv.getByTestId('confetti')).toHaveCount(1);
    await expect(pages[winner].getByTestId('ceremony-mine')).toHaveText('You won!');
    await expect(pages[loser].getByTestId('ceremony-mine')).toHaveText('Not this time.');
    await expect(pages[loser].getByTestId('confetti')).toHaveCount(0);
    // the one announcement: the ceremony also says what the duel was for (the room works it out from resolveDuel)
    await expect(tv.getByTestId('ceremony-stakes')).toContainText(new RegExp(`${winner} (takes|keeps) `));
    // and the pot flies to the winner's picture, on the TV and on the phones (the old verdict's flying chips)
    await expect.poll(() => tv.evaluate(() => (window.__chips || []).filter((c) => c.drawn > 0).length), { timeout: 5000 }).toBe(1);
    for (let k = 0; k < 7; k++) await shot(tv, `chips-${k}`);                       // (a burst: the flight is under a second)
    for (const page of [tv, pages[winner], pages[loser]]) await expect.poll(() => page.evaluate(() => (window.__chips || []).filter((c) => c.drawn > 0).length), { timeout: 5000 }).toBe(1);
    await expect(pages[loser].getByTestId('ceremony-stakes')).toBeVisible();
    await tv.waitForTimeout(500);
    await shot(tv, 'ceremony-tv'); await shot(pages[winner], 'ceremony-won'); await shot(pages[loser], 'ceremony-lost');

    // then the duel is settled with that winner, and the announcement is gone
    await H.waitForGame(roomCode, (g) => (g.lastDuel ? g : false), { label: 'the duel settled', timeout: 15000 });
    expect((await H.dbGet(roomCode, 'lastDuel')).winnerId).toBe(r.p[winner].id);
    await expect(tv.getByTestId('arena-ceremony')).toHaveCount(0);
    // ... and it is not announced a second time: no duel verdict on the TV, the console or either phone
    // (the verdict used to come up over the main screen as soon as the duel was settled, for 6.5 s)
    expect((await H.dbGet(roomCode, 'lastDuel')).announced).toBe(true);
    await tv.waitForTimeout(1200);
    await expect(tv.locator('#duel-verdict')).toHaveCount(0);
    await expect(r.host.locator('#duel-verdict')).toHaveCount(0);
    for (const n of NAMES) await expect(pages[n].locator('#duel-result-modal.active')).toHaveCount(0);
    expect([...r.errors, ...r.phoneErrors, ...r.tvErrors]).toEqual([]);
  } finally { await r.close(); }
});

test('the host can cut practice short, and the room remembers the game has been tried tonight', async ({ browser }) => {
  test.setTimeout(180000);
  const r = await toArena(browser, { 'practice-round': 'on' });
  try {
    const { tv, roomCode } = r;
    await expect(tv.getByTestId('practice-ribbon')).toBeVisible();
    await r.host.getByTestId('arena-skip-practice').click();
    await expect(tv.getByTestId('arena-briefing')).toContainText('Now for real', { timeout: 5000 });
    await H.waitForGame(roomCode, (g) => (g.arena?.lifecycle === 'active' && !g.arena.practice ? g : false), { label: 'the real game', timeout: 12000 });
    expect((await H.dbGet(roomCode, 'triedGames')).pass_the_gremlin).toBe(true);
    expect([...r.errors, ...r.phoneErrors, ...r.tvErrors]).toEqual([]);
  } finally { await r.close(); }
});

test('the end of the night: the finale on every screen, then Play again puts the same people back in the lobby', async ({ browser }) => {
  test.setTimeout(180000);
  const r = await H.createGame(browser, {});
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, ['Alice', 'Bob', 'Charlie']);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    const tvErrors = [];
    tv.on('pageerror', (e) => tvErrors.push(String(e)));
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    await H.startGame(r.host, r.roomCode);
    const p = await H.playersByName(r.roomCode);
    await H.dbPatch(r.roomCode, {
      gifts: { g1: { name: 'Fuzzy socks', ownerId: p.Bob.id, naughtyLevel: 2 } }, [`players/${p.Bob.id}/giftId`]: 'g1',
      [`players/${p.Bob.id}/partyPoints`]: 5, [`players/${p.Bob.id}/duelWins`]: 2, [`players/${p.Alice.id}/partyPoints`]: 2, [`players/${p.Charlie.id}/chips`]: 1, phase: 'ended',
    });

    // the TV runs the show by itself: the MVP, then the podium with the winner on the top step, then who got what
    await expect(tv.getByTestId('mvp-name')).toHaveText('Bob', { timeout: 8000 });
    await expect(tv.getByTestId('recap-podium')).toBeVisible({ timeout: 15000 });
    await expect(tv.getByTestId('recap-step-1')).toContainText('Bob', { timeout: 15000 });
    await expect(tv.getByTestId('recap-list')).toBeVisible({ timeout: 30000 });
    await expect(tv.getByTestId(`recap-line-${p.Bob.id}`)).toContainText('Fuzzy socks');
    await shot(tv, 'recap-tv');

    // a phone: past the MVP moment, your own place and the gift you went home with, and a picture to keep
    const bob = pages.Bob;
    await bob.locator('#mvp-reveal').click();
    await expect(bob.getByTestId('recap-place')).toContainText('1st');
    await expect(bob.getByTestId('recap')).toContainText('Fuzzy socks');
    await expect(bob.getByTestId('recap-save')).toBeVisible();
    await shot(bob, 'recap-phone');

    // the host closes the finale and starts over with the same people
    await r.host.locator('#mvp-reveal').click();
    await r.host.getByTestId('recap-close').click();
    await r.host.getByTestId('play-again').click();
    await r.host.getByTestId('dialog-confirm').click();
    const g = await H.waitForGame(r.roomCode, (x) => (x.phase === 'lobby' ? x : false), { label: 'back in the lobby' });
    expect(Object.values(g.players).map((x) => x.name).sort()).toEqual(['Alice', 'Bob', 'Charlie']);
    for (const x of Object.values(g.players)) { expect(x.giftId ?? null).toBeNull(); expect(x.partyPoints ?? 0).toBe(0); expect(x.chips).toBe(10); }
    expect(g.gifts ?? null).toBeNull();
    await expect(tv.getByTestId('recap')).toHaveCount(0);
    await expect(bob.getByTestId('recap')).toHaveCount(0);
    await expect(r.host.locator('#btn-start-game')).toBeEnabled();          // and off they go again
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
