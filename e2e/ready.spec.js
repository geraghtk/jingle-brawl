// @ts-check
// Before a mini-game (the house default; the other specs switch it off): the instructions come up on
// the TV and on every phone, nobody starts until every player has pressed Ready, then a 3, 2, 1 and the
// game runs. The host can start without waiting. Whack-an-Elf's own intro waits the same way.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const lab = async (r, gameId) => {
  await r.host.getByTestId('lab-open').click();
  await r.host.getByTestId('lab-game').selectOption(gameId);
  await r.host.getByTestId('lab-start').click();
};
const lifecycle = async (room) => (await H.dbGet(room, 'arena'))?.lifecycle;

test('a mini-game shows how to play and waits for every phone\'s Ready, then counts 3, 2, 1 and starts', async ({ browser }) => {
  test.setTimeout(120000);
  const r = await H.createGame(browser, { 'ready-check': 'on' });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, ['Alice', 'Bob']);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    const tvErrors = [];
    tv.on('pageerror', (e) => tvErrors.push(String(e)));
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    await lab(r, 'reindeer_ring_toss');

    // the TV: the game's name, how to play in numbered lines, who is ready (nobody yet)
    const brief = tv.getByTestId('arena-briefing');
    await expect(brief).toBeVisible();
    await expect(brief).toContainText('Reindeer Ring Toss');
    await expect(brief.getByTestId('briefing-how').locator('li')).toHaveCount(3);
    await expect(brief.getByTestId('briefing-how')).toContainText('Flick up on your phone');
    await expect(brief.getByTestId('briefing-wait')).toContainText('Waiting for Alice, Bob');
    await expect(brief.locator('.brief-roster li.ready')).toHaveCount(0);

    // a phone: the same instructions and its own button, not the game's controls yet
    const [alice, bob] = [pages.Alice, pages.Bob];
    await expect(alice.getByTestId('briefing-how')).toContainText('Flick up on your phone');
    await expect(alice.getByTestId('ringtoss-card')).toHaveCount(0);
    await alice.getByTestId('briefing-ready').click();
    await expect(alice.getByTestId('briefing-ready')).toHaveCount(0);
    await expect(alice.getByTestId('briefing-wait')).toContainText('Ready! Waiting for Bob');
    await expect(brief.locator('.brief-roster li.ready')).toHaveCount(1);
    await expect(brief.getByTestId('briefing-wait')).toContainText('Waiting for Bob');

    // it really waits: nothing has started
    await tv.waitForTimeout(2500);
    expect(await lifecycle(r.roomCode)).toBe('briefing');
    await expect(brief.getByTestId('briefing-count')).toHaveCount(0);

    // the last Ready: the count, then the game (the panel goes, the phones get their controls)
    await bob.getByTestId('briefing-ready').click();
    await expect(brief.getByTestId('briefing-count')).toBeVisible();
    await expect(bob.getByTestId('briefing-count')).toBeVisible();
    await H.waitForGame(r.roomCode, (g) => (g.arena?.lifecycle === 'active' ? g : false), { label: 'the game started', timeout: 10000 });
    await expect(tv.getByTestId('arena-briefing')).toHaveCount(0);
    await expect(alice.getByTestId('arena-briefing')).toHaveCount(0);
    await expect(tv.getByTestId('ringtoss-stage')).toBeVisible();
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});

test('the host can start without waiting; Whack-an-Elf\'s intro waits for Ready too', async ({ browser }) => {
  test.setTimeout(120000);
  const r = await H.createGame(browser, { 'ready-check': 'on' });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, ['Alice', 'Bob', 'Charlie']);
  try {
    // nobody presses Ready: the console's own button starts the count
    await lab(r, 'pass_the_gremlin');
    await expect(pages.Alice.getByTestId('briefing-ready')).toBeVisible();
    expect(await lifecycle(r.roomCode)).toBe('briefing');
    await r.host.getByTestId('arena-start-now').click();
    await H.waitForGame(r.roomCode, (g) => (g.arena?.lifecycle === 'active' ? g : false), { label: 'started by the host', timeout: 10000 });
    await expect(pages.Alice.getByTestId('arena-briefing')).toHaveCount(0);
    await r.host.getByTestId('lab-end').click();

    // a party game: its rules were always there; now it waits for the table instead of counting down
    await lab(r, 'whack_elf');
    for (const n of ['Alice', 'Bob', 'Charlie']) await expect(pages[n].getByTestId('party-ready-btn')).toBeVisible();
    await pages.Alice.getByTestId('party-ready-btn').click();
    await pages.Bob.getByTestId('party-ready-btn').click();
    await pages.Alice.waitForTimeout(2500);
    expect((await H.dbGet(r.roomCode, 'minigame')).phase).toBe('intro');           // Charlie is still reading
    await expect(pages.Alice.getByTestId('party-ready')).toContainText('Waiting for Charlie');
    await pages.Charlie.getByTestId('party-ready-btn').click();
    await H.waitForGame(r.roomCode, (g) => (g.minigame?.phase === 'playing' ? g : false), { label: 'whack started', timeout: 10000 });
    expect([...r.errors, ...phoneErrors]).toEqual([]);
  } finally {
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
