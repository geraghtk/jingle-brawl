// @ts-check
// Losing the room and finding it again, on real pages. The phone's WebSocket runs through
// Playwright's router so the test can cut it: cleanly (Wi-Fi drops), or silently (a phone that slept:
// the socket looks open but nothing arrives). Either way the phone says it is reconnecting, comes
// back by itself, still in its seat (and a 'no hello' on a dead line doesn't make it give up). And a dead phone's player gets back in
// their seat on another phone from a link the host makes.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

/** Route a page's WebSocket through a switch: `mode` 'up' passes everything, 'cut' refuses and
 *  closes, 'silent' keeps the socket open but drops what the room sends. */
async function switchable(page) {
  const net = { mode: 'up', live: /** @type {any[]} */ ([]) };
  await page.routeWebSocket(/\/ws$/, (ws) => {
    if (net.mode === 'cut') return ws.close();
    const server = ws.connectToServer();
    const link = { ws, server };
    net.live.push(link);
    ws.onMessage((m) => { if (net.mode === 'up') server.send(m); });
    ws.onClose(() => { if (net.mode === 'up') server.close(); });
    server.onMessage((m) => { if (net.mode === 'up') ws.send(m); });
  });
  return {
    cut() { net.mode = 'cut'; for (const l of net.live.splice(0)) { l.ws.close(); l.server.close(); } },
    silence() { net.mode = 'silent'; },
    restore() { net.mode = 'up'; },
  };
}

test('a phone that loses the room says so, comes back by itself, still in its seat', async ({ browser }) => {
  test.setTimeout(90000);
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  try {
    const phone = await ctx.newPage();
    const net = await switchable(phone);
    await phone.goto(`${H.BASE}/player.html?room=${roomCode}&testId=1&name=Alice`);
    await phone.getByRole('button', { name: /Join Game/i }).click();
    await phone.waitForSelector('#game-screen', { state: 'visible' });
    await expect(phone.getByTestId('reconnecting')).toHaveCount(0);

    // the Wi-Fi drops: the banner shows (after a moment) and stays while the room can't be reached
    net.cut();
    await expect(phone.getByTestId('reconnecting')).toBeVisible();
    await expect(phone.getByTestId('reconnecting')).toContainText('Reconnecting');
    // something changes in the room meanwhile
    await H.dbPut(roomCode, 'settings/startingChips', 15);
    net.restore();
    await expect(phone.getByTestId('reconnecting')).toHaveCount(0, { timeout: 10000 });

    // the phone slept: its socket looks open, but nothing comes through. It notices and starts over
    net.silence();
    await expect(phone.getByTestId('reconnecting')).toBeVisible({ timeout: 20000 });
    net.restore();
    await expect(phone.getByTestId('reconnecting')).toHaveCount(0, { timeout: 10000 });
    await expect(host.getByText('Alice').first()).toBeVisible();
    const g = await H.dbGet(roomCode);
    const alice = Object.values(g.players).find((p) => p.name === 'Alice');
    expect(alice.connected).toBe(true);
  } finally {
    await Promise.all([ctx.close(), hostCtx.close()]);
  }
});

test("the host gives a player whose phone died a link back to their seat", async ({ browser }) => {
  test.setTimeout(90000);
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, ['Alice', 'Bob']);
  const spare = await browser.newContext({ viewport: { width: 390, height: 844 } });
  try {
    await H.startGame(host, roomCode);
    const ids = await H.playersByName(roomCode);
    await H.dbPut(roomCode, `players/${ids.Alice.id}/chips`, 7);        // something to bring along
    await contexts[0].close();                                       // Alice's phone dies

    await host.getByTestId('seat-player').selectOption({ label: 'Alice' });
    await host.getByTestId('seat-make').click();
    const url = await host.getByTestId('seat-link').getAttribute('href');
    expect(url).toContain(`room=${roomCode}`);
    expect(url).toContain('seat=');

    const phone = await spare.newPage();
    const errors = [];
    phone.on('pageerror', (e) => errors.push(e.message));
    await phone.goto(url.replace(/^https?:\/\/[^/]+/, H.BASE));
    await phone.waitForSelector('#game-screen', { state: 'visible' });
    expect(phone.url()).not.toContain('seat=');                      // the link is a key: off the address bar
    const seat = await phone.evaluate(() => Object.fromEntries(Object.entries(localStorage).filter(([k]) => k.includes('player_id'))));
    expect(Object.values(seat)).toContain(ids.Alice.id);
    await H.waitForGame(roomCode, (g) => g.players[ids.Alice.id].connected === true, { label: 'Alice back' });
    await expect(phone.locator('#chip-display')).toHaveText('7');      // her seat, with her chips
    await expect(pages.Bob.locator('#game-screen')).toBeVisible();   // Bob never noticed
    expect(errors).toEqual([]);
    await H.assertInvariants(roomCode, 'after a seat moved phones');
  } finally {
    await Promise.all([spare.close(), hostCtx.close(), ...contexts.slice(1).map((c) => c.close())]);
  }
});
