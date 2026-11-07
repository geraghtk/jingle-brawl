// @ts-check
// The room server is the referee (runs only with TRANSPORT=server: npm run test:e2e:server).
// A phone can't do what isn't on its screen even by talking to the server directly; no phone or
// TV is ever sent a sealed bid; the server picks a game up after a restart; a reloaded phone
// rejoins with its token.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

test.skip(process.env.TRANSPORT !== 'server', 'room server only (npm run test:e2e:server)');

const NAMES = ['Alice', 'Bob', 'Charlie'];

/** Tap whatever elf is up for `ms` (never the Grinch). */
async function whack(page, ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const cell = page.locator('#party-card .whack-cell.up:not([data-kind="grinch"])').first();
    if (await cell.count()) await cell.dispatchEvent('pointerdown').catch(() => {});
    await page.waitForTimeout(60);
  }
}

async function setup(browser, settings = {}) {
  const { host, hostCtx, roomCode, errors } = await H.createGame(browser, settings);
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, NAMES);
  await H.startGame(host, roomCode);
  return { host, roomCode, pages, errors, close: async () => { await hostCtx.close(); for (const c of contexts) await c.close(); } };
}

/** Talk to the room server directly from a phone's page, with its own token: what a cheat would do. */
async function rawPress(page, roomCode, intent) {
  return page.evaluate(async ({ roomCode, intent }) => {
    const key = Object.keys(localStorage).find((k) => k.startsWith('jingleBrawl_player_token'));
    const token = key && localStorage.getItem(key);
    const ws = new WebSocket(`${location.origin.replace('http', 'ws')}/ws`);
    return new Promise((resolve) => {
      ws.onopen = () => ws.send(JSON.stringify({ t: 'hello', v: 1, room: roomCode, token }));
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.t === 'welcome') ws.send(JSON.stringify({ t: 'intent', id: `cheat-${Math.random()}`, intent }));
        if (m.t === 'ack') { ws.close(); resolve(m); }
      };
    });
  }, { roomCode, intent });
}

test('a phone talking to the server directly still can only press its own buttons', async ({ browser }) => {
  const { host, roomCode, pages, close } = await setup(browser);
  try {
    await H.drawGift(host, roomCode, 'Cheat Gift');
    const bob = pages.Bob;
    for (const intent of [{ type: 'reveal' }, { type: 'admin_set_chips', pid: 'x', chips: 99 }, { type: 'end' }, { type: 'bid', amount: 999 }]) {
      const ack = await rawPress(bob, roomCode, intent);
      expect(ack.ok, JSON.stringify(intent)).toBe(false);
    }
    expect((await H.dbGet(roomCode, 'currentTurn/phase'))).toBe('bidding');
    const chips = Object.values(await H.dbGet(roomCode, 'players')).map((p) => p.chips);
    expect(chips.every((c) => c <= 15)).toBe(true);
  } finally {
    await close();
  }
});

test('no phone and no TV is ever sent another sealed bid (checked on the wire)', async ({ browser }) => {
  const { host, roomCode, pages, close } = await setup(browser);
  const tvCtx = await browser.newContext();
  try {
    const frames = { tv: [], ...Object.fromEntries(NAMES.map((n) => [n, []])) };
    const listen = (page, who) => page.on('websocket', (ws) => ws.on('framereceived', (f) => frames[who].push(String(f.payload))));
    for (const n of NAMES) listen(pages[n], n);
    for (const n of NAMES) await pages[n].reload();                  // reconnect so the listener sees the socket
    for (const n of NAMES) { const b = pages[n].getByRole('button', { name: /Rejoin/i }); if (await b.count()) await b.click(); }
    const tv = await tvCtx.newPage();
    listen(tv, 'tv');
    await tv.goto(H.BASE + '/healthz');
    await tv.evaluate((room) => new Promise((r) => { const ws = new WebSocket(`${location.origin.replace('http', 'ws')}/ws`); ws.onopen = () => { ws.send(JSON.stringify({ t: 'hello', v: 1, room })); r(null); }; }), roomCode);

    const turn = await H.drawGift(host, roomCode, 'Secret Gift');
    const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
    const bids = Object.fromEntries(NAMES.filter((n) => n !== opener).map((n, i) => [n, i + 2]));
    await H.allBid(pages, opener, bids);
    await H.waitForGame(roomCode, (g) => Object.keys(g.currentTurn?.bids || {}).length === 2, { label: 'bids in' });
    const ids = await H.playersByName(roomCode);
    const leaks = (who) => frames[who].map((f) => JSON.parse(f)).filter((m) => m.t === 'state' && m.game.currentTurn?.phase === 'bidding')
      .flatMap((m) => Object.entries(m.game.currentTurn.bids || {}))
      .filter(([id, b]) => b.amount !== 0 && (who === 'tv' || ids[who].id !== id));
    for (const who of ['tv', ...NAMES]) expect(leaks(who), who).toEqual([]);
    expect(frames.tv.length).toBeGreaterThan(0);
  } finally {
    await tvCtx.close();
    await close();
  }
});

test('the server restarting mid party game picks it up: it finishes and pays once', async ({ browser }) => {
  test.setTimeout(90000);
  const { host, roomCode, pages, close } = await setup(browser, { 'tiebreaker-method': 'whack_elf' });
  try {
    const turn = await H.drawGift(host, roomCode, 'Restart Gift');
    const opener = H.openerNameOf(turn, await H.playersByName(roomCode));
    const [t1, t2] = NAMES.filter((n) => n !== opener);
    await H.allBid(pages, opener, { [t1]: 2, [t2]: 2 });
    await H.revealBids(host);
    await H.openerFights(pages[opener]);
    await H.waitForGame(roomCode, (g) => g.minigame?.phase === 'intro', { label: 'party intro' });
    const res = await fetch(`${H.BASE}/debug/rooms/${roomCode}/reload`, { method: 'POST' });  // the room reloads from storage
    expect(res.ok).toBe(true);                                       // (every screen is dropped and reconnects)
    await H.waitForGame(roomCode, (g) => g.minigame?.phase === 'playing', { label: 'playing after the restart', timeout: 20000 });
    await Promise.all(NAMES.map((n, i) => whack(pages[n], 4000 + i * 1500)));   // everyone plays, scores differ
    await H.waitForGame(roomCode, (g) => g.currentTurn?.phase === 'duel' || !g.minigame, { label: 'party over', timeout: 45000 });
    const players = Object.values(await H.dbGet(roomCode, 'players'));
    expect(players.reduce((s, p) => s + (p.partyPoints || 0), 0)).toBe(3 + 2 + 1);
  } finally {
    await close();
  }
});

test('a phone that reloads rejoins with its token, as the same player', async ({ browser }) => {
  const { roomCode, pages, close } = await setup(browser);
  try {
    const before = await H.playersByName(roomCode);
    await pages.Alice.reload();
    await pages.Alice.getByRole('button', { name: /Rejoin/i }).click();
    await pages.Alice.waitForSelector('#game-screen', { state: 'visible' });
    const after = await H.playersByName(roomCode);
    expect(Object.keys(after).length).toBe(Object.keys(before).length);
    expect(after.Alice.id).toBe(before.Alice.id);
    await H.waitForGame(roomCode, (g) => g.players[before.Alice.id].connected === true, { label: 'Alice back' });
  } finally {
    await close();
  }
});
