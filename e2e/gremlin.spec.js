// @ts-check
// Pass the Gremlin as the duel game, on real pages: the holder's phone shows the problem and a
// keypad, a right answer unlocks passing, the TV follows (no answer, no deadline), and the doom
// settles the duel with exactly one of the two duellists winning. The test makes the secret
// deadline due now through the debug API instead of waiting up to 100 s.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

/** Do whatever challenge the holder's phone shows, right (or `wrong`). */
async function solve(page, { wrong = false } = {}) {
  const shown = { quick_math: page.getByTestId('gremlin-prompt'), memory_flash: page.getByTestId('gremlin-memory-show'), perfect_timing: page.getByTestId('gremlin-timing') };
  await shown.quick_math.or(shown.memory_flash).or(shown.perfect_timing).first().waitFor();
  let kind = 'quick_math';
  for (const [k, loc] of Object.entries(shown)) if (await loc.count()) kind = k;
  if (kind === 'quick_math') {
    // read "7 × 6 = ?" off the phone and type the answer on its keypad
    const prompt = (await page.getByTestId('gremlin-prompt').textContent()).replace('= ?', '').trim();
    const [a, op, b] = prompt.split(' ');
    const answer = (op === '+' ? +a + +b : op === '-' ? +a - +b : +a * +b) + (wrong ? 1 : 0);
    for (const k of String(answer)) await page.locator(`.grem-pad button[data-key="${k}"]`).click();
    await page.locator('.grem-pad button[data-key="OK"]').click();
  } else if (kind === 'memory_flash') {
    // remember the row while it shows, then tap it back once it hides
    const seq = await page.locator('[data-testid=gremlin-memory-show] [data-sym]').evaluateAll((els) => els.map((e) => Number(e.getAttribute('data-sym'))));
    if (wrong) seq[0] = (seq[0] + 1) % 6;
    await page.getByTestId('gremlin-memory-pad').waitFor({ timeout: 6000 });
    for (const n of seq) await page.locator(`[data-testid=gremlin-memory-pad] button[data-sym="${n}"]`).click();
  } else {
    // press STOP at a moment the marker is mid-zone (or well outside it), timed inside the page
    // on the room's clock as the phone keeps it, the way a player's thumb would
    await page.evaluate((miss) => new Promise((done) => {
      const el = document.querySelector('[data-testid=gremlin-timing]');
      const start = +el.dataset.start, period = +el.dataset.period, off = +el.dataset.offset;
      const [lo, hi] = el.dataset.band.split(',').map(Number);
      const x = miss ? (lo > 0.5 ? lo / 3 : (hi + 2) / 3) : (lo + hi) / 2;
      const now = Date.now() + off;
      const k = Math.max(0, Math.floor((now - start) / period));
      const at = [k, k + 1, k + 2].flatMap((j) => [start + (j + x / 2) * period, start + (j + 1 - x / 2) * period]).filter((t) => t > now + 40).sort((a, b) => a - b)[0];
      setTimeout(() => {
        document.querySelector('[data-testid=gremlin-stop]').dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
        done();
      }, at - now);
    }), wrong);
  }
  return kind;
}

/** Three at the table, a duel whose game is Pass the Gremlin, started; the TV watching. */
async function toArena(browser, settings = {}) {
  const r = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-pass_the_gremlin': 'on', ...settings });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const tv = await tvCtx.newPage();
  await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
  await H.startGame(r.host, r.roomCode);
  const turn = await H.drawGift(r.host, r.roomCode, 'Gremlin Gift');
  const p = await H.playersByName(r.roomCode);
  const opener = H.openerNameOf(turn, p);
  const [bidder] = NAMES.filter((n) => n !== opener);
  await H.allBid(pages, opener, { [bidder]: 1 });
  await H.revealBids(r.host);
  await H.clickStartDuel(r.host);
  await expect(r.host.getByTestId('duel-game-name')).toContainText('Pass the Gremlin', { timeout: 10000 });
  await r.host.getByTestId('start-duel-app').click();
  await H.waitForTurnPhase(r.roomCode, 'minigame_arena');
  await expect(tv.getByTestId('gremlin-stage')).toBeVisible();
  const a0 = await H.dbGet(r.roomCode, 'arena');
  const holderName = Object.keys(p).find((n) => p[n].id === a0.state.holder);
  const close = async () => { await tvCtx.close(); await r.hostCtx.close(); for (const c of contexts) await c.close(); };
  return { ...r, pages, phoneErrors, tv, p, opener, bidder, a0, holderName, holder: pages[holderName], close };
}

test('Pass the Gremlin settles a duel: solve, pass, doom; the TV never shows the answer', async ({ browser }) => {
  test.setTimeout(120000);
  const { host, hostCtx, roomCode, errors } = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-pass_the_gremlin': 'on' });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    await tv.goto(`${H.BASE}/tv.html?room=${roomCode}`);
    await H.startGame(host, roomCode);
    const turn = await H.drawGift(host, roomCode, 'Gremlin Gift');
    const p = await H.playersByName(roomCode);
    const opener = H.openerNameOf(turn, p);
    const [bidder] = NAMES.filter((n) => n !== opener);
    await H.allBid(pages, opener, { [bidder]: 1 });
    await H.revealBids(host);
    await H.clickStartDuel(host);
    await expect(host.getByTestId('duel-game-name')).toContainText('Pass the Gremlin', { timeout: 10000 });
    await host.getByTestId('start-duel-app').click();
    await H.waitForTurnPhase(roomCode, 'minigame_arena');
    await expect(tv.getByTestId('gremlin-stage')).toBeVisible();

    // the holder solves and passes at random
    const a0 = await H.dbGet(roomCode, 'arena');
    expect(a0.mode).toBe('party_with_duelists');                     // three at the table: everyone plays
    const holderName = Object.keys(p).find((n) => p[n].id === a0.state.holder);
    const holder = pages[holderName];
    await expect(holder.getByTestId('gremlin-holder')).toHaveText('You have the Gremlin!');
    await expect(tv.getByTestId('gremlin-tv-holder')).toContainText(holderName);
    for (const n of NAMES.filter((x) => x !== holderName)) await expect(pages[n].getByTestId('gremlin-prompt')).toHaveCount(0);
    await solve(holder);
    await holder.getByTestId('gremlin-random').click();
    await H.waitForGame(roomCode, (g) => (g.arena?.state.holder !== a0.state.holder ? g : false), { label: 'passed' });
    const tvText = await tv.locator('body').textContent();
    expect(tvText).not.toContain(String(a0.state.doomAt));

    // the doom (made due now), maybe a sudden death, until the duel is settled
    for (let i = 0; i < 20; i++) {
      const g = await H.dbGet(roomCode, '');
      if (g.lastDuel) break;
      const a = g.arena;
      if (a?.lifecycle === 'active' && a.deadlines?.some((d) => d.id === 'doom')) {
        await H.dbPut(roomCode, 'arena/deadlines', [{ id: 'doom', at: Date.now() }]);
      }
      await host.waitForTimeout(700);
    }
    const g = await H.waitForGame(roomCode, (x) => (x.lastDuel ? x : false), { label: 'duel settled', timeout: 30000 });
    expect([p[opener].id, p[bidder].id]).toContain(g.lastDuel.winnerId);
    expect([...errors, ...phoneErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await hostCtx.close();
    for (const c of contexts) await c.close();
  }
});

for (const [kind, label] of [['memory_flash', 'a memory test'], ['perfect_timing', 'a timing test']]) {
  test(`${kind}: only the holder's phone gets the challenge; a miss gets another, a hit unlocks the pass`, async ({ browser }) => {
    test.setTimeout(120000);
    const t = await toArena(browser, { 'gremlin-challenges': kind });
    try {
      expect(t.a0.state.cfg.challenges).toEqual([kind]);
      await expect(t.holder.getByTestId('gremlin-holder')).toHaveText('You have the Gremlin!');
      await expect(t.tv.getByTestId('gremlin-tv-kind')).toContainText(label);
      for (const n of NAMES.filter((x) => x !== t.holderName)) {
        await expect(t.pages[n].getByTestId('gremlin-memory-show')).toHaveCount(0);
        await expect(t.pages[n].getByTestId('gremlin-timing')).toHaveCount(0);
        await expect(t.pages[n].getByTestId('gremlin-card')).toContainText(label);
      }
      for (const page of [t.tv, t.host]) {
        await expect(page.getByTestId('gremlin-memory-show')).toHaveCount(0);
        await expect(page.getByTestId('gremlin-timing')).toHaveCount(0);
      }
      // a miss: told so, and a new one of the same kind once the short wait is over
      await solve(t.holder, { wrong: true });
      await expect(t.holder.locator('.grem-error')).toContainText('Wrong!');
      const after = await H.dbGet(t.roomCode, 'arena/state');
      expect(after.stage).toBe('holding');
      expect(after.challenge.kind).toBe(kind);
      await expect(t.holder.locator('.grem-error')).not.toContainText('Wrong!', { timeout: 5000 });
      // a hit: the pass buttons, and the TV says so
      await solve(t.holder);
      await expect(t.holder.getByTestId('gremlin-random')).toBeVisible({ timeout: 5000 });
      await expect(t.tv.getByTestId('gremlin-tv-kind')).toContainText('Passing it on');
      await t.holder.getByTestId('gremlin-random').click();
      await H.waitForGame(t.roomCode, (g) => (g.arena?.state.holder !== t.a0.state.holder ? g : false), { label: 'passed' });
      expect([...t.errors, ...t.phoneErrors]).toEqual([]);
    } finally {
      await t.close();
    }
  });
}

test("the Gremlin's sound switch is remembered on the phone", async ({ browser }) => {
  test.setTimeout(120000);
  const t = await toArena(browser, { 'gremlin-challenges': 'quick_math' });
  try {
    const mute = t.holder.getByTestId('gremlin-mute');
    await expect(mute).toHaveAttribute('aria-pressed', 'false');
    await mute.click();
    await expect(mute).toHaveAttribute('aria-pressed', 'true');
    await t.holder.reload();
    await t.holder.getByRole('button', { name: /Rejoin/i }).click().catch(() => {});
    await expect(t.holder.getByTestId('gremlin-mute')).toHaveAttribute('aria-pressed', 'true');
    await expect(t.holder.locator('.grem-art').first()).toBeVisible();             // the sticker, not the emoji
    expect(await t.holder.locator('.grem-art').first().evaluate((img) => img.naturalWidth)).toBeGreaterThan(0);
  } finally {
    await t.close();
  }
});
