// @ts-check
// Pass the Gremlin as the duel game, on real pages: the holder's phone shows the problem and a
// keypad, a right answer unlocks passing, the TV follows (no answer, no deadline), and the doom
// settles the duel with exactly one of the two duellists winning. The test makes the secret
// deadline due now through the debug API instead of waiting up to 100 s.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

/** Read "7 × 6 = ?" off a phone and type the answer on its keypad. */
async function solve(page) {
  const prompt = (await page.getByTestId('gremlin-prompt').textContent()).replace('= ?', '').trim();
  const [a, op, b] = prompt.split(' ');
  const answer = op === '+' ? +a + +b : op === '-' ? +a - +b : +a * +b;
  for (const k of String(answer)) await page.locator(`.grem-pad button[data-key="${k}"]`).click();
  await page.locator('.grem-pad button[data-key="OK"]').click();
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
