// @ts-check
// The Great Gift Conveyor on real pages: four at the table, the duel's game is the Conveyor, so the
// two duelists captain two teams of two. Each phone reads the order slip and taps the matching
// button, the checker compares the gift with the order and loads it (or sends it back); one wrong
// gift is loaded on purpose. The TV shows both factories; the team that fills the sleigh first wins
// its captain the duel, and only that team gets Party points.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie', 'Dana'];

/** Do the job on this phone's table, if there is one, as a player would: read the order slip, tap the
 *  matching button (`slip`: a wrong one, or load a wrong gift). Read in one go (the gift may move on). */
async function work(page, { slip = false } = {}) {
  const seen = await page.evaluate(() => {
    const job = document.querySelector('[data-testid=conveyor-job]');
    if (!job) return null;
    const txt = (sel, root = document) => (root.querySelector(sel)?.textContent || '').trim();
    const order = document.querySelector('[data-testid=conveyor-order]');
    const gift = document.querySelector('[data-testid=conveyor-gift]');
    return {
      step: job.getAttribute('data-step'),
      order: { toy: txt('.cv-toy', order), paper: txt('.cv-paper', order), tag: txt('.cv-tag', order).replace(/^for /, '') },
      gift: gift && { toy: txt('.cv-toy', gift), paper: txt('.cv-paper', gift), tag: txt('.cv-tag', gift).replace(/^for /, '') },
      shoo: !!document.querySelector('[data-testid=conveyor-shoo]'),
      choices: [...document.querySelectorAll('.cv-choice')].map((b) => (b.getAttribute('aria-label') || b.textContent || '').trim()),
    };
  });
  if (!seen) return null;
  const click = (loc) => loc.click({ timeout: 2000 }).then(() => true, () => false);   // it may have moved on: fine
  const want = seen.step === 'pick' ? seen.order.toy : seen.step === 'wrap' ? seen.order.paper : seen.step === 'label' ? seen.order.tag : null;
  if (want !== null) {
    const k = seen.choices.findIndex((c) => (c === want) !== slip);
    return (await click(page.locator('.cv-choice').nth(k))) ? seen.step : null;
  }
  if (seen.shoo) return (await click(page.getByTestId('conveyor-shoo'))) ? 'shoo' : null;
  const same = seen.gift && seen.gift.toy === seen.order.toy && seen.gift.paper === seen.order.paper && seen.gift.tag === seen.order.tag;
  if (!(await click(page.getByTestId(same || slip ? 'conveyor-load' : 'conveyor-back')))) return null;
  return same ? 'load' : slip ? 'load-wrong' : 'back';
}

test('the Gift Conveyor: captains, teams, phone-to-phone gifts, the TV factories, the winning captain', async ({ browser }) => {
  test.setTimeout(180000);
  const r = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-gift_conveyor': 'on' });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    await H.startGame(r.host, r.roomCode);
    const turn = await H.drawGift(r.host, r.roomCode, 'Factory Gift');
    const p = await H.playersByName(r.roomCode);
    const opener = H.openerNameOf(turn, p);
    const [bidder] = NAMES.filter((n) => n !== opener);
    await H.allBid(pages, opener, { [bidder]: 1 });
    await H.revealBids(r.host);
    await H.clickStartDuel(r.host);
    await expect(r.host.getByTestId('duel-game-name')).toContainText('Gift Conveyor', { timeout: 10000 });
    await r.host.getByTestId('start-duel-app').click();
    await H.waitForTurnPhase(r.roomCode, 'minigame_arena');

    const a = await H.dbGet(r.roomCode, 'arena');
    expect(a.mode).toBe('captain_teams');
    const [teamA, teamB] = a.state.teams;
    const byId = Object.fromEntries(Object.entries(p).map(([n, x]) => [x.id, n]));
    expect([byId[teamA.captainId], byId[teamB.captainId]].sort()).toEqual([opener, bidder].sort());
    expect(teamA.members).toHaveLength(2);
    expect(teamB.members).toHaveLength(2);
    await expect(tv.getByTestId('conveyor-lane-A')).toBeVisible();
    await expect(tv.getByTestId('conveyor-lane-B')).toBeVisible();
    for (const id of [...teamA.members, ...teamB.members]) {
      await expect(pages[byId[id]].getByTestId('conveyor-my-team')).toContainText(teamA.members.includes(id) ? 'Team A' : 'Team B');
    }

    // team A works; their first gift gets the wrong toy and the checker loads it anyway
    const A = teamA.members.map((id) => pages[byId[id]]);
    const stepOn = (page) => page.evaluate(() => document.querySelector('[data-testid=conveyor-job]')?.getAttribute('data-step') ?? null);
    let slipped = false, wrongLoaded = false;
    for (let i = 0; i < 300; i++) {
      const g = await H.dbGet(r.roomCode, 'arena');
      if (!g || g.result) break;
      for (const page of A) {
        const step = await stepOn(page);
        if (!step) continue;
        const slip = (step === 'pick' && !slipped) || (step === 'check' && slipped && !wrongLoaded);
        const did = await work(page, { slip });
        if (step === 'pick' && slip) slipped = true;
        if (did === 'load-wrong') {
          wrongLoaded = true;
          await expect(tv.getByTestId('conveyor-delivered-A')).toContainText('1 wrong');   // the TV shows it
        }
      }
      await A[0].waitForTimeout(60);
    }
    expect(wrongLoaded).toBe(true);

    // team A filled the sleigh: its captain wins the duel; only team A got Party points
    const g = await H.waitForGame(r.roomCode, (x) => (x.lastDuel || x.currentTurn?.duelGameWinner ? x : false), { label: 'conveyor decided', timeout: 30000 });
    const winner = g.lastDuel?.winnerId ?? g.currentTurn?.duelGameWinner;
    expect(winner).toBe(teamA.captainId);
    for (const id of teamA.members) expect(g.players[id].partyPoints || 0).toBe(1);          // a flat point each
    for (const id of teamB.members) expect(g.players[id].partyPoints || 0).toBe(0);
    expect([...r.errors, ...phoneErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
