// @ts-check
// The Great Gift Conveyor on real pages (from the Game Lab): four at the table, so the two captains
// lead teams of two and everyone works two stations. Unfinished toys roll along each team's belt on
// the TV; a phone shows its own stations close up, and a piece can only be fitted while the toy is
// there. One team works, the other watches: the workers' toys reach the sleigh and their belt speeds
// up; the watchers' toys fall in the bin. At the bell the team with more toys wins.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie', 'Dana'];
const shot = async (page, name) => { if (process.env.CV_SHOTS) await page.screenshot({ path: `${process.env.CV_SHOTS}/${name}.png` }); };
const state = async (room) => (await H.dbGet(room, 'arena')).state;

/** Do what this phone can do right now, as a player would: at each of its stations, if a piece can be
 *  fitted, read what the toy wants and tap that piece (`slip`: another one); shoo a raccoon. */
const work = (page, slip = false) => page.evaluate((wrong) => {
  let did = 0;
  for (const st of document.querySelectorAll('[data-testid^=conveyor-station-]')) {
    const shoo = st.querySelector('.cv-shoo:not(:disabled)');
    if (shoo) { shoo.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })); did++; continue; }
    const want = st.querySelector('.cv-ask strong')?.textContent?.trim();
    const open = [...st.querySelectorAll('.cv-choice:not(:disabled)')];
    if (!want || !open.length) continue;
    const label = (b) => (b.querySelector(':scope > span:last-child')?.textContent || '').trim();   // the piece's name (the swatch carries a mark too)
    const pick = open.find((b) => (label(b) === want) !== wrong);
    if (pick) { pick.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })); did++; }
  }
  return did;
}, slip);

test('the Gift Conveyor: toys roll past your station, fit the piece in time; the working team fills its sleigh and its belt speeds up', async ({ browser }) => {
  test.setTimeout(180000);
  const r = await H.createGame(browser, {});
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    const tvErrors = [];
    tv.on('pageerror', (e) => tvErrors.push(String(e)));
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    const p = await H.playersByName(r.roomCode);
    const nameOf = (id) => NAMES.find((n) => p[n].id === id);

    await r.host.getByTestId('lab-open').click();
    await r.host.getByTestId('lab-game').selectOption('gift_conveyor');
    await expect(r.host.getByTestId('lab-mode')).toHaveValue('captain_teams');
    await r.host.getByTestId('lab-start').click();
    await expect(tv.getByTestId('conveyor-stage')).toBeVisible();

    // two teams of two: each captain's team has all four stations between its two players
    const s0 = await state(r.roomCode);
    expect(s0.teams.map((t) => t.members.length)).toEqual([2, 2]);
    for (const t of s0.teams) expect(t.crew.map((c) => c.length)).toEqual([1, 1, 1, 1]);
    const [A, B] = s0.teams;
    const workers = A.members.map((id) => pages[nameOf(id)]);
    const idle = B.members.map((id) => pages[nameOf(id)]);
    for (const page of [...workers, ...idle]) {
      await expect(page.getByTestId('conveyor-card')).toBeVisible();
      await expect(page.locator('[data-testid^=conveyor-station-]')).toHaveCount(2);     // two stations each
    }
    await expect(workers[0].getByTestId('conveyor-my-team')).toContainText('Team A');
    // a small phone (320 x 568): both stations are on the screen, every piece in reach, nothing to scroll
    // (the second station used to be below the fold)
    await workers[0].setViewportSize({ width: 320, height: 568 });
    for (const station of await workers[0].locator('[data-testid^=conveyor-station-]').all()) await expect(station).toBeInViewport({ ratio: 1 });
    for (const piece of await workers[0].locator('[data-testid^=conveyor-fit-], [data-testid^=conveyor-shoo-]').all()) await expect(piece).toBeInViewport({ ratio: 1 });
    expect(await workers[0].evaluate(() => { const c = document.querySelector('[data-testid=conveyor-card]'); return c.scrollHeight <= c.clientHeight + 1; })).toBe(true);
    await workers[0].setViewportSize({ width: 390, height: 844 });
    // the TV: both belts, the first toy on each, every station signed with who works it
    await expect(tv.getByTestId('conveyor-toy-A-0')).toBeVisible();
    await expect(tv.getByTestId('conveyor-toy-B-0')).toBeVisible();
    await expect(tv.getByTestId('conveyor-zone-A-0')).toContainText('Wheels');
    await expect(tv.getByTestId('conveyor-zone-A-0')).toContainText(nameOf(A.crew[0][0]));

    // the phone says what the toy coming to its station wants
    const first = pages[nameOf(A.crew[0][0])];
    await expect(first.getByTestId('conveyor-ask-0')).toContainText(/Next one wants|It wants/);
    // the toy glides on the TV with no message per frame: it has moved along within a second
    const x0 = (await tv.getByTestId('conveyor-toy-A-0').boundingBox()).x;
    await tv.waitForTimeout(1000);
    expect((await tv.getByTestId('conveyor-toy-A-0').boundingBox()).x).toBeGreaterThan(x0 + 20);

    // team A works its stations for a while; team B does nothing
    let presses = 0, shown = false;
    for (let i = 0; i < 175 && (await state(r.roomCode)).teams[0].delivered < 2; i++) {
      for (const page of workers) presses += await work(page);
      if (!shown && presses >= 2) { shown = true; await shot(tv, 'belt'); await shot(workers[0], 'phone'); }
      await tv.waitForTimeout(150);
    }
    const s1 = await state(r.roomCode);
    expect(s1.teams[0].delivered).toBeGreaterThanOrEqual(2);
    expect(s1.teams[0].rejected).toBe(0);
    expect(s1.teams[1].delivered).toBe(0);
    expect(s1.teams[1].rejected).toBeGreaterThanOrEqual(1);             // nobody fitted a thing: into the bin
    await expect(tv.getByTestId('conveyor-sleigh-A')).toContainText(String(s1.teams[0].delivered));
    await expect(workers[0].getByTestId('conveyor-ours')).toHaveText(String(s1.teams[0].delivered));
    // team A's belt runs faster for it; team B's is only as fast as the clock made it
    const speed = async (team) => Number((await tv.getByTestId(`conveyor-speed-${team}`).getAttribute('aria-label')).replace(/\D/g, ''));
    expect(await speed('A')).toBeGreaterThan(await speed('B'));
    await shot(tv, 'faster');

    // a wrong piece: the toy is spoiled and the phone says so
    let slipped = 0;
    let told = false;                                                   // (the words stay 1.4 s: look for them as we go, not afterwards)
    const wrong = workers[0].locator('.cv-said').filter({ hasText: 'Wrong piece' });
    for (let i = 0; i < 80 && !(slipped && told); i++) { slipped = (await work(workers[0], true)) || slipped; told = told || (await wrong.count()) > 0; await tv.waitForTimeout(150); told = told || (await wrong.count()) > 0; }
    expect(slipped).toBeGreaterThan(0);
    expect(told).toBe(true);

    // the bell: more toys on the sleigh wins, on the TV and on every phone
    await H.dbPut(r.roomCode, 'arena/state/endsAt', Date.now());
    await H.dbPut(r.roomCode, 'arena/deadlines', [{ id: 'end', at: Date.now() }]);
    await expect.poll(async () => (await H.dbGet(r.roomCode, 'arena')).lifecycle, { timeout: 10000 }).toBe('completed');
    await expect(tv.getByTestId('conveyor-clock')).toHaveText('Team A wins!');
    await expect(workers[0].getByTestId('conveyor-card')).toContainText('Your team won!');
    await expect(idle[0].getByTestId('conveyor-card')).toContainText('The other team won.');
    await shot(tv, 'done');
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
