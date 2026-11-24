// @ts-check
// Nutcracker Knockout on real pages, from the Game Lab: the two duelists' phones are controllers
// (a jab, a haymaker, hold to block, tap to dodge), the TV draws the ring, and the room decides every
// blow. A part takes two blows (a haymaker is both); the last part is the head, and that is the round.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];
const shot = async (page, name) => { if (process.env.NK_SHOTS) await page.screenshot({ path: `${process.env.NK_SHOTS}/${name}.png` }); };
const state = async (room) => (await H.dbGet(room, 'arena')).state;
/** The room's pending deadlines fall due now (the intro, a punch in the air, the pause after a knockout). */
const dueNow = async (room) => {
  const a = await H.dbGet(room, 'arena');
  if ((a.deadlines || []).length) await H.dbPut(room, 'arena/deadlines', a.deadlines.map((d) => ({ ...d, at: Date.now() })));
};
/** Press a controller pad the way a thumb does. */
const tap = (page, id) => page.getByTestId(id).dispatchEvent('pointerdown', { pointerId: 3, pointerType: 'touch', isPrimary: true });

test('Nutcracker Knockout: phones are controllers, the room lands the punches, parts come off, the head last', async ({ browser }) => {
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

    // the host starts it in the lab for Alice and Bob; Charlie is at the table but not in the fight
    await r.host.getByTestId('lab-open').click();
    await r.host.getByTestId('lab-game').selectOption('nutcracker_knockout');
    await r.host.getByTestId('lab-player-Charlie').setChecked(false);
    await r.host.getByTestId('lab-start').click();
    await expect(tv.getByTestId('robot-stage')).toBeVisible();
    await expect(tv.getByTestId('robot-splash')).toHaveText('Round 1');
    const s0 = await state(r.roomCode);
    expect(s0.stage).toBe('intro');
    const [first, second] = s0.order;                               // fighter 0 (left of the TV) and fighter 1
    const nameOf = (id) => NAMES.find((n) => p[n].id === id);
    const A = pages[nameOf(first)], B = pages[nameOf(second)];

    // the pads follow the robot's side of the ring: jab over haymaker in the column facing the other robot, block over
    // dodge behind. Fighter 0 starts on the left of the TV (punches under the right thumb), fighter 1 on the right.
    const box = async (page, id) => page.getByTestId(id).boundingBox();
    const layout = async (page, side) => {
      await expect(page.getByTestId('robot-pads')).toHaveAttribute('data-side', side);
      await page.getByTestId('robot-pads').evaluate((el) => Promise.all(el.getAnimations().map((a) => a.finished)));   // (they pop in: measure them at rest)
      const [jab, heavy, block, dodge] = [await box(page, 'robot-punch-jab'), await box(page, 'robot-punch-heavy'), await box(page, 'robot-block'), await box(page, 'robot-dodge')];
      expect(Math.abs(jab.x - heavy.x)).toBeLessThan(2);                // jab and haymaker: one above the other
      expect(Math.abs(block.x - dodge.x)).toBeLessThan(2);              // block and dodge: one above the other
      expect(jab.y).toBeLessThan(heavy.y);                              // jab and block on top
      expect(block.y).toBeLessThan(dodge.y);
      expect(Math.abs(jab.y - block.y)).toBeLessThan(2);
      if (side === 'left') expect(jab.x).toBeGreaterThan(block.x); else expect(jab.x).toBeLessThan(block.x);   // fists towards the other robot
    };
    await layout(A, 'left');
    await layout(B, 'right');
    await expect(A.getByTestId('robot-side')).toContainText('Left');
    // the next round they change sides: the TV's corners and both phones' pads follow
    const sw = await H.dbGet(r.roomCode, 'arena');
    await H.dbPut(r.roomCode, 'arena/state/left', 1);
    await H.dbPut(r.roomCode, 'arena/stateVersion', sw.stateVersion + 20);
    await layout(A, 'right');
    await layout(B, 'left');
    await expect(tv.locator('.nk-corner.left')).toHaveAttribute('data-testid', `robot-corner-${second}`);
    await H.dbPut(r.roomCode, 'arena/state/left', 0);
    await H.dbPut(r.roomCode, 'arena/stateVersion', sw.stateVersion + 40);
    await layout(A, 'left');
    // before the bell the pads do nothing; a phone that isn't fighting is told to watch
    await expect(A.getByTestId('robot-punch-jab')).toBeDisabled();
    await expect(A.getByTestId('robot-dodge')).toBeDisabled();
    const watcher = NAMES.find((n) => p[n].id !== first && p[n].id !== second);
    if (watcher) { await expect(pages[watcher].getByTestId('robot-card')).toContainText('Watch the TV'); await expect(pages[watcher].getByTestId('robot-card')).toHaveClass(/(^| )ap( |$)/); if (process.env.NK_SHOTS) await pages[watcher].screenshot({ path: `${process.env.NK_SHOTS}/watcher.png` }); }   // the same full-screen frame as the players
    await tv.waitForTimeout(1200);                                  // the 3D scene has loaded and drawn
    await shot(tv, 'intro');

    await dueNow(r.roomCode);                                       // the bell
    await expect.poll(async () => (await state(r.roomCode)).stage).toBe('fighting');
    await expect(A.getByTestId('robot-punch-jab')).toBeEnabled();
    await expect(A.getByTestId('robot-word')).toHaveText('Jab, haymaker, block or dodge');
    expect((await state(r.roomCode)).fight.f.map((f) => f.health)).toEqual([18, 18]);                    // six parts, three blows each

    // a jab: the phone only says "jab"; the room lands it a moment later, and the hat is knocked loose (not off)
    await tap(A, 'robot-punch-jab');
    await expect.poll(async () => (await state(r.roomCode)).fight.f[1].health, { timeout: 5000 }).toBe(17);
    await expect(tv.getByTestId(`robot-parts-${second}`)).toHaveAttribute('aria-label', '6 of 6 parts left, the hat is loose');
    await expect(B.getByTestId('robot-word')).toHaveText(/Ouch!|Your hat is loose/);
    await tv.waitForTimeout(300);
    await shot(tv, 'loose');

    // a haymaker: slow, and it takes a whole part (the loose hat goes, and the next part is loosened)
    await expect(A.getByTestId('robot-punch-heavy')).toBeEnabled({ timeout: 5000 });
    await tap(A, 'robot-punch-heavy');
    await expect.poll(async () => (await state(r.roomCode)).fight.thrown?.[0] || 0).toBe(2);
    await tv.waitForTimeout(450);
    await shot(tv, 'punch');
    await expect.poll(async () => (await state(r.roomCode)).fight.f[1].health, { timeout: 5000 }).toBe(14);
    await expect(tv.getByTestId(`robot-parts-${second}`)).toHaveAttribute('aria-label', '5 of 6 parts left, the left shoulder plate is loose');
    await expect(B.getByTestId('robot-my-parts')).toHaveAttribute('aria-label', 'Your robot has 5 of 6 parts left');
    // and a health bar for each robot, on the TV and on both phones: 14 of 18 left
    await expect(tv.getByTestId(`robot-health-${second}`)).toHaveAttribute('aria-label', 'Health 78%');
    await expect(tv.getByTestId(`robot-health-${first}`)).toHaveAttribute('aria-label', 'Health 100%');
    expect(await B.getByTestId('robot-my-health').locator('i').evaluate((el) => el.style.width)).toMatch(/^77\.7/);
    expect(await A.getByTestId('robot-their-health').locator('i').evaluate((el) => el.style.width)).toMatch(/^77\.7/);
    await tv.waitForTimeout(350);
    await shot(tv, 'hit');

    // a block: B holds it, A's jab is stopped and A is the one left open
    await B.getByTestId('robot-block').dispatchEvent('pointerdown', { pointerId: 5, pointerType: 'touch', isPrimary: true });
    await expect.poll(async () => !!(await state(r.roomCode)).fight.f[1].guard).toBe(true);
    await expect(B.getByTestId('robot-word')).toHaveText('Blocking');
    await B.waitForTimeout(400);                                    // held long enough that it is a block, not a parry
    await expect(A.getByTestId('robot-punch-jab')).toBeEnabled({ timeout: 5000 });
    await tap(A, 'robot-punch-jab');
    await tv.waitForTimeout(300);
    await shot(tv, 'block');
    await expect.poll(async () => (await state(r.roomCode)).fight.events?.at(-1)?.kind, { timeout: 5000 }).toBe('blocked');
    expect((await state(r.roomCode)).fight.f[1].health).toBe(14);
    await B.getByTestId('robot-block').dispatchEvent('pointerup', { pointerId: 5 });
    await expect.poll(async () => !!(await state(r.roomCode)).fight.f[1].guard).toBe(false);

    // a dodge: A throws a haymaker, B taps Dodge as it comes in: it misses, and A is left stumbling
    await expect(A.getByTestId('robot-punch-heavy')).toBeEnabled({ timeout: 5000 });
    await tap(A, 'robot-punch-heavy');
    await expect(B.getByTestId('robot-word')).not.toContainText(/coming|dodge!|block!/i);   // the phone never says what to press or when: the wind-up is on the TV
    await B.waitForTimeout(330);
    await tap(B, 'robot-dodge');
    await tv.waitForTimeout(200);
    await shot(tv, 'dodge');
    await expect.poll(async () => (await state(r.roomCode)).fight.events?.at(-1)?.kind, { timeout: 5000 }).toBe('dodged');
    expect((await state(r.roomCode)).fight.f[1].health).toBe(14);
    await expect(A.getByTestId('robot-word')).toHaveText('Missed! You are open');

    // the rest of the round: A lands until the head comes off (the room has one hit left to give)
    const before = await H.dbGet(r.roomCode, 'arena');
    await H.dbPut(r.roomCode, 'arena/state/fight/f/1/health', 1);
    await H.dbPut(r.roomCode, 'arena/stateVersion', before.stateVersion + 1);
    await expect(tv.getByTestId(`robot-parts-${second}`)).toHaveAttribute('aria-label', '1 of 6 parts left, the head is loose');
    await tv.waitForTimeout(1200);
    await shot(tv, 'stripped');
    await expect(A.getByTestId('robot-punch-jab')).toBeEnabled({ timeout: 5000 });
    await tap(A, 'robot-punch-jab');
    await expect.poll(async () => (await state(r.roomCode)).stage, { timeout: 5000 }).toBe('ko');
    await expect(tv.getByTestId('robot-splash')).toHaveText('Knockout!');
    await expect(A.getByTestId('robot-word')).toHaveText('You took the head off!');
    await expect(B.getByTestId('robot-word')).toHaveText('You lost your head');
    await tv.waitForTimeout(700);
    await shot(tv, 'knockout');
    await shot(B, 'phone');
    const s1 = await state(r.roomCode);
    expect(s1.wins[first]).toBe(1);
    expect(s1.last).toMatchObject({ round: 1, winner: first, loser: second, how: 'knockout' });

    // round 2 starts with both robots whole again
    await dueNow(r.roomCode);
    await expect(tv.getByTestId('robot-splash')).toHaveText('Round 2');
    await expect(tv.getByTestId(`robot-parts-${second}`)).toHaveAttribute('aria-label', '6 of 6 parts left');
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
