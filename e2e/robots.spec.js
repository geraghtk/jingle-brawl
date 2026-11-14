// @ts-check
// Nutcracker Knockout on real pages, from the Game Lab: the two duelists' phones are controllers
// (punch left, punch right, hold to block), the TV draws the ring, and the room decides every hit.
// A part comes off for each hit; the last one is the head, and that is the round.
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

    // before the bell the pads do nothing; a phone that isn't fighting is told to watch
    await expect(A.getByTestId('robot-punch-L')).toBeDisabled();
    const watcher = NAMES.find((n) => p[n].id !== first && p[n].id !== second);
    if (watcher) await expect(pages[watcher].getByTestId('robot-card')).toContainText('Watch the TV');
    await tv.waitForTimeout(1200);                                  // the 3D scene has loaded and drawn
    await shot(tv, 'intro');

    await dueNow(r.roomCode);                                       // the bell
    await expect.poll(async () => (await state(r.roomCode)).stage).toBe('fighting');
    await expect(A.getByTestId('robot-punch-L')).toBeEnabled();
    await expect(A.getByTestId('robot-word')).toHaveText('Punch, or hold Block');

    // a punch: the phone only says "left"; the room starts the wind-up and lands it a moment later
    await tap(A, 'robot-punch-L');
    await expect.poll(async () => (await state(r.roomCode)).fight.thrown?.[0] || 0).toBe(1);
    await tv.waitForTimeout(250);
    await shot(tv, 'punch');
    await expect.poll(async () => (await state(r.roomCode)).fight.f[1].health, { timeout: 5000 }).toBe(5);
    await expect(tv.getByTestId(`robot-parts-${second}`)).toHaveAttribute('aria-label', '5 of 6 parts left');
    await expect(B.getByTestId('robot-my-parts')).toHaveAttribute('aria-label', 'Your robot has 5 of 6 parts left');
    await tv.waitForTimeout(350);
    await shot(tv, 'hit');

    // a block: B holds it, A's punch is stopped and A is the one left open
    await B.getByTestId('robot-block').dispatchEvent('pointerdown', { pointerId: 5, pointerType: 'touch', isPrimary: true });
    await expect.poll(async () => !!(await state(r.roomCode)).fight.f[1].guard).toBe(true);
    await expect(B.getByTestId('robot-word')).toHaveText('Blocking');
    await B.waitForTimeout(400);                                    // held long enough that it is a block, not a parry
    await expect(A.getByTestId('robot-punch-R')).toBeEnabled();
    await tap(A, 'robot-punch-R');
    await tv.waitForTimeout(300);
    await shot(tv, 'block');
    await expect.poll(async () => (await state(r.roomCode)).fight.events?.at(-1)?.kind, { timeout: 5000 }).toBe('blocked');
    expect((await state(r.roomCode)).fight.f[1].health).toBe(5);
    await B.getByTestId('robot-block').dispatchEvent('pointerup', { pointerId: 5 });
    await expect.poll(async () => !!(await state(r.roomCode)).fight.f[1].guard).toBe(false);

    // the rest of the round: A lands until the head comes off (the room has one hit left to give)
    const before = await H.dbGet(r.roomCode, 'arena');
    await H.dbPut(r.roomCode, 'arena/state/fight/f/1/health', 1);
    await H.dbPut(r.roomCode, 'arena/stateVersion', before.stateVersion + 1);
    await expect(tv.getByTestId(`robot-parts-${second}`)).toHaveAttribute('aria-label', '1 of 6 parts left');
    await tv.waitForTimeout(1200);
    await shot(tv, 'stripped');
    await expect(A.getByTestId('robot-punch-L')).toBeEnabled({ timeout: 5000 });
    await tap(A, 'robot-punch-L');
    await expect.poll(async () => (await state(r.roomCode)).stage, { timeout: 5000 }).toBe('ko');
    await expect(tv.getByTestId('robot-splash')).toHaveText('Knockout!');
    await expect(A.getByTestId('robot-word')).toHaveText('You took the head off!');
    await expect(B.getByTestId('robot-word')).toHaveText('You lost your head');
    await tv.waitForTimeout(450);
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
