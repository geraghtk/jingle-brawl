// @ts-check
// Elf Tower Wars on real pages (from the Game Lab): four at the table, so the two captains lead teams
// of two. Nobody has a job: every phone has the same hand of cards and its own energy (pick a card,
// tap where it goes). A tower built and a toy sent, by the same phone, show on the TV; a toy that
// arrives hurts the other workshop; a workshop that falls ends it for the other team.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie', 'Dana'];
const shot = async (page, name) => { if (process.env.TW_SHOTS) await page.screenshot({ path: `${process.env.TW_SHOTS}/${name}.png` }); };
const arena = (room) => H.dbGet(room, 'arena');
const tap = (page, id) => page.getByTestId(id).dispatchEvent('pointerdown', { pointerId: 1, pointerType: 'touch' });

test('Elf Tower Wars: plan (a tower goes up, toys line up at the door), then the march on the TV with the phones quiet; a fallen workshop loses', async ({ browser }) => {
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
    await r.host.getByTestId('lab-game').selectOption('elf_tower_wars');
    await expect(r.host.getByTestId('lab-mode')).toHaveValue('captain_teams');
    await r.host.getByTestId('lab-start').click();
    await expect(tv.getByTestId('tower-stage')).toBeVisible();

    const s0 = (await arena(r.roomCode)).state;
    expect(s0.teams.map((t) => t.members.length)).toEqual([2, 2]);
    const [A, B] = s0.teams;
    const one = pages[nameOf(A.members[0])], two = pages[nameOf(A.members[1])];
    await expect(one.getByTestId('tower-my-team')).toContainText('Team A');
    // it opens with a plan: round 1, the six cards and Done on every phone (the reactions are for the march)
    await expect(tv.getByTestId('tower-round')).toHaveText('Round 1');                 // (no "of 5": it goes on until a workshop falls)
    await expect(tv.getByTestId('tower-phase')).toHaveText('Plan');
    await expect(one.getByTestId('tower-phase')).toContainText('Plan');
    for (const page of [one, two]) for (const card of ['soldier', 'mouse', 'tank', 'snow', 'cannon', 'frost']) await expect(page.getByTestId(`tower-card-${card}`)).toBeVisible();   // the same hand for everyone
    await expect(one.getByTestId('tower-card-zap')).toHaveCount(0);
    await expect(one.getByTestId('tower-done')).toBeVisible();
    await expect(one.getByTestId('tower-hint')).toContainText('A snow tower stops it');     // a card is picked from the start: the hint says what it is for and what answers it
    await expect(tv.getByTestId('tower-hp-A')).toContainText('20');
    for (const id of [...A.members, ...B.members]) await expect(tv.getByTestId(`tower-member-${id}`)).toBeVisible();   // everyone's own energy

    // (plenty of time to plan in this test)
    await H.dbPut(r.roomCode, 'arena/state/phaseEndsAt', Date.now() + 120000);
    await H.dbPut(r.roomCode, 'arena/deadlines', [{ id: 'phase', at: Date.now() + 120000 }]);
    // a tower: its card, then one of the team's spots; it stands on the TV at once
    await tap(one, 'tower-card-snow');
    await expect(one.getByTestId('tower-card-snow')).toHaveClass(/picked/);
    await expect(one.getByTestId('tower-spot-btn-0-0')).toHaveClass(/open/);
    await tap(one, 'tower-spot-btn-0-0');
    await H.waitForGame(r.roomCode, (g) => ((g.arena.state.field.sides[0].towers || []).length === 1 ? g : false), { label: 'tower built' });
    await expect(tv.getByTestId('tower-spot-0-0-0')).toHaveClass(/built/);
    await expect(one.getByTestId('tower-spot-btn-0-0')).toHaveClass(/built/);
    // a tower looks its level: built again (here: set to level 2 through the debug API) it is a bigger, grander picture on the TV and the phone
    await expect(tv.getByTestId('tower-spot-0-0-0').locator('img')).toHaveAttribute('src', /snow-a\.webp/);
    const lv = await H.dbGet(r.roomCode, 'arena');
    const up = [{ ...lv.state.field.sides[0], towers: lv.state.field.sides[0].towers.map((t) => ({ ...t, level: 2 })) }, lv.state.field.sides[1]];
    await H.dbPut(r.roomCode, 'arena/state', { ...lv.state, field: { ...lv.state.field, sides: up } });
    await H.dbPut(r.roomCode, 'arena/stateVersion', lv.stateVersion + 20);
    await expect(tv.getByTestId('tower-spot-0-0-0').locator('img')).toHaveAttribute('src', /snow2-a\.webp/, { timeout: 8000 });
    await expect(one.getByTestId('tower-spot-btn-0-0').locator('img')).toHaveAttribute('src', /snow2-a\.webp/);

    // every path starts at a gate at both workshops (the middle one's stands clear of the building: toys used to slide out from under the house)
    for (const side of [0, 1]) for (const lane of [0, 1, 2]) await expect(tv.getByTestId(`tower-gate-${side}-${lane}`)).toBeVisible();

    // the other teammate lines a soldier up at the middle path: it waits at the door (on the TV too), it does not walk yet
    await tap(two, 'tower-lane-1');                                           // the soldier card is the one picked to begin with
    await H.waitForGame(r.roomCode, (g) => ((g.arena.state.queue || []).length === 1 ? g : false), { label: 'toy lined up' });
    await expect(tv.getByTestId('tower-queue-0-1')).toBeVisible();
    await expect(two.getByTestId('tower-lane-info-1')).toContainText('yours 1');
    await expect(tv.getByTestId('tower-waiting')).toContainText('Planning');   // the TV says who the table is waiting for
    expect(((await arena(r.roomCode)).state.field.units || []).length).toBe(0);
    await shot(tv, 'tv-plan'); await shot(one, 'phone-plan');

    // everyone says Done: nobody waits out the clock, the march starts and the toy walks on the TV
    for (const n of NAMES) await tap(pages[n], 'tower-done');
    const sent = await H.waitForGame(r.roomCode, (g) => (g.arena.state.phase === 'march' && (g.arena.state.field.units || []).length === 1 ? g : false), { label: 'the march', timeout: 10000 });
    await expect(tv.getByTestId('tower-phase')).toHaveText('March!');
    await expect(one.getByTestId('tower-word')).toHaveText('Watch the TV!');
    // the phone is quiet now: three reactions, no cards to plan with
    for (const card of ['zap', 'shield', 'combo']) await expect(one.getByTestId(`tower-card-${card}`)).toBeVisible();
    await expect(one.getByTestId('tower-card-soldier')).toHaveCount(0);
    await expect(one.getByTestId('tower-done')).toHaveCount(0);
    const unit = tv.getByTestId(`tower-unit-${sent.arena.state.field.units[0].id}`);
    await expect(unit).toBeVisible();
    const x0 = (await unit.boundingBox()).x;
    await tv.waitForTimeout(1200);
    expect((await unit.boundingBox()).x).toBeGreaterThan(x0 + 10);          // it glides between the room's steps
    await shot(tv, 'tv-march'); await shot(one, 'phone-march');

    // team B's workshop is down to its last (the long battle is the unit tests' job): the soldier arriving ends it
    const a1 = await arena(r.roomCode);
    const sides = [a1.state.field.sides[0], { ...a1.state.field.sides[1], hp: 1 }];
    const units = a1.state.field.units.map((u) => ({ ...u, pos: 0.9, hp: 99 }));
    await H.dbPut(r.roomCode, 'arena/state', { ...a1.state, phaseEndsAt: Date.now() + 60000, field: { ...a1.state.field, asOf: Date.now(), sides, units } });
    await H.dbPut(r.roomCode, 'arena/deadlines', [{ id: 'phase', at: Date.now() + 60000 }, { id: 'tick', at: Date.now() + 400 }]);
    await H.dbPut(r.roomCode, 'arena/stateVersion', a1.stateVersion + 50);
    const done = await H.waitForGame(r.roomCode, (g) => (g.arena?.lifecycle === 'completed' ? g : false), { label: 'workshop fell', timeout: 40000 });
    expect(done.arena.result).toMatchObject({ status: 'resolved', reason: 'the other workshop fell' });
    expect(done.arena.result.duel.winningDuelistId).toBe(A.captainId);
    await expect(tv.getByTestId('ceremony-headline')).toContainText('team wins!', { timeout: 8000 });   // the one announcement every game ends with
    await expect(tv.getByTestId('tower-end')).toHaveCount(0);
    await expect(two.getByTestId('tower-word')).toContainText('Your team won!');
    await shot(tv, 'tv-end');
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
