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

test('Elf Tower Wars: build a tower, send toys, they march on the TV and hurt the other workshop; a fallen workshop loses', async ({ browser }) => {
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
    for (const page of [one, two]) for (const card of ['soldier', 'mouse', 'tank', 'zap', 'shield', 'combo', 'snow', 'cannon', 'frost', 'patch']) await expect(page.getByTestId(`tower-card-${card}`)).toBeVisible();   // the same hand for everyone
    await expect(one.getByTestId('tower-hint')).toContainText('A snow tower stops it');     // a card is picked from the start: the hint says what it is for and what answers it
    await expect(tv.getByTestId('tower-hp-A')).toContainText('20');
    for (const id of [...A.members, ...B.members]) await expect(tv.getByTestId(`tower-member-${id}`)).toBeVisible();   // everyone's own energy

    // a tower: its card, then one of the team's spots; it stands on the TV and the energy is spent (a cannon costs too much yet)
    await expect(one.getByTestId('tower-card-cannon')).toHaveClass(/poor/);
    await tap(one, 'tower-card-snow');
    await expect(one.getByTestId('tower-card-snow')).toHaveClass(/picked/);
    await expect(one.getByTestId('tower-spot-btn-0-0')).toHaveClass(/open/);
    await tap(one, 'tower-spot-btn-0-0');
    await H.waitForGame(r.roomCode, (g) => ((g.arena.state.field.sides[0].towers || []).length === 1 ? g : false), { label: 'tower built' });
    await expect(tv.getByTestId('tower-spot-0-0-0')).toHaveClass(/built/);
    await expect(one.getByTestId('tower-spot-btn-0-0')).toHaveClass(/built/);
    await expect(one.getByTestId('tower-hint')).toContainText('Tap yours again to make it better');

    // the same phone, another card: a toy soldier down the middle; it marches on the TV, towards team B's workshop
    await tap(one, 'tower-card-soldier');
    await expect(one.getByTestId('tower-lane-1')).toHaveClass(/open/, { timeout: 10000 });       // once its energy is back
    await tap(one, 'tower-lane-1');
    const sent = await H.waitForGame(r.roomCode, (g) => ((g.arena.state.field.units || []).length === 1 ? g : false), { label: 'toy sent' });
    const unit = tv.getByTestId(`tower-unit-${sent.arena.state.field.units[0].id}`);
    await expect(unit).toBeVisible();
    const x0 = (await unit.boundingBox()).x;
    await tv.waitForTimeout(1200);
    expect((await unit.boundingBox()).x).toBeGreaterThan(x0 + 10);          // it glides between the room's steps
    await shot(tv, 'tv-march'); await shot(one, 'phone');

    // nothing in its way: it arrives and team B's workshop is hurt (their phones too)
    const hurt = await H.waitForGame(r.roomCode, (g) => (g.arena.state.field.sides[1].hp < 20 ? g : false), { label: 'workshop hurt', timeout: 30000 });
    expect(hurt.arena.state.field.sides[1].hp).toBe(19);
    await expect(tv.getByTestId('tower-hp-B')).toContainText('19');
    await expect(pages[nameOf(B.members[0])].getByTestId('tower-score')).toContainText('Us 19');

    // team B's workshop is down to its last (the long battle is the unit tests' job): one more toy ends it
    const a1 = await arena(r.roomCode);
    const sides = [a1.state.field.sides[0], { ...a1.state.field.sides[1], hp: 1 }];
    await H.dbPut(r.roomCode, 'arena/state', { ...a1.state, field: { ...a1.state.field, sides } });
    await H.dbPut(r.roomCode, 'arena/stateVersion', a1.stateVersion + 50);
    await expect(two.getByTestId('tower-lane-2')).toHaveClass(/open/, { timeout: 10000 });      // the soldier card is the one picked to begin with
    await tap(two, 'tower-lane-2');
    const done = await H.waitForGame(r.roomCode, (g) => (g.arena?.lifecycle === 'completed' ? g : false), { label: 'workshop fell', timeout: 40000 });
    expect(done.arena.result).toMatchObject({ status: 'resolved', reason: 'the other workshop fell' });
    expect(done.arena.result.duel.winningDuelistId).toBe(A.captainId);
    await expect(tv.getByTestId('tower-end')).toContainText('Team A wins!');
    await expect(two.getByTestId('tower-word')).toContainText('Your team won!');
    await shot(tv, 'tv-end');
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
