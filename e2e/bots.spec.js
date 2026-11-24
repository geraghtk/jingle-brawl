// @ts-check
// Practice bots and the party games in the Game Lab: a host with one phone (or none) tries a game that
// needs a table. The bots are added in the lobby, play on the room server's clock, show on the TV like
// anyone else, and are not players of the real game.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

async function openTv(browser, room) {
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const tv = await ctx.newPage();
  await tv.goto(`${H.BASE}/tv.html?room=${room}`);
  return { ctx, tv };
}

test('Whack-an-Elf from the Game Lab, one person against two practice bots: practice only, the bots score, nothing is won', async ({ browser }) => {
  test.setTimeout(120000);
  const r = await H.createGame(browser, {});
  const { pages, contexts } = await H.joinPlayers(browser, r.roomCode, ['Alice']);
  const { ctx: tvCtx, tv } = await openTv(browser, r.roomCode);
  try {
    // the host adds two bots from the lab's picker: they are in the lobby, marked as bots, and ticked to play
    await r.host.getByTestId('lab-open').click();
    await r.host.getByTestId('lab-add-bot').click();
    await r.host.getByTestId('lab-add-bot').click();
    await expect(r.host.locator('[data-testid^="lab-player-Bot "]')).toHaveCount(2);
    for (const box of await r.host.locator('[data-testid^="lab-player-Bot "]').all()) await expect(box).toBeChecked();
    await expect(tv.locator('#lobby-roster-list')).toContainText('Practice bot');
    const players = await H.dbGet(r.roomCode, 'players');
    const bots = Object.keys(players).filter((id) => players[id].isBot);
    expect(bots).toHaveLength(2);

    // Whack-an-Elf is on the list (it is not one of the runtime's games); everyone picked plays
    await r.host.getByTestId('lab-game').selectOption('whack_elf');
    await r.host.getByTestId('lab-start').click();
    await expect(tv.getByTestId('whack-stage')).toContainText('Practice');
    await expect(pages.Alice.getByTestId('party-card')).toContainText('Practice');
    await expect(tv.locator('#phase-badge')).toContainText('GAME LAB');
    expect((await H.dbGet(r.roomCode, '')).phase).toBe('lobby');

    // playing: the bots' scores climb on the TV's race
    await H.dbPatch(r.roomCode, { 'minigame/phaseEndsAt': Date.now() });
    await expect(tv.getByTestId(`whack-racer-${bots[0]}`)).toBeVisible();
    await expect.poll(async () => {
      const s = (await H.dbGet(r.roomCode, 'minigame/scores')) || {};
      return bots.every((b) => (s[b]?.score || 0) > 0);
    }, { timeout: 20000 }).toBe(true);
    await expect(pages.Alice.getByTestId('whack-score')).toBeVisible();

    // the end: results stay up, nobody won anything, the host can play again or leave
    await H.dbPatch(r.roomCode, { 'minigame/phaseEndsAt': Date.now() });
    await expect(tv.getByTestId('whack-stage')).toContainText('Practice: nothing won', { timeout: 15000 });
    await expect(pages.Alice.getByTestId('party-card')).toContainText('Practice: nothing won');
    await expect(r.host.getByTestId('lab-again').first()).toBeVisible();
    const after = await H.dbGet(r.roomCode, 'players');
    for (const p of Object.values(after)) { expect(p.partyPoints || 0).toBe(0); expect(p.chips).toBe(players[Object.keys(players)[0]].chips); }
    await r.host.getByTestId('lab-end').first().click();
    await expect(tv.locator('#lobby-roster-list')).toBeVisible();

    // bots are not players of the real game: one person and two bots can't start it
    await expect(r.host.locator('#btn-start-game')).toBeDisabled();
    await expect(r.host.locator('#lobby-start-hint')).toContainText('at least 2 players');
  } finally {
    await tvCtx.close();
    for (const c of contexts) await c.close();
    await r.hostCtx.close();
  }
});

test('bots alone run the Great Gift Conveyor on the TV: the factory shows who works each machine, and the sleigh fills', async ({ browser }) => {
  test.setTimeout(150000);
  const r = await H.createGame(browser, {});
  const { ctx: tvCtx, tv } = await openTv(browser, r.roomCode);
  const errors = [];
  tv.on('pageerror', (e) => errors.push(e.message));
  try {
    await r.host.getByTestId('lab-open').click();
    for (let i = 0; i < 4; i++) {
      await r.host.getByTestId('lab-add-bot').click();
      await expect(r.host.locator('[data-testid^="lab-player-Bot "]')).toHaveCount(i + 1);
    }
    await r.host.getByTestId('lab-game').selectOption('gift_conveyor');
    await r.host.getByTestId('lab-start').click();

    // each line is a factory: a machine per station with its sign and its worker, toys on the belt
    await expect(tv.getByTestId('conveyor-stage')).toBeVisible();
    for (const team of ['A', 'B']) {
      await expect(tv.getByTestId(`conveyor-zone-${team}-0`)).toContainText('Wheels');
      await expect(tv.getByTestId(`conveyor-zone-${team}-3`)).toContainText('Bow');
      await expect(tv.getByTestId(`conveyor-zone-${team}-0`)).toContainText('Bot ');
      await expect(tv.getByTestId(`conveyor-toy-${team}-0`)).toBeVisible();
    }

    // the bots build toys: a delivery lands a gift on that team's sleigh
    let state;
    await expect.poll(async () => {
      state = (await H.dbGet(r.roomCode, 'arena/state')) || { teams: [] };
      return state.teams.some((t) => t.delivered > 0);
    }, { timeout: 70000, intervals: [500] }).toBe(true);
    const ahead = state.teams.find((t) => t.delivered > 0);
    await expect(tv.getByTestId(`conveyor-sleigh-${ahead.teamId}`).locator('.cvf-gift').first()).toBeVisible();
    await expect(tv.getByTestId(`conveyor-sleigh-${ahead.teamId}`)).toContainText('of 8');
    expect(errors).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
  }
});

test('Sketch-Off fills the screens: the phone is the pad, and the TV shows the drawings growing with no names, then the gallery', async ({ browser }) => {
  test.setTimeout(150000);
  const r = await H.createGame(browser, {});
  const { pages, contexts } = await H.joinPlayers(browser, r.roomCode, ['Alice']);
  const { ctx: tvCtx, tv } = await openTv(browser, r.roomCode);
  try {
    await r.host.getByTestId('lab-open').click();
    for (let i = 0; i < 2; i++) { await r.host.getByTestId('lab-add-bot').click(); await expect(r.host.locator('[data-testid^="lab-player-Bot "]')).toHaveCount(i + 1); }
    await r.host.getByTestId('lab-game').selectOption('sketch_off');
    await r.host.getByTestId('lab-start').click();
    // full screen on the TV like the other games: the prompt first
    await expect(tv.getByTestId('party-full')).toBeVisible();
    await expect(tv.getByTestId('sketch-stage')).toContainText('Everyone draws');
    await H.dbPatch(r.roomCode, { 'minigame/phaseEndsAt': Date.now() });

    // drawing: an easel each, nothing on them yet; the phone's pad takes the screen
    await expect(tv.getByTestId('sketch-wall').locator('.sk-easel')).toHaveCount(3);
    const pad = pages.Alice.getByTestId('sketch-pad');
    await expect(pad).toBeVisible();
    const view = pages.Alice.viewportSize(), box = await pad.boundingBox();
    expect(box.width).toBeGreaterThan(view.width * 0.9);                  // as wide as the phone
    expect(Math.abs(box.width - box.height)).toBeLessThan(2);             // and square

    // Alice draws a line: within a couple of seconds it is on the TV's wall, on an easel with no name
    await pages.Alice.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.2);
    await pages.Alice.mouse.down();
    await pages.Alice.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.7, { steps: 8 });
    await pages.Alice.mouse.up();
    await expect.poll(async () => !!(await H.dbGet(r.roomCode, 'minigame/live'))?.[Object.keys(await H.dbGet(r.roomCode, 'players')).find((id) => !id.startsWith('bot_'))], { timeout: 8000 }).toBe(true);
    await expect(tv.getByTestId('sketch-wall').locator('.sk-easel:not(.done) .sk-paper > img')).toHaveCount(1, { timeout: 8000 });
    await expect(tv.getByTestId('sketch-wall')).not.toContainText('Alice');
    await expect(tv.getByTestId('sketch-wall')).not.toContainText('Bot ');
    // another phone never gets it: Alice's own view carries no wall and no snapshots
    const mine = await pages.Alice.evaluate(() => document.body.innerHTML.includes('sk-wall'));
    expect(mine).toBe(false);

    // she hands in; the bots too (they draw a scribble); then the gallery fills the TV, numbered
    await pages.Alice.getByTestId('sketch-done').click();
    await expect(tv.getByTestId('sketch-gallery').locator('.sk-paper > img')).toHaveCount(3, { timeout: 60000 });
    expect(await H.dbGet(r.roomCode, 'minigame/live')).toBeFalsy();                       // the snapshots are thrown away
    const tile = await tv.getByTestId('sketch-gallery').locator('.sk-paper').first().boundingBox();
    expect(tile.height).toBeGreaterThan(900 * 0.45);                                      // three drawings: one big row
    // ...with a margin: no drawing touches the edge of the wall (they used to reach the frame of the TV)
    const wall = await tv.getByTestId('sketch-gallery').boundingBox();
    for (const paper of await tv.getByTestId('sketch-gallery').locator('.sk-paper').all()) {
      const b = await paper.boundingBox(), m = wall.height * 0.02;
      expect(b.x - wall.x).toBeGreaterThan(m); expect(wall.x + wall.width - (b.x + b.width)).toBeGreaterThan(m);
      expect(b.y - wall.y).toBeGreaterThan(m); expect(wall.y + wall.height - (b.y + b.height)).toBeGreaterThan(m);
    }
    await expect(pages.Alice.locator('.pg-gallery figure')).toHaveCount(3);
  } finally {
    await tvCtx.close();
    for (const c of contexts) await c.close();
    await r.hostCtx.close();
  }
});
