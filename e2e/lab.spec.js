// @ts-check
// The Game Lab: from the lobby, the host tries Reindeer Ring Toss on the real phones and the TV,
// with no gift, bids or duel. The result stays on the screens; nothing is won (no Party points, no
// chips); the host plays again, then goes back to the lobby and starts the real game.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob'];
const endStage = (room) => H.dbPut(room, 'arena/deadlines', [{ id: 'stage', at: Date.now() }]);

/** Run the clock out: every pending deadline of the lab game falls due now, until it's over. */
async function runOut(room, page) {
  for (let i = 0; i < 80; i++) {
    const a = await H.dbGet(room, 'arena');
    if (!a || a.lifecycle === 'completed' || a.lifecycle === 'cancelled') return a;
    if ((a.deadlines || []).length) await H.dbPut(room, 'arena/deadlines', a.deadlines.map((d) => ({ ...d, at: Date.now() })));
    await page.waitForTimeout(250);
  }
  throw new Error('the lab game never finished');
}

/** A brisk flick straight up the phone's flick surface (pointer events at a finger's pace). */
const flick = (page) => page.getByTestId('ringtoss-pad').evaluate((pad) => new Promise((done) => {
  const r = pad.getBoundingClientRect(), x0 = r.left + r.width / 2, y0 = r.bottom - 120, dist = 0.45 * window.innerHeight;
  const fire = (type, f) => pad.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 9, pointerType: 'touch', isPrimary: true, clientX: x0, clientY: y0 - dist * f }));
  fire('pointerdown', 0);
  let k = 0;
  const step = () => { k++; const f = Math.min(1, (k * 16) / 100); fire(f >= 1 ? 'pointerup' : 'pointermove', f); if (f >= 1) done(); else setTimeout(step, 16); };
  setTimeout(step, 16);
}));

test('the Game Lab: a mini-game from the lobby, practice only, play again, back to the lobby', async ({ browser }) => {
  test.setTimeout(120000);
  const r = await H.createGame(browser, {});
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    const chipsBefore = Object.fromEntries(Object.entries(await H.dbGet(r.roomCode, 'players')).map(([id, p]) => [id, p.chips]));

    // the host picks the game; two players: head to head by default
    await r.host.getByTestId('lab-open').click();
    await r.host.getByTestId('lab-game').selectOption('reindeer_ring_toss');
    await expect(r.host.getByTestId('lab-mode')).toHaveValue('direct_duel');
    await r.host.getByTestId('lab-start').click();
    await expect(r.host.getByTestId('lab-modal')).toBeHidden();

    // the phones and the TV play it; the room is still in the lobby
    for (const n of NAMES) await expect(pages[n].getByTestId('ringtoss-card')).toBeVisible();
    await expect(tv.getByTestId('ringtoss-stage')).toBeVisible();
    await expect(tv.locator('#phase-badge')).toContainText('GAME LAB');
    const g1 = await H.dbGet(r.roomCode, '');
    expect(g1.phase).toBe('lobby');
    expect(g1.arena).toMatchObject({ gameId: 'reindeer_ring_toss', mode: 'direct_duel', lifecycle: 'active' });
    const first = g1.arena.instanceId;

    // a throw with the buttons, then run the clock out: the result stays on the screens
    const alice = pages.Alice;
    await alice.getByTestId('ringtoss-controls-toggle').click();
    await alice.getByTestId('ringtoss-hold').dispatchEvent('pointerdown');
    await alice.waitForTimeout(400);
    await alice.getByTestId('ringtoss-hold').dispatchEvent('pointerup');
    await expect.poll(async () => ((await H.dbGet(r.roomCode, 'arena')).state.throws || []).length).toBe(1);
    for (let i = 0; i < 8; i++) {
      const a = await H.dbGet(r.roomCode, 'arena');
      if (a.lifecycle === 'completed') break;
      await endStage(r.roomCode);
      await r.host.waitForTimeout(600);
    }
    await expect(r.host.getByTestId('lab-again')).toBeVisible({ timeout: 10000 });
    await r.host.waitForTimeout(1500);                                      // no hand-off comes along
    const g2 = await H.dbGet(r.roomCode, '');
    expect(g2.arena?.lifecycle).toBe('completed');
    expect(g2.phase).toBe('lobby');
    for (const [id, p] of Object.entries(g2.players)) {
      expect(p.chips).toBe(chipsBefore[id]);
      expect(p.partyPoints || 0).toBe(0);
    }

    // play again: a fresh game, same set-up
    await r.host.getByTestId('lab-again').click();
    await expect.poll(async () => (await H.dbGet(r.roomCode, 'arena'))?.instanceId).not.toBe(first);
    expect((await H.dbGet(r.roomCode, 'arena')).lifecycle).toBe('active');

    // the replay's first ring flies its whole arc on the TV (a stale frame loop once timed it from the
    // last game's ring #1, so it was "already down" and just appeared): many frames, real movement
    await alice.getByTestId('ringtoss-controls-toggle').click();     // back to flicking: the whole screen
    const arc = tv.evaluate(() => new Promise((done) => {
      const ys = []; const t0 = performance.now();
      const f = () => { const r = document.querySelector('.rt-flights:not(.behind) .rt-ring'); if (r) ys.push(+r.getAttribute('cy')); if (performance.now() - t0 < 2500) requestAnimationFrame(f); else done(ys); };
      requestAnimationFrame(f);
    }));
    await flick(alice);
    const ys = await arc;
    expect(ys.length).toBeGreaterThan(20);
    expect(new Set(ys.map((y) => Math.round(y))).size).toBeGreaterThan(15);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(100);    // it rose and fell, not a dot

    // back to the lobby (stops it), then the real game starts as usual
    await r.host.getByTestId('lab-end').click();
    await expect(r.host.locator('#btn-start-game')).toBeVisible();
    for (const n of NAMES) await expect(pages[n].getByTestId('ringtoss-card')).toBeHidden();
    const g3 = await H.dbGet(r.roomCode, '');
    expect(g3.currentTurn ?? null).toBeNull();
    expect(g3.arena ?? null).toBeNull();
    await H.startGame(r.host, r.roomCode);
    expect((await H.dbGet(r.roomCode, 'phase'))).toBe('main');
    expect([...r.errors, ...phoneErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});

test('the Game Lab: every game runs to its result on the TV, the console and the phones (Gremlin, Conveyor, solo Ring Toss)', async ({ browser }) => {
  test.setTimeout(150000);
  const names = ['Alice', 'Bob', 'Charlie', 'Dana'];
  const r = await H.createGame(browser, {});
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, names);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const tvErrors = [];
  try {
    const tv = await tvCtx.newPage();
    tv.on('pageerror', (e) => tvErrors.push(String(e)));
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    const runs = [
      { game: 'pass_the_gremlin', mode: 'party_with_duelists', card: 'gremlin-card' },
      { game: 'gift_conveyor', mode: 'captain_teams', card: 'conveyor-card' },
      { game: 'reindeer_ring_toss', mode: 'free_for_all', card: 'ringtoss-card', only: 'Charlie' },
    ];
    for (const run of runs) {
      await r.host.getByTestId('lab-open').click();
      await r.host.getByTestId('lab-game').selectOption(run.game);
      if (run.only) for (const n of names.filter((x) => x !== run.only)) await r.host.getByTestId(`lab-player-${n}`).uncheck();
      await expect(r.host.getByTestId('lab-mode')).toHaveValue(run.mode);
      await r.host.getByTestId('lab-start').click();
      const playing = run.only ? [run.only] : names;
      for (const n of playing) await expect(pages[n].getByTestId(run.card)).toBeVisible();
      const a = await runOut(r.roomCode, r.host);
      expect(a.lifecycle).toBe('completed');
      // the result stays up everywhere, and nothing crashed
      await expect(r.host.getByTestId('lab-again')).toBeVisible();
      await expect(tv.locator('#phase-badge')).toContainText('GAME LAB');
      for (const n of playing) await expect(pages[n].getByTestId(run.card)).toBeVisible();
      await r.host.waitForTimeout(500);
      await r.host.getByTestId('lab-end').click();
      await expect(r.host.locator('#btn-start-game')).toBeVisible();
    }
    const g = await H.dbGet(r.roomCode, '');
    expect(g.teamPairs ?? null).toBeNull();                                   // the lab doesn't remember teammates
    for (const p of Object.values(g.players)) expect(p.partyPoints || 0).toBe(0);
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
