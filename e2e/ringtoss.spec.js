// @ts-check
// Reindeer Ring Toss on real pages: three at the table, the duel's game is Ring Toss, so everyone
// throws and the two duelists are compared. Phones swipe (pointer events fired inside the page at a
// real finger's pace) or use the hold-to-charge buttons; the TV draws each ring in its thrower's
// colour; the room scores every ring from the swipe alone; the duel goes to the duelist with more.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

/** Swipe up the phone's pad: `angle` radians (negative left), `power` about 1 for a brisk flick,
 *  as finger moves every 16 ms over 100 ms. */
async function swipe(page, angle, power) {
  await page.getByTestId('ringtoss-pad').evaluate((pad, [a, p]) => new Promise((done) => {
    const r = pad.getBoundingClientRect();
    const x0 = r.left + r.width / 2, y0 = r.bottom - 40;
    const dist = p * 3 * 0.1 * window.innerHeight;                    // FLICK_SPEED 3 screen heights a second, 100 ms
    const fire = (type, f) => pad.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 7, pointerType: 'touch', isPrimary: true,
      clientX: x0 + Math.sin(a) * dist * f, clientY: y0 - Math.cos(a) * dist * f }));
    fire('pointerdown', 0);
    let k = 0;
    const step = () => {
      k++;
      const f = Math.min(1, (k * 16) / 100);
      fire(f >= 1 ? 'pointerup' : 'pointermove', f);
      if (f >= 1) done(); else setTimeout(step, 16);
    };
    setTimeout(step, 16);
  }), [angle, power]);
}

async function toRingToss(browser) {
  const r = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-reindeer_ring_toss': 'on' });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const tv = await tvCtx.newPage();
  await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
  await H.startGame(r.host, r.roomCode);
  const turn = await H.drawGift(r.host, r.roomCode, 'Ring Gift');
  const p = await H.playersByName(r.roomCode);
  const opener = H.openerNameOf(turn, p);
  const [bidder] = NAMES.filter((n) => n !== opener);
  await H.allBid(pages, opener, { [bidder]: 1 });
  await H.revealBids(r.host);
  await H.clickStartDuel(r.host);
  await expect(r.host.getByTestId('duel-game-name')).toContainText('Ring Toss', { timeout: 10000 });
  await r.host.getByTestId('start-duel-app').click();
  await H.waitForTurnPhase(r.roomCode, 'minigame_arena');
  const close = async () => { await tvCtx.close(); await r.hostCtx.close(); for (const c of contexts) await c.close(); };
  return { ...r, pages, phoneErrors, tv, p, opener, bidder, close };
}
const stateOf = async (room) => (await H.dbGet(room, 'arena')).state;
const endStage = (room) => H.dbPut(room, 'arena/deadlines', [{ id: 'stage', at: Date.now() }]);

test('Ring Toss: practice, swipes and buttons, rings on the TV, the duel to the better duelist', async ({ browser }) => {
  test.setTimeout(150000);
  const t = await toRingToss(browser);
  try {
    const a0 = await H.dbGet(t.roomCode, 'arena');
    expect(a0.mode).toBe('party_with_duelists');
    await expect(t.tv.getByTestId('ringtoss-stage')).toBeVisible();
    await expect(t.tv.getByTestId('ringtoss-stage-label')).toContainText('Practice');
    const bystander = NAMES.find((n) => n !== t.opener && n !== t.bidder);

    // practice: a swipe makes a ring on the TV (dashed: not scored); the phone says how it went
    const ringsOnTv = () => t.tv.locator('[data-testid=ringtoss-scene] .rt-ring').count();
    await swipe(t.pages[t.opener], 0, 1.07);
    await expect.poll(async () => (await stateOf(t.roomCode)).throws?.length || 0).toBe(1);
    await expect.poll(ringsOnTv).toBeGreaterThan(0);
    await expect(t.pages[t.opener].getByTestId('ringtoss-feedback')).toContainText(/Ringer|Missed/, { timeout: 5000 });
    expect((await stateOf(t.roomCode)).scores?.[t.p[t.opener].id] || 0).toBe(0);

    // scored rings: the opener swipes, the bidder throws with the buttons, the bystander too
    await endStage(t.roomCode);
    await expect(t.tv.getByTestId('ringtoss-stage-label')).toContainText('Throw!');
    for (let i = 0; i < 3; i++) {
      await swipe(t.pages[t.opener], (i - 1) * 0.12, 0.85 + i * 0.1);
      await t.pages[t.opener].waitForTimeout(700);
    }
    const bob = t.pages[t.bidder];
    await bob.getByTestId('ringtoss-controls-toggle').click();
    await bob.getByTestId('ringtoss-aim').fill('0');
    const hold = bob.getByTestId('ringtoss-hold');
    await hold.dispatchEvent('pointerdown');                       // a tiny charge: it falls short
    await hold.dispatchEvent('pointerup');
    await swipe(t.pages[bystander], 0, 1.07);
    await expect.poll(async () => (await stateOf(t.roomCode)).throws.filter((x) => x.stage === 'throwing').length).toBe(5);

    // the TV's scores are the room's; every phone's own score too
    const s = await stateOf(t.roomCode);
    for (const n of NAMES) {
      await expect(t.tv.getByTestId(`ringtoss-score-${t.p[n].id}`)).toContainText(String(s.scores[t.p[n].id] || 0));
      await expect(t.pages[n].getByTestId('ringtoss-my-score')).toHaveText(String(s.scores[t.p[n].id] || 0));
    }
    // a page that reloads keeps its rings (nothing doubled)
    await t.pages[t.opener].reload();
    await t.pages[t.opener].getByRole('button', { name: /Rejoin/i }).click().catch(() => {});
    await expect(t.pages[t.opener].getByTestId('ringtoss-card')).toBeVisible();
    expect((await stateOf(t.roomCode)).throws.filter((x) => x.pid === t.p[t.opener].id && x.stage === 'throwing')).toHaveLength(3);

    // time's up (and any tiebreak): the duel goes to the duelist the room scored higher
    for (let i = 0; i < 8; i++) {
      const g = await H.dbGet(t.roomCode, '');
      if (g.lastDuel || g.currentTurn?.phase === 'duel') break;
      if (g.arena?.lifecycle === 'active') await endStage(t.roomCode);
      await t.host.waitForTimeout(800);
    }
    const g = await H.waitForGame(t.roomCode, (x) => (x.lastDuel || x.currentTurn?.duelGameWinner || x.currentTurn?.duelGameOutcome ? x : false), { label: 'ring toss decided', timeout: 30000 });
    const [o, b] = [t.p[t.opener].id, t.p[t.bidder].id];
    const bobRing = s.throws.find((x) => x.pid === b && x.stage === 'throwing');
    expect(bobRing).toMatchObject({ pegId: null, points: 0 });      // the short one scored nothing
    const winner = g.lastDuel?.winnerId ?? g.currentTurn?.duelGameWinner;
    if ((s.scores[o] || 0) > 0) expect(winner).toBe(o);              // more points: the opener wins the duel
    else expect([o, b]).toContain(winner);                           // both on 0: the closest ring decided it
    expect([...t.errors, ...t.phoneErrors]).toEqual([]);
  } finally {
    await t.close();
  }
});
