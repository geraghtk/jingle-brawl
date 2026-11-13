// Which duel the duellists play: the defender picks from the host's list, or the wheel spins
// (on the TV and every phone, landing on the same duel); an in-app game settles the duel itself.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie'];

async function setup(browser, settings) {
  const { host, hostCtx, roomCode, errors } = await H.createGame(browser, { 'vote-window': '0', ...settings });
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, roomCode, NAMES);
  await H.startGame(host, roomCode);
  const turn = await H.drawGift(host, roomCode, 'Duel Gift');
  const p = await H.playersByName(roomCode);
  const opener = H.openerNameOf(turn, p);
  const [bidder, spectator] = NAMES.filter((n) => n !== opener);
  await H.allBid(pages, opener, { [bidder]: 1 });
  await H.revealBids(host);
  await H.clickStartDuel(host);
  await H.waitForTurnPhase(roomCode, 'duel');
  return {
    host, roomCode, pages, p, opener, bidder, spectator,
    noErrors: () => expect([...errors, ...phoneErrors]).toEqual([]),
    cleanup: async () => { await hostCtx.close(); for (const c of contexts) await c.close(); },
  };
}

test('the defender picks the duel from the host\'s list; every screen shows it; then the winner can be called', async ({ browser }) => {
  const s = await setup(browser, { 'duel-pick': 'defender', 'duel-games': 'Rock, Paper, Scissors\nThumb War\nStaring Contest' });
  try {
    const { host, pages, p, opener, bidder, spectator } = s;
    // the opener defends: their phone lists the duels
    await expect(pages[opener].getByTestId('duel-game-1')).toHaveText('Thumb War');
    await expect(pages[spectator].locator('#status-card')).toContainText(`${opener} is picking the duel`);
    await expect(pages[spectator].getByTestId(`vote-${p[bidder].id}`)).toHaveCount(0); // no voting yet
    await expect(host.getByTestId('duel-game-stage')).toContainText(`${opener} is picking the duel`);
    await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toBeDisabled();

    await pages[opener].getByTestId('duel-game-1').click();
    await expect(host.getByTestId('duel-game-name')).toHaveText('Thumb War');
    await expect(host.getByTestId('duel-game-stage')).toContainText(`${opener} picked it`);
    for (const n of [bidder, spectator, opener]) await expect(pages[n].getByTestId('phone-duel-game')).toContainText('Thumb War');
    await pages[spectator].getByTestId(`vote-${p[bidder].id}`).click(); // voting is open now
    await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toBeEnabled();
    await H.declareWinner(host, opener);
    await H.waitForGame(s.roomCode, (g) => (g.lastDuel?.winnerId === p[opener].id ? g : false), { label: 'duel resolved' });
    s.noErrors();
  } finally {
    await s.cleanup();
  }
});

test('the wheel spins on the TV and on every phone and lands on the same duel', async ({ browser }) => {
  const s = await setup(browser, { 'duel-pick': 'wheel', 'duel-games': 'Thumb War\nStaring Contest\nArm Wrestle\nOdds & Evens' });
  try {
    const { host, pages, p, opener, bidder, spectator } = s;
    await expect(host.getByTestId('duel-wheel')).toBeVisible();
    for (const n of NAMES) await expect(pages[n].getByTestId('duel-wheel')).toBeVisible();
    await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toBeDisabled(); // until it stops
    const turn = await H.waitForGame(s.roomCode, (g) => g.currentTurn?.duelGame && g.currentTurn, { label: 'duel game' });
    expect(turn.duelGame.via).toBe('wheel');
    expect(turn.wheel.options.map((o) => o.name)).toEqual(['Thumb War', 'Staring Contest', 'Arm Wrestle', 'Odds & Evens']);
    await expect(host.getByTestId('duel-game-name')).toHaveText(turn.duelGame.name, { timeout: 8000 });
    for (const n of [opener, bidder, spectator]) await expect(pages[n].getByTestId('phone-duel-game')).toContainText(turn.duelGame.name);
    await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toBeEnabled();
    s.noErrors();
  } finally {
    await s.cleanup();
  }
});

test('the Under-16 Exception: a player who joined as under 16 picks the duel, even where the wheel would', async ({ browser }) => {
  const { host, hostCtx, roomCode } = await H.createGame(browser, { 'vote-window': '0', 'duel-pick': 'defender', 'duel-games': 'Thumb War\nArm Wrestle' });
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, ['Alice', 'Bob']);
  const kidCtx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  try {
    // the kid ticks "I'm under 16" and gives an age; adults are never asked
    const kid = await kidCtx.newPage();
    await kid.goto(`${H.PLAYER_BASE}/player.html?room=${roomCode}&testId=9&name=Kid`);
    await kid.getByTestId('under-16').check();
    await expect(kid.getByTestId('join')).toBeDisabled(); // pick an age first
    await kid.getByTestId('age').selectOption('10');
    await kid.getByTestId('join').click();
    await kid.waitForSelector('#game-screen', { state: 'visible' });
    const p = await H.playersByName(roomCode);
    expect(p.Kid.age).toBe(10);
    expect(p.Alice.age).toBeUndefined();

    await H.startGame(host, roomCode);
    const turn = await H.drawGift(host, roomCode, 'Kid Gift');
    // a yield duel would normally spin the wheel — the kid's age overrides that
    await H.dbPatch(roomCode, { currentTurn: { phase: 'duel', challengerId: p.Kid.id, defenderId: p.Alice.id, giftId: turn.giftId, pot: 0, isYieldDuel: true } });
    await expect(kid.locator('#status-card')).toContainText("You're under 16, so you pick");
    await expect(pages.Alice.locator('#status-card')).toContainText('Kid is picking the duel');
    await expect(host.getByTestId('duel-game-stage')).toContainText('Kid is picking the duel · under-16 rule');
    await expect(host.getByTestId('duel-wheel')).toHaveCount(0);

    await kid.getByTestId('duel-game-1').click();
    await expect(host.getByTestId('duel-game-name')).toHaveText('Arm Wrestle');
    await expect(host.getByTestId('duel-game-stage')).toContainText('Kid picked it');
    await expect(pages.Bob.getByTestId('phone-duel-game')).toContainText('Arm Wrestle');
  } finally {
    await hostCtx.close();
    await kidCtx.close();
    for (const c of contexts) await c.close();
  }
});

test('an in-app duel game settles it: everyone plays Whack-an-Elf, the better-placed duellist wins the duel', async ({ browser }) => {
  test.setTimeout(90000);
  const s = await setup(browser, { 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-whack_elf': 'on' });
  try {
    const { host, roomCode, pages, p, opener, bidder } = s;
    await expect(host.getByTestId('duel-game-name')).toContainText('Whack-an-Elf', { timeout: 8000 });
    await expect(host.getByTestId(`modal-winner-${p[opener].id}`)).toHaveCount(0, { timeout: 10000 }); // the game decides (after the 4.8 s spin)
    await host.getByTestId('start-duel-app').click();
    await H.waitForTurnPhase(roomCode, 'minigame_party');
    for (const n of NAMES) await expect(pages[n].locator('#party-card')).toContainText('Whack-an-Elf');
    const skip = () => host.getByTestId('skip-party-phase').click();
    await skip(); // intro → playing
    await H.waitForGame(roomCode, (g) => (g.minigame?.phase === 'playing' ? g : false), { label: 'playing' });
    // the defender (opener) out-whacks the challenger
    await H.dbPatch(roomCode, { [`minigame/scores/${p[opener].id}`]: { score: 9, final: true }, [`minigame/scores/${p[bidder].id}`]: { score: 3, final: true } });
    await skip(); // → results
    const mg = await H.waitForGame(roomCode, (g) => (g.minigame?.phase === 'results' ? g.minigame : false), { label: 'results' });
    expect(mg.results.winners).toEqual([p[opener].id]);
    await skip(); // → back to the duel, which the host settles with the game's winner
    const g = await H.waitForGame(roomCode, (x) => (x.lastDuel?.winnerId ? x : false), { label: 'duel resolved', timeout: 10000 });
    expect(g.lastDuel).toMatchObject({ winnerId: p[opener].id, loserId: p[bidder].id });
    await expect(host.locator('#duel-verdict .verdict-name')).toHaveText(opener);
    s.noErrors();
  } finally {
    await s.cleanup();
  }
});

/** Scribble on a phone's Draw Off canvas, then submit it. */
async function drawAndSubmit(page, strokes) {
  const box = await page.locator('#drawing-canvas').boundingBox();
  for (let i = 0; i < strokes; i++) {
    await page.mouse.move(box.x + 20 + i * 9, box.y + 20);
    await page.mouse.down();
    await page.mouse.move(box.x + 40 + i * 11, box.y + box.height - 30, { steps: 4 });
    await page.mouse.up();
  }
  await page.getByTestId('drawoff-submit').click();
}

test('AI Draw Off settles the duel: the two duellists draw, the critic, with the host key, picks the higher score', async ({ browser }) => {
  test.setTimeout(90000);
  // the server runs with JB_FAKE_CRITIC=1 in tests: scores come from the drawings, no network, no key spent
  const s = await setup(browser, { 'duel-pick': 'wheel', 'duel-games': '', 'duel-app-draw_off': 'on', 'ai-critic-key': 'sk-e2e-secret-key' });
  try {
    const { host, roomCode, pages, p, opener, bidder, spectator } = s;
    await expect(host.getByTestId('duel-game-name')).toContainText('AI Draw Off', { timeout: 8000 });
    await expect(host.getByTestId('critic-status')).toContainText('Ready');
    await expect(host.locator('body')).not.toContainText('sk-e2e-secret-key');
    await host.getByTestId('start-duel-app').click();
    await H.waitForTurnPhase(roomCode, 'minigame_draw_off');
    await H.waitForGame(roomCode, (g) => (g.minigame?.phase === 'drawing' ? g : false), { label: 'drawing', timeout: 15000 });
    await expect(pages[spectator].getByTestId('drawoff-card')).toContainText('Watch the TV');
    await drawAndSubmit(pages[opener], 6);
    await drawAndSubmit(pages[bidder], 2);
    const mg = await H.waitForGame(roomCode, (g) => (g.minigame?.phase === 'results' ? g.minigame : false), { label: 'judged', timeout: 15000 });
    const [o, b] = [mg.scores[p[opener].id].rating, mg.scores[p[bidder].id].rating];
    const g = await H.waitForGame(roomCode, (x) => (x.lastDuel?.winnerId || x.currentTurn?.duelGameOutcome ? x : false), { label: 'back to the duel', timeout: 30000 });
    if (o === b) {
      // a tie: nobody wins by default, the host picks
      await expect(host.getByTestId('duel-actions-hint')).toContainText('you pick the winner');
      await host.getByTestId(`modal-winner-${p[opener].id}`).click();
      await H.waitForGame(roomCode, (x) => (x.lastDuel?.winnerId ? x : false), { label: 'host picked' });
    } else {
      expect(g.lastDuel.winnerId).toBe(o > b ? p[opener].id : p[bidder].id);
    }
    s.noErrors();
  } finally {
    await s.cleanup();
  }
});
