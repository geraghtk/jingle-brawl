// @ts-check
// The phone's side of the polish list: the revealed bids on the phone (and nothing of them before
// the reveal), a reload that lands straight back in the game, the app's own Leave sheet, and long
// names that fit (no break inside a word, no overflow, no ellipsis on the name the moment is about).
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const LONG = 'Bartholomew-Alexander';
const LONGEST = 'Bartholomew-Alexander II';   // 24 letters: the most a name may be (core/actions.ts NAME_MAX)

/** Join with a name that needs encoding and a phone of a given size. */
async function joinAs(browser, roomCode, names, sizes = []) {
  const pages = {}, contexts = [];
  for (let i = 0; i < names.length; i++) {
    const ctx = await browser.newContext({ viewport: sizes[i] || { width: 390, height: 844 } });
    contexts.push(ctx);
    const page = await ctx.newPage();
    await page.goto(`${H.PLAYER_BASE}/player.html?room=${roomCode}&testId=${i}&name=${encodeURIComponent(names[i])}`);
    await page.getByRole('button', { name: /Join Game/i }).click();
    await page.waitForSelector('#game-screen', { state: 'visible' });
    pages[names[i]] = page;
  }
  return { pages, contexts };
}

/** Nothing on the page runs off the screen, and no fitted name is wider than its place or broken inside a word. */
async function namesFit(page, label) {
  const bad = await page.evaluate(() => {
    const out = [];
    if (document.documentElement.scrollWidth > window.innerWidth + 1) out.push('the page scrolls sideways');
    for (const el of document.querySelectorAll('.fit-name')) {
      const box = el.getBoundingClientRect();
      if (!box.width) continue;
      const range = document.createRange(); range.selectNodeContents(el);
      const rects = [...range.getClientRects()].filter((r) => r.width > 0);
      const right = Math.max(...rects.map((r) => r.right)), left = Math.min(...rects.map((r) => r.left));
      const room = el.parentElement.getBoundingClientRect();
      if (right > room.right + 2 || left < room.left - 2) out.push(`"${el.textContent}" is wider than its place`);
      if (right > window.innerWidth + 1) out.push(`"${el.textContent}" runs off the screen`);
      const cs = getComputedStyle(el);
      if (cs.textOverflow === 'ellipsis' && cs.overflow !== 'visible') out.push(`"${el.textContent}" may be cut with an ellipsis`);
      if (el.classList.contains('fit-break')) out.push(`"${el.textContent}" may break inside a word`);
    }
    return out;
  });
  expect(bad, label).toEqual([]);
}

test('after the reveal a phone lists every bid, the top bids and its own row marked, a tie said plainly; before it, nothing', async ({ browser }) => {
  test.setTimeout(120000);
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const names = ['Alice', 'Bob', 'Cara', 'Dan'];
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, names);
  try {
    await H.startGame(host, roomCode);
    const turn = await H.drawGift(host, roomCode, 'Socks');
    const ids = await H.playersByName(roomCode);
    const opener = H.openerNameOf(turn, ids);
    const [a, b, c] = names.filter((n) => n !== opener);
    await H.allBid(pages, opener, { [a]: 3, [b]: 3, [c]: 0 });
    await H.waitForGame(roomCode, (g) => Object.keys(g.currentTurn?.bids || {}).length === 3, { label: 'three bids in' });

    // sealed: no list on any phone, and the amounts are nowhere on the page
    for (const n of names) await expect(pages[n].getByTestId('revealed-bids')).toHaveCount(0);

    await H.revealBids(host);
    await H.waitForTurnPhase(roomCode, 'reveal');
    for (const n of names) {
      const page = pages[n];
      const card = page.getByTestId('revealed-bids');
      await expect(card).toBeVisible();
      await expect(card.locator('.rb-row')).toHaveCount(3);
      // highest first: the two tied at 3, then the pass
      await expect(card.locator('.rb-row').nth(2)).toHaveAttribute('data-testid', `bid-row-${ids[c].id}`);
      await expect(page.getByTestId(`bid-amount-${ids[a].id}`)).toContainText('3');
      await expect(page.getByTestId(`bid-amount-${ids[b].id}`)).toContainText('3');
      await expect(page.getByTestId(`bid-amount-${ids[c].id}`)).toHaveText('pass');
      await expect(page.getByTestId(`bid-row-${ids[a].id}`)).toHaveAttribute('data-top', 'true');
      await expect(page.getByTestId(`bid-row-${ids[b].id}`)).toHaveAttribute('data-top', 'true');
      await expect(page.getByTestId(`bid-row-${ids[c].id}`)).not.toHaveAttribute('data-top', 'true');
      await expect(page.getByTestId('bids-summary')).toContainText('A tie at 3 chips');
      await expect(card.locator('[data-mine]')).toHaveCount(n === opener ? 0 : 1);
      if (n !== opener) await expect(page.getByTestId(`bid-row-${ids[n].id}`)).toHaveAttribute('data-mine', 'true');
    }
    await expect(pages[a].getByTestId('bids-summary')).toContainText(`${b} and you`);
    await expect(pages[c].getByTestId('bids-next')).toContainText(`${opener} is choosing`);
    // the opener still has their choice, above the list
    await expect(pages[opener].getByTestId('fight')).toBeVisible();
    await H.assertInvariants(roomCode, 'after the reveal');
  } finally {
    await Promise.all([hostCtx.close(), ...contexts.map((x) => x.close())]);
  }
});

test('a phone that reloads mid-turn lands straight back in the game: same seat, chips and gift, no tap', async ({ browser }) => {
  test.setTimeout(120000);
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, ['Alice', 'Bob', 'Cara']);
  try {
    await H.startGame(host, roomCode);
    const turn = await H.drawGift(host, roomCode, 'Socks');
    const ids = await H.playersByName(roomCode);
    const opener = H.openerNameOf(turn, ids);
    await H.dbPut(roomCode, `players/${ids[opener].id}/chips`, 7);      // something to bring along
    const phone = pages[opener];
    await expect(phone.locator('#chip-display')).toHaveText('7');

    await phone.reload();                                               // mid-turn: the others are bidding on their gift
    await expect(phone.locator('#game-screen')).toBeVisible();
    await expect(phone.getByTestId('rejoin')).toHaveCount(0);
    await expect(phone.getByTestId('join')).toHaveCount(0);
    await expect(phone.locator('#display-name')).toHaveText(opener);
    await expect(phone.locator('#chip-display')).toHaveText('7');
    await expect(phone.locator('#your-gift-name')).toHaveText('Socks');
    await expect(phone.locator('#status-card')).toContainText('You opened');
    const after = await H.playersByName(roomCode);
    expect(Object.keys(after).sort()).toEqual(['Alice', 'Bob', 'Cara']);  // nobody joined twice
    expect(after[opener].id).toBe(ids[opener].id);

    // the join link again (the QR code scanned a second time): still straight in
    await phone.goto(`${H.PLAYER_BASE}/player.html?room=${roomCode}&testId=${['Alice', 'Bob', 'Cara'].indexOf(opener)}`);
    await expect(phone.locator('#game-screen')).toBeVisible();
    await expect(phone.locator('#chip-display')).toHaveText('7');
    await H.assertInvariants(roomCode, 'after a reload');
  } finally {
    await Promise.all([hostCtx.close(), ...contexts.map((x) => x.close())]);
  }
});

test('a link to a different room is the one time a phone with a seat is asked which game', async ({ browser }) => {
  test.setTimeout(120000);
  const one = await H.createGame(browser);
  const two = await H.createGame(browser);
  const { pages, contexts } = await H.joinPlayers(browser, one.roomCode, ['Alice']);
  try {
    const phone = pages.Alice;
    await phone.goto(`${H.PLAYER_BASE}/player.html?room=${two.roomCode}&testId=0&name=Alice`);
    await expect(phone.getByTestId('rejoin-choice')).toContainText(one.roomCode);
    await expect(phone.getByTestId('rejoin-choice')).toContainText(two.roomCode);
    await phone.getByTestId('rejoin').click();                           // back to the seat they had
    await expect(phone.locator('#game-screen')).toBeVisible();
    expect(Object.keys(await H.playersByName(one.roomCode))).toEqual(['Alice']);

    await phone.goto(`${H.PLAYER_BASE}/player.html?room=${two.roomCode}&testId=0&name=Alice`);
    await phone.getByTestId('join-other').click();                       // or the other party: the join form, room filled in
    await expect(phone.locator('#room-code-input')).toHaveValue(two.roomCode);
    await phone.getByTestId('join').click();
    await expect(phone.locator('#game-screen')).toBeVisible();
    expect(Object.keys(await H.playersByName(two.roomCode))).toEqual(['Alice']);
  } finally {
    await Promise.all([one.hostCtx.close(), two.hostCtx.close(), ...contexts.map((x) => x.close())]);
  }
});

test("Leave asks in the app's own sheet (no browser dialog): Stay keeps the seat, Leave gives it up", async ({ browser }) => {
  const { hostCtx, roomCode } = await H.createGame(browser);
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, ['Alice']);
  try {
    const phone = pages.Alice;
    const dialogs = [];
    phone.on('dialog', (d) => { dialogs.push(d.message()); d.dismiss().catch(() => {}); });
    await phone.getByRole('button', { name: 'Leave game' }).click();
    const sheet = phone.getByTestId('leave-sheet');
    await expect(sheet).toBeVisible();
    await expect(sheet).toHaveAttribute('role', 'alertdialog');
    await expect(phone.getByTestId('leave-sheet-cancel')).toBeFocused();   // the safe answer has the focus
    await phone.getByTestId('leave-sheet-cancel').click();
    await expect(sheet).toHaveCount(0);
    await expect(phone.locator('#game-screen')).toBeVisible();

    await phone.getByRole('button', { name: 'Leave game' }).click();
    await phone.keyboard.press('Escape');                                  // Escape is "stay" too
    await expect(sheet).toHaveCount(0);

    await phone.getByRole('button', { name: 'Leave game' }).click();
    await phone.getByTestId('leave-sheet-confirm').click();
    await expect(phone.getByTestId('join')).toBeVisible();                 // the join form: the seat is forgotten
    await phone.reload();
    await expect(phone.getByTestId('join')).toBeVisible();                 // and a reload does not bring it back
    expect(dialogs).toEqual([]);
  } finally {
    await Promise.all([hostCtx.close(), ...contexts.map((x) => x.close())]);
  }
});

test('long names fit: duel corners, the verdict, the vote and the bids, on the console and on small phones', async ({ browser }) => {
  test.setTimeout(150000);
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const names = [LONG, 'Christopher', LONGEST, 'Al'];
  const small = { width: 320, height: 568 };
  const { pages, contexts } = await joinAs(browser, roomCode, names, [undefined, small, undefined, small]);
  try {
    const typed = await H.playersByName(roomCode);
    expect(typed[LONGEST].name.length).toBe(24);                          // the whole name was kept
    await H.startGame(host, roomCode);
    const turn = await H.drawGift(host, roomCode, 'A Suspiciously Heavy Fruitcake');
    const ids = await H.playersByName(roomCode);
    const opener = H.openerNameOf(turn, ids);
    const bidders = names.filter((n) => n !== opener);
    // one clear top bid: the longest name there is, unless they opened
    const top = bidders.includes(LONGEST) ? LONGEST : LONG;
    await H.allBid(pages, opener, Object.fromEntries(bidders.map((n) => [n, n === top ? 2 : 0])));
    await H.revealBids(host);
    await H.waitForTurnPhase(roomCode, 'reveal');
    for (const n of names) {
      await expect(pages[n].getByTestId('revealed-bids')).toBeVisible();
      await namesFit(pages[n], `${n}'s phone at the reveal`);
    }

    await H.clickStartDuel(host);
    await H.waitForTurnPhase(roomCode, 'duel');
    await expect(host.locator('#duel-modal-content .corner-name .fit-name')).toHaveCount(2);
    await namesFit(host, 'the console in the duel');
    for (const n of names) {
      await expect(pages[n].locator('.duel-you-row, .callit-corners')).toBeVisible();
      await namesFit(pages[n], `${n}'s phone in the duel`);
    }
    // the far avatar stays on the row with the names (it used to drop to a line of its own)
    const duelist = pages[top];
    const tops = await duelist.locator('.duel-you-row .f6-avatar').evaluateAll((els) => els.map((e) => Math.round(e.getBoundingClientRect().top + e.getBoundingClientRect().height / 2)));
    expect(Math.abs(tops[0] - tops[1])).toBeLessThan(4);

    if (await host.getByTestId('close-voting').count()) await host.getByTestId('close-voting').click();
    await host.getByTestId(`modal-winner-${ids[top].id}`).click();
    const name = host.locator('#duel-verdict .verdict-name');
    await expect(name).toHaveText(top);
    await namesFit(host, "the console's verdict");
    // the name the moment is about is whole: every line of it ends at a space or a hyphen
    const lines = await name.locator('.fit-name').evaluate((el) => {
      const text = el.firstChild, out = []; let line = '', lastTop = null;
      for (let i = 0; i < text.length; i++) {
        const r = document.createRange(); r.setStart(text, i); r.setEnd(text, i + 1);
        const top = Math.round(r.getBoundingClientRect().top);
        if (lastTop != null && top > lastTop + 2) { out.push(line); line = ''; }
        line += text.data[i]; lastTop = top;
      }
      return [...out, line];
    });
    for (const line of lines.slice(0, -1)) expect(line, `"${line}" ends inside a word`).toMatch(/[\s-]$/);
  } finally {
    await Promise.all([hostCtx.close(), ...contexts.map((x) => x.close())]);
  }
});
