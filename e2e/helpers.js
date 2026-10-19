// @ts-check
// Shared helpers for game-flow tests.
//
// Philosophy: drive the game the way a real host + players would (clicking every
// button the game legitimately requires), then assert INVARIANTS that must hold
// no matter the rules — chips never negative, gifts never orphaned or duplicated,
// player.giftId <-> gift.ownerId always consistent, the turn always resolves, and
// zero page errors. We do NOT bake in expected chip totals for ambiguous rules
// (e.g. Santa Tax split); those are reported for review instead of asserted.
//
// State is read straight from the RTDB emulator (Bearer owner = admin bypass).

const BASE = process.env.BASE_URL || 'http://localhost:8080';
const NS = 'jingle-brawl-e8308-default-rtdb';
const DB_BASE = 'http://127.0.0.1:9000';
const ADMIN = { Authorization: 'Bearer owner', 'Content-Type': 'application/json' };

async function dbGet(roomCode, path = '') {
  const url = `${DB_BASE}/games/${roomCode}${path ? '/' + path : ''}.json?ns=${NS}`;
  const res = await fetch(url, { headers: ADMIN });
  return res.json();
}

/** Admin write (bypasses rules) — for injecting state a test can't reach by clicking. */
async function dbPut(roomCode, path, value) {
  const url = `${DB_BASE}/games/${roomCode}/${path}.json?ns=${NS}`;
  const res = await fetch(url, { method: 'PUT', headers: ADMIN, body: JSON.stringify(value) });
  if (!res.ok) throw new Error(`dbPut ${path} failed: ${res.status}`);
  return res.json();
}

/** Poll until fn(game) returns truthy; returns that value. */
async function waitForGame(roomCode, fn, { timeout = 15000, label = 'condition' } = {}) {
  const t0 = Date.now();
  let last;
  while (Date.now() - t0 < timeout) {
    last = await dbGet(roomCode);
    const result = fn(last);
    if (result) return result;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(
    `Timed out waiting for ${label}.\n  phase: ${last?.phase}\n  currentTurn: ${JSON.stringify(last?.currentTurn)}`
  );
}

/** phase === null waits for the turn to be over (currentTurn cleared). */
async function waitForTurnPhase(roomCode, phase, timeout = 15000) {
  return waitForGame(
    roomCode,
    (g) =>
      phase === null
        ? g.currentTurn == null ? { over: true } : false
        : g.currentTurn?.phase === phase ? g.currentTurn : false,
    { timeout, label: `turn phase ${phase === null ? 'null (turn resolved)' : phase}` }
  );
}

// --- Invariants -----------------------------------------------------------

/**
 * Returns a list of invariant-violation strings (empty = healthy).
 * These are properties that must hold in every stable game state.
 */
function invariantErrors(game) {
  const errors = [];
  const players = game.players || {};
  const gifts = game.gifts || {};
  const ids = new Set(Object.keys(players));

  for (const [id, p] of Object.entries(players)) {
    if (typeof p.chips !== 'number' || Number.isNaN(p.chips)) errors.push(`${p.name}: chips not a number (${p.chips})`);
    else if (p.chips < 0) errors.push(`${p.name}: negative chips (${p.chips})`);
  }

  // Gift owner must be a real player or the misfit pile
  for (const g of Object.values(gifts)) {
    if (g.ownerId != null && g.ownerId !== 'misfit' && !ids.has(g.ownerId)) {
      errors.push(`gift "${g.name}": dangling ownerId ${g.ownerId}`);
    }
  }

  // No player holds more than one gift
  const held = {};
  for (const [gid, g] of Object.entries(gifts)) {
    if (g.ownerId && g.ownerId !== 'misfit') (held[g.ownerId] ||= []).push(g.name);
  }
  for (const [id, names] of Object.entries(held)) {
    if (names.length > 1) errors.push(`${players[id]?.name || id}: holds ${names.length} gifts (${names.join(', ')})`);
  }

  // player.giftId and gift.ownerId must agree both ways
  for (const [id, p] of Object.entries(players)) {
    if (p.giftId) {
      const g = gifts[p.giftId];
      if (!g) errors.push(`${p.name}: giftId ${p.giftId} points to no gift`);
      else if (g.ownerId !== id) errors.push(`${p.name}: giftId ${p.giftId} but that gift.ownerId is ${g.ownerId}`);
    }
  }
  for (const [gid, g] of Object.entries(gifts)) {
    if (g.ownerId && g.ownerId !== 'misfit') {
      const p = players[g.ownerId];
      if (p && p.giftId !== gid) errors.push(`gift "${g.name}": owned by ${p.name} but their giftId is ${p.giftId}`);
    }
  }

  return errors;
}

async function assertInvariants(roomCode, label = '') {
  const game = await dbGet(roomCode);
  const errors = invariantErrors(game);
  if (errors.length) {
    throw new Error(`Invariant violation${label ? ` [${label}]` : ''}:\n  - ${errors.join('\n  - ')}`);
  }
  return game;
}

// --- Page-error capture ---------------------------------------------------

/** Attach a pageerror listener; pushes into and returns a live array. */
function trackErrors(page, who) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(`${who}: ${e.message}`));
  return errors;
}

// --- Setup ----------------------------------------------------------------

/**
 * Create a game from the host setup screen. `settings` maps setup-screen <select>
 * ids (without the "setting-" prefix) to option values, e.g.
 * { 'starting-chips': '15', 'tiebreaker-method': 'minigame' }.
 */
async function createGame(browser, settings = {}) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const host = await ctx.newPage();
  const errors = trackErrors(host, 'HOST');
  await host.goto(`${BASE}/host.html`);
  for (const [id, value] of Object.entries(settings)) {
    await host.locator(`#setting-${id}`).selectOption(value);
  }
  await host.getByRole('button', { name: /Create New Game/i }).click();
  await host.waitForSelector('#game-screen', { state: 'visible' });
  const roomCode = (await host.locator('#room-code-display').textContent()).trim();
  return { host, hostCtx: ctx, roomCode, errors };
}

/** Admin-bypass multi-path update of a room (keys are paths under games/{room}). */
async function dbPatch(roomCode, updates) {
  const url = `${DB_BASE}/games/${roomCode}.json?ns=${NS}`;
  const res = await fetch(url, { method: 'PATCH', headers: ADMIN, body: JSON.stringify(updates) });
  if (!res.ok) throw new Error(`dbPatch failed: ${res.status}`);
  return res.json();
}

async function joinPlayers(browser, roomCode, names) {
  const pages = {};
  const contexts = [];
  const errors = []; // shared live array of all player page errors
  for (let i = 0; i < names.length; i++) {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
    contexts.push(ctx);
    const page = await ctx.newPage();
    page.on('pageerror', (e) => errors.push(`PLAYER ${names[i]}: ${e.message}`));
    await page.goto(`${BASE}/player.html?room=${roomCode}&testId=${i}&name=${names[i]}`);
    await page.getByRole('button', { name: /Join Game/i }).click();
    await page.waitForSelector('#game-screen', { state: 'visible' });
    pages[names[i]] = page;
  }
  return { pages, contexts, errors };
}

async function playersByName(roomCode) {
  const players = (await dbGet(roomCode, 'players')) || {};
  const map = {};
  for (const [id, p] of Object.entries(players)) map[p.name] = { id, ...p };
  return map;
}

async function startGame(host, roomCode) {
  await host.getByRole('button', { name: /Start Game/i }).click();
  await waitForGame(roomCode, (g) => g.phase === 'main', { label: 'main phase' });
}

async function drawGift(host, roomCode, giftName) {
  host.once('dialog', (d) => d.accept(giftName));
  await host.getByRole('button', { name: /Draw.*Santa|Santa.*Sack/i }).click();
  return waitForTurnPhase(roomCode, 'bidding');
}

function openerNameOf(turn, playersMap) {
  return Object.values(playersMap).find((p) => p.id === turn.openerId).name;
}

async function submitBid(page, amount) {
  const panel = page.locator('#bid-panel');
  await panel.waitFor({ state: 'visible', timeout: 10000 });
  if (amount === 0) await panel.getByRole('button', { name: 'Pass', exact: true }).click();
  else await panel.getByRole('button', { name: String(amount), exact: true }).click();
  await panel.getByRole('button', { name: /Seal it|Lock In Bid/i }).click();
}

/** Everyone except opener bids; bids = { Bob: 2, Charlie: 0 }. Missing => 0 (pass). */
async function allBid(pages, openerName, bids) {
  for (const [name, page] of Object.entries(pages)) {
    if (name === openerName || name.startsWith('_')) continue;
    await submitBid(page, bids[name] ?? 0);
  }
}

async function revealBids(host) {
  await host.getByRole('button', { name: /Reveal All Bids/i }).click();
}

// --- Host / player choice actions ----------------------------------------

const clickStartDuel = (host) => host.getByRole('button', { name: /Start Duel$|Start Duel!/i }).first().click();
const clickStartYieldDuel = (host) => host.getByRole('button', { name: /Start Yield Duel/i }).click();
const clickStartTiebreakerDuel = (host) => host.getByRole('button', { name: /Start Tiebreaker Duel/i }).click();

/** Normal duel winner — button lives in the duel modal footer. */
async function declareWinner(host, name) {
  const btn = host.locator('#duel-modal-content').getByRole('button', { name: new RegExp(`${name} wins`, 'i') });
  await btn.waitFor({ state: 'visible', timeout: 10000 });
  await btn.click();
}

/** Tiebreak winner — buttons live in the turn panel (no modal for tiebreaks). */
async function declareTiebreakWinner(host, name) {
  const btn = host.locator('#action-buttons').getByRole('button', { name: new RegExp(`${name} Wins`, 'i') });
  await btn.waitFor({ state: 'visible', timeout: 10000 });
  await btn.click();
}

const openerKeeps = (page) => page.getByRole('button', { name: /Keep My Gift/i }).click();
const openerFights = (page) => page.getByRole('button', { name: /Fight/i }).first().click();
const openerYields = (page) => page.getByRole('button', { name: /Yield!/i }).first().click();
const skipReprisal = (page) => page.getByRole('button', { name: /Skip Reprisal/i }).click();

/**
 * After a winner is declared, the turn resolves to either null or 'reprisal'.
 * If reprisal, the loser skips it (pass `pagesById` to click their phone).
 */
async function resolveDuel(host, roomCode, pagesById) {
  const state = await waitForGame(
    roomCode,
    (g) => (g.currentTurn == null ? { over: true } : g.currentTurn.phase === 'reprisal' ? g.currentTurn : false),
    { label: 'duel resolution (turn null or reprisal)', timeout: 20000 }
  );
  if (state.over) return;
  // Reprisal offered — skip it to end the turn cleanly
  const page = pagesById?.[state.reprisalPlayerId];
  if (page) {
    await skipReprisal(page);
    await waitForTurnPhase(roomCode, null);
  }
}

module.exports = {
  BASE, dbGet, dbPut, dbPatch, waitForGame, waitForTurnPhase,
  invariantErrors, assertInvariants,
  createGame, joinPlayers, playersByName, startGame, drawGift, openerNameOf,
  submitBid, allBid, revealBids,
  clickStartDuel, clickStartYieldDuel, clickStartTiebreakerDuel,
  declareWinner, declareTiebreakWinner,
  openerKeeps, openerFights, openerYields, skipReprisal, resolveDuel,
};
