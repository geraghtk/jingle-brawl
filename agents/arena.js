// Runs ONE real game: the host page + one phone page per agent, all in a real browser
// against the local emulators. The loop reads the room from the emulator, works out
// whose move it is, asks that agent's persona, and clicks the matching button on that
// agent's own page — exactly what a person would do. The host is driven by a
// "referee" that clicks Draw/Reveal/Start and declares duel winners (fair coin: real
// duels are rock-paper-scissors, thumb wars …).
//
// Playability checks while it plays: no progress for `stallMs` = STUCK (snapshot +
// screenshots saved), page errors, unexpected alerts, missing buttons, and the
// e2e invariants (chips ≥ 0, one gift per player, owner links consistent).

const fs = require('fs');
const path = require('path');
const H = require('../e2e/helpers');
const { decide, rng, clamp } = require('./personas');

const GIFT_CATALOG = [
  'Lava Lamp', 'Fancy Socks', 'Scented Candle', 'Rubber Duck', 'Ugly Sweater', 'Bluetooth Speaker',
  'Chia Pet', 'Hot Sauce Kit', 'Fondue Set', 'Snow Globe', 'Puzzle Box', 'Garden Gnome',
  'Coffee Sampler', 'Desk Cactus', 'Board Game', 'Bath Bombs', 'Novelty Mug', 'Beanie',
  'Plush Llama', 'Mini Drone', 'Waffle Maker', 'Tea Towel', 'Cookbook', 'Yoga Mat',
  'Fart Machine', 'Lottery Tickets', 'Phone Stand', 'Bacon Jam', 'Glow Stars', 'Karaoke Mic',
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Match a button in either app: the React test id, or the legacy page's onclick. */
const sel = (testid, legacy) => `[data-testid="${testid}"], ${legacy}`;

class Arena {
  /**
   * @param {object} o
   * @param {import('@playwright/test').Browser} o.browser
   * @param {{name: string, persona: string}[]} o.seats
   * @param {Record<string,string>} o.settings  setup-screen <select> ids → values
   * @param {number} o.seed
   * @param {boolean} o.watch      slow pacing + live commentary
   * @param {number} o.pace        ms between actions (watch mode)
   * @param {string} o.outDir      where stuck snapshots/screenshots go
   * @param {(page, role, index) => Promise<void>} [o.onPage]  called for every page before it loads (watch mode streams it)
   * @param {import('@playwright/test').Page} [o.commentary]
   * @param {string} [o.join]     JOIN MODE: a room a person is hosting. The agents join it and play
   *                              only their own seats; the host page and any human players are left
   *                              to the people (no host clicks, no stall timeout).
   */
  constructor(o) {
    Object.assign(this, o);
    this.random = rng(o.seed);
    this.pace = o.watch ? (o.pace ?? 900) : (o.pace ?? 0);
    this.stallMs = o.stallMs ?? (o.join ? Infinity : o.watch ? 60000 : 25000); // people think slowly
    this.agents = {};       // playerId → { name, persona, brain, page, random, rivals }
    this.byName = {};
    this.values = {};       // giftName → base quality
    this.taste = {};        // playerId → giftName → offset
    this.giftQueue = [...GIFT_CATALOG].sort(() => this.random() - 0.5);
    this.issues = [];       // playability problems (deduped by text)
    this.duels = [];        // resolved duel records
    this.decisions = [];    // every agent decision (for per-persona stats)
    this.partyGames = [];   // party mini-game results
    this.turns = 0;
    this.acted = new Map(); // state key → { at, tries }
    this.voted = new Set();
  }

  // --- logging ---------------------------------------------------------------

  issue(kind, text, extra) {
    const key = `${kind}:${text}`;
    if (this.issues.some((i) => i.key === key)) return;
    this.issues.push({ key, kind, text, ...extra, at: Date.now() - this.t0 });
    this.say(`⚠️ ${kind}: ${text}`, 'warn');
  }

  say(text, kind = 'info') {
    if (this.log) this.log(text);
    if (this.commentary) this.commentary.evaluate(([t, k]) => window.feed && window.feed(t, k), [text, kind]).catch(() => {});
  }

  think(pid, text) {
    const a = this.agents[pid];
    this.say(`${a.persona.emoji} ${a.name}: ${text}`, 'agent');
  }

  // --- setup -----------------------------------------------------------------

  async setup() {
    this.t0 = Date.now();
    if (this.join) return this.setupJoin();
    const hostCtx = await this.browser.newContext({ viewport: { width: 1440, height: 900 } });
    this.contexts = [hostCtx];
    this.host = await hostCtx.newPage();
    this.hookPage(this.host, 'HOST');
    if (this.onPage) await this.onPage(this.host, 'host', 0);
    await this.host.goto(`${H.BASE}/host.html`);
    // The referee calls duels once the agents have voted, so no vote window (unless asked)
    for (const [id, value] of Object.entries({ 'vote-window': '0', ...this.settings })) {
      const select = this.host.locator(`#setting-${id}`);
      if ((id === 'vote-window' || id === 'duel-pick') && !(await select.count())) continue;
      await select.selectOption(value);
    }
    await this.host.getByRole('button', { name: /Create New Game/i }).click();
    await this.host.waitForSelector('#game-screen', { state: 'visible' });
    this.room = (await this.host.locator('#room-code-display').textContent()).trim();
    this.say(`🎄 Room ${this.room} — ${this.seats.length} agents, seed ${this.seed}`);

    await this.joinSeats();
    await this.host.getByRole('button', { name: /Start Game/i }).click();
    await this.started();
  }

  /** JOIN MODE: sit the agents down at a person's table, then wait for them to press Start. */
  async setupJoin() {
    this.room = this.join.toUpperCase();
    this.contexts = [];
    this.host = null;
    const g = await H.dbGet(this.room);
    if (!g) throw new Error(`Room ${this.room} not found — create the game on the host page first`);
    if (g.phase !== 'lobby') throw new Error(`Room ${this.room} has already started — agents can only join in the lobby`);
    const humans = Object.values(g.players || {}).map((x) => x.name);
    this.say(`🎄 Joining room ${this.room} — ${this.seats.length} agents${humans.length ? ` alongside ${humans.join(', ')}` : ''}`);
    await this.joinSeats();
    this.say('⏳ Agents are seated. Join on your phone too if you haven\'t, then press Start Game on the host.');
    await H.waitForGame(this.room, (x) => (x.phase !== 'lobby' ? x : false), { label: 'game start', timeout: 6 * 3600000 });
    await this.started();
  }

  async started() {
    const g = await H.waitForGame(this.room, (x) => (x.phase === 'main' ? x : false), { label: 'main phase' });
    this.startingChips = Object.values(g.players)[0].chips;
    this.say(`▶️ Game on — everyone starts with ${this.startingChips} chips`);
  }

  async joinSeats() {
    for (let i = 0; i < this.seats.length; i++) {
      const { name, persona } = this.seats[i];
      const ctx = await this.browser.newContext({ viewport: { width: 390, height: 844 } });
      this.contexts.push(ctx);
      const page = await ctx.newPage();
      this.hookPage(page, name);
      if (this.onPage) await this.onPage(page, 'phone', i);
      await page.goto(`${H.PLAYER_BASE}/player.html?room=${this.room}&testId=${i}&name=${encodeURIComponent(name)}`);
      const logs = [];
      page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
      await page.getByRole('button', { name: /Join Game/i }).click();
      if (!(await page.waitForSelector('#game-screen', { state: 'visible', timeout: 15000 }).catch(() => null))) {
        // Report what the phone showed, then tap Join again like a person would
        const toast = (await page.locator('#toast').textContent().catch(() => ''))?.trim();
        const shot = path.join(this.outDir, 'issues', `${this.room}-${name}-join.png`);
        fs.mkdirSync(path.dirname(shot), { recursive: true });
        await page.screenshot({ path: shot }).catch(() => {});
        this.issue('join-failed', `${name}: still on the join screen after 15s (toast: "${toast || 'none'}"; console: ${logs.slice(-3).join(' | ') || 'none'})`, { screenshot: shot });
        await page.getByRole('button', { name: /Join Game/i }).click().catch(() => {});
        await page.waitForSelector('#game-screen', { state: 'visible' });
      }
      this.byName[name] = { name, persona, page };
    }
    const players = await H.playersByName(this.room);
    for (const s of Object.values(this.byName)) {
      const id = players[s.name].id;
      const brain = decide(s.persona);
      this.agents[id] = {
        id, name: s.name, personaKey: s.persona, persona: brain.persona, brain, page: s.page,
        random: rng(this.seed * 31 + Object.keys(this.agents).length + 1), rivals: {},
      };
      this.taste[id] = {};
    }
    // Everyone gets a nemesis (only the Competitive Sibling cares)
    const ids = Object.keys(this.agents);
    for (const id of ids) {
      const others = ids.filter((x) => x !== id);
      this.agents[id].nemesis = others[Math.floor(this.random() * others.length)];
    }
    this.recentTopBids = []; // winning bids revealed so far (public)
  }

  hookPage(page, who) {
    page.on('pageerror', (e) => this.issue('page-error', `${who}: ${e.message}`));
    // The pages catch most errors and console.error them — those are bugs too
    page.on('console', (m) => {
      if (m.type() === 'error' && !/favicon|ERR_CONNECTION|net::/i.test(m.text())) this.issue('console-error', `${who}: ${m.text().slice(0, 160)}`);
    });
    page.on('dialog', async (d) => {
      try {
        if (d.type() === 'prompt' && /gift they're opening/i.test(d.message())) {
          const name = this.giftQueue.shift() || `Gift ${this.turns + 1}`;
          this.values[name] = 1 + Math.floor(this.random() * 10);
          await d.accept(name);
          return;
        }
        if (d.type() === 'confirm') { await d.accept(); return; }
        this.issue('alert', `${who}: ${d.message().slice(0, 120)}`);
        await d.accept();
      } catch { /* page closed */ }
    });
  }

  /** Ask play() to wrap up after its current step (join mode never stalls out by itself). */
  stop() { this.stopped = true; }

  async close() {
    for (const c of this.contexts || []) await c.close().catch(() => {});
  }

  // --- what an agent may know ---------------------------------------------------

  valueFor(pid, giftName) {
    const t = this.taste[pid];
    if (!(giftName in t)) t[giftName] = (this.random() + this.random() + this.random() - 1.5) * 2; // ~N(0, 1.4)
    // Gifts we didn't name (a person typed them in join mode, or an admin add) get a quality
    // the first time anyone sees them, like the ones we draw from GIFT_CATALOG
    this.values[giftName] ??= 1 + Math.floor(this.random() * 10);
    const base = this.values[giftName];
    return clamp(base + t[giftName], 0, 12);
  }

  view(g, pid) {
    const a = this.agents[pid];
    const p = g.players[pid];
    const gifts = {};
    for (const [id, gift] of Object.entries(g.gifts || {})) {
      gifts[id] = { name: gift.name, ownerId: gift.ownerId, naughtyLevel: gift.naughtyLevel || 0, value: this.valueFor(pid, gift.name) };
    }
    const players = {};
    for (const [id, x] of Object.entries(g.players)) players[id] = { name: x.name, giftId: x.giftId || null };
    const n = Object.keys(g.players).length;
    const baseCost = Number(g.settings?.baseCost ?? 1);
    return {
      me: { id: pid, name: p.name, chips: p.chips, giftId: p.giftId || null },
      gifts, players,
      expectedValue: 5.5,
      startingChips: this.startingChips,
      maxBid: Number(g.settings?.maxBid ?? 4),
      minCost: (naughty) => baseCost + naughty,
      progress: clamp(Object.keys(g.gifts || {}).length / n, 0, 1),
      endgame: g.phase === 'endgame',
      rivals: a.rivals,
      nemesis: a.nemesis,
      // who holds the gift I'd most like (the "leader", for Kingmakers)
      leaderId: Object.values(gifts).filter((x) => players[x.ownerId] && x.ownerId !== pid)
        .sort((x, y) => y.value - x.value)[0]?.ownerId || null,
      recentTopBids: this.recentTopBids.slice(-4),
      betsOn: !!g.settings?.spectatorBets,
      random: a.random,
    };
  }

  record(pid, kind, decision, extra = {}) {
    const a = this.agents[pid];
    this.decisions.push({ pid, persona: a.personaKey, kind, ...decision, ...extra });
    if (decision.thought) this.think(pid, decision.thought);
  }

  // --- clicking -----------------------------------------------------------------

  /** Click the first VISIBLE match; report a playability issue if it never shows. */
  async click(page, selector, who, what, timeout = 8000) {
    const loc = page.locator(selector).filter({ visible: true }).first();
    try {
      await loc.waitFor({ state: 'visible', timeout });
      await loc.click({ timeout });
      return true;
    } catch {
      const shot = path.join(this.outDir, 'issues', `${this.room}-${who}-${Date.now() - this.t0}.png`);
      fs.mkdirSync(path.dirname(shot), { recursive: true });
      await page.screenshot({ path: shot }).catch(() => {});
      this.issue('missing-button', `${who}: no visible "${what}" (${selector})`, { screenshot: shot });
      return false;
    }
  }

  async bid(pid, amount) {
    const { page, name } = this.agents[pid];
    await page.locator('#bid-panel').waitFor({ state: 'visible', timeout: 8000 }).catch(() => {});
    const ok = await this.click(page, sel(`bid-${amount}`, `#bid-quick-buttons button[onclick="setBid(${amount})"]`), name, `bid ${amount}`);
    if (ok) await this.click(page, '#submit-bid-btn', name, 'Seal bid');
  }

  // --- the loop -----------------------------------------------------------------

  /** Progress fingerprint: if this doesn't change for stallMs the game is stuck. */
  key(g) {
    const t = g.currentTurn;
    return JSON.stringify([
      g.phase, Object.keys(g.gifts || {}).length, t?.phase, t?.giftId, t?.openerId, t?.activePlayerId,
      t?.challengerId, t?.defenderId, t?.reprisalPlayerId, Object.keys(t?.bids || {}).length,
      Object.keys(t?.duelVotes || {}).length, g.lastDuel?.at,
      // party games: phase, drawings, votes and live scores count as progress
      g.minigame?.phase, Object.keys(g.minigame?.submissions || {}).length, Object.keys(g.minigame?.votes || {}).length,
      Object.values(g.minigame?.scores || {}).reduce((n, x) => n + (x?.score || 0), 0),
      Object.values(g.players).map((p) => [p.chips, p.giftId || null]),
    ]);
  }

  async play({ maxMs = 30 * 60000 } = {}) {
    let lastKey = null;
    let lastChange = Date.now();
    let lastDuelAt = null;
    let pendingDuel = null;
    for (;;) {
      const g = await H.dbGet(this.room);
      if (!g) throw new Error('room vanished');
      const k = this.key(g);
      if (k !== lastKey) { lastKey = k; lastChange = Date.now(); }

      // duel bookkeeping (lastDuel is what the host publishes for every resolved duel)
      const t = g.currentTurn;
      if (t?.phase === 'duel' && (!pendingDuel || pendingDuel.k !== `${t.challengerId}|${t.defenderId}|${t.giftId}|${!!t.isTiebreak}`)) {
        if (pendingDuel?.type === 'tiebreak') this.duels.push({ ...pendingDuel, k: undefined });
        this.duelSeq = (this.duelSeq || 0) + 1;
        pendingDuel = {
          k: `${t.challengerId}|${t.defenderId}|${t.giftId}|${!!t.isTiebreak}`,
          type: t.isTiebreak ? 'tiebreak' : t.isGambit ? 'gambit' : t.isReprisal ? 'reprisal' : t.isYieldDuel ? 'yield'
            : t.isEndgameSteal ? 'steal' : t.isEndgameAuction ? 'auction' : t.isNoBidsDuel ? 'nobids' : 'sealed',
          challengerId: t.challengerId, defenderId: t.defenderId, pot: t.pot || 0, tax: t.tax || 0,
        };
      }
      if (g.lastDuel?.at && g.lastDuel.at !== lastDuelAt) {
        if (lastDuelAt !== null || this.duels.length || pendingDuel) {
          const d = g.lastDuel;
          this.duels.push({ ...(pendingDuel || {}), k: undefined, winnerId: d.winnerId, loserId: d.loserId, potWon: d.pot, dividend: d.dividend });
          const loser = this.agents[d.loserId];
          if (loser) loser.rivals[d.winnerId] = (loser.rivals[d.winnerId] || 0) + 1;
          pendingDuel = null;
        }
        lastDuelAt = g.lastDuel.at;
      }

      if (g.phase === 'ended') return this.finish(g, 'ended');
      if (this.stopped) return this.finish(g, 'stopped');
      // Invariants only on a settled state (host writes gift/owner/turn in separate steps)
      if (!t && Date.now() - lastChange > 1500) {
        for (const e of H.invariantErrors(g)) this.issue('invariant', e);
      }
      if (Date.now() - lastChange > this.stallMs) return this.finish(g, 'stuck', await this.snapshot(g));
      if (Date.now() - this.t0 > maxMs) return this.finish(g, 'timeout', await this.snapshot(g));

      // Between turns the host writes several paths in a row — wait until it settles
      const action = !t && Date.now() - lastChange < 800 ? null : this.nextAction(g);
      if (action) {
        const prev = this.acted.get(action.key);
        // Act once per state; retry only if nothing moved for a while (click lost?)
        if (!prev || (k === prev.stateKey && Date.now() - prev.at > 7000 && prev.tries < 3)) {
          this.acted.set(action.key, { at: Date.now(), stateKey: k, tries: (prev?.tries || 0) + 1 });
          if (prev) this.issue('retry', `had to retry: ${action.desc}`);
          if (this.verbose) console.log(`  [${((Date.now() - this.t0) / 1000).toFixed(1)}s] ${action.desc}`);
          if (this.pace) await sleep(this.pace);
          try { await action.run(); } catch (e) { this.issue('action-failed', `${action.desc}: ${e.message.split('\n')[0]}`); }
          continue;
        }
      }
      await sleep(this.watch ? 250 : 150);
    }
  }

  /** Work out the single next move (or null = waiting on the game itself). */
  nextAction(g) {
    const t = g.currentTurn;
    const host = this.host;
    const players = g.players;
    const ids = Object.keys(players);
    const gifts = g.gifts || {};
    const name = (id) => players[id]?.name || id;
    const A = (key, desc, run) => ({ key, desc, run });
    // Join mode: a person runs the host page, and people make their own moves
    const hostA = (key, desc, run) => (this.host ? A(key, desc, run) : null);
    const agent = (id) => !!this.agents[id];

    if (!t) {
      if (g.phase === 'main') {
        const wrapped = ids.length - Object.keys(gifts).length;
        const inBag = ids.filter((id) => !players[id].giftId).length;
        if (wrapped > 0 && inBag > 0) {
          // keyed by turns SEEN (not clicked): the host writes gift → owner → turn in steps
          return hostA(`draw#${this.turns}`, 'draw from the sack', async () => {
            await this.click(host, '#btn-draw-player:not([disabled])', 'HOST', 'Draw from Santa\'s Sack');
          });
        }
        if (inBag > 0) return hostA('endgame', 'start the endgame', async () => {
          this.say('🎰 All gifts open — Misfit Lottery!');
          await this.click(host, '#btn-endgame:not([disabled])', 'HOST', 'Endgame');
        });
        return hostA('end', 'end the game', () => this.click(host, sel('end-game', 'button[onclick="endGame()"]'), 'HOST', 'End Game for All'));
      }
      return null; // endgame: host draws the next player itself
    }

    // One key per decision point. lastDuel.at separates repeats of an identical state
    // (e.g. two Reprisals by the same player) — but NOT inside a duel: the host publishes
    // lastDuel before it clears the turn, which would look like a brand-new duel.
    const tk = `${t.phase}|${t.giftId}|${t.openerId}|${t.activePlayerId}|${t.challengerId}|${t.defenderId}|${t.reprisalPlayerId}|` +
      (t.phase === 'duel' ? `duel#${this.duelSeq}` : g.lastDuel?.at);
    const opened = t.giftId ? gifts[t.giftId] : null;
    const positiveBids = Object.entries(t.bids || {}).filter(([, b]) => b.amount > 0);

    switch (t.phase) {
      case 'bidding': {
        if (opened && !this.announced?.has(t.giftId)) {
          (this.announced ||= new Set()).add(t.giftId);
          this.turns++;
          this.valueFor(ids.find((id) => this.agents[id]) || t.openerId, opened.name); // assigns its quality
          this.say(`🎁 ${name(t.openerId)} opens "${opened.name}" (quality ${this.values[opened.name]}/10)`, 'gift');
        }
        const waiting = ids.find((id) => id !== t.openerId && !t.bids?.[id] && agent(id));
        if (waiting) {
          return A(`${tk}|bid|${waiting}`, `${name(waiting)} bids`, async () => {
            const d = this.agents[waiting].brain.bid(this.view(g, waiting), this.valueFor(waiting, opened.name));
            this.record(waiting, 'bid', d, { gift: opened.name });
            await this.bid(waiting, d.amount);
          });
        }
        if (ids.some((id) => id !== t.openerId && !t.bids?.[id])) return null; // a person is still bidding
        return hostA(`${tk}|reveal`, 'reveal bids', () => this.click(host, sel('reveal-bids', 'button[onclick="revealBids()"]'), 'HOST', 'Reveal All Bids'));
      }

      case 'reveal': {
        if (!this.seenReveal?.has(t.giftId)) {
          (this.seenReveal ||= new Set()).add(t.giftId);
          if (positiveBids.length) this.recentTopBids.push(Math.max(...positiveBids.map(([, b]) => b.amount)));
        }
        const opener = t.openerId;
        if (positiveBids.length === 0) {
          if (!agent(opener)) return null;
          return A(`${tk}|keep-or-gambit`, `${name(opener)} keep/Gambit`, async () => {
            const targets = Object.entries(gifts).filter(([id, x]) => id !== t.giftId && x.ownerId !== opener && players[x.ownerId]);
            const v = this.view(g, opener);
            const d = this.agents[opener].brain.keepOrGambit(v, targets.map(([id]) => [id, v.gifts[id]]));
            this.record(opener, 'keep-or-gambit', d);
            const page = this.agents[opener].page;
            if (d.action === 'gambit') {
              if (await this.click(page, sel('gambit', 'button[onclick="showGambitOptions()"]'), name(opener), 'Grinch\'s Gambit')) {
                if (await this.click(page, sel(`gambit-target-${d.giftId}`, `[onclick="executeGambit('${d.giftId}')"]`), name(opener), 'Gambit target')) return;
                await this.click(page, sel('cancel-gambit', 'button[onclick="cancelGambit()"]'), name(opener), 'Cancel Gambit'); // back to Keep
              }
            }
            await this.click(page, sel('keep-gift', 'button[onclick="keepMyGift()"]'), name(opener), 'Keep My Gift');
          });
        }
        if (positiveBids.length === 1) {
          return hostA(`${tk}|start-duel`, 'start the duel', () => this.click(host, sel('start-duel', 'button[onclick^="startDuel("]'), 'HOST', 'Start Duel'));
        }
        if (!agent(opener)) return null;
        return A(`${tk}|fight-or-yield`, `${name(opener)} fight/yield`, () => this.fightOrYield(g, opener, opened));
      }

      case 'yield_choice':
        if (!agent(t.openerId)) return null;
        return A(`${tk}|fight-or-yield`, `${name(t.openerId)} fight/yield`, () => this.fightOrYield(g, t.openerId, opened));
      case 'duel_pending':
        return hostA(`${tk}|start`, 'start the duel', () => this.click(host, sel('start-duel', 'button[onclick^="startDuel("]'), 'HOST', 'Start Duel!'));
      case 'yield_pending':
        return hostA(`${tk}|start`, 'start the yield duel', () => this.click(host, sel('start-yield-duel', 'button[onclick^="openerYields("]'), 'HOST', 'Start Yield Duel!'));
      case 'tiebreaker_pending':
        // Mini-game tiebreakers start themselves; only the random method has a button
        if ((g.settings?.tiebreakerMethod || 'random') !== 'random') return null;
        return hostA(`${tk}|start`, 'start the tiebreaker', () => this.click(host, sel('start-tiebreaker', 'button[onclick^="startTiebreaker("]'), 'HOST', 'Start Tiebreaker Duel!'));

      case 'duel': {
        const dk = `${t.challengerId}|${t.defenderId}|${t.giftId}|${!!t.isTiebreak}`;
        // Which duel gets played (React host: settings.duelPick; the original pages have none)
        const duelPick = g.settings?.duelPick || 'off';
        const listed = g.settings?.duelGames; // text, one per line (older games: an array)
        const tableGames = typeof listed === 'string' ? listed.split('\n').filter((x) => x.trim()).length : Object.values(listed || {}).length;
        const duelGames = tableGames + Object.values(g.settings?.duelAppGames || {}).length;
        if (!t.duelGame && duelPick !== 'off' && duelGames > 0) {
          // who picks: an under-16 duellist (the youngest; a tie → the defender), else the house rule
          const young = (id) => (typeof players[id]?.age === 'number' && players[id].age < 16 ? players[id].age : null);
          const c = young(t.challengerId), d = young(t.defenderId);
          const picker = c != null && (d == null || c < d) ? t.challengerId
            : d != null ? t.defenderId
            : duelPick === 'wheel' || t.isYieldDuel || t.isTiebreak ? null : t.defenderId;
          if (!picker || !agent(picker)) return null; // the host spins / a person picks
          return A(`${tk}|duel-game`, `${name(picker)} picks the duel`, async () => {
            const page = this.agents[picker].page;
            const a = this.agents[picker];
            if (a.random() < 0.2) {
              this.record(picker, 'duel-game', { pick: 'wheel', thought: 'lets the wheel decide' });
              await this.click(page, '[data-testid="duel-game-spin"]', name(picker), 'Let the wheel pick');
              return;
            }
            const i = Math.floor(a.random() * duelGames);
            this.record(picker, 'duel-game', { pick: i, thought: 'picks the duel' });
            await this.click(page, `[data-testid="duel-game-${i}"]`, name(picker), 'a duel');
          });
        }
        if (typeof t.wheel?.spunAt === 'number' && Date.now() < t.wheel.spunAt + 5200) return null; // let it stop
        if (t.duelGameWinner) return null; // an in-app duel game decided it; the host settles it
        // Every spectator decides (vote or not) before the referee calls the duel
        const spectators = t.isTiebreak ? [] : ids.filter((id) =>
          id !== t.challengerId && id !== t.defenderId && !t.duelVotes?.[id] && !this.voted.has(`${dk}|${id}`) && agent(id));
        for (const voter of spectators) {
          this.voted.add(`${dk}|${voter}`);
          const d = this.agents[voter].brain.vote(this.view(g, voter));
          if (!d) continue;
          const pick = d.pick === 'challenger' ? t.challengerId : t.defenderId;
          return A(`${tk}|vote|${voter}`, `${name(voter)} votes`, () =>
            this.click(this.agents[voter].page, sel(`vote-${pick}`, `button[onclick="voteOnDuel('${pick}')"]`), name(voter), 'vote', 10000)); // a busy machine can lag the phone behind the wheel
        }
        // An in-app duel game (Whack-an-Elf…): the referee starts it, the game decides the winner
        if (t.duelGame?.kind === 'app') {
          return hostA(`${tk}|duel-app`, `start ${t.duelGame.name}`, () =>
            this.click(host, '[data-testid="start-duel-app"]:not([disabled])', 'HOST', `Start ${t.duelGame.name}`));
        }
        const fn = t.isTiebreak ? 'recordTiebreakerWinner' : 'recordDuelWinner';
        return hostA(`${tk}|winner`, 'declare the duel winner', async () => {
          await sleep(this.watch ? 1800 : 200); // let people see the duel
          const winner = this.random() < 0.5 ? t.challengerId : t.defenderId; // fair coin
          this.say(`⚔️ ${name(t.challengerId)} vs ${name(t.defenderId)}${t.isTiebreak ? ' (tiebreak)' : opened ? ` for "${opened.name}"` : ''} → 🏆 ${name(winner)}`, 'duel');
          // Sealed/Gambit/Reprisal duels show a modal over the turn panel's copy of the buttons
          const modalOpen = await host.locator('#duel-modal-overlay.active').count();
          await this.click(host, modalOpen
            ? sel(`modal-winner-${winner}`, `#duel-modal-overlay button[onclick="${fn}('${winner}')"]`)
            : sel(`${t.isTiebreak ? 'tiebreak' : 'duel'}-winner-${winner}`, `#action-buttons button[onclick="${fn}('${winner}')"]`), 'HOST', `${name(winner)} wins`);
          // The TV verdict: let watchers enjoy it, otherwise tap it away
          if (!t.isTiebreak) {
            if (this.watch) {
              await host.locator('#duel-verdict').waitFor({ state: 'attached', timeout: 2500 }).catch(() => {});
              await host.locator('#duel-verdict').waitFor({ state: 'detached', timeout: 9000 }).catch(() => {});
            }
            else await host.locator('#duel-verdict').click({ timeout: 2500 }).catch(() => {});
          }
        });
      }

      case 'reprisal': {
        const pid = t.reprisalPlayerId;
        if (!agent(pid)) return null;
        return A(`${tk}|reprisal`, `${name(pid)} reprisal`, async () => {
          const v = this.view(g, pid);
          const targets = Object.entries(v.gifts).filter(([id, x]) =>
            id !== t.excludedGiftId && x.ownerId !== pid && (x.ownerId === 'misfit' || players[x.ownerId]));
          const d = this.agents[pid].brain.reprisal(v, targets);
          this.record(pid, 'reprisal', d, { misfit: d.giftId ? gifts[d.giftId]?.ownerId === 'misfit' : undefined });
          const page = this.agents[pid].page;
          if (d.giftId && await this.click(page, sel(`reprisal-target-${d.giftId}`, `[onclick="selectReprisalTarget('${d.giftId}')"]`), name(pid), 'Reprisal target')) return;
          await this.click(page, sel('skip-reprisal', 'button[onclick="skipReprisal()"]'), name(pid), 'Skip Reprisal');
        });
      }

      case 'endgame_choice': {
        const pid = t.activePlayerId;
        if (!agent(pid)) return null;
        return A(`${tk}|path`, `${name(pid)} picks a path`, async () => {
          const v = this.view(g, pid);
          const steal = Object.entries(v.gifts).filter(([, x]) => x.ownerId !== pid && players[x.ownerId]);
          const d = this.agents[pid].brain.endgamePath(v, steal, this.valueFor(pid, t.misfitName));
          this.record(pid, 'endgame-path', d, { misfit: t.misfitName });
          (this.stealPlan ||= {})[tk.replace('endgame_choice', 'endgame_steal_select')] = d.giftId;
          const page = this.agents[pid].page;
          if (d.path === 'steal' && await this.click(page, sel('path-a', 'button[onclick="showEndgameStealOptions()"]'), name(pid), 'Path A')) return;
          await this.click(page, sel('path-b', 'button[onclick="startEndgameAuction()"]'), name(pid), 'Path B');
        });
      }

      case 'endgame_steal_select': {
        const pid = t.activePlayerId;
        if (!agent(pid)) return null;
        return A(`${tk}|steal`, `${name(pid)} steals`, async () => {
          const target = this.stealPlan?.[tk];
          const page = this.agents[pid].page;
          if (target && await this.click(page, sel(`steal-target-${target}`, `[onclick="executeEndgameSteal('${target}')"]`), name(pid), 'steal target')) return;
          await this.click(page, sel('cancel-steal', 'button[onclick="cancelEndgameSteal()"]'), name(pid), 'Cancel steal');
        });
      }

      case 'endgame_bidding': {
        const waiting = ids.find((id) => id !== t.activePlayerId && !t.bids?.[id] && agent(id));
        if (waiting) {
          return A(`${tk}|bid|${waiting}`, `${name(waiting)} bids on the Misfit`, async () => {
            const d = this.agents[waiting].brain.bid(this.view(g, waiting), this.valueFor(waiting, t.misfitName));
            this.record(waiting, 'endgame-bid', d, { gift: t.misfitName });
            await this.bid(waiting, d.amount);
          });
        }
        if (ids.some((id) => id !== t.activePlayerId && !t.bids?.[id])) return null; // a person is still bidding
        return hostA(`${tk}|reveal`, 'reveal auction bids', () =>
          this.click(host, sel('reveal-auction', 'button[onclick="revealEndgameAuctionBids()"]'), 'HOST', 'Reveal Bids'));
      }

      case 'endgame_reveal':
        return hostA(`${tk}|duel`, 'duel the highest bidder', () =>
          this.click(host, sel('auction-duel', 'button[onclick^="startEndgameAuctionDuel("]'), 'HOST', 'Duel Highest Bidder'));

      case 'endgame_nobids': {
        const pid = t.activePlayerId;
        if (!agent(pid)) return null;
        return A(`${tk}|defender`, `${name(pid)} picks a defender`, async () => {
          const page = this.agents[pid].page;
          const pickSel = '[data-testid^="nobids-defender-"], [onclick^="executeNoBidsDuel("]';
          await page.locator(pickSel).first().waitFor({ state: 'visible', timeout: 8000 }).catch(() => {});
          const candidates = await page.locator(pickSel).evaluateAll((els) => els.map((e) =>
            e.dataset.testid?.slice('nobids-defender-'.length) || (e.getAttribute('onclick')?.match(/'([^']+)'/) || [])[1]).filter(Boolean));
          const d = this.agents[pid].brain.noBidsDefender(this.view(g, pid), candidates);
          this.record(pid, 'nobids-defender', d);
          if (d.playerId) await this.click(page, sel(`nobids-defender-${d.playerId}`, `[onclick="executeNoBidsDuel('${d.playerId}')"]`), name(pid), 'defender');
          else this.issue('dead-end', `${name(pid)}: no-bids Misfit duel but no defender to pick`);
        });
      }

      case 'minigame_party': return this.partyAction(g, A);

      case 'minigame_draw_off':
        if (!this.host) return null; // people draw this one; the agents sit it out
        this.issue('unsupported', 'AI Draw Off is not agent-driven (needs the AI critic) — use another tiebreaker method');
        return null;

      default:
        this.issue('unknown-phase', `turn phase "${t.phase}"`);
        return null;
    }
  }

  // --- party mini-games (Sketch-Off, Whack-an-Elf): the whole table plays ---------------

  partyAction(g, A) {
    const mg = g.minigame;
    if (!mg) return null;
    const mk = `party|${mg.seed}|${mg.phase}`;
    const name = (id) => g.players[id]?.name || id;
    const ids = (mg.participants || []).filter((id) => this.agents[id]);
    if (mg.phase === 'intro' || mg.phase === 'results') {
      if (mg.phase === 'results' && mg.results && !this.partyGames.some((x) => x.seed === mg.seed)) {
        this.partyGames.push({ type: mg.type, seed: mg.seed, tied: mg.tiedBidders, ...mg.results });
        const r = mg.results;
        this.say(`🎉 ${r.winners.map(name).join(' & ')} win${r.winners.length > 1 ? '' : 's'} the tie${r.bonus ? ` · Elf Bonus → ${name(r.bonus)}` : ''}`, 'duel');
      }
      if (!this.host) return null;
      return A(mk, mg.phase === 'intro' ? 'start the party game' : 'continue to the duel', async () => {
        await sleep(this.watch ? 4000 : 300); // let people read the rules / results
        await this.click(this.host, sel('skip-party-phase', 'button[onclick="skipPartyPhase()"]'), 'HOST', 'Skip ahead');
      });
    }
    if (mg.phase === 'playing' && mg.type === 'whack_elf') {
      return A(mk, 'everyone whacks elves', () => Promise.all(ids.map((id) => this.whack(id, mg))));
    }
    if (mg.phase === 'playing' && mg.type === 'sketch_off') {
      const who = ids.find((id) => !mg.submissions?.[id]);
      return who ? A(`${mk}|${who}`, `${name(who)} sketches`, () => this.sketch(who)) : null;
    }
    if (mg.phase === 'voting') {
      const drawings = Object.keys(mg.submissions || {});
      const voter = ids.find((id) => !mg.votes?.[id] && drawings.some((d) => d !== id));
      if (!voter) return null;
      return A(`${mk}|${voter}`, `${name(voter)} votes`, async () => {
        const options = drawings.filter((d) => d !== voter);
        const pick = options[Math.floor(this.agents[voter].random() * options.length)];
        this.record(voter, 'sketch-vote', { pick, thought: `votes for ${name(pick)}'s drawing` });
        await this.click(this.agents[voter].page, `#party-card [data-vote="${pick}"]`, name(voter), 'vote for a drawing');
      });
    }
    return null;
  }

  /** One agent plays Whack-an-Elf on its own phone until time's up. */
  async whack(pid, mg) {
    const a = this.agents[pid];
    const p = a.persona;
    let hits = 0;
    let slips = 0;
    const seen = {}; // "cell|kind" → when this player can react to it
    while (Date.now() < mg.phaseEndsAt - 200) { // phaseEndsAt is server time; the emulator is local
      // Human-ish: an elf can only be hit once it has been up for this player's reaction
      // time (reflexMs ± 40%), so slow players miss the quick ones late in the game.
      const now = Date.now();
      const up = await a.page.locator('#party-card .whack-cell.up').evaluateAll((els) => els.map((e) => `${e.dataset.cell}|${e.dataset.kind}`));
      for (const key of Object.keys(seen)) if (!up.includes(key)) delete seen[key];
      for (const key of up) seen[key] ??= now + p.reflexMs * (0.6 + a.random() * 0.8);
      const ready = up.filter((key) => seen[key] <= now);
      const good = ready.filter((key) => !key.endsWith('|grinch'));
      const bad = ready.filter((key) => key.endsWith('|grinch'));
      const target = bad.length && a.random() < p.slip ? bad[0] : good[0];
      if (target) {
        delete seen[target];
        await a.page.locator(`#party-card .whack-cell[data-cell="${target.split('|')[0]}"]`).dispatchEvent('pointerdown').catch(() => {});
        if (target.endsWith('|grinch')) slips++; else hits++;
      }
      await sleep(60);
    }
    this.record(pid, 'whack', { hits, slips, thought: `whacked ${hits} el${hits === 1 ? 'f' : 'ves'}${slips ? `, hit the Grinch ${slips}×` : ''}` });
  }

  /** One agent scribbles something and hands it in. */
  async sketch(pid) {
    const a = this.agents[pid];
    const pad = a.page.locator('#party-card canvas.pg-pad');
    await pad.waitFor({ state: 'visible', timeout: 8000 });
    const b = await pad.boundingBox();
    const strokes = 2 + Math.floor(a.random() * 4);
    for (let s = 0; s < strokes; s++) {
      let x = b.x + b.width * (0.2 + a.random() * 0.6);
      let y = b.y + b.height * (0.2 + a.random() * 0.6);
      await a.page.mouse.move(x, y);
      await a.page.mouse.down();
      for (let i = 0; i < 8; i++) {
        x = Math.min(b.x + b.width - 4, Math.max(b.x + 4, x + (a.random() - 0.5) * b.width * 0.25));
        y = Math.min(b.y + b.height - 4, Math.max(b.y + 4, y + (a.random() - 0.5) * b.height * 0.25));
        await a.page.mouse.move(x, y, { steps: 2 });
      }
      await a.page.mouse.up();
    }
    if (this.watch) await sleep(800);
    this.record(pid, 'sketch', { strokes, thought: `draws a masterpiece (${strokes} strokes)` });
    await this.click(a.page, '#party-card [data-act="done"]', a.name, 'Done drawing');
  }

  async fightOrYield(g, opener, opened) {
    const d = this.agents[opener].brain.fightOrYield(this.view(g, opener), this.valueFor(opener, opened.name));
    this.record(opener, 'fight-or-yield', d);
    const page = this.agents[opener].page;
    const fn = d.action === 'yield' ? 'openerChooseYield' : 'openerChooseFight';
    await this.click(page, sel(d.action === 'yield' ? 'yield' : 'fight', `button[onclick^="${fn}("]`), g.players[opener].name, d.action === 'yield' ? 'Yield!' : 'Fight!');
  }

  // --- results ------------------------------------------------------------------

  async snapshot(g) {
    const dir = path.join(this.outDir, `stuck-${this.room}`);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'state.json'), JSON.stringify(g, null, 2));
    await this.host?.screenshot({ path: path.join(dir, 'host.png') }).catch(() => {});
    for (const a of Object.values(this.agents)) await a.page.screenshot({ path: path.join(dir, `${a.name}.png`) }).catch(() => {});
    this.say(`🧊 STUCK — snapshot in ${dir}`, 'warn');
    return dir;
  }

  finish(g, outcome, snapshotDir) {
    const players = g.players;
    const seats = Object.values(this.agents).map((a) => {
      const p = players[a.id];
      const gift = p.giftId ? g.gifts[p.giftId] : null;
      const myVal = gift ? this.valueFor(a.id, gift.name) : null;
      // rank of what I ended with among every opened gift, by MY taste (1 = best)
      const all = Object.values(g.gifts || {}).map((x) => this.valueFor(a.id, x.name)).sort((x, y) => y - x);
      const best = all[0] ?? null;
      return {
        name: a.name, persona: a.personaKey, chips: p.chips, partyPoints: p.partyPoints || 0,
        votingWins: p.votingWins || 0, votingLosses: p.votingLosses || 0,
        gift: gift?.name || null, value: myVal, best,
        rank: myVal == null ? null : all.indexOf(myVal) + 1, gifts: all.length,
        wins: this.duels.filter((d) => d.winnerId === a.id).length,
        losses: this.duels.filter((d) => d.loserId === a.id).length,
      };
    });
    const result = {
      room: this.room, seed: this.seed, outcome, snapshotDir,
      settings: { ...this.settings }, gameSettings: g.settings,
      ms: Date.now() - this.t0, turns: this.turns,
      startingChips: this.startingChips,
      seats, duels: this.duels.map(({ k, ...d }) => d), decisions: this.decisions, partyGames: this.partyGames,
      issues: this.issues.map(({ key, ...i }) => i),
    };
    this.say(`🏁 ${outcome.toUpperCase()} after ${(result.ms / 1000).toFixed(0)}s, ${this.turns} draws, ${this.duels.length} duels`);
    for (const s of [...seats].sort((a, b) => (b.value ?? -1) - (a.value ?? -1))) {
      this.say(`   ${this.agents[Object.keys(this.agents).find((id) => this.agents[id].name === s.name)].persona.emoji} ${s.name}: ${s.gift ? `"${s.gift}" (${s.value.toFixed(1)}, #${s.rank}/${s.gifts} for them)` : 'NO GIFT'} · ${s.chips} chips`, 'result');
    }
    return result;
  }
}

module.exports = { Arena, GIFT_CATALOG };
