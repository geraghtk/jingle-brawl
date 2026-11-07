// The same table, played on the real app: the host page and one phone per person in a real
// browser, against the room server. Each offer (core/table.ts) is pressed by clicking that
// button on that person's own page, so the app's screens, timers and sync are all in play. What a
// person sees is the real pages' text: the TV (the host page) and their own phone. Real time:
// advance(ms) waits. People's skills, tastes and hands, and the scripted host, are sim/world.ts.
//
// Ids: the people are addressed by their own ids (as in the engine table); the app's player ids
// are translated both ways, so an offer reads the same on either backend.
import { createRequire } from 'node:module';
import type { Browser, Page } from '@playwright/test';
import type { PartyGame } from '../core/party';
import { GAMES } from '../core/party';
import { rng as makeRng } from '../core/rng';
import { HOST, isOffered, offers } from '../core/table';
import type { Offer, Who } from '../core/table';
import type { Game, PlayerId, Settings } from '../core/types';
import type { Observation, TableEvent } from './table';
import { describe as describeOffer } from './table';
import { avatarLabel, drawingsOf, GIFTS, PatientHost, Physics, pickAvatar, situation } from './world';
import type { AvatarPicks, HostOptions, Person } from './world';

const require = createRequire(import.meta.url);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Settings → the host's setup screen (ids of its fields). Unset fields keep the screen's defaults. */
export function setupFields(s: Settings): Record<string, string> {
  const f: Record<string, string> = { 'vote-window': '30', 'duel-pick': 'defender' }; // the screen's own defaults
  const put = (id: string, v: unknown) => { if (v !== undefined && v !== null) f[id] = String(v); };
  put('reprisal-depth', s.maxReprisalDepth); put('max-bid', s.maxBid); put('max-payout', s.maxWinnerPayout);
  put('starting-chips', s.startingChips); put('loser-dividend', s.loserDividendRule); put('spectator-bets', s.spectatorBets);
  put('vote-window', s.voteWindow); put('duel-pick', s.duelPick); put('tax-threshold', s.taxThreshold); put('base-cost', s.baseCost);
  put('tiebreaker-method', s.tiebreakerMethod); put('party-prize', s.partyPrize);
  if (s.duelGames !== undefined) f['duel-games'] = Array.isArray(s.duelGames) ? s.duelGames.join('\n') : s.duelGames;
  for (const a of s.duelAppGames || []) f[`duel-app-${a}`] = 'on';
  if (s.publicChips) f['public-chips'] = 'on';
  return f;
}

export interface LiveOptions {
  seed: number;
  people: Person[];
  settings?: Settings;
  autoHost?: boolean | HostOptions;
  base?: string;                // the app (default: the Vite dev server)
  headed?: boolean;
  slowMo?: number;
}

export class LiveTable {
  game: Game = { phase: 'lobby' };
  room = '';
  readonly log: TableEvent[] = [];
  readonly did: Record<string, Record<string, number>> = {};
  readonly issues: string[] = [];
  readonly physics: Physics;
  private browser!: Browser;
  private hostPage!: Page;
  private phones: Record<string, Page> = {};
  private idOf: Record<string, PlayerId> = {};       // person → app player id
  private personOf: Record<PlayerId, string> = {};   // app player id → person
  private host: PatientHost | null;
  private drawn: Record<PlayerId, number> = {};      // drawings these hands made, and how good
  private busy: Promise<unknown>[] = [];
  private whacking = new Set<string>();             // people playing Whack-an-Elf right now
  readonly avatars: AvatarPicks = {};                // who picked which avatar as they joined
  private t0 = Date.now();
  private lastKey = '';
  private H: any;                                    // e2e/helpers.js (emulator REST, setup)

  private constructor(private o: LiveOptions) {
    this.physics = new Physics(o.seed, o.people);
    const ah = o.autoHost === true ? {} : o.autoHost || null;
    this.host = ah ? new PatientHost(ah, Date.now()) : null;
  }

  get people() { return this.o.people; }

  static async open(o: LiveOptions): Promise<LiveTable> {
    process.env.BASE_URL = o.base || process.env.BASE_URL || 'http://localhost:5173';
    const t = new LiveTable(o);
    t.H = require('../../../e2e/helpers.js');
    const { chromium } = require('@playwright/test');
    t.browser = await chromium.launch({ headless: !o.headed, slowMo: o.slowMo });
    const { host, roomCode } = await t.H.createGame(t.browser, setupFields(o.settings || {}));
    t.hostPage = host;
    t.room = roomCode;
    t.watch(host, 'HOST');
    const r = makeRng(o.seed + 7919), taken = new Set<string>();                  // the avatar picker, as each joins
    for (const [i, p] of o.people.entries()) {
      const { press, avatar } = pickAvatar(p, taken, r);
      taken.add(avatar);
      t.avatars[p.id] = { avatar, label: avatarLabel(avatar), picked: !!press };
      const ctx = await t.browser.newContext({ viewport: { width: 390, height: 844 } });
      const page = await ctx.newPage();
      t.watch(page, p.name);
      const join = async () => {
        await page.goto(`${process.env.PLAYER_BASE || process.env.BASE_URL}/player.html?room=${roomCode}&testId=${100 + i}&name=${encodeURIComponent(p.name)}`);
        if (p.age != null && p.age < 16) {
          await page.getByTestId('under-16').check();
          await page.getByTestId('age').selectOption(String(p.age));
        }
        if (press) {
          const opt = page.locator(`[data-avatar="${press}"]`);
          await opt.click();
          await page.waitForFunction((el) => el?.getAttribute('aria-pressed') === 'true', await opt.elementHandle());
        }
        await page.getByTestId('join').click();
        await page.waitForSelector('#game-screen', { state: 'visible', timeout: 15000 });
      };
      // a phone that doesn't get in is a playability issue (kept), not the end of the run: one retry
      await join().catch(async (e) => { t.issue(`${p.name}: joining failed, tried again (${String(e.message).split('\n')[0]})`); await join(); });
      t.phones[p.id] = page;
    }
    const byName = await t.H.playersByName(roomCode);
    for (const p of o.people) { t.idOf[p.id] = byName[p.name].id; t.personOf[byName[p.name].id] = p.id; }
    await t.refresh();
    return t;
  }

  private watch(page: Page, who: string) {
    page.on('pageerror', (e) => this.issue(`${who}: page error: ${e.message}`));
    page.on('dialog', () => {});                      // answered where they're expected
  }

  private issue(text: string) {
    this.issues.push(text);
    this.note(`⚠️ ${text}`);
  }

  private note(text: string, who?: Who, kind: TableEvent['kind'] = 'event', priv = false) {
    this.log.push({ seq: this.log.length + 1, t: Date.now() - this.t0, text, who, kind, ...(priv ? { private: true } : {}) });
  }

  async refresh(): Promise<Game> {
    this.game = (await this.H.dbGet(this.room)) || this.game;
    const t = this.game.currentTurn;
    const key = JSON.stringify([this.game.phase, t?.phase, t?.giftId, t?.challengerId, t?.defenderId, Object.keys(t?.bids || {}).length,
      Object.keys(t?.duelVotes || {}).length, this.game.lastDuel?.at, (this.game.minigame as PartyGame | undefined)?.phase]);
    if (key !== this.lastKey) {
      const d = this.game.lastDuel;
      const before = this.lastKey ? JSON.parse(this.lastKey) : [];
      if (d && d.at !== before[7]) this.note(`${this.game.players?.[d.winnerId]?.name} beat ${this.game.players?.[d.loserId]?.name}`);
      this.lastKey = key;
      this.host?.changed(Date.now());
    }
    return this.game;
  }

  // --- ids ----------------------------------------------------------------------------------------

  private real(who: Who): Who { return who === HOST ? HOST : this.idOf[who] ?? who; }
  private out(o: Offer): Offer {
    const p = (id: string) => this.personOf[id] ?? id;
    if (o.type === 'call_duel' || o.type === 'vote_sketch') return { ...o, pick: p(o.pick) };
    if (o.type === 'duel_winner') return { ...o, id: p(o.id) };
    if (o.type === 'nobids_defender') return { ...o, playerId: p(o.playerId) };
    return o;
  }
  private in(o: Offer): Offer {
    const r = (id: string) => this.idOf[id] ?? id;
    if (o.type === 'call_duel' || o.type === 'vote_sketch') return { ...o, pick: r(o.pick) };
    if (o.type === 'duel_winner') return { ...o, id: r(o.id) };
    if (o.type === 'nobids_defender') return { ...o, playerId: r(o.playerId) };
    return o;
  }

  // --- seeing -------------------------------------------------------------------------------------

  offers(who: Who): Offer[] {
    return offers(this.game, this.real(who), Date.now()).map((o) => this.out(o))
      .filter((o) => !(o.type === 'whack' && this.whacking.has(who)));   // already at it: the phone's busy
  }

  private async screen(page: Page) {
    const text = await page.locator('#game-screen').innerText({ timeout: 3000 }).catch(() => '');
    const lines = text.split('\n').map((x) => x.trim()).filter(Boolean).slice(0, 60);
    return { title: lines[0] || '', lines: lines.slice(1) };
  }

  async observe(who: Who): Promise<Observation> {
    await this.refresh();
    const person = who === HOST ? null : who;
    const mg = this.game.minigame as PartyGame | undefined;
    const drawings = mg?.phase === 'voting' && mg.type in GAMES
      ? Object.fromEntries(Object.entries(drawingsOf(this.game, this.drawn)).map(([id, q]) => [this.personOf[id] ?? id, q])) : undefined;
    return {
      now: Date.now() - this.t0, tv: await this.screen(this.hostPage), phone: who === HOST ? null : await this.screen(this.phones[who]),
      offers: this.offers(who), ...situation(this.game, person && this.idOf[person], person, this.physics, (id) => this.personOf[id] ?? id),
      ...(drawings ? { drawings } : {}),
    };
  }

  // --- pressing -----------------------------------------------------------------------------------

  private async click(page: Page, selector: string, what: string) {
    const loc = page.locator(selector).filter({ visible: true }).first();
    try {
      await loc.waitFor({ state: 'visible', timeout: 8000 });
      await loc.click({ timeout: 8000 });
      return true;
    } catch {
      this.issue(`no visible "${what}" (${selector})`);
      return false;
    }
  }

  async act(who: Who, offer: Offer): Promise<{ ok: boolean; error?: string }> {
    await this.refresh();
    const me = this.real(who);
    const o = this.in(offer);
    if (!isOffered(this.game, me, o, Date.now())) return { ok: false, error: `${offer.type} isn't on that screen now` };
    const page = who === HOST ? this.hostPage : this.phones[who];
    // the host covering for a player: the console's Manual Override, folded away until opened
    if (who === HOST && ['keep', 'fight', 'yield', 'skip_reprisal', 'path', 'nobids_defender'].includes(o.type)) return this.override(o);
    const tid = (id: string) => `[data-testid="${id}"]`;
    const t = this.game.currentTurn;
    let ok = true;
    switch (o.type) {
      case 'bid': ok = await this.click(page, tid(`bid-${o.amount}`), `bid ${o.amount}`) && await this.click(page, tid('seal-bid'), 'Seal bid'); break;
      case 'keep': ok = await this.click(page, tid('keep-gift'), 'Keep My Gift'); break;
      case 'gambit': ok = await this.click(page, tid('gambit'), "Grinch's Gambit") && await this.click(page, tid(`gambit-target-${o.giftId}`), 'Gambit target'); break;
      case 'fight': case 'yield': ok = await this.click(page, tid(o.type), o.type); break;
      case 'pick_duel': ok = await this.click(page, tid(`duel-game-${o.index}`), o.name); break;
      case 'spin': ok = await this.click(page, tid('duel-game-spin'), 'Let the wheel pick'); break;
      case 'call_duel': ok = await this.click(page, tid(`vote-${o.pick}`), 'call the duel'); break;
      case 'reprisal': ok = await this.click(page, tid(`reprisal-target-${o.giftId}`), 'Reprisal target'); break;
      case 'skip_reprisal': ok = await this.click(page, tid('skip-reprisal'), 'Skip Reprisal'); break;
      case 'path': ok = await this.click(page, tid(o.path === 'steal' ? 'path-a' : 'path-b'), `Path ${o.path}`); break;
      case 'steal': ok = await this.click(page, tid(`steal-target-${o.giftId}`), 'steal target'); break;
      case 'cancel_steal': ok = await this.click(page, tid('cancel-steal'), 'Cancel steal'); break;
      case 'nobids_defender': ok = await this.click(page, tid(`nobids-defender-${o.playerId}`), 'defender'); break;
      case 'whack':                                                    // hands keep going; the protocol doesn't wait
        this.whacking.add(who);
        this.busy.push(this.whack(who, page).finally(() => this.whacking.delete(who)));
        break;
      case 'sketch': ok = await this.sketch(who, page); break;
      case 'vote_sketch': ok = await this.click(page, `#party-card [data-vote="${o.pick}"]`, 'vote for a drawing'); break;
      case 'start': ok = await this.click(page, '#btn-start-game:not([disabled])', 'Start Game'); break;
      case 'draw': {
        const name = o.giftName || GIFTS[Object.keys(this.game.gifts || {}).length % GIFTS.length];
        page.once('dialog', (d) => d.accept(name).catch(() => {}));
        ok = await this.click(page, tid('draw'), 'Draw from the sack');
        break;
      }
      case 'reveal': ok = await this.click(page, tid('reveal-bids'), 'Reveal All Bids'); break;
      case 'start_duel': ok = await this.click(page, tid('start-duel'), 'Start Duel'); break;
      case 'start_yield_duel': ok = await this.click(page, tid('start-yield-duel'), 'Start Yield Duel'); break;
      case 'host_spin': ok = await this.click(page, tid('host-spin'), 'Spin'); break;
      case 'close_voting': ok = await this.click(page, tid('close-voting'), 'Close voting'); break;
      case 'start_duel_app': ok = await this.click(page, tid('start-duel-app'), 'Start the duel game'); break;
      case 'duel_winner': {
        const modal = await page.locator('#duel-modal-overlay.active').count();
        ok = await this.click(page, tid(modal && !t?.isTiebreak ? `modal-winner-${o.id}` : `${t?.isTiebreak ? 'tiebreak' : 'duel'}-winner-${o.id}`), 'X wins');
        if (ok) this.busy.push(sleep(3000).then(() => page.locator('#duel-verdict').click({ timeout: 2000 }).catch(() => {}))); // the host taps the verdict away
        break;
      }
      case 'skip_party': ok = await this.click(page, tid('skip-party-phase'), 'Skip ahead'); break;
      case 'endgame': ok = await this.click(page, tid('endgame'), 'Endgame'); break;
      case 'reveal_auction': ok = await this.click(page, tid('reveal-auction'), 'Reveal Bids'); break;
      case 'auction_duel': ok = await this.click(page, tid('auction-duel'), 'Duel Highest Bidder'); break;
      case 'end': page.once('dialog', (d) => d.accept().catch(() => {})); ok = await this.click(page, tid('end-game'), 'End Game for All'); break;
    }
    if (!ok) return { ok: false, error: `couldn't press ${offer.type} on the page` };
    (this.did[who] ||= {})[offer.type] = (this.did[who][offer.type] || 0) + 1;
    this.note(`${who === HOST ? 'The host' : this.o.people.find((p) => p.id === who)?.name || who} ${describeOffer(o, this.game)}`, who, 'act', o.type === 'bid');
    await sleep(150);
    await this.refresh();
    return { ok: true };
  }

  private async override(o: Offer): Promise<{ ok: boolean; error?: string }> {
    const page = this.hostPage;
    await page.locator('summary:has-text("Manual Override")').first().click({ timeout: 5000 }).catch(() => {});
    const id = o.type === 'skip_reprisal' ? 'override-skip-reprisal' : o.type === 'path' ? 'override-auction' : o.type === 'nobids_defender' ? 'override-defender' : `override-${o.type}`;
    let ok = await this.click(page, `[data-testid="${id}"]`, `override ${o.type}`);
    if (ok && o.type === 'nobids_defender') ok = await this.click(page, `[data-testid="override-defender-${o.playerId}"]`, 'the defender');
    if (!ok) return { ok: false, error: `couldn't press the override for ${o.type}` };
    (this.did[HOST] ||= {})[o.type] = (this.did[HOST][o.type] || 0) + 1;
    this.note(`The host covers: ${describeOffer(o, this.game)}`, HOST, 'act');
    await sleep(150);
    await this.refresh();
    return { ok: true };
  }

  /** Whack-an-Elf on the real phone until time's up: an elf can be hit once it has been up for
   *  this person's reaction time, and now and then they tap the Grinch by mistake. */
  private async whack(who: string, page: Page) {
    const mg = this.game.minigame as PartyGame;
    const seen: Record<string, number> = {};
    while (Date.now() < mg.phaseEndsAt - 150) {
      const now = Date.now();
      const up: string[] = await page.locator('#party-card .whack-cell.up').evaluateAll((els) => els.map((e) => `${(e as HTMLElement).dataset.cell}|${(e as HTMLElement).dataset.kind}`)).catch(() => []);
      for (const k of Object.keys(seen)) if (!up.includes(k)) delete seen[k];
      for (const k of up) seen[k] ??= now + this.physics.reaction(who);
      const ready = up.filter((k) => seen[k] <= now);
      const good = ready.filter((k) => !k.endsWith('|grinch'));
      const bad = ready.filter((k) => k.endsWith('|grinch'));
      const target = bad.length && this.physics.slips(who) ? bad[0] : good[0];
      if (target) {
        delete seen[target];
        await page.locator(`#party-card .whack-cell[data-cell="${target.split('|')[0]}"]`).dispatchEvent('pointerdown').catch(() => {});
      }
      await sleep(50);
    }
  }

  /** A drawing on the real canvas: better artists take more care (more strokes). */
  private async sketch(who: string, page: Page) {
    const q = this.physics.sketchQuality(who);
    const pad = page.locator('#party-card canvas.pg-pad');
    if (!(await pad.waitFor({ state: 'visible', timeout: 8000 }).then(() => true).catch(() => false))) { this.issue(`${who}: no drawing pad`); return false; }
    const b = (await pad.boundingBox())!;
    for (let s = 0; s < 2 + Math.round(q * 6); s++) {
      const x0 = b.x + b.width * (0.2 + 0.6 * ((s * 37) % 10) / 10), y0 = b.y + b.height * (0.2 + 0.6 * ((s * 53) % 10) / 10);
      await page.mouse.move(x0, y0);
      await page.mouse.down();
      for (let i = 1; i <= 6; i++) await page.mouse.move(x0 + Math.sin(i + s) * b.width * 0.15, y0 + Math.cos(i * s) * b.height * 0.15, { steps: 2 });
      await page.mouse.up();
    }
    const ok = await this.click(page, '#party-card [data-act="done"]', 'Done drawing');
    if (ok) this.drawn[this.idOf[who]] = q;
    return ok;
  }

  // --- time ---------------------------------------------------------------------------------------

  /** Let `ms` of real time pass; the host page runs its own timers, the scripted host its job. */
  async advance(ms: number): Promise<TableEvent[]> {
    const from = this.log.length;
    const until = Date.now() + ms;
    while (Date.now() < until) {
      await this.refresh();
      if (this.host && this.game.phase !== 'ended') {
        const due = this.host.due(this.game, Date.now(), () => this.physics.duelOutcome(this.game, (id) => this.personOf[id] ?? id));
        if (due && due.at <= Date.now()) {
          const r = await this.act(HOST, this.out(due.offer));
          if (!r.ok) this.host.waits(this.game, Date.now());
          continue;
        }
      }
      await sleep(Math.min(250, Math.max(0, until - Date.now())));
    }
    return this.log.slice(from);
  }

  get ended() { return this.game.phase === 'ended'; }

  result() {
    const seats = this.o.people.map((p) => {
      const pl = this.game.players?.[this.idOf[p.id]];
      const gift = pl?.giftId ? this.game.gifts?.[pl.giftId] : undefined;
      return { id: p.id, name: p.name, chips: pl?.chips ?? 0, gift: gift?.name ?? null, value: gift ? this.physics.value(p.id, gift.name) : 0,
        partyPoints: pl?.partyPoints || 0, did: this.did[p.id] || {} };
    });
    return { ended: this.ended, minutes: Math.round((Date.now() - this.t0) / 600) / 100, seats, host: this.did[HOST] || {}, room: this.room, issues: this.issues };
  }

  /** The pages as pictures (the TV and each phone, JPEG base64), for whoever watches; a page that
   *  can't be captured right now is left out. */
  async frames(): Promise<Record<string, string>> {
    const shot = async (page: Page | undefined, fullPage = false) => {
      try { return page ? (await page.screenshot({ type: 'jpeg', quality: 55, timeout: 3000, fullPage })).toString('base64') : null; } catch { return null; }
    };
    const out: Record<string, string> = {};
    const all = await Promise.all([['tv', this.hostPage] as const, ...this.o.people.map((p) => [p.id, this.phones[p.id]] as const)]
      .map(async ([k, page]) => [k, await shot(page, k === 'tv')] as const));   // the TV whole: its boards below the fold too
    for (const [k, b] of all) if (b) out[k] = b;
    return out;
  }

  async close() {
    await Promise.allSettled(this.busy);
    await this.browser?.close().catch(() => {});
  }
}
