#!/usr/bin/env node
// Agent play-testing harness — real host + phone pages, played by persona agents.
//
//   npm run agents:watch                       # one paced game in tiled windows + live commentary
//   npm run agents:sweep                       # headless: every preset × N games → balance report
//   node agents/play.js --games 5 --set max-payout=3,loser-dividend=everyone
//   node agents/play.js --lineup brawler,miser,miser,sniper --seed 7 --watch
//   npm run agents:join ABCD                   # YOU host room ABCD on React (+ play a phone); agents fill the other seats
//
// Options:
//   --watch            slowed down, streamed live to one window with commentary
//   --sweep            run all PRESETS (see presets.js) instead of one configuration
//   --games N          games per configuration (default 1 watch / 8 otherwise)
//   --lineup a,b,c     persona per seat (default: a random mix of the cast, new each game)
//   --players N        table size for the random mix (default 6, max 12)
//   --personas         list the cast and exit
//   --set k=v,...      setup-screen settings (ids without "setting-"), e.g. max-bid=5
//   --presets a,b      with --sweep: only these presets
//   --parallel P       games at once when headless (default 3)
//   --seed S           base seed (game i uses S+i)
//   --pace MS          delay between actions in watch mode (default 900)
//   --out DIR          report directory (default agents/reports/<timestamp>)
//   --verbose          print every action and agent thought
//   --join ROOM        join a game a person is hosting (lobby only): the agents play their own
//                      seats and leave the host page and your phone to you (--players = agent count, default 4)
//   --phones URL       app the agents' phones load (default $PLAYER_BASE / $BASE_URL / the dev server :5173)
//
// Needs the local emulators + page server; starts them if they aren't running.

const fs = require('fs');
const path = require('path');
const { spawn, execSync } = require('child_process');
const { chromium } = require('@playwright/test');
let Arena; // required in main(), once --phones is known
const { PERSONAS, rng } = require('./personas');
const { PRESETS } = require('./presets');
const { summarize } = require('./report');

function parseArgs(argv) {
  const o = { set: {} };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--watch') o.watch = true;
    else if (a === '--verbose') o.verbose = true;
    else if (a === '--sweep') o.sweep = true;
    else if (a === '--games') o.games = Number(next());
    else if (a === '--lineup') o.lineup = next().split(',');
    else if (a === '--players') o.players = Number(next());
    else if (a === '--personas') { for (const [k, p] of Object.entries(PERSONAS)) console.log(`${p.emoji} ${k.padEnd(12)} ${p.label} — ${p.blurb}`); process.exit(0); }
    else if (a === '--presets') o.presets = next().split(',');
    else if (a === '--parallel') o.parallel = Number(next());
    else if (a === '--seed') o.seed = Number(next());
    else if (a === '--pace') o.pace = Number(next());
    else if (a === '--out') o.out = next();
    else if (a === '--join') o.join = next().toUpperCase();
    else if (a === '--phones') o.phones = next().replace(/\/$/, '');
    else if (a === '--set') for (const kv of next().split(',')) { const [k, v] = kv.split('='); o.set[k] = v; }
    else if (a === '-h' || a === '--help') { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(1, 27).join('\n')); process.exit(0); }
    // Bare values, so `npm run agents:join ABCD` works in PowerShell too (it drops the `--`
    // before npm's args, and npm then swallows any --flags meant for this script)
    else if (/^https?:\/\//.test(a)) o.phones = a.replace(/\/$/, '');
    else if (/^[a-z]{4}$/i.test(a)) o.join = a.toUpperCase();
    else throw new Error(`Unknown option ${a}`);
  }
  return o;
}

// --- servers ------------------------------------------------------------------

const PAGES = () => (process.env.PLAYER_BASE || process.env.BASE_URL || 'http://localhost:5173').replace(/\/$/, '');
const PAGE_URL = () => `${PAGES()}/player.html`;
const DB_URL = 'http://127.0.0.1:9000/.json?ns=jingle-brawl-e8308-default-rtdb';
const AUTH_URL = 'http://127.0.0.1:9099/emulator/v1/projects/jingle-brawl-e8308/config';
// Any answer means it is running (the database answers 401 at its root: the rules forbid reading it)
const up = async (url) => { try { return (await fetch(url)).status < 500; } catch { return false; } };

async function ensureServers() {
  const started = [];
  const root = path.join(__dirname, '..');
  if (!(await up(PAGE_URL()))) {
    if (/:5173$/.test(PAGES())) {
      console.log('Starting the app (npm run dev)…');
      started.push(spawn('npx', ['vite', '--config', 'vite.config.mts'], { cwd: root, stdio: 'ignore', shell: true }));
    } else {
      throw new Error(`Nothing is serving ${PAGES()} — start the app first`);
    }
  }
  if (!(await up(DB_URL)) || !(await up(AUTH_URL))) {
    console.log('Starting the Firebase emulators (database + auth)…');
    started.push(spawn('firebase', ['emulators:start', '--only', 'database,auth', '--project', 'jingle-brawl-e8308'],
      { cwd: root, stdio: 'ignore', shell: true }));
  }
  const deadline = Date.now() + 90000;
  for (const url of [PAGE_URL(), DB_URL, AUTH_URL]) {
    while (!(await up(url))) {
      if (Date.now() > deadline) throw new Error(`Not reachable: ${url} (is Java 21+ / the Firebase CLI installed?)`);
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  return () => {
    for (const p of started) {
      try {
        if (process.platform === 'win32') execSync(`taskkill /pid ${p.pid} /T /F`, { stdio: 'ignore' });
        else p.kill('SIGTERM');
      } catch { /* already gone */ }
    }
  };
}

// --- watch mode: one "stage" window streaming every page live -------------------
// The game's pages run headless at their real sizes (host 1440×900, phones 390×844);
// the stage shows a live screencast of each next to the agents' running commentary.
// One window fits any screen, and nothing on the stage is clickable by accident.

async function openStage(seats, { host = true } = {}) {
  const browser = await chromium.launch({ headless: false, args: ['--start-maximized'] });
  const ctx = await browser.newContext({ viewport: null });
  const page = await ctx.newPage();
  const tiles = seats.map((s, i) => `
    <figure class="phone"><img id="phone-${i}" alt=""><figcaption>${PERSONAS[s.persona].emoji} <b>${s.name}</b> · ${PERSONAS[s.persona].label}</figcaption></figure>`).join('');
  const legend = seats.map((s) => `<li>${PERSONAS[s.persona].emoji} <b>${s.name}</b> — ${PERSONAS[s.persona].blurb}</li>`).join('');
  await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><title>Agent table</title>
  <style>
    :root { --bg:#0b1220; --card:#162033; --line:#22304a; --text:#e2e8f0; --muted:#94a3b8; --gold:#f59e0b; --red:#f87171; --green:#4ade80; --blue:#7dd3fc; }
    * { box-sizing:border-box; }
    body { margin:0; height:100vh; background:var(--bg); color:var(--text); font:13px/1.4 system-ui,sans-serif;
      display:grid; grid-template-columns: minmax(0, 1.75fr) minmax(0, 1fr); grid-template-rows: auto minmax(0, 1fr); gap:10px; padding:10px; }
    #host { grid-column:1; grid-row:1; margin:0; }
    #host img { width:100%; aspect-ratio: 1440 / 900; display:block; border-radius:10px; background:#000; }
    #feed-wrap { grid-column:1; grid-row:2; background:var(--card); border-radius:10px; display:flex; flex-direction:column; min-height:0; }
    #feed-wrap h1 { margin:0; padding:8px 12px; font-size:14px; color:var(--gold); border-bottom:1px solid var(--line); }
    #feed { flex:1; overflow-y:auto; padding:6px 12px; }
    .l { padding:2px 0; } .agent { color:var(--text); } .gift { color:var(--gold); font-weight:600; margin-top:6px; }
    .duel { color:var(--blue); } .warn { color:var(--red); font-weight:600; } .result { color:var(--green); } .info { color:var(--muted); }
    #side { grid-column:2; grid-row:1 / span 2; display:flex; flex-direction:column; gap:8px; min-height:0; }
    #phones { flex:1; min-height:0; display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); grid-auto-rows: minmax(0, 1fr); gap:8px; }
    .phone { margin:0; min-width:0; min-height:0; display:flex; flex-direction:column; }
    .phone img { flex:1; min-height:0; width:100%; object-fit:contain; display:block; }
    figcaption { padding:3px 2px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    figcaption b { color:var(--text); }
    ul { margin:0; padding:8px 12px 8px 26px; background:var(--card); border-radius:10px; color:var(--muted); font-size:12px; }
    ${host ? '' : '#host { display:none; } #feed-wrap { grid-row: 1 / span 2; }'}
  </style></head><body>
  <figure id="host"><img id="host-img" alt=""></figure>
  <section id="feed-wrap"><h1>🎄 Agent commentary</h1><div id="feed"></div></section>
  <aside id="side"><div id="phones">${tiles}</div><ul>${legend}</ul></aside>
  <script>
    window.frame = (id, data) => { const el = document.getElementById(id); if (el) el.src = 'data:image/jpeg;base64,' + data; };
    window.feed = (t, k) => {
      const d = document.createElement('div'); d.className = 'l ' + (k || 'info'); d.textContent = t;
      const f = document.getElementById('feed'); f.appendChild(d); f.scrollTop = f.scrollHeight;
    };
  </script></body></html>`);
  return { browser, page };
}

/** Stream a (headless) page onto the stage. Drops frames while the stage is busy. */
function streamer(stage) {
  return async (page, role, i) => {
    const cdp = await page.context().newCDPSession(page);
    const id = role === 'host' ? 'host-img' : `phone-${i}`;
    let busy = false;
    cdp.on('Page.screencastFrame', ({ data, sessionId }) => {
      cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
      if (busy) return;
      busy = true;
      stage.evaluate(([el, d]) => window.frame(el, d), [id, data]).catch(() => {}).finally(() => { busy = false; });
    });
    const size = role === 'host' ? { maxWidth: 1440, maxHeight: 900 } : { maxWidth: 390, maxHeight: 844 };
    await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 70, everyNthFrame: 1, ...size });
  };
}

// --- running games -------------------------------------------------------------

const NAMES = ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy', 'Mallory', 'Niaj'];

function seatsFor(config, gameIndex, seed, taken = []) {
  let personas;
  if (config.lineup) {
    // Fixed lineup: rotate through the seats so seat/draw order isn't confounded with persona
    personas = config.lineup.map((_, i) => config.lineup[(i + gameIndex) % config.lineup.length]);
  } else {
    // Mix: a different random table each game (same seed → same table in every preset)
    const r = rng(seed * 7919);
    const pool = Object.keys(PERSONAS);
    personas = Array.from({ length: config.players }, (_, i) => {
      if (i % pool.length === 0) pool.sort(() => r() - 0.5);
      return pool[i % pool.length];
    });
  }
  const lower = new Set(taken.map((n) => n.toLowerCase()));
  const names = NAMES.filter((n) => !lower.has(n.toLowerCase()));
  return personas.map((persona, i) => ({ name: names[i] || `Agent ${i + 1}`, persona }));
}

async function runOne(browser, { config, gameIndex, seed, watch, verbose, pace, outDir, join, taken = [] }) {
  const seats = seatsFor(config, gameIndex, seed, taken);
  const stage = watch ? await openStage(seats, { host: !join }) : null;
  const arena = new Arena({
    browser, seats, settings: config.settings, seed, watch, pace, outDir, join,
    onPage: stage ? streamer(stage.page) : null, commentary: stage?.page,
    log: watch || verbose || join ? (t) => console.log(t) : null, verbose,
  });
  try {
    await arena.setup();
    const result = await arena.play(join ? { maxMs: 6 * 3600000 } : undefined);
    result.preset = config.name;
    result.lineup = seats;
    return result;
  } catch (e) {
    return { preset: config.name, seed, outcome: 'crashed', error: e.stack, issues: arena.issues, seats: [], duels: [], decisions: [] };
  } finally {
    if (stage) {
      console.log('\nGame over — the stage stays open for 2 minutes (close it or Ctrl+C to quit).');
      await Promise.race([new Promise((r) => setTimeout(r, 120000)), new Promise((r) => stage.page.on('close', r))]);
      await stage.browser.close().catch(() => {});
    }
    await arena.close();
  }
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  if (o.phones) process.env.PLAYER_BASE = o.phones;
  // The host page (sweeps) runs on the same app as the phones: the dev server unless told otherwise
  process.env.BASE_URL ||= process.env.PLAYER_BASE || 'http://localhost:5173';
  ({ Arena } = require('./arena')); // after PLAYER_BASE is set (e2e/helpers reads it on load)
  const lineup = o.lineup || null; // null = random mix of the whole cast per game
  for (const p of lineup || []) if (!PERSONAS[p]) throw new Error(`Unknown persona ${p}. Known: ${Object.keys(PERSONAS).join(', ')}`);
  const players = lineup ? lineup.length : (o.players ?? (o.join ? 4 : 6));
  if (players < 2 || players > 12) throw new Error('--players must be 2-12');
  const base = { ...o.set }; // ties: the house default (a party game mix)
  const configs = o.sweep
    ? PRESETS.filter((p) => !o.presets || o.presets.includes(p.name)).map((p) => ({ name: p.name, settings: { ...base, ...p.settings }, lineup, players }))
    : [{ name: Object.keys(o.set).length ? 'custom' : 'default', settings: base, lineup, players }];
  if (o.join && (o.sweep || o.games > 1)) throw new Error('--join plays one game (your game) — drop --sweep / --games');
  const games = o.join ? 1 : o.games ?? (o.watch ? 1 : 8);
  const seed0 = o.seed ?? Math.floor(Math.random() * 1e6);
  const outDir = o.out || path.join(__dirname, 'reports', new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19));
  fs.mkdirSync(outDir, { recursive: true });

  const stop = await ensureServers();
  process.on('SIGINT', () => { stop(); process.exit(130); });
  const browser = await chromium.launch(); // game pages are always headless; --watch streams them
  let taken = [];
  if (o.join) {
    const room = await fetch(`http://127.0.0.1:9000/games/${o.join}.json?ns=jingle-brawl-e8308-default-rtdb`).then((r) => r.json()).catch(() => null);
    if (!room) throw new Error(`Room ${o.join} not found on the local emulator — create it on the host page (localhost) first`);
    taken = Object.values(room.players || {}).map((x) => x.name);
  }

  const jobs = [];
  for (const config of configs) for (let i = 0; i < games; i++) jobs.push({ config, gameIndex: i, seed: seed0 + i });
  console.log(`Running ${jobs.length} game(s): ${configs.map((c) => c.name).join(', ')} · ${lineup ? `lineup ${lineup.join(', ')}` : `random ${players}-player mix of ${Object.keys(PERSONAS).length} personas`} · seed ${seed0}`);

  const results = [];
  const parallel = o.watch ? 1 : Math.max(1, o.parallel ?? 3);
  let next = 0;
  const t0 = Date.now();
  await Promise.all(Array.from({ length: parallel }, async () => {
    while (next < jobs.length) {
      const job = jobs[next++];
      const r = await runOne(browser, { ...job, watch: o.watch, verbose: o.verbose, pace: o.pace, outDir, join: o.join, taken });
      results.push(r);
      const issues = (r.issues || []).length;
      console.log(`  [${results.length}/${jobs.length}] ${r.preset} seed ${job.seed}: ${r.outcome}` +
        (r.ms ? ` in ${(r.ms / 1000).toFixed(0)}s, ${r.duels.length} duels` : '') + (issues ? `, ${issues} issue(s)` : '') +
        (r.error ? `\n${r.error}` : ''));
      fs.writeFileSync(path.join(outDir, 'results.json'), JSON.stringify(results, null, 2));
    }
  }));

  await browser.close();
  stop();
  const md = summarize(results, { lineup, players, games, seed: seed0, minutes: (Date.now() - t0) / 60000 });
  fs.writeFileSync(path.join(outDir, 'summary.md'), md);
  console.log('\n' + md);
  console.log(`\nReport: ${path.join(outDir, 'summary.md')}  (raw: results.json)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
