// Party mini-games: tiebreakers the WHOLE table plays — Sketch-Off and Whack-an-Elf.
//
// One ranking settles everything:
//   • the best-placed tied bidder wins the tiebreak (the best two, if the Opener yielded)
//   • everyone scores party points by placing (3 / 2 / 1), and — when the host allows it —
//     the best finisher who WASN'T in the tie wins an Elf Bonus chip from the North Pole Fund.
//
// Shared by host.html, player.html and Node unit tests (party-games.test.js): the pure
// functions come first; the browser screens (phone + TV) are only defined in a browser.
//
// Data lives at games/{room}/minigame:
//   { type: 'sketch_off'|'whack_elf', phase: 'intro'|'playing'|'voting'|'results',
//     participants, tiedBidders, minigameGoal: 'pick_one'|'pick_two', seed, prompt?,
//     phaseEndsAt, playStartAt, scores/{pid}: {score, final}, submissions/{pid}: dataURL,
//     votes/{voterId}: pid, results: {ranking, scoreOf, points, bonus, winners} }
(function (root) {
  const GAMES = {
    sketch_off: {
      label: 'Sketch-Off', emoji: '✏️', introMs: 7000, playMs: 60000, voteMs: 30000,
      rules: 'Everyone draws the same prompt. Then everyone votes for their favourite (not your own!).',
    },
    whack_elf: {
      label: 'Whack-an-Elf', emoji: '🔨', introMs: 7000, playMs: 20000,
      rules: 'Tap the elves as they pop up. The faster you tap, the more it scores (up to 3). Golden elves score double. Tap the Grinch and lose 3!',
    },
  };
  const RESULTS_MS = 9000; // how long the results stay up before the duel starts
  const PLACE_POINTS = [3, 2, 1];

  const SKETCH_PROMPTS = [
    'A reindeer on a skateboard', 'Santa stuck in a chimney', 'A snowman at the beach',
    'An elf lifting weights', 'The Grinch doing yoga', 'A penguin opening a present',
    'A gingerbread house on fire', 'Rudolph with a cold', 'A Christmas tree in a tiny car',
    'Mrs Claus riding a motorbike', 'A polar bear ice skating', 'A sleigh with a flat tyre',
    'A cat attacking the wrapping paper', 'Santa on a treadmill', 'A reindeer at the dentist',
    'An elf asleep on the job', 'A snowman with a sunburn', 'The ugliest Christmas sweater ever',
    'A present that is obviously a bike', 'Grandma winning a snowball fight', 'A turkey hiding from the cook',
    'A gift wrapped by a toddler', 'A Christmas tree falling over', 'A penguin delivering presents',
    'A sleigh stuck in traffic', 'A dog wearing antlers and hating it', 'Mistletoe in an awkward place',
    'A stocking that is far too full', 'The Grinch at a gift exchange', 'A reindeer singing karaoke',
    'A fruitcake used as a doorstop', 'A gingerbread man running late', 'Carolers nobody asked for',
    'A candle that smells like socks', 'An elf on a coffee break', 'Santa squeezing into his suit',
  ];

  /** Seeded PRNG (mulberry32): every phone generates the same Whack-an-Elf sequence. */
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Points for a tap `frac` of the way through a pop-up (0 = just appeared): speed pays. */
  function whackPoints(kind, frac) {
    if (kind === 'grinch') return -3;
    const speed = frac < 1 / 3 ? 3 : frac < 2 / 3 ? 2 : 1;
    return kind === 'gold' ? speed * 2 : speed;
  }

  /** Pop-ups for one Whack-an-Elf game: [{ at, cell (0-8), kind, upMs }], at = ms from start. */
  function whackSequence(seed, playMs = GAMES.whack_elf.playMs) {
    const r = rng(seed);
    const items = [];
    const busyUntil = Array(9).fill(0);
    for (let at = 600; at < playMs - 700; at += 260 + r() * 260) {
      const progress = at / playMs;
      const upMs = Math.round(1000 - 450 * progress); // speeds up as the game goes on
      const free = busyUntil.map((t, cell) => (t <= at ? cell : -1)).filter((c) => c >= 0);
      if (!free.length) continue;
      const cell = free[Math.floor(r() * free.length)];
      const roll = r();
      const kind = roll < 0.1 ? 'gold' : roll < 0.26 ? 'grinch' : 'elf';
      busyUntil[cell] = at + upMs + 120;
      items.push({ at: Math.round(at), cell, kind, upMs });
    }
    return items;
  }

  /** Sketch-Off score = votes received (no self-votes, only for real drawings). */
  function sketchScores(votes, submissions) {
    const scoreOf = {};
    for (const pid of Object.keys(submissions || {})) scoreOf[pid] = 0;
    for (const [voter, target] of Object.entries(votes || {})) {
      if (voter !== target && target in scoreOf) scoreOf[target]++;
    }
    return scoreOf;
  }

  /** Whack-an-Elf score = the latest score each phone reported. */
  function whackScores(scores) {
    const scoreOf = {};
    for (const [pid, s] of Object.entries(scores || {})) scoreOf[pid] = Number(s?.score) || 0;
    return scoreOf;
  }

  /** Everyone, best first. Equal scores are ordered by a seeded coin, not by seat. */
  function rank(participants, scoreOf, seed) {
    const r = rng(seed ^ 0x5eed);
    const coin = Object.fromEntries(participants.map((id) => [id, r()]));
    return [...participants].sort((a, b) =>
      (scoreOf[b] ?? -Infinity) - (scoreOf[a] ?? -Infinity) || coin[a] - coin[b]);
  }

  /** Tiebreak winners: tied bidders in ranking order — 1 to duel the Opener, 2 after a yield. */
  function tieWinners(ranking, tiedBidders, goal) {
    const placed = ranking.filter((id) => tiedBidders.includes(id));
    const missing = tiedBidders.filter((id) => !placed.includes(id)); // e.g. left the game
    return [...placed, ...missing].slice(0, goal === 'pick_two' ? 2 : 1);
  }

  /**
   * Party points by place (3/2/1 for anyone who actually scored) and the Elf Bonus chip
   * for the best finisher outside the tie. prize: 'points_chips' | 'points' | 'none'.
   */
  function rewards(ranking, scoreOf, tiedBidders, prize = 'points_chips') {
    const points = {};
    let bonus = null;
    if (prize === 'none') return { points, bonus };
    const scorers = ranking.filter((id) => (scoreOf[id] ?? 0) > 0);
    scorers.slice(0, PLACE_POINTS.length).forEach((id, i) => { points[id] = PLACE_POINTS[i]; });
    if (prize === 'points_chips') bonus = scorers.find((id) => !tiedBidders.includes(id)) || null;
    return { points, bonus };
  }

  /** Everything the host writes to minigame.results. */
  function results(minigame) {
    const participants = minigame.participants || [];
    const scoreOf = minigame.type === 'sketch_off'
      ? sketchScores(minigame.votes, minigame.submissions)
      : whackScores(minigame.scores);
    const ranking = rank(participants, scoreOf, minigame.seed || 1);
    const tied = minigame.tiedBidders || [];
    const { points, bonus } = rewards(ranking, scoreOf, tied, minigame.prize);
    return { ranking, scoreOf, points, bonus, winners: tieWinners(ranking, tied, minigame.minigameGoal) };
  }

  const METHODS = Object.keys(GAMES); // tiebreakerMethod values that are party games
  const api = { GAMES, METHODS, SKETCH_PROMPTS, RESULTS_MS, whackPoints, rng, whackSequence, sketchScores, whackScores, rank, tieWinners, rewards, results };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.PartyGames = api;
  if (typeof document === 'undefined') return;

  // ===================================================================================
  // Browser screens
  // ===================================================================================

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const GRINCH_SVG = `<svg viewBox="0 0 120 120" aria-hidden="true">
    <path d="M24 56 C24 22 96 22 96 56 L92 84 C88 104 32 104 28 84 Z" fill="#6cbf3c"/>
    <path d="M30 50 L50 56 M90 50 L70 56" stroke="#2f5a17" stroke-width="5" stroke-linecap="round"/>
    <circle cx="48" cy="64" r="6" fill="#f4e04d"/><circle cx="72" cy="64" r="6" fill="#f4e04d"/>
    <circle cx="49" cy="65" r="2.5" fill="#c0392b"/><circle cx="71" cy="65" r="2.5" fill="#c0392b"/>
    <path d="M40 82 C52 92 70 92 82 78" stroke="#2f5a17" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M24 40 C30 10 70 4 98 18 C84 22 80 30 82 40 Z" fill="#d7263d"/><circle cx="100" cy="18" r="7" fill="#fff"/>
  </svg>`;
  const elfSvg = () => root.Mascots?.svg?.elf || '<svg viewBox="0 0 120 120"><circle cx="60" cy="70" r="30" fill="#1f9d55"/></svg>';

  const css = `
    .pg { text-align: center; }
    .pg h3 { margin: 0 0 4px; font-size: 22px; }
    .pg .pg-sub { color: var(--muted, #6b7280); font-size: 14px; margin: 0 0 10px; }
    .pg .pg-timer { font: 800 28px/1 system-ui, sans-serif; font-variant-numeric: tabular-nums; margin: 6px 0 10px; }
    .pg .pg-prompt { font-size: 20px; font-weight: 800; margin: 6px 0 10px; }
    .pg .pg-btns { display: flex; gap: 8px; justify-content: center; margin-top: 10px; flex-wrap: wrap; }
    .pg button.pg-btn { border: 0; border-radius: 12px; padding: 10px 16px; font-weight: 700; cursor: pointer; background: var(--bg, #eef1f6); color: inherit; }
    .pg button.pg-btn.primary { background: var(--accent, #ff3b5c); color: #fff; }
    .pg canvas.pg-pad { width: 100%; max-width: 340px; aspect-ratio: 1; background: #fff; border-radius: 14px; touch-action: none; box-shadow: inset 0 0 0 2px var(--line, #e5e7eb); }
    .pg .pg-colors { display: flex; gap: 8px; justify-content: center; margin-top: 8px; }
    .pg .pg-colors button { width: 30px; height: 30px; border-radius: 50%; border: 3px solid transparent; cursor: pointer; }
    .pg .pg-colors button.on { border-color: var(--ink, #111); }
    .pg .whack-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; max-width: 340px; margin: 0 auto; }
    .pg .whack-cell { aspect-ratio: 1; border-radius: 50%; border: 0; padding: 0; cursor: pointer; position: relative; overflow: hidden;
      background: radial-gradient(circle at 50% 75%, #3b2a1e 0 38%, #6b4a33 39% 60%, #d9e7c4 61%); }
    .pg .whack-cell .pop { position: absolute; inset: 8% 12% 18%; transform: translateY(130%); transition: transform .12s ease-out; pointer-events: none; }
    .pg .whack-cell.up .pop { transform: translateY(0); }
    .pg .whack-cell.gold .pop { filter: sepia(1) saturate(4) hue-rotate(5deg) brightness(1.1); }
    .pg .whack-cell.hit .pop { transform: translateY(130%) scale(.8); transition-duration: .2s; }
    .pg .whack-cell svg { width: 100%; height: 100%; }
    .pg .whack-score { font: 800 26px/1 system-ui, sans-serif; margin-bottom: 8px; }
    .pg .whack-float { position: absolute; left: 50%; top: 20%; transform: translateX(-50%); font-weight: 900; pointer-events: none; animation: pgFloat .6s ease-out forwards; }
    @keyframes pgFloat { to { transform: translate(-50%, -30px); opacity: 0; } }
    .pg .pg-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
    .pg .pg-gallery figure { margin: 0; background: #fff; border-radius: 12px; padding: 6px; box-shadow: 0 2px 10px rgba(15,20,35,.08); position: relative; }
    .pg .pg-gallery img { width: 100%; aspect-ratio: 1; border-radius: 8px; display: block; }
    .pg .pg-gallery figcaption { font-weight: 700; padding-top: 4px; font-size: 13px; }
    .pg .pg-gallery button { all: unset; display: block; cursor: pointer; }
    .pg .pg-gallery figure.mine { opacity: .5; }
    .pg .pg-gallery figure.voted { box-shadow: 0 0 0 3px var(--accent, #ff3b5c); }
    .pg .pg-rank { list-style: none; padding: 0; margin: 8px auto; max-width: 520px; text-align: left; }
    .pg .pg-rank li { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 12px; margin-bottom: 6px; background: var(--bg, #eef1f6); }
    .pg .pg-rank li.tied { box-shadow: inset 4px 0 0 var(--accent, #ff3b5c); }
    .pg .pg-rank li.win { background: color-mix(in srgb, var(--accent, #ff3b5c) 14%, white); font-weight: 800; }
    .pg .pg-rank .pos { width: 26px; font-weight: 800; color: var(--muted, #6b7280); }
    .pg .pg-rank .nm { flex: 1; }
    .pg .pg-rank .tag { font-size: 11px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; background: #fff; }
    .pg .pg-bars { max-width: 520px; margin: 0 auto; text-align: left; }
    .pg .pg-bar { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
    .pg .pg-bar .nm { width: 110px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pg .pg-bar .track { flex: 1; height: 18px; border-radius: 9px; background: var(--bg, #eef1f6); overflow: hidden; }
    .pg .pg-bar .fill { height: 100%; background: var(--accent, #ff3b5c); transition: width .3s; }
    .pg .pg-bar .v { width: 36px; text-align: right; font-weight: 800; font-variant-numeric: tabular-nums; }
  `;
  function mountCss() {
    if (document.getElementById('party-games-style')) return;
    const s = document.createElement('style');
    s.id = 'party-games-style';
    s.textContent = css;
    document.head.appendChild(s);
  }

  // Server-synced clock: phones and host agree on when a phase ends.
  let offset = 0;
  const now = () => Date.now() + offset;
  function syncClock(db) {
    db.ref('.info/serverTimeOffset').on('value', (s) => { offset = Number(s.val()) || 0; });
  }
  const secsLeft = (mg) => Math.max(0, Math.ceil(((mg.phaseEndsAt || 0) - now()) / 1000));

  function timer(el, mg) {
    const tick = () => { if (el.isConnected) { el.textContent = `${secsLeft(mg)}s`; setTimeout(tick, 250); } };
    tick();
  }

  // ---------- phone --------------------------------------------------------------------
  // renderPhone(container, minigame, ctx) is called on EVERY minigame update; it only
  // rebuilds the screen when the game or phase changes, so a drawing or a running
  // Whack-an-Elf survives other players' updates.
  // ctx = { me, players, write(path, value) }  (path is under games/{room}/minigame)

  let screen = { key: null, cleanup: null };

  function resetPhone() {
    screen.cleanup?.();
    screen = { key: null, cleanup: null };
  }

  function renderPhone(el, mg, ctx) {
    mountCss();
    const key = `${mg.type}|${mg.seed}|${mg.phase}`;
    if (screen.key === key) return refreshPhone(el, mg, ctx);
    screen.cleanup?.();
    screen = { key, cleanup: null };
    const g = GAMES[mg.type];
    const tied = (mg.tiedBidders || []).includes(ctx.me);
    const stake = tied ? '<b>You\'re in the tie!</b> Beat the other tied bidders to win it.' : 'Not in the tie? Play for party points' + (mg.prize === 'points_chips' ? ' and the Elf Bonus chip!' : '!');
    el.classList.add('pg');
    if (mg.phase === 'intro') {
      el.innerHTML = `<h3>${g.emoji} ${g.label}!</h3><p class="pg-sub">${g.rules}</p>
        ${mg.prompt ? `<div class="pg-prompt">“${esc(mg.prompt)}”</div>` : ''}
        <p class="pg-sub">${stake}</p><div class="pg-timer"></div><p class="pg-sub">Get ready…</p>`;
      timer(el.querySelector('.pg-timer'), mg);
    } else if (mg.phase === 'playing' && mg.type === 'sketch_off') {
      sketchPad(el, mg, ctx);
    } else if (mg.phase === 'playing' && mg.type === 'whack_elf') {
      whackBoard(el, mg, ctx);
    } else if (mg.phase === 'voting') {
      el.innerHTML = `<h3>🗳️ Vote for the best drawing</h3><p class="pg-sub">“${esc(mg.prompt)}” (not your own!)</p>
        <div class="pg-timer"></div><div class="pg-gallery"></div>`;
      timer(el.querySelector('.pg-timer'), mg);
      refreshPhone(el, mg, ctx);
    } else if (mg.phase === 'results') {
      refreshPhone(el, mg, ctx);
    }
  }

  function refreshPhone(el, mg, ctx) {
    if (mg.phase === 'voting') {
      const myVote = mg.votes?.[ctx.me];
      const order = rank(Object.keys(mg.submissions || {}), {}, mg.seed); // stable shuffle
      const gallery = el.querySelector('.pg-gallery');
      if (!gallery) return;
      gallery.innerHTML = order.map((pid, i) => `
        <figure class="${pid === ctx.me ? 'mine' : ''} ${myVote === pid ? 'voted' : ''}">
          ${pid === ctx.me ? `<img src="${mg.submissions[pid]}" alt="your drawing"><figcaption>Yours</figcaption>`
            : `<button data-vote="${esc(pid)}"><img src="${mg.submissions[pid]}" alt="drawing ${i + 1}"><figcaption>#${i + 1}${myVote === pid ? ' ✓' : ''}</figcaption></button>`}
        </figure>`).join('') || '<p class="pg-sub">Nobody finished a drawing!</p>';
      gallery.querySelectorAll('[data-vote]').forEach((b) => b.addEventListener('click', () => ctx.write(`votes/${ctx.me}`, b.dataset.vote)));
    } else if (mg.phase === 'results' && mg.results) {
      el.innerHTML = resultsHtml(mg, ctx.players, ctx.me);
    }
  }

  function sketchPad(el, mg, ctx) {
    const done = !!mg.submissions?.[ctx.me];
    el.innerHTML = `<h3>✏️ Draw: “${esc(mg.prompt)}”</h3><div class="pg-timer"></div>
      <canvas class="pg-pad" width="640" height="640"></canvas>
      <div class="pg-colors">${['#111', '#d7263d', '#1f9d55', '#1e6fd9', '#f2a900', '#8b5a2b'].map((c, i) =>
        `<button style="background:${c}" data-color="${c}" class="${i ? '' : 'on'}" aria-label="colour"></button>`).join('')}</div>
      <div class="pg-btns"><button class="pg-btn" data-act="undo">↶ Undo</button><button class="pg-btn" data-act="clear">Clear</button>
      <button class="pg-btn primary" data-act="done">${done ? '✓ Submitted' : 'Done!'}</button></div>`;
    timer(el.querySelector('.pg-timer'), mg);
    const canvas = el.querySelector('canvas');
    const c2d = canvas.getContext('2d');
    const strokes = [];
    let color = '#111';
    let current = null;
    let submitted = done;
    const paint = () => {
      c2d.fillStyle = '#fff'; c2d.fillRect(0, 0, 640, 640);
      c2d.lineCap = 'round'; c2d.lineJoin = 'round'; c2d.lineWidth = 12;
      for (const s of strokes) {
        c2d.strokeStyle = s.color; c2d.beginPath();
        s.pts.forEach(([x, y], i) => (i ? c2d.lineTo(x, y) : c2d.moveTo(x, y)));
        if (s.pts.length === 1) c2d.lineTo(s.pts[0][0] + 0.1, s.pts[0][1]);
        c2d.stroke();
      }
    };
    const at = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) * 640 / r.width, (e.clientY - r.top) * 640 / r.height]; };
    canvas.addEventListener('pointerdown', (e) => { if (submitted) return; canvas.setPointerCapture?.(e.pointerId); current = { color, pts: [at(e)] }; strokes.push(current); paint(); });
    canvas.addEventListener('pointermove', (e) => { if (current) { current.pts.push(at(e)); paint(); } });
    const end = () => { current = null; };
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
    el.querySelectorAll('[data-color]').forEach((b) => b.addEventListener('click', () => {
      color = b.dataset.color; el.querySelectorAll('[data-color]').forEach((x) => x.classList.toggle('on', x === b));
    }));
    const submit = () => {
      if (submitted || !strokes.length) return;
      submitted = true;
      const out = document.createElement('canvas'); out.width = out.height = 320;
      out.getContext('2d').drawImage(canvas, 0, 0, 320, 320);
      ctx.write(`submissions/${ctx.me}`, out.toDataURL('image/jpeg', 0.7));
      el.querySelector('[data-act="done"]').textContent = '✓ Submitted';
    };
    el.querySelector('[data-act="undo"]').addEventListener('click', () => { if (!submitted) { strokes.pop(); paint(); } });
    el.querySelector('[data-act="clear"]').addEventListener('click', () => { if (!submitted) { strokes.length = 0; paint(); } });
    el.querySelector('[data-act="done"]').addEventListener('click', submit);
    paint();
    // Time's up: hand in whatever is on the canvas
    const t = setTimeout(submit, Math.max(0, (mg.phaseEndsAt || 0) - now() - 400));
    screen.cleanup = () => clearTimeout(t);
  }

  function whackBoard(el, mg, ctx) {
    el.innerHTML = `<h3>🔨 Whack-an-Elf!</h3><div class="pg-timer"></div><div class="whack-score">0</div>
      <div class="whack-grid">${Array.from({ length: 9 }, (_, i) => `<button class="whack-cell" data-cell="${i}"><div class="pop"></div></button>`).join('')}</div>`;
    timer(el.querySelector('.pg-timer'), mg);
    const items = whackSequence(mg.seed, GAMES.whack_elf.playMs);
    const cells = [...el.querySelectorAll('.whack-cell')];
    const hit = new Set();
    let score = 0;
    let sent = null;
    const showing = Array(9).fill(-1);
    const scoreEl = el.querySelector('.whack-score');
    const t0 = mg.playStartAt;
    const frame = () => {
      const t = now() - t0;
      for (let c = 0; c < 9; c++) {
        const idx = items.findIndex((it) => it.cell === c && t >= it.at && t < it.at + it.upMs);
        if (idx !== showing[c]) {
          showing[c] = idx;
          const cell = cells[c];
          const it = items[idx];
          cell.classList.toggle('up', idx >= 0 && !hit.has(idx));
          cell.classList.remove('hit');
          cell.classList.toggle('gold', it?.kind === 'gold');
          cell.dataset.kind = it ? it.kind : '';
          if (it) cell.querySelector('.pop').innerHTML = it.kind === 'grinch' ? GRINCH_SVG : elfSvg();
        }
      }
    };
    const loop = setInterval(frame, 40);
    cells.forEach((cell, c) => cell.addEventListener('pointerdown', () => {
      const idx = showing[c];
      if (idx < 0 || hit.has(idx)) return;
      hit.add(idx);
      const it = items[idx];
      const v = whackPoints(it.kind, (now() - t0 - it.at) / it.upMs);
      score += v;
      scoreEl.textContent = score;
      cell.classList.add('hit'); cell.classList.remove('up');
      const f = document.createElement('div');
      f.className = 'whack-float'; f.textContent = v > 0 ? `+${v}` : `${v}`; f.style.color = v > 0 ? '#1f9d55' : '#d7263d';
      cell.appendChild(f); setTimeout(() => f.remove(), 650);
    }));
    // Report the score live (the TV shows a leaderboard) and once more at the end
    const report = (final) => { if (final || sent !== score) { sent = score; ctx.write(`scores/${ctx.me}`, { score, final }); } };
    const live = setInterval(() => report(false), 700);
    const end = setTimeout(() => { clearInterval(loop); clearInterval(live); report(true); }, Math.max(0, mg.phaseEndsAt - now() - 150));
    frame();
    screen.cleanup = () => { clearInterval(loop); clearInterval(live); clearTimeout(end); };
  }

  function resultsHtml(mg, players, me) {
    const r = mg.results;
    const tied = mg.tiedBidders || [];
    const name = (id) => esc(players?.[id]?.name || '?');
    const unit = mg.type === 'sketch_off' ? 'vote' : 'pt';
    const rows = r.ranking.map((id, i) => {
      const tags = [
        r.winners.includes(id) ? '<span class="tag">wins the tie</span>' : tied.includes(id) ? '<span class="tag">tied</span>' : '',
        r.points?.[id] ? `<span class="tag">+${r.points[id]} party pts</span>` : '',
        r.bonus === id ? '<span class="tag">+1 chip Elf Bonus</span>' : '',
      ].join('');
      const s = r.scoreOf?.[id];
      return `<li class="${tied.includes(id) ? 'tied' : ''} ${r.winners.includes(id) ? 'win' : ''}">
        <span class="pos">${i + 1}</span><span class="nm">${name(id)}${id === me ? ' (you)' : ''}</span>
        <span>${s == null ? 'didn\'t play' : `${s} ${unit}${Math.abs(s) === 1 ? '' : 's'}`}</span>${tags}</li>`;
    }).join('');
    const who = r.winners.map(name).join(' & ');
    return `<h3>${GAMES[mg.type].emoji} Results</h3>
      <p class="pg-sub">${who} ${r.winners.length > 1 ? 'win' : 'wins'} the tie${mg.minigameGoal === 'pick_two' ? ' and duel for the gift' : ' and duel the Opener'}.</p>
      <ol class="pg-rank">${rows}</ol>`;
  }

  // ---------- TV (host) ----------------------------------------------------------------

  function renderHost(el, mg, players) {
    mountCss();
    el.classList.add('pg');
    const g = GAMES[mg.type];
    const name = (id) => esc(players?.[id]?.name || '?');
    const ids = mg.participants || [];
    if (mg.phase === 'intro') {
      el.innerHTML = `<h3>${g.emoji} ${g.label}!</h3><p class="pg-sub">${g.rules}</p>
        ${mg.prompt ? `<div class="pg-prompt">“${esc(mg.prompt)}”</div>` : ''}
        <p class="pg-sub">Tied: <b>${(mg.tiedBidders || []).map(name).join(', ')}</b>. Everyone else plays for party points${mg.prize === 'points_chips' ? ' and the Elf Bonus chip' : ''}.</p>
        <div class="pg-timer"></div>`;
    } else if (mg.phase === 'playing' && mg.type === 'sketch_off') {
      const n = Object.keys(mg.submissions || {}).length;
      el.innerHTML = `<h3>✏️ Drawing “${esc(mg.prompt)}”</h3><div class="pg-timer"></div><p class="pg-sub">${n}/${ids.length} drawings in</p>`;
    } else if (mg.phase === 'playing' && mg.type === 'whack_elf') {
      const scores = whackScores(mg.scores);
      const top = Math.max(5, ...Object.values(scores));
      el.innerHTML = `<h3>🔨 Whack-an-Elf!</h3><div class="pg-timer"></div><div class="pg-bars">${rank(ids, scores, mg.seed).map((id) => `
        <div class="pg-bar"><span class="nm">${name(id)}</span><span class="track"><span class="fill" style="width:${Math.max(0, (scores[id] || 0) / top * 100)}%"></span></span><span class="v">${scores[id] || 0}</span></div>`).join('')}</div>`;
    } else if (mg.phase === 'voting') {
      const order = rank(Object.keys(mg.submissions || {}), {}, mg.seed);
      el.innerHTML = `<h3>🗳️ Vote on your phones!</h3><p class="pg-sub">“${esc(mg.prompt)}” · ${Object.keys(mg.votes || {}).length}/${ids.length} voted</p><div class="pg-timer"></div>
        <div class="pg-gallery">${order.map((pid, i) => `<figure><img src="${mg.submissions[pid]}" alt="drawing ${i + 1}"><figcaption>#${i + 1}</figcaption></figure>`).join('')}</div>`;
    } else if (mg.phase === 'results' && mg.results) {
      el.innerHTML = resultsHtml(mg, players, null);
      return;
    }
    const t = el.querySelector('.pg-timer');
    if (t) timer(t, mg);
  }

  Object.assign(api, { now, syncClock, renderPhone, resetPhone, renderHost });
})(typeof window !== 'undefined' ? window : globalThis);
