// Turns a list of game results (Arena.finish) into a markdown balance report.
const { PERSONAS } = require('./personas');
const { PRESETS } = require('./presets');

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const sd = (xs) => { const m = mean(xs); return Math.sqrt(mean(xs.map((x) => (x - m) ** 2))); };
const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : '–');
const pct = (x) => (Number.isFinite(x) ? `${Math.round(x * 100)}%` : '–');
const ratio = (n, d) => (d ? n / d : NaN);

function gameStats(r) {
  const d = r.decisions || [];
  const bidsByGift = {};
  for (const x of d.filter((x) => x.kind === 'bid')) (bidsByGift[x.gift] ||= []).push(x.amount);
  const opened = Object.values(bidsByGift);
  const offers = d.filter((x) => x.kind === 'reprisal');
  const kog = d.filter((x) => x.kind === 'keep-or-gambit');
  const foy = d.filter((x) => x.kind === 'fight-or-yield');
  const paths = d.filter((x) => x.kind === 'endgame-path');
  const seats = r.seats || [];
  const values = seats.filter((s) => s.value != null).map((s) => s.value);
  const types = {};
  for (const x of r.duels || []) types[x.type] = (types[x.type] || 0) + 1;
  return {
    minutes: r.ms / 60000,
    draws: r.turns,
    duels: (r.duels || []).length,
    types,
    contested: ratio(opened.filter((b) => b.some((a) => a > 0)).length, opened.length),
    avgTopBid: mean(opened.map((b) => Math.max(0, ...b))),
    gambitRate: ratio(kog.filter((x) => x.action === 'gambit').length, kog.length),
    yieldRate: ratio(foy.filter((x) => x.action === 'yield').length, foy.length),
    reprisalOffers: offers.length,
    reprisalTaken: ratio(offers.filter((x) => x.giftId).length, offers.length),
    endgameTurns: paths.length,
    stealRate: ratio(paths.filter((x) => x.path === 'steal').length, paths.length),
    chips: mean(seats.map((s) => s.chips)),
    moved: (r.duels || []).reduce((n, d) => n + (d.potWon || 0), 0), // chips handed over in pots
    gap: seats.length ? Math.max(...seats.map((s) => s.chips)) - Math.min(...seats.map((s) => s.chips)) : NaN,
    unspent: ratio(mean(seats.map((s) => s.chips)), r.startingChips),
    broke: ratio(seats.filter((s) => s.chips === 0).length, seats.length),
    hoarders: ratio(seats.filter((s) => s.chips >= r.startingChips * 0.75).length, seats.length),
    value: mean(values),
    spread: sd(values),
    top3: ratio(seats.filter((s) => s.rank != null && s.rank <= 3).length, seats.length),
    giftless: seats.filter((s) => s.gift == null).length,
    predPts: seats.reduce((n, s) => n + (s.votingWins || 0), 0),   // Party Score from correct calls
    partyPts: seats.reduce((n, s) => n + (s.partyPoints || 0), 0), // Party Score from party games
  };
}

function summarize(results, meta) {
  const out = [];
  const ok = results.filter((r) => r.outcome === 'ended');
  out.push(`# Agent play-test report`);
  out.push('');
  out.push(`${results.length} games (${ok.length} finished) · ${meta.lineup ? `lineup: ${meta.lineup.map((p) => `${PERSONAS[p].emoji} ${p}`).join(', ')}` : `random ${meta.players}-player tables from ${Object.keys(PERSONAS).length} personas`} · ` +
    `seed ${meta.seed} · ${f1(meta.minutes)} min wall time. Duels are fair coin flips; gift quality is random 1–10 with per-player taste (±~1.4).`);
  out.push('');

  // --- per preset --------------------------------------------------------------
  const byPreset = {};
  for (const r of results) (byPreset[r.preset] ||= []).push(r);
  out.push('## Settings comparison');
  out.push('');
  out.push('| preset | done | min | duels | contested | top bid | gambit | yield | reprisal taken | endgame turns | chips left | chips moved | rich–poor gap | hoarders | broke | call pts | party pts | gift value | spread | top-3 gift |');
  out.push('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const [name, rs] of Object.entries(byPreset)) {
    const done = rs.filter((r) => r.outcome === 'ended');
    const s = done.map(gameStats);
    const m = (k) => mean(s.map((x) => x[k]).filter(Number.isFinite));
    out.push(`| ${name} | ${done.length}/${rs.length} | ${f1(m('minutes'))} | ${f1(m('duels'))} | ${pct(m('contested'))} | ${f1(m('avgTopBid'))} | ` +
      `${pct(m('gambitRate'))} | ${pct(m('yieldRate'))} | ${pct(m('reprisalTaken'))} | ${f1(m('endgameTurns'))} | ${f1(m('chips'))} (${pct(m('unspent'))}) | ${f1(m('moved'))} | ${f1(m('gap'))} | ` +
      `${pct(m('hoarders'))} | ${pct(m('broke'))} | ${f1(m('predPts'))} | ${f1(m('partyPts'))} | ${f1(m('value'))} | ${f1(m('spread'))} | ${pct(m('top3'))} |`);
  }
  out.push('');
  out.push('Columns: **contested** = opened gifts with ≥1 bid · **chips left** = average final chips (share of starting chips) · **chips moved** = chips handed from loser to winner in pots, per game · **rich–poor gap** = richest minus poorest player at the end · **call pts / party pts** = Party Score handed out per game for correct duel calls / party mini-games · ' +
    '**hoarders** = players ending with ≥75% of their chips (chips didn\'t matter to them) · **broke** = ending on 0 · ' +
    '**gift value** = how much each player values what they took home (0–12) · **spread** = how unequal that is within a game · ' +
    '**top-3 gift** = ended with one of their 3 favourite opened gifts.');
  const described = PRESETS.filter((p) => byPreset[p.name]);
  if (described.length) {
    out.push('');
    for (const p of described) out.push(`- **${p.name}** — ${p.why}`);
  }

  // --- duel mix ----------------------------------------------------------------
  out.push('');
  out.push('## Duel mix (per game)');
  out.push('');
  const typeNames = ['sealed', 'yield', 'tiebreak', 'gambit', 'reprisal', 'steal', 'auction', 'nobids'];
  out.push(`| preset | ${typeNames.join(' | ')} |`);
  out.push(`|---|${typeNames.map(() => '---').join('|')}|`);
  for (const [name, rs] of Object.entries(byPreset)) {
    const s = rs.filter((r) => r.outcome === 'ended').map(gameStats);
    out.push(`| ${name} | ${typeNames.map((t) => f1(mean(s.map((x) => x.types[t] || 0)))).join(' | ')} |`);
  }

  // --- personas ------------------------------------------------------------------
  out.push('');
  out.push('## Personas (all finished games)');
  out.push('');
  out.push('| persona | games | gift value | vs table | top-3 gift | chips left | duels W-L | avg bid when bidding | bids placed | party pts |');
  out.push('|---|---|---|---|---|---|---|---|---|---|');
  const allSeats = ok.flatMap((r) => r.seats.map((s) => ({ ...s, tableMean: mean(r.seats.filter((x) => x.value != null).map((x) => x.value)) })));
  const allBids = ok.flatMap((r) => r.decisions.filter((x) => x.kind === 'bid' || x.kind === 'endgame-bid'));
  for (const p of Object.keys(PERSONAS)) {
    const seats = allSeats.filter((s) => s.persona === p);
    if (!seats.length) continue;
    const bids = allBids.filter((x) => x.persona === p);
    const placed = bids.filter((x) => x.amount > 0);
    const valued = seats.filter((s) => s.value != null);
    out.push(`| ${PERSONAS[p].emoji} ${PERSONAS[p].label} | ${seats.length} | ${f1(mean(valued.map((s) => s.value)))} | ` +
      `${f1(mean(valued.map((s) => s.value - s.tableMean)))} | ${pct(ratio(seats.filter((s) => s.rank != null && s.rank <= 3).length, seats.length))} | ` +
      `${f1(mean(seats.map((s) => s.chips)))} | ${seats.reduce((a, s) => a + s.wins, 0)}-${seats.reduce((a, s) => a + s.losses, 0)} | ` +
      `${f1(mean(placed.map((x) => x.amount)))} | ${pct(ratio(placed.length, bids.length))} | ${f1(mean(seats.map((s) => s.partyPoints || 0)))} |`);
  }
  out.push('');
  out.push('**vs table** = persona\'s gift value minus the average at their table (positive = the style pays off).');

  // --- playability -----------------------------------------------------------------
  out.push('');
  out.push('## Playability');
  out.push('');
  const outcomes = {};
  for (const r of results) outcomes[r.outcome] = (outcomes[r.outcome] || 0) + 1;
  out.push(`Outcomes: ${Object.entries(outcomes).map(([k, v]) => `${k} ${v}`).join(' · ')}`);
  const issues = {};
  for (const r of results) for (const i of r.issues || []) {
    const k = `${i.kind}: ${r.room ? i.text.split(r.room).join('ROOM') : i.text}`;
    (issues[k] ||= { n: 0, seeds: new Set() }).n++;
    issues[k].seeds.add(`${r.preset}#${r.seed}`);
  }
  const list = Object.entries(issues).sort((a, b) => b[1].n - a[1].n);
  if (!list.length) out.push('', 'No stalls, page errors, alerts, missing buttons or invariant violations. ✅');
  else {
    out.push('');
    out.push('| issue | games | e.g. |');
    out.push('|---|---|---|');
    for (const [k, v] of list) out.push(`| ${k.replace(/\|/g, '\\|')} | ${v.n} | ${[...v.seeds].slice(0, 3).join(', ')} |`);
  }
  const stuck = results.filter((r) => r.outcome !== 'ended');
  for (const r of stuck) out.push(`- ${r.outcome} ${r.preset} seed ${r.seed}${r.snapshotDir ? ` → \`${r.snapshotDir}\`` : ''}${r.error ? `: ${r.error.split('\n')[0]}` : ''}`);
  return out.join('\n');
}

module.exports = { summarize, gameStats };
