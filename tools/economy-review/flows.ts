// Faucets / sinks / velocity, spend-down, Gini over time, snowballing, position effects, allocative efficiency, by table size.
import { allIn, aggressive, corr, f0, gini, hoarder, mixTable, passive, play, stat, valueScaled } from './sim2';
import type { Res, Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 2000);
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
// best total value over all assignments of gifts to seats (bitmask DP)
function optimal(v: number[][]): number {
  const n = v.length; const dp = new Float64Array(1 << n).fill(-1); dp[0] = 0;
  for (let m = 0; m < (1 << n); m++) { if (dp[m] < 0) continue; let s = 0; for (let b = m; b; b &= b - 1) s++; if (s >= n) continue; for (let k = 0; k < n; k++) if (!(m & (1 << k))) { const q = dp[m] + v[s][k]; if (q > dp[m | (1 << k)]) dp[m | (1 << k)] = q; } }
  return dp[(1 << n) - 1];
}
function batch(n: number, table: (i: number) => Strategy2[], games: number, seed0: number, extra: Record<string, unknown> = {}): Res[] {
  const out: Res[] = []; for (let i = 0; i < games; i++) out.push(play({ strategies: table(i), seed: seed0 + i, ...extra })); return out;
}

console.log(`## Chip flows per game by table size (mixed table: bid2/value-scaled/bid3/passive/random/bid1 rotating), seeds 50000+, mean (SE)`);
console.log('| n | games | supply | duels | sealed bids paid | pots back | tax burn | cost burn | dividends | elf bonus | net change | chips left | gross spend / supply | players >=75% chips | broke | max Naughty |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
const byN: Record<number, Res[]> = {};
for (const n of [3, 4, 6, 8, 12]) {
  const g = n === 12 ? Math.round(G / 2) : G;
  const rs = batch(n, (i) => mixTable(n, i), g, 50000);
  byN[n] = rs;
  const L = (k: keyof Res['ledger']) => stat(rs.map((r) => r.ledger[k]));
  const start = rs[0].ledger.start;
  const hoard = mean(rs.flatMap((r) => r.seats.map((s) => (s.chipsLeft >= 0.75 * (start / n) ? 1 : 0))));
  const broke = mean(rs.flatMap((r) => r.seats.map((s) => (s.chipsLeft === 0 ? 1 : 0))));
  const vel = stat(rs.map((r) => (r.ledger.bidsPaid + r.ledger.costsBurned + r.ledger.tollsMoved) / r.ledger.start));
  const net = stat(rs.map((r) => r.ledger.end - r.ledger.start));
  console.log(`| ${n} | ${g} | ${start} | ${f0(stat(rs.map((r) => r.duels.length)), 1)} | ${f0(L('bidsPaid'), 1)} | ${f0(L('potsPaid'), 1)} | ${f0(L('taxBurned'), 1)} | ${f0(L('costsBurned'), 1)} | ${f0(L('dividendsMinted'), 1)} | ${f0(L('elfBonusMinted'), 2)} | ${f0(net, 1)} | ${(100 * mean(rs.map((r) => r.ledger.end / r.ledger.start))).toFixed(1)}% | ${f0(vel, 3)} | ${(hoard * 100).toFixed(0)}% | ${(broke * 100).toFixed(1)}% | ${f0(stat(rs.map((r) => r.maxNaughty)), 1)} |`);
}

console.log(`\n## Spend-down? bids by game progress (mixed table). share of bid decisions that are >0, mean positive bid, chips spent per player`);
console.log('| n | early third: bid rate / mean bid | middle | late third | total net chips spent per player (start - end) | max a player could be charged if top bidder every turn |');
console.log('|---|---|---|---|---|---|');
for (const n of [3, 4, 6, 8, 12]) {
  const rs = byN[n]; const cell = (lo: number, hi: number) => { const b = rs.flatMap((r) => r.bids.filter((q) => q.turn / n > lo && q.turn / n <= hi)); const pos = b.filter((q) => q.amount > 0); return `${(100 * pos.length / Math.max(1, b.length)).toFixed(0)}% / ${mean(pos.map((q) => q.amount)).toFixed(2)}`; };
  console.log(`| ${n} | ${cell(0, 1 / 3)} | ${cell(1 / 3, 2 / 3)} | ${cell(2 / 3, 1.01)} | ${mean(rs.map((r) => (r.ledger.start - r.ledger.end) / n)).toFixed(2)} | ${4 * (n - 1)} |`);
}

console.log(`\n## Inequality over a game (mixed table): Gini of chips after 25/50/75/100% of gifts, Gini of final gift value, correlations`);
console.log('| n | chip Gini 25% | 50% | 75% | end | gift-value Gini end | corr(final chips, final gift value) | P(chip leader has above-median gift) | corr(chips at 50%, final value) |');
console.log('|---|---|---|---|---|---|---|---|---|');
for (const n of [3, 4, 6, 8, 12]) {
  const rs = byN[n];
  const at = (frac: number) => stat(rs.map((r) => { const s = r.snaps[Math.min(r.snaps.length - 1, Math.max(0, Math.round(frac * r.snaps.length) - 1))]; return s ? gini(s.chips) : 0; }));
  const endG = stat(rs.map((r) => gini(r.seats.map((s) => s.chipsLeft))));
  const vG = stat(rs.map((r) => gini(r.seats.map((s) => s.giftValue))));
  const chips = rs.flatMap((r) => { const m = mean(r.seats.map((s) => s.chipsLeft)); return r.seats.map((s) => s.chipsLeft - m); });
  const vals = rs.flatMap((r) => { const m = mean(r.seats.map((s) => s.giftValue)); return r.seats.map((s) => s.giftValue - m); });
  const lead = mean(rs.map((r) => { const mx = Math.max(...r.seats.map((s) => s.chipsLeft)); const leaders = r.seats.filter((s) => s.chipsLeft === mx); const sorted = r.seats.map((s) => s.giftValue).sort((a, b) => a - b); const med = (sorted[Math.floor((n - 1) / 2)] + sorted[Math.ceil((n - 1) / 2)]) / 2; return mean(leaders.map((s) => (s.giftValue > med ? 1 : 0))); }));
  const midC: number[] = [], finV: number[] = [];
  rs.forEach((r) => { const s = r.snaps[Math.floor(r.snaps.length / 2)]; if (!s) return; const mc = mean(s.chips), mv = mean(r.seats.map((q) => q.giftValue)); s.chips.forEach((c, i) => { midC.push(c - mc); finV.push(r.seats[i].giftValue - mv); }); });
  console.log(`| ${n} | ${f0(at(0.25), 3)} | ${f0(at(0.5), 3)} | ${f0(at(0.75), 3)} | ${f0(endG, 3)} | ${f0(vG, 3)} | ${corr(chips, vals).toFixed(3)} | ${(lead * 100).toFixed(0)}% | ${corr(midC, finV).toFixed(3)} |`);
}

console.log(`\n## Snowballing: does winning your first duel compound? (n=6, by dividend rule; relative final gift value = mine - table mean)`);
console.log('| dividend rule | won first duel: rel. value | lost first duel | difference (SE) | later-duel win rate after a first win | after a first loss | final chips after first win / loss |');
console.log('|---|---|---|---|---|---|---|');
for (const rule of ['defenders', 'everyone', 'nobody']) {
  const rs = rule === 'defenders' ? byN[6] : batch(6, (i) => mixTable(6, i), G, 50000, { settings: { loserDividendRule: rule } });
  const w: number[] = [], l: number[] = [], wr: number[] = [], lr: number[] = [], wc: number[] = [], lc: number[] = [];
  rs.forEach((r) => { const m = mean(r.seats.map((s) => s.giftValue)); r.seats.forEach((s) => { if (s.firstDuelWon == null) return; const later = s.duelWins + s.duelLosses - 1; const lw = s.duelWins - (s.firstDuelWon ? 1 : 0); (s.firstDuelWon ? w : l).push(s.giftValue - m); (s.firstDuelWon ? wc : lc).push(s.chipsLeft); if (later > 0) (s.firstDuelWon ? wr : lr).push(lw / later); }); });
  const sw = stat(w), sl = stat(l);
  console.log(`| ${rule} | ${f0(sw)} | ${f0(sl)} | ${(sw.mean - sl.mean).toFixed(2)} (${Math.sqrt(sw.se ** 2 + sl.se ** 2).toFixed(2)}) | ${mean(wr).toFixed(3)} | ${mean(lr).toFixed(3)} | ${mean(wc).toFixed(2)} / ${mean(lc).toFixed(2)} |`);
}

console.log(`\n## Position: when you first open (mixed table). relative final gift value, P(keep the gift you opened), relative to quality luck`);
console.log('| n | first third of openers: rel. value / kept own / duels fought | middle | last third | first-vs-last diff (SE) |');
console.log('|---|---|---|---|---|');
for (const n of [4, 6, 8, 12]) {
  const rs = byN[n]; const groups: { v: number[]; k: number[]; d: number[] }[] = [0, 1, 2].map(() => ({ v: [], k: [], d: [] }));
  rs.forEach((r) => { const m = mean(r.seats.map((s) => s.giftValue)); r.seats.forEach((s) => { if (s.firstOpenTurn < 0) return; const gq = Math.min(2, Math.floor((3 * s.firstOpenTurn) / n)); groups[gq].v.push(s.giftValue - m); groups[gq].k.push(s.keptFirstGift ? 1 : 0); groups[gq].d.push(s.duelsFought); }); });
  const c = (q: typeof groups[0]) => `${f0(stat(q.v))} / ${(100 * mean(q.k)).toFixed(0)}% / ${mean(q.d).toFixed(2)}`;
  const a = stat(groups[0].v), b = stat(groups[2].v);
  console.log(`| ${n} | ${c(groups[0])} | ${c(groups[1])} | ${c(groups[2])} | ${(a.mean - b.mean).toFixed(2)} (${Math.sqrt(a.se ** 2 + b.se ** 2).toFixed(2)}) |`);
}
{
  const rs = byN[6]; const never = mean(rs.flatMap((r) => r.seats.map((s) => (s.firstOpenTurn < 0 ? 1 : 0))));
  const nv = stat(rs.flatMap((r) => { const m = mean(r.seats.map((s) => s.giftValue)); return r.seats.filter((s) => s.firstOpenTurn < 0).map((s) => s.giftValue - m); }));
  console.log(`n=6: ${(never * 100).toFixed(1)}% of players never open a gift (won one by challenge first); their rel. value ${f0(nv)}`);
}

console.log(`\n## Allocative efficiency (n=6, 1000 games, seeds 70000+): 0% = everyone keeps what they open (random), 100% = best possible assignment`);
console.log('| table | avg gift value | efficiency | value SD within game | duels |');
console.log('|---|---|---|---|---|');
const tables: [string, (i: number) => Strategy2[]][] = [['hoarder (no trades)', () => Array(6).fill(hoarder)], ['passive', () => Array(6).fill(passive)], ['bid 2', () => Array(6).fill(aggressive(2))], ['bid 4', () => Array(6).fill(aggressive(4))], ['value-scaled', () => Array(6).fill(valueScaled)], ['all-in', () => Array(6).fill(allIn)], ['mixed', (i) => mixTable(6, i)]];
for (const [nm, tb] of tables) {
  const rs = batch(6, tb, 1000, 70000);
  const eff = stat(rs.map((r) => { const ach = r.seats.reduce((a, s) => a + s.giftValue, 0); const rnd = mean(r.value.flat()) * 6; const opt = optimal(r.value); return (ach - rnd) / Math.max(1e-9, opt - rnd); }));
  const sd = stat(rs.map((r) => { const v = r.seats.map((s) => s.giftValue); const m = mean(v); return Math.sqrt(mean(v.map((q) => (q - m) ** 2))); }));
  console.log(`| ${nm} | ${f0(stat(rs.map((r) => mean(r.seats.map((s) => s.giftValue)))))} | ${(100 * eff.mean).toFixed(1)}% ±${(100 * eff.se).toFixed(1)} | ${f0(sd)} | ${mean(rs.map((r) => r.duels.length)).toFixed(1)} |`);
}

console.log(`\n## What is a chip worth? one seat's endowment changed, same seeds (n=6, others 10 chips, mixed table; the seat plays the named strategy)`);
console.log('| focal strategy | 0 chips | 1 | 2 | 4 | 6 | 10 | 20 | 40 |  (focal final gift value - table mean, SE) |');
console.log('|---|---|---|---|---|---|---|---|---|---|');
for (const s of [valueScaled, aggressive(4), allIn]) {
  const cells = [0, 1, 2, 4, 6, 10, 20, 40].map((c) => {
    const d: number[] = [];
    for (let i = 0; i < G; i++) { const seat = i % 6; const tb = mixTable(6, i); tb[seat] = s; const r = play({ strategies: tb, seed: 90000 + i, chips: (k) => (k === seat ? c : undefined) }); const oth = r.seats.filter((_, k) => k !== seat).map((q) => q.giftValue); d.push(r.seats[seat].giftValue - mean(oth)); }
    return f0(stat(d));
  });
  console.log(`| ${s.name} | ${cells.join(' | ')} | |`);
}
