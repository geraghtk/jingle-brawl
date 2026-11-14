// Sensitivity: one setting at a time around the defaults, n=6, same seeds in every cell (paired differences vs default).
import { aggressive, allIn, devGain, f, mixTable, passive, play, stat, valueScaled } from './sim2';
import type { Res, Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 1500);
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
type M = { duels: number; value: number; spread: number; left: number; broke: number; topbid: number; repr: number; gamb: number; naughty: number; spentPer: number };
const metrics = (r: Res): M => {
  const v = r.seats.map((s) => s.giftValue); const m = mean(v);
  const sealed = r.duels.filter((d) => d.kind === 'sealed');
  return { duels: r.duels.length, value: m, spread: Math.sqrt(mean(v.map((q) => (q - m) ** 2))), left: r.ledger.end / r.ledger.start, broke: mean(r.seats.map((s) => (s.chipsLeft === 0 ? 1 : 0))), topbid: sealed.length ? mean(sealed.map((d) => d.bid)) : 0, repr: r.reprisals, gamb: r.gambits, naughty: r.maxNaughty, spentPer: (r.ledger.start - r.ledger.end) / r.n };
};
function run(settings: Record<string, unknown>, table: (i: number) => Strategy2[]): M[] {
  const out: M[] = []; for (let i = 0; i < G; i++) out.push(metrics(play({ strategies: table(i), seed: 30000 + i, settings }))); return out;
}
const levers: [string, string, unknown[]][] = [
  ['startingChips', 'startingChips', [2, 3, 5, 8, 10, 15, 25]],
  ['maxBid', 'maxBid', [1, 2, 4, 6, 10]],
  ['maxWinnerPayout (cap)', 'maxWinnerPayout', [0, 1, 2, 4, 99]],
  ['taxThreshold', 'taxThreshold', [1, 2, 3, 99]],
  ['loserDividendRule', 'loserDividendRule', ['nobody', 'defenders', 'everyone']],
  ['baseCost', 'baseCost', [0, 1, 2, 3, 5]],
  ['maxReprisalDepth', 'maxReprisalDepth', [0, 1, 2]],
];
for (const [tname, table] of [['mixed table', (i: number) => mixTable(6, i)], ['all-in table (chips treated as free)', () => Array(6).fill(allIn)]] as [string, (i: number) => Strategy2[]][]) {
  const base = run({}, table);
  console.log(`\n## Sensitivity, ${tname}, n=6, ${G} games/cell, seeds 30000+. Each cell: mean (paired diff vs default ± SE)`);
  console.log('| lever | value | duels/game | avg gift value | value spread (SD) | chips left | broke players | chips spent/player | reprisals | gambits |');
  console.log('|---|---|---|---|---|---|---|---|---|---|');
  for (const [label, key, vals] of levers) for (const v of vals) {
    const ms = run({ [key]: v }, table);
    const c = (k: keyof M, d = 2, pct = false) => { const diff = stat(ms.map((m, i) => m[k] - base[i][k])); const mu = mean(ms.map((m) => m[k])); const s = pct ? 100 : 1; const sig = Math.abs(diff.mean) > 2 * diff.se && diff.se > 0 ? '*' : ''; return `${(mu * s).toFixed(d)}${pct ? '%' : ''} (${diff.mean >= 0 ? '+' : ''}${(diff.mean * s).toFixed(d)}±${(diff.se * s).toFixed(d)}${sig})`; };
    console.log(`| ${label} | ${String(v)} | ${c('duels', 1)} | ${c('value')} | ${c('spread')} | ${c('left', 1, true)} | ${c('broke', 1, true)} | ${c('spentPer')} | ${c('repr')} | ${c('gamb')} |`);
  }
}

console.log(`\n## Do the settings change the best response? deviation gains (gift value, w=0) at a bid-2 table, 1000 games/cell, mean±SE`);
console.log('| settings | bid 1 | bid 3 | bid 4 | passive | all-in | value-scaled |');
console.log('|---|---|---|---|---|---|---|');
const variants: [string, Record<string, unknown>][] = [['defaults', {}], ['startingChips 3', { startingChips: 3 }], ['startingChips 5', { startingChips: 5 }], ['startingChips 25', { startingChips: 25 }], ['no tax (threshold 99)', { taxThreshold: 99 }], ['cap 0 (bid always lost)', { maxWinnerPayout: 0, taxThreshold: 1 }], ['dividend nobody', { loserDividendRule: 'nobody' }], ['dividend everyone', { loserDividendRule: 'everyone' }], ['baseCost 0', { baseCost: 0 }], ['baseCost 3', { baseCost: 3 }], ['maxBid 10', { maxBid: 10 }]];
for (const [label, settings] of variants) {
  const cell = (d: Strategy2) => { const g = devGain(aggressive(2), d, { games: 1000, settings }).gain; return `${f(g)}${Math.abs(g.mean) > 2 * g.se ? '*' : ''}`; };
  const hi = settings.maxBid ? aggressive(10) : aggressive(4);
  console.log(`| ${label} | ${cell(aggressive(1))} | ${cell(aggressive(3))} | ${cell(hi)} | ${cell(passive)} | ${cell(allIn)} | ${cell(valueScaled)} |`);
}
