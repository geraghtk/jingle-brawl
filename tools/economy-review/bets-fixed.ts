// Spectator-betting designs with FIXED-HABIT bettors (ROADMAP Step 2c). One focal seat (rotating) at an otherwise
// honest table, and at a value-betting table (live-odds designs only). Chips per game, mean ± SE over game replays.
//   npx vite-node tools/economy-review/bets-fixed.ts > tools/economy-review/out-bets-fixed.txt
//   GAMES=3000 REPS=2 SIZES=4,6,8,12 by default (eval seeds 900000+ per size; the learner scripts use the same eval games)
import { calib, DESIGNS, evalTable, f, genGames, SIGMAS, table } from './bets-core';
import type { Kind } from './bets-core';

const G = Number(process.env.GAMES || 3000), REPS = Number(process.env.REPS || 2);
const SIZES = (process.env.SIZES || '4,6,8,12').split(',').map(Number);
const KEYS = (process.env.DESIGNS || Object.keys(DESIGNS).join(',')).split(',');
const HABITS: Kind[] = ['honest', 'value', 'fav', 'contra', 'random', 'valueLast'];
console.log(`Eval: ${G} engine games per table size (seeds 900000 + 10000*n + i), each replayed ${REPS}x with fresh beliefs/orders. Abstainer = 0 by construction.`);
console.log('Columns: focal seat profit per game by habit; "pair" = joint profit of two seats, one always backing A, one always B (collusion probe);');
console.log('net = chips the betting adds to (+) or removes from (-) the whole table per game (table of the row habit, no deviant); tax = what the duel tax burned in those games;');
console.log('pool vs p: mean |closing pool share - true p| (line = the record line alone, half = always 50/50), slope of pool share on p (1 = calibrated), two-sided = duels with real backers on both sides.\n');
for (const n of SIZES) {
  const games = genGames(n, G, 900000 + 10000 * n);
  const dpg = games.reduce((a, g) => a + g.duels.length, 0) / games.length;
  console.log(`\n## n=${n} (${n - 2} spectators a duel), ${games.length} games, ${dpg.toFixed(2)} duels a game`);
  console.log(`| design | crowd | table | ${HABITS.join(' | ')} | pair A+B | net chips/game | staked/game | tax/game | bets placed | MAE pool / line / half | slope | two-sided |`);
  console.log(`|---|---|---|${HABITS.map(() => '---').join('|')}|---|---|---|---|---|---|---|---|`);
  for (const key of KEYS) for (const [sn, sigma] of Object.entries(SIGMAS)) for (const tk of ['honest', 'value'] as Kind[]) {
    const d = DESIGNS[key];
    if (tk === 'value' && !(d.seq && d.settle === 'pari')) continue;
    const base = evalTable(d, games, sigma, (_gi, fs) => table(n, tk, fs, tk), REPS, 11);
    const cells = HABITS.map((h) => f(evalTable(d, games, sigma, (_gi, fs) => table(n, tk, fs, h), REPS, 11).focal));
    const pair = evalTable(d, games, sigma, (_gi, fs) => { const t = table(n, tk, fs, 'alwaysA'); t[(fs + 1) % n] = { kind: 'alwaysB' }; return t; }, REPS, 11).pair;
    const c = calib(base.rec);
    console.log(`| ${key} | ${sn} | ${tk} | ${cells.join(' | ')} | ${f(pair)} | ${f(base.net)} | ${base.staked.mean.toFixed(1)} | ${base.tax.mean.toFixed(1)} | ${(100 * base.part).toFixed(0)}% | ${c.maePool.toFixed(3)} / ${c.maeLine.toFixed(3)} / ${c.maeHalf.toFixed(3)} | ${c.slope.toFixed(2)} | ${(100 * c.twoSided).toFixed(0)}% |`);
  }
}
console.log('\nDesigns:'); for (const k of KEYS) console.log(`  ${k}: ${DESIGNS[k].name}`);
