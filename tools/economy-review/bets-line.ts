// How good is an opening line made from the two duelists' records so far in the game? (ROADMAP Step 2c, design D)
// line = ra/(ra+rb), r = (wins+K)/(duels+2K). Compared with a flat 50/50, against the true p from the skills.
//   npx vite-node tools/economy-review/bets-line.ts > tools/economy-review/out-bets-line.txt
import { genGames, lineOf } from './bets-core';

const G = Number(process.env.GAMES || 3000);
console.log(`${G} engine games per size, seeds 900000 + 10000*n + i. MAE = mean |line - true p|; Brier on the actual winner.`);
console.log('| n | duels/game | tax burned/game | duels with a tax | both without a record | 50/50: MAE, Brier | K=1 | K=2 | K=3 | K=5 | K=8 | truth: Brier |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|');
for (const n of [4, 6, 8, 12]) {
  const games = genGames(n, G, 900000 + 10000 * n);
  const duels = games.flatMap((g) => g.duels);
  const brier = (fn: (d: typeof duels[0]) => number) => duels.reduce((a, d) => a + (fn(d) - (d.aWon ? 1 : 0)) ** 2, 0) / duels.length;
  const mae = (fn: (d: typeof duels[0]) => number) => duels.reduce((a, d) => a + Math.abs(fn(d) - d.p), 0) / duels.length;
  const cell = (fn: (d: typeof duels[0]) => number) => `${mae(fn).toFixed(3)}, ${brier(fn).toFixed(4)}`;
  const ks = [1, 2, 3, 5, 8].map((K) => cell((d) => lineOf(d.rec[0], d.rec[1], d.rec[2], d.rec[3], K)));
  const tax = games.reduce((a, g) => a + g.tax, 0) / games.length;
  console.log(`| ${n} | ${(duels.length / games.length).toFixed(2)} | ${tax.toFixed(2)} | ${(100 * duels.filter((d) => d.tax > 0).length / duels.length).toFixed(0)}% | ${(100 * duels.filter((d) => d.rec.every((x) => x === 0)).length / duels.length).toFixed(0)}% | ${cell(() => 0.5)} | ${ks.join(' | ')} | ${brier((d) => d.p).toFixed(4)} |`);
}
