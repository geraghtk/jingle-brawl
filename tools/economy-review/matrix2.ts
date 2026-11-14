// Is "bid the max only on a big upgrade" (snipe) or budget smoothing (spend-down) an equilibrium? Rows the first matrix lacked.
import { aggressive, allIn, devGain, f, passive, snipe, spendDown, valueScaled } from './sim2';
import type { Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 1200);
const cols: Strategy2[] = [aggressive(1), aggressive(2), aggressive(4), valueScaled, passive, snipe(2), snipe(3), snipe(4), spendDown, allIn];
const rows: Strategy2[] = [spendDown, snipe(2), snipe(3), snipe(4)];
for (const n of [6, 4, 10]) {
  console.log(`\n## Deviation gain in gift value (w=0), ${n} players, ${G} games/cell, seed base 5, mean±SE (* = |mean|>2SE)`);
  console.log(`| table \\ deviant | ${cols.map((c) => c.name).join(' | ')} |`);
  console.log(`|---|${cols.map(() => '---').join('|')}|`);
  for (const t of rows) {
    if (n !== 6 && t.name !== 'snipe3' && t.name !== 'spend-down') continue;
    const cells = cols.map((d) => { if (d.name === t.name) return '—'; const r = devGain(t, d, { games: G, seed: 5, n }).gain; return `${f(r)}${Math.abs(r.mean) > 2 * r.se ? '*' : ''}`; });
    console.log(`| **${t.name}** | ${cells.join(' | ')} |`);
  }
}
