// Extended deviation matrix (w = 0: payoff = final gift value), who tops the chip standings, action-level deviations.
import { aggressive, allIn, devGain, f, hoarder, passive, snipe, spendDown, valueScaled, withFight, withGambit, withReprisal } from './sim2';
import type { Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 1200);
const cols: Strategy2[] = [aggressive(1), aggressive(2), aggressive(3), aggressive(4), valueScaled, passive, snipe(3), spendDown, allIn, hoarder];
const rows: Strategy2[] = [aggressive(1), aggressive(2), aggressive(3), aggressive(4), valueScaled, passive, snipe(3), allIn];
console.log(`## Deviation gain in gift value (w=0), ${G} games/cell, 6 players, seed base 1, mean±SE (|mean|>2SE marked *)`);
console.log(`| table \\ deviant | ${cols.map((c) => c.name).join(' | ')} |`);
console.log(`|---|${cols.map(() => '---').join('|')}|`);
const firsts: string[] = [];
for (const t of rows) {
  const cells: string[] = []; const fc: string[] = [];
  for (const d of cols) {
    if (d.name === t.name) { cells.push('—'); fc.push('—'); continue; }
    const r = devGain(t, d, { games: G });
    cells.push(`${f(r.gain)}${Math.abs(r.gain.mean) > 2 * r.gain.se ? '*' : ''}`);
    fc.push(`${(r.chipFirst * 100).toFixed(0)}%`);
  }
  console.log(`| **${t.name}** | ${cells.join(' | ')} |`);
  firsts.push(`| **${t.name}** | ${fc.join(' | ')} |`);
}
console.log(`\n## P(the deviant is 1st in the chip standings) (fair share = 17%)`);
console.log(`| table \\ deviant | ${cols.map((c) => c.name).join(' | ')} |`);
console.log(`|---|${cols.map(() => '---').join('|')}|`);
firsts.forEach((l) => console.log(l));

console.log(`\n## Same matrix corner at w=0.25 and w=0.5 (chips count)`);
for (const w of [0.25, 0.5]) for (const t of [aggressive(2), passive, hoarder]) {
  console.log(`w=${w} table ${t.name}: ` + [aggressive(1), aggressive(2), aggressive(4), passive, hoarder, allIn].filter((d) => d.name !== t.name).map((d) => `${d.name} ${f(devGain(t, d, { games: G, w }).gain)}`).join(' | '));
}

console.log(`\n## Action-level deviations (one player changes ONE habit), ${2 * G} games/cell`);
for (const base of [aggressive(2), valueScaled]) {
  const vs = [withFight(base, 'always'), withFight(base, 'never'), withReprisal(base, 'never'), withReprisal(base, 'always'), withGambit(base, 'never'), withGambit(base, 'always')];
  console.log(`table ${base.name}: ` + vs.map((d) => { const r = devGain(base, d, { games: 2 * G, seed: 3 }); return `${d.name.replace(base.name + '+', '')} ${f(r.gain)}`; }).join(' | '));
}
