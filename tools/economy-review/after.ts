// After the rule changes of 2026-11-13 (final standings by Party Score, +1 Party point per duel fought, a flat
// point to each winning team member): who tops the final standings now? One deviant at a mixed table of six,
// as extra.ts, but scored by the ENGINE's own Party Score (partyPoints + votingWins + duelsFought), not a model.
// Team games are still sim2's model (credit by skill), at a flat 1 point; shown at no team duels and at every duel.
//   npx vite-node tools/economy-review/after.ts        (GAMES=500 for a quick look)
import { allIn, hoarder, mixTable, passive, play, spendDown, valueScaled } from './sim2';
import type { Opts, Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 2000);
const SK6 = [0.5, 0.7, 0.85, 1.15, 1.4, 2.0];
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const share = (v: number[], k: number) => { const mx = Math.max(...v); return v[k] === mx ? 1 / v.filter((q) => q === mx).length : 0; };
let mismatched = 0, games = 0;
console.log(`n=6, ${G} games/row, seeds 150000+, fair share = 17%`);
console.log('| style | team duels | rel. gift value | duels fought | tops the OLD standings (chips) | tops the NEW standings (Party Score) | most skilled seat tops the new standings |');
console.log('|---|---|---|---|---|---|---|');
const conds: [string, Partial<Opts>][] = [['none', {}], ['every duel, flat 1', { teamGame: 1, teamBase: 1, teamBonus: 0 }]];
for (const d of [hoarder, passive, valueScaled, spendDown, allIn] as Strategy2[]) for (const [label, extra] of conds) {
  const val: number[] = [], chip: number[] = [], top: number[] = [], skilled: number[] = [], df: number[] = [];
  for (let i = 0; i < G; i++) {
    const seat = i % 6; const tb = mixTable(6, i); tb[seat] = d;
    const r = play({ strategies: tb, seed: 150000 + i, skill: SK6, votes: 'truthful', ...extra });
    games++;
    if (r.seats.some((s) => s.engineDuels !== s.duelsFought)) mismatched++;      // the engine counts the duels the simulator saw
    const score = r.seats.map((s) => s.engineScore + s.teamPts);
    val.push(r.seats[seat].giftValue - mean(r.seats.filter((_, k) => k !== seat).map((s) => s.giftValue)));
    chip.push(share(r.seats.map((s) => s.chipsLeft), seat));
    top.push(share(score, seat));
    skilled.push(share(score, 5));
    df.push(r.seats[seat].duelsFought);
  }
  const pc = (x: number[]) => `${(100 * mean(x)).toFixed(0)}%`;
  console.log(`| ${d.name} | ${label} | ${mean(val) >= 0 ? '+' : ''}${mean(val).toFixed(2)} | ${mean(df).toFixed(2)} | ${pc(chip)} | ${pc(top)} | ${pc(skilled)} |`);
}
console.log(`engine duelsFought differs from the simulator's own count in ${mismatched} of ${games} games`);
