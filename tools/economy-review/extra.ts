// Who do the two leaderboards crown? One deviant at a mixed table (n=6, skills fixed per seat, truthful duel calls).
// Chip standings (the app's "1st place") vs Party MVP, today and with team games in every duel (MODELLED rewards),
// and a what-if: +1 Party point to each duelist per duel fought.
import { allIn, hoarder, mixTable, passive, play, spendDown, valueScaled } from './sim2';
import type { Opts, Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 2000);
const SK6 = [0.5, 0.7, 0.85, 1.15, 1.4, 2.0];
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const share = (v: number[], k: number) => { const mx = Math.max(...v); return v[k] === mx ? 1 / v.filter((q) => q === mx).length : 0; };
console.log(`n=6, ${G} games/row, seeds 150000+, fair share of any title = 17%`);
console.log('| deviant | Party points | rel. gift value | duels fought | P(1st in chip standings) | P(Party MVP) | P(MVP) if each duelist also gets +1 per duel | P(MVP) if +2 per duel |');
console.log('|---|---|---|---|---|---|---|---|');
const conds: [string, Partial<Opts>][] = [['today (calls + tie games)', {}], ['team games every duel, 2 + up to 3', { teamGame: 1 }], ['team games every duel, flat 1', { teamGame: 1, teamBase: 1, teamBonus: 0 }]];
for (const d of [hoarder, passive, valueScaled, spendDown, allIn] as Strategy2[]) for (const [label, extra] of conds) {
  const val: number[] = [], chip: number[] = [], mvp: number[] = [], mvp1: number[] = [], mvp2: number[] = [], df: number[] = [];
  for (let i = 0; i < G; i++) {
    const seat = i % 6; const tb = mixTable(6, i); tb[seat] = d;
    const r = play({ strategies: tb, seed: 150000 + i, skill: SK6, votes: 'truthful', ...extra });
    val.push(r.seats[seat].giftValue - mean(r.seats.filter((_, k) => k !== seat).map((s) => s.giftValue)));
    chip.push(share(r.seats.map((s) => s.chipsLeft), seat));
    mvp.push(share(r.seats.map((s) => s.partyPoints), seat));
    mvp1.push(share(r.seats.map((s) => s.partyPoints + s.duelsFought), seat));
    mvp2.push(share(r.seats.map((s) => s.partyPoints + 2 * s.duelsFought), seat));
    df.push(r.seats[seat].duelsFought);
  }
  const pc = (x: number[]) => `${(100 * mean(x)).toFixed(0)}%`;
  console.log(`| ${d.name} | ${label} | ${mean(val) >= 0 ? '+' : ''}${mean(val).toFixed(2)} | ${mean(df).toFixed(2)} | ${pc(chip)} | ${pc(mvp)} | ${pc(mvp1)} | ${pc(mvp2)} |`);
}
