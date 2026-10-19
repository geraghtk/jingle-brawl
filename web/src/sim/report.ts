// npm run analyze:economics — runs the experiments and prints a markdown report
// (GAMES=n per cell, REPORT_OUT=path to also save it).
import { writeFileSync } from 'node:fs';
import { DEFAULTS } from '../core/settings';
import { deviationGain, fmt, profile, significant } from './experiments';
import { aggressive, passive, random, valueScaled } from './strategies';
import type { Strategy } from './strategies';

const GAMES = Number(process.env.GAMES || 400);
const S: Strategy[] = [aggressive(1), aggressive(2), aggressive(3), aggressive(4), valueScaled, passive];
const out: string[] = [];
const say = (s = '') => { out.push(s); console.log(s); };

say(`# Jingle Brawl — economic simulation (${GAMES} games per cell, 6 players, default settings)`);
say(`Defaults: max bid ${DEFAULTS.maxBid}, Santa Tax: pots of ${DEFAULTS.taxThreshold}+ pay at most ${DEFAULTS.maxWinnerPayout}, Loser's Dividend: ${DEFAULTS.loserDividendRule}, base cost ${DEFAULTS.baseCost} + Naughty.`);
say('Payoff = value (0–12) of the gift you end with. Duels are coin flips. ± = 95% CI.');
say();
say('## Who beats whom — deviation gain (row = the table, column = one player deviating)');
say('Positive = the lone deviant ends with a better gift than the table players in the same games.');
say();
say(`| table \\ deviant | ${S.map((s) => s.name).join(' | ')} |`);
say(`|---|${S.map(() => '---').join('|')}|`);
for (const t of S) {
  const cells = S.map((d) => {
    if (d === t) return '—';
    const g = deviationGain(t, d, { games: GAMES });
    return `${fmt(g)}${significant(g) ? (g.mean > 0 ? ' ▲' : ' ▼') : ''}`;
  });
  say(`| **${t.name}** | ${cells.join(' | ')} |`);
}
say();
say('▲ / ▼ = significantly better / worse than staying with the table. A row with no ▲ is an equilibrium among these strategies.');
say();
say('## What each table looks like (everyone plays the same strategy)');
say();
say('| strategy | duels | sealed duels | Gambits | avg gift value | giftless | chips left | Santa Tax burned | costs burned | dividends minted | max Naughty |');
say('|---|---|---|---|---|---|---|---|---|---|---|');
for (const s of [...S, random]) {
  const p = profile(s, { games: Math.max(100, GAMES / 2) });
  say(`| ${s.name} | ${p.duels.mean.toFixed(1)} | ${p.sealedDuels.mean.toFixed(1)} | ${p.gambits.mean.toFixed(1)} | ${p.payoff.mean.toFixed(2)} | ${p.giftless.mean.toFixed(2)} | ${(p.chipsLeftShare.mean * 100).toFixed(0)}% | ${p.taxBurned.mean.toFixed(1)} | ${p.costsBurned.mean.toFixed(1)} | ${p.dividendsMinted.mean.toFixed(1)} | ${p.maxNaughty.mean.toFixed(1)} |`);
}
say();
say('## Does a higher bid pay at a bid-2 table — and do the proposed tax changes matter?');
say();
const variants: [string, Record<string, unknown>][] = [
  ['defaults (cap 2 from pot 2)', {}],
  ['cap 3 from pot 3 ("progressive")', { maxWinnerPayout: 3, taxThreshold: 3 }],
  ['no Santa Tax', { taxThreshold: 99 }],
  ["dividend for every loser", { loserDividendRule: 'everyone' }],
];
say(`| settings | bid 3 vs bid 2 | bid 4 vs bid 2 | bid 1 vs bid 2 | passive vs bid 2 |`);
say('|---|---|---|---|---|');
for (const [label, settings] of variants) {
  const cell = (d: Strategy) => { const g = deviationGain(aggressive(2), d, { games: GAMES, settings }); return `${fmt(g)}${significant(g) ? (g.mean > 0 ? ' ▲' : ' ▼') : ''}`; };
  say(`| ${label} | ${cell(aggressive(3))} | ${cell(aggressive(4))} | ${cell(aggressive(1))} | ${cell(passive)} |`);
}
say();
say('## If leftover chips count (the final standings are sorted by chips)');
say('Payoff = gift value + w × chips left. w = 0 is the rules as written (chips are worth nothing at the end).');
say();
say('| w | bid 2 at a passive table | passive at a bid-2 table | bid 4 at a bid-2 table |');
say('|---|---|---|---|');
for (const w of [0, 0.25, 0.5]) {
  const c = (t: Strategy, d: Strategy) => { const g = deviationGain(t, d, { games: GAMES, chipWeight: w }); return `${fmt(g)}${significant(g) ? (g.mean > 0 ? ' ▲' : ' ▼') : ''}`; };
  say(`| ${w} | ${c(passive, aggressive(2))} | ${c(aggressive(2), passive)} | ${c(aggressive(2), aggressive(4))} |`);
}
if (process.env.REPORT_OUT) writeFileSync(process.env.REPORT_OUT, out.join('\n') + '\n');
