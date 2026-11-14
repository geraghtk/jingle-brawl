// Contest theory + side systems: skill-based duels (Tullock ratio), the value of bidding by skill, Party Score with
// duel calls (real engine votes), tie party games (real engine) and a MODELLED team mini-game (conveyor-like rewards), spectator bets.
import { aggressive, corr, f0, mixTable, passive, play, stat } from './sim2';
import type { Opts, Res, Strategy2 } from './sim2';

const G = Number(process.env.GAMES || 2000);
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const SK6 = [0.5, 0.7, 0.85, 1.15, 1.4, 2.0];
const skillFor = (n: number) => Array.from({ length: n }, (_, i) => SK6[Math.round((i * 5) / Math.max(1, n - 1))]);

console.log(`## Skill and the gift (n=6, mixed strategies rotate over seats, skills fixed per seat; ${G} games, seeds 110000+)`);
console.log('| duels | ' + SK6.map((s) => `skill ${s}`).join(' | ') + ' | best - worst (SE) |');
console.log('|---|' + SK6.map(() => '---').join('|') + '|---|');
for (const [label, extra] of [['coin flips', {}], ['1v1 skill: rel. gift value', { skill: SK6 }], ['team games in every duel: rel. gift value', { skill: SK6, teamGame: 1 }]] as [string, Partial<Opts>][]) {
  const per: number[][] = SK6.map(() => []); const wr: number[][] = SK6.map(() => []);
  for (let i = 0; i < G; i++) { const r = play({ strategies: mixTable(6, i), seed: 110000 + i, ...extra }); const m = mean(r.seats.map((s) => s.giftValue)); r.seats.forEach((s, k) => { per[k].push(s.giftValue - m); if (s.duelWins + s.duelLosses) wr[k].push(s.duelWins / (s.duelWins + s.duelLosses)); }); }
  const a = stat(per[5]), b = stat(per[0]);
  console.log(`| ${label} | ${per.map((p, k) => `${f0(stat(p))} (wins ${(100 * mean(wr[k])).toFixed(0)}%)`).join(' | ')} | ${(a.mean - b.mean).toFixed(2)} (${Math.sqrt(a.se ** 2 + b.se ** 2).toFixed(2)}) |`);
}

console.log(`\n## Value of bidding by duel skill: one seat with win prob p vs everyone, at a bid-2 table (n=6, ${G} games, seeds 120000+). Focal rel. gift value`);
console.log('| focal win prob | passive | bid 1 | bid 2 | bid 4 | bid 4 - passive (SE) |');
console.log('|---|---|---|---|---|---|');
for (const [p, sk] of [[0.25, 1 / 3], [0.5, 1], [0.75, 3]] as [number, number][]) {
  const cells: ReturnType<typeof stat>[] = [];
  for (const s of [passive, aggressive(1), aggressive(2), aggressive(4)]) {
    const d: number[] = [];
    for (let i = 0; i < G; i++) { const seat = i % 6; const tb: Strategy2[] = Array(6).fill(aggressive(2)); tb[seat] = s; const skill = Array(6).fill(1); skill[seat] = sk; const r = play({ strategies: tb, seed: 120000 + i, skill }); d.push(r.seats[seat].giftValue - mean(r.seats.filter((_, k) => k !== seat).map((q) => q.giftValue))); }
    cells.push(stat(d));
  }
  console.log(`| ${p} | ${cells.map((c) => f0(c)).join(' | ')} | ${(cells[3].mean - cells[0].mean).toFixed(2)} (${Math.sqrt(cells[3].se ** 2 + cells[0].se ** 2).toFixed(2)}) |`);
}

console.log(`\n## Party Score by source (truthful duel calls through the engine; tie party games through the engine; team game MODELLED: winners 2 + round(3*credit/top))`);
console.log('| n | team-game share of duels | calls/player | tie-game pts/player | team pts/player | Party Score mean | SD across players | share of score variance from team pts | corr(score, skill) | corr(score, gift value) | corr(score, duels fought) | P(MVP is the most skilled) | P(MVP also best gift) | MVP margin over 2nd |');
console.log('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
const partyRow = (n: number, label: string, extra: Partial<Opts>, games: number) => {
  const skill = skillFor(n); const rs: Res[] = [];
  for (let i = 0; i < games; i++) rs.push(play({ strategies: mixTable(n, i), seed: 130000 + i, skill, votes: 'truthful', ...extra }));
  const per = (k: 'callPts' | 'tiePts' | 'teamPts' | 'partyPoints') => mean(rs.flatMap((r) => r.seats.map((s) => s[k])));
  const sd = mean(rs.map((r) => { const v = r.seats.map((s) => s.partyPoints); const m = mean(v); return Math.sqrt(mean(v.map((q) => (q - m) ** 2))); }));
  const dev = (k: 'teamPts' | 'partyPoints' | 'giftValue' | 'duelsFought') => rs.flatMap((r) => { const m = mean(r.seats.map((s) => s[k])); return r.seats.map((s) => s[k] - m); });
  const sc = dev('partyPoints'), tp = dev('teamPts');
  const varShare = mean(tp.map((q) => q * q)) / Math.max(1e-9, mean(sc.map((q) => q * q)));
  const skl = rs.flatMap(() => skill.map((s) => s - mean(skill)));
  let mvpSk = 0, mvpGift = 0; const margin: number[] = [];
  rs.forEach((r) => { const v = r.seats.map((s) => s.partyPoints); const mx = Math.max(...v); const w = v.map((q, i) => (q === mx ? i : -1)).filter((i) => i >= 0); const topSkill = skill.indexOf(Math.max(...skill)); mvpSk += w.includes(topSkill) ? 1 / w.length : 0; const gv = r.seats.map((s) => s.giftValue); const bg = gv.indexOf(Math.max(...gv)); mvpGift += w.includes(bg) ? 1 / w.length : 0; const sorted = [...v].sort((a, b) => b - a); margin.push(sorted[0] - sorted[1]); });
  console.log(`| ${n} | ${label} | ${per('callPts').toFixed(2)} | ${per('tiePts').toFixed(2)} | ${per('teamPts').toFixed(2)} | ${per('partyPoints').toFixed(2)} | ${sd.toFixed(2)} | ${(100 * varShare).toFixed(0)}% | ${corr(sc, skl).toFixed(2)} | ${corr(sc, dev('giftValue')).toFixed(2)} | ${corr(sc, dev('duelsFought')).toFixed(2)} | ${(100 * mvpSk / rs.length).toFixed(0)}% | ${(100 * mvpGift / rs.length).toFixed(0)}% | ${mean(margin).toFixed(2)} |`);
};
for (const n of [6, 8, 12]) {
  const g = n === 12 ? Math.round(G / 3) : G;
  partyRow(n, '0% (today)', {}, g);
  partyRow(n, '50%', { teamGame: 0.5 }, g);
  partyRow(n, '100%', { teamGame: 1 }, g);
  partyRow(n, '100%, rewards 1 + up to 1', { teamGame: 1, teamBase: 1, teamBonus: 1 }, g);
  partyRow(n, '100%, flat 1 to winners', { teamGame: 1, teamBase: 1, teamBonus: 0 }, g);
  partyRow(n, '100%, 2+3 and losers get 1', { teamGame: 1, teamLoserPoints: 1 }, g);
}

console.log(`\n## Spectator bets (spectatorBets=1) through the engine: n=6, skills as above, ${G} games, seeds 140000+`);
console.log('| the table votes | seat 0 votes | chips burned to the Fund / game | chips changing hands / game | seat 0 final chips vs truthful baseline (paired, SE) | seat 0 correct calls / game |');
console.log('|---|---|---|---|---|---|');
{
  const skill = SK6;
  const runB = (tableVote: 'truthful' | 'random', me: 'same' | 'contrarian' | 'abstain' | 'herd-against-self') => {
    const out: { chips: number; calls: number; burned: number; moved: number }[] = [];
    for (let i = 0; i < G; i++) {
      const focal = i % 6;
      const r = play({ strategies: mixTable(6, i), seed: 140000 + i, skill, settings: { spectatorBets: 1 },
        votes: (seat, _a, _b, pA, x) => { const truth = pA > 0.5 ? 1 : pA < 0.5 ? 0 : (x() < 0.5 ? 1 : 0); const base = tableVote === 'truthful' ? truth : (x() < 0.5 ? 1 : 0); if (seat !== focal || me === 'same') return base; if (me === 'abstain') return null; return 1 - truth; } });
      out.push({ chips: r.seats[focal].chipsLeft, calls: r.seats[focal].callPts, burned: r.ledger.betsBurned, moved: r.ledger.betsMoved });
    }
    return out;
  };
  for (const tv of ['truthful', 'random'] as const) {
    const base = runB(tv, 'same');
    for (const me of ['same', 'contrarian', 'abstain'] as const) {
      const o = me === 'same' ? base : runB(tv, me);
      const d = stat(o.map((q, i) => q.chips - base[i].chips));
      console.log(`| ${tv} | ${me === 'same' ? 'like the table' : me} | ${mean(o.map((q) => q.burned)).toFixed(2)} | ${mean(o.map((q) => q.moved)).toFixed(2)} | ${d.mean >= 0 ? '+' : ''}${d.mean.toFixed(2)} ± ${d.se.toFixed(2)} | ${mean(o.map((q) => q.calls)).toFixed(2)} |`);
    }
  }
  // no bets, for the chips baseline
  const nb: number[] = []; const wb: number[] = [];
  for (let i = 0; i < G; i++) { nb.push(play({ strategies: mixTable(6, i), seed: 140000 + i, skill, votes: 'truthful' }).ledger.end); wb.push(play({ strategies: mixTable(6, i), seed: 140000 + i, skill, votes: 'truthful', settings: { spectatorBets: 1 } }).ledger.end); }
  console.log(`chips left at the end, table total: bets off ${mean(nb).toFixed(1)}, bets on (truthful) ${mean(wb).toFixed(1)}`);
}
