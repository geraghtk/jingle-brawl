// THE LEARNING AGENT for the spectator-betting study (ROADMAP Step 2c). Self-written, no libraries:
// a tabular contextual bandit (bets-core.ts QTable; 384 contexts x up to 5 actions), epsilon-greedy (0.4 -> 0.02),
// sample-average values (best response) or a step floor of 0.001 (self-play, where the others keep moving).
// Context: own belief strength (4 buckets), whether the record line agrees (2), what my favourite / my underdog would pay
// if I joined now (4 x 4; constant where the pool is sealed), bettors still to act (3). Actions: abstain, back my
// favourite, back my underdog (and the same at stake 2 where allowed).
//   MODE=br : one learner (rotating seat) at an honest table and at a value-betting table.
//   MODE=sp : every seat is an independent learner (its own table).
// Training: TRAIN_G engine games (seeds 500000 + 10000*n + i) replayed with fresh beliefs/orders until BR_DUELS / SP_DUELS
// duels; evaluation, greedy and frozen, on the EVAL games bets-fixed.ts uses (seeds 900000 + 10000*n + i), REPS replays.
// The curve is the greedy policy checked on 500 held-out games (seeds 700000 + 10000*n + i) after each fifth of training.
//   SIZES=6 npx vite-node tools/economy-review/bets-learn.ts > tools/economy-review/out-bets-learn-n6.txt
import { ACTIONS, calib, DESIGNS, evalTable, f, genGames, QTable, SIGMAS, table, train } from './bets-core';
import type { Agent, Eval, Kind } from './bets-core';

const TRAIN_G = Number(process.env.TRAIN_G || 4000), EVAL_G = Number(process.env.GAMES || 3000), REPS = Number(process.env.REPS || 2);
const BR_DUELS = Number(process.env.BR_DUELS || 2e6), SP_DUELS = Number(process.env.SP_DUELS || 3e6);
const SIZES = (process.env.SIZES || '4,6,8,12').split(',').map(Number);
const KEYS = (process.env.DESIGNS || 'A,B,C,D1,D2,D3,D2w,G1,Tc,H,E1').split(',');
const MODES = (process.env.MODE || 'br,sp').split(',');
const HABITS: Kind[] = ['honest', 'value', 'fav', 'contra', 'random', 'valueLast'];
const pc = (x: number) => `${(100 * x).toFixed(0)}%`;

function describe(ev: Eval, live: boolean) {
  const r = ev.rec; const mix = [0, 0, 0, 0, 0];
  for (let b = 0; b < 3; b++) for (let a = 0; a < 5; a++) mix[a] += r.acts[b * 5 + a];
  const tot = Math.max(1, mix.reduce((a, b) => a + b, 0)); const sh = mix.map((x) => x / tot);
  const ah = r.agreeHonest / Math.max(1, r.decisions), av = r.agreeValue / Math.max(1, r.decisions);
  const words = ah >= 0.9 ? 'bets its belief' : sh[0] >= 0.9 ? 'abstains' : live && av >= 0.85 ? 'value rule: bets only when payout x belief beats the stake' : sh[2] + sh[4] >= 0.6 ? 'mostly backs its underdog (contrarian)' : 'mixed';
  const last = [0, 1, 2].map((a) => r.acts[a] / Math.max(1, r.acts[0] + r.acts[1] + r.acts[2] + r.acts[3] + r.acts[4]));
  return `${words}; abstain ${pc(sh[0])} / favourite ${pc(sh[1] + sh[3])} / underdog ${pc(sh[2] + sh[4])}${sh[3] + sh[4] > 0.005 ? ` (stake 2: ${pc(sh[3] + sh[4])})` : ''}; same as honest ${pc(ah)}${live ? `, as value rule ${pc(av)}; acting last: ${pc(last[0])}/${pc(last[1])}/${pc(last[2])}` : ''}`;
}
function grid(ev: Eval) {   // modal action by belief strength x what the favourite pays (summed over the underdog payout)
  const B = ['<.55', '.55-.65', '.65-.75', '>.75'], P = ['<1.2', '1.2-1.6', '1.6-2.2', '>2.2']; const out: string[] = [];
  for (let b0 = 0; b0 < 4; b0++) {
    const cells: string[] = [];
    for (let b2 = 0; b2 < 4; b2++) {
      const m = [0, 0, 0, 0, 0]; for (let b3 = 0; b3 < 4; b3++) for (let a = 0; a < 5; a++) m[a] += ev.rec.pol[((b0 * 4 + b2) * 4 + b3) * 5 + a];
      const tot = m.reduce((a, b) => a + b, 0); const k = m.indexOf(Math.max(...m));
      cells.push(tot < 30 ? `${P[b2]}: (rare)` : `${P[b2]}: ${ACTIONS[k].replace('back my ', '')} ${pc(m[k] / tot)}`);
    }
    out.push(`      belief ${B[b0]} | fav pays ${cells.join(' | ')}`);
  }
  return out.join('\n');
}

for (const n of SIZES) {
  const trainG = genGames(n, TRAIN_G, 500000 + 10000 * n), evalG = genGames(n, EVAL_G, 900000 + 10000 * n), valG = genGames(n, 500, 700000 + 10000 * n);
  const dpg = evalG.reduce((a, g) => a + g.duels.length, 0) / evalG.length;
  console.log(`\n# n=${n} (${n - 2} spectators a duel; ${dpg.toFixed(2)} duels a game). Train ${trainG.length} games, eval ${evalG.length} games x ${REPS}, chips per game ± SE.`);
  if (MODES.includes('br')) {
    console.log(`\n## Best response: one learner, ${BR_DUELS} training duels (it spectates ${(100 * (n - 2) / n).toFixed(0)}% of them), 2 training seeds`);
    console.log('| design | crowd | table | learner (seed 1) | learner (seed 2) | honest in that seat | best fixed habit | learner - honest | learner - best habit | what it learned (seed 1) | curve: greedy profit/game after each fifth |');
    console.log('|---|---|---|---|---|---|---|---|---|---|---|');
    for (const key of KEYS) for (const [sn, sigma] of Object.entries(SIGMAS)) for (const tk of ['honest', 'value'] as Kind[]) {
      const d = DESIGNS[key]; const live = d.seq && d.settle === 'pari';
      if (tk === 'value' && !live) continue;
      const runs = [1, 2].map((ts) => {
        const Q = new QTable(); const at = (_gi: number, fs: number) => table(n, tk, fs, 'learner', Q);
        const curve = train(d, trainG, sigma, at, BR_DUELS, 4242 * ts + n, 0, 5, () => evalTable(d, valG, sigma, at, 1, 77).focal.mean);
        return { curve, ev: evalTable(d, evalG, sigma, at, REPS, 11) };
      });
      const fixed = HABITS.map((h) => ({ h, s: evalTable(d, evalG, sigma, (_gi, fs) => table(n, tk, fs, h), REPS, 11).focal }));
      fixed.push({ h: 'abstain', s: { mean: 0, se: 0, n: 1 } });
      const honest = fixed[0].s; const best = fixed.reduce((a, b) => (b.s.mean > a.s.mean ? b : a));
      const L = runs[0].ev.focal; const diff = (a: typeof L, b: typeof L) => `${a.mean - b.mean >= 0 ? '+' : ''}${(a.mean - b.mean).toFixed(2)}±${Math.sqrt(a.se ** 2 + b.se ** 2).toFixed(2)}`;
      console.log(`| ${key} | ${sn} | ${tk} | ${f(L)} | ${f(runs[1].ev.focal)} | ${f(honest)} | ${best.h} ${f(best.s)} | ${diff(L, honest)} | ${diff(L, best.s)} | ${describe(runs[0].ev, live)} | ${runs[0].curve.map((x) => x.toFixed(2)).join(', ')} |`);
      if (live && n === Number(process.env.GRID_N || 8) && sn === 'sharp') console.log(grid(runs[0].ev));
    }
  }
  if (MODES.includes('sp')) {
    console.log(`\n## Self-play: ${n} independent learners, ${SP_DUELS} training duels`);
    console.log('| design | crowd | bets placed | staked/game | net chips/game (tax burned) | a seat\'s profit/game | MAE pool / line / half | slope | two-sided | best fixed-habit deviation (habit profit - learner profit in that seat) | all deviations | what the focal learner does | curve: share of chances bet after each fifth |');
    console.log('|---|---|---|---|---|---|---|---|---|---|---|---|---|');
    for (const key of KEYS) for (const [sn, sigma] of Object.entries(SIGMAS)) {
      const d = DESIGNS[key]; const live = d.seq && d.settle === 'pari';
      const Qs = Array.from({ length: n }, () => new QTable());
      const all = (): Agent[] => Qs.map((Q) => ({ kind: 'learner' as Kind, Q }));
      const curve = train(d, trainG, sigma, all, SP_DUELS, 9191 + n, 0.001, 5, () => evalTable(d, valG, sigma, all, 1, 77).part);
      const ev = evalTable(d, evalG, sigma, all, REPS, 11); const c = calib(ev.rec);
      const devs = [...HABITS, 'abstain' as Kind].map((h) => { const s = evalTable(d, evalG, sigma, (_gi, fs) => { const t = all(); t[fs] = { kind: h }; return t; }, REPS, 11).focal; return { h, gain: s.mean - ev.focal.mean, se: Math.sqrt(s.se ** 2 + ev.focal.se ** 2) }; });
      const best = devs.reduce((a, b) => (b.gain > a.gain ? b : a));
      console.log(`| ${key} | ${sn} | ${pc(ev.part)} | ${ev.staked.mean.toFixed(1)} | ${f(ev.net)} (${ev.tax.mean.toFixed(1)}) | ${f(ev.focal)} | ${c.maePool.toFixed(3)} / ${c.maeLine.toFixed(3)} / ${c.maeHalf.toFixed(3)} | ${c.slope.toFixed(2)} | ${pc(c.twoSided)} | ${best.h} ${best.gain >= 0 ? '+' : ''}${best.gain.toFixed(2)}±${best.se.toFixed(2)} | ${devs.map((x) => `${x.h} ${x.gain >= 0 ? '+' : ''}${x.gain.toFixed(2)}`).join(', ')} | ${describe(ev, live)} | ${curve.map(pc).join(', ')} |`);
    }
  }
}
console.log('\nDesigns:'); for (const k of KEYS) console.log(`  ${k}: ${DESIGNS[k].name}`);
