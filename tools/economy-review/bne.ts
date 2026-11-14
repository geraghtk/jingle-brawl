// THEORY (no engine): the symmetric Bayes-Nash bid of the one-shot stage game, by best-response iteration on a grid.
// m rivals; each is interested with prob q, gain v ~ U(0,V). Integer bids 0..B. Only the top bidder pays (ties: uniform draw).
// The top bidder then duels: wins with prob p. Chip shadow price lambda (gift points per chip).
//   Jingle Brawl rule: pays b, gets min(b,cap) back on a win  -> expected chip cost (1-p)*b + p*max(0,b-cap)
//   plain first-price:  pays b whatever happens               -> b
// payoff(b,v) = Pwin(b) * (p*v - lambda*cost(b)).
const B = 4, V = 8, GRID = 400;
function solve(o: { m: number; q: number; p: number; lambda: number; cap: number; plain?: boolean }) {
  const cost = (b: number) => (o.plain ? b : (1 - o.p) * b + o.p * Math.max(0, b - o.cap));
  let dist = Array(B + 1).fill(1 / (B + 1)); // a rival's bid distribution
  let bidOf: number[] = [];
  for (let it = 0; it < 400; it++) {
    // P(win | bid b) against m iid rivals with distribution dist, uniform tie-break
    const cum = dist.map((_, b) => dist.slice(0, b).reduce((a, c) => a + c, 0)); // P(rival < b)
    const pwin = dist.map((pb, b) => { let s = 0; let comb = 1; for (let k = 0; k <= o.m; k++) { if (k > 0) comb = (comb * (o.m - k + 1)) / k; s += (comb * pb ** k * cum[b] ** (o.m - k)) / (k + 1); } return b === 0 ? 0 : s; });
    bidOf = []; const nd = Array(B + 1).fill(0);
    for (let i = 0; i < GRID; i++) {
      const v = ((i + 0.5) / GRID) * V; let best = 0, bu = 0;
      for (let b = 1; b <= B; b++) { const u = pwin[b] * (o.p * v - o.lambda * cost(b)); if (u > bu + 1e-12) { bu = u; best = b; } }
      bidOf.push(best); nd[best] += o.q / GRID;
    }
    nd[0] += 1 - o.q;
    dist = dist.map((d, b) => 0.7 * d + 0.3 * nd[b]);
  }
  const th = Array.from({ length: B }, (_, k) => { const i = bidOf.findIndex((b) => b >= k + 1); return i < 0 ? Infinity : ((i + 0.5) / GRID) * V; });
  const pos = bidOf.filter((b) => b > 0);
  // expected top bid and chips destroyed/moved per auction, by sampling the equilibrium
  let top = 0, paid = 0, anyBid = 0, tieAtTop = 0; const N = 40000; let seed = 12345; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let k = 0; k < N; k++) {
    const bids = Array.from({ length: o.m + 1 }, () => (rnd() < o.q ? bidOf[Math.floor(rnd() * GRID)] : 0));
    const t = Math.max(...bids); top += t; if (t > 0) { anyBid++; paid += cost(t); if (bids.filter((b) => b === t).length > 1) tieAtTop++; }
  }
  return { th, meanPosBid: pos.reduce((a, b) => a + b, 0) / Math.max(1, pos.length), dist, top: top / N, contested: anyBid / N, cost: paid / N, tie: tieAtTop / Math.max(1, anyBid) };
}
const show = (label: string, r: ReturnType<typeof solve>) => console.log(`| ${label} | ${r.th.map((t) => (Number.isFinite(t) ? t.toFixed(2) : 'never')).join(' / ')} | ${r.dist.map((d) => (100 * d).toFixed(0)).join('/')} | ${r.top.toFixed(2)} | ${(100 * r.contested).toFixed(0)}% | ${(100 * r.tie).toFixed(0)}% | ${r.cost.toFixed(2)} |`);
console.log('THEORY: one-shot symmetric equilibrium (5 rivals, each interested w.p. 0.5 with gain U(0,8); cap 2; max bid 4)');
console.log('| case | gain needed to bid 1 / 2 / 3 / 4 | a rival\'s bid mix 0/1/2/3/4 % | E[top bid] | contested | tie at the top (given a bid) | expected chips the top bidder loses |');
console.log('|---|---|---|---|---|---|---|');
for (const lambda of [0, 0.05, 0.25, 0.5, 1, 2]) show(`lambda=${lambda}, p=0.5`, solve({ m: 5, q: 0.5, p: 0.5, lambda, cap: 2 }));
for (const p of [0.25, 0.75]) show(`lambda=0.5, p=${p}`, solve({ m: 5, q: 0.5, p, lambda: 0.5, cap: 2 }));
show('lambda=0.5, p=0.5, no cap (full refund on a win)', solve({ m: 5, q: 0.5, p: 0.5, lambda: 0.5, cap: 99 }));
show('lambda=0.5, p=0.5, cap 0 (bid always lost)', solve({ m: 5, q: 0.5, p: 0.5, lambda: 0.5, cap: 0 }));
show('lambda=0.5, p=0.5, plain first-price', solve({ m: 5, q: 0.5, p: 0.5, lambda: 0.5, cap: 2, plain: true }));
for (const m of [2, 3, 7, 11]) show(`lambda=0.5, p=0.5, ${m} rivals`, solve({ m, q: 0.5, p: 0.5, lambda: 0.5, cap: 2 }));
