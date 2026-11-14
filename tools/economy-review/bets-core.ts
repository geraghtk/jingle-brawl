// Spectator-betting study (ROADMAP Step 2c): a betting layer replayed over real-engine duel sequences.
//   - genGames(): sim3 games (the real rules engine, mixed bidding styles, votes off) -> per game the list of duels
//     (who, true win probability from the skills, winner, every seat's chips and the record-based opening line).
//   - Designs A..H (DESIGNS below): settlement written here (only A exists in the engine: scoring.ts settleBets, mirrored below).
//   - Fixed-habit bettors and a tabular epsilon-greedy contextual-bandit learner (QTable), self-written.
// LIMITATION: betting chips do not feed back into the engine's bidding. A seat's betting purse at a duel is
// (its engine chips at that duel + its betting profit so far this game), floored at 0, so nobody stakes chips they lack.
import { mixTable, play } from './sim3';

export const rng32 = (seed: number) => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
export const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

// ---- modelling assumptions -------------------------------------------------------------------------
export const SKILL_SD = 0.5;                 // a seat's duel skill = exp(0.5 * N(0,1)), drawn per game; P(a beats b) = sa/(sa+sb)
export const TAU2 = 2 * SKILL_SD * SKILL_SD; // so logit(p) ~ N(0, TAU2)
// A spectator's belief: private signal logit(p) + sigma*N(0,1), shrunk to the prior (posterior mean in logit space).
export const SIGMAS: Record<string, number> = { sharp: 0.3, noisy: 1.0 };

export interface Duel { a: number; b: number; p: number; aWon: boolean; chips: number[]; line: number; rec: number[]; tax: number }
export interface GameSeq { n: number; duels: Duel[]; tax: number }

/** Opening line from the two records so far this game: shrunk win rate (w+K)/(d+2K) each, then a ratio. No record = 0.5. */
export const LINE_K = Number(process.env.LINE_K || 3);   // shrinkage: each record counts as if it also held K wins and K losses (bets-line.ts picks K)
export const lineOf = (wa: number, la: number, wb: number, lb: number, K = LINE_K) => { const ra = (wa + K) / (wa + la + 2 * K), rb = (wb + K) / (wb + lb + 2 * K); return ra / (ra + rb); };

export function genGames(n: number, count: number, seed0: number): GameSeq[] {
  const out: GameSeq[] = [];
  for (let i = 0; i < count; i++) {
    const r = rng32(seed0 + i);
    const skill = Array.from({ length: n }, () => Math.exp(SKILL_SD * gauss(r)));
    const duels: Duel[] = [];
    try {
      const res = play({ strategies: mixTable(n, i), seed: seed0 + i, skill, onDuel: (d) => duels.push({ a: d.a, b: d.b, p: d.pA, aWon: d.aWon, chips: d.chips, line: lineOf(d.wins[d.a], d.losses[d.a], d.wins[d.b], d.losses[d.b]), rec: [d.wins[d.a], d.losses[d.a], d.wins[d.b], d.losses[d.b]], tax: d.tax }) });
      out.push({ n, duels, tax: res.ledger.taxBurned });
    } catch { /* a refused plan: skip the game */ }
  }
  return out;
}

// ---- designs -----------------------------------------------------------------------------------------
export interface Design {
  name: string;
  settle: 'A' | 'pari';
  seq: boolean;            // bettors act in a random order and see the pool so far
  revise?: boolean;        // a second pass: everyone may change their bet seeing the whole pool
  seed: number;            // chips from the Fund per duel
  seedMode: 'none' | 'house' | 'dead' | 'tax';   // tax: this duel's own tax goes into the pool as bonus chips (dead); house: the seed is the house's own bet at its line (its winning share returns to the Fund);
                                         // dead: bonus chips in the pool, paid to the winning backers
  free?: boolean;          // calls cost nothing (the prize is the seed)
  maxW: number;            // largest stake
  taxBonus?: boolean;      // with a house seed: that duel's tax also goes into the pool as bonus chips
  cap?: number;            // most a winning chip can collect (incl. itself); the excess rolls over into the next duel's pool
}
export const DESIGNS: Record<string, Design> = {
  A: { name: 'A today (sealed, losers pay winners, odd chip burned)', settle: 'A', seq: false, seed: 0, seedMode: 'none', maxW: 1 },
  B: { name: 'B pool, sealed', settle: 'pari', seq: false, seed: 0, seedMode: 'none', maxW: 1 },
  C: { name: 'C pool, live odds', settle: 'pari', seq: true, seed: 0, seedMode: 'none', maxW: 1 },
  Cr: { name: 'C + change allowed', settle: 'pari', seq: true, revise: true, seed: 0, seedMode: 'none', maxW: 1 },
  D1: { name: 'D house line, S=1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1 },
  D2: { name: 'D house line, S=2', settle: 'pari', seq: true, seed: 2, seedMode: 'house', maxW: 1 },
  D3: { name: 'D house line, S=3', settle: 'pari', seq: true, seed: 3, seedMode: 'house', maxW: 1 },
  D2r: { name: 'D S=2 + change allowed', settle: 'pari', seq: true, revise: true, seed: 2, seedMode: 'house', maxW: 1 },
  D2w: { name: 'D S=2, stake 1 or 2', settle: 'pari', seq: true, seed: 2, seedMode: 'house', maxW: 2 },
  E1: { name: 'E1 live pool + 1 bonus chip (no line)', settle: 'pari', seq: true, seed: 1, seedMode: 'dead', maxW: 1 },
  E2: { name: 'E2 live pool + 2 bonus chips', settle: 'pari', seq: true, seed: 2, seedMode: 'dead', maxW: 1 },
  Cc: { name: 'C + payout cap 3 for 1 (excess rolls over)', settle: 'pari', seq: true, seed: 0, seedMode: 'none', maxW: 1, cap: 3 },
  G1: { name: 'G1 house line S=1 + cap 3 for 1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, cap: 3 },
  G2: { name: 'G2 house line S=2 + cap 3 for 1', settle: 'pari', seq: true, seed: 2, seedMode: 'house', maxW: 1, cap: 3 },
  T: { name: 'T live pool + that duel tax as bonus chips', settle: 'pari', seq: true, seed: 0, seedMode: 'tax', maxW: 1 },
  Tc: { name: 'T + cap 3 for 1', settle: 'pari', seq: true, seed: 0, seedMode: 'tax', maxW: 1, cap: 3 },
  H: { name: 'H house line S=1 + that duel tax as bonus + cap 3 for 1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, cap: 3, taxBonus: true },
  H0: { name: 'H without the cap', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, taxBonus: true },
  H2: { name: 'H with cap 2 for 1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, cap: 2, taxBonus: true },
  H4: { name: 'H with cap 4 for 1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, cap: 4, taxBonus: true },
  H5: { name: 'H with cap 5 for 1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, cap: 5, taxBonus: true },
  Hr: { name: 'H + change allowed', settle: 'pari', seq: true, revise: true, seed: 1, seedMode: 'house', maxW: 1, cap: 3, taxBonus: true },
  Hw: { name: 'H, stake 1 or 2', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 2, cap: 3, taxBonus: true },
  G4: { name: 'G house line S=1 + cap 4 for 1', settle: 'pari', seq: true, seed: 1, seedMode: 'house', maxW: 1, cap: 4 },
  Tc4: { name: 'T + cap 4 for 1', settle: 'pari', seq: true, seed: 0, seedMode: 'tax', maxW: 1, cap: 4 },
  F1: { name: 'F free call, 1 Fund chip split by right callers', settle: 'pari', seq: true, seed: 1, seedMode: 'dead', free: true, maxW: 1 },
  F2: { name: 'F free call, 2 Fund chips', settle: 'pari', seq: true, seed: 2, seedMode: 'dead', free: true, maxW: 1 },
};

// ---- bettors -----------------------------------------------------------------------------------------
export type Kind = 'honest' | 'value' | 'fav' | 'contra' | 'abstain' | 'random' | 'alwaysA' | 'alwaysB' | 'valueLast' | 'learner';
export const NS = 384, NA = 5;   // belief 4 x line 2 x favourite payout 4 x underdog payout 4 x bettors still to act 3
export class QTable { q = new Float64Array(NS * NA); c = new Float64Array(NS * NA); }
export interface Agent { kind: Kind; Q?: QTable }
export const ACTIONS = ['abstain', 'back my favourite', 'back my underdog', 'favourite x2', 'underdog x2'];

export interface Rec {
  focal: number;
  // calibration of the closing pool (implied P(A) = weight on A / weight on both, incl. the house's seed)
  duels: number; priced: number; twoSided: number; absPool: number; absLine: number; absHalf: number; sx: number; sy: number; sxy: number; sxx: number;
  // the focal learner's choices
  pol: Float64Array;       // [belief bucket 4][fav payout bucket 4][dog payout bucket 4][action 5]
  acts: Float64Array;      // [toAct bucket 3][action 5]
  agreeHonest: number; agreeValue: number; decisions: number;
}
export const newRec = (): Rec => ({ focal: -1, duels: 0, priced: 0, twoSided: 0, absPool: 0, absLine: 0, absHalf: 0, sx: 0, sy: 0, sxy: 0, sxx: 0, pol: new Float64Array(320), acts: new Float64Array(15), agreeHonest: 0, agreeValue: 0, decisions: 0 });

export interface GameOut { delta: number[]; staked: number; bets: number; opps: number }

export function runGame(d: Design, gm: GameSeq, agents: Agent[], sigma: number, r: () => number, learn: { eps: number; alphaMin: number } | null, rec: Rec | null): GameOut {
  const n = gm.n; const k = TAU2 / (TAU2 + sigma * sigma);
  const cum: number[] = new Array(n).fill(0); let carry = 0; let staked = 0, nb = 0, opps = 0;
  const cost = d.free ? 0 : 1; const house = d.seedMode === 'house'; const cap = d.cap || 1e9; const taxBonus = d.seedMode === 'tax' || !!d.taxBonus;
  const side = new Int8Array(n), wt = new Int8Array(n); const q = new Float64Array(n);
  const order: number[] = []; const visits: number[] = [];
  for (const du of gm.duels) {
    order.length = 0; visits.length = 0;
    for (let s = 0; s < n; s++) { side[s] = 0; wt[s] = 0; if (s !== du.a && s !== du.b) order.push(s); }
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
    for (let i = 0; i < order.length - 1; i++) if (agents[order[i]].kind === 'valueLast') { const t = order[i]; order.splice(i, 1); order.push(t); break; }
    const lp = Math.log(du.p / (1 - du.p));
    const S = (d.seedMode === 'none' || d.seedMode === 'tax' ? 0 : d.seed) + (taxBonus ? du.tax : 0);
    const hA = house ? d.seed * du.line : 0, hB = house ? d.seed * (1 - du.line) : 0;
    let WA = hA, WB = hB, C = 0;
    const passes = d.revise ? 2 : 1;
    for (let pass = 0; pass < passes; pass++) for (let j = 0; j < order.length; j++) {
      const s = order[j]; const ag = agents[s];
      if (pass === 0) { q[s] = 1 / (1 + Math.exp(-k * (lp + sigma * gauss(r)))); opps++; }
      else if (side[s]) { if (side[s] > 0) WA -= wt[s]; else WB -= wt[s]; C -= cost * wt[s]; side[s] = 0; wt[s] = 0; }
      const avail = d.free ? 99 : Math.max(0, du.chips[s] + cum[s]);
      if (avail < 1) continue;
      const vA = d.seq ? WA : hA, vB = d.seq ? WB : hB, T = S + carry + (d.seq ? C : 0);
      const toAct = order.length - 1 - j + (passes === 2 && pass === 0 ? order.length : 0);
      const qa = q[s]; const m = qa >= 0.5 ? 1 : -1;
      const pay = (sd: number, w: number) => Math.min(cap * w, (T + cost * w) * w / ((sd > 0 ? vA : vB) + w));
      const valueRule = (): number => {
        const evA = qa * pay(1, 1) - cost, evB = (1 - qa) * pay(-1, 1) - cost;
        if (evA > 1e-9 && evA >= evB) return 1; if (evB > 1e-9) return -1; return 0;
      };
      let sd = 0, w = 1;
      switch (ag.kind) {
        case 'honest': sd = m; break;
        case 'value': case 'valueLast': sd = d.seq && d.settle === 'pari' ? valueRule() : m; break;
        case 'fav': sd = du.line > 0.5 ? 1 : du.line < 0.5 ? -1 : (r() < 0.5 ? 1 : -1); break;
        case 'contra': sd = d.seq ? (vA < vB ? 1 : vB < vA ? -1 : -m) : -m; break;
        case 'random': sd = r() < 0.5 ? 1 : -1; break;
        case 'alwaysA': sd = 1; break;
        case 'alwaysB': sd = -1; break;
        case 'abstain': break;
        case 'learner': {
          const qs = m > 0 ? qa : 1 - qa; const Lf = m > 0 ? du.line : 1 - du.line;
          const b0 = qs < 0.55 ? 0 : qs < 0.65 ? 1 : qs < 0.75 ? 2 : 3;
          const b1 = Lf < 0.5 ? 0 : 1;
          let b2 = 0, b3 = 0, b4 = 0;
          if (d.settle === 'pari') {
            if (d.free) { const wf = m > 0 ? vA : vB, wd = m > 0 ? vB : vA; b2 = wf < 0.5 ? 3 : wf < 1.5 ? 2 : wf < 2.5 ? 1 : 0; b3 = wd < 0.5 ? 3 : wd < 1.5 ? 2 : wd < 2.5 ? 1 : 0; }
            else { const pf = pay(m, 1), pd = pay(-m, 1); b2 = pf < 1.2 ? 0 : pf < 1.6 ? 1 : pf < 2.2 ? 2 : 3; b3 = pd < 1.8 ? 0 : pd < 2.6 ? 1 : pd < 4 ? 2 : 3; }
          }
          if (d.seq) b4 = toAct === 0 ? 0 : toAct <= 2 ? 1 : 2;
          const st = (((b0 * 2 + b1) * 4 + b2) * 4 + b3) * 3 + b4;
          const na = d.maxW >= 2 && avail >= 2 ? 5 : 3;
          let a = 0;
          if (learn && r() < learn.eps) a = Math.floor(r() * na);
          else { let best = ag.Q!.q[st * NA]; for (let i = 1; i < na; i++) if (ag.Q!.q[st * NA + i] > best + 1e-12) { best = ag.Q!.q[st * NA + i]; a = i; } }
          if (learn) visits.push(s, st * NA + a);
          sd = a === 0 ? 0 : (a === 1 || a === 3) ? m : -m; w = a >= 3 ? 2 : 1;
          if (rec && s === rec.focal && pass === passes - 1) {
            rec.pol[((b0 * 4 + b2) * 4 + b3) * 5 + a]++; rec.acts[b4 * 5 + a]++; rec.decisions++;
            if (sd === m) rec.agreeHonest++;
            if (sd === (d.seq && d.settle === 'pari' ? valueRule() : m)) rec.agreeValue++;
          }
          break;
        }
      }
      if (sd) { side[s] = sd; wt[s] = w; if (sd > 0) WA += w; else WB += w; C += cost * w; }
    }
    // ---- settle ----
    const ws = du.aWon ? 1 : -1;
    const delta: number[] = new Array(n).fill(0);
    let nBets = 0, winW = 0, loseN = 0;
    for (const s of order) if (side[s]) { nBets++; staked += cost * wt[s]; if (side[s] === ws) winW += wt[s]; else loseN++; }
    nb += nBets;
    if (d.settle === 'A') {
      // scoring.ts settleBets: wrong callers lose 1; split equally (integer) among right callers; the rest is burned
      const right = nBets - loseN; const share = right ? Math.floor(loseN / right) : 0;
      for (const s of order) if (side[s]) delta[s] = side[s] === ws ? share : -1;
    } else if (nBets && winW > 0) {
      const T = S + carry + C; const Wwin = winW + (ws > 0 ? hA : hB);
      let paid = 0;
      for (const s of order) if (side[s]) {
        if (side[s] === ws) { const p = Math.min(cap * wt[s], Math.floor(T * wt[s] / Wwin + 1e-9)); paid += p; delta[s] = p - cost * wt[s]; }
        else delta[s] = -cost * wt[s];
      }
      const housePay = house ? Math.floor(Math.min(cap, T / Wwin) * (ws > 0 ? hA : hB) + 1e-9) : 0;   // back to the Fund, whole chips
      carry = T - paid - housePay;                                                       // remainder waits in the next pool
    } // else: nobody (real) backed the winner: stakes returned, the seed goes back to the Fund, carry stays
    for (const s of order) cum[s] += delta[s];
    if (learn) for (let i = 0; i < visits.length; i += 2) {
      const ag = agents[visits[i]]; const ix = visits[i + 1]; const c = ++ag.Q!.c[ix];
      ag.Q!.q[ix] += Math.max(1 / c, learn.alphaMin) * (delta[visits[i]] - ag.Q!.q[ix]);
    }
    if (rec) {
      rec.duels++;
      if (WA + WB > 0) {
        const imp = WA / (WA + WB); rec.priced++;
        let ra = 0, rb = 0; for (const s of order) { if (side[s] > 0) ra++; else if (side[s] < 0) rb++; }
        if (ra && rb) rec.twoSided++;
        rec.absPool += Math.abs(imp - du.p); rec.absLine += Math.abs(du.line - du.p); rec.absHalf += Math.abs(0.5 - du.p);
        rec.sx += du.p; rec.sy += imp; rec.sxy += du.p * imp; rec.sxx += du.p * du.p;
      }
    }
  }
  return { delta: cum, staked, bets: nb, opps };   // carry left at the end of the game goes to the Fund (it is in no seat's delta)
}

// ---- evaluation / training ---------------------------------------------------------------------------
export interface Stat { mean: number; se: number; n: number }
export const stat = (xs: number[]): Stat => {
  const n = xs.length; const mean = xs.reduce((a, b) => a + b, 0) / Math.max(1, n);
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n - 1));
  return { mean, se: sd / Math.sqrt(Math.max(1, n)), n };
};
export const f = (s: Stat, d = 2) => `${s.mean >= 0 ? '+' : ''}${s.mean.toFixed(d)}±${s.se.toFixed(d)}`;

export interface Eval { focal: Stat; others: Stat; pair: Stat; net: Stat; staked: Stat; tax: Stat; part: number; rec: Rec; duelsPerGame: number }
/** Replays every game `reps` times with fresh beliefs and orders. focal seat = game index % n; pair = focal + the next seat. */
export function evalTable(d: Design, games: GameSeq[], sigma: number, agentsFor: (gi: number, focal: number) => Agent[], reps: number, seed: number): Eval {
  const r = rng32(seed); const rec = newRec();
  const focal: number[] = [], others: number[] = [], pair: number[] = [], net: number[] = [], staked: number[] = [], tax: number[] = [];
  let bets = 0, opps = 0, duels = 0;
  for (let rep = 0; rep < reps; rep++) games.forEach((gm, gi) => {
    const fs = gi % gm.n; rec.focal = fs;
    const o = runGame(d, gm, agentsFor(gi, fs), sigma, r, null, rec);
    const tot = o.delta.reduce((a, b) => a + b, 0);
    focal.push(o.delta[fs]); pair.push(o.delta[fs] + o.delta[(fs + 1) % gm.n]); others.push((tot - o.delta[fs]) / (gm.n - 1));
    net.push(tot); staked.push(o.staked); tax.push(gm.tax); bets += o.bets; opps += o.opps; duels += gm.duels.length;
  });
  return { focal: stat(focal), others: stat(others), pair: stat(pair), net: stat(net), staked: stat(staked), tax: stat(tax), part: bets / Math.max(1, opps), rec, duelsPerGame: duels / Math.max(1, games.length * reps) };
}
export const calib = (rec: Rec) => {
  const n = Math.max(1, rec.priced); const mx = rec.sx / n, my = rec.sy / n;
  const slope = (rec.sxy / n - mx * my) / Math.max(1e-12, rec.sxx / n - mx * mx);
  return { priced: rec.priced / Math.max(1, rec.duels), twoSided: rec.twoSided / Math.max(1, rec.duels), maePool: rec.absPool / n, maeLine: rec.absLine / n, maeHalf: rec.absHalf / n, slope };
};

/** Trains whatever learners `agentsFor` seats (epsilon-greedy, epsilon 0.4 -> 0.02 over the first 80%), `duels` duels in all.
 *  After each of `blocks` blocks calls `check()` (a greedy evaluation) for the learning curve. */
export function train(d: Design, games: GameSeq[], sigma: number, agentsFor: (gi: number, focal: number) => Agent[], duels: number, seed: number, alphaMin: number, blocks: number, check: () => number): number[] {
  const r = rng32(seed); const curve: number[] = []; let done = 0, gi = 0, next = duels / blocks;
  while (done < duels) {
    const gm = games[gi % games.length]; const eps = Math.max(0.02, 0.4 * (1 - done / (0.8 * duels)));
    runGame(d, gm, agentsFor(gi, gi % gm.n), sigma, r, { eps, alphaMin }, null);
    done += gm.duels.length; gi++;
    if (done >= next) { curve.push(check()); next += duels / blocks; }
  }
  return curve;
}
export const table = (n: number, kind: Kind, focal: number, fk: Kind, Q?: QTable): Agent[] => Array.from({ length: n }, (_, s) => (s === focal ? { kind: fk, Q } : { kind }));
