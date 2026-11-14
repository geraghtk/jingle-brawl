// Scratch copy of web/src/sim/simulate.ts (the repo file is untouched), with extras for the economy review:
// skill-based duels (Tullock ratio), per-turn snapshots, opener order, real-engine duel votes + spectator bets,
// a MODELLED team mini-game (Party points like conveyor rewards), per-seat chip overrides.
// With no extras it draws the same random numbers as the repo simulator (validate.ts checks that).
import * as A from 'C:/Users/Kevin/jingle-brawl/web/src/core/actions';
import * as D from 'C:/Users/Kevin/jingle-brawl/web/src/core/duel';
import { advanceParty, reportWhack, submitSketch, voteSketch } from 'C:/Users/Kevin/jingle-brawl/web/src/core/partyflow';
import { rng as makeRng } from 'C:/Users/Kevin/jingle-brawl/web/src/core/rng';
import { gambitTargets, misfitPile, reprisalTargets, stealTargets, topTiedBidders } from 'C:/Users/Kevin/jingle-brawl/web/src/core/rules';
import { config, minCost } from 'C:/Users/Kevin/jingle-brawl/web/src/core/settings';
import { getPath, applyUpdates } from 'C:/Users/Kevin/jingle-brawl/web/src/core/tx';

type Game = any; type Plan = any; type PlayerId = string;
export interface View2 {
  id: string; chips: number; maxBid: number; baseCost: number; holding: number; hasGift: boolean; progress: number; endgame: boolean;
  rng: () => number; value: (giftId: string) => number;
  n: number; wrappedLeft: number; topBid: number; skill: number;
}
export interface Target { id: string; cost: number }
export interface Strategy2 {
  name: string;
  bid(v: View2, giftId: string): number;
  gambit(v: View2, targets: Target[]): string | null;
  fight(v: View2, giftId: string): boolean;
  reprisal(v: View2, targets: Target[]): string | null;
  steal(v: View2, targets: Target[], misfitId: string): string | null;
}
export interface Opts {
  strategies: Strategy2[]; seed: number; settings?: Record<string, unknown>;
  skill?: number[];                      // duel win prob = s_a / (s_a + s_b)
  chips?: (seat: number) => number | undefined; // endowment override after Start
  votes?: 'none' | 'truthful' | 'random' | ((seat: number, a: number, b: number, pA: number, x: () => number) => number | null);
  teamGame?: number;                     // probability a (non-tiebreak) duel is a team mini-game (MODELLED, not the engine's conveyor)
  teamBase?: number; teamBonus?: number; // conveyor defaults 2 and 3
  teamLoserPoints?: number;              // what-if: points to each member of the losing team
}
export interface Res {
  n: number; seats: { strategy: string; chipsLeft: number; giftValue: number; gotGift: boolean; spent: number; partyPoints: number; callPts: number; teamPts: number; tiePts: number; duelWins: number; duelLosses: number; duelsFought: number; firstOpenTurn: number; firstDuelWon: number | null; keptFirstGift: boolean }[];
  ledger: { start: number; end: number; taxBurned: number; costsBurned: number; dividendsMinted: number; tollsMoved: number; elfBonusMinted: number; betsBurned: number; betsMoved: number; bidsPaid: number; potsPaid: number };
  duels: { kind: string; bid: number; pot: number; tax: number; dividend: number; turn: number }[];
  bids: { turn: number; amount: number; n: number }[];   // every bid decision incl. 0
  snaps: { opened: number; chips: number[]; held: number[] }[]; // after each main-game turn
  value: number[][]; quality: number[];
  gambits: number; reprisals: number; yields: number; steals: number; auctions: number; ties: number; maxNaughty: number; ended: boolean; finalGift: (number | null)[];
}
const NOW = 1_700_000_000_000;
const sum = (g: Game) => Object.values(g.players || {}).reduce((n: number, p: any) => n + (p.chips || 0), 0) as number;
function apply(g: Game, plan: Plan): Game {
  if (plan.error) throw new Error(`refused: ${plan.error}`);
  if (plan.claim && getPath(g, plan.claim.path) != null) return g;
  const claimed = plan.claim ? applyUpdates(g, { [plan.claim.path]: plan.claim.value }, NOW) : g;
  return applyUpdates(claimed, plan.updates, NOW);
}

export function play(o: Opts): Res {
  const r = makeRng(o.seed);
  const x = makeRng(o.seed * 7919 + 13);       // extras only: never disturbs the engine's stream
  const n = o.strategies.length;
  const ctx = () => ({ now: NOW, rng: r });
  const quality: number[] = Array.from({ length: n }, () => 1 + Math.floor(r() * 10));
  const taste: number[][] = Array.from({ length: n }, () => Array.from({ length: n }, () => (r() + r() + r() - 1.5) * 2));
  const valueOf = (seat: number, gi: number) => Math.max(0, Math.min(12, quality[gi] + taste[seat][gi]));
  const skill = o.skill || Array(n).fill(1);

  let g: Game = A.newGame({ hostId: 'h', hostUid: null, settings: { duelPick: 'off', voteWindow: 0, ...(o.settings || {}) } as any, now: NOW });
  const ids = Array.from({ length: n }, (_, i) => `p${i}`);
  for (const id of ids) g = apply(g, A.joinPlayer(g, { playerId: id, name: id.toUpperCase(), uid: null, now: NOW }));
  g = apply(g, A.startGame(g));
  if (o.chips) for (let i = 0; i < n; i++) { const c = o.chips(i); if (c != null) g = applyUpdates(g, { [`players/p${i}/chips`]: c }, NOW); }
  const cfg = config(g.settings);
  const seat = (id: PlayerId) => Number(id.slice(1));
  const gi = (giftId: string) => Number(giftId.slice(1));

  const res: Res = {
    n, seats: [], duels: [], bids: [], snaps: [], value: Array.from({ length: n }, (_, s) => Array.from({ length: n }, (_, k) => valueOf(s, k))), quality,
    gambits: 0, reprisals: 0, yields: 0, steals: 0, auctions: 0, ties: 0, maxNaughty: 0, ended: false, finalGift: [],
    ledger: { start: sum(g), end: 0, taxBurned: 0, costsBurned: 0, dividendsMinted: 0, tollsMoved: 0, elfBonusMinted: 0, betsBurned: 0, betsMoved: 0, bidsPaid: 0, potsPaid: 0 },
  };
  const spent = Array(n).fill(0), teamPts = Array(n).fill(0), fought = Array(n).fill(0);
  const firstOpen = Array(n).fill(-1), firstGift = Array(n).fill(-1), firstDuel: (number | null)[] = Array(n).fill(null);
  let opened = 0;
  let topBid = 0;
  const view = (id: PlayerId): View2 => {
    const me = g.players[id];
    return {
      id, chips: me.chips, maxBid: cfg.maxBid, baseCost: cfg.baseCost, holding: me.giftId ? valueOf(seat(id), gi(me.giftId)) : 0, hasGift: !!me.giftId,
      progress: opened / n, endgame: g.phase === 'endgame', rng: r, value: (giftId: string) => valueOf(seat(id), gi(giftId)),
      n, wrappedLeft: n - opened, topBid, skill: skill[seat(id)],
    };
  };
  const run = (plan: Plan) => {
    g = apply(g, plan);
    if (plan.then === 'drawEndgamePlayer') g = apply(g, D.drawEndgamePlayer(g, r));
    if (plan.then === 'checkEndgameContinue') g = apply(g, D.checkEndgameContinue(g, r));
  };
  const pay = (id: PlayerId, c: number) => { spent[seat(id)] += c; };
  const S = (id: PlayerId) => o.strategies[seat(id)];
  const snap = () => res.snaps.push({ opened, chips: ids.map((id) => g.players[id].chips), held: ids.map((id, i) => (g.players[id].giftId ? valueOf(i, gi(g.players[id].giftId)) : 0)) });

  const resolve = (kind: string, bid: number) => {
    const t = g.currentTurn;
    const a = t.challengerId as PlayerId, b = t.defenderId as PlayerId;
    let pA = skill[seat(a)] / (skill[seat(a)] + skill[seat(b)]);
    // modelled team game: the two duelists captain, everyone else is dealt alternately; team skill = sum of members
    let teams: [number[], number[]] | null = null;
    if (o.teamGame && x() < o.teamGame && n >= 4) {
      const rest = ids.map(seat).filter((s) => s !== seat(a) && s !== seat(b));
      for (let i = rest.length - 1; i > 0; i--) { const j = Math.floor(x() * (i + 1)); [rest[i], rest[j]] = [rest[j], rest[i]]; }
      teams = [[seat(a)], [seat(b)]];
      rest.forEach((s, i) => teams![i % 2].push(s));
      const sk = (m: number[]) => m.reduce((q, s) => q + skill[s], 0) / m.length;
      pA = sk(teams[0]) / (sk(teams[0]) + sk(teams[1]));
    }
    // votes through the real engine (voteOnDuel); spectators stake a chip when spectatorBets is on
    if (o.votes && o.votes !== 'none') {
      for (const id of ids) {
        if (id === a || id === b) continue;
        let pickA: number | null;
        if (o.votes === 'truthful') pickA = pA > 0.5 ? 1 : pA < 0.5 ? 0 : (x() < 0.5 ? 1 : 0);
        else if (o.votes === 'random') pickA = x() < 0.5 ? 1 : 0;
        else pickA = o.votes(seat(id), seat(a), seat(b), pA, x);
        if (pickA == null) continue;
        run(A.voteOnDuel(g, id, pickA ? a : b, NOW));
      }
    }
    const w = r() < pA ? a : b;      // same single draw as the repo's coin
    const loser = w === a ? b : a;
    const loserChips = g.players[loser].chips;
    const before = sum(g);
    const { tax = 0, pot = 0 } = t;
    const chipsBefore = ids.map((id) => g.players[id].chips);
    run(D.resolveDuel(g, w, NOW));
    const dividend = g.lastDuel?.dividend || 0;
    res.ledger.taxBurned += tax; res.ledger.dividendsMinted += dividend; res.ledger.potsPaid += pot;
    let toll = 0;
    if (kind === 'nobids' && loser === a && g.players[loser].chips === loserChips - 1 - 0) { /* toll (bets may also move chips; handled below) */ }
    const bets = g.lastDuel?.bets;
    if (bets) { res.ledger.betsBurned += bets.toBank || 0; res.ledger.betsMoved += bets.pool || 0; }
    // toll: whatever of the supply change is not pot/dividend/bets is zero-sum; detect it directly
    if (t.isNoBidsDuel && loser === a) { const d = g.players[b].chips - chipsBefore[seat(b)]; if (d >= 1 && !bets) toll = 1; else if (bets) { toll = chipsBefore[seat(a)] >= 1 ? 1 : 0; } }
    res.ledger.tollsMoved += toll;
    void before;
    res.duels.push({ kind, bid, pot, tax, dividend, turn: opened });
    fought[seat(a)]++; fought[seat(b)]++;
    if (firstDuel[seat(a)] == null) firstDuel[seat(a)] = w === a ? 1 : 0;
    if (firstDuel[seat(b)] == null) firstDuel[seat(b)] = w === b ? 1 : 0;
    if (teams) {
      const win = teams[0].includes(seat(w)) ? teams[0] : teams[1];
      const lose = win === teams[0] ? teams[1] : teams[0];
      const credit = win.map((s) => skill[s] * (0.5 + x()));
      const top = Math.max(...credit);
      win.forEach((s, i) => { teamPts[s] += (o.teamBase ?? 2) + Math.round(((o.teamBonus ?? 3) * credit[i]) / top); });
      if (o.teamLoserPoints) lose.forEach((s) => { teamPts[s] += o.teamLoserPoints!; });
    }
  };

  let guard = 0;
  while (guard++ < 4000) {
    const t = g.currentTurn;
    for (const gift of Object.values(g.gifts || {}) as any[]) res.maxNaughty = Math.max(res.maxNaughty, gift.naughtyLevel || 0);
    if (g.phase === 'ended') { res.ended = true; break; }
    if (!t) {
      if (g.phase === 'main') {
        if (opened > 0) snap();
        const inBag = ids.filter((id) => !g.players[id].giftId);
        if (opened < n && inBag.length) {
          run(A.drawGift(g, `Gift ${opened}`, `g${opened}`, ctx()));
          const op = seat(g.currentTurn.openerId);
          if (firstOpen[op] < 0) { firstOpen[op] = opened; firstGift[op] = opened; }
          opened++; continue;
        }
        if (inBag.length && misfitPile(g).length) { run(D.startEndgame(g)); continue; }
        run(A.endGame(g)); continue;
      }
      if (g.phase === 'endgame') { run(A.endGame(g)); continue; }
      break;
    }
    switch (t.phase) {
      case 'bidding': case 'endgame_bidding': {
        const gift = t.phase === 'bidding' ? t.giftId : t.misfitId;
        const skip = t.phase === 'bidding' ? t.openerId : t.activePlayerId;
        topBid = 0;
        for (const id of ids) {
          if (id === skip || t.bids?.[id]) continue;
          const max = Math.min(cfg.maxBid, g.players[id].chips);
          const amt = Math.max(0, Math.min(Math.floor(Number(S(id).bid(view(id), gift)) || 0), max));
          run(A.submitBid(g, id, amt, NOW));
          res.bids.push({ turn: opened, amount: amt, n });
        }
        run(t.phase === 'bidding' ? A.revealBids(g) : D.revealAuction(g));
        break;
      }
      case 'reveal': {
        const opener = t.openerId;
        const bids = (Object.entries(t.bids || {}) as [string, any][]).filter(([, b]) => b.amount > 0).sort(([, a], [, b]) => b.amount - a.amount);
        if (!bids.length) {
          const targets = gambitTargets(g, t.giftId, opener).map(([id, q]: any) => ({ id, cost: minCost(cfg, q.naughtyLevel || 0) }));
          const pick = S(opener).gambit(view(opener), targets.filter((q: Target) => q.cost <= g.players[opener].chips));
          if (pick) { const c = minCost(cfg, g.gifts[pick].naughtyLevel || 0); pay(opener, c); res.ledger.costsBurned += c; res.gambits++; run(A.gambit(g, pick)); }
          else run(A.keepGift(g));
          break;
        }
        if (bids.length === 1) { pay(bids[0][0], bids[0][1].amount); res.ledger.bidsPaid += bids[0][1].amount; run(A.startDuel(g, bids[0][0], opener)); break; }
        const tied = topTiedBidders(t);
        topBid = bids[0][1].amount;
        const fight = Boolean(S(opener).fight(view(opener), t.giftId));
        if (tied.length >= 2) { res.ties++; run(fight ? A.openerFight(g, null, null, true, r) : A.openerYield(g, null, null, r)); }
        else run(fight ? A.openerFight(g, bids[0][0], bids[1][0], false, r) : A.openerYield(g, bids[0][0], bids[1][0], r));
        if (!fight) res.yields++;
        break;
      }
      case 'yield_choice': run(A.openerFight(g, t.selectedC1, t.selectedC2, true, r)); break;
      case 'duel_pending': { const b = t.bids?.[t.challengerId]?.amount || 0; pay(t.challengerId, b); res.ledger.bidsPaid += b; run(A.startDuel(g, t.challengerId, t.openerId)); break; }
      case 'tiebreaker_pending': run(A.startTiebreaker(g, t.tiebreakerC1, t.tiebreakerC2, ctx())); break;
      case 'yield_pending': {
        for (const c of [t.yieldC1, t.yieldC2]) { const b = Math.floor((t.bids?.[c]?.amount || 0) / 2); pay(c, b); res.ledger.bidsPaid += b; }
        run(A.startYieldDuel(g, t.yieldC1, t.yieldC2, false, ctx()));
        break;
      }
      case 'duel': {
        if (t.isTiebreak) { const w = r() < 0.5 ? t.challengerId : t.defenderId; pay(w, t.bids?.[w]?.amount || 0); run(A.recordTiebreakWinner(g, w)); break; }
        const kind = t.isGambit ? 'gambit' : t.isReprisal ? 'reprisal' : t.isYieldDuel ? 'yield' : t.isEndgameSteal ? 'steal' : t.isEndgameAuction ? 'auction' : t.isNoBidsDuel ? 'nobids' : 'sealed';
        resolve(kind, t.bids?.[t.challengerId]?.amount || 0);
        break;
      }
      case 'reprisal': {
        const pid = t.reprisalPlayerId;
        const targets = reprisalTargets(g, t.excludedGiftId, pid).map(([id, q]: any) => ({ id, cost: minCost(cfg, q.naughtyLevel || 0) }));
        const pick = S(pid).reprisal(view(pid), targets.filter((q: Target) => q.cost <= g.players[pid].chips));
        if (pick) { const c = minCost(cfg, g.gifts[pick].naughtyLevel || 0); pay(pid, c); res.ledger.costsBurned += c; res.reprisals++; run(D.chooseReprisal(g, pick, NOW)); }
        else run(D.skipReprisal(g));
        break;
      }
      case 'endgame_choice': {
        const pid = t.activePlayerId;
        const targets = stealTargets(g, pid).map(([id, q]: any) => ({ id, cost: minCost(cfg, q.naughtyLevel || 0) })).filter((q: Target) => q.cost <= g.players[pid].chips);
        const pick = S(pid).steal(view(pid), targets, t.misfitId);
        if (pick) { const c = minCost(cfg, g.gifts[pick].naughtyLevel || 0); pay(pid, c); res.ledger.costsBurned += c; res.steals++; run(D.chooseSteal(g)); run(D.executeSteal(g, pick)); }
        else { res.auctions++; run(D.startAuction(g)); }
        break;
      }
      case 'endgame_reveal': {
        const [top] = (Object.entries(t.bids || {}) as [string, any][]).sort(([, a], [, b]) => b.amount - a.amount);
        pay(top[0], top[1].amount); res.ledger.bidsPaid += top[1].amount;
        run(D.auctionDuel(g, top[0]));
        break;
      }
      case 'endgame_nobids': {
        const others = ids.filter((id) => id !== t.activePlayerId);
        run(D.noBidsDuel(g, others[Math.floor(r() * others.length)]));
        break;
      }
      case 'minigame_party': {
        const mg = g.minigame;
        if (mg.phase === 'intro') { run(advanceParty(g, 'intro', NOW)); break; }
        if (mg.phase === 'playing') {
          if (mg.type === 'whack_elf') for (const id of mg.participants) run(reportWhack(g, id, Math.floor(r() * 20), true));
          else for (const id of mg.participants) run(submitSketch(g, id, 'data:sketch'));
        }
        if (mg.phase === 'voting') for (const id of mg.participants) { const others = mg.participants.filter((q: string) => q !== id); run(voteSketch(g, id, others[Math.floor(r() * others.length)])); }
        if (mg.phase === 'playing' || mg.phase === 'voting') {
          const from = g.minigame.phase;
          run(advanceParty(g, from, NOW));
          const after = g.minigame;
          if (after.phase === 'results' && after.results?.bonus && after.prize === 'points_chips') res.ledger.elfBonusMinted += 1;
          break;
        }
        if (mg.phase === 'results') {
          const winners = mg.results?.winners || [];
          if (mg.minigameGoal === 'pick_two') for (const c of winners) { const b = Math.floor((t.bids?.[c]?.amount || 0) / 2); pay(c, b); res.ledger.bidsPaid += b; }
          else if (winners[0]) { const b = t.bids?.[winners[0]]?.amount || 0; pay(winners[0], b); res.ledger.bidsPaid += b; }
          run(advanceParty(g, 'results', NOW));
        }
        break;
      }
      default: throw new Error(`sim2: unhandled turn phase ${t.phase}`);
    }
  }
  res.ledger.end = sum(g);
  res.finalGift = ids.map((id) => (g.players[id].giftId ? gi(g.players[id].giftId) : null));
  res.seats = ids.map((id, i) => {
    const p = g.players[id];
    return {
      strategy: o.strategies[i].name, chipsLeft: p.chips, gotGift: !!p.giftId, spent: spent[i], giftValue: p.giftId ? valueOf(i, gi(p.giftId)) : 0,
      partyPoints: (p.partyPoints || 0) + (p.votingWins || 0) + teamPts[i], callPts: p.votingWins || 0, teamPts: teamPts[i], tiePts: p.partyPoints || 0,
      duelWins: p.duelWins || 0, duelLosses: p.duelLosses || 0, duelsFought: fought[i], firstOpenTurn: firstOpen[i], firstDuelWon: firstDuel[i],
      keptFirstGift: p.giftId ? gi(p.giftId) === firstGift[i] : false,
    };
  });
  return res;
}

// ---- strategies (the repo's, re-typed, + additions) ---------------------------------------------
function bestTarget(v: View2, targets: Target[], costWeight: number, over = v.holding): string | null {
  let best: string | null = null; let bestNet = 0;
  for (const t of targets) { const net = v.value(t.id) - over - costWeight * t.cost; if (net > bestNet) { bestNet = net; best = t.id; } }
  return best;
}
const common = (cw: number) => ({
  fight: (v: View2, giftId: string) => v.value(giftId) >= 5,
  reprisal: (v: View2, targets: Target[]) => bestTarget(v, targets, cw),
  steal: (v: View2, targets: Target[], misfitId: string) => bestTarget(v, targets, cw, v.value(misfitId)),
});
export const aggressive = (k: number): Strategy2 => ({ name: `bid ${k}`, bid: (v, id) => (v.value(id) > v.holding + 0.5 ? Math.min(k, v.maxBid, v.chips) : 0), gambit: (v, t) => bestTarget(v, t, 0.5), ...common(0.5) });
export const passive: Strategy2 = { name: 'passive', bid: () => 0, gambit: (v, t) => bestTarget(v, t, 0.3), ...common(0.3) };
export const valueScaled: Strategy2 = { name: 'value-scaled', bid: (v, id) => Math.max(0, Math.min(Math.round((v.value(id) - v.holding) / 2), v.maxBid, v.chips)), gambit: (v, t) => bestTarget(v, t, 0.5), ...common(0.5) };
export const random: Strategy2 = {
  name: 'random', bid: (v) => (v.rng() < 0.5 ? Math.floor(v.rng() * (v.maxBid + 1)) : 0),
  gambit: (v, t) => (t.length && v.rng() < 0.5 ? t[Math.floor(v.rng() * t.length)].id : null), fight: (v) => v.rng() < 0.5,
  reprisal: (v, t) => (t.length && v.rng() < 0.5 ? t[Math.floor(v.rng() * t.length)].id : null), steal: (v, t) => (t.length && v.rng() < 0.5 ? t[Math.floor(v.rng() * t.length)].id : null),
};
// additions
/** Bids the max, but only on a big upgrade (gain >= g). */
export const snipe = (gmin = 3): Strategy2 => ({ name: `snipe${gmin}`, bid: (v, id) => (v.value(id) - v.holding >= gmin ? Math.min(v.maxBid, v.chips) : 0), gambit: (v, t) => bestTarget(v, t, 0.5), ...common(0.5) });
/** Budget smoothing from sequential-auction theory: bid chips / gifts still to come (spend-down by the end). */
export const spendDown: Strategy2 = { name: 'spend-down', bid: (v, id) => (v.value(id) > v.holding + 0.5 ? Math.max(1, Math.min(v.maxBid, v.chips, Math.ceil(v.chips / (v.wrappedLeft + 1)))) : 0), gambit: (v, t) => bestTarget(v, t, 0), ...common(0) };
/** Never spends a chip (plays for the chip standings). */
export const hoarder: Strategy2 = { name: 'hoarder', bid: () => 0, gambit: () => null, fight: (v, id) => v.value(id) >= 5, reprisal: () => null, steal: () => null };
/** Chips are free: max bid on any gain, direct challenges at any cost. */
export const allIn: Strategy2 = { name: 'all-in', bid: (v, id) => (v.value(id) > v.holding ? Math.min(v.maxBid, v.chips) : 0), gambit: (v, t) => bestTarget(v, t, 0), ...common(0) };
/** Threshold bidder: bid b when gain >= th[b-1] (the one-shot equilibrium shape). */
export const threshold = (th: number[], nm = `thr(${th.join(',')})`): Strategy2 => ({ name: nm, bid: (v, id) => { const gain = v.value(id) - v.holding; let b = 0; th.forEach((q, i) => { if (gain >= q) b = i + 1; }); return Math.min(b, v.maxBid, v.chips); }, gambit: (v, t) => bestTarget(v, t, 0.5), ...common(0.5) });
export const withFight = (s: Strategy2, mode: 'always' | 'never'): Strategy2 => ({ ...s, name: `${s.name}+${mode === 'always' ? 'never-yield' : 'always-yield'}`, fight: () => mode === 'always' });
export const withReprisal = (s: Strategy2, mode: 'never' | 'always'): Strategy2 => ({ ...s, name: `${s.name}+reprisal-${mode}`, reprisal: mode === 'never' ? () => null : (v, t) => bestTarget(v, t, 0) });
export const withGambit = (s: Strategy2, mode: 'never' | 'always'): Strategy2 => ({ ...s, name: `${s.name}+gambit-${mode}`, gambit: mode === 'never' ? () => null : (v, t) => bestTarget(v, t, 0) });

// ---- stats ------------------------------------------------------------------------------------------
export interface Stat { mean: number; se: number; n: number }
export const stat = (xs: number[]): Stat => {
  const n = xs.length; const mean = xs.reduce((a, b) => a + b, 0) / Math.max(1, n);
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n - 1));
  return { mean, se: sd / Math.sqrt(Math.max(1, n)), n };
};
export const f = (s: Stat, d = 2) => `${s.mean >= 0 ? '+' : ''}${s.mean.toFixed(d)}±${s.se.toFixed(d)}`;
export const f0 = (s: Stat, d = 2) => `${s.mean.toFixed(d)}±${s.se.toFixed(d)}`;
export const gini = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); const n = s.length; const tot = s.reduce((a, b) => a + b, 0); if (!tot) return 0; let c = 0; s.forEach((v, i) => { c += (2 * (i + 1) - n - 1) * v; }); return c / (n * tot); };
export const corr = (a: number[], b: number[]) => { const ma = stat(a).mean, mb = stat(b).mean; let sab = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return sab / Math.sqrt(sa * sb || 1); };
export const MIX: Strategy2[] = [aggressive(2), valueScaled, aggressive(3), passive, random, aggressive(1)];
export const mixTable = (n: number, rot: number): Strategy2[] => Array.from({ length: n }, (_, i) => MIX[(i + rot) % MIX.length]);

/** Paired deviation gain: the deviant's payoff minus the mean table player's, same games (SE, not CI). */
export function devGain(table: Strategy2, deviant: Strategy2, o: { n?: number; games?: number; seed?: number; settings?: Record<string, unknown>; w?: number; skill?: number[] } = {}) {
  const n = o.n ?? 6; const d: number[] = []; let first = 0; let chips = 0; let devVal = 0;
  const G = o.games ?? 1000;
  for (let i = 0; i < G; i++) {
    const s = i % n;
    const r = play({ strategies: Array.from({ length: n }, (_, k) => (k === s ? deviant : table)), seed: (o.seed ?? 1) * 100003 + i, settings: o.settings, skill: o.skill });
    const pay = (q: Res['seats'][0]) => q.giftValue + (o.w ?? 0) * q.chipsLeft;
    const others = r.seats.filter((_, k) => k !== s).map(pay);
    d.push(pay(r.seats[s]) - others.reduce((a, b) => a + b, 0) / others.length);
    const mx = Math.max(...r.seats.map((q) => q.chipsLeft));
    const top = r.seats.filter((q) => q.chipsLeft === mx).length;
    if (r.seats[s].chipsLeft === mx) first += 1 / top;
    chips += r.seats[s].chipsLeft; devVal += r.seats[s].giftValue;
  }
  return { gain: stat(d), chipFirst: first / G, chipsLeft: chips / G, value: devVal / G };
}
