// Plays whole games in memory through the real rules engine (the same plans the apps apply), with
// pluggable strategies, so economic claims can be measured instead of argued: chip flows, who
// ends up with what, whether a strategy is a best response. Duels are fair coin flips (real duels
// are rock-paper-scissors, thumb wars…). Thousands of games run in seconds.
import * as A from '../core/actions';
import * as D from '../core/duel';
import type { PartyGame } from '../core/party';
import { advanceParty, reportWhack, submitSketch, voteSketch } from '../core/partyflow';
import { rng as makeRng } from '../core/rng';
import type { Rng } from '../core/rng';
import { gambitTargets, misfitPile, reprisalTargets, stealTargets, topTiedBidders, wrappedGifts } from '../core/rules';
import { config, minCost } from '../core/settings';
import { getPath, applyUpdates } from '../core/tx';
import type { Game, Plan, PlayerId, Settings } from '../core/types';
import type { Strategy, Target, View } from './strategies';

export interface Ledger {
  start: number;          // chips in play after Start
  end: number;
  taxBurned: number;      // Santa Tax (destroyed: there is no stored bank)
  costsBurned: number;    // Gambit / Reprisal / Endgame-steal costs (paid to nobody)
  dividendsMinted: number;// Loser's Dividends (created from nothing)
  tollsMoved: number;     // Misfit toll: player → player (zero-sum)
  elfBonusMinted: number; // party games: +1 chip to the best finisher outside the tie
}

export interface GameResult {
  seats: { id: PlayerId; strategy: string; chipsLeft: number; giftValue: number; gotGift: boolean; spent: number }[];
  ledger: Ledger;
  duels: { kind: string; bid: number; pot: number; tax: number; dividend: number }[];
  bidsPlaced: number;
  bidsByAmount: Record<number, number>;
  gambits: number;
  reprisals: number;
  yields: number;
  steals: number;
  auctions: number;
  maxNaughty: number;
  steps: number;
  ended: boolean;
  partyGames: Record<string, number>; // ties settled by each party game
}

const NOW = 1_700_000_000_000;
const sum = (g: Game) => Object.values(g.players || {}).reduce((n, p) => n + (p.chips || 0), 0);

/** Apply a plan like the room server: refused → throw; a lost claim → no change. */
function apply(g: Game, plan: Plan): Game {
  if (plan.error) throw new Error(`refused: ${plan.error}`);
  if (plan.claim && getPath(g, plan.claim.path) != null) return g;
  const claimed = plan.claim ? applyUpdates(g, { [plan.claim.path]: plan.claim.value }, NOW) : g;
  return applyUpdates(claimed, plan.updates, NOW);
}

export interface SimOptions {
  settings?: Settings;
  strategies: Strategy[];   // one per seat
  seed: number;
  maxSteps?: number;
}

/** A choice the game is waiting on: whose it is, what they see, and what they may choose from.
 *  `game` is the state at that moment (for traces); `view` is what the strategy reads. */
export type Decision =
  | { kind: 'bid'; player: PlayerId; giftId: string; max: number; view: View; game: Game }
  | { kind: 'gambit'; player: PlayerId; targets: Target[]; view: View; game: Game }
  | { kind: 'fight'; player: PlayerId; giftId: string; view: View; game: Game }
  | { kind: 'reprisal'; player: PlayerId; targets: Target[]; view: View; game: Game }
  | { kind: 'steal'; player: PlayerId; targets: Target[]; misfitId: string; view: View; game: Game };

export interface StepOptions {
  settings?: Settings;
  names: string[];          // one per seat (seat i is player "p<i>")
  seed: number;
  maxSteps?: number;
}

/** What a strategy decides at a decision. */
export function answer(s: Strategy, d: Decision): number | string | boolean | null {
  switch (d.kind) {
    case 'bid': return s.bid(d.view, d.giftId);
    case 'gambit': return s.gambit(d.view, d.targets);
    case 'fight': return s.fight(d.view, d.giftId);
    case 'reprisal': return s.reprisal(d.view, d.targets);
    case 'steal': return s.steal(d.view, d.targets, d.misfitId);
  }
}

/** One game, played by strategies (each seat's answers its own decisions). */
export function playGame(o: SimOptions): GameResult {
  const it = steps({ settings: o.settings, names: o.strategies.map((s) => s.name), seed: o.seed, maxSteps: o.maxSteps });
  let s = it.next();
  while (!s.done) s = it.next(answer(o.strategies[Number(s.value.player.slice(1))], s.value));
  return s.value;
}

/** One game as a sequence of decisions: it pauses at every choice a player makes and resumes
 *  with their answer (playGame answers from strategies; sim/bridge.ts from outside). The same
 *  code and the same random draws either way. */
export function* steps(o: StepOptions): Generator<Decision, GameResult, unknown> {
  const r: Rng = makeRng(o.seed);
  const n = o.names.length;
  const ctx = () => ({ now: NOW, rng: r });
  // valuations: a shared quality per gift + each player's own taste (as in the agent harness)
  const quality: number[] = Array.from({ length: n }, () => 1 + Math.floor(r() * 10));
  const taste: number[][] = Array.from({ length: n }, () => Array.from({ length: n }, () => (r() + r() + r() - 1.5) * 2));
  const valueOf = (seat: number, giftIdx: number) => Math.max(0, Math.min(12, quality[giftIdx] + taste[seat][giftIdx]));

  let g: Game = A.newGame({ hostId: 'h', hostUid: null, settings: { duelPick: 'off', voteWindow: 0, ...o.settings }, now: NOW });
  const ids = Array.from({ length: n }, (_, i) => `p${i}`);
  for (const id of ids) g = apply(g, A.joinPlayer(g, { playerId: id, name: id.toUpperCase(), uid: null, now: NOW }));
  g = apply(g, A.startGame(g));
  const cfg = config(g.settings);
  const seat = (id: PlayerId) => ids.indexOf(id);
  const giftIdx = (giftId: string) => Number(giftId.slice(1)); // "g3" → 3

  const res: GameResult = {
    seats: [], duels: [], bidsPlaced: 0, bidsByAmount: {}, gambits: 0, reprisals: 0, yields: 0, steals: 0, auctions: 0,
    maxNaughty: 0, steps: 0, ended: false, partyGames: {},
    ledger: { start: sum(g), end: 0, taxBurned: 0, costsBurned: 0, dividendsMinted: 0, tollsMoved: 0, elfBonusMinted: 0 },
  };
  const spent: Record<PlayerId, number> = Object.fromEntries(ids.map((id) => [id, 0]));
  let opened = 0;

  const view = (id: PlayerId): View => {
    const me = g.players![id];
    const held = me.giftId ? valueOf(seat(id), giftIdx(me.giftId)) : 0;
    return {
      id, chips: me.chips, maxBid: cfg.maxBid, baseCost: cfg.baseCost, holding: held, hasGift: !!me.giftId,
      progress: opened / n, endgame: g.phase === 'endgame', rng: r,
      value: (giftId: string) => valueOf(seat(id), giftIdx(giftId)),
    };
  };
  const run = (plan: Plan) => {
    g = apply(g, plan);
    if (plan.then === 'drawEndgamePlayer') g = apply(g, D.drawEndgamePlayer(g, r));
    if (plan.then === 'checkEndgameContinue') g = apply(g, D.checkEndgameContinue(g, r));
  };
  const coin = (a: PlayerId, b: PlayerId) => (r() < 0.5 ? a : b);
  const resolve = (winner: PlayerId, kind: string, bid: number) => {
    const t = g.currentTurn!;
    const before = sum(g);
    const loser = winner === t.challengerId ? t.defenderId! : t.challengerId!;
    const loserChips = g.players![loser].chips;
    const { tax = 0, pot = 0 } = t;
    run(D.resolveDuel(g, winner, NOW));
    const dividend = g.lastDuel?.dividend || 0;
    res.ledger.taxBurned += tax;
    res.ledger.dividendsMinted += dividend;
    if (t.isNoBidsDuel && loser === t.challengerId && g.players![loser].chips === loserChips - 1) res.ledger.tollsMoved += 1;
    res.duels.push({ kind, bid, pot, tax, dividend });
    void before;
  };
  const pay = (id: PlayerId, cost: number) => { spent[id] += cost; };

  const maxSteps = o.maxSteps ?? 4000;
  while (res.steps++ < maxSteps) {
    const t = g.currentTurn;
    for (const gift of Object.values(g.gifts || {})) res.maxNaughty = Math.max(res.maxNaughty, gift.naughtyLevel || 0);
    if (g.phase === 'ended') { res.ended = true; break; }

    if (!t) {
      if (g.phase === 'main') {
        const inBag = ids.filter((id) => !g.players![id].giftId);
        if (opened < n && inBag.length) { run(A.drawGift(g, `Gift ${opened}`, `g${opened}`, ctx())); opened++; continue; }
        if (inBag.length && misfitPile(g).length) { run(D.startEndgame(g)); continue; }
        run(A.endGame(g)); continue;
      }
      if (g.phase === 'endgame') { run(A.endGame(g)); continue; } // no misfits left for the giftless
      break;
    }

    switch (t.phase) {
      case 'bidding': {
        const gift = t.giftId!;
        for (const id of ids) {
          if (id === t.openerId || t.bids?.[id]) continue;
          const max = Math.min(cfg.maxBid, g.players![id].chips);
          const amt = Math.max(0, Math.min(Number(yield { kind: 'bid', player: id, giftId: gift, max, view: view(id), game: g }) || 0, max));
          run(A.submitBid(g, id, amt, NOW));
          if (amt > 0) { res.bidsPlaced++; res.bidsByAmount[amt] = (res.bidsByAmount[amt] || 0) + 1; }
        }
        run(A.revealBids(g));
        break;
      }
      case 'reveal': {
        const opener = t.openerId!;
        const bids = Object.entries(t.bids || {}).filter(([, b]) => b.amount > 0).sort(([, a], [, b]) => b.amount - a.amount);
        if (!bids.length) {
          const targets = gambitTargets(g, t.giftId, opener).map(([id, x]) => ({ id, cost: minCost(cfg, x.naughtyLevel || 0) }));
          const pick = (yield { kind: 'gambit', player: opener, targets: targets.filter((x) => x.cost <= g.players![opener].chips), view: view(opener), game: g }) as string | null;
          if (pick) { pay(opener, minCost(cfg, g.gifts![pick].naughtyLevel || 0)); res.ledger.costsBurned += minCost(cfg, g.gifts![pick].naughtyLevel || 0); res.gambits++; run(A.gambit(g, pick)); }
          else run(A.keepGift(g));
          break;
        }
        if (bids.length === 1) { pay(bids[0][0], bids[0][1].amount); run(A.startDuel(g, bids[0][0], opener)); break; }
        const tied = topTiedBidders(t);
        const fight = Boolean(yield { kind: 'fight', player: opener, giftId: t.giftId!, view: view(opener), game: g });
        if (tied.length >= 2) run(fight ? A.openerFight(g, null, null, true, r) : A.openerYield(g, null, null, r));
        else run(fight ? A.openerFight(g, bids[0][0], bids[1][0], false, r) : A.openerYield(g, bids[0][0], bids[1][0], r));
        if (!fight) res.yields++;
        break;
      }
      case 'yield_choice': run(A.openerFight(g, t.selectedC1!, t.selectedC2!, true, r)); break;
      case 'duel_pending': pay(t.challengerId!, t.bids?.[t.challengerId!]?.amount || 0); run(A.startDuel(g, t.challengerId!, t.openerId!)); break;
      case 'tiebreaker_pending': run(A.startTiebreaker(g, t.tiebreakerC1!, t.tiebreakerC2!, ctx())); break;
      case 'yield_pending': {
        for (const c of [t.yieldC1!, t.yieldC2!]) pay(c, Math.floor((t.bids?.[c]?.amount || 0) / 2));
        run(A.startYieldDuel(g, t.yieldC1!, t.yieldC2!, false, ctx()));
        break;
      }
      case 'duel': {
        const w = coin(t.challengerId!, t.defenderId!);
        if (t.isTiebreak) {
          pay(w, t.bids?.[w]?.amount || 0);
          run(A.recordTiebreakWinner(g, w));
          break;
        }
        const kind = t.isGambit ? 'gambit' : t.isReprisal ? 'reprisal' : t.isYieldDuel ? 'yield' : t.isEndgameSteal ? 'steal'
          : t.isEndgameAuction ? 'auction' : t.isNoBidsDuel ? 'nobids' : 'sealed';
        resolve(w, kind, t.bids?.[t.challengerId!]?.amount || 0);
        break;
      }
      case 'reprisal': {
        const pid = t.reprisalPlayerId!;
        const targets = reprisalTargets(g, t.excludedGiftId, pid).map(([id, x]) => ({ id, cost: minCost(cfg, x.naughtyLevel || 0) }));
        const pick = (yield { kind: 'reprisal', player: pid, targets: targets.filter((x) => x.cost <= g.players![pid].chips), view: view(pid), game: g }) as string | null;
        if (pick) {
          const cost = minCost(cfg, g.gifts![pick].naughtyLevel || 0);
          pay(pid, cost); res.ledger.costsBurned += cost; res.reprisals++;
          run(D.chooseReprisal(g, pick, NOW));
        } else run(D.skipReprisal(g));
        break;
      }
      case 'endgame_choice': {
        const pid = t.activePlayerId!;
        const targets = stealTargets(g, pid).map(([id, x]) => ({ id, cost: minCost(cfg, x.naughtyLevel || 0) }))
          .filter((x) => x.cost <= g.players![pid].chips);
        const pick = (yield { kind: 'steal', player: pid, targets, misfitId: t.misfitId!, view: view(pid), game: g }) as string | null;
        if (pick) {
          const cost = minCost(cfg, g.gifts![pick].naughtyLevel || 0);
          pay(pid, cost); res.ledger.costsBurned += cost; res.steals++;
          run(D.chooseSteal(g)); run(D.executeSteal(g, pick));
        } else { res.auctions++; run(D.startAuction(g)); }
        break;
      }
      case 'endgame_bidding': {
        for (const id of ids) {
          if (id === t.activePlayerId || t.bids?.[id]) continue;
          const max = Math.min(cfg.maxBid, g.players![id].chips);
          const amt = Math.max(0, Math.min(Number(yield { kind: 'bid', player: id, giftId: t.misfitId!, max, view: view(id), game: g }) || 0, max));
          run(A.submitBid(g, id, amt, NOW));
          if (amt > 0) { res.bidsPlaced++; res.bidsByAmount[amt] = (res.bidsByAmount[amt] || 0) + 1; }
        }
        run(D.revealAuction(g));
        break;
      }
      case 'endgame_reveal': {
        const [top] = Object.entries(t.bids || {}).sort(([, a], [, b]) => b.amount - a.amount);
        pay(top[0], top[1].amount);
        run(D.auctionDuel(g, top[0]));
        break;
      }
      case 'endgame_nobids': {
        const others = ids.filter((id) => id !== t.activePlayerId);
        run(D.noBidsDuel(g, others[Math.floor(r() * others.length)]));
        break;
      }
      case 'minigame_party': {
        // Every tie is a whole-table party game. Skill isn't modelled: scores / votes are random.
        const mg = g.minigame as PartyGame;
        if (mg.phase === 'intro') { res.partyGames[mg.type] = (res.partyGames[mg.type] || 0) + 1; run(advanceParty(g, 'intro', NOW)); break; }
        if (mg.phase === 'playing') {
          if (mg.type === 'whack_elf') for (const id of mg.participants) run(reportWhack(g, id, Math.floor(r() * 20), true));
          else for (const id of mg.participants) run(submitSketch(g, id, 'data:sketch'));
        }
        if (mg.phase === 'voting') {
          for (const id of mg.participants) {
            const others = mg.participants.filter((x) => x !== id);
            run(voteSketch(g, id, others[Math.floor(r() * others.length)]));
          }
        }
        if (mg.phase === 'playing' || mg.phase === 'voting') {
          const from = (g.minigame as PartyGame).phase;
          run(advanceParty(g, from, NOW));
          const after = g.minigame as PartyGame;
          if (after.phase === 'results' && after.results?.bonus && after.prize === 'points_chips') res.ledger.elfBonusMinted += 1;
          break;
        }
        if (mg.phase === 'results') {
          // the tie winner(s) now pay their bids and duel (pick_one) / duel each other (pick_two)
          const goal = mg.minigameGoal;
          const winners = mg.results?.winners || [];
          if (goal === 'pick_two') for (const c of winners) pay(c, Math.floor((t.bids?.[c]?.amount || 0) / 2));
          else if (winners[0]) pay(winners[0], t.bids?.[winners[0]]?.amount || 0);
          run(advanceParty(g, 'results', NOW));
        }
        break;
      }
      default:
        throw new Error(`sim: unhandled turn phase ${t.phase}`);
    }
  }
  void wrappedGifts;
  res.ledger.end = sum(g);
  res.seats = ids.map((id, i) => {
    const p = g.players![id];
    return {
      id, strategy: o.names[i], chipsLeft: p.chips, gotGift: !!p.giftId, spent: spent[id],
      giftValue: p.giftId ? valueOf(i, giftIdx(p.giftId)) : 0,
    };
  });
  return res;
}
