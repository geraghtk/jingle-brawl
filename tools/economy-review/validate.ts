// 1) sim2 reproduces the repo simulator game for game (same seeds). 2) the chip ledger identity, incl. votes + bets.
import { playGame } from 'C:/Users/Kevin/jingle-brawl/web/src/sim/simulate';
import * as R from 'C:/Users/Kevin/jingle-brawl/web/src/sim/strategies';
import { play, mixTable, MIX } from './sim2';

const repoMix = [R.aggressive(2), R.valueScaled, R.aggressive(3), R.passive, R.random, R.aggressive(1)];
let same = 0, total = 0;
for (let i = 0; i < 400; i++) {
  const n = [3, 4, 6, 8, 12][i % 5];
  const a = playGame({ strategies: Array.from({ length: n }, (_, k) => repoMix[(k + i) % 6]), seed: 4242 + i });
  const b = play({ strategies: mixTable(n, i), seed: 4242 + i });
  total++;
  const ok = a.ledger.end === b.ledger.end && a.ledger.taxBurned === b.ledger.taxBurned && a.ledger.costsBurned === b.ledger.costsBurned
    && a.seats.every((s, k) => s.chipsLeft === b.seats[k].chipsLeft && Math.abs(s.giftValue - b.seats[k].giftValue) < 1e-9) && a.duels.length === b.duels.length;
  if (ok) same++;
}
console.log(`sim2 == repo simulator on ${same}/${total} games (seeds 4242..4641, n in 3/4/6/8/12, mixed table)`);

let bad = 0, games = 0, neg = 0, unfinished = 0;
for (const bets of [0, 1]) for (const rule of ['defenders', 'everyone', 'nobody']) for (let i = 0; i < 500; i++) {
  const n = [3, 4, 6, 8, 12][i % 5];
  const r = play({ strategies: mixTable(n, i), seed: 9000 + i, settings: { spectatorBets: bets, loserDividendRule: rule }, votes: 'random' });
  const L = r.ledger; games++;
  const expect = L.start - L.taxBurned - L.costsBurned + L.dividendsMinted + L.elfBonusMinted - L.betsBurned;
  if (expect !== L.end) { bad++; if (bad < 4) console.log('ledger mismatch', JSON.stringify(L), expect); }
  if (r.seats.some((s) => s.chipsLeft < 0)) neg++;
  if (!r.ended) unfinished++;
}
console.log(`ledger identity end = start - tax - costs + dividends + elf bonus - bets-to-bank: ${games - bad}/${games} games hold; negative chips ${neg}; unfinished ${unfinished}`);
void MIX;
