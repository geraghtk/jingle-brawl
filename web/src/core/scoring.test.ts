// Party Score + spectator bets.
import { test } from 'vitest';
import assert from 'node:assert';
import * as S from './scoring';

test('Party Score = party points + correct duel calls + duels fought', () => {
  assert.strictEqual(S.partyScore({ partyPoints: 3, votingWins: 2, votingLosses: 5 }), 5);
  assert.strictEqual(S.partyScore({ partyPoints: 3, votingWins: 2, duelsFought: 4 }), 9);
  assert.strictEqual(S.partyScore({}), 0);
});

test('the standings are by Party Score, never by chips: sitting on your chips does not win', () => {
  const rows = S.partyStandings({
    saver: { name: 'Saver', chips: 10 }, fighter: { name: 'Fighter', chips: 1, duelsFought: 3, partyPoints: 1 }, caller: { name: 'Caller', chips: 6, votingWins: 2 },
  });
  assert.deepStrictEqual(rows.map((r) => [r.name, r.score, r.place, r.duels]), [['Fighter', 4, 1, 3], ['Caller', 2, 2, 0], ['Saver', 0, 3, 0]]);
  assert.deepStrictEqual(S.partyMvps({ saver: { chips: 10 }, fighter: { duelsFought: 1 } }), ['fighter']);
});

test('Party MVP: highest score, ties share it, nobody if nobody scored', () => {
  assert.deepStrictEqual(S.partyMvps({ a: { partyPoints: 3 }, b: { votingWins: 3 }, c: { votingWins: 1 } }), ['a', 'b']);
  assert.deepStrictEqual(S.partyMvps({ a: {}, b: { votingLosses: 2 } }), []);
});

test('Party Score standings: best first, ties share a place, then by name', () => {
  const t = S.partyStandings({ a: { name: 'Cy', partyPoints: 2 }, b: { name: 'Al', votingWins: 3 }, c: { name: 'Bo', partyPoints: 1, votingWins: 1 }, d: { name: 'Di' } });
  assert.deepStrictEqual(t.map((r) => [r.name, r.score, r.place]), [['Al', 3, 1], ['Bo', 2, 2], ['Cy', 2, 2], ['Di', 0, 4]]);
  assert.deepStrictEqual([t[1].points, t[1].calls], [1, 1]);
});

// --- spectator bets: the pool (an even line unless said: the Fund's chip is half on each duelist) ---
const even = { a: 'w', line: 0.5, carry: 0, tax: 0 };
const on = (who: string) => ({ votedFor: who, stake: 1 });

test('bets: the pool is the stakes + 1 from the Fund; the winner\'s backers share it, whole chips each', () => {
  // 5 stakes + 1 seed = 6; two right callers and the Fund's half chip share it: 6 / 2.5 = 2.4, so 2 each
  const r = S.settleBets({ s1: on('w'), s2: on('w'), s3: on('l'), s4: on('l'), s5: on('l') }, 'w', even);
  assert.deepStrictEqual(r.delta, { s1: 1, s2: 1, s3: -1, s4: -1, s5: -1 });
  assert.deepStrictEqual([r.pool, r.each, r.winners, r.toBank, r.carry, r.returned], [6, 2, 2, 1, 1, false]);
  // nothing leaks: what the players put in and the Fund seeded is what was paid, went back or waits
  assert.strictEqual(r.each * r.winners + r.toBank + r.carry, r.pool);
});

test('bets: a lone backer of the underdog gets at most 3 for 1; the rest waits for the next pool', () => {
  const votes = { s1: on('w'), s2: on('l'), s3: on('l'), s4: on('l'), s5: on('l'), s6: on('l'), s7: on('l'), s8: on('l') };
  const r = S.settleBets(votes, 'w', even);                       // 8 stakes + 1 = 9 to one chip and the Fund's half: capped
  assert.strictEqual(r.each, S.BETS.cap);
  assert.strictEqual(r.delta.s1, 2);
  assert.strictEqual(r.toBank, 1);                                // the Fund's half chip at the cap: 1.5, whole chips only
  assert.strictEqual(r.carry, 9 - 3 - 1);
  // and the next duel's pool starts with it, plus that duel's tax
  const next = S.settleBets({ s1: on('w'), s2: on('l') }, 'w', { ...even, carry: r.carry, tax: 2 });
  assert.strictEqual(next.pool, 1 + 2 + 5 + 2);
  assert.strictEqual(next.each, 3);
});

test('bets: nobody backed the winner: every stake goes back and the pool waits as it was', () => {
  const r = S.settleBets({ s1: on('l'), s2: on('l'), s3: { votedFor: 'w', stake: 0 } }, 'w', { ...even, carry: 4, tax: 3 });
  assert.deepStrictEqual(r.delta, {});
  assert.deepStrictEqual([r.returned, r.carry, r.pool], [true, 4, 0]);
});

test('bets: free votes are not bets; no bets at all leaves the pool alone', () => {
  const r = S.settleBets({ s1: { votedFor: 'w', stake: 0 }, s2: { votedFor: 'l' } }, 'w', { ...even, carry: 2 });
  assert.deepStrictEqual([r.delta, r.carry, r.returned], [{}, 2, false]);
  assert.deepStrictEqual(S.settleBets(undefined, 'w', even).delta, {});
});

test('bets: everyone right: the Fund\'s chip is all there is to win', () => {
  const r = S.settleBets({ s1: on('w'), s2: on('w') }, 'w', even);  // 3 chips to 2.5: 1 each, their own chip back
  assert.deepStrictEqual(r.delta, { s1: 0, s2: 0 });
  assert.strictEqual(r.each * 2 + r.toBank + r.carry, 3);
});

test('bets: a bet never costs more than the stake, and the pool always adds up', () => {
  for (let right = 1; right <= 10; right++) for (let wrong = 0; wrong <= 10; wrong++) for (const line of [0.2, 0.5, 0.8]) for (const extra of [0, 1, 5]) {
    const votes = Object.fromEntries([...Array(right)].map((_, i) => [`r${i}`, on('w')]).concat([...Array(wrong)].map((_, i) => [`x${i}`, on('l')])));
    const r = S.settleBets(votes, 'w', { a: 'w', line, carry: extra, tax: extra });
    assert.ok(r.each >= 0 && r.each <= S.BETS.cap);
    assert.ok(r.carry >= 0 && r.toBank >= 0);
    assert.strictEqual(r.each * right + r.toBank + r.carry, r.pool);
    assert.strictEqual(r.pool, 1 + 2 * extra + right + wrong);
  }
});

test('the line: even with no record, leaning to the better record, never certain', () => {
  assert.strictEqual(S.betLine({}, undefined), 0.5);
  assert.strictEqual(S.betLine({ duelWins: 2, duelLosses: 1 }, { duelWins: 2, duelLosses: 1 }), 0.5);
  const strong = S.betLine({ duelWins: 5 }, { duelLosses: 5 });
  assert.ok(strong > 0.6 && strong < 0.8);
  assert.ok(Math.abs(S.betLine({ duelWins: 5 }, { duelLosses: 5 }) + S.betLine({ duelLosses: 5 }, { duelWins: 5 }) - 1) < 1e-9);
});

test('the odds on show: what one more chip on each duelist would get back, never more than the cap', () => {
  const o = S.poolOdds({ s1: on('a'), s2: on('a'), s3: on('a') }, 'a', 'b', { a: 'a', line: 0.5, carry: 0, tax: 0 });
  assert.strictEqual(o.pool, 4);
  assert.deepStrictEqual(o.backers, { a: 3, b: 0 });
  assert.ok(Math.abs(o.pays.a - 5 / 4.5) < 1e-9);                 // crowded: barely your chip back
  assert.strictEqual(o.pays.b, 3);                                // alone on the other side: 5 / 1.5, capped
  assert.ok(o.pays.b > o.pays.a);
});
