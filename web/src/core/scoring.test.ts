// Party Score + spectator bets.
import { test } from 'vitest';
import assert from 'node:assert';
import * as S from './scoring';

test('Party Score = party points + correct duel calls', () => {
  assert.strictEqual(S.partyScore({ partyPoints: 3, votingWins: 2, votingLosses: 5 }), 5);
  assert.strictEqual(S.partyScore({}), 0);
});

test('Party MVP: highest score, ties share it, nobody if nobody scored', () => {
  assert.deepStrictEqual(S.partyMvps({ a: { partyPoints: 3 }, b: { votingWins: 3 }, c: { votingWins: 1 } }), ['a', 'b']);
  assert.deepStrictEqual(S.partyMvps({ a: {}, b: { votingLosses: 2 } }), []);
});

test('bets: wrong callers pay 1 each, split between right callers; odd chip to the Bank', () => {
  const r = S.settleBets({
    s1: { votedFor: 'w', stake: 1 }, s2: { votedFor: 'w', stake: 1 },
    s3: { votedFor: 'l', stake: 1 }, s4: { votedFor: 'l', stake: 1 }, s5: { votedFor: 'l', stake: 1 },
  }, 'w');
  assert.deepStrictEqual(r.delta, { s1: 1, s2: 1, s3: -1, s4: -1, s5: -1 });
  assert.deepStrictEqual([r.pool, r.toBank, r.winners], [3, 1, 2]);
});

test('bets: nobody right → the pool goes to the Bank; zero-stake votes are free', () => {
  const r = S.settleBets({ s1: { votedFor: 'l', stake: 1 }, s2: { votedFor: 'w', stake: 0 } }, 'w');
  assert.deepStrictEqual(r.delta, { s1: -1 });
  assert.strictEqual(r.toBank, 1);
});

test('bets: everyone right → nobody loses anything', () => {
  const r = S.settleBets({ s1: { votedFor: 'w', stake: 1 }, s2: { votedFor: 'w', stake: 1 } }, 'w');
  assert.deepStrictEqual(r.delta, { s1: 0, s2: 0 });
});
