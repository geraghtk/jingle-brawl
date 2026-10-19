// Party mini-game rules.  Run: npm run test:party
const test = require('node:test');
const assert = require('node:assert');
const P = require('./party-games');

test('Whack-an-Elf: the same seed gives every phone the same pop-ups; no cell double-booked', () => {
  const a = P.whackSequence(42);
  assert.deepStrictEqual(a, P.whackSequence(42));
  assert.notDeepStrictEqual(a, P.whackSequence(43));
  assert.ok(a.length > 25, `${a.length} pop-ups in 20s`);
  for (let c = 0; c < 9; c++) {
    const inCell = a.filter((it) => it.cell === c);
    for (let i = 1; i < inCell.length; i++) assert.ok(inCell[i].at >= inCell[i - 1].at + inCell[i - 1].upMs, `cell ${c} overlaps`);
  }
  assert.ok(a.some((it) => it.kind === 'grinch') && a.some((it) => it.kind === 'gold'));
});

test('Whack-an-Elf points: faster taps score more, golden elves double, the Grinch costs 3', () => {
  assert.deepStrictEqual([0.1, 0.5, 0.9].map((f) => P.whackPoints('elf', f)), [3, 2, 1]);
  assert.deepStrictEqual([0.1, 0.9].map((f) => P.whackPoints('gold', f)), [6, 2]);
  assert.strictEqual(P.whackPoints('grinch', 0.1), -3);
});

test('Sketch-Off: votes count, self-votes and votes for non-drawings do not', () => {
  const s = P.sketchScores({ a: 'b', b: 'b', c: 'b', d: 'a', e: 'ghost' }, { a: 'x', b: 'x', c: 'x' });
  assert.deepStrictEqual(s, { a: 1, b: 2, c: 0 });
});

test('the best-placed tied bidder wins; the best two after a yield', () => {
  const ranking = ['x', 't2', 'y', 't1', 't3'];
  assert.deepStrictEqual(P.tieWinners(ranking, ['t1', 't2', 't3'], 'pick_one'), ['t2']);
  assert.deepStrictEqual(P.tieWinners(ranking, ['t1', 't2', 't3'], 'pick_two'), ['t2', 't1']);
  assert.deepStrictEqual(P.tieWinners(['x'], ['t1', 't2'], 'pick_one'), ['t1'], 'a tied bidder with no score still resolves');
});

test('rewards: 3/2/1 party points to scorers; Elf Bonus to the best non-tied finisher', () => {
  const ranking = ['t1', 'x', 'y', 'z', 't2'];
  const scoreOf = { t1: 9, x: 7, y: 5, z: 0, t2: 0 };
  assert.deepStrictEqual(P.rewards(ranking, scoreOf, ['t1', 't2']), { points: { t1: 3, x: 2, y: 1 }, bonus: 'x' });
  assert.deepStrictEqual(P.rewards(ranking, scoreOf, ['t1', 't2'], 'points').bonus, null);
  assert.deepStrictEqual(P.rewards(ranking, scoreOf, ['t1', 't2'], 'none'), { points: {}, bonus: null });
  assert.strictEqual(P.rewards(['a'], { a: 0 }, []).bonus, null, 'nobody scored → no bonus');
});

test('equal scores are ordered by the seed, not by seat', () => {
  const ids = ['a', 'b', 'c', 'd', 'e', 'f'];
  const flat = Object.fromEntries(ids.map((id) => [id, 1]));
  const orders = new Set([1, 2, 3, 4, 5].map((seed) => P.rank(ids, flat, seed).join('')));
  assert.ok(orders.size > 1);
  assert.deepStrictEqual(P.rank(ids, flat, 7), P.rank(ids, flat, 7));
});

test('results() puts it together for Whack-an-Elf', () => {
  const r = P.results({
    type: 'whack_elf', seed: 3, participants: ['a', 'b', 'c'], tiedBidders: ['a', 'b'], minigameGoal: 'pick_one',
    scores: { a: { score: 4 }, b: { score: 6 }, c: { score: 9 } }, prize: 'points_chips',
  });
  assert.deepStrictEqual(r.ranking, ['c', 'b', 'a']);
  assert.deepStrictEqual(r.winners, ['b']);
  assert.deepStrictEqual(r.points, { c: 3, b: 2, a: 1 });
  assert.strictEqual(r.bonus, 'c');
});
