// Persona sanity checks.  Run: npm run test:agents
const test = require('node:test');
const assert = require('node:assert');
const { PERSONAS, decide, rng } = require('./personas');

function view(over = {}) {
  return {
    me: { id: 'me', name: 'Me', chips: 10, giftId: null },
    gifts: {}, players: { me: { name: 'Me' }, rival: { name: 'Rival' } },
    expectedValue: 5.5, startingChips: 10, maxBid: 4, minCost: (n) => 1 + n,
    progress: 0.2, endgame: false, rivals: {}, nemesis: 'rival', leaderId: null,
    recentTopBids: [], random: rng(1),
    ...over,
  };
}

test('every persona bids within its chips and the max bid', () => {
  for (const key of Object.keys(PERSONAS)) {
    const brain = decide(key);
    for (let i = 0; i < 50; i++) {
      const v = view({ me: { id: 'me', name: 'Me', chips: i % 6, giftId: null }, random: rng(i + 1) });
      const { amount } = brain.bid(v, 1 + (i % 11));
      assert.ok(amount >= 0 && amount <= Math.min(4, i % 6), `${key} bid ${amount} with ${i % 6} chips`);
    }
  }
});

test('the Endgame Shark spends nothing early, then bids in the endgame', () => {
  const shark = decide('shark');
  assert.strictEqual(shark.bid(view(), 11).amount, 0);
  assert.ok(shark.bid(view({ endgame: true, progress: 1 }), 11).amount > 0);
});

test('the High Roller goes all-in or passes', () => {
  const hr = decide('highroller');
  assert.strictEqual(hr.bid(view(), 11).amount, 4);
  assert.strictEqual(hr.bid(view(), 5).amount, 0);
});

test('the Sentimentalist never yields its gift; the People-Pleaser yields a decent one', () => {
  assert.strictEqual(decide('sentimental').fightOrYield(view(), 0.5).action, 'fight');
  assert.strictEqual(decide('peacekeeper').fightOrYield(view(), 5).action, 'yield');
});

test('the Competitive Sibling Gambits their nemesis over a slightly better gift', () => {
  const gifts = {
    a: { name: 'Nemesis Gift', ownerId: 'rival', naughtyLevel: 0, value: 6 },
    b: { name: 'Better Gift', ownerId: 'other', naughtyLevel: 0, value: 7 },
  };
  const v = view({ gifts, me: { id: 'me', name: 'Me', chips: 10, giftId: null } });
  const d = decide('sibling').keepOrGambit(v, Object.entries(gifts));
  assert.deepStrictEqual([d.action, d.giftId], ['gambit', 'a']);
});

test('the Bandwagoner follows the room', () => {
  const follower = decide('follower');
  const quiet = follower.bid(view({ recentTopBids: [1, 1, 1] }), 8).amount;
  const hot = follower.bid(view({ recentTopBids: [4, 4, 4] }), 8).amount;
  assert.ok(hot > quiet, `room at 4 → ${hot}, room at 1 → ${quiet}`);
});
