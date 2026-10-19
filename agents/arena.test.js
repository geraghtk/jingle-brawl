// node --test agents/arena.test.js
const test = require('node:test');
const assert = require('node:assert');
const { Arena } = require('./arena');

// Join mode: a person types the gift names, so the harness never assigned them a quality.
// They all used to count as exactly average (5.5) — so almost nothing looked worth a bid.
test('a gift the harness did not name gets its own quality (1-10), the same for everyone', () => {
  const arena = new Arena({ seed: 7, seats: [], settings: {} });
  arena.taste.p1 = {};
  arena.taste.p2 = {};
  const names = ['cat house', 'microwave oven', 'dog treats', 'house', 'nickers', 'socks', 'mug', 'lamp'];
  for (const n of names) arena.valueFor('p1', n);
  const qualities = names.map((n) => arena.values[n]);
  for (const q of qualities) assert.ok(Number.isInteger(q) && q >= 1 && q <= 10, `quality ${q}`);
  assert.ok(new Set(qualities).size > 2, `qualities should vary, got ${qualities}`);
  // the base quality is shared: p2 sees the same gift, only their taste differs
  const before = { ...arena.values };
  for (const n of names) arena.valueFor('p2', n);
  assert.deepStrictEqual(arena.values, before);
});
