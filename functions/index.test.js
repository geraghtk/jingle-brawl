// Unit tests for the critique reply parser.  Run: cd functions && npm test
const test = require("node:test");
const assert = require("node:assert");
const { parseCritique, critiquePrompt, noDashes } = require("./index.js")._internal;

test("parses the DESCRIPTION / SCORE / CRITIQUE format", () => {
  const r = parseCritique("DESCRIPTION: A cat on a mat.\nSCORE: 77\nCRITIQUE: Purr-fectly adequate.");
  assert.deepStrictEqual(r, { description: "A cat on a mat.", rating: 77, critique: "Purr-fectly adequate.", error: null });
});

test("takes the first number of 'SCORE: 85/100'", () => {
  assert.strictEqual(parseCritique("SCORE: 85/100\nCRITIQUE: ok").rating, 85);
});

test("clamps and defaults the score", () => {
  assert.strictEqual(parseCritique("SCORE: 250").rating, 100);
  assert.strictEqual(parseCritique("no score here").rating, 50);
});

test("falls back to the raw text as the critique", () => {
  assert.strictEqual(parseCritique("Just vibes.").critique, "Just vibes.");
});

test("prompt embeds the secret word", () => {
  assert.match(critiquePrompt("Penguin"), /"Penguin"/);
});

test("no em dashes reach players: the prompt asks for none and replies are cleaned", () => {
  const em = "\u2014";
  assert.match(critiquePrompt("Penguin"), /Do not use em dashes/);
  assert.ok(!critiquePrompt("Penguin").includes(em));
  const r = parseCritique(`DESCRIPTION: A penguin ${em} wobbly.\nSCORE: 70\nCRITIQUE: Charming${em}truly ${em}.`);
  assert.strictEqual(r.description, "A penguin, wobbly.");
  assert.strictEqual(r.critique, "Charming, truly.");
  assert.strictEqual(noDashes(`${em} Alice wins \u2013 easily, pages 1\u20135.`), "Alice wins, easily, pages 1\u20135.");
});
