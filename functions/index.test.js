// Unit tests for the critique reply parser.  Run: cd functions && npm test
const test = require("node:test");
const assert = require("node:assert");
const { parseCritique, critiquePrompt } = require("./index.js")._internal;

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
