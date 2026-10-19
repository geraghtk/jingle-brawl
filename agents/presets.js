// Settings presets for `--sweep`. Keys are the host setup-screen <select> ids without
// the "setting-" prefix; values must be one of that select's <option> values.
// Anything not listed keeps the app default (see GAME_CONFIG in firebase-config.js).
const PRESETS = [
  { name: 'default', why: 'app defaults: payout cap 2 from pot 2, dividend to defenders, 1 reprisal, max bid 4', settings: {} },
  { name: 'v2-classic', why: 'closest to the original written V2 rules', settings: { 'tax-threshold': '3', 'max-payout': '3', 'loser-dividend': 'everyone', 'reprisal-depth': '2' } },
  { name: 'dividend-all', why: 'every loser gets +1, not just defenders', settings: { 'loser-dividend': 'everyone' } },
  { name: 'no-dividend', why: 'nobody gets a dividend', settings: { 'loser-dividend': 'nobody' } },
  { name: 'no-tax', why: 'winner keeps the whole pot', settings: { 'max-payout': '99', 'tax-threshold': '99' } },
  { name: 'heavy-tax', why: 'winner keeps at most 1 chip', settings: { 'max-payout': '1' } },
  { name: 'chips-8', why: 'tight economy', settings: { 'starting-chips': '8' } },
  { name: 'chips-15', why: 'loose economy', settings: { 'starting-chips': '15' } },
  { name: 'max-bid-5', why: 'higher bid ceiling', settings: { 'max-bid': '5' } },
  { name: 'cheap-challenges', why: 'Gambit/Reprisal/Steal cost = Naughty level only', settings: { 'base-cost': '0' } },
  { name: 'reprisal-chain-2', why: 'a Reprisal can trigger one more', settings: { 'reprisal-depth': '2' } },
];

module.exports = { PRESETS };
