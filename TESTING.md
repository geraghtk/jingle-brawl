# Multi-Device Testing Guide

Yuel Duel is a party game where a host and multiple players interact in real time from their own devices. This guide explains how to test that interaction on a single machine.

## Quick Start

1. `npm run dev` (the room server and the app on :5173)
2. Open `http://localhost:5173/host.html` → create a game → note the room code
3. Open player tabs: `http://localhost:5173/player.html?room=ABCD&testId=1&name=Alice` (`testId=2`, `3`, … for more)
4. Start the game from the host when everyone is in

Each player tab uses `?testId=N` so it has **isolated localStorage**: tabs from the same browser act as different players.
To watch a whole party on the real app instead: `npx vite-node web/src/sim/play-table.ts -- --live --headed --players 4`.

## Manual Testing

Other ways to simulate multiple devices:

| Method | How | Notes |
|-------|-----|------|
| **Different browsers** | Host in Chrome, Player 1 in Firefox, Player 2 in Edge | Each browser has separate storage |
| **Incognito + normal** | Host in normal Chrome, Player 1 in Incognito | Incognito has isolated storage |
| **Multiple Incognito windows** | Each in its own Incognito window | May share storage depending on browser |
| **URL with testId** | `player.html?room=ABCD&testId=1&name=Alice` | Same browser, isolated storage per testId |

## URL Parameters for player.html

| Param | Example | Purpose |
|-------|---------|---------|
| `room` | `?room=ABCD` | Pre-fills room code |
| `testId` | `?testId=1` | Isolated localStorage (for multi-tab testing) |
| `name` | `?name=Alice` | Pre-fills player name |

## What to Test

- **Lobby**: Host sees players join; players see each other
- **Bidding**: All non-opener players can submit bids
- **Duels**: Challenger vs defender; votes from other players
- **Sketch-Off**: Everyone draws the prompt, votes (not for themselves); tie + party points
- **Whack-an-Elf**: Everyone taps for 20 s; live TV leaderboard; Elf Bonus chip
- **Draw Off**: Each player draws; AI critiques; witty verdict; duel
- **Reprisals**: Loser can target another gift
- **Endgame**: Misfit lottery, steals

## Automated E2E (Playwright)

Automated multi-user flows run with Playwright. Each "user" gets an isolated browser
context, so one run drives a host + several phones. Playwright builds the app and the
room server serves it (`--db :memory: --debug-api`), so every run starts clean.

### Setup

```bash
npm install
npx playwright install chromium
```

### Run Tests

```bash
npm run test:e2e          # everything, headless: builds, then the room server serves it (--db :memory: --debug-api)
npm run test:flows        # just the game-flow permutations, list output
npm run test:e2e:ui       # visual debugging
npm run test:e2e:headed   # watch the browsers
npx playwright test -g "Draw Off"   # one test by name
```


### What's Covered

| Spec | Covers |
|------|--------|
| `lobby-and-join.spec.js` | create → 3 join → start |
| `bidding-and-duel.spec.js` | draw → bid → reveal → duel (smoke) |
| `game-flows.spec.js` | every main-phase fork: no bids, single bidder win/loss, two bids fight/yield, ties fight/yield, reprisal offered, full game → final tally |
| `regressions.spec.js` | one test per bug fixed in the 2026-10 review: starting-chips setting, duel voting, Gambit cancel + Misfit targets, clean reprisal duels, host reprisal override, own-gift reprisal, yield rounding, win/loss verdict, admin Misfit, Sketch-Off + Whack-an-Elf (tie, points, Elf Bonus), Draw Off canvas wipe, 3-way ties, Draw Off yield loop, endgame auction, settings-driven help text, apostrophe rejoin |

Tests assert **invariants** (`invariantErrors()` in `e2e/helpers.js`: chips never
negative, gift ↔ owner consistent both ways, no player holds 2 gifts, zero page errors)
rather than encoding expected chip totals. `dbGet` / `dbPut` / `dbPatch` read and inject
state through the server's debug API (this machine only) for setups that clicking can't
reach quickly (e.g. an endgame with a Misfit pile).

**Rule:** every bug fix or new code path ships with a test that fails without it.

### Other test suites

```bash
python -m unittest discover -s tests          # ai_art_critic.py parsing (needs its requirements)
```

### Agent play-testing (playability & balance)

`agents/` has persona agents (The Accountant, The Alpha, The Endgame Shark …) that play
real games through the real pages: `npm run agents:watch` to watch one live,
`npm run agents:sweep` to compare settings presets. Stalls, page/console errors,
missing buttons and invariant breaks are reported with screenshots. See
[`agents/README.md`](agents/README.md).

### Requirements

- Node 22.5+ (the room server uses `node:sqlite`); Playwright serves everything on :4173

## Logic-Only Testing

For logic-only runs without browsers, `web/src/sim/` plays whole games through the real engine in memory (`npm run test:unit`, `npx vite-node web/src/sim/play-table.ts`). It does **not** exercise the UI.

## Manual testing on real phones (iPhone Safari, Android Chrome)

`npm run dev` on a PC; phones on the same Wi-Fi. (Windows: allow Node through the firewall on private networks.)

1. **Host on a phone:** open the PC's Wi-Fi address from `npm run dev`'s output → `player.html` → "host a new game" → Create → "Join as a player too". The phone shows two tabs, My game and Host: switching is instant, the Host tab gets a dot when the game waits on you (two players in: Start), and the console's dialogs (prompts, confirms) work inside the tab on iPhone Safari too.
2. **TV:** on the casting/TV device open the console's TV link (`tv.html?room=CODE`): no buttons, the join QR.
3. **Players:** scan the TV's QR on 2+ more phones; pick avatars; join.
4. **A round:** draw, bid (change a bid once), check the TV shows bids only as "in", the host's 👀 Peek shows amounts on the console only, reveal, duel, verdict.
**Trying one mini-game on its own: the Game Lab.** In the lobby, the console's Controls card has **🧪 Game Lab: try a mini-game**: pick the game, who plays (one player can practise alone: a free-for-all), how (head to head, everyone with two duelists, two captains' teams) and the duelists or captains, then **Start practice**. It runs on the real phones and the TV with nothing at stake (no gift, bids, chips or Party points); the result stays up until **Play again** or **Back to the lobby**, and the real game can't start until then. The quickest way to tune a game: the host on the PC (also "Join as a player too" for a second seat, or untick everyone else), your phone as a player. (`core/minigames/lab.ts`, `e2e/lab.spec.js`.)

5. **Pass the Gremlin** (tick it at setup; Duel pick: wheel, no room duels; or the Game Lab): the holder's phone buzzes (Android; iPhone has no vibration API), the keypad is big enough, a wrong answer waits briefly, Random and named passes work, the TV never shows the answer; Pause / Resume on the console. With the challenge mix: the memory row is readable in the time it shows; the timing marker moves smoothly (no stutter on an older phone) and a tap that looks in the zone counts (if good taps miss, the phone's clock offset or touch latency needs a look); the sound switch works and is remembered; sounds play after the first tap (iPhone's silent switch mutes them).
6. **AI Draw Off** (tick it, enter your key): both duellists draw; the TV hides drawings until judging; the critic decides; try a wrong key → the console says so without showing it.
7. **Resilience:** lock a phone for 30 s and unlock (it reconnects, same screen; a "Reconnecting..." strip shows if it takes a moment); a dead phone: the host's "A phone died?" link seats that player on another phone; turn Wi-Fi off/on mid-game; reload the host console; restart `npm run dev` mid-Gremlin (it comes back paused).
8. **Accessibility:** iOS "Reduce Motion" stops the Gremlin's shaking; text zoom keeps the keypad usable.
