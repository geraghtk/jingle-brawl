# Multi-Device Testing Guide

Yuel Duel is a party game where a host and multiple players interact in real time from their own devices. This guide explains how to test that interaction on a single machine.

## Quick Start: test-multi-device.html

1. **Start your local server** (e.g. `python -m http.server 8080`)
2. **Open** `http://localhost:8080/test-multi-device.html`
3. Click **Open Host** → create a game → copy the room code
4. Enter the room code in the harness
5. Click **Open All Players** (or individual Player 1, Player 2, etc.)
6. In each player tab: the room code and name are pre-filled; click **Join**
7. Start the game from the host when everyone is in

Each player tab uses `?testId=N` so they have **isolated localStorage**—multiple tabs from the same browser act as different players instead of sharing one session.

## Manual Testing (without harness)

To simulate multiple devices without the harness:

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
context, so one run drives a host + several phones. Everything runs against the **local
Firebase emulators** (Realtime Database :9000 + Auth :9099) with `database.rules.json`
enforced — tests never touch production data or create real anonymous users.

### Setup

```bash
npm install
npx playwright install chromium
# Java 21+ is required by the Firebase emulators
```

### Run Tests

```bash
npm run test:e2e          # everything, headless (Playwright boots http.server + emulators)
npm run test:flows        # just the game-flow permutations, list output
npm run test:e2e:ui       # visual debugging
npm run test:e2e:headed   # watch the browsers
npx playwright test -g "Draw Off"   # one test by name
```

`e2e/global-setup.js` waits until both emulators answer before any test starts.
Keep the emulators up between runs with `npm run emulator` (Playwright reuses them).

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
emulator state as admin (`Authorization: Bearer owner`) for setups that clicking can't
reach quickly (e.g. an endgame with a Misfit pile).

**Rule:** every bug fix or new code path ships with a test that fails without it.

### Other test suites

```bash
cd functions && npm test                      # Cloud Function reply parsing (node --test)
python -m unittest discover -s tests          # ai_art_critic.py parsing (needs its requirements)
```

### Agent play-testing (playability & balance)

`agents/` has persona agents (The Accountant, The Alpha, The Endgame Shark …) that play
real games through the real pages: `npm run agents:watch` to watch one live,
`npm run agents:sweep` to compare settings presets. Stalls, page/console errors,
missing buttons and invariant breaks are reported with screenshots. See
[`agents/README.md`](agents/README.md).

### Requirements

- Local server on port 8080 (Playwright starts `python -m http.server 8080` if not running)
- Firebase CLI + Java 21+ for the emulators (Playwright starts them if not running)

## Logic-Only Testing (test-game.html)

For logic-only testing without real browser interaction, use `test-game.html`. It creates bot players (in the emulator on localhost) and runs the game flow automatically. It does **not** exercise the player UI, drawing, or voting—only the core game state transitions.
