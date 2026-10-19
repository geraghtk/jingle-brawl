# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Jingle Brawl** (also called "Yuel Duel") is a real-time multiplayer party game for gift exchanges. Players bid Jingle Chips on gifts, duel to win them, and can challenge each other with special actions like Grinch's Gambit and Reindeer Reprisals. A tied top bid can be settled by a party mini-game the whole table plays ("Sketch-Off", "Whack-an-Elf" — `party-games.js`) or by "AI Draw Off", with an AI Art Critic judging the results.

## Commands

```bash
npm install                      # Playwright
npm run emulator                 # local Realtime Database (:9000) + Auth (:9099) emulators — needs Java 21+
python -m http.server 8080       # serve the pages (Playwright starts this automatically)

npm run test:e2e                 # full Playwright suite (boots server + emulators if not running)
npm run test:flows               # game-flow permutations only
npm run test:e2e:ui / :headed    # visual debugging
cd functions && npm test         # Cloud Function parser unit tests
python -m unittest discover -s tests   # ai_art_critic.py unit tests

npm run agents:watch             # persona agents play a real game, streamed to one window
npm run agents:sweep             # all settings presets × N headless games → agents/reports/…/summary.md
npm run test:agents              # persona unit tests

python ai_art_critic.py          # optional local AI critic (Ollama llava / OpenAI) on :5050

firebase deploy --only database  # security rules
firebase deploy --only functions # AI critique cloud functions
firebase deploy --only hosting
```

## Local Development

On localhost the pages talk to the **emulators**, never production (`useLocalEmulators()` in `firebase-config.js`; add `?prod=1` to opt out). Start them with `npm run emulator`, then:
- Host: `http://localhost:8080/host.html`
- Player tabs: `http://localhost:8080/player.html?room=ABCD&testId=1&name=Alice`

`?testId=N` isolates localStorage and uses per-tab auth persistence, so one browser can simulate many players. `test-multi-device.html` opens these tabs. Emulator UI: http://127.0.0.1:4000.

## Architecture

**Stack:** Vanilla HTML/JS (no framework, no build), Firebase Realtime Database (compat SDK 9.22) + Anonymous Auth, Cloud Functions (Node 22) for production AI critique, optional local Flask server for AI critique.

**Key files:**
- `host.html` (~7.4k lines) — host/TV screen: creates games, draws names/gifts, resolves every duel (chip math, swaps, naughty level), runs minigames, admin panel, bot auto-pilot
- `player.html` (~4.4k lines) — phone: join/rejoin, sealed bids, opener choices (keep/Gambit/fight/yield), reprisals, endgame paths, drawing canvas, votes
- `firebase-config.js` — Firebase config, `connectGameDatabase()` / `connectGameAuth()` (emulator routing), `GAME_CONFIG` defaults
- `database.rules.json` — security rules (auth required; players own their node via `ownerUid`, host via `hostUid`)
- `functions/index.js` — `critiqueDrawing` (OpenAI) and `critiqueDrawingGemini` (Vertex AI via `@google/genai`); model IDs from `functions/.env`
- `ai_art_critic.py` — local Flask critic: `/critique`, `/witty-verdict` (Draw Off verdict), `/health`
- `jingle_brawl_env.py` — Deep RL self-play simulation for balance testing
- `e2e/` — Playwright specs + `helpers.js` (invariants, emulator REST helpers)
- `agents/` — persona-agent play-testing harness (playability + settings/balance sweeps); see `agents/README.md`
- `chip-fx.js` — Jingle Chip look + `flyChips()` (loser → winner flights), shared by both pages
- `mascots.js` — Santa / elf / reindeer pop-ups reacting to game moments (`?mascots=0` disables)
- `scoring.js` — Party Score (party points + correct duel calls → Party MVP) and spectator bets (`settings.spectatorBets`, experimental); pure + unit-tested
- `party-games.js` — Sketch-Off + Whack-an-Elf: pure ranking/reward rules (unit-tested, `npm run test:party`) plus the phone and TV screens; host.html only runs the phase timers (`startPartyGame` / `advancePartyGame`)

**Real-time data flow:** host creates `games/{roomCode}` (claimed atomically by transaction) → all clients attach listeners → state changes propagate. Key paths: `players`, `gifts`, `currentTurn` (turn state machine), `phase`, `settings`, `minigame`, `lastDuel` (outcome the host publishes so phones show the right verdict).

**Game state machine:**
```
lobby → main (draw → bidding → reveal → [duel_pending | yield_pending | tiebreaker_pending → minigame_*] → duel → reprisal?) → endgame (endgame_choice → steal | auction) → ended
```

**Conventions that matter:**
- A Misfit-pile gift has `ownerId: 'misfit'` (+ `misfitOrder` for FIFO). Never `null`.
- "In the bag" = player has no `giftId`; there is no separate bag list.
- The host is the resolver; phones mostly write *intents* (bids, choices, votes). Writes are still client-authoritative — see `CODE_REVIEW.md`.
- Settings live in `games/{room}/settings` and are copied into `GAME_CONFIG` on both sides.
- Host actions that move chips (`startDuel`, `openerYields`, `recordDuelWinner`) first `claimTurn()` — an atomic transaction on the turn — so a double-tap or a second host tab can't charge/pay twice. New chip-moving host actions should do the same.
- Duel help text is generated from settings + the turn (`dividendNote` / `reprisalNote` / `spectatorLoserNote` / `taxNote` in player.html) — don't hard-code rules in copy.
- Target-legality helpers exist on both sides — `reprisalTargets()` / `gambitTargets()` / `topTiedBidders()` — use them instead of re-filtering gifts.
- Inline `onclick` strings must only interpolate ids, never names (names may contain quotes).
- Escape user/AI text with `escapeHtml()` before `innerHTML`.

## Game Rules Reference

Authoritative rules: `Jingle Brawl V2.md` (player-facing: `rules.html`, `quick-reference.html`). The written rules were synced to the app on 2026-10-18 (`CODE_REVIEW.md` → "Rules vs code"); when behaviour changes, update all four rule docs (`Jingle Brawl V2.md`, `JingleBrawlV2.html`, `rules.html`, `quick-reference.html`). `GAME_FLOW_AUDIT.md` documents scenario status. `AI_ART_CRITIC.md` covers the critic backends.

## Testing Notes

Playwright runs sequentially (1 worker) against the emulators. Tests drive real host + phone pages and assert invariants (`e2e/helpers.js`). **Every bug fix or new code path must ship with a test that fails without it** (`e2e/regressions.spec.js` for fixes). See `TESTING.md`.
