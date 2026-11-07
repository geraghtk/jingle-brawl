# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Jingle Brawl** (also called "Yuel Duel") is a real-time multiplayer party game for gift exchanges. Players bid Jingle Chips on gifts, duel to win them, and can challenge each other with special actions like Grinch's Gambit and Reindeer Reprisals. A tied top bid can be settled by a party mini-game the whole table plays ("Sketch-Off", "Whack-an-Elf", `web/src/core/party.ts`) or by "AI Draw Off", with an AI Art Critic judging the results.

## Commands

```bash
npm install                      # Playwright
npm run emulator                 # local Realtime Database (:9000) + Auth (:9099) emulators — needs Java 21+
npm run dev                      # the app on :5173 (Vite); Playwright builds + previews on :4173 itself
npm run server -- --db :memory:  # the room server (dev, TS via vite-node) on :8787; --host 0.0.0.0 for phones on Wi-Fi
npm run build && npm start       # build the app + the server bundle (dist-server/), run it (rooms in rooms.db)

npm run test:e2e                 # full Playwright suite (boots server + emulators if not running)
npm run test:e2e:server          # the same suite with the screens on the room server (no emulators, no Java)
npm run test:flows               # game-flow permutations only
npm run test:e2e:ui / :headed    # visual debugging
cd functions && npm test         # Cloud Function parser unit tests
python -m unittest discover -s tests   # ai_art_critic.py unit tests
npm run test:unit                # rules engine + economics (vitest: web/src/core, web/src/sim)
npm run analyze:economics        # game-theory report: deviation matrix, chip flows (see ECONOMICS.md)

npm run agents:watch             # persona agents play a real game, streamed to one window
npm run agents:sweep             # all settings presets × N headless games → agents/reports/…/summary.md
npm run test:agents              # persona unit tests

npx vite-node web/src/sim/play-table.ts -- [--live] [--headed] [--players 4]   # a party at the table: engine or the real app
npx vite-node web/src/sim/table-bridge.ts       # the table over JSON lines (showrunner's Simulation Lab drives it)

python ai_art_critic.py          # optional local AI critic (Ollama llava / OpenAI) on :5050

firebase deploy --only database  # security rules
firebase deploy --only functions # AI critique cloud functions
firebase deploy --only hosting
```

## Local Development

On localhost the app talks to the **emulators**, never production (`web/src/firebase/app.ts`; add `?prod=1` to opt out). Start them with `npm run emulator` and the app with `npm run dev`, then:
- Host: `http://localhost:5173/host.html`
- Player tabs: `http://localhost:5173/player.html?room=ABCD&testId=1&name=Alice`

`?testId=N` isolates localStorage and uses per-tab auth persistence, so one browser can simulate many players. Emulator UI: http://127.0.0.1:4000.

## Architecture

**Stack:** React 18 + TypeScript + Vite (`web/`: entries `web/host.html`, `web/player.html`, static files in `web/public` incl. the rule pages), Firebase Realtime Database (modular SDK) + Anonymous Auth, Cloud Functions (Node 22) for production AI critique, optional local Flask server for AI critique. The original vanilla pages were retired on 2026-11-07. Moving to an own Node room server (server-authoritative; plan in `ARCHITECTURE.md`).

**Key files:**
- `web/src/core/` — the rules engine, pure: actions are `(game, …) => Plan` built with `Tx` (`tx.ts`); `actions.ts`, `duel.ts`, `admin.ts`, `voting.ts`, `party.ts`/`partyflow.ts` (Sketch-Off + Whack-an-Elf), `duelgame.ts` (duel picker, wheel, in-app duel games), `scoring.ts` (Party Score, spectator bets), `settings.ts` (`DEFAULTS`, `config()`), `rules.ts` (target helpers)
- `web/src/host/` — host/TV screen (`TurnPanel.tsx`, `Sidebar.tsx` admin, `PartyPanel.tsx`, `DuelModal.tsx`), `useHostAutomation.ts` arms the `hostAuto` timers
- `web/src/player/` — phone: join/rejoin, sealed bids (`BidPanel.tsx`), opener choices and duels (`StatusCard.tsx`), reprisals, party games (`PartyCard.tsx`), votes
- `web/src/firebase/` — `app.ts` (config, emulator routing, auth), `game.ts` (`useGame`, `applyPlan`, `useServerOffset`)
- `web/src/ui/` — shared UI: `mascots.ts` (Santa / elf / reindeer pop-ups, `?mascots=0` disables), `whack.tsx`, `icons.tsx`, `chips.ts`, `notes.tsx` (duel help text)
- `database.rules.json` — security rules (auth required; players own their node via `ownerUid`, host via `hostUid`)
- `functions/index.js` — `critiqueDrawing` (OpenAI) and `critiqueDrawingGemini` (Vertex AI via `@google/genai`); model IDs from `functions/.env`
- `ai_art_critic.py` — local Flask critic: `/critique`, `/witty-verdict` (Draw Off verdict), `/health`
- `jingle_brawl_env.py` — Deep RL self-play simulation for balance testing
- `e2e/` — Playwright specs + `helpers.js` (invariants, emulator REST helpers)
- `agents/` — persona-agent play-testing harness (playability + settings/balance sweeps); see `agents/README.md`
- `server/` — the room server (own `tsconfig.json`: no DOM, so `web/src/core` must stay DOM-free). `room/actor.ts` is one room's referee (intent → checks → `dispatch` → commit → save → timers from `hostAuto` → each screen its `projectGame` view), runtime-agnostic (Clock/Store/Conn); `room/rooms.ts` the registry + token secret; `store/` SQLite (`node:sqlite`) or memory; `node/` HTTP API (`/api/rooms`, join, lobby), WebSocket (`/ws`), static `dist/`, and `--debug-api` (localhost-only test helpers). Wire types: `web/src/net/protocol.ts`. Screens still talk to Firebase until the client switch (plan S4, `ARCHITECTURE.md`).

**Real-time data flow:** host creates `games/{roomCode}` (claimed atomically by transaction) → all clients attach listeners → state changes propagate. Key paths: `players`, `gifts`, `currentTurn` (turn state machine), `phase`, `settings`, `minigame`, `lastDuel` (outcome the host publishes so phones show the right verdict).

**Game state machine:**
```
lobby → main (draw → bidding → reveal → [duel_pending | yield_pending | tiebreaker_pending → minigame_*] → duel → reprisal?) → endgame (endgame_choice → steal | auction) → ended
```

**Conventions that matter:**
- A Misfit-pile gift has `ownerId: 'misfit'` (+ `misfitOrder` for FIFO). Never `null`.
- "In the bag" = player has no `giftId`; there is no separate bag list.
- **Buttons send intents, never plans:** `send({ type: ... })` (`core/intents.ts`). An intent is a table offer (`core/table.ts`: `offers()` says who may press it and when, `perform()` builds the plan) or a host `admin_*` intent; `dispatch()` refuses anything not on that person's screen. A new button = an offer + a `perform` case (+ its shape in `parseIntent`). `core/boundary.test.ts` fails if a screen imports an action. Writes are still applied by the clients until the room server lands (`ARCHITECTURE.md`).
- One commit path for plans: `core/commit.ts` (claims), used by the tests, the simulator and (next) the room server.
- Settings live in `games/{room}/settings`; read them through `config()` (`core/settings.ts`).
- Actions that move chips claim a path first (`tx.claim(...)`, e.g. `currentTurn/resolvedBy` in `resolveDuel`): applied as a set-if-empty transaction, so a double-tap, a timer racing a button, or a second host tab can't charge/pay twice. New chip-moving actions should do the same.
- Duel help text is generated from settings + the turn (`web/src/ui/notes.tsx`) — don't hard-code rules in copy.
- Target-legality helpers live in `core/rules.ts` (`reprisalTargets()` / `gambitTargets()` / `topTiedBidders()`): use them instead of re-filtering gifts.
- Never put user/AI text into `dangerouslySetInnerHTML`; let React escape it.
- Avatars and icons are sticker pictures (`web/public/art`, made with `tools/art`, see its README): a player picks an avatar when they join (`core/avatars.ts`, `players/{id}/avatar`); emoji in the React app render as icons through `<Icon>`/`<Glyphs>` (`ui/icons.tsx`, alt text = the emoji, so role names in tests are unchanged).
- No em dashes in anything players see (screens, mascots, rule pages, critic replies): `web/src/ui/copy.test.ts` checks.

## Economics

`ECONOMICS.md` + `web/src/sim/` (in-memory games through the real engine): the chip ledger must balance; incentive findings are pinned in `web/src/sim/economics.test.ts`. Key fact: leftover chips are worthless at the end, but the standings rank by chips — which of the two players play for flips the equilibrium. A rule change that moves chips or bids should re-run `npm run analyze:economics`.

## Simulated guests: the table (`web/src/sim/table*.ts`, showrunner's virtual world)

Showrunner's Simulation Lab (`../showrunner`, `simlab_adapters/yule_duel/party.py`) seats minds in a
virtual living room to play this game as people would; this repo gives it the table:
- `core/table.ts`: what each person can press right now (`offers(game, who, now)`: their phone's
  buttons, or the host's console incl. the Manual Override) and `perform()` (the same plan the click
  runs). Anything that drives the game as people do should go through these.
- `core/hostauto.ts`: the host page's automatic steps (party-game phases, the duel vote, the wheel,
  in-app duel winners, ties) as data; `useHostAutomation` arms a timer per step, the simulated table
  runs them on its clock. A new automatic host step goes here, not in the hook.
- `core/screens.ts`: the TV and a phone in words, sealed bids sealed.
- `sim/table.ts` (engine, virtual clock, seeded) and `sim/live.ts` (the real host page and phones in
  a browser against the emulators) behind one JSON-lines protocol (`sim/table-bridge.ts`);
  `sim/world.ts` holds the people's skills, tastes and hands, and the scripted patient host.
- Tests: `sim/table.test.ts`, `core/hostauto.test.ts`, `e2e/table-live.spec.js` (runs
  `sim/live-check.ts` against the served build).
- Watch a party in showrunner's Studio: `python -m showrunner.studio ../jingle-brawl` (from
  `../showrunner`) → Watch a party. `showrunner.json` here tells it this is a game and which Lab adapter
  and minds play it; on the real app the tiles are the pages' screenshots (`frames` on the bridge).

## Game Rules Reference

Authoritative rules: `Jingle Brawl V2.md` (player-facing: `rules.html`, `quick-reference.html`). The written rules were synced to the app on 2026-10-18 (`CODE_REVIEW.md` → "Rules vs code"); when behaviour changes, update all four rule docs (`Jingle Brawl V2.md`, `web/public/JingleBrawlV2.html`, `web/public/rules.html`, `web/public/quick-reference.html`). `GAME_FLOW_AUDIT.md` documents scenario status. `AI_ART_CRITIC.md` covers the critic backends.

## Testing Notes

Playwright runs sequentially (1 worker) against the emulators. Tests drive real host + phone pages and assert invariants (`e2e/helpers.js`). **Every bug fix or new code path must ship with a test that fails without it** (`e2e/regressions.spec.js` for fixes). See `TESTING.md`.
