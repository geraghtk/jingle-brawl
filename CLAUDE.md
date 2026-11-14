# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Jingle Brawl** (also called "Yuel Duel") is a real-time multiplayer party game for gift exchanges. Players bid Jingle Chips on gifts, duel to win them, and can challenge each other with special actions like Grinch's Gambit and Reindeer Reprisals. A tied top bid is settled by a party mini-game the whole table plays ("Sketch-Off", "Whack-an-Elf", `web/src/core/party.ts`). Duels can be played in the room or in the app; one in-app duel game is "AI Draw Off" (`core/drawoff.ts`): the two duellists draw, an AI Art Critic picks the winner (`AI_ART_CRITIC.md`).

## Commands

```bash
npm install                      # Playwright
npm run dev                      # the room server (:8787, rooms.db) + the app with hot reload (:5173, phones on Wi-Fi too)
npm run server -- --db :memory:  # the room server alone (TS via vite-node)
npm run build && npm start       # build the app + the server bundle (dist-server/), run it (rooms in rooms.db)

npm run test:e2e                 # full Playwright suite on the room server (builds, serves, --debug-api)
npm run test:flows               # game-flow permutations only
npm run test:e2e:ui / :headed    # visual debugging
python -m unittest discover -s tests   # ai_art_critic.py unit tests
npm run test:unit                # rules engine + economics (vitest: web/src/core, web/src/sim)
npm run analyze:economics        # game-theory report: deviation matrix, chip flows (see ECONOMICS.md)
npm run analyze:gremlin          # Pass the Gremlin: thousands of simulated rounds (who gets doomed, passes, sudden death)

npm run agents:watch             # persona agents play a real game, streamed to one window
npm run agents:sweep             # all settings presets × N headless games → agents/reports/…/summary.md
npm run test:agents              # persona unit tests

npx vite-node web/src/sim/play-table.ts -- [--live] [--headed] [--players 4]   # a party at the table: engine or the real app
npx vite-node web/src/sim/table-bridge.ts       # the table over JSON lines (showrunner's Simulation Lab drives it)

python ai_art_critic.py          # optional local AI critic (Ollama llava / OpenAI) on :5050

npm run build && npm start       # production: one Node process serves the app and the rooms (DEPLOY.md: Docker, env, limits, logs, /metrics)
```

## Local Development

`npm run dev`, then (three kinds of screen):
- Host console: `http://localhost:5173/host.html` (on the host's own device; the host plays too: "Join as a player too", then their phone has two tabs, "My game" and "Host" (the console in a same-origin frame, `player/SeatTabs.tsx`; a dot when the game waits on the host, `core/nudge.ts`))
- TV (the shared screen, its own device): `http://localhost:5173/tv.html?room=ABCD` (public view, no controls; linked from the console)
- Player tabs: `http://localhost:5173/player.html?room=ABCD&testId=1&name=Alice` (join links and QR codes use this PC's Wi-Fi address)

`?testId=N` isolates localStorage, so one browser can simulate many players.

## Architecture

**Stack:** React 18 + TypeScript + Vite (`web/`: entries `web/host.html`, `web/player.html`, static files in `web/public` incl. the rule pages) and our own Node room server (`server/`: WebSockets, SQLite via `node:sqlite`, HMAC session tokens), which is the referee and serves the built app. Optional local Flask critic for AI critique. Firebase and the original vanilla pages were retired on 2026-11-07 (`ARCHITECTURE.md`).

**Key files:**
- `web/src/core/` — the rules engine, pure: actions are `(game, …) => Plan` built with `Tx` (`tx.ts`); `actions.ts`, `duel.ts`, `admin.ts`, `voting.ts`, `party.ts`/`partyflow.ts` (Sketch-Off + Whack-an-Elf), `duelgame.ts` (duel picker, wheel, in-app duel games), `scoring.ts` (Party Score, spectator bets), `settings.ts` (`DEFAULTS`, `config()`), `rules.ts` (target helpers)
- `web/src/host/` — the host console and the TV's screens (`TurnPanel.tsx`, `Sidebar.tsx` admin, `PartyPanel.tsx`, `DuelModal.tsx`); `HostProvider tv` hides every control (`web/src/tv/` is the TV page)
- `web/src/player/` — phone: join/rejoin, sealed bids (`BidPanel.tsx`), opener choices and duels (`StatusCard.tsx`), reprisals, party games (`PartyCard.tsx`), votes
- `web/src/net/` — the screens' side of the room server: `client.ts` (`RoomClient`), `hooks.tsx` (`useRoom`, `useServerOffset`), `api.ts` (create / lobby / join), `protocol.ts`, `lan.ts` (the TV's Wi-Fi join link)
- `web/src/ui/` — shared UI: `mascots.ts` (Santa / elf / reindeer pop-ups, `?mascots=0` disables), `whack.tsx`, `icons.tsx`, `chips.ts`, `notes.tsx` (duel help text)
- `server/src/critic/` — AI Draw Off's critic (OpenAI / Gemini with the host's own key, held in server memory only; the local critic; a fake one for tests: `JB_FAKE_CRITIC=1`)
- `ai_art_critic.py` — local Flask critic: `/critique`, `/witty-verdict` (Draw Off verdict), `/health`
- `jingle_brawl_env.py` — Deep RL self-play simulation for balance testing
- `e2e/` — Playwright specs + `helpers.js` (invariants; `dbGet`/`dbPut`/`dbPatch` through the server's debug API)
- `agents/` — persona-agent play-testing harness (playability + settings/balance sweeps); see `agents/README.md`
- `server/` — the room server (own `tsconfig.json`: no DOM, so `web/src/core` must stay DOM-free). `room/actor.ts` is one room's referee (intent → checks → `dispatch` → commit → save → timers from `hostAuto` → each screen its `projectGame` view), runtime-agnostic (Clock/Store/Conn); `room/rooms.ts` the registry + token secret; `store/` SQLite (`node:sqlite`) or memory; `node/` HTTP API (`/api/rooms`, join, lobby), WebSocket (`/ws`), static `dist/`, and `--debug-api` (localhost-only test helpers). Wire types: `web/src/net/protocol.ts`. Screens talk to it through `web/src/net/` (`RoomClient`, `useRoom`; first a whole view, then only what changed, `core/diff.ts`).

**Real-time data flow:** the host creates a room (`POST /api/rooms` → code + host token), phones join (`/join` → player token), every screen opens one WebSocket and sends intents; the room applies them and sends each screen its own view (then only what changed). Key paths: `players`, `gifts`, `currentTurn` (turn state machine), `phase`, `settings`, `minigame`, `lastDuel` (outcome the host publishes so phones show the right verdict).

**Game state machine:**
```
lobby → main (draw → bidding → reveal → [duel_pending | yield_pending | tiebreaker_pending → minigame_*] → duel → reprisal?) → endgame (endgame_choice → steal | auction) → ended
```

**Conventions that matter:**
- A Misfit-pile gift has `ownerId: 'misfit'` (+ `misfitOrder` for FIFO). Never `null`.
- "In the bag" = player has no `giftId`; there is no separate bag list.
- **Buttons send intents, never plans:** `send({ type: ... })` (`core/intents.ts`). An intent is a table offer (`core/table.ts`: `offers()` says who may press it and when, `perform()` builds the plan) or a host `admin_*` intent; `dispatch()` refuses anything not on that person's screen. A new button = an offer + a `perform` case (+ its shape in `parseIntent`). `core/boundary.test.ts` fails if a screen imports an action. The room server applies them (`server/`).
- One commit path for plans: `core/commit.ts` (claims), used by the tests, the simulator and (next) the room server.
- Settings live in `games/{room}/settings`; read them through `config()` (`core/settings.ts`).
- Actions that move chips claim a path first (`tx.claim(...)`, e.g. `currentTurn/resolvedBy` in `resolveDuel`): applied as a set-if-empty transaction, so a double-tap, a timer racing a button, or a second host tab can't charge/pay twice. New chip-moving actions should do the same.
- Duel help text is generated from settings + the turn (`web/src/ui/notes.tsx`) — don't hard-code rules in copy.
- Target-legality helpers live in `core/rules.ts` (`reprisalTargets()` / `gambitTargets()` / `topTiedBidders()`): use them instead of re-filtering gifts.
- Never put user/AI text into `dangerouslySetInnerHTML`; let React escape it.
- Avatars and icons are sticker pictures (`web/public/art`, made with `tools/art`, see its README): a player picks an avatar when they join (`core/avatars.ts`, `players/{id}/avatar`); emoji in the React app render as icons through `<Icon>`/`<Glyphs>` (`ui/icons.tsx`, alt text = the emoji, so role names in tests are unchanged).
- The server's guards (`server/src/node/`): per-address limits (`limits.ts`; `--debug-api` turns them off, so dev and the suite never hit them), security headers + CSP (no inline scripts or `on*=` handlers in pages: use a file), JSON logs + `/metrics` (`log.ts`; never names, drawings, keys or addresses). Rooms are deleted 2 h after the end / 24 h idle (`Rooms.purge`), so codes come back: tokens carry the room's `born` and the socket checks it.
- No reduced-motion mode: the game ignores the OS `prefers-reduced-motion` setting everywhere (its animations are the feedback; a site-wide rule once blanked Ring Toss's stars and smoke). Don't add `@media (prefers-reduced-motion)` rules or `matchMedia` checks.
- No em dashes in anything players see (screens, mascots, rule pages, critic replies): `web/src/ui/copy.test.ts` checks.

## Mini-game platform (`web/src/core/minigames/`)

From `yule_duel_agent_spec_updated.md` (M1/M2). A game is a pure `GameDefinition` (`contracts.ts`): config, initial state, `validateAction`, `reduce`, `onDeadline`, `offers` (for the simulator), a view per player / TV / host, and a `GameResult`. Register it in `registry.ts`; the runtime (`runtime.ts`) needs no change. The running game is `game.arena` (server only); screens get `game.arenaView`. Results are explicit (`resolved` / `requires_tiebreaker` / `no_contest` / `cancelled`); a game never moves chips or gifts: `handoff()` gives the duel its winner only for a `resolved` result with a nominated winning duelist, else the host picks. With duelists, a mode that can't always resolve them is refused at start (`canAlwaysResolve`). Game state is stored like the rest of the game (empty objects vanish): read it defensively. Registered games are in-app duel games (2 at the table: `direct_duel`; 3+: `party_with_duelists` if supported). Game #1: Pass the Gremlin (`gremlin/`, UI `player/GremlinCard.tsx`, `host/ArenaStage.tsx`; challenges in `gremlin/challenges.ts`: quick_math, memory_flash, perfect_timing, the host's pick via `settings.gremlinChallenges`; perfect_timing trusts a phone's tap time only within `TIMING_LATENCY_MS`; art `web/public/art/gremlin/`, sounds `ui/sound.ts`); tests in `gremlin/*.test.ts`, `server/src/room/gremlin-room.test.ts`, `e2e/gremlin.spec.js`. M3 (challenges, art, sound) done 2026-11-08. Game #2: Reindeer Ring Toss (`ringtoss/`: every target 1, each full set (all five since your last) +`setBonus`; the scored round is unlimited rings to the clock, `batch` at a time with a `reloadMs` reload: read rings in hand with `inHand`, screens show the refill from `reloadAt`; UI `player/RingTossCard.tsx`: the phone is only the controller, the whole screen a flick surface; `host/RingTossStage.tsx`: the art `web/public/art/ringtoss/` from `tools/art/reindeer.py`, PEGS measured from it (move them together); rings in the air are moved per frame outside React (`Flights`: read `live.current`, never a stale closure); a mini-game is full screen on the TV and, by default, the console: `host/ArenaFull.tsx`; the show (runs of ringers, the leader, the last-five count, the one target a set still needs) is derived from the view in `ringtoss/flair.ts`: it never scores), built on theme-free throw engines in `minigames/engines/` (`gesture.ts`: the phone sends its raw swipe path in screen heights, the server makes the launch: the whole stroke aims, turned down by `AIM_GAIN`, the release speed is the power; `projectile.ts`: deterministic 2.5D flight (where it lands); how it looks in the air is `arcFor`/`arcAt` there: one gravity for every throw, air drag, a ringer drops onto its tip; the TV adds a spin (`tumble`) and the screen's flight time is `visMs`; `targets.ts`: peg scoring); `testgames/snowballs.ts` proves another game reuses them. Game #3: the Great Gift Conveyor (`conveyor/`, UI `player/ConveyorCard.tsx`, `host/ConveyorStage.tsx`), mode `captain_teams`: `startDuelApp` deals the teams with `engines/teams.ts` (balanced, splitting past teammates: `game.teamPairs`) and the game runs a line on `engines/workflow.ts` (one holder per item, one item per station, a dropped phone's item to the next person). A result may carry `rewards` (a recommendation); `handoff()` applies them as Party points. Views drop nulls and empty maps: compare view fields with `!= null` and normalise a view on read (`ringView`). **The Game Lab** (`minigames/lab.ts`, host `LabModal.tsx`): in the lobby the host starts any registered game with chosen players / mode / duelists (`lab_start`, `lab_end`; `currentTurn.lab`, the room stays in `lobby`); practice only (no hand-off, no rewards, no `teamPairs`), the result stays until Play again / Back to the lobby; `lab_*` aren't in `offers()` so the simulated host never wanders in.

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
  a browser against the room server) behind one JSON-lines protocol (`sim/table-bridge.ts`);
  `sim/world.ts` holds the people's skills, tastes and hands, and the scripted patient host.
- Tests: `sim/table.test.ts`, `core/hostauto.test.ts`, `e2e/table-live.spec.js` (runs
  `sim/live-check.ts` against the served build).
- Watch a party in showrunner's Studio: `python -m showrunner.studio ../jingle-brawl` (from
  `../showrunner`) → Watch a party. `showrunner.json` here tells it this is a game and which Lab adapter
  and minds play it; on the real app the tiles are the pages' screenshots (`frames` on the bridge).

## Game Rules Reference

Authoritative rules: `Jingle Brawl V2.md` (player-facing: `rules.html`, `quick-reference.html`). The written rules were synced to the app on 2026-10-18 (`CODE_REVIEW.md` → "Rules vs code"); when behaviour changes, update all four rule docs (`Jingle Brawl V2.md`, `web/public/JingleBrawlV2.html`, `web/public/rules.html`, `web/public/quick-reference.html`). `GAME_FLOW_AUDIT.md` documents scenario status. `AI_ART_CRITIC.md` covers the critic backends.

## Testing Notes

Playwright runs sequentially (1 worker) against the room server (`--debug-api`). Tests drive real host + phone pages and assert invariants (`e2e/helpers.js`). **Every bug fix or new code path must ship with a test that fails without it** (`e2e/regressions.spec.js` for fixes). See `TESTING.md`.
