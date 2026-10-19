# Jingle Brawl — Code Review & Improvement Roadmap

_Review date: 2026-06-24. Scope: full solution — security, correctness bugs, best
practices, scalability for concurrent games, and de-monolithing the HTML files._

## 2026-10-18 review — bugs found and fixed

A full read of host.html, player.html, the critic backends, functions and tooling.
Each fix has a test in `e2e/regressions.spec.js` (written first and confirmed failing
on the old code for the first 11), or in `tests/` / `functions/index.test.js`.

**Game logic (host.html / player.html)**
1. **Duplicate `executeReprisal` / `skipReprisal`** in host.html — the later copy
   silently replaced the real one: no affordability check (negative chips), naughty
   bumped twice, chain depth double-counted, Misfit targets became a duel vs "misfit"
   (stuck turn), modal never closed. Duplicate removed.
2. **Reprisal could target the loser's own gift** (self-duel) and **Gambit could target a
   Misfit** (no defender → stuck). Now `reprisalTargets()` / `gambitTargets()` on both sides.
3. **Counterfeit Canvas froze after the first artist.** The turn-rotation listener was
   only attached if an inline `<canvas>` happened to exist (it doesn't during role reveal).
   Also three callers raced to start turns (would double-advance once it did work). One
   listener now drives rotation; timer advances are guarded by the turn they were set for.
4. **Draw Off wiped an artist's drawing** whenever a rival submitted (canvas re-init on
   every minigame update). Also undo/resize produced a transparent PNG (black to vision models).
5. **3+-way tie, yield, with a minigame → infinite minigame loop** (`openerYields` re-routed
   the minigame's own result back into a new minigame). Phones also always sent the first two
   tied bidders by key order — now a random pair, per the rules.
6. **Minigames could auto-start twice** (async existence check raced the first start).
7. **Yield duels charged half the bid rounded UP**; the V2 rules say rounded down.
8. **Phones showed the wrong duel verdict** (guessed from chip deltas: pot 1 + dividend 1
   looked like a win). The host now publishes `games/{room}/lastDuel`.
9. **Starting Chips setting was never saved** (added after the Firebase copy was taken).
10. **Admin panel Misfit moves** used `ownerId: null` instead of `'misfit'` → orphaned gifts
    the Misfit Lottery never found. Removed the dead "Toggle In Bag" control (wrote an unused `bag` node).
11. **Duel vote showed "Error submitting vote"** and **Gambit "Cancel" did nothing**
    (calls to non-existent `updateTurnDisplay()` / `renderPlayerUI()`).
12. Endgame auction slider reset on every rival bid; sealed-bid counts included the active
    player; the game never moved to `ended` when the last endgame duel resolved it.
13. Minigame card never hid afterwards and old strokes leaked into the next minigame;
    Counterfeit undo didn't disappear on other phones.
14. Rejoin button broke for names with an apostrophe (`O'Brien`); AI critic text and the
    host event log are now escaped before `innerHTML`.
15. Phone rule blurbs hard-coded "Santa Tax = 1 if pot ≥ 3 / +1 dividend" — now generated
    from the game's settings.

**Tooling / infra**
- **Tests created real anonymous users in the PRODUCTION Auth project** — only the
  database was routed to the emulator. Auth now uses the Auth emulator on localhost
  (`connectGameAuth()`); the run had started failing with `auth/too-many-requests`.
  Consider deleting the stray anonymous users in the console.
- `test-game.html` could not write under the rules (no auth) — fixed.
- `firebase.json` hosting published `functions/`, `e2e/`, reports, logs, the design-handoff
  folder and a zip — now ignored. `node_modules/` added to `.gitignore`.
- **AI critic (prod) was broken:** `critiqueDrawingGemini` used `gemini-2.0-flash-001`
  (retired 2026-06-01) through `@google-cloud/vertexai` (generative module removed
  2026-06-24). Migrated to `@google/genai`; model IDs are now env config
  (`GEMINI_MODEL`, `OPENAI_VISION_MODEL`). Functions runtime Node 20 (EOL) → 22.
  **Needs `firebase deploy --only functions`, and confirm the Gemini model ID** (sources
  disagreed; default `gemini-3.8-flash`).
- `ai_art_critic.py`: Ollama score "85/100" parsed as 100; GPT-5 rejects `max_tokens`;
  empty `WINNER:` crowned the first player.

### Found by the agent harness (2026-10-18, later)
| Bug | Fix | Test |
|---|---|---|
| Double-tapping a host button resolved/started a duel twice: winner paid twice, bids and yield halves charged twice | `claimTurn()` transaction in `startDuel`, `openerYields`, `recordDuelWinner` | 4 × "double-tapping …" |
| Duel help text promised a Reprisal after a Gambit and told spectators every loser gets +1 | `reprisalNote()` / `spectatorLoserNote()` from settings + turn | "duel help text matches the duel" |
| Base Challenge Cost "0 + Naughty" never worked: `getMinCost` used `baseCost || 1` | `??` in `firebase-config.js` | "Base Challenge Cost 0 + Naughty …" |
| All-pass Misfit auction: Reveal gave the HOST a Choose-Defender modal; the active player never picked on their phone | `revealEndgameAuctionBids` routes all-pass to `endgame_nobids` | "endgame auction where everyone passes …" |
| Host Draw/Endgame buttons + stats ignored gift-only changes (stale after admin edits; flaky test) | gifts listener calls `updateControls()` | endgame tests (were order-dependent) |
| Coins flew from screen centre (host) / to the middle (phone), not loser → winner | `chip-fx.js` `flyChips()` + "from/to X" on the verdict | "won chips fly from the loser to the winner" |

Rule recorded from code: a Reprisal that targets a Misfit claims it with no duel (your current gift goes to the Misfit pile).

### Rules vs code — resolved 2026-10-18: the written rules now follow the app
Kevin chose to keep the app's behaviour and update `Jingle Brawl V2.md`, `JingleBrawlV2.html`,
`rules.html` and `quick-reference.html` to match. What changed in the text:

| Topic | Old V2 text | Now (= code) |
|---|---|---|
| Reprisal after Grinch's Gambit | allowed | none for either player (commit `7b13ca2`, "attacker's risk") |
| Reprisal after yield / tie-break / endgame duels | allowed after any sealed-bid duel | none — only after Challenger-vs-Opener sealed-bid duels |
| Reprisal targets | any gift but the one lost | also never your own gift |
| Reprisal chain (default) | 2 max | 1 / no chaining (setting) |
| Naughty level after first duel | +1 | set to the highest bid; +1 afterwards |
| Santa Tax (default) | 1 chip if pot ≥ 3 | winner capped at 2 once pot ≥ 2 (settings) |
| Loser's Dividend (default) | every loser | Defenders only (setting) |
| Max bid | none | 4 (setting) |
| Tied top bid | tie-break, then yield choice | Opener chooses fight/yield first, then random pair or mini-game |
| Swap with 2 gifts | loser chooses which to Misfit | loser's original is auto-Misfited |
| Head Elf rotation | Misfit loser becomes Head Elf | table play only; the host screen draws |

Still not modelled: chip supply is not conserved (tax deletes chips, the dividend mints
them — there is no stored bank).

Also still open from the June review: client-authoritative writes (#3), readable sealed
bids (#2), unauthenticated paid AI endpoints (#5), no game GC (#10), de-monolith (#11).

## Progress (updated 2026-07-07)

**Done (safe, non-breaking — smoke-tested):**
- **#1 (partial)** — `database.rules.json` created + wired into `firebase.json` +
  un-ignored in `.gitignore`. Root is locked, `games` stays readable/writable per
  room, `hostId`/`createdAt` are immutable, whole-game deletion is blocked, and
  `chips`/`bid amount`/`name` are range/type-validated. **Needs `firebase deploy
  --only database` — verify in a test project / the console Rules simulator first.**
- **#3 (partial)** — chip values are now bounded by rules (0–9999), a first anti-cheat
  guardrail without the full host-authority refactor.
- **#7** — `createGame` now claims its room code with an atomic transaction (writes
  only if the code is free), so a collision can never wipe a live game.
- **#9** — player names are sanitized at the source in `joinGame` (strips `<>`, caps
  at 60 chars) and escaped at the roster sinks; verified an `<img onerror>` name no
  longer executes.
- **P3 hygiene** — removed the stray `Untitled` scratch file.

**Done — #4 Anonymous Auth + ownership rules (code complete, smoke-tested, inert until activated):**
- Added `firebase-auth-compat` + `signInAnonymously()` (gated via an `authReady`
  promise `await`-ed before the first write) to both files. `createGame` stamps
  `hostUid = auth.uid`; `joinGame`/bots stamp `ownerUid = auth.uid`.
- `database.rules.json` now scopes writes by identity: a player may only write a
  `players/$pid` node they **own** (`ownerUid === auth.uid`); the host may write any
  player (`auth.uid === hostUid`); shared client-authoritative paths (`currentTurn`,
  `gifts`, `minigame`, …) still accept any authenticated user (that's the #3 refactor,
  still pending). All writes now require `auth != null`.
- **Multi-tab testing preserved:** ownership keys off `ownerUid`, not `uid === playerId`,
  so one browser's single uid may own many players. For per-tab *distinct* identities
  (to exercise the rules), `?testId` tabs use SESSION (per-tab) auth persistence.
- **To activate:** (1) Firebase console → Authentication → Sign-in method → **enable
  Anonymous**; (2) verify rules in the console **Rules Playground** / emulator; (3)
  `firebase deploy --only database`. Until then the app runs exactly as before.

**Deferred (need a decision, console setup, or a larger refactor):**
- **#2 full bid privacy / #3 host-authority refactor** — deep changes to game-critical
  write flows; require auth first + playtesting.
- **#5 AI endpoint hardening** — App Check / rate limiting need console + reCAPTCHA
  setup and a functions deploy.
- **#6 remove client OpenAI key** — left as-is to avoid breaking the bring-your-own-key
  local/Ollama dev path (the key is already never persisted to Firebase).
- **#10 GC**, **#11 de-monolith**, **#13 CI** — larger separate efforts.

---

## TL;DR

The game works, but it is built on a **fully client-authoritative, zero-auth trust
model with no database security rules**. Multiple games can already coexist under
`games/{roomCode}`, but nothing isolates or protects them: any visitor can read or
overwrite any game, and any player can edit their own chips, read other players'
"sealed" bids, or wipe the entire database. Closing these gaps is the prerequisite
for running real concurrent games. The two HTML files (6.6k + 4k lines) also need to
be broken into modules with a build step.

Priorities: **P0 = security/integrity, P1 = correctness & cost-abuse, P2 =
architecture/de-monolith, P3 = quality/DX.**

---

## P0 — Security & game integrity (do first)

### 1. No Firebase Realtime Database security rules
`firebase.json` has no `database` section and there is no `database.rules.json`. The
RTDB is wide open: **any internet client can read/write/delete every game**, including
`db.ref('games').remove()`. This is the single biggest issue.
- Add `database.rules.json` and wire it in `firebase.json`.
- Validate writes per-path: a player may only write their own
  `players/{playerId}` node and their own bid; only the host may write `phase`,
  `currentTurn` resolution, `gifts`, and other players' chips.
- Note: `.gitignore` ignores `*.json`, so add `!database.rules.json` or the rules
  file will be silently untracked.

### 2. Sealed bids are not sealed
Bids are written in plaintext to `games/{room}/currentTurn/bids/{playerId}` with open
read rules (`player.html:2483`). Any player can read everyone else's bid before the
reveal — this defeats the core sealed-bid auction mechanic. Even with rules locking
reads, a malicious client could still see writes. Options:
- Restrict read on `currentTurn/bids` until host flips a `revealed` flag (rules-based).
- Or commit–reveal: players write a salted hash of their bid, reveal the salt at
  reveal time. Stronger, more work.

### 3. Client-authoritative everything (cheating)
All game logic — chip math, duel resolution, payouts, Santa Tax — runs in the
browser, and clients write results directly to Firebase
(e.g. player edits `players/{id}/chips` at `player.html:2643`). A player can set their
chip count to anything. There is no server referee.
- Short term: lock down with rules so only the host writes authoritative state, and
  make the host the single source of truth for resolution.
- Long term: move resolution into a Cloud Function / trusted server so even the host
  can't trivially fabricate state, and clients only submit *intents*.

### 4. No authentication at all
There is no Firebase Auth. `hostId`/`playerId` are just `'host_' + Date.now()` /
`'player_' + Date.now() + random` strings stored in localStorage — trivially
spoofable, and rules can't bind a write to an identity without auth.
- Add Firebase Anonymous Auth at minimum; key rules off `auth.uid`. Store the host's
  uid on the game and gate host-only writes on `auth.uid === game.hostId`.

### 5. Unauthenticated, unthrottled, paid AI endpoints
`critiqueDrawing` / `critiqueDrawingGemini` use `Access-Control-Allow-Origin: *` with
no auth, no rate limit, no App Check (`functions/index.js:29,177`). Anyone who finds
the URL can burn your OpenAI/Vertex budget at will.
- Add Firebase App Check, require auth, add per-IP/per-room rate limiting, cap image
  size, and set function `maxInstances`/budget alerts.

### 6. OpenAI API key handled in the browser
The OpenAI key is typed into a host form, kept in `sessionStorage`
(`host.html:1780`), and POSTed in the request body to non-cloud-function URLs
(`host.html:4259`). Keys in client storage/requests leak easily.
- Never accept a user OpenAI key client-side. Use the server-side secret
  (`OPENAI_API_KEY`) exclusively via the Cloud Function, and remove the key field.

---

## P1 — Correctness bugs & abuse

### 7. Room-code collision / overwrite
`createGame` generates a 4-char code (32-symbol alphabet ≈ 1M combos) with
`Math.random` and **never checks if the room already exists** before
`db.ref('games/'+roomCode).set(...)` (`host.html:1745,1794`). A collision silently
wipes an in-progress game. Codes are also short/enumerable, enabling griefing once a
game is found.
- Use a transaction to claim an unused code (retry on collision); consider 5–6 chars.
- With rules in place, also prevent overwriting an existing non-empty game node.

### 8. `Math.random` for identifiers
Room codes and IDs use `Math.random` — fine for UX, not for anything security-adjacent
once rules/identity matter. Move identity to `auth.uid`; keep codes as a human-friendly
*join* token only.

### 9. XSS surface from `innerHTML` + user input
160+ `innerHTML` assignments across the two files, several interpolating
player-supplied names (e.g. rejoin/lobby rendering). A player named
`<img onerror=...>` can run script in the host/other players' tabs.
- Escape all user-derived strings before injecting, or build nodes with
  `textContent`. Centralize an `escapeHtml()` helper.

### 10. Disconnect/rejoin edge cases
`onDisconnect().set(false)` marks players disconnected, but there's no TTL/cleanup of
abandoned games, and host disconnect has no failover. Long-lived `games/*` nodes
accumulate forever.
- Add a scheduled Cloud Function to GC games older than N hours, and a host-disconnect
  grace/heartbeat.

---

## P2 — Architecture: de-monolith & scale

### 11. Monolithic HTML (6,620 + 4,042 lines)
`host.html` and `player.html` each bundle HTML, CSS, and all JS inline (162 `<script>`/
`<style>`/firebase refs in host alone, 118 + 62 top-level functions). This is the main
maintainability blocker and a big driver of the bugs above (logic duplicated across two
files: `GAME_CONFIG` referenced 44× in host, 21× in player).
Recommended target structure:
```
src/
  shared/        gameConfig, rules engine, firebase client, escapeHtml, types
  host/          host UI + controllers
  player/        player UI + controllers
  styles/        extracted CSS (yd-design.css already a start)
functions/       cloud functions (resolution + AI critique)
database.rules.json
```
- Introduce a build step (Vite). Extract the shared **rules/resolution engine** into
  one module imported by host, player, and (eventually) the server referee — this
  kills the duplication and makes the logic testable in isolation.
- Migrate Firebase v9 compat (`db.ref(...)`) to modular v9 imports for tree-shaking.

### 12. Concurrency is data-modeled but not isolated
Multiple games already key off `games/{roomCode}`, so simultaneous games *function*.
What's missing for running them "securely at the same time" is items 1–6: per-game
rules isolation, identity, and cost controls. No schema change needed — just
enforcement. Add a lightweight index/limit on concurrent active games if you host
publicly.

### 13. No CI / typed contracts
Plain JS, no types, no CI. The Firebase data shape is an implicit contract spread
across both files.
- Add TypeScript (or JSDoc types) for the game-state schema; run lint + Playwright in
  CI (GitHub Actions).

---

## P3 — Quality, testing, DX

- **Inline `onclick` handlers** (71 host / 40 player) couple markup to globals and add
  to the XSS surface; move to `addEventListener` after modularizing.
- **Sparse error handling**: only 11 `catch` blocks in host, 1 in player; Firebase
  awaits and `fetch` calls to the AI critic can reject unhandled. Add try/catch + user
  feedback around all network writes.
- **Tests**: only 2 Playwright specs (lobby/join, bidding/duel). Add coverage for
  reprisals, ties/tiebreakers, endgame steals/lottery, and the rules engine once
  extracted (fast unit tests vs. slow E2E).
- **Repo hygiene**: stray `Untitled` file; `node_modules/` and many `.md`/`.py` files
  live in the hosting root (mostly ignored, but tighten). Confirm
  `firebase-config.js` exposure is acceptable (it is for Firebase web *only once rules
  exist*).
- **AI critique parsing** is brittle regex on `DESCRIPTION:/SCORE:/CRITIQUE:`; request
  structured JSON output instead.

---

## Suggested execution order
1. `database.rules.json` + Firebase Anonymous Auth (items 1, 4) — unblocks everything.
2. Lock bids + chips/host-only writes (items 2, 3).
3. Secure AI endpoints + remove client key (items 5, 6).
4. Room-code transaction + XSS escaping (items 7, 9).
5. Extract shared rules engine + Vite build; then split host/player (items 11–13).
6. Game GC, CI, expanded tests (items 10, 13, P3).
