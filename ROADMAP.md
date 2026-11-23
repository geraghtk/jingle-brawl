# Jingle Brawl: where we are, and the road to a public release

(The plan Kevin approved on 2026-11-13. Update the Progress log at the end as steps land.)

## Context

Kevin wants Jingle Brawl to be a real product strangers can use, local-first. The architecture plan
(own Node room server, Firebase retired, mini-game platform) finished on 2026-11-07; the production
track since then added hardening, three mini-games and the Game Lab. This plan (1) records where that
track stands, (2) adds what a full review found (five review agents on 2026-11-13: security, platform
code, test gaps, duel-game selection, public-release readiness; the top security findings re-checked
by hand), (3) adds the work to make it look and behave like a professional public app, and (4) adds
one feature: whole-table and team games as normal options for regular duels, so nobody sits out.

Standing rules for every step: each fix or new path ships with a test that fails without it; rule
changes update the four rule docs; chip or bid changes re-run `npm run analyze:economics`; no em
dashes in player copy; no reduced-motion rules.

## 1. Where we are (2026-11-13, branch `review-2026-10-agents`)

| Item | State |
|---|---|
| Architecture plan P0.1 to M2 (intents, projections, room server, TV page, runtime, Pass the Gremlin) | Done, pushed 2026-11-07 |
| P0 hardening (limits, CSP, room TTL, logs + /metrics, connection banner, privacy page, Dockerfile, CI) | Done, pushed 2026-11-08 |
| M3 Gremlin challenges/art/sound; host tab on the host's phone | Done, pushed |
| M4 Reindeer Ring Toss + throw engines | Done, pushed; playtested PC + phone 2026-11-12 |
| M5 Great Gift Conveyor (teams); Game Lab | Done |
| Ring Toss: unlimited rings in batches, full sets | Done on disk, **not committed** |
| Ring Toss: the show (`ringtoss/flair.ts`: snow, trails, nose glow, runs, crown, countdown, sounds) | Done on disk 2026-11-13, tests green, **not committed** |
| Hosting choice (Cloudflare Durable Objects adapter vs one Fly machine / VM) | **Open: Kevin's decision** |
| Real-phone playtests (Gremlin skew, Ring Toss flick, Conveyor pace) | Not started |
| P1: PWA, wake lock, Litestream backups, device matrix | Not started |
| Seat revocation; `JingleBrawlV2.html` re-export; Dockerfile never built locally | Open |

Mini-game spec milestones M0 to M5 are all done; M6+ is the backlog catalogue (mostly captain-team games).

## 2. The road, in order

### Step 0. Commit what is on disk
Two commits on the current branch: (a) Ring Toss batches + full sets, (b) the show. Run the full
Playwright suite first (only `e2e/ringtoss.spec.js` was run today).

### Step 1. Security fixes (before anything is hosted publicly)
Anyone can create a room, so "host-only" means "any stranger".

| # | Sev | Problem | Fix | Files |
|---|---|---|---|---|
| S1 | High | A host admin message with `ownerId`/`pid` `__proto__` writes to `Object.prototype`: every room on the server is corrupted until restart. Confirmed by hand. | `setPath`/`getPath` refuse `__proto__`/`constructor`/`prototype`; existence checks use `Object.hasOwn`; `ID` regex excludes them | `web/src/core/tx.ts`, `intents.ts`, `admin.ts` |
| S2 | High | Settings at room creation are stored unvalidated; `startingChips: 1e9` makes the bid offer list enumerate a billion entries and the process dies. Confirmed by hand. | Whitelist + clamp settings in `rooms.create` (one `sanitizeSettings` in `core/settings.ts`, reused by Setup); bids checked by range, not by enumerating offers | `server/src/room/rooms.ts`, `server/src/node/http.ts`, `core/settings.ts`, `core/table.ts` |
| S3 | Med | Whack-an-Elf score is whatever the phone reports: a tied bidder can send 1000 and win the tie | Cap by elapsed time x max rate now; server-counted taps against the seeded schedule when Whack moves onto the platform (Step 4) | `core/table.ts`, `partyflow.ts`, `party.ts` |
| S4 | Med | One room of 50 players with 600 KB drawings makes ~30 MB of state cloned, projected and rewritten on every intent; host connect parses up to 100k event rows | Drawings stored by reference outside `game` with a smaller cap; `LIMIT 300` on the event read; per-room intent budget | `server/src/room/actor.ts`, `store/sqlite.ts`, `core/intents.ts` |
| S5 | Med | `ai_art_critic.py` binds 0.0.0.0 with open CORS and falls back to the operator's OpenAI key | Bind 127.0.0.1, drop CORS, set `MAX_CONTENT_LENGTH` | `ai_art_critic.py`, `tests/` |
| S6 | Low/Med | 4-character room code is the only gate to watch or join; no way to remove someone | Failed hellos count against the lobby limit; host "Lock the room" switch; **seat revocation** (the long-open item) | `server/src/node/ws.ts`, `limits.ts`, `core/actions.ts`, `host/Sidebar.tsx` |
| S7 | Low | `/debug` ignores `trustProxy`; no Host check (DNS rebinding in dev) | Refuse `--debug-api` with `--trust-proxy` or `NODE_ENV=production`; require a localhost Host header | `server/src/node/http.ts`, `main.ts` |
| S8 | Low | Metrics token compared with `===`; raw `req.name` logged; env-key names differ between `critic/source.ts` and DEPLOY.md | timing-safe compare; coerce name; align the docs | `http.ts`, `actor.ts`, `DEPLOY.md` |

Found sound (no work): intent authorization, per-screen projections, token HMAC, static path guard,
CSP/HTML escaping, no SSRF, debug API absent from shipped paths. Unknown: whether Fly's
`X-Forwarded-For` last hop is the client (check when hosting is chosen).

### Step 2. Whole-table and team games in regular duels (the new feature)

What the survey found: this mostly exists but is hidden. `startDuelApp` (`web/src/core/duelgame.ts:97-123`)
already runs the Conveyor as `captain_teams`, the Gremlin and Ring Toss as `party_with_duelists` (3+
players), and Sketch-Off / Whack-an-Elf with goal `'duel'`; `handoff()` gives the gift to the winning
duelist or captain. But `DEFAULTS.duelAppGames` is `[]`, the tick-boxes live only on the pre-game
`host/Setup.tsx`, there is no way to change them mid-game, and only one game has real teams.

Kevin's decision (2026-11-13): **team games are available for regular duels by default; a game
setting opens it to all whole-table games.**

0. **Bug first (confirmed by hand):** in the `party_with_duelists` branch a duelist whose phone is
   "away" is left out of `participants` (`duelgame.ts:108,118`), so they cannot play and lose the gift.
   The Conveyor branch unions the duelists in (`:113`); do the same. Test in `duelgame.test.ts`.
1. **One setting, `duelTableGames`** in `settings.ts` `DEFAULTS` + `config()`:
   `teams` (**default**: every game with a `captain_teams` mode is offered for regular duels) /
   `all` (also the everyone-plays games: Gremlin and Ring Toss as `party_with_duelists`, Sketch-Off,
   Whack-an-Elf) / `off` (only the in-room games and whatever the host ticks). `duelOptions`
   (`duelgame.ts:39-46`) builds the list from it; the per-game tick-boxes in `host/Setup.tsx` stay as
   the fine control (un-tick one you don't want). AI Draw Off stays opt-in: it needs a key.
   Team games need 4+ at the table (two captains and a teammate each); with fewer they are not offered.
2. **Say who plays.** In the picker (`host/DuelGameStage.tsx`, `player/StatusCard.tsx:369-388`) and on the
   wheel, tag each option from the game's modes: "Teams", "Everyone plays", "Just the two of you".
   Derived from `supportedModes`, never hard-coded.
3. **Change it mid-game.** New host intent `admin_set_duel_games` (offer + `perform` case + `parseIntent`
   shape) and a small editor in `host/Sidebar.tsx`; today nothing writes these after setup.
4. **More team games, so the default is not one game.** Add `captain_teams` to Ring Toss (team total
   decides, captains are the duelists; level: captains throw the extra rings) and to the Gremlin (the
   team holding it at doom loses; sudden death between captains). Reuse `engines/teams.ts`
   `assignTeams` + `recordPairs` exactly as the Conveyor does; the winning team gets `rewards` as Party
   points through `handoff()`. `startDuelApp` picks `captain_teams` when the game supports it and 4+
   are at the table; under `all` with 3 players it falls back to `party_with_duelists`.
5. **Balance check before it ships as default:** Party points now flow in most duels. Re-run
   `npm run analyze:economics` and `npm run agents:sweep`; tune `rewards` if team membership swamps the standings.
6. **Tie-break duels.** `duelOptions` drops app games when `currentTurn.isTiebreak` (line 43): keep that
   (the table has just played a party game) but make it a visible rule in the help text (`ui/notes.tsx`).
7. Rule docs (all four), `ECONOMICS.md` re-run (Party points flow more often), simulator offers
   (`sim/world.ts` hands) so the virtual table plays them.

Tests: `duelgame.test.ts` (defaults, preference weighting, team mode choice), `ringtoss.test.ts` /
`gremlin` tests for `captain_teams`, `e2e/duel-game.spec.js` (default picker shows the tags; mid-game
switch), one e2e per new team mode.

### Step 2b. The economy (review of 2026-11-13: simulated on the real engine, 1000 to 2400 games a cell)

Run on a copy of `web/src/sim/simulate.ts` with added strategies (spend-down, sniping, never-spend,
contrarian voters), checked against auction, contest, sequential-auction, faucet/sink and inequality
models. The team-reward numbers come from a model of team credit, not the Conveyor engine, and the
bots have fixed habits: treat sizes as indications, directions as solid.

**Healthy (leave alone):** the chip ledger balances; no all-zero or all-max bidding equilibrium; no
seat-order advantage (0.03 +- 0.09 gift points); winning early does not snowball; Reprisal is a real
decision (never taking one costs 0.65); mild deflation at every table size.

**Decorative (stop tuning):** tax threshold changes nothing at all; payout cap, dividend rule and base
cost move the chip counter but not who gets which gift. The only levers that change a game are
`maxReprisalDepth` (duels per game 6.2 / 8.3 / 9.6: it is the game-length setting) and very low
starting chips (5 or fewer: 10 to 22% of players go broke).

**Problems, by how much they matter at a party (each needs Kevin's call: they are rule changes):**
1. **Both leaderboards reward not playing.** A player who never spends a chip goes home with a gift
   1.24 points worse, yet tops the chip standings 42% of the time and is Party MVP 34% (fair share
   17%); the most active style gets 2% and 6%. Duelists cannot call their own duel, so Party Score
   falls as you fight more (correlation -0.59).
   - Cheap (screens only): stop ranking the final tally by chips (leftover chips are worthless).
   - Rule change (four docs): **+1 Party point for each duel you fight.** Non-player MVP 34% -> 9%,
     most active 6% -> 27%.
2. **Team rewards are too big for a default.** At the Conveyor's 2 + up to 3 per winning team member,
   team games would supply 93% of Party Score variance and make the most skilled player MVP 51% of
   the time (14% today). Flat 1 point to each winner: 49% of variance, most skilled MVP 26%.
   Recommended with team duels as the default: **flat 1 (or 1 + up to 1), together with the duelist
   point above.** Upside either way: team games halve skill's grip on who gets the gift (best-minus-
   worst duelist 1.32 -> 0.52 gift points), which is good for weak duelists, who today have no reason to bid.
3. **Spectator bets: keep off.** With stakes on, honest calling is not the best play (a lone contrarian
   gains 3.55 chips a game, an abstainer 1.61) and the table burns 11.5 chips. If wanted later: pay
   right callers from the Fund with no stake.
4. **Chips go unspent by construction** (80 to 88% left at the end): only the top bidder pays and a
   winner is refunded. Theory says spend down late; a spend-down bot is the best deviant at every
   table. Not a bug, but it is why chip settings feel inert; worth a line of advice on the phone
   near the end ("chips are worth nothing after the last gift").
5. `ECONOMICS.md` says "bid 3 is stable": no longer reproduces (bid 4 gains 0.47 +- 0.10). Update the
   doc and `economics.test.ts` when the rules above are settled.

**Measure in real playtests** (simulation can't): log every bid with the bidder's chips, ties,
fight/yield, the duel game and who picked it, the winner, calls, team deals, Reprisals offered and
taken; ask each player to rate their final gift 0 to 10 and which number they thought meant winning.

The analysis scripts and their outputs are kept as they were run in `tools/economy-review/` (one-off
scripts: they import the engine by absolute path and are not part of the build or the tests; run with
`npx vite-node tools/economy-review/<name>.ts`). Fold the useful ones (`sim2.ts` strategies,
`sweep.ts`, `flows.ts`, `skill.ts`) into `web/src/sim/` with tests when Step 2b is acted on.

**Decided and built (Kevin, 2026-11-13: "let's make these changes"):** final standings are by Party
Score, not chips; +1 Party point for each duel you fight (`players/{id}/duelsFought`, added in
`resolveDuel`); a flat 1 point to each winning Conveyor team member. Re-run on the engine's own score
(`tools/economy-review/after.ts`, 1000 games a row, six players, fair share 17%): the never-spends
player tops the standings 8% of the time (was 42% by chips), the most active 27% (was 2%); with a
team game in every duel at a flat point, 12% and 25%, and the most skilled seat 27%.

### Step 2c. Spectator bets as a real market (proposal: Kevin wants to keep betting, 2026-11-13)

**Simulated and built (2026-11-13).** Kevin: keep staked betting optional, fix it, decide later whether it
becomes default. The simulation (`tools/economy-review/bets-*.ts`, a tabular learning bettor plus fixed
habits on duels replayed from the real engine) found the proposal below does NOT pass: with no bonus a
rational player never bets, and with no cap "always back the underdog" farms honest callers (12 to 23
chips a game at twelve players). What passed and is now in the engine (`scoring.ts` `BETS`,
`settleBets`, `poolOdds`, `betLine`; `duelgame.ts` `betsOpen`): one pool of the stakes + 1 chip from
the Fund at the house line + that duel's tax; the winner's backers share it, at most 3 back for 1;
the rest waits in `game.betPool`; nobody backed the winner: stakes returned; only with 4+ spectators,
never on a duel the spectators play in. Honest betting beats sitting out (+0.5 to +0.8 chips a game),
the best learned policy gains at most about 1.4 over honest, collusion gains nothing; worth at most
about 0.1 gift points. **Still to do:** live "pays X for 1" and the pool on the phone and TV
(`poolOdds` is ready for it), the four rule docs, a browser test, undo restoring `betPool`, re-run
`analyze:economics` (the pool hands back roughly what the tax burns).

The original proposal, kept for the record:

Today (setting off by default): each call stakes 1 chip, the wrong callers' chips are split between
the right ones, the odd chip and "nobody right" go to the Fund. The review found why it misbehaves:
the leftovers make it negative-sum (abstaining beats betting by 1.6 chips a game) and the crowd piles
on the favourite, so a lone contrarian gains 3.5. Both are fixable, and the second is what a market
is supposed to do once people can SEE the odds. Proposed design, to simulate before building:
1. **A pool with live odds (parimutuel).** All stakes on a duel go into one pool, split between the
   backers of the winner in proportion to their stakes. The phone and TV show what each duelist pays
   right now ("Bob pays 3 for 1"), moving as calls come in: backing the underdog pays more, so the
   payout depends on who is backing whom.
2. **An opening line from performance.** Before anyone bets, the house prices the duel from each
   duelist's record (duels won and lost so far, a neutral 50/50 for a first duel) and seeds the pool
   with a couple of chips from the North Pole Fund laid against that line. The seed makes betting
   positive-sum (so playing beats abstaining) and recycles taxed chips back to the table.
3. **Nothing leaks.** Nobody backed the winner: stakes are returned. Remainders carry to the next duel's pool.
4. **Who may bet.** Never the two duelists. In a team duel everyone is on a team, so no bets (a
   teammate could throw it): the Party point for the winning team is the stake there.
5. **Stake size**: 1 chip, or let a bettor choose 1 to 2 (a bigger say in the odds). Calls still score
   +1 Party Score when right, bet or no bet.
Simulate in `tools/economy-review/` first (current rule vs pool vs pool + seeded line; honest,
contrarian, favourite-only and abstaining bettors): the target is that betting your real belief is at
least as good as abstaining, and no fixed habit beats it. Then rule docs, `scoring.ts` `settleBets`,
the vote UI, `ECONOMICS.md`.

### Other simulators (asked 2026-11-13)
Ours (`web/src/sim/`, `tools/economy-review/`) is home-built and runs the real rules engine, which
no outside tool can do. Outside tools model the game rather than run it; the ones worth knowing:
- **Learning agents** (self-play reinforcement learning: OpenSpiel, or PettingZoo with RLlib): find
  exploits that fixed-habit bots never try. Highest value. `jingle_brawl_env.py` in this repo was an
  earlier attempt and is out of date with the rules; better to wrap the real engine through
  `sim/table-bridge.ts` (JSON lines) so the rules stay in one place.
- **Machinations** (a visual game-economy modeller): good for explaining chip faucets and sinks to
  people; it would only restate what `flows.ts` already measures.
- **Gambit** (a game-theory solver): exact equilibria for a small cut-down model, e.g. one sealed bid
  followed by a duel; a check on the review's `bne.ts`.

### Step 3. Test gaps and the bugs they hide
Unit suite: 390 pass. The review read the specs; it did not run Playwright. Ranked by risk:

1. **Endgame follow-up timer is neither saved nor cancelled** (`server/src/room/actor.ts:266-273`): a
   restart inside the 500 ms window strands the room in `endgame` with no turn; a closed actor can
   still commit. Likely a real bug (not reproduced). Fix: make it a `hostAuto` step; test in `actor.test.ts`.
2. **Three security gates with no test** (`server/src/node/server.test.ts`): `/debug` is 404 without
   `--debug-api`; an old token on a recycled room code is refused (`ws.ts:58`); a path-traversal request is refused.
3. **Gift routing in `resolveDuel`** (`core/duel.ts`), reached only by random simulation: auction won by
   someone already holding a gift, endgame steal lost, Misfit toll at 0 chips, reprisal chain stopping
   at `maxReprisalDepth` and the can't-afford skip. Unit tests in `engine.test.ts`.
4. **Admin actions** (`core/admin.ts`): `resetTurn` (leaves `arena`, no refund of a paid bid) and
   `removePlayerGift` untested; `undoDuel` restores `players` wholesale (a late joiner is deleted) and
   leaves `currentTurn`; `saveGift` with an unknown owner creates a phantom player. Decide the right
   behaviour for each, fix, pin in `admin.test.ts`.
5. **`RoomClient`** (`web/src/net/client.ts`): patch-gap resync, a retry timer surviving stop/start
   (possible second socket), pending presses on stop; `e2e/reconnect.spec.js` never asserts the phone caught up.
6. **Server failure paths**: `store.commit` throwing after memory changed, the critic throwing
   mid-judging and its call cap, `Rooms.recover()` (never called in a test), malformed socket messages,
   HTTP 400/404/oversize.
7. Smaller: a patched `arenaView` round-trip in `diff.test.ts`; Gremlin with someone away; equal-point
   overlapping targets; a source check that no `prefers-reduced-motion` rule returns (like
   `copy.test.ts`); delete dead `yield_choice` / `recordTiebreakWinner`.
8. **Flaky spots to harden**: `e2e/ringtoss.spec.js` (the new show test asserts things that live about
   1.3 s, one after another; the read-then-write of `arena` can overwrite a live throw; the five-swipe
   reload timing), `e2e/helpers.js:244,262` (sleep then skip races the server's phase timer),
   `e2e/regressions.spec.js:779-784` (real-time Whack). Give the debug API an atomic "patch arena
   state" call and wait on state, not sleeps.

### Step 4. Platform tidy (ten small commits, each independently testable)
Defects first: (a) error text is unstyled in Ring Toss and Conveyor (`.grem-error` only exists under
`.grem`); (b) simulated guests are told every mini-game is the Gremlin (`core/screens.ts:74-80,134-139`);
(c) check `ConveyorStage.tsx:42` `t.gifts.filter` on the finished state.

1. `ui/arena.css`: `.arena-full*`, `.arena-banner`, `.arena-error` out of `ringtoss.css` (+ fix a).
2. `useServerNow()` in `net/hooks.tsx`, `secsLeft()` (five copies today).
3. `host/GremlinStage.tsx` out of `ArenaStage.tsx`; one game-to-component map for host and player.
4. View types beside each game: `core/minigames/<game>/view.ts` (`RingView`, `GremlinView`, `ConveyorView`).
5. Shared per-game UI out of host/ and player/ into `ui/<game>/`; `boundary.test.ts` gains a host/player cross-import check.
6. Split `host/RingTossStage.tsx` (662 lines): `ui/ringtoss/scene.ts`, `effects.tsx`, `Flights.tsx`; delete dead `.rt-pad`, `compact`.
7. Optional `hydrate()` on `GameDefinition`, called once by `runtime.ts` (replaces a dozen `whole()` calls; settles c).
8. Result builders in `minigames/scoring.ts`; `projectForHost` optional; `asConfig()`.
9. Per-game `words()` for `core/screens.ts` (fixes b), tested in `sim/table.test.ts`.
10. `configFromSettings?` on the definition; core RNG helpers last (may shift seeded expectations).
Then: **move Sketch-Off and Whack-an-Elf onto the platform** as `GameDefinition`s (they are the older
separate `game.minigame` system). That gives server-counted Whack scores (S3), platform games as
tie-breakers, and one code path for every game. Largest item here; its own sub-plan when reached.

Left alone on purpose until a second game needs them: `flair.ts` helpers, placements, tiebreak
cascades, stage helpers, confetti/snow components.

### Step 5. Professional public app

**Blockers**
- **A front door.** `/` currently opens the phone join form. New `web/index.html` landing page: what it
  is, how it works in three steps, screenshots, "Host a game" and "Join with a code", links to rules and
  privacy. Join form stays at `/player.html` (and `/join`). Rules linked from the join screen and host console.
- **One name everywhere. FOLLOW-UP: Kevin is still deciding** (Yule Duel / Yuel Duel / Jingle Brawl).
  Until then: put the name in one constant (`ui/brand.ts`) and one logo path so the switch is a
  one-line change; the landing page, manifest, link previews and logo wait on the answer.
  `web/public/JingleBrawlV2.html` (a Google Docs export with blocked fonts) is re-exported or retired
  with its 2 MB of images.
- **Error boundary** around each entry (`host`, `player`, `tv`): a friendly "something broke, tap to
  reload" card that reports through the existing `net/errors.ts` beacon. No blank pages.
- **Terms of use + operator contact** (Kevin, 2026-11-13: private code, public site): a `terms.html`
  beside `privacy.html`, a contact address on both, an all-rights-reserved notice in the repo and
  footer; art and font licence notes (`tools/art/README.md`: check the Flux model's terms for a
  public commercial-looking site). No open-source licence.

**Should-have**
- Page heads: meta description, theme-color, Open Graph/Twitter card (join links get shared in group
  chats), a proper small favicon set (today a 633 KB PNG), apple-touch-icon, web manifest; then the
  P1 PWA (installable, no offline play).
- Screen Wake Lock on TV and phones during a game; allow pinch-zoom on `player.html`.
- Friendly states: human wording for "no such room" / "not your room" / room ended; a loading
  skeleton during the session check; replace the 14 `alert`/`confirm` calls with the app's own dialogs.
- Onboarding: host setup as presets ("Quick party", "Family", "Chaos") over the full form; a first-run
  tour of the console; "How to play" on the phone (60-second version generated from settings, like
  `ui/notes.tsx`); a demo room against simulated guests (`sim/` already plays the table).
- Design system: one `ui/tokens.css` for the Frost tokens (copied today in `host.css` and `player.css`);
  sweep the ~316 hard-coded colours and 255 inline styles onto tokens, worst files first; self-host the
  fonts (privacy + CSP + speed); rule pages restyled to match; fix contrast (gold room code on white
  1.6:1, accent links 3.4:1, muted text 4.2:1); labels on the join inputs; the 390 px lobby card crowding.
- Server: gzip/brotli; HTML 404 and 500 pages; build stamp shown in the console footer and sent with
  client errors; robots.txt; `rooms.db` backups (Litestream, P1); ESLint + Prettier in CI; CHANGELOG.
- Repo hygiene: fix or delete the stale `.claude/agents/{host,player,architect}.md` and
  `trace-mechanic` skill (they describe the retired 6600-line `host.html` and Firebase); move root
  clutter (`spec_package.md`, `GAME_STATE_AUDIT.md`, RL files, duplicate logos, `yd-design.css`,
  `YuleDule/`, `images/`) into `docs/archive/` or delete; track `yule_duel_agent_spec_updated.md`
  (CLAUDE.md cites it); one `ROADMAP.md` at the root replacing the scattered roadmap sections.

**Nice-to-have**: i18n layer, dark mode, a mute button on every game card (only the Gremlin has one).

### Step 6. Host it and play it for real
**Hosting: decide later** (Kevin, 2026-11-13); nothing before this step depends on it. Real-phone
playtests on the home Wi-Fi do not need hosting and can start any time after Step 2. When decided: deploy with the Dockerfile (build it locally once), check the
forwarded-address rate limiting, load test (rooms x players x drawings after S4), Litestream, uptime
check on `/healthz`. Real-phone playtests from `TESTING.md` on iPhone Safari + Android Chrome + a real
TV: Gremlin skew, Ring Toss flick and the new show, Conveyor pace, the new team duels.

### Step 7. More games (M6+)

**Next new game (Kevin, 2026-11-13): robot boxing** (working title "Rock 'Em Robots" is taken by a
real toy: pick an original name). The two duelists each drive a robot on the TV from their phone,
throwing punches and blocking; best of three rounds; a round is lost when your robot's head comes off.
3D robots in the game's look.
- **Rules as a `GameDefinition`** (`minigames/robots/`), mode `direct_duel` first. The room is the
  referee: phones send inputs (jab left, jab right, block, maybe dodge), the room resolves them on its
  own clock with wind-up, hit and recovery windows, so a block has to be timed and mashing loses to a
  counter. Each head has a "neck" meter; at zero it pops off and the round ends. Round timer; level on
  time goes to the healthier neck, then sudden death. Trust a phone's tap time only within a latency
  window, as `perfect_timing` does (`TIMING_LATENCY_MS`).
- **A theme-free engine**, `engines/fight.ts` (moves, windows, stamina, damage), beside the throw
  engines, with its own tests and a simulated hand in `offers()` so the virtual table can play it.
- **Phone = controller only** (as Ring Toss: eyes on the TV): two big punch zones and a hold-to-block,
  haptics on a hit taken, buttons large enough to use without looking.
- **Everyone else**: with the "everyone plays" setting, bystanders back a corner and tap to charge
  that robot's one special punch per round (Party points to the winning corner through `rewards`). A
  `captain_teams` mode can follow the same way.
- **3D look: decide with a prototype first.** Kevin called the Blender reindeer "amateurish" and chose
  flat 2D art, so the robots are judged in the Game Lab before the game is built around them. Try, in
  this order: (1) real-time three.js bundled with the app (no CDN: the CSP allows only our own
  scripts), simple chunky robots with a toon shader and dark outlines to sit beside the sticker art,
  head as a separate rigid body so it really flies off; (2) if that looks cheap, pre-rendered sprite
  sheets from better models. Check frame rate on a smart-TV browser and a Chromecast, not just a PC.
- Tests: engine and game unit tests, a room test for timing trust, `e2e/robots.spec.js`, the Lab.

After that, from the spec's catalogue, after Step 4 makes a new game cheap: pick two that reuse existing engines
(e.g. Snowball Siege on the throw engines: `testgames/snowballs.ts` already proves it; a captain-team
relay on `engines/workflow.ts`). One agent per game in its own worktree against the `GameDefinition` contract.

## How the work is run
Steps 1, 3 and 4 are many small independent commits: suited to parallel agents in separate worktrees
(one per fix, each with its test), merged and checked by the coordinator. Steps 2 and 5 touch shared
files and product taste: done in the main session with Kevin reviewing screens. Pause for review
after Step 1, after Step 2, and before the landing page design.

## Verification
- Every step: `npx tsc -p tsconfig.json --noEmit`, `npm run test:unit`, the step's own spec, then the
  full `npm run test:e2e` before each push; CI green.
- Step 1: each finding gets a failing-first test (unit for S1/S2/S3, server test for S4/S6/S7, `tests/` for S5).
- Step 2: `npm run agents:sweep` and `npm run analyze:economics` before and after; a live table run
  (`npx vite-node web/src/sim/play-table.ts -- --live --players 6`) showing team duels being picked.
- Step 5: Lighthouse on landing/player/host (performance, accessibility, best practices), a link-preview
  check of a join URL, and a walk through `npm run dev` as host + phone + TV.
- Step 6: the `TESTING.md` real-phone checklist.

## Open decisions (Kevin)
- **Product name** (thinking about it): blocks the landing page copy, logo, manifest and link previews.
- **Hosting**: later; blocks only Step 6's deploy.

## Progress log
- 2026-11-13: plan approved. **S1 and S2 fixed** (`core/tx.ts` `own()` + unsafe-path guard, `ID` regex,
  `admin.ts` own-key checks, `sanitizeSettings` + `LIMITS` in `core/settings.ts`, applied in
  `Rooms.create` and read through `config()`; tests `core/security.test.ts`, `core/settings.test.ts`,
  `server/src/room/settings.test.ts`). With `maxBid` capped at 99 the bid list is at most 100 offers,
  so bids are still checked against offers. Also fixed on the way: `saveGift` to an unknown owner made
  a phantom player (now refused; `'misfit'` means the pile). **Step 2 item 0 fixed** (an away duelist
  is always a participant; `duelgame.test.ts`).
- 2026-11-13: connection client (5 commits: one socket per screen, every press answered, a room that
  came back older), platform tidy (6 commits: Step 4 items 1 to 6 done), server (7 commits: the
  endgame draw is a host step, a malformed path can't stop the server, debug API gate S7, S8, failed
  saves, late presses, oversized bodies, floods) merged from agent worktrees. Step 1: S3 and S5 done
  too; S4 (drawings out of game state) and S6 (lock the room, seat revocation) still open.
- 2026-11-13: Step 2 items 0, 1, 2 built (`duelTableGames`: teams by default, all, off; the pickers
  say who plays). Open: item 3 (mid-game switch), item 4 (team modes for Ring Toss and the Gremlin:
  until then the only team game is the Conveyor).
- 2026-11-13: economy review done (Step 2b) and its three changes built; betting market proposed (Step 2c).
- 2026-11-13: betting simulated (learning bettor + fixed habits) and the passing design built in the
  engine (commit 141e33f; screens, rule docs and browser test still to do: see Step 2c). OpenSpiel
  2.0.2 installed in `.venv-openspiel/` (not in git); the bridge to the engine is not built
  (`tools/rl/README.md`). Next session: finish Step 2c's screens and docs, then Step 2 items 3 and 4.
- 2026-11-13: **Nutcracker Knockout started** (Step 7; Kevin chose the name and asked that parts come off hit by hit, the head last). Built: `engines/fight.ts`, `robots/` rules and tests, the phone controller, the TV stage with real-time 3D (three.js 0.170, toon shading and ink outlines), `e2e/robots.spec.js`, rule docs. Open: Kevin's verdict on the look and the feel of the timings on real phones; hands are cosmetic (left and right punches are the same); no mode for the rest of the table yet (bystanders backing a corner); sounds are the stock cues; frame rate on a real TV untested.
- 2026-11-13 (evening): Kevin tried to reach the Game Lab to play Nutcracker Knockout and could not find it: his room had already been started (the lab is lobby-only), and he took the greyed-out Endgame button for the way out (the way out is Quick actions > Exit Game, below the fold). He has not yet seen the robots. To fix (Step 5 onboarding): make the lab findable from a started game (or say on the Controls card why it is gone and how to get a fresh room), and make Exit Game easier to find.
- 2026-11-16: Kevin played Nutcracker Knockout and likes it ("cool", "good to go"). His asks, built: rounds were over too soon, so a part now takes two blows (12 health, 60 s rounds); a dodge (it replaced the timed parry: one timed defence is enough for a phone you are not looking at); left/right became a quick jab and a slow haymaker that smashes a block. Also a slow-motion knockout with the camera following the head, and tin sounds. **Not wanted yet (his call): anything for the other players during a fight.** Still open: feel on real phones, avatars on the robots, a comeback lean (needs simulating). He also wants the **Gift Conveyor rebuilt as a real belt**: unfinished toys roll along a conveyor on the TV, each player adds their piece (wheels, headlights) as toys pass their station; my additions he has not yet answered: the captain runs the belt-speed lever, keep the stowaway, small teams double up stations, the phone shows a close-up of your stretch; replace the current game, do not add a second.
- 2026-11-16: **the Gift Conveyor rebuilt as a real belt** (Kevin's design; it replaces the phone-to-phone version). `engines/belt.ts` + tests, `conveyor/` rewritten, `ui/conveyorBelt.tsx`, new phone card and TV stage, `e2e/conveyor.spec.js`. Kevin's call on speed: no captain's lever; each belt speeds up with its own team's deliveries, and (my addition, to confirm with him) both creep up as the round goes on. Open: his verdict on look and feel; pacing numbers (quota 8, 90 s, base 10 s a traverse, top speed leaves about 0.85 s at a station) are untested on real phones; `engines/workflow.ts` is now unused (delete or keep for a future game); the TV belt has a lot of empty height.
- 2026-11-16: betting follow-ups done: the phone only says a call stakes a chip when bets really are open on that duel; browser tests for the pool (six players) and for a small table (free vote). Still open in Step 2c: live odds on the phone and TV, the rule docs' betting text, undo restoring `betPool`.
- 2026-11-16: paused mini-games really stand still (`shiftTime`, `arenaView.pausedAt`). **Whack-an-Elf polished** after a review Kevin asked for: the TV's live bars never filled (fixed); full screen on the TV and console with the board and a live race (`host/WhackStage.tsx`); the phone's board fills the screen with sound, vibration, score, place and a 3-2-1; a roll (three in a row: +1 a hit, `whackTap`); new flat 2D pop-ups from Flux (`tools/art/whack2d.py`, `whack2d_place.py`) replacing the Blender ones. **The Grinch is the Grump everywhere** (Kevin's choice; 115 places, incl. "Grump's Gambit"). Open: the holes and mallet are still the Blender renders; the `gambit` icon was drawn for the old name; `JingleBrawlV2.html` (the old export) now says Grump too but is otherwise stale; Sketch-Off has had no such pass.
- 2026-11-18: Kevin tested and came back with three things. (1) He could not find Whack-an-Elf to try: **the Game Lab now lists Whack-an-Elf and Sketch-Off** (practice: no prize, results stay up). (2) "Is there a way to test the games with AI NPC players?": **practice bots** (`core/bots.ts`): the host adds them in the lab's picker, the room server plays them through each game's own `offers()`; any game runs with one person or none. Lab only: they are not in the real game (a whole-game demo room against simulated guests is still Step 5). (3) "The conveyor host screen is pretty rudimentary. This is supposed to be a polished, professional looking product": **the Conveyor's TV screen redrawn as a factory floor** (`ui/conveyorFactory.tsx`): a machine per station with its sign, its workers' avatars, a waiting lamp and a tool that stamps; a belt with a hazard rail and rollers; hopper and checker; a sleigh that fills with gifts; sparks, crosses, "Faster!", a winners flag. Open: his verdict on it; the phone card has not had the same pass; bots are a little too tidy to be a real table (pace and slip are constants in `bots.ts`); the same eye should go over the Gremlin and Sketch-Off screens (his bar: "polished, professional").
- 2026-11-18 (later, Kevin testing with bots): (1) Ring Toss: the grey smudge below the reindeer was the flight shadow of a ring on its way to an antler falling on the snowdrift: a ring now has a shadow only near the ground (`scene.ts shadowOpacity`); "a ring shows up below the antler" was `stack` counting every earlier ringer of the game: only rings still showing count (`stackOn`); the trail is two faint copies, not four ("too busy"). (2) **Pass the Gremlin's TV screen is a table**: everyone seated round it, the Gremlin hopping to the holder, the last passes drawn, hold time, wrong answers, how often each has held it. (3) **Sketch-Off fills the screens**: the phone's pad is full screen, the TV is a wall of easels where the drawings grow live with no names (`sketch_live`, `minigame.wall`), a big numbered gallery for the vote, then whose was whose. (4) **Sounds** rebuilt as synthesized instruments with reverb (he found them "too 8 bit"); not heard by me: his ears decide, and recorded samples are the next step if it is still not right. Asked and answered: an everyone-plays game has no score of its own for bystanders (Gremlin: one loser, a doomed bystander starts a duelists-only sudden death; Ring Toss: the better duelist wins; only the Conveyor pays Party points to the winning team); whether bystanders should earn something is open.

**2026-11-18 (third entry): rewards for the table playing along (Kevin's design).** Pass the Gremlin with
the whole table: a caught bystander is out and the rest play on with a shorter fuse (35 to 55 s); after
two are out (or the only one) the duelists finish in sudden death. Bystanders earn Party points for
lasting: 0 for the first out, 1 for the second, 2 if never caught. Ring Toss: everyone but the duelists
earns 3 / 2 / 1 by their own score. `handoff()` logs who was paid. `npm run analyze:gremlin` (6 players):
no bystander caught in 68% of games, one in 22%, two in 10%; median game 89 s, longest 189 s; a
bystander averages 1.6 to 1.9 points a game; every bystander passing to one duelist moves that duelist's
losses only from 52% to 54%. At first that was 0 / 1 / 2, and a bystander nearly always earned 2, twice the 1 point a duelist gets
for fighting; Kevin had it fixed the same day: 1 Party point for never being caught, nothing otherwise. The end-of-game result is on screen only for a moment before the duel verdict
takes over (true of every arena game): a short results hold would fix that.

**2026-11-18 (fourth entry): two new team games, built back to back at Kevin's request (he had not
played either when this was written).** From the spec he copied into `agents/`
(`yule_duel_priority_games_4_to_9_spec.md`: Mecha Santa is its #6, Elf Tower Wars its #9; Pie Face,
Sleigh Kart, QR Lineup and Claim the Chimneys are not built).
- **Platform first:** a finished mini-game's ending stays up 6 s before the duel takes over (host can
  skip); jobs on a team are a shared engine (`engines/roles.ts`); Setup says what a game needs (the TV,
  four players).
- **Mecha Santa Mayhem** (`mecha_santa`): 3D toon robots like Knockout (his choice). Simulation: an
  average team of four finishes in about 55 s, a slow one is still out at the 90 s bell; teams of 2, 3,
  4 and 5 win about evenly after two fixes (shared jobs take turns; small teams stride further).
- **Elf Tower Wars** (`elf_tower_wars`): the armies march past each other (his choice). Each job has
  its own energy (not one shared pool: so nobody can spend a teammate's share). Simulation: a team that
  never builds, never attacks or never defends loses every time, and team sizes come out even. **Its
  balance is not finished:** with towers upgradable (a cannon over a snowball tower) defence is strong,
  so between even teams no workshop falls and the bell decides (about 5 of 20 health lost each); and a
  team that saves its energy for tanks and cannons beats one that spends as fast as it can (94% in the
  simulation), which may be good strategy or may be a dominant habit: to settle after he plays. Its
  pictures are drawn SVG, not Flux sticker art (the plan said Flux): to redo if he wants it richer.
- **Not done:** real-phone feel for both; tilt control (needs HTTPS); bots-only browser cases for the two
  (the room test covers bots finishing both); the spec's contribution bonus (flat reward kept).

**2026-11-22: graphics pass (Kevin: "improve the graphics, especially Tower Wars", "the mecha santas
too", and the goal: consistent across games and fit to release).** Tower Wars redrawn with Flux art
(toys, towers, workshops, scenery, snowy field; `tools/art/towers.py`), lobbed snowballs, puffs, bangs,
smoke from a hurt workshop. Mecha Santa rebuilt as a riveted clockwork robot (piston legs, shoulder
plates, dials, chimney, lamp eyes); the camera no longer shrinks both robots to specks when one is far
behind. All eight games photographed side by side: the family is sticker avatars + flat Flux cartoon +
toon 3D on one palette. Still uneven: Knockout's nutcrackers are blockier than the new Mecha Santas;
Whack-an-Elf's TV leaves a lot of empty space; awaiting his verdict.

**2026-11-22 (second entry): instructions and a Ready button before every mini-game (Kevin: "some
games have instructions, but others don't. It might also be good for everyone to hit a button when
they're ready as opposed to a countdown", then "have a countdown after everyone has confirmed").**
Every runtime game now opens with how to play it (TV, console, phones), waits for each player's Ready,
then counts 3, 2, 1. Whack-an-Elf and Sketch-Off wait the same way. Bots are always ready; a dropped
phone is not waited for; the host can start without waiting; a setting switches it off. Also that day:
Knockout's nutcrackers rounded to match the Mecha Santas, with new faces (he asked for better
moustaches and thinner eyebrows). Open: Sketch-Off shows its prompt while people are still pressing
Ready (thinking time is uneven); the games' own short intros (Ring Toss practice, Knockout's "Round 1")
still run after the count.

**2026-11-22 (third entry): Kevin played Tower Wars and Mecha Santa.** Tower Wars ("I was the builder
and didn't really know what to do. It wasn't that fun just waiting... perhaps instead of everyone doing
a different role, we have the same role(s)"): the three jobs are gone. Every player has the same hand
of cards and their own energy; the phone is a map plus the cards (pick a card, tap where). Free
spending made an all-out rush win every time in simulation, so each workshop now throws its own
snowballs at toys that get close and toys toughen faster; even teams of all-rounders are close and
last about 100 s. Open: the simulated hands that press less often (so save for dearer things) beat the
quick ones, and uneven team sizes can only be checked roughly with them; needs real players. Mecha
Santa ("boring to play as you tell me when I need to press a button... each player can activate their
part whenever, so the fun is in having to communicate"): the phone is now plain buttons that never
light up or say when; every press does something (wrong timing trips, bonks or lurches the robot);
shared jobs no longer take turns. Also that day: the Conveyor's toy cars redrawn without dashed
outlines, and the Tower Wars workshop recut (its roof had holes where the snow was cut out with the
background).

**2026-11-22 (fourth entry): polish plan from a look at similar games (Kevin: "research against other
types of similar games... come up with a plan... see what we can do to polish things further
(animations etc)"). Proposed, not approved.** Sources were thin on Gartic Phone, Kahoot and AirConsole
specifics; those items are from general knowledge of them.
- Phase 1, one look for every game: a shared motion kit (`ui/motion.css`: enter, exit, pop, count-up,
  shake); one opening sequence (title card, how to play, 3, 2, 1 with sound) and one results ceremony
  (winners in, confetti, points counting up) used by all eight games; keep phones awake during a game
  (Screen Wake Lock: nothing does this today).
- Phase 2, per-game animation: Tower Wars (toys bob as they walk, towers recoil, a workshop shakes and
  crumbles, a card flies to where it was played); Conveyor (a fitted piece snaps on, the sleigh
  bounces); Mecha Santa (dust at each footfall, a camera punch on a smash); Gremlin, Whack-an-Elf and
  Sketch-Off entrances; then the main game's own moments (bid reveal, gift opening, duel verdict,
  chips moving).
- Phase 3, what similar games ship that this does not: an end-of-night recap (podium, a few awards, a
  picture to share, the Sketch-Off drawings to keep); "play again with the same people"; a relaxed
  pace setting (longer or no timers); an error boundary.
- Phase 4, bigger: music under the lobby and the games; an unscored practice round the first time a
  game comes up in a night; a bot holding a dropped player's seat in a team game.
- Deliberately not copying: accounts or sign-up (the commonest complaint about gift-exchange apps),
  motion or camera controls (they shut out older phones).
Also that day: the workshop and snowball tower art recut again (tolerance 4): the right eave's snow and
the tower's loaded snowball were still see-through.

**2026-11-22 (fifth entry): Tower Wars after a look at the genre (Kevin: "take this research... and make
the changes").** Built: a third toy (a pack of three wind-up mice, Flux art) and real counters (the
cannon hits a crowd, the snowball tower one toy; mice swamp a snow tower, a cannon stops them, a teddy
tank shrugs off a cannon); towers level up by building the same one again (stars, bigger, reach shown
when built); an avalanche that needs two teammates on the same path within a moment (shown waiting on
the TV only); double energy for the last half minute; each card's hint says what it is for and what
answers it. Not built: sending raising income (only if real players turtle). Open after simulation:
hands that save for dear things still beat quick cheap spending; needs real players.

**2026-11-23: the polish plan built (Kevin: "commit and push this and then implement the wider polish
plan"; the fourth entry above was the plan).** Committed and pushed first (`76b4949`); everything below
is uncommitted.
- Phase 1: the motion kit (`ui/motion.css`, `ui/motion.tsx`); the opening (title lands, lines come up,
  "GO!"; a long briefing now fits the TV: Mecha Santa's and Tower Wars's used to run off the bottom);
  one results ceremony for all six runtime games on the TV, console and phones; phones kept awake in a
  mini-game and the TV always (only works on https or localhost: on a plain Wi-Fi address browsers do
  not offer it, so until hosting a phone can still dim).
- Phase 2: Tower Wars (toys walk, towers recoil, a workshop is knocked and crumbles, a lost toy tips
  over), Conveyor (a piece snaps on, the sleigh bounces, a binned toy tumbles, the raccoon is shooed),
  Mecha Santa (snow at each footfall, a camera punch, a rocket trail); the main game's turn panel comes
  in with the same moves. Not done: new entrances for Gremlin, Whack-an-Elf and Sketch-Off (they had
  their own already), and nothing new for the bid reveal, verdict or chips (already animated).
- Phase 3: the finale (MVP, podium, awards from what the night recorded, who got what, a picture to
  save or share); Play again with the same people; a relaxed pace; an error boundary on every page.
  Sketch-Off drawings to keep: not possible yet, a party game's drawings are thrown away when it ends
  (keeping them is tied to open security item S4).
- Phase 4: music (four original loops, TV on by default, console off); a practice round the first time
  a game comes up in a night; the room plays a dropped teammate's part in a team game.
- Nobody has heard the music or the new sound timings, and the animations were judged from stills: both
  need Kevin's ears and eyes. Simulated tables and the persona agents now get the practice round too
  (it is the default); `npm run analyze:economics` is unaffected (no chips or points move in practice).

**2026-11-23 (second entry): Kevin played again and asked for more.**
- Mecha Santa: a robot left behind walked off the TV (now: the picture splits in two, his choice); a new
  course every race; chimney stacks to climb, rope swings, ice, with the TV and the phones saying what
  is coming and what it takes (his choice: what, never when). He then found, in minutes, a robot
  sinking through the stack it vaulted, a rope let-go snapping back to the edge, a climb that looked
  like walking, and the old winner card under the new celebration card ("this is the sort of polish I
  was expecting you to pick up on"): all fixed, the old cards removed, and motion is now judged from
  burst captures (`tools/shots/burst.mjs`, `sheet.py`), not stills.
- Tower Wars: towers no longer look as if they hover on the soldiers; levels 2 and 3 have their own
  Flux pictures. Then ("the pacing is a bit too fast, like I'm always looking at my phone screen...
  research some of the well rated tower defense games") it became rounds of plan, then watch (his
  choice, after Clash Royale's card pace and Legion TD's build and battle phases), with six cards and
  three reactions (his choice); the glue card went. Open after simulation: an all-toys rush beats the
  simulated all-rounder about 7 in 10; uneven team sizes swing either way; between even teams few
  battles end with a workshop falling (most are decided on health after five rounds).
- Knockout: pads laid out by the robot's side of the ring (jab over haymaker facing the other robot,
  block over dodge behind), and the robots change sides every round (a decider is tossed for).
- A full polish sweep by three reviewers (every game and the main flow, in motion) was asked for; its
  findings are recorded below when in.

