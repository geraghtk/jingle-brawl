# Agent play-testing harness

Persona agents play **real** games: the real `host.html` and one real `player.html` per
agent, in a browser, against the local emulators. Each agent reads only what its player
could see, decides in character, and clicks the same buttons a person would. A referee
drives the host (Draw, Reveal, Start, and declaring duel winners by fair coin flip).

Use it to:
- **Watch** a table of personalities play (`--watch`).
- **Check playability:** it catches stalls, page and console errors, alerts, buttons that never
  appear, and broken invariants. Each issue comes with screenshots.
- **Compare settings and balance:** `--sweep` runs every preset over the same seeded tables
  and writes a report.

```bash
npm run agents:watch                 # one paced game, streamed live to a single window
npm run agents:sweep                 # all presets × 8 games, headless → agents/reports/<time>/summary.md
npm run agents -- --games 5 --set max-payout=3,loser-dividend=everyone
npm run agents -- --lineup brawler,miser,shark,sibling --watch --seed 7
npm run agents -- --personas         # list the cast
npm run test:agents                  # persona unit tests
```

It starts the page server and the emulators if they aren't running, and stops them
afterwards. Options are documented at the top of `play.js`.

## Playing with the agents (join mode)

You host and play a phone; agents fill the other seats.

1. `npm run emulator`, and the app: `npm run dev` (React on :5173) or the legacy pages on :8080.
2. Open `http://localhost:5173/host.html`, create a game, note the room code.
3. Your phone: a tab on the **same computer** at `http://localhost:5173/player.html?room=CODE&testId=1`
   (the local emulators are only reachable from this machine).
4. `npm run agents:join ABCD` with your room code (`agents:join:legacy` for the :8080 pages).
   A window streams their phones and thoughts. More options (`--players N`, default 4;
   `--lineup a,b,c`) need `node agents/play.js --watch --phones http://localhost:5173 --join ABCD …`,
   because npm swallows `--flags` in PowerShell.
5. Press **Start Game** on the host and play. The agents never touch the host page or your
   moves; they bid, vote, fight/yield, Reprise, steal and play party games for themselves.
   Gifts you type get an average value (plus each agent's own taste).

## The cast (`personas.js`)

Real-world gift-exchange types, built from shared traits: bid threshold and scale,
reserve, endgame saving, all-in, herd/FOMO, attachment to what they hold, yield
threshold, Gambit/Reprisal appetite, cost aversion, conflict aversion, spite, nemesis,
and leader-hunting.

| key | who | plays like |
|---|---|---|
| `miser` | 🧾 The Accountant | small bids, clear upgrades only, hoards |
| `brawler` | 🥊 The Alpha | bids big and often, never yields, always attacks |
| `sniper` | 🎯 The Collector | ignores average gifts, all-in on great ones |
| `wildcard` | 🎲 The Party Animal | random everything |
| `grinch` | 😈 The Grudge Holder | goes after whoever beat them |
| `strategist` | ♟️ The Chess Player | modest early, saves for the endgame |
| `peacekeeper` | 🕊️ The People-Pleaser | avoids conflict, yields readily |
| `sibling` | 🤼 The Competitive Sibling | fixated on one nemesis |
| `follower` | 🐑 The Bandwagoner | bids what the room has been bidding |
| `highroller` | 🎰 The High Roller | all-in or nothing, loves a Gambit |
| `sentimental` | 🥹 The Sentimentalist | overvalues what they hold, always defends |
| `shark` | 🦈 The Endgame Shark | silent all game, steals in the Misfit Lottery |
| `contrarian` | 👑 The Kingmaker | targets whoever's winning |
| `newbie` | 🐣 The First-Timer | tiny bids, misses options |

With no `--lineup`, each game seats a random mix (`--players N`, default 6). The mix
is seeded, so every preset in a sweep sees the same tables.

**Gift value model:** each gift has a hidden quality (1–10), plus a personal taste offset of
about ±1.4 per player. The report's *gift value* is how much each player values what they
took home.

## Files

- `play.js`: CLI, server start-up, watch-mode stage (screencasts of every page plus commentary)
- `arena.js`: runs one game. Contains the state → whose-move → click loop, stall detection, issue capture and results
- `personas.js`: the cast and their decisions (pure functions over a "what I can see" view)
- `presets.js`: settings presets for `--sweep`
- `report.js`: results → `summary.md`

## Limits

- Party tiebreakers are agent-driven: in Whack-an-Elf each agent taps on its own phone with its
  persona's reaction time (`reflexMs`) and Grinch slip rate; in Sketch-Off agents scribble and
  vote at random (drawing quality isn't modelled). Presets `party-whack` / `party-sketch`.
- AI Draw Off isn't agent-driven (it needs the AI critic).
- Duels are coin flips, so skill is not modelled.
- Agents don't see other players' chip counts, which matches the default `publicChips: false`.
