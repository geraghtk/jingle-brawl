# Architecture

Where Jingle Brawl (Yuel Duel) is today, where it is going, and why. Decided 2026-11-07; the
mini-game spec is `yule_duel_agent_spec_updated.md`.

## Before (until 2026-11-07)

```
host.html (TV + host console, one page)        player.html (phones)
        └──────── Firebase Realtime Database ──────────┘
                  games/{room}: whole game, world-readable,
                  mostly writable by any signed-in user
```

What stopped this being a product: no secrets (any phone could read sealed bids, so a hidden
timer or a challenge answer couldn't exist); any signed-in user could rewrite the game; the host
tab was the referee and the only clock (close it and the game stalled); the TV and the host
console were one page; open, paid AI-critic endpoints; no CI, staging, room cleanup or
observability. The rules engine was already pure, which made the move cheap:

- **Rules engine** `web/src/core`: actions are `(game, …) => Plan` built with `Tx` (`tx.ts`).
- **Who can press what**: `core/table.ts` `offers()` / `perform()`; `core/intents.ts` adds the
  host's admin intents, `parseIntent` (shape and bounds) and `dispatch`.
- **Automatic steps**: `core/hostauto.ts` derives every timer from state.
- **Duel hand-off**: duel games set `currentTurn/duelGameWinner`; a hostauto step calls
  `resolveDuel` (`core/duel.ts`), which moves chips and gifts once (claim `resolvedBy`).

## Now (S1-S7 done 2026-11-07)

```
LOCAL (now)                          LATER (when chosen)
PC: node room server                 same server on Fly.io / Railway / Cloud Run,
  ├ serves the built app             or a Durable Objects adapter
  ├ rooms: one action at a time,     (room logic unchanged)
  │  SQLite state + event log
  └ server timers (hostAuto)
    ▲            ▲            ▲
  host console   TV view      phones (Wi-Fi), each socket gets ITS view
```

- **Own small Node room server** (`server/`), WebSockets. Phones, the TV and the host console send
  *intents*; the room validates (who, phase, schema, action id), applies the existing core
  engine, persists an event, and publishes a projection per viewer. Secrets never leave it.
- **Durable timers**: steps are derived from persisted state (`hostAuto`), so a restart re-arms
  them. Due steps run before each intent, so a deadline beats a late press deterministically.
- **Identity**: server-issued signed host/player tokens (Firebase auth retired).
- **Portability**: the room actor depends only on Clock / Storage / Publisher / RNG interfaces, so
  a Cloudflare Durable Objects adapter can be added without touching rules.

### Why this and not …
| Option | Verdict |
|---|---|
| Firebase + Cloud Run listener | Works, but we'd hand-build an intent queue, room leases, private nodes and deadline persistence; always-on instance ~$45/mo; two DB hops per action. |
| Cloudflare Durable Objects | Best-fit model (room actor, alarms, hibernation, ~$5/mo) but production is Cloudflare-only; self-hosted workerd is single-instance. Kept as a future adapter. |
| Colyseus / Nakama / Rivet | Colyseus duplicates our state model and keeps rooms in memory; Nakama is a heavy Go+DB stack whose TS runtime is not Node; Rivet now targets AI-agent workloads. |

## Product decisions
- Sealed bids are hidden from everyone, the host included, until the reveal; the host has an
  audited "peek".
- With `publicChips` off, others' chips are hidden until the game ends.
- AI Draw Off comes back as an in-app duel game (the two duelists draw, the AI critic judges),
  run by the server. Keys are BYO per room, held only in server memory (never in the game, the
  database, logs or any view), rate-limited, forgotten at the game's end, after inactivity, on
  request or on restart; a server env key can be a default. If the critic fails, the host picks
  the winner (marked as a host decision); nobody wins by default.
- Hosting: Cloudflare's free tier (a Durable Objects adapter for the same room actor) is the
  proposed hosted target, after a load test of its free limits; the Node server stays the
  local/LAN version.
- Pass the Gremlin ships as an in-app duel game: 2 players direct duel; 3+ everyone plays and the
  two duelists are resolved (duelist-only sudden death if a bystander is doomed).
- "Yuel Duel" is the brand (the logo says so), not a typo.

## Collision risks
1. UI code builds plans itself at ~60 call sites; moving to intents can drift from `perform()`.
   Parity tests per call site; the e2e suite stays green before any transport change.
2. `isOffered` ignores hand-filled fields (name, giftName, score, final, drawing); intents over
   the network need strict per-type validation and bounds.
3. Host-tab-only behaviour must move server-side: `then` follow-ups, the event log, Undo, the
   Draw Off critic pipeline.
4. Presence (`connected`) decides who may vote; phones sleeping on Wi-Fi flap, so the server
   needs a grace period.
5. Spec conflicts: the spec says reuse the existing transport/auth (we replace them on purpose);
   `party.rank()` breaks duel-goal ties with a seeded coin (spec forbids arbitrary winners);
   Whack-an-Elf scores are client-reported (spec: never trust client scores).
6. `game.minigame` is already a union (PartyGame | DrawOff); the new runtime gets its own field.

## Roadmap
Done 2026-11-07: P0.1 retire the original pages → S1 intents + single commit path → S2
per-viewer projections → S3 room server → S4 client transport → S5 suite on the server → S6
server by default, `npm run dev`, Wi-Fi join link → S7 Firebase removed → AI Draw Off as a duel
game (BYO keys) → S8 the TV as its own screen, the host playing from their own phone → the
host's audited Peek → M1 the mini-game platform → M2 Pass the Gremlin.

Done 2026-11-08, the hosting-independent production track (`DEPLOY.md`): per-address abuse limits
(`server/src/node/limits.ts`), live-room and 50-player caps, security headers + CSP; rooms deleted
from the store 2 h after the end / 24 h idle (`Rooms.purge`), tokens bound to one room (`born`) so a
reused code doesn't open for old tokens; JSON logs, `/metrics`, browsers' error reports; the
reconnect banner and dead-socket detection; the host's "A phone died?" seat link; `privacy.html`;
Dockerfile; CI (`.github/workflows/ci.yml`).

Done 2026-11-08, M3: the Gremlin's memory_flash and perfect_timing challenges (server-checked; the
host picks one kind or the mix), its sticker art, sounds and a mute switch, accessibility labels.

Done 2026-11-08, M4: Reindeer Ring Toss, game #2 (the TV as the arena), on theme-free throw engines
(`minigames/engines/`: swipe → launch, deterministic flight, peg scoring) that a second game reuses.

Done 2026-11-08, M5: the Great Gift Conveyor, game #3: captain-led teams (`engines/teams.ts`), a
phone-to-phone production line (`engines/workflow.ts`), the winning captain wins the duel, reward
recommendations applied as Party points by the hand-off.

Done 2026-11-08: the Game Lab: from the lobby the host tries any registered mini-game on the real
phones and TV (chosen players, mode, duelists), practice only, no gift engine involved
(`minigames/lab.ts`).

Next: choose the hosting (Cloudflare free tier via a Durable Objects adapter, or one small VM/Fly
machine with the Dockerfile); real-phone playtests in the Game Lab to tune the games (the doom's
skew towards slow players, Ring Toss flick and peg sizes, the Conveyor's pace).
