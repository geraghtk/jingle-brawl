# Architecture

Where Jingle Brawl (Yuel Duel) is today, where it is going, and why. Decided 2026-11-07; the
mini-game spec is `yule_duel_agent_spec_updated.md`.

## Today

```
host.html (TV + host console, one page)        player.html (phones)
        │  React + TS (web/src)                        │
        └──────── Firebase Realtime Database ──────────┘
                  games/{room}: whole game, world-readable,
                  mostly writable by any signed-in user
```

- **Rules engine** `web/src/core`: pure. Actions are `(game, …) => Plan` built with `Tx`
  (`tx.ts`); a Plan is a multi-path update plus an optional claim (set-if-empty guard).
- **Who can press what**: `core/table.ts` `offers()` / `perform()` (the phone's buttons, the
  host console incl. Manual Override).
- **Automatic host steps**: `core/hostauto.ts` derives every timer from state (party-game
  phases, duel vote, wheel, in-app duel winner, ties). The host tab arms them
  (`host/useHostAutomation.ts`); the simulator runs them on a virtual clock (`sim/table.ts`).
- **Transport**: `web/src/firebase/game.ts` (`useGame`, `applyPlan`, `useServerOffset`).
- **Duel hand-off**: in-app duel games set `currentTurn/duelGameWinner`; the hostauto step calls
  `resolveDuel` (`core/duel.ts`), which moves chips and gifts once (claim `resolvedBy`).

### What stops this being a product
1. No secrets: every phone (and anyone with the room code) can read sealed bids; a hidden
   timer or a challenge answer cannot exist.
2. Client-authoritative writes: any signed-in user can rewrite the turn, gifts or mini-game.
3. The host tab is the referee and the only clock: close it and the game stalls.
4. The TV and the host console are one page.
5. Open, paid AI-critic endpoints (no auth, CORS `*`, no rate limit). No CI, no staging, no
   room cleanup, no observability.

## Target

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
- AI Draw Off stays, ported behind the server (keys on the server, host-only, rate-limited).
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
Plan: P0.1 retire the original pages (done 2026-11-07) → S1 intents + single commit path → S2
per-viewer projections → S3 room server → S4 client transport → S5 tests on the server → S6
flip default + LAN join QR → S7 remove Firebase → S8 separate TV view → M1 mini-game runtime
contracts → M2 Pass the Gremlin. Production track alongside: deploy (Docker + Fly.io), CI, abuse
limits, room TTL, observability, privacy page, reconnect UX. Ring Toss (M4) and the Gift Conveyor
(M5) come after, with their own approval.
