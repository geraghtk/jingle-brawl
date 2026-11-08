# Yuel Duel (Jingle Brawl)

A real-time party game for gift exchanges: players bid Jingle Chips on opened gifts, duel
for them, and strike back with Grinch's Gambits and Reindeer Reprisals. One screen runs
`host.html` (TV/laptop); everyone else plays on their phone via `player.html` (React + TypeScript app in `web/`).

Rules: [`rules.html`](web/public/rules.html) · [`quick-reference.html`](web/public/quick-reference.html) · full text in `Jingle Brawl V2.md`.

## Quick start (local)

```bash
npm install
npm run dev                   # the room server + the app on :5173
```

Open http://localhost:5173/host.html → **Create New Game**, then join from
`player.html?room=CODE` (or scan the QR code: it carries this PC's Wi-Fi address, so phones on the
same Wi-Fi can join). Rooms live in rooms.db. To simulate many players in one browser use
`player.html?room=CODE&testId=1&name=Alice`.

## Tests

```bash
npm run test:e2e                         # Playwright: host + phones end-to-end on the room server
python -m unittest discover -s tests     # local AI critic tests
```

See [`TESTING.md`](TESTING.md). Every fix or new code path ships with a test.

**Watch AI agents play:** `npm run agents:watch` streams a game played by persona agents
(conservative, aggressive, spiteful, endgame sharks …) into one window; `npm run agents:sweep`
plays every settings preset and writes a balance report. See [`agents/README.md`](agents/README.md).

## Runbook

### Game night (production)
1. On a PC: `npm run build && npm start`. On the host's phone, open the address it prints, tap
   **host a new game**, choose settings, **Create New Game**, then **Join as a player too**.
2. On the device showing the TV, open the TV link from the host console (`tv.html?room=CODE`):
   the public view, no controls. Players on the same Wi-Fi scan the join QR and enter a name. Start needs ≥ 2 players.
3. If a phone dies, the host has **Manual Override** under every waiting step, and the
   **Admin** tab (edit chips, move gifts, skip/reset a turn, undo the last duel).
4. A refreshed host or phone offers **Rejoin** (rooms live in `rooms.db`; a restarted server picks them up).
5. Tiebreakers: Sketch-Off and Whack-an-Elf need nothing extra (everyone plays on their phone; the host has
   **Skip ahead** if someone stalls).

### Deploying
One Node process serves the app and the rooms: `npm run build && npm start` (`--port`, `--db`, `JB_TOKEN_SECRET`).
Hosting it online (HTTPS, a disk for `rooms.db`) is planned in `ARCHITECTURE.md`.

### Troubleshooting
| Symptom | Fix |
|---|---|
| Phones can't open the join link | same Wi-Fi as the PC? Windows firewall allowing Node on private networks? |
| "please reload: the game was updated" | the page is older than the server: reload it |
| AI critic "Connection refused" locally | start `python ai_art_critic.py` (and `ollama serve`) |

## Project docs
- [`CLAUDE.md`](CLAUDE.md) — architecture & conventions (start here when changing code)
- [`CODE_REVIEW.md`](CODE_REVIEW.md) — review findings, fixed bugs, open roadmap, rules-vs-code decisions
- [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`AI_ART_CRITIC.md`](AI_ART_CRITIC.md) · [`GAME_FLOW_AUDIT.md`](GAME_FLOW_AUDIT.md)
