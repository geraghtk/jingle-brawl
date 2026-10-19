# Yuel Duel (Jingle Brawl)

A real-time party game for gift exchanges: players bid Jingle Chips on opened gifts, duel
for them, and strike back with Grinch's Gambits and Reindeer Reprisals. One screen runs
`host.html` (TV/laptop); everyone else plays on their phone via `player.html`.

Rules: [`rules.html`](rules.html) · [`quick-reference.html`](quick-reference.html) · full text in `Jingle Brawl V2.md`.

## Quick start (local)

```bash
npm install
npm run emulator              # Realtime Database + Auth emulators (Java 21+, Firebase CLI)
python -m http.server 8080    # in a second terminal
```

Open http://localhost:8080/host.html → **Create New Game**, then join from
`player.html?room=CODE` (or scan the QR code). On localhost everything uses the
emulators; nothing touches production. To simulate many players in one browser use
`player.html?room=CODE&testId=1&name=Alice` or `test-multi-device.html`.

## Tests

```bash
npm run test:e2e                         # Playwright: host + phones end-to-end on the emulators
cd functions && npm test                 # Cloud Function parser tests
python -m unittest discover -s tests     # local AI critic tests
```

See [`TESTING.md`](TESTING.md). Every fix or new code path ships with a test.

**Watch AI agents play:** `npm run agents:watch` streams a game played by persona agents
(conservative, aggressive, spiteful, endgame sharks …) into one window; `npm run agents:sweep`
plays every settings preset and writes a balance report. See [`agents/README.md`](agents/README.md).

## Runbook

### Game night (production)
1. Open `https://<project>.web.app/host.html` on the big screen, choose settings, **Create New Game**.
2. Players scan the QR code / open the join URL and enter a name. Start needs ≥ 2 players.
3. If a phone dies, the host has **Manual Override** under every waiting step, and the
   **Admin** tab (edit chips, move gifts, skip/reset a turn, undo the last duel).
4. A refreshed host or phone offers **Rejoin** (state lives in Firebase).
5. AI tiebreakers (Counterfeit Canvas critic / Draw Off) need the critic URL in settings —
   on Firebase Hosting it defaults to the `critiqueDrawingGemini` function.

### Deploying
```bash
firebase deploy --only database     # rules (database.rules.json) — test with the emulator first
firebase deploy --only functions    # AI critic; set model IDs in functions/.env (see AI_ART_CRITIC.md)
firebase deploy --only hosting
```
Pre-flight: `npm run test:e2e` green; Anonymous Auth enabled in the console; for
functions, Blaze plan + Vertex AI API enabled (Gemini) or `OPENAI_API_KEY` secret set.
After deploying functions, `GET https://us-central1-<project>.cloudfunctions.net/health`
shows the models in use.

### Troubleshooting
| Symptom | Fix |
|---|---|
| "Could not create a game" / `permission_denied` locally | emulators not running → `npm run emulator` |
| Same in production | Anonymous sign-in disabled, or rules not deployed |
| AI critic error in production | model ID retired → update `GEMINI_MODEL` in `functions/.env`, redeploy |
| AI critic "Connection refused" locally | start `python ai_art_critic.py` (and `ollama serve`) |

## Project docs
- [`CLAUDE.md`](CLAUDE.md) — architecture & conventions (start here when changing code)
- [`CODE_REVIEW.md`](CODE_REVIEW.md) — review findings, fixed bugs, open roadmap, rules-vs-code decisions
- [`FIREBASE_SETUP.md`](FIREBASE_SETUP.md) · [`AI_ART_CRITIC.md`](AI_ART_CRITIC.md) · [`GAME_FLOW_AUDIT.md`](GAME_FLOW_AUDIT.md)
