# AI Art Critic (AI Draw Off)

AI Draw Off is an in-app **duel game**: the two duellists get the same prompt and draw on their
phones, the room watches on the TV, and an AI art critic scores both drawings. The higher score
wins the duel; a short witty verdict names the winner (flavour only: it can't overrule the
scores). The room server calls the critic (`server/src/critic/`); no screen ever talks to an AI
provider.

## Turning it on

At setup, under **Duels you'll play in the room**, tick **🎨 AI Draw Off** and choose a critic:

| Critic | What you need |
|---|---|
| **OpenAI** | your own API key (platform.openai.com/api-keys) |
| **Google Gemini** | your own API key (aistudio.google.com/apikey) |
| **Local critic on this PC** | `ollama pull llava` and `python ai_art_critic.py` on the PC running the game (free) |

## Your key (BYO)

- It goes from the host's screen to the room server **once**, over the host's own connection.
- The server keeps it **in memory only**, for that room: never in the game, the database
  (`rooms.db`), the event log, or any screen's view. The host console shows only "Ready".
- It is checked once with the provider; a refused key is reported without the key in the message.
- It is **forgotten** when the game ends, when the room has been idle for 2 hours, when the host
  presses **Forget**, or when the server restarts (the host console then asks for it again).
- Calls are billed to your account; each room is limited to 12 critic calls per 10 minutes.
- Hosted online, the site must be HTTPS so the key is encrypted on the way to the server.

## Server defaults (for your own parties)

Environment variables on the room server:

| Variable | Meaning |
|---|---|
| `JB_OPENAI_KEY` / `JB_GEMINI_KEY` | a default key for rooms that pick that provider and don't bring one |
| `JB_CRITIC_URL` | the local critic's address (default `http://127.0.0.1:5050`); rooms can't set a URL |
| `OPENAI_VISION_MODEL` / `GEMINI_MODEL` | model IDs (they churn; defaults in `critic.ts`) |
| `JB_FAKE_CRITIC=1` | tests: a fake critic, no network |

## When the critic can't decide

No key, a provider error, or a tie: nobody wins by default. The TV shows the drawings and the
reason; back in the duel, the host's winner buttons unlock ("you pick the winner") and the event
log marks the result as a host decision.

## The local critic

`ai_art_critic.py` (Flask on :5050, Ollama `llava`): `/critique` scores one drawing,
`/witty-verdict` writes the verdict, `/health`. Unit tests: `python -m unittest discover -s tests`.
