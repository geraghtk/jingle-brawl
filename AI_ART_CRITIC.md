# AI Art Critic - Setup Guide

The AI Art Critic adds a fun "robot judge" that analyzes drawings in two mini-games:

**The Counterfeit Canvas** – Collaborative drawing; AI rates the shared canvas against the secret word.

**AI Draw Off** – Each player draws individually (same prompt, 3 minutes); AI judges and ranks all submissions.

## Prerequisites

### 1. Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download).

### 2. Pull a vision model

```bash
ollama pull llava
```

Other options: `llava:13b` (larger), `llava:34b` (largest), or `llava3` (newer).

### 3. Install Python dependencies

```bash
pip install -r ai_art_critic_requirements.txt
```

## Running the AI Art Critic

1. **Start Ollama** (usually runs automatically; if not: `ollama serve`)
2. **Start the AI Art Critic server:**

   ```bash
   python ai_art_critic.py
   ```

   Server runs at `http://localhost:5050` by default.

3. **Create a Jingle Brawl game** with:
   - Tiebreaker: **The Counterfeit Canvas Mini-Game** or **AI Draw Off** (Draw Off requires the critic and auto-enables it)
   - ✅ **Enable AI Art Critic**
   - Engine: **Ollama** (local, free) or **OpenAI** (paste a key; it stays in the host's browser session and is only sent to this local server)
   - Server URL: `http://localhost:5050` (default)

Counterfeit Canvas: when drawing ends, the AI rates the shared canvas and the rating appears during voting.
Draw Off: after 3 minutes the AI scores each submission, then `/witty-verdict` writes a critic's verdict and picks the winner (falls back to the highest score if that call fails).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5050 | Server port |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama API URL |
| `OLLAMA_VISION_MODEL` | llava | Vision model name |
| `OPENAI_VISION_MODEL` | gpt-5.2 | OpenAI model when a key is supplied |
| `OPENAI_API_KEY` | — | Server-side key (used if the host sends none) |

Unit tests for the reply parsing (no model needed): `python -m unittest discover -s tests`.

---

## Production (Firebase)

For production on Firebase Hosting, use a **Cloud Function** instead of local Ollama. Two options:

### Option A: Gemini (recommended – no API key)

Uses Firebase / Vertex AI Gemini. No secrets to manage – uses your project credentials.

1. **Enable Vertex AI API**  
   [Enable the API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) for your Firebase project.

2. **Deploy**
   ```bash
   npm install --prefix functions
   firebase deploy --only functions
   ```

3. **Use the Gemini function URL** (default when hosted on Firebase):
   ```
   https://us-central1-YOUR_PROJECT.cloudfunctions.net/critiqueDrawingGemini
   ```

### Option B: OpenAI

Uses an OpenAI vision model (`gpt-5.2` by default). Requires an API key.

1. **Set secret**
   ```bash
   firebase functions:secrets:set OPENAI_API_KEY
   ```
   Get a key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

2. **Deploy**
   ```bash
   firebase deploy --only functions
   ```

3. **Use the OpenAI function URL**
   ```
   https://us-central1-YOUR_PROJECT.cloudfunctions.net/critiqueDrawing
   ```

### Choosing models (Cloud Functions)

Model IDs are configuration, not code — they get retired (`gemini-2.0-flash-001` was
retired 2026-06-01, which broke the old Gemini function). Set them in `functions/.env`
and redeploy:

```
GEMINI_MODEL=gemini-3.8-flash        # check the Vertex AI "Model versions" page for a current stable ID
OPENAI_VISION_MODEL=gpt-5.2
```

`GET .../health` reports which models a deployed function is using. The Gemini function
uses the Google Gen AI SDK (`@google/genai`); the old `@google-cloud/vertexai` generative
module was removed by Google on 2026-06-24. Parser tests: `cd functions && npm test`.

**Draw Off on Cloud Functions:** there is no `witty-verdict` function, so the verdict
step is skipped and the highest AI score wins. Use the local server for verdicts.

### Firebase Blaze plan

Cloud Functions require the [Blaze (pay-as-you-go) plan](https://firebase.google.com/pricing).

### Deploy hosting

```bash
firebase deploy --only hosting
# or
firebase deploy
```

When the game is served from `*.web.app` or `*.firebaseapp.com`, the AI Critic URL defaults to the Gemini function.

---

## Troubleshooting

**Local (Ollama):**
- **"Vision model error"** – Ensure Ollama is running and you've pulled a vision model: `ollama pull llava`
- **"Connection refused"** – Ensure `ai_art_critic.py` is running and the URL in host settings matches

**Production (Cloud Functions):**
- **Gemini: "Vertex AI error"** – Enable Vertex AI API: [console.cloud.google.com/apis/library/aiplatform.googleapis.com](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)
- **OpenAI: "API key not configured"** – Run `firebase functions:secrets:set OPENAI_API_KEY`
- **CORS errors** – Functions have `cors: true`; ensure you're using the correct function URL

**Both:**
- **Empty drawing** – The AI critic runs even if the canvas is nearly empty; it will describe what it sees
