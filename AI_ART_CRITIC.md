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
   - Tiebreaker: **The Counterfeit Canvas Mini-Game**
   - ✅ **Enable AI Art Critic**
   - Server URL: `http://localhost:5050` (default)

When the drawing phase ends, the AI will analyze the canvas and its rating will appear during the voting phase.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5050 | Server port |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama API URL |
| `OLLAMA_VISION_MODEL` | llava | Vision model name |

## Troubleshooting

- **"Vision model error"** – Ensure Ollama is running and you've pulled a vision model: `ollama pull llava`
- **"Connection refused"** – Ensure `ai_art_critic.py` is running and the URL in host settings matches
- **CORS errors** – The server enables CORS; if issues persist, serve `host.html` from the same origin as the API
- **Empty drawing** – The AI critic runs even if the canvas is nearly empty; it will describe what it sees
