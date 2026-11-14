#!/usr/bin/env python3
"""
AI Art Critic - Local vision model server for Jingle Brawl's Counterfeit Canvas minigame.

Uses Ollama with a vision model (e.g., llava) to:
1. Describe what was drawn in the collaborative canvas
2. Rate how well the drawing matches the secret word the players were asked to draw

Prerequisites:
- Ollama installed: https://ollama.com/download
- Vision model pulled: ollama pull llava
- Dependencies: pip install flask ollama

Run: python ai_art_critic.py
Default: http://localhost:5050
"""

import base64
import json
import os
import re
from flask import Flask, request, jsonify

app = Flask(__name__)
# Only the room server calls this (server to server, JB_CRITIC_URL), never a browser: no CORS, so a
# web page open on this PC can't use it; and no request bigger than a couple of drawings.
MAX_BODY_BYTES = 4 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
# This machine only, unless told otherwise (CRITIC_HOST=0.0.0.0 to serve a room server on another PC:
# anyone who can reach it can then spend the OPENAI_API_KEY it falls back to).
DEFAULT_HOST = "127.0.0.1"

# Config - can override via env
OLLAMA_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENAI_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-5.2")

# House style for everything players see: no em dashes. Asked for in every prompt, and
# stripped from replies anyway (models love them).
NO_DASHES = "Do not use em dashes; use commas, colons or full stops instead. "


def no_dashes(text):
    """Replace em dashes (and en dashes spaced like them) with commas; drop a leading one."""
    if not text:
        return text
    text = re.sub(r"(^|\n)[ \t]*(?:\u2014|\u2013)[ \t]*", r"\1", text)
    text = re.sub(r"[ \t]*\u2014[ \t]*|[ \t]+\u2013[ \t]+", ", ", text)
    return re.sub(r",\s*([,.!?;:])", r"\1", text)


def get_ollama_client():
    """Lazy import to avoid startup failure if ollama not installed."""
    try:
        from ollama import Client
        return Client(host=OLLAMA_HOST)
    except ImportError:
        return None


@app.before_request
def refuse_oversized():
    """Too big to be a drawing or two: say so before anything reads it (the handlers catch every error as a 500)."""
    if (request.content_length or 0) > MAX_BODY_BYTES:
        return jsonify({"error": "Request too large"}), 413


@app.route("/health", methods=["GET"])
def health():
    """Health check - useful for testing if server is running."""
    return jsonify({"status": "ok", "model": OLLAMA_MODEL})


def run_openai_critique(image_b64, secret_word, api_key):
    """Use OpenAI GPT-4 Vision for critique. Returns (description, rating, critique_text, raw_response)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI library not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    image_data = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64

    prompt = (
        f'This is a drawing made on a phone or tablet. Expect messy lines and blocky shapes. '
        f'The prompt was: "{secret_word}".\n\n'
        f'1. Describe what you see in detail.\n'
        f'2. Rate 0-100 how well it matches the prompt. Be forgiving of technical quality.\n'
        f'3. Give a brief, witty one-sentence critique.\n\n'
        f'{NO_DASHES}'
        f'Format EXACTLY as:\nDESCRIPTION: [description]\nSCORE: [number]\nCRITIQUE: [one sentence]'
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }],
        # GPT-5-family models reject max_tokens; their reasoning tokens also count
        # against this budget, so leave headroom or the visible answer comes back empty.
        max_completion_tokens=2000
    )
    text = response.choices[0].message.content or ""
    model_used = getattr(response, "model", None) or OPENAI_MODEL

    description = ""
    rating = 50
    critique_text = text
    if "DESCRIPTION:" in text:
        m = re.search(r"DESCRIPTION:\s*(.+?)(?=SCORE:|$)", text, re.DOTALL)
        description = m.group(1).strip() if m else ""
    if "SCORE:" in text.upper():
        m = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
        if m:
            rating = max(0, min(100, int(m.group(1))))
    if "CRITIQUE:" in text:
        parts = text.split("CRITIQUE:", 1)
        if len(parts) > 1:
            critique_text = parts[1].strip()

    return no_dashes(description), rating, no_dashes(critique_text), text, model_used


@app.route("/critique", methods=["POST"])
def critique():
    """
    Analyze a drawing and rate it against the secret word.
    
    Expects JSON: { "image": "base64-png", "secretWord": "...", "openaiApiKey": "sk-..." (optional) }
    If openaiApiKey provided, uses OpenAI. Otherwise uses Ollama.
    Returns: { "description", "rating", "critique", "rawVisionResponse", "rawRateResponse", "error" }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        
        image_b64 = data.get("image")
        secret_word = data.get("secretWord", "something")
        openai_key = data.get("openaiApiKey") or os.environ.get("OPENAI_API_KEY")
        
        if not image_b64:
            return jsonify({"error": "Missing 'image' (base64 PNG)"}), 400
        
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        # Use OpenAI if key provided
        if openai_key:
            try:
                desc, rating, critique_text, raw, model_used = run_openai_critique(image_b64, secret_word, openai_key)
                return jsonify({
                    "description": desc,
                    "rating": rating,
                    "critique": critique_text,
                    "rawVisionResponse": desc,
                    "rawRateResponse": raw,
                    "backend": "openai",
                    "model": model_used,
                    "error": None
                })
            except Exception as e:
                return jsonify({
                    "error": f"OpenAI error: {str(e)}",
                    "description": None,
                    "rating": 0,
                    "critique": None,
                    "rawVisionResponse": None,
                    "rawRateResponse": None,
                    "backend": None,
                    "model": None
                }), 500

        # Fall back to Ollama
        client = get_ollama_client()
        if not client:
            return jsonify({
                "error": "Neither OpenAI key nor Ollama available. Add openaiApiKey to request or run: pip install ollama",
                "description": None,
                "rating": 0,
                "critique": None
            }), 500
        
        # Step 1: Get detailed description of what's in the image
        describe_prompt = (
            "This is a drawing made on a phone or tablet with a finger or stylus. "
            "Expect messy lines, blocky shapes, and rough proportions. That's normal for device drawings. "
            "Describe in detail what you see: shapes, objects, figures, colors, composition. "
            "What does the drawing depict? Be specific but concise. Focus on intent and recognition, not technical polish. "
            + NO_DASHES
        )
        
        try:
            describe_response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{
                    "role": "user",
                    "content": describe_prompt,
                    "images": [image_b64],
                }],
            )
            description = no_dashes(describe_response.message.content) or "Could not analyze the image."
        except Exception as e:
            return jsonify({
                "error": f"Vision model error: {str(e)}. Is Ollama running? Run: ollama pull llava",
                "description": None,
                "rating": 0,
                "critique": None,
                "rawVisionResponse": None,
                "rawRateResponse": None
            }), 500
        
        # Step 2: Rate the drawing against the secret word
        rate_prompt = (
            f"The prompt players were asked to draw was: \"{secret_word}\".\n\n"
            f"Based on this description of what was actually drawn: \"{description}\"\n\n"
            "Remember: this was drawn on a device (finger/stylus), so lines may be messy and shapes blocky. "
            "Be forgiving of technical quality. Focus on: Is the intended subject recognizable? Are key elements present?\n\n"
            "Rate how well the drawing matches the prompt from 0-100. "
            "Give a single number (0-100) and a brief, WITTY, snarky one-sentence critique "
            "(like a roguish art critic - playful, not mean). "
            + NO_DASHES
            + "Format your response as: SCORE: [number] | CRITIQUE: [your witty one sentence]"
        )
        
        try:
            rate_response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{
                    "role": "user",
                    "content": rate_prompt,
                }],
            )
            rate_text = rate_response.message.content or "Could not rate."
        except Exception as e:
            return jsonify({
                "error": str(e),
                "description": description,
                "rating": 50,
                "critique": "Rating failed",
                "rawVisionResponse": description,
                "rawRateResponse": None
            }), 500
        
        # Parse score and critique from response
        rating = 50
        critique_text = rate_text
        # First number after SCORE: ("SCORE: 85/100" -> 85; joining every digit gave 851)
        m = re.search(r"SCORE:\s*(\d{1,3})", rate_text, re.IGNORECASE)
        if m:
            rating = max(0, min(100, int(m.group(1))))
        if "CRITIQUE:" in rate_text:
            parts = rate_text.split("CRITIQUE:", 1)
            if len(parts) > 1:
                critique_text = parts[1].strip()
        
        return jsonify({
            "description": description,
            "rating": rating,
            "critique": no_dashes(critique_text),
            "rawVisionResponse": description,  # What Ollama "saw" (first call)
            "rawRateResponse": rate_text,      # Full LLM output before parsing (second call)
            "backend": "ollama",
            "model": OLLAMA_MODEL,
            "error": None
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "description": None,
            "rating": 0,
            "critique": None,
            "rawVisionResponse": None,
            "rawRateResponse": None
        }), 500


def parse_winner(text, scores):
    """Map the model's 'WINNER: <name>' line to a player id (None if no clear match)."""
    m = re.search(r"WINNER:\s*(.*)", text, re.IGNORECASE)
    raw = m.group(1).strip().strip("[]*").strip().upper() if m else ""
    if not raw:  # an empty name would "match" every player via `"" in name`
        return None
    names = {pid: (s.get("name") or "").upper() for pid, s in scores.items()}
    exact = next((pid for pid, n in names.items() if n == raw), None)
    return exact or next((pid for pid, n in names.items() if n and (raw in n or n in raw)), None)


@app.route("/witty-verdict", methods=["POST"])
def witty_verdict():
    """
    Given multiple drawings' scores, generate a witty art critic verdict and pick the overall winner.
    Expects JSON: { "prompt": "Penguin", "scores": { "playerId": { "rating": 85, "critique": "...", "name": "Alice" }, ... } }
    Returns: { "wittyVerdict": "...", "winnerId": "playerId", "error": null }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided", "wittyVerdict": None, "winnerId": None}), 400

        prompt = data.get("prompt", "something")
        scores = data.get("scores", {})
        openai_key = data.get("openaiApiKey") or os.environ.get("OPENAI_API_KEY")

        if not scores:
            return jsonify({"error": "Missing 'scores'", "wittyVerdict": None, "winnerId": None}), 400

        # Build detailed summary for LLM (include full description + critique for richer verdict)
        lines = []
        for pid, s in scores.items():
            name = s.get("name", f"Player_{pid[:8]}")
            r = s.get("rating", 0)
            desc = s.get("description", "")
            crit = s.get("critique", "")
            entry = f"- {name}: {r}/100. "
            if desc:
                entry += f"What was drawn: {desc} "
            if crit:
                entry += f"Critique: {crit}"
            lines.append(entry.strip())
        summary = "\n\n".join(lines)
        names_list = ", ".join(s.get("name", f"Player_{pid[:8]}") for pid, s in scores.items())

        verdict_prompt = (
            f'You are a witty, roguish art critic hosting an art competition, like a mix of a sharp-tongued talent-show judge and a gallery owner who\'s seen it all. '
            f'The prompt artists were given was: "{prompt}".\n\n'
            f'These were drawn on phones/tablets (finger or stylus), so lines can be messy and shapes blocky. Judge on creativity and intent, not polish.\n\n'
            f'Here are the submissions with full analysis:\n\n{summary}\n\n'
            f'Write an IN-DEPTH, WITTY verdict (4-8 sentences). '
            f'Address each submission: reference what was actually drawn, riff on the descriptions, and deliver sharp but playful commentary. '
            f'Build tension. Be specific: mention details from the descriptions. Then announce the overall winner with dramatic flair. '
            f'Make it memorable and fun, not generic. {NO_DASHES}\n\n'
            f'After your verdict, on a new line write exactly: WINNER: [name]\n'
            f'Use one of these names: {names_list}'
        )

        try:
            verdict_backend = None
            verdict_model = None
            if openai_key:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": verdict_prompt}],
                    max_completion_tokens=3000
                )
                text = resp.choices[0].message.content or "Could not generate verdict."
                verdict_backend = "openai"
                verdict_model = getattr(resp, "model", None) or OPENAI_MODEL
            else:
                client = get_ollama_client()
                if not client:
                    return jsonify({
                        "error": "Provide openaiApiKey or run Ollama (pip install ollama)",
                        "wittyVerdict": None,
                        "winnerId": None
                    }), 500
                resp = client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": verdict_prompt}])
                text = resp.message.content or "Could not generate verdict."
                verdict_backend = "ollama"
                verdict_model = OLLAMA_MODEL
        except Exception as e:
            return jsonify({
                "error": str(e),
                "wittyVerdict": None,
                "winnerId": None
            }), 500

        # Parse winner by name
        winner_id = parse_winner(text, scores)
        if not winner_id:
            winner_id = max(scores.keys(), key=lambda p: scores[p].get("rating", 0))

        verdict_text = no_dashes(re.split(r"WINNER:", text, maxsplit=1, flags=re.IGNORECASE)[0].strip())

        return jsonify({
            "wittyVerdict": verdict_text,
            "winnerId": winner_id,
            "backend": verdict_backend,
            "model": verdict_model,
            "error": None
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "wittyVerdict": None,
            "winnerId": None
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"AI Art Critic server starting on http://localhost:{port}")
    print(f"Using vision model: {OLLAMA_MODEL}")
    print("Make sure Ollama is running: ollama serve")
    print("And pull a vision model: ollama pull llava")
    app.run(host=os.environ.get("CRITIC_HOST", DEFAULT_HOST), port=port, debug=False)
