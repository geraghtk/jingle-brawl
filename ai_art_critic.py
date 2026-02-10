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
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow host.html to call from same origin or different port

# Config - can override via env
OLLAMA_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def get_ollama_client():
    """Lazy import to avoid startup failure if ollama not installed."""
    try:
        from ollama import Client
        return Client(host=OLLAMA_HOST)
    except ImportError:
        return None


@app.route("/health", methods=["GET"])
def health():
    """Health check - useful for testing if server is running."""
    return jsonify({"status": "ok", "model": OLLAMA_MODEL})


@app.route("/critique", methods=["POST"])
def critique():
    """
    Analyze a drawing and rate it against the secret word.
    
    Expects JSON body: { "image": "base64-encoded-png", "secretWord": "Snowman" }
    Returns: { "description": "...", "rating": 0-100, "critique": "...", "error": null }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        
        image_b64 = data.get("image")
        secret_word = data.get("secretWord", "something")
        
        if not image_b64:
            return jsonify({"error": "Missing 'image' (base64 PNG)"}), 400
        
        # Strip data URL prefix if present (e.g., "data:image/png;base64,...")
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        
        client = get_ollama_client()
        if not client:
            return jsonify({
                "error": "Ollama Python library not installed. Run: pip install ollama",
                "description": None,
                "rating": 0,
                "critique": None
            }), 500
        
        # Step 1: Get detailed description of what's in the image
        describe_prompt = (
            "Look at this collaborative drawing. It was made by multiple people taking turns "
            "adding one stroke at a time. Describe in detail what you see: shapes, objects, "
            "figures, colors, composition. What does the drawing depict? Be specific but concise."
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
            description = describe_response.message.content or "Could not analyze the image."
        except Exception as e:
            return jsonify({
                "error": f"Vision model error: {str(e)}. Is Ollama running? Run: ollama pull llava",
                "description": None,
                "rating": 0,
                "critique": None
            }), 500
        
        # Step 2: Rate the drawing against the secret word
        rate_prompt = (
            f"The secret word players were asked to draw was: \"{secret_word}\".\n\n"
            f"Based on this description of what was actually drawn: \"{description}\"\n\n"
            "Rate how well the drawing matches the secret word from 0-100. "
            "Consider: Is the intended subject recognizable? Are key features present? "
            "Give a single number (0-100) and a brief one-sentence critique. "
            "Format your response as: SCORE: [number] | CRITIQUE: [your one sentence]"
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
                "critique": "Rating failed"
            }), 500
        
        # Parse score and critique from response
        rating = 50
        critique_text = rate_text
        if "SCORE:" in rate_text.upper():
            parts = rate_text.upper().split("SCORE:", 1)
            if len(parts) > 1:
                score_part = parts[1].split("|")[0].strip()
                try:
                    rating = int("".join(c for c in score_part if c.isdigit())[:3])
                    rating = max(0, min(100, rating))
                except (ValueError, IndexError):
                    pass
        if "CRITIQUE:" in rate_text:
            parts = rate_text.split("CRITIQUE:", 1)
            if len(parts) > 1:
                critique_text = parts[1].strip()
        
        return jsonify({
            "description": description,
            "rating": rating,
            "critique": critique_text,
            "error": None
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "description": None,
            "rating": 0,
            "critique": None
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"AI Art Critic server starting on http://localhost:{port}")
    print(f"Using vision model: {OLLAMA_MODEL}")
    print("Make sure Ollama is running: ollama serve")
    print("And pull a vision model: ollama pull llava")
    app.run(host="0.0.0.0", port=port, debug=False)
