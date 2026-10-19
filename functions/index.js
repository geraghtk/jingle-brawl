/**
 * Jingle Brawl - Firebase Cloud Functions (AI Art Critic for production)
 *
 * critiqueDrawing:       OpenAI vision model   — needs: firebase functions:secrets:set OPENAI_API_KEY
 * critiqueDrawingGemini: Gemini on Vertex AI   — uses project credentials (enable the Vertex AI API)
 * Both: POST { image, secretWord } -> { description, rating, critique, error }
 * (same contract as the local ai_art_critic.py /critique endpoint)
 *
 * Model IDs churn (gemini-2.0-flash-001 was retired 2026-06-01), so they are config,
 * not code. Override without a code change by setting them in functions/.env:
 *   GEMINI_MODEL=gemini-3.8-flash
 *   OPENAI_VISION_MODEL=gpt-5.2
 */

const functions = require("firebase-functions");
const { onRequest } = require("firebase-functions/v2/https");
const { defineSecret } = require("firebase-functions/params");
const OpenAI = require("openai");

const openaiApiKey = defineSecret("OPENAI_API_KEY");
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.8-flash";
const OPENAI_MODEL = process.env.OPENAI_VISION_MODEL || "gpt-5.2";

// These endpoints are public and paid per call: bound the inputs
const MAX_IMAGE_B64_CHARS = 8 * 1024 * 1024;
const MAX_SECRET_WORD_CHARS = 200;

// CORS for browser requests from Firebase Hosting
const cors = (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return true;
  }
  return false;
};

const critiquePrompt = (secretWord) => `Look at this drawing. It may be a collaborative drawing or a single person's sketch.

1. Describe in detail what you see: shapes, objects, figures, colors, composition. What does the drawing depict?
2. The intended subject was: "${secretWord}". Rate how well the drawing matches this from 0-100.
3. Give a brief one-sentence critique.

Format your response EXACTLY as:
DESCRIPTION: [your detailed description]
SCORE: [number 0-100]
CRITIQUE: [one sentence]`;

/** Validate a critique request; responds and returns null on bad input. */
function readCritiqueRequest(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return null;
  }
  let data;
  try {
    data = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
  } catch {
    res.status(400).json({ error: "Invalid JSON body" });
    return null;
  }
  let image = data?.image;
  if (!image || typeof image !== "string") {
    res.status(400).json({ error: "Missing 'image' (base64 PNG)" });
    return null;
  }
  if (image.includes(",")) image = image.split(",", 2)[1] || image;
  if (image.length > MAX_IMAGE_B64_CHARS) {
    res.status(413).json({ error: "Image too large" });
    return null;
  }
  const secretWord = String(data?.secretWord || "something").slice(0, MAX_SECRET_WORD_CHARS);
  return { image, secretWord };
}

/** Parse the DESCRIPTION / SCORE / CRITIQUE reply format. */
function parseCritique(text) {
  const description = (text.match(/DESCRIPTION:\s*(.+?)(?=SCORE:|$)/s)?.[1] || "").trim();
  const score = text.match(/SCORE:\s*(\d{1,3})/i);
  const rating = score ? Math.min(100, Math.max(0, parseInt(score[1], 10))) : 50;
  const critique = (text.match(/CRITIQUE:\s*(.+)/s)?.[1] || text).trim();
  return { description, rating, critique, error: null };
}

const failure = (res, err, fallbackMessage) =>
  res.status(500).json({ error: err?.message || fallbackMessage, description: null, rating: 0, critique: null });

exports.critiqueDrawing = onRequest(
  { cors: true, secrets: [openaiApiKey] },
  async (req, res) => {
    if (cors(req, res)) return;
    const input = readCritiqueRequest(req, res);
    if (!input) return;

    const apiKey = openaiApiKey.value();
    if (!apiKey) return failure(res, null, "OpenAI API key not configured. Set the OPENAI_API_KEY secret.");

    try {
      const openai = new OpenAI({ apiKey });
      const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [{
          role: "user",
          content: [
            { type: "text", text: critiquePrompt(input.secretWord) },
            { type: "image_url", image_url: { url: `data:image/png;base64,${input.image}` } },
          ],
        }],
        // GPT-5-family models reject max_tokens, and reasoning tokens count against this
        max_completion_tokens: 2000,
      });
      res.json(parseCritique(response.choices?.[0]?.message?.content || ""));
    } catch (err) {
      functions.logger.error("OpenAI error", err);
      failure(res, err, "OpenAI API error");
    }
  }
);

exports.health = onRequest({ cors: true }, (req, res) => {
  if (cors(req, res)) return;
  res.json({ status: "ok", openaiModel: OPENAI_MODEL, geminiModel: GEMINI_MODEL });
});

// Google Gen AI SDK (@google/genai) — replaces @google-cloud/vertexai, whose generative
// module was removed on 2026-06-24.
async function runGeminiCritique(imageData, secretWord) {
  const { GoogleGenAI } = require("@google/genai");
  const project = process.env.GCLOUD_PROJECT || process.env.GCP_PROJECT;
  if (!project) throw new Error("Project ID not found");

  const ai = new GoogleGenAI({ vertexai: true, project, location: "us-central1" });
  const response = await ai.models.generateContent({
    model: GEMINI_MODEL,
    contents: [{
      role: "user",
      parts: [
        { text: critiquePrompt(secretWord) },
        { inlineData: { data: imageData, mimeType: "image/png" } },
      ],
    }],
  });
  return response.text || "";
}

exports.critiqueDrawingGemini = onRequest(
  { cors: true },
  async (req, res) => {
    if (cors(req, res)) return;
    const input = readCritiqueRequest(req, res);
    if (!input) return;

    try {
      res.json(parseCritique(await runGeminiCritique(input.image, input.secretWord)));
    } catch (err) {
      functions.logger.error("Gemini error", err);
      failure(res, err, "Vertex AI error. Enable the Vertex AI API on your project.");
    }
  }
);

// Exported for unit tests
exports._internal = { parseCritique, critiquePrompt };
