/**
 * Jingle Brawl - Firebase Cloud Functions
 *
 * critiqueDrawing: AI Art Critic for production (uses OpenAI GPT-4 Vision)
 * Same API as local ai_art_critic.py: POST { image, secretWord } -> { description, rating, critique }
 *
 * Set secret: firebase functions:secrets:set OPENAI_API_KEY
 */

const functions = require("firebase-functions");
const { onRequest } = require("firebase-functions/v2/https");
const { defineSecret } = require("firebase-functions/params");
const OpenAI = require("openai");

const openaiApiKey = defineSecret("OPENAI_API_KEY");

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

exports.critiqueDrawing = onRequest(
  { cors: true, secrets: [openaiApiKey] },
  async (req, res) => {
    if (cors(req, res)) return;

    if (req.method !== "POST") {
      res.status(405).json({ error: "Method not allowed" });
      return;
    }

    const apiKey = openaiApiKey.value();
    if (!apiKey) {
      res.status(500).json({
        error: "OpenAI API key not configured. Set OPENAI_API_KEY secret or openai.api_key config.",
        description: null,
        rating: 0,
        critique: null,
      });
      return;
    }

    let data;
    try {
      data = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    } catch {
      res.status(400).json({ error: "Invalid JSON body" });
      return;
    }

    const imageB64 = data?.image;
    const secretWord = data?.secretWord || "something";

    if (!imageB64) {
      res.status(400).json({ error: "Missing 'image' (base64 PNG)" });
      return;
    }

    let imageData = imageB64;
    if (imageData.includes(",")) {
      imageData = imageData.split(",", 2)[1] || imageData;
    }

    const openai = new OpenAI({ apiKey });

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `Look at this drawing. It may be a collaborative drawing or a single person's sketch.

1. Describe in detail what you see: shapes, objects, figures, colors, composition. What does the drawing depict?
2. The intended subject was: "${secretWord}". Rate how well the drawing matches this from 0-100.
3. Give a brief one-sentence critique.

Format your response EXACTLY as:
DESCRIPTION: [your detailed description]
SCORE: [number 0-100]
CRITIQUE: [one sentence]`,
              },
              {
                type: "image_url",
                image_url: { url: `data:image/png;base64,${imageData}` },
              },
            ],
          },
        ],
        max_tokens: 400,
      });

      const text = response.choices?.[0]?.message?.content || "";
      let description = "";
      let rating = 50;
      let critique = text;

      if (text.includes("DESCRIPTION:")) {
        const descMatch = text.match(/DESCRIPTION:\s*(.+?)(?=SCORE:|$)/s);
        description = descMatch ? descMatch[1].trim() : "";
      }
      if (text.includes("SCORE:")) {
        const scoreMatch = text.match(/SCORE:\s*(\d+)/);
        if (scoreMatch) rating = Math.min(100, Math.max(0, parseInt(scoreMatch[1], 10)));
      }
      if (text.includes("CRITIQUE:")) {
        const critMatch = text.match(/CRITIQUE:\s*(.+)/s);
        critique = critMatch ? critMatch[1].trim() : critique;
      }

      res.json({ description, rating, critique, error: null });
    } catch (err) {
      functions.logger.error("OpenAI error", err);
      res.status(500).json({
        error: err.message || "OpenAI API error",
        description: null,
        rating: 0,
        critique: null,
      });
    }
  }
);

exports.health = onRequest({ cors: true }, (req, res) => {
  if (cors(req, res)) return;
  res.json({ status: "ok", model: "gpt-4o" });
});

/**
 * critiqueDrawingGemini: Same API as critiqueDrawing but uses Vertex AI Gemini.
 * No API key needed - uses Firebase project credentials.
 * Enable Vertex AI API: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
 */
async function runGeminiCritique(imageData, secretWord) {
  const { VertexAI } = require("@google-cloud/vertexai");
  const projectId = process.env.GCLOUD_PROJECT || process.env.GCP_PROJECT;
  if (!projectId) throw new Error("Project ID not found");

  const vertexAI = new VertexAI({ project: projectId, location: "us-central1" });
  const model = vertexAI.getGenerativeModel({ model: "gemini-2.0-flash-001" });

  const prompt = `Look at this drawing. It may be a collaborative drawing or a single person's sketch.

1. Describe in detail what you see: shapes, objects, figures, colors, composition. What does the drawing depict?
2. The intended subject was: "${secretWord}". Rate how well the drawing matches this from 0-100.
3. Give a brief one-sentence critique.

Format your response EXACTLY as:
DESCRIPTION: [your detailed description]
SCORE: [number 0-100]
CRITIQUE: [one sentence]`;

  const request = {
    contents: [{
      role: "user",
      parts: [
        { text: prompt },
        { inlineData: { data: imageData, mimeType: "image/png" } },
      ],
    }],
  };

  const result = await model.generateContent(request);
  return result.response?.candidates?.[0]?.content?.parts?.[0]?.text || "";
}

exports.critiqueDrawingGemini = onRequest(
  { cors: true },
  async (req, res) => {
    if (cors(req, res)) return;

    if (req.method !== "POST") {
      res.status(405).json({ error: "Method not allowed" });
      return;
    }

    let data;
    try {
      data = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    } catch {
      res.status(400).json({ error: "Invalid JSON body" });
      return;
    }

    const imageB64 = data?.image;
    const secretWord = data?.secretWord || "something";

    if (!imageB64) {
      res.status(400).json({ error: "Missing 'image' (base64 PNG)" });
      return;
    }

    let imageData = imageB64;
    if (imageData.includes(",")) {
      imageData = imageData.split(",", 2)[1] || imageData;
    }

    try {
      const text = await runGeminiCritique(imageData, secretWord);
      let description = "";
      let rating = 50;
      let critique = text;

      if (text.includes("DESCRIPTION:")) {
        const descMatch = text.match(/DESCRIPTION:\s*(.+?)(?=SCORE:|$)/s);
        description = descMatch ? descMatch[1].trim() : "";
      }
      if (text.includes("SCORE:")) {
        const scoreMatch = text.match(/SCORE:\s*(\d+)/);
        if (scoreMatch) rating = Math.min(100, Math.max(0, parseInt(scoreMatch[1], 10)));
      }
      if (text.includes("CRITIQUE:")) {
        const critMatch = text.match(/CRITIQUE:\s*(.+)/s);
        critique = critMatch ? critMatch[1].trim() : critique;
      }

      res.json({ description, rating, critique, error: null });
    } catch (err) {
      functions.logger.error("Gemini error", err);
      res.status(500).json({
        error: err.message || "Vertex AI error. Enable Vertex AI API on your project.",
        description: null,
        rating: 0,
        critique: null,
      });
    }
  }
);
