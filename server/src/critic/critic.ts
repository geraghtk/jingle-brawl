// The AI Art Critic, called by the room server for an AI Draw Off duel. Each room brings its own
// key (BYO): it is handed in here and never stored. Providers: OpenAI, Gemini (an AI Studio key),
// the local critic (ai_art_critic.py, Ollama) and a fake one for tests. The winner is always the
// higher score; the witty verdict is flavour and can't overrule it.

export interface Score { rating: number; description: string; critique: string; backend: string; model: string }
export interface Critic {
  readonly backend: string;
  /** Throws if the key (or the local critic) doesn't work: called once when the key is given. */
  check(): Promise<void>;
  judge(image: string, prompt: string): Promise<Score>;
  /** A witty line about the two drawings, naming the winner already decided (null if it can't). */
  verdict(prompt: string, winner: string, entries: { name: string; rating: number; description: string }[]): Promise<string | null>;
}
export type Provider = 'openai' | 'gemini' | 'local';

const OPENAI_MODEL = process.env.OPENAI_VISION_MODEL || 'gpt-5.2';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const TIMEOUT_MS = 45_000;

export const critiquePrompt = (word: string) => `Look at this drawing. It may be a quick sketch on a phone.

1. Describe in detail what you see: shapes, objects, figures, colors, composition. What does the drawing depict?
2. The intended subject was: "${word.slice(0, 200)}". Rate how well the drawing matches this from 0-100.
3. Give a brief, witty one-sentence critique.

Do not use em dashes; use commas, colons or full stops instead.

Format your response EXACTLY as:
DESCRIPTION: [your detailed description]
SCORE: [number 0-100]
CRITIQUE: [one sentence]`;

const verdictPrompt = (word: string, winner: string, entries: { name: string; rating: number; description: string }[]) =>
  `You are a witty, roguish art critic judging a two-person drawing duel at a holiday party. The prompt was "${word.slice(0, 200)}".
${entries.map((e) => `${e.name} (score ${e.rating}/100): ${e.description.slice(0, 400)}`).join('\n')}
The winner is ${winner}. In at most three sentences, build a little tension and announce ${winner} as the winner with dramatic flair.
Be kind to the loser. Do not use em dashes.`;

/** No em dashes in anything players see (models use them anyway). */
const EM = String.fromCharCode(0x2014), EN = String.fromCharCode(0x2013); // (not literal: copy.test checks this file)
export const noDashes = (text: string) => text
  .replace(new RegExp(`(^|\\n)[ \\t]*[${EM}${EN}][ \\t]*`, 'g'), '$1')
  .replace(new RegExp(`[ \\t]*${EM}[ \\t]*|[ \\t]+${EN}[ \\t]+`, 'g'), ', ')
  .replace(/,\s*([,.!?;:])/g, '$1');

/** The DESCRIPTION / SCORE / CRITIQUE reply. */
export function parseCritique(text: string): Omit<Score, 'backend' | 'model'> {
  const description = noDashes((text.match(/DESCRIPTION:\s*([\s\S]+?)(?=SCORE:|$)/)?.[1] || '').trim()).slice(0, 1000);
  const score = text.match(/SCORE:\s*(\d{1,3})/i);
  const rating = score ? Math.min(100, Math.max(0, parseInt(score[1], 10))) : 50;
  const critique = noDashes((text.match(/CRITIQUE:\s*([\s\S]+)/)?.[1] || text).trim()).slice(0, 400);
  return { description, rating, critique };
}

const split = (dataUrl: string) => {
  const m = /^data:(image\/[a-z]+);base64,(.+)$/.exec(dataUrl);
  if (!m) throw new Error('not an image');
  return { mime: m[1], data: m[2] };
};

async function post(url: string, headers: Record<string, string>, body: unknown): Promise<unknown> {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body), signal: AbortSignal.timeout(TIMEOUT_MS) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(((data as { error?: { message?: string } | string }).error as { message?: string })?.message || String((data as { error?: string }).error || res.statusText));
  return data;
}

export function openaiCritic(key: string, model = OPENAI_MODEL): Critic {
  const auth = { Authorization: `Bearer ${key}` };
  const ask = async (content: unknown[]) => {
    const r = await post('https://api.openai.com/v1/chat/completions', auth, { model, messages: [{ role: 'user', content }], max_completion_tokens: 2000 }) as
      { choices?: { message?: { content?: string } }[] };
    return r.choices?.[0]?.message?.content || '';
  };
  return {
    backend: 'openai',
    async check() {
      const res = await fetch('https://api.openai.com/v1/models', { headers: auth, signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (!res.ok) throw new Error(res.status === 401 ? 'OpenAI refused that key' : `OpenAI: ${res.status}`);
    },
    async judge(image, prompt) {
      const text = await ask([{ type: 'text', text: critiquePrompt(prompt) }, { type: 'image_url', image_url: { url: image } }]);
      return { ...parseCritique(text), backend: 'openai', model };
    },
    async verdict(prompt, winner, entries) {
      return noDashes(await ask([{ type: 'text', text: verdictPrompt(prompt, winner, entries) }])).slice(0, 600) || null;
    },
  };
}

export function geminiCritic(key: string, model = GEMINI_MODEL): Critic {
  const auth = { 'x-goog-api-key': key };
  const ask = async (parts: unknown[]) => {
    const r = await post(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, auth, { contents: [{ parts }] }) as
      { candidates?: { content?: { parts?: { text?: string }[] } }[] };
    return (r.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('');
  };
  return {
    backend: 'gemini',
    async check() {
      const res = await fetch('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1', { headers: auth, signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (!res.ok) throw new Error(res.status === 400 || res.status === 403 ? 'Google refused that key' : `Gemini: ${res.status}`);
    },
    async judge(image, prompt) {
      const { mime, data } = split(image);
      const text = await ask([{ text: critiquePrompt(prompt) }, { inline_data: { mime_type: mime, data } }]);
      return { ...parseCritique(text), backend: 'gemini', model };
    },
    async verdict(prompt, winner, entries) {
      return noDashes(await ask([{ text: verdictPrompt(prompt, winner, entries) }])).slice(0, 600) || null;
    },
  };
}

/** The local critic (python ai_art_critic.py, Ollama), same contract as before. */
export function localCritic(url: string): Critic {
  const base = url.replace(/\/$/, '');
  return {
    backend: 'local',
    async check() {
      const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error('the local critic is not answering');
    },
    async judge(image, prompt) {
      const d = await post(`${base}/critique`, {}, { image, secretWord: prompt }) as Partial<Score> & { rating?: number };
      return { rating: Math.min(100, Math.max(0, Number(d.rating) || 0)), description: noDashes(String(d.description || '')), critique: noDashes(String(d.critique || '')),
        backend: String(d.backend || 'local'), model: String(d.model || '') };
    },
    async verdict(prompt, winner, entries) {
      const scores = Object.fromEntries(entries.map((e, i) => [`p${i}`, { ...e }]));
      const d = await post(`${base}/witty-verdict`, {}, { prompt, scores }).catch(() => null) as { wittyVerdict?: string } | null;
      return d?.wittyVerdict ? noDashes(d.wittyVerdict).slice(0, 600) : `${winner} takes it!`;
    },
  };
}

/** Tests: a drawing's score is its length (so a test can choose the winner), no network. */
export function fakeCritic(): Critic {
  return {
    backend: 'fake',
    async check() {},
    async judge(image) { return { rating: image.length % 101, description: 'a drawing', critique: 'Bold lines.', backend: 'fake', model: 'fake' }; },
    async verdict(_p, winner) { return `${winner} wins it, by a whisker.`; },
  };
}

export function makeCritic(provider: Provider, keyOrUrl: string): Critic {
  if (provider === 'openai') return openaiCritic(keyOrUrl);
  if (provider === 'gemini') return geminiCritic(keyOrUrl);
  return localCritic(keyOrUrl);
}
