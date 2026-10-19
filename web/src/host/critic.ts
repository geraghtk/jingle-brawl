// AI Art Critic client: the local Flask server (/critique, /witty-verdict) or the Cloud Function.
import type { DrawOffScore } from '../core/drawoff';

export async function critique(baseUrl: string, image: string, prompt: string, openaiApiKey?: string): Promise<DrawOffScore> {
  const base = baseUrl.replace(/\/$/, '');
  const cloud = base.includes('cloudfunctions.net');
  const body: Record<string, string> = { image, secretWord: prompt };
  if (openaiApiKey && !cloud) body.openaiApiKey = openaiApiKey;
  const res = await fetch(cloud ? base : `${base}/critique`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return {
    rating: data.rating || 0, description: data.description || '', critique: data.critique || '',
    rawVisionResponse: data.rawVisionResponse || data.description || '', rawRateResponse: data.rawRateResponse || '',
    backend: data.backend, model: data.model,
  };
}

/** Optional: a witty overall verdict that may name the winner. */
export async function wittyVerdict(baseUrl: string, prompt: string, scores: Record<string, DrawOffScore & { name: string }>, openaiApiKey?: string) {
  try {
    const body: Record<string, unknown> = { prompt, scores };
    if (openaiApiKey) body.openaiApiKey = openaiApiKey;
    const res = await fetch(`${baseUrl.replace(/\/$/, '')}/witty-verdict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error || !data.wittyVerdict) return {};
    return { text: data.wittyVerdict as string, winnerId: data.winnerId as string | undefined, backend: data.backend, model: data.model };
  } catch {
    return {};
  }
}

/** OpenAI key for the critic: kept in this tab only (never written to the database). */
export const openaiKeyFor = (room: string) => sessionStorage.getItem(`jingleBrawl_openai_key_${room}`) || undefined;
