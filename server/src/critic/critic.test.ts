// The critic's reply parsing and the providers' calls (fetch mocked: no network, no key spent).
import { afterEach, describe, expect, it, vi } from 'vitest';
import { geminiCritic, noDashes, openaiCritic, parseCritique } from './critic';

afterEach(() => vi.unstubAllGlobals());

describe('parseCritique', () => {
  it('reads the three parts, clamps the score, and keeps em dashes out', () => {
    const r = parseCritique('DESCRIPTION: A cat — maybe a dog.\nSCORE: 140\nCRITIQUE: Bold — and brave.');
    expect(r).toEqual({ description: 'A cat, maybe a dog.', rating: 100, critique: 'Bold, and brave.' });
    expect(parseCritique('no format at all').rating).toBe(50);
    expect(noDashes('— Leading')).toBe('Leading');
  });
});

describe('providers', () => {
  const reply = (body: unknown, ok = true, status = 200) => vi.fn(async () => ({ ok, status, statusText: 'x', json: async () => body }));

  it('OpenAI: the key goes in the Authorization header only; the drawing goes as an image', async () => {
    const fetchMock = reply({ choices: [{ message: { content: 'DESCRIPTION: a tree\nSCORE: 77\nCRITIQUE: Festive.' } }] });
    vi.stubGlobal('fetch', fetchMock);
    const s = await openaiCritic('sk-test', 'm1').judge('data:image/png;base64,AAAA', 'A tree');
    expect(s).toMatchObject({ rating: 77, description: 'a tree', backend: 'openai', model: 'm1' });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://api.openai.com/v1/chat/completions');
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer sk-test');
    expect(String(init.body)).not.toContain('sk-test');
    expect(String(init.body)).toContain('data:image/png;base64,AAAA');
  });

  it('Gemini: the key goes in its header; a refused key fails the check with a plain reason', async () => {
    const ok = reply({ candidates: [{ content: { parts: [{ text: 'DESCRIPTION: a star\nSCORE: 60\nCRITIQUE: Shiny.' }] } }] });
    vi.stubGlobal('fetch', ok);
    expect((await geminiCritic('g-key', 'gm').judge('data:image/jpeg;base64,BBBB', 'A star')).rating).toBe(60);
    const [url, init] = ok.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).not.toContain('g-key');
    expect((init.headers as Record<string, string>)['x-goog-api-key']).toBe('g-key');
    vi.stubGlobal('fetch', reply({}, false, 403));
    await expect(geminiCritic('bad').check()).rejects.toThrow(/refused that key/);
  });
});
