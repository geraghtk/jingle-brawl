// Where a room's AI critic comes from on this server: the host's own key (BYO), else a default
// from the environment (JB_OPENAI_KEY / JB_GEMINI_KEY, for your own parties), else the local
// critic (JB_CRITIC_URL, default http://127.0.0.1:5050). JB_FAKE_CRITIC=1: the fake one (tests).
// The local critic's address is the server's to set, never a room's (no fetching URLs a host typed).
import type { CriticSource } from '../room/actor';
import { fakeCritic, localCritic, makeCritic } from './critic';

export function criticSource(env: NodeJS.ProcessEnv = process.env): CriticSource {
  if (env.JB_FAKE_CRITIC === '1') return { make: () => fakeCritic(), fallback: () => fakeCritic() };
  return {
    make: (provider, key) => makeCritic(provider, key),
    fallback(provider) {
      if (provider === 'local') return localCritic(env.JB_CRITIC_URL || 'http://127.0.0.1:5050');
      const key = provider === 'openai' ? env.JB_OPENAI_KEY : env.JB_GEMINI_KEY;
      return key ? makeCritic(provider, key) : null;
    },
  };
}
