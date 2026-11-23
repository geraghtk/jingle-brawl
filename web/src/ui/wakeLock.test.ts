import { describe, expect, it } from 'vitest';
import { holdAwake } from './wakeLock';
import type { WakeEnv } from './wakeLock';

function fake(visible = true) {
  const log: string[] = [];
  let onRelease: (() => void) | null = null, onVisible: (() => void) | null = null;
  const env: WakeEnv & { show(v: boolean): void; systemRelease(): void } = {
    locker: { request: async () => { log.push('ask'); return { release: async () => { log.push('release'); }, addEventListener: (_t, fn) => { onRelease = fn; } }; } },
    visible: () => visible,
    onVisible: (fn) => { onVisible = fn; return () => { onVisible = null; log.push('unlisten'); }; },
    show(v) { visible = v; onVisible?.(); },
    systemRelease() { onRelease?.(); },
  };
  return { env, log };
}
const settle = () => new Promise((r) => setTimeout(r, 0));

describe('keeping the screen awake', () => {
  it('asks once, asks again when the page comes back after the browser let go, and lets go at the end', async () => {
    const { env, log } = fake();
    const stop = holdAwake(env);
    await settle();
    expect(log).toEqual(['ask']);
    env.show(true);                         // still held: no second lock
    await settle();
    expect(log).toEqual(['ask']);
    env.show(false); env.systemRelease();   // the page was hidden: the browser dropped it
    env.show(true);
    await settle();
    expect(log).toEqual(['ask', 'ask']);
    stop();
    await settle();
    expect(log).toEqual(['ask', 'ask', 'unlisten', 'release']);
  });

  it('a hidden page waits; a browser without it, or one that refuses, is no error', async () => {
    const { env, log } = fake(false);
    const stop = holdAwake(env);
    await settle();
    expect(log).toEqual([]);
    stop();
    expect(() => holdAwake({ visible: () => true, onVisible: () => () => {} })()).not.toThrow();
    const no: WakeEnv = { locker: { request: () => Promise.reject(new Error('no')) }, visible: () => true, onVisible: () => () => {} };
    const end = holdAwake(no);
    await settle();
    expect(() => end()).not.toThrow();
  });

  it('stopped before the lock arrived: it is given straight back', async () => {
    const { env, log } = fake();
    holdAwake(env)();
    await settle();
    expect(log).toEqual(['ask', 'unlisten', 'release']);
  });
});
