// Keep the screen awake (the Screen Wake Lock API): a phone must not dim in the middle of a mini-game,
// and the TV must not go to sleep at all. The browser lets go of the lock whenever the page is hidden,
// so it is asked for again when the page comes back. Browsers only offer it on a secure page (https,
// or localhost): on a plain Wi-Fi address there is nothing to ask, and this quietly does nothing.
import { useEffect } from 'react';

interface Sentinel { release(): Promise<void>; addEventListener(type: 'release', fn: () => void): void }
interface Locker { request(type: 'screen'): Promise<Sentinel> }
export interface WakeEnv { locker?: Locker; visible(): boolean; onVisible(fn: () => void): () => void }

/** Hold the screen awake until the returned function is called. Exported for the test; screens use `useWakeLock`. */
export function holdAwake(env: WakeEnv): () => void {
  let held: Sentinel | null = null, asking = false, done = false;
  const ask = () => {
    if (done || held || asking || !env.locker || !env.visible()) return;
    asking = true;
    env.locker.request('screen').then((s) => {
      asking = false;
      if (done) { void s.release().catch(() => {}); return; }
      held = s;
      s.addEventListener('release', () => { if (held === s) held = null; });
    }, () => { asking = false; });           // refused (battery saver, not allowed): the screen may dim, the game still works
  };
  ask();
  const off = env.onVisible(ask);
  return () => { done = true; off(); const s = held; held = null; if (s) void s.release().catch(() => {}); };
}

const browserEnv = (): WakeEnv => ({
  locker: (navigator as unknown as { wakeLock?: Locker }).wakeLock,
  visible: () => document.visibilityState === 'visible',
  onVisible: (fn) => { document.addEventListener('visibilitychange', fn); return () => document.removeEventListener('visibilitychange', fn); },
});

/** Keep this screen awake while `on`. */
export function useWakeLock(on: boolean): void {
  useEffect(() => {
    if (!on || typeof navigator === 'undefined') return;
    try { return holdAwake(browserEnv()); } catch { return; }
  }, [on]);
}
