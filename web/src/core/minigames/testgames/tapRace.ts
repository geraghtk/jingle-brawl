// Tap Race: a trivial game for tests only. It proves a game registers and runs on the runtime
// without changing it: the first duelist to N taps wins; when time runs out, more taps wins.
import type { GameDefinition } from '../contracts';
import { headToHead } from '../scoring';

interface Config { target: number; ms: number }
interface State { taps: Record<string, number>; endsAt: number }

export const tapRace: GameDefinition<Config, State, { tap: true }> = {
  id: 'tap_race', version: '1', label: 'Tap Race', emoji: '👆', rules: 'Tap fast.',
  categories: ['real_time_action'], supportedModes: ['direct_duel', 'party_with_duelists'],
  compatibility: { version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'none', topology: ['independent_phone'],
    devices: [{ feature: 'touch', required: true }], players: { min: 2, max: 20 }, modes: ['direct_duel', 'party_with_duelists'] },
  validateConfig: (raw) => ({ target: 5, ms: 10_000, ...(raw as Partial<Config>) }),
  canAlwaysResolve: (mode) => (mode === 'direct_duel' ? { ok: true } : { ok: false, reason: 'two duelists can tie on taps' }),
  createInitialState: (ctx, c) => ({ state: { taps: {}, endsAt: ctx.now + c.ms }, deadlines: [{ id: 'end', at: ctx.now + c.ms }] }),
  validateAction: (ctx, _s, who, action) => (
    ctx.duelists?.includes(who) && (action as { tap?: unknown })?.tap === true ? { ok: true, action: { tap: true } } : { ok: false, error: 'not a tap' }),
  reduce(ctx, s, who) {
    const taps = { ...s.taps, [who]: (s.taps?.[who] || 0) + 1 };
    const state = { ...s, taps };
    if (ctx.mode === 'direct_duel' && taps[who] >= 5) return { state, result: headToHead.result('tap_race', ctx.instanceId, ctx.mode, { scores: taps, reason: 'first to 5' }, ctx.duelists) };
    return { state };
  },
  onDeadline: (ctx, s) => ({ state: s, result: headToHead.result('tap_race', ctx.instanceId, ctx.mode, { scores: s.taps || {}, reason: 'more taps' }, ctx.duelists) }),
  offers: (ctx, _s, who) => (ctx.duelists?.includes(who) ? [{ tap: true }] : []),
  projectForPlayer: (s, pid) => ({ mine: s.taps?.[pid] || 0 }),
  projectForTv: (s) => ({ taps: s.taps || {} }),
  projectForHost: (s) => ({ taps: s.taps || {} }),
};
