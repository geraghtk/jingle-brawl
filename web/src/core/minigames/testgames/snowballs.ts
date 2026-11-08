// A second throw game on the same engines (spec, M4 acceptance: the gesture and flight capabilities
// register in another game without touching the runtime). Snowball Splat: one big target, ten
// throws, the highest total wins. For tests only; not registered in the app.
import type { GameDefinition } from '../contracts';
import { launchFrom, parsePath } from '../engines/gesture';
import type { Launch } from '../engines/gesture';
import { DEFAULT_FLIGHT, landing } from '../engines/projectile';
import { score } from '../engines/targets';
import type { Peg } from '../engines/targets';
import { headToHead } from '../scoring';

const TARGET: Peg[] = [{ id: 'bull', x: 0, z: 1.2, rx: 0.08, rz: 0.12, points: 10 }, { id: 'board', x: 0, z: 1.2, rx: 0.25, rz: 0.35, points: 2 }];
interface S { left: Record<string, number>; scores: Record<string, number> }

export const snowballs: GameDefinition<{ throws: number }, S, { launch: Launch }> = {
  id: 'snowball_splat', version: '1', label: 'Snowball Splat', emoji: '⛄', rules: 'Hit the target.',
  categories: ['real_time_action'], supportedModes: ['direct_duel'],
  compatibility: { version: 1, environments: ['in_person'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display'], devices: [], players: { min: 2, max: 2 }, modes: ['direct_duel'] },
  validateConfig: () => ({ throws: 3 }),
  canAlwaysResolve: () => ({ ok: true }),
  createInitialState: (ctx, c) => ({ state: { left: Object.fromEntries(ctx.participants.map((p) => [p, c.throws])), scores: {} } }),
  validateAction(_ctx, s, who, raw) {
    const path = parsePath((raw as { path?: unknown })?.path);
    const launch = path && launchFrom(path);
    if (!launch || !(s.left[who] > 0)) return { ok: false, error: 'no' };
    return { ok: true, action: { launch } };
  },
  reduce(ctx, s, who, a) {
    const l = landing(a.launch, DEFAULT_FLIGHT);
    const scores = s.scores || {};                     // stored state: an empty object comes back missing
    const state = { left: { ...s.left, [who]: s.left[who] - 1 }, scores: { ...scores, [who]: (scores[who] || 0) + score(TARGET, l.x, l.z).points } };
    const over = Object.values(state.left).every((n) => n === 0);
    return { state, ...(over ? { result: headToHead.result('snowball_splat', ctx.instanceId, ctx.mode, { scores: state.scores, reason: 'all thrown' }, ctx.duelists) } : {}) };
  },
  onDeadline: (_c, s) => ({ state: s }),
  offers: () => [],
  projectForPlayer: (s) => s, projectForTv: (s) => s, projectForHost: (s) => s,
};
