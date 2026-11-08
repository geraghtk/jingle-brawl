// Pass the Gremlin (spec, game #1). A gremlin jumps between phones. Whoever holds it must solve a
// quick challenge before passing it on, to someone they pick or at random (never straight back to
// whoever just passed it, if anyone else can take it; not before a short minimum hold). A secret
// deadline, kept by the server and never shown, makes it go berserk: whoever holds it then loses.
//   direct_duel          the two duelists pass it; the doomed one loses, the other wins
//   free_for_all         one loser, everyone else survives; no winner is invented
//   party_with_duelists  everyone passes it; a doomed duelist loses the duel. A doomed bystander
//                        starts a duelist-only sudden death with a fresh secret deadline, which
//                        always ends on one of the two.
// The TV and the phones see only a coarse danger level, from public settings and elapsed time.
import type { Rng } from '../../rng';
import type { PlayerId } from '../../types';
import type { GameDefinition, MatchCtx, Transition } from '../contracts';
import { singleLoser } from '../scoring';
import { ALL_CHALLENGES, CHALLENGES, nextChallenge } from './challenges';
import type { Challenge } from './challenges';

export interface GremlinConfig {
  doomMinMs: number; doomMaxMs: number;            // the main round's secret deadline window
  tiebreakMinMs: number; tiebreakMaxMs: number;    // the sudden death's
  minHoldMs: number;                               // the holder keeps it at least this long
  wrongCooldownMs: number;                         // a wrong answer: a new problem after this
  introMs: number;                                 // before the sudden death starts
  challenges: string[];                            // the kinds a holder may be given (each new one: another kind)
  challenge?: string;                              // (before M3: one kind)
}
export const DEFAULTS: GremlinConfig = {
  doomMinMs: 60_000, doomMaxMs: 100_000, tiebreakMinMs: 20_000, tiebreakMaxMs: 35_000,
  minHoldMs: 1000, wrongCooldownMs: 700, introMs: 4000, challenges: ALL_CHALLENGES,
};

/** The kinds this game uses (a game saved before M3 has one `challenge`). */
export const kindsOf = (c: Partial<GremlinConfig>): string[] => (c.challenges?.length ? c.challenges : [c.challenge || 'quick_math']);

export type Danger = 'calm' | 'uneasy' | 'unstable';
export interface Pass { from: PlayerId; to: PlayerId; at: number; random: boolean }
export interface Stats { holds: number; passes: number; ok: number; fail: number }

export interface GremlinState {
  cfg: GremlinConfig;                   // this game's settings (public)
  stage: 'holding' | 'pass_ready' | 'sudden_death_intro' | 'doomed';
  round: 'main' | 'sudden_death';
  pool: PlayerId[];                     // who can hold it this round
  holder: PlayerId;
  prevHolder?: PlayerId | null;
  holdSince: number;
  challenge: Challenge;                 // secret: the answer
  cooldownUntil?: number;
  roundStartedAt: number;
  doomAt: number;                       // secret: never in a view
  window: [number, number];             // this round's public window (for the danger level)
  passes?: Pass[];
  stats?: Record<PlayerId, Stats>;
  bystanderDoomed?: PlayerId | null;    // the main round's loser when it wasn't a duelist
  doomed?: PlayerId | null;
}

export type GremlinAction = { kind: 'answer'; value: unknown } | { kind: 'pass'; to: PlayerId } | { kind: 'pass_random' };

const sample = (rng: Rng, [lo, hi]: [number, number]) => lo + Math.floor(rng() * (hi - lo + 1));
const stat = (s: GremlinState, pid: PlayerId): Stats => ({ holds: 0, passes: 0, ok: 0, fail: 0, ...(s.stats?.[pid] || {}) });

export function danger(s: Pick<GremlinState, 'roundStartedAt' | 'window'>, now: number): Danger {
  const t = now - s.roundStartedAt;
  return t < s.window[0] * 0.5 ? 'calm' : t < s.window[0] ? 'uneasy' : 'unstable';
}

/** Who the holder may pass to: the pool but themselves, not straight back if anyone else can take it, and connected ones if any are. */
export function eligible(ctx: Pick<MatchCtx, 'connected'>, s: GremlinState): PlayerId[] {
  let out = s.pool.filter((p) => p !== s.holder);
  const notBack = out.filter((p) => p !== s.prevHolder);
  if (notBack.length) out = notBack;
  const here = out.filter((p) => ctx.connected(p));
  return here.length ? here : out;
}

function newRound(ctx: MatchCtx, c: GremlinConfig, round: GremlinState['round'], pool: PlayerId[], extra: Partial<GremlinState> = {}): Transition<GremlinState> {
  const window: [number, number] = round === 'main' ? [c.doomMinMs, c.doomMaxMs] : [c.tiebreakMinMs, c.tiebreakMaxMs];
  const holder = pool[Math.floor(ctx.rng() * pool.length)];
  const doomAt = ctx.now + sample(ctx.rng, window);
  const state: GremlinState = {
    ...extra, cfg: c, stage: 'holding', round, pool, holder, prevHolder: null, holdSince: ctx.now,
    challenge: nextChallenge(kindsOf(c), ctx.rng, ctx.now), roundStartedAt: ctx.now, doomAt, window,
  };
  state.stats = { ...(extra.stats || {}), [holder]: { ...stat(state, holder), holds: stat(state, holder).holds + 1 } };
  return { state, deadlines: [{ id: 'doom', at: doomAt }] };
}

export const gremlin: GameDefinition<GremlinConfig, GremlinState, GremlinAction> = {
  id: 'pass_the_gremlin', version: '1', label: 'Pass the Gremlin', emoji: '👹',
  rules: "Holding the Gremlin? Solve the problem, then pass it on. It goes berserk at a secret moment: don't be holding it!",
  categories: ['real_time_action', 'party_social', 'cross_device'],
  supportedModes: ['direct_duel', 'party_with_duelists', 'free_for_all'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'optional_status', topology: ['independent_phone', 'phone_to_phone'],
    devices: [{ feature: 'touch', required: true }, { feature: 'haptics', required: false }, { feature: 'audio', required: false }],
    players: { min: 2, max: 20 }, modes: ['direct_duel', 'party_with_duelists', 'free_for_all'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<GremlinConfig>;
    const c: GremlinConfig = { ...DEFAULTS, ...r, challenges: r.challenges || r.challenge ? kindsOf(r) : DEFAULTS.challenges };
    delete c.challenge;
    if (!(c.doomMinMs > 0 && c.doomMaxMs >= c.doomMinMs && c.tiebreakMinMs > 0 && c.tiebreakMaxMs >= c.tiebreakMinMs)) throw new Error('bad Gremlin deadline windows');
    for (const k of c.challenges) if (!CHALLENGES[k]) throw new Error(`no challenge ${k}`);
    return c;
  },

  // the doom always lands on someone holding it; with duelists, a bystander's doom goes to sudden death between them
  canAlwaysResolve: (mode) => (mode === 'free_for_all' ? { ok: false, reason: 'a free-for-all has no duel winner' } : { ok: true }),

  createInitialState(ctx, c) {
    const pool = ctx.mode === 'direct_duel' && ctx.duelists ? [...ctx.duelists] : [...ctx.participants];
    return newRound(ctx, c, 'main', pool);
  },

  validateAction(ctx, s, who, raw) {
    const a = raw as Partial<GremlinAction> & { value?: unknown; to?: unknown };
    if (s.stage === 'doomed' || s.stage === 'sudden_death_intro') return { ok: false, error: 'Wait for it…' };
    if (who !== s.holder) return { ok: false, error: "You're not holding the Gremlin" };
    if (a?.kind === 'answer') {
      if (s.stage !== 'holding') return { ok: false, error: 'Already solved: pass it on!' };
      if (s.cooldownUntil && ctx.now < s.cooldownUntil) return { ok: false, error: 'Wait for the next one' };
      const value = CHALLENGES[s.challenge.kind].parse(a.value);
      if (value === null) return { ok: false, error: s.challenge.kind === 'quick_math' ? 'Type a number' : 'Not an answer' };
      return { ok: true, action: { kind: 'answer', value } };
    }
    if (a?.kind === 'pass' || a?.kind === 'pass_random') {
      if (s.stage !== 'pass_ready') return { ok: false, error: 'Solve the problem first' };
      if (ctx.now < s.holdSince + s.cfg.minHoldMs) return { ok: false, error: 'Hold on a moment…' };
      const can = eligible(ctx, s);
      if (a.kind === 'pass') {
        if (typeof a.to !== 'string' || !can.includes(a.to)) return { ok: false, error: a.to === s.prevHolder ? "Not straight back to them!" : "You can't pass to them" };
        return { ok: true, action: { kind: 'pass', to: a.to } };
      }
      return { ok: true, action: { kind: 'pass_random' } };
    }
    return { ok: false, error: 'Not a Gremlin move' };
  },

  reduce(ctx, s, who, a) {
    const cfg = s.cfg;
    if (a.kind === 'answer') {
      const st = stat(s, who);
      if (CHALLENGES[s.challenge.kind].check(s.challenge, a.value, ctx.now)) {
        return { state: { ...s, stage: 'pass_ready', cooldownUntil: undefined, stats: { ...s.stats, [who]: { ...st, ok: st.ok + 1 } } } };
      }
      // wrong: another of the same kind, after a short wait
      return { state: { ...s, challenge: CHALLENGES[s.challenge.kind].make(ctx.rng, ctx.now), cooldownUntil: ctx.now + cfg.wrongCooldownMs,
        stats: { ...s.stats, [who]: { ...st, fail: st.fail + 1 } } } };
    }
    const can = eligible(ctx, s);
    const random = a.kind === 'pass_random';
    const to = random ? can[Math.floor(ctx.rng() * can.length)] : (a as { to: PlayerId }).to;
    const from = stat(s, who), next = stat(s, to);
    return {
      state: {
        ...s, stage: 'holding', holder: to, prevHolder: who, holdSince: ctx.now, cooldownUntil: undefined,
        challenge: nextChallenge(kindsOf(cfg), ctx.rng, ctx.now, s.challenge.kind),
        passes: [...(s.passes || []), { from: who, to, at: ctx.now, random }].slice(-40),
        stats: { ...s.stats, [who]: { ...from, passes: from.passes + 1 }, [to]: { ...next, holds: next.holds + 1 } },
      },
    };
  },

  onDeadline(ctx, s, id) {
    if (id === 'sudden_death') {
      // the duelists only, a fresh secret deadline
      return newRound(ctx, s.cfg, 'sudden_death', [...ctx.duelists!], { stats: s.stats, passes: s.passes, bystanderDoomed: s.bystanderDoomed });
    }
    // the doom: whoever holds it now loses
    const loser = s.holder;
    const doomed: GremlinState = { ...s, stage: 'doomed', doomed: loser };
    const metrics = { passes: s.passes?.length || 0, stats: s.stats || {}, bystanderDoomed: s.bystanderDoomed ?? null };
    if (ctx.mode === 'party_with_duelists' && ctx.duelists && !ctx.duelists.includes(loser)) {
      return {
        state: { ...doomed, stage: 'sudden_death_intro', bystanderDoomed: loser },
        deadlines: [{ id: 'sudden_death', at: ctx.now + s.cfg.introMs }],
        events: ['👹 The Gremlin went berserk on a bystander! Sudden death: just the duellists now.'],
      };
    }
    const result = singleLoser.result(gremlin.id, ctx.instanceId, ctx.mode,
      { loser, participants: ctx.participants, reason: s.round === 'sudden_death' ? 'holding the Gremlin at sudden death' : 'holding the Gremlin when it went berserk' },
      ctx.mode === 'free_for_all' ? undefined : ctx.duelists);
    return { state: doomed, result: { ...result, metrics: { ...result.metrics, ...metrics } }, events: ['👹 The Gremlin went berserk!'] };
  },

  offers(ctx, s, who) {
    if (who !== s.holder || s.stage === 'doomed' || s.stage === 'sudden_death_intro') return [];
    if (s.stage === 'holding') {   // simulated hands; never sent to a screen
      const value = ctx.now < (s.cooldownUntil || 0) ? null : CHALLENGES[s.challenge.kind].solve(s.challenge, ctx.now);
      return value === null ? [] : [{ kind: 'answer', value }];
    }
    if (ctx.now < s.holdSince + s.cfg.minHoldMs) return [];
    return [{ kind: 'pass_random' }, ...eligible(ctx, s).map((to): GremlinAction => ({ kind: 'pass', to }))];
  },

  projectForPlayer(s, pid, now) {
    const mine = s.holder === pid;
    return {
      ...publicView(s, now),
      ...(mine && s.stage === 'holding' ? { ...CHALLENGES[s.challenge.kind].view(s.challenge), issuedAt: s.challenge.issuedAt, cooldownUntil: s.cooldownUntil ?? null } : {}),
      ...(mine && s.stage === 'pass_ready' ? { canPassAt: s.holdSince + s.cfg.minHoldMs, targets: eligible({ connected: () => true }, s) } : {}),
    };
  },
  projectForTv: (s, now) => publicView(s, now),
  projectForHost: (s, now) => publicView(s, now),
};

/** What everyone may know: never the answer, never the deadline. */
function publicView(s: GremlinState, now: number) {
  return {
    stage: s.stage, round: s.round, pool: s.pool, holder: s.holder, prevHolder: s.prevHolder ?? null,
    challengeKind: s.challenge.kind, challengeLabel: CHALLENGES[s.challenge.kind].label, solved: s.stage === 'pass_ready', danger: danger(s, now),
    trail: (s.passes || []).slice(-8), bystanderDoomed: s.bystanderDoomed ?? null, doomed: s.doomed ?? null,
  };
}
