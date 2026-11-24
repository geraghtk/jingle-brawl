// Pass the Gremlin (spec, game #1). A gremlin jumps between phones. Whoever holds it must solve a
// quick challenge before passing it on, to someone they pick or at random (never straight back to
// whoever just passed it, if anyone else can take it; not before a short minimum hold). A secret
// deadline, kept by the server and never shown, makes it go berserk: whoever holds it then loses.
//   direct_duel          the two duelists pass it; the doomed one loses, the other wins
//   free_for_all         one loser, everyone else survives; no winner is invented
//   party_with_duelists  everyone passes it; a doomed duelist loses the duel. A doomed bystander is
//                        out and the rest play on with a shorter fuse; after `maxOut` of them (or
//                        when no bystander is left) the two duelists finish alone in sudden death,
//                        which always ends on one of the two. Bystanders earn Party points for
//                        how long they lasted (`rewards`: the game only recommends them).
// The TV and the phones see only a coarse danger level, from public settings and elapsed time.
import type { Rng } from '../../rng';
import type { PlayerId } from '../../types';
import type { GameDefinition, MatchCtx, Transition } from '../contracts';
import { singleLoser } from '../scoring';
import { ALL_CHALLENGES, CHALLENGES, nextChallenge } from './challenges';
import type { Challenge } from './challenges';
import { outPoints } from './view';
import type { GremlinView } from './view';

export interface GremlinConfig {
  doomMinMs: number; doomMaxMs: number;            // the main round's secret deadline window
  tiebreakMinMs: number; tiebreakMaxMs: number;    // the sudden death's
  minHoldMs: number;                               // the holder keeps it at least this long
  wrongCooldownMs: number;                         // a wrong answer: a new problem after this
  introMs: number;                                 // before the sudden death (or the next round) starts
  laterMinMs: number; laterMaxMs: number;          // a round after a bystander went out: a shorter fuse
  maxOut: number;                                  // this many bystanders out, then sudden death
  surviveReward: number;                           // Party points for a bystander still in at the end
  challenges: string[];                            // the kinds a holder may be given (each new one: another kind)
  challenge?: string;                              // (before M3: one kind)
}
export const DEFAULTS: GremlinConfig = {
  doomMinMs: 60_000, doomMaxMs: 100_000, tiebreakMinMs: 20_000, tiebreakMaxMs: 35_000,
  minHoldMs: 1000, wrongCooldownMs: 700, introMs: 4000, laterMinMs: 35_000, laterMaxMs: 55_000, maxOut: 2, surviveReward: 1,
  challenges: ALL_CHALLENGES,
};

/** The kinds this game uses (a game saved before M3 has one `challenge`). */
export const kindsOf = (c: Partial<GremlinConfig>): string[] => (c.challenges?.length ? c.challenges : [c.challenge || 'quick_math']);

export type Danger = 'calm' | 'uneasy' | 'unstable';
export interface Pass { from: PlayerId; to: PlayerId; at: number; random: boolean }
export interface Stats { holds: number; passes: number; ok: number; fail: number }

export interface GremlinState {
  cfg: GremlinConfig;                   // this game's settings (public)
  stage: 'holding' | 'pass_ready' | 'out_intro' | 'sudden_death_intro' | 'doomed';
  round: 'main' | 'sudden_death';
  roundNo?: number;                     // 1, 2, ... (a game saved before elimination has none: 1)
  out?: PlayerId[];                     // bystanders it caught, first out first
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
  bystanderDoomed?: PlayerId | null;    // the bystander it caught last
  doomed?: PlayerId | null;
}

export type GremlinAction = { kind: 'answer'; value: unknown } | { kind: 'pass'; to: PlayerId } | { kind: 'pass_random' };

const sample = (rng: Rng, [lo, hi]: [number, number]) => lo + Math.floor(rng() * (hi - lo + 1));
const stat = (s: GremlinState, pid: PlayerId): Stats => ({ holds: 0, passes: 0, ok: 0, fail: 0, ...(s.stats?.[pid] || {}) });

export function danger(s: Pick<GremlinState, 'roundStartedAt' | 'window'>, now: number): Danger {
  const t = now - s.roundStartedAt;
  return t < s.window[0] * 0.5 ? 'calm' : t < s.window[0] ? 'uneasy' : 'unstable';
}

const between = (s: GremlinState) => s.stage === 'doomed' || s.stage === 'sudden_death_intro' || s.stage === 'out_intro';

/** A bystander's Party points for lasting: `surviveReward` for whoever is still in (and at the table)
 *  when a duelist is caught; less for one who was caught (`outPoints`: by default nothing). Kept small
 *  on purpose: most bystanders are never caught, and a duelist earns 1 for fighting. */
export function survivalRewards(c: GremlinConfig, participants: PlayerId[], duelists: PlayerId[], out: PlayerId[], here: (p: PlayerId) => boolean) {
  const rewards: { playerId: PlayerId; points: number; reason: string }[] = [];
  for (const p of participants) {
    if (duelists.includes(p)) continue;
    const i = out.indexOf(p);
    const points = i >= 0 ? outPoints(i, c.surviveReward) : here(p) ? c.surviveReward : 0;
    if (points > 0) rewards.push({ playerId: p, points, reason: i >= 0 ? 'lasting a while' : 'never caught' });
  }
  return rewards;
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
  const window: [number, number] = round === 'sudden_death' ? [c.tiebreakMinMs, c.tiebreakMaxMs] : (extra.roundNo || 1) > 1 ? [c.laterMinMs, c.laterMaxMs] : [c.doomMinMs, c.doomMaxMs];
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
    if (!(c.doomMinMs > 0 && c.doomMaxMs >= c.doomMinMs && c.tiebreakMinMs > 0 && c.tiebreakMaxMs >= c.tiebreakMinMs && c.laterMinMs > 0 && c.laterMaxMs >= c.laterMinMs)) throw new Error('bad Gremlin deadline windows');
    for (const k of c.challenges) if (!CHALLENGES[k]) throw new Error(`no challenge ${k}`);
    return c;
  },

  // the doom always lands on someone holding it; with duelists, a bystander's doom goes to sudden death between them
  // every time the state holds moves on by the pause (the hold timer, the mood, a challenge's own clock, the secret moment)
  shiftTime(s, ms) {
    const later = (t: number | undefined) => (t ? t + ms : t);
    const c = s.challenge as GremlinState['challenge'] & { params?: { startAt?: number } };
    const challenge = c ? { ...c, issuedAt: c.issuedAt + ms, ...(c.params && typeof c.params.startAt === 'number' ? { params: { ...c.params, startAt: c.params.startAt + ms } } : {}) } : c;
    return { ...s, holdSince: s.holdSince + ms, roundStartedAt: s.roundStartedAt + ms, doomAt: s.doomAt + ms, ...(s.cooldownUntil ? { cooldownUntil: later(s.cooldownUntil) } : {}), challenge: challenge as GremlinState['challenge'] };
  },

  canAlwaysResolve: (mode) => (mode === 'free_for_all' ? { ok: false, reason: 'a free-for-all has no duel winner' } : { ok: true }),

  createInitialState(ctx, c) {
    const pool = ctx.mode === 'direct_duel' && ctx.duelists ? [...ctx.duelists] : [...ctx.participants];
    return newRound(ctx, c, 'main', pool);
  },

  validateAction(ctx, s, who, raw) {
    const a = raw as Partial<GremlinAction> & { value?: unknown; to?: unknown };
    if (between(s)) return { ok: false, error: 'Wait for it…' };
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
    const carry = { stats: s.stats, passes: s.passes, bystanderDoomed: s.bystanderDoomed, out: s.out, roundNo: (s.roundNo || 1) + 1 };
    // the duelists only, a fresh secret deadline
    if (id === 'sudden_death') return newRound(ctx, s.cfg, 'sudden_death', [...ctx.duelists!], carry);
    // a bystander is out: everyone else plays on, on a shorter fuse
    if (id === 'next_round') return newRound(ctx, s.cfg, 'main', s.pool.filter((p) => !(s.out || []).includes(p)), carry);
    // the doom: whoever holds it now loses
    const loser = s.holder;
    const doomed: GremlinState = { ...s, stage: 'doomed', doomed: loser };
    if (ctx.mode === 'party_with_duelists' && ctx.duelists && !ctx.duelists.includes(loser)) {
      const out = [...(s.out || []), loser];
      const left = s.pool.filter((p) => !ctx.duelists!.includes(p) && !out.includes(p));
      const on = out.length < s.cfg.maxOut && left.length > 0;
      return {
        state: { ...doomed, stage: on ? 'out_intro' : 'sudden_death_intro', bystanderDoomed: loser, out },
        deadlines: [{ id: on ? 'next_round' : 'sudden_death', at: ctx.now + s.cfg.introMs }],
        events: [on ? '👹 The Gremlin went berserk on a bystander! They are out; everyone else plays on.'
          : '👹 The Gremlin went berserk on a bystander! Sudden death: just the duellists now.'],
      };
    }
    const out = s.out || [];
    const metrics = { passes: s.passes?.length || 0, stats: s.stats || {}, bystanderDoomed: s.bystanderDoomed ?? null, out, rounds: s.roundNo || 1 };
    const rewards = ctx.mode === 'party_with_duelists' && ctx.duelists ? survivalRewards(s.cfg, ctx.participants, ctx.duelists, out, ctx.connected) : [];
    const result = singleLoser.result(gremlin.id, ctx.instanceId, ctx.mode,
      { loser, participants: ctx.participants, reason: s.round === 'sudden_death' ? 'holding the Gremlin at sudden death' : 'holding the Gremlin when it went berserk' },
      ctx.mode === 'free_for_all' ? undefined : ctx.duelists);
    return { state: doomed, result: { ...result, ...(rewards.length ? { rewards } : {}), metrics: { ...result.metrics, ...metrics } }, events: ['👹 The Gremlin went berserk!'] };
  },

  offers(ctx, s, who) {
    if (who !== s.holder || between(s)) return [];
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
function publicView(s: GremlinState, now: number): GremlinView {
  return {
    stage: s.stage, round: s.round, pool: s.pool, holder: s.holder, prevHolder: s.prevHolder ?? null,
    challengeKind: s.challenge.kind, challengeLabel: CHALLENGES[s.challenge.kind].label, solved: s.stage === 'pass_ready', danger: danger(s, now),
    trail: (s.passes || []).slice(-8), bystanderDoomed: s.bystanderDoomed ?? null, doomed: s.doomed ?? null,
    roundNo: s.roundNo || 1, out: s.out || [], maxOut: s.cfg.maxOut, survive: s.cfg.surviveReward,
    // for the big screen's table: since when the holder has had it, their wrong answers so far, and how
    // often each player has held it (nothing here helps anyone guess the deadline)
    holdSince: s.holdSince, misses: stat(s, s.holder).fail,
    held: Object.fromEntries(Object.entries(s.stats || {}).map(([p, st]) => [p, st.holds || 0]).filter(([, n]) => n)),
  };
}
