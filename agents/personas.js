// Agent personas for the play-testing harness (agents/play.js).
//
// A persona is a set of PURE decision functions. Each receives a `view` — only what
// that player could legitimately know at the table (no peeking at sealed bids, no
// other players' chip counts unless the game shows them) — and returns a decision
// plus a one-line "thought" the commentary feed shows.
//
// Gift value: every gift has a hidden base quality (1-10) and every player a personal
// taste offset, so v(gift) differs per player. A player only knows v() for gifts that
// have been opened (everyone sees them unwrapped). Giftless players compare against
// `view.expectedValue` — what an unknown/misfit gift is worth to them on average.
//
// The cast is modelled on people you actually meet at a gift exchange. Each is a mix
// of the same traits (see PARAMETERS below), so new types are a few numbers away.

/** Seeded PRNG (mulberry32) so a game is reproducible from its seed. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// --- PARAMETERS ----------------------------------------------------------------
// Every persona sets these (DEFAULTS fills the rest):
//   bidThreshold   minimum desire (value gain) before bidding at all
//   bidScale       chips per point of desire
//   noise          ± random wobble on bids
//   reserve        fraction of starting chips held back early (released as the game ends)
//   endgameSaver   keep chips for the Misfit Lottery: spend nothing until the last gifts
//   allIn          bid everything allowed, or nothing (no half measures)
//   herd           0-1: pull my bid toward the room's recent winning bids (FOMO)
//   attachment     extra value on the gift I'm holding (endowment effect)
//   yieldBelow     as Opener facing 2+ bidders, yield when my gift is worth less than this
//   gambitMin      Grinch's Gambit only if the (spite-adjusted) gain beats this
//   reprisalMin    Reindeer Reprisal only if the gain beats this
//   costAversion   how much each chip of Minimum Cost reduces a target's appeal
//   conflictAverse 0-1: chance of backing out of any attack it would otherwise make
//   spite          bonus value for hitting someone who beat me
//   nemesis        bonus value for hitting my one chosen rival (whoever the arena assigns)
//   leaderHunter   bonus value for hitting whoever holds the best gift (to me)
//   stealBias      endgame: preference for Path A (steal) over Path B (auction), 0-1
//   voteRate       how often they vote on other people's duels
//   random         ignore all of the above and do something random
const DEFAULTS = {
  bidThreshold: 2, bidScale: 0.6, noise: 0.4, reserve: 0.3, endgameSaver: false, allIn: false,
  herd: 0, attachment: 0, yieldBelow: 3, gambitMin: 2, reprisalMin: 2, costAversion: 1,
  conflictAverse: 0, spite: 0, nemesis: 0, leaderHunter: 0, stealBias: 0.5, voteRate: 0.5, random: false,
};

const PERSONAS = {
  // --- the original six, now named for who they are ---------------------------
  miser: {
    label: 'The Accountant', emoji: '🧾',
    blurb: 'Counts every chip: small bids, only on clear upgrades, hoards for later.',
    bidThreshold: 3, bidScale: 0.35, reserve: 0.6, yieldBelow: 3, gambitMin: 5, reprisalMin: 4, costAversion: 1.5, stealBias: 0.15, voteRate: 0.3,
  },
  brawler: {
    label: 'The Alpha', emoji: '🥊',
    blurb: 'Office-party alpha: bids big and often, never yields, always takes the Gambit/Reprisal.',
    bidThreshold: 0.5, bidScale: 1.0, noise: 0.8, reserve: 0, yieldBelow: -99, gambitMin: 0, reprisalMin: 0, costAversion: 0.2, stealBias: 0.9, voteRate: 0.9,
  },
  sniper: {
    label: 'The Collector', emoji: '🎯',
    blurb: 'Knows exactly what they want: ignores average gifts, goes all-in on great ones.',
    bidThreshold: 4, bidScale: 2.0, noise: 0.3, yieldBelow: 5, gambitMin: 3, reprisalMin: 3, costAversion: 0.6, stealBias: 0.6,
  },
  wildcard: {
    label: 'The Party Animal', emoji: '🎲',
    blurb: 'Three eggnogs in: random bids and choices, never read the rules.',
    random: true, voteRate: 0.7,
  },
  grinch: {
    label: 'The Grudge Holder', emoji: '😈',
    blurb: 'Never forgets: medium bids, but goes after whoever beat them.',
    bidThreshold: 1.5, bidScale: 0.7, noise: 0.6, reserve: 0.2, yieldBelow: 2, gambitMin: 1, reprisalMin: -1, costAversion: 0.5, stealBias: 0.8, voteRate: 0.6, spite: 3,
  },
  strategist: {
    label: 'The Chess Player', emoji: '♟️',
    blurb: 'Plays the long game: modest bids early, chips saved for the endgame, yields weak gifts.',
    bidThreshold: 2, bidScale: 0.6, noise: 0.3, reserve: 0.5, yieldBelow: 5, gambitMin: 2.5, reprisalMin: 2, costAversion: 1.0,
  },

  // --- more real-world types -----------------------------------------------------
  peacekeeper: {
    label: 'The People-Pleaser', emoji: '🕊️',
    blurb: 'Hates conflict: rarely bids, yields readily, almost never Gambits or Reprisals.',
    bidThreshold: 4, bidScale: 0.4, reserve: 0.4, yieldBelow: 6, gambitMin: 6, reprisalMin: 6, costAversion: 1.5, conflictAverse: 0.8, stealBias: 0.05, voteRate: 0.9,
  },
  sibling: {
    label: 'The Competitive Sibling', emoji: '🤼',
    blurb: 'Only cares about beating one person: targets their nemesis relentlessly.',
    bidThreshold: 2, bidScale: 0.7, reserve: 0.2, yieldBelow: 3, gambitMin: 1, reprisalMin: 0, costAversion: 0.5, nemesis: 5, stealBias: 0.8, voteRate: 0.6,
  },
  follower: {
    label: 'The Bandwagoner', emoji: '🐑',
    blurb: 'FOMO-driven: bids what the room has been bidding, whatever the gift.',
    bidThreshold: 1, bidScale: 0.4, herd: 0.8, reserve: 0.2, yieldBelow: 4, gambitMin: 3, reprisalMin: 2.5, stealBias: 0.4, voteRate: 0.8,
  },
  highroller: {
    label: 'The High Roller', emoji: '🎰',
    blurb: 'Casino regular: all-in or nothing, loves a Gambit, treats chips as confetti.',
    bidThreshold: 2.5, allIn: true, reserve: 0, yieldBelow: -99, gambitMin: -1, reprisalMin: -1, costAversion: 0, stealBias: 1, voteRate: 0.7,
  },
  sentimental: {
    label: 'The Sentimentalist', emoji: '🥹',
    blurb: 'Falls in love with whatever they hold: always defends it, rarely chases others.',
    bidThreshold: 3.5, bidScale: 0.5, attachment: 3, yieldBelow: -99, gambitMin: 4, reprisalMin: 3, costAversion: 1.2, stealBias: 0.2,
  },
  shark: {
    label: 'The Endgame Shark', emoji: '🦈',
    blurb: 'Sits quietly all game, then spends everything stealing in the Misfit Lottery.',
    endgameSaver: true, bidThreshold: 3, bidScale: 0.8, yieldBelow: 5, gambitMin: 99, reprisalMin: 4, costAversion: 0.3, stealBias: 1,
  },
  contrarian: {
    label: 'The Kingmaker', emoji: '👑',
    blurb: 'Contrarian: whoever is winning becomes the target.',
    bidThreshold: 2, bidScale: 0.6, yieldBelow: 3, gambitMin: 1.5, reprisalMin: 1, costAversion: 0.6, leaderHunter: 3, stealBias: 0.7,
  },
  newbie: {
    label: 'The First-Timer', emoji: '🐣',
    blurb: 'Nervous newcomer: tiny bids, misses half the options, mostly keeps what they get.',
    bidThreshold: 2.5, bidScale: 0.3, noise: 1.0, reserve: 0.5, yieldBelow: 2, gambitMin: 4, reprisalMin: 4, costAversion: 2, conflictAverse: 0.5, stealBias: 0.2, voteRate: 0.4,
  },
};
for (const p of Object.values(PERSONAS)) for (const [k, v] of Object.entries(DEFAULTS)) if (!(k in p)) p[k] = v;

// --- shared helpers over a view --------------------------------------------------

/** What I hold now is worth this much to me (giftless => expected value). */
function myValue(view, p) {
  const g = view.me.giftId && view.gifts[view.me.giftId];
  return g ? g.value + (p?.attachment || 0) : view.expectedValue;
}

/** Chips I'm willing to spend right now, after holding back my reserve. */
function spendable(view, p) {
  if (p.endgameSaver && !view.endgame && view.progress < 0.8) return 0;
  const reserve = Math.round(view.startingChips * p.reserve * (1 - view.progress));
  return Math.max(0, view.me.chips - reserve);
}

/** Sealed bid: scale desire into chips, nudged toward the room, within max bid and reserve. */
function bidFor(view, p, desire) {
  if (desire < p.bidThreshold) return 0;
  const cap = Math.min(view.maxBid, spendable(view, p), view.me.chips);
  if (p.allIn) return cap;
  let raw = p.bidScale * desire + (view.random() - 0.5) * 2 * p.noise;
  if (p.herd && view.recentTopBids.length) {
    const room = view.recentTopBids.reduce((a, b) => a + b, 0) / view.recentTopBids.length;
    raw = raw * (1 - p.herd) + room * p.herd;
  }
  return clamp(Math.round(raw), 0, cap);
}

/** Bonus value for hitting the gift's owner: grudge, nemesis, or the current leader. */
function targetBonus(view, p, ownerId) {
  if (!ownerId || ownerId === 'misfit') return 0;
  let bonus = 0;
  if (p.spite && (view.rivals[ownerId] || 0) > 0) bonus += p.spite;
  if (p.nemesis && ownerId === view.nemesis) bonus += p.nemesis;
  if (p.leaderHunter && ownerId === view.leaderId) bonus += p.leaderHunter;
  return bonus;
}

/** Best affordable target in [giftId, gift] pairs by a score (default: gain − cost aversion). */
function bestTarget(view, targets, p, score) {
  let best = null;
  for (const [id, g] of targets) {
    const cost = view.minCost(g.naughtyLevel || 0);
    if (cost > view.me.chips) continue;
    const gain = g.value - myValue(view, p);
    const s = score(id, g, gain, cost);
    if (!best || s > best.s) best = { id, g, gain, cost, s };
  }
  return best;
}

function pickRandom(view, list) {
  return list[Math.floor(view.random() * list.length)];
}

/** Decisions. Every function returns { ...decision, thought }. */
function decide(personaKey) {
  const p = PERSONAS[personaKey];
  if (!p) throw new Error(`Unknown persona "${personaKey}". Known: ${Object.keys(PERSONAS).join(', ')}`);
  const attackScore = (view) => (id, g, gain, cost) => gain + targetBonus(view, p, g.ownerId) - p.costAversion * cost;
  const chickensOut = (view) => p.conflictAverse && view.random() < p.conflictAverse;
  const affordable = (view, targets) => targets.filter(([, g]) => view.minCost(g.naughtyLevel || 0) <= view.me.chips);

  return {
    persona: p,

    /** Sealed bid on the gift being contested (main phase or a Misfit auction). */
    bid(view, giftValue) {
      const desire = giftValue - myValue(view, p);
      if (p.random) {
        const cap = Math.min(view.maxBid, view.me.chips);
        const amount = view.random() < 0.45 ? 0 : 1 + Math.floor(view.random() * cap);
        return { amount: clamp(amount, 0, cap), thought: amount ? `feeling lucky — ${amount}!` : 'nah, pass' };
      }
      const amount = bidFor(view, p, desire);
      if (!amount && p.endgameSaver && desire >= p.bidThreshold && !view.endgame) {
        return { amount, thought: `nice (gain ${desire.toFixed(1)}), but I'm saving for the endgame` };
      }
      const why = desire < p.bidThreshold ? `not worth it (gain ${desire.toFixed(1)})` : `want it (gain ${desire.toFixed(1)})`;
      return { amount, thought: amount ? `${why} → bid ${amount}` : `${why} → pass` };
    },

    /** Opener, nobody bid: keep, or Grinch's Gambit on another player's gift. */
    keepOrGambit(view, targets) {
      if (p.random) {
        const t = affordable(view, targets);
        if (t.length && view.random() < 0.4) return { action: 'gambit', giftId: pickRandom(view, t)[0], thought: 'chaos Gambit!' };
        return { action: 'keep', thought: 'keeping it, whatever it is' };
      }
      const best = bestTarget(view, targets, p, attackScore(view));
      if (best && best.s > p.gambitMin) {
        if (chickensOut(view)) return { action: 'keep', thought: `could Gambit "${best.g.name}"… but I don't want to upset anyone` };
        return { action: 'gambit', giftId: best.id, thought: `Gambit for "${best.g.name}" (gain ${best.gain.toFixed(1)}, cost ${best.cost})` };
      }
      return { action: 'keep', thought: best ? `no Gambit worth it (best ${best.s.toFixed(1)})` : 'nothing affordable to Gambit' };
    },

    /** Opener facing 2+ bidders (or a tie): fight, or yield and go back in the bag. */
    fightOrYield(view, giftValue) {
      if (p.random) return view.random() < 0.5
        ? { action: 'fight', thought: 'fight! why not' } : { action: 'yield', thought: 'eh, let them fight' };
      const held = giftValue + p.attachment;
      if (held < p.yieldBelow) return { action: 'yield', thought: `meh gift (${giftValue.toFixed(1)}) — yield and redraw` };
      return { action: 'fight', thought: p.attachment ? `I love my ${giftValue.toFixed(1)} gift — not giving it up` : `defending my ${giftValue.toFixed(1)} gift` };
    },

    /** After losing: Reindeer Reprisal target, or skip. */
    reprisal(view, targets) {
      if (p.random) {
        const t = affordable(view, targets);
        if (t.length && view.random() < 0.6) return { giftId: pickRandom(view, t)[0], thought: 'revenge on… that one!' };
        return { giftId: null, thought: 'skip' };
      }
      const best = bestTarget(view, targets, p, attackScore(view));
      if (best && best.s > p.reprisalMin) {
        if (chickensOut(view)) return { giftId: null, thought: 'I could strike back… nah, let it go' };
        return { giftId: best.id, thought: `Reprisal on "${best.g.name}" (gain ${best.gain.toFixed(1)}, cost ${best.cost})` };
      }
      return { giftId: null, thought: best ? `Reprisal not worth it (${best.s.toFixed(1)})` : "can't afford a Reprisal" };
    },

    /** Endgame: Path A (steal a player's gift) or Path B (auction the Misfit). */
    endgamePath(view, stealTargets, misfitValue) {
      const best = bestTarget(view, stealTargets, p, (id, g, gain, cost) => g.value + targetBonus(view, p, g.ownerId) - p.costAversion * cost);
      if (!best) return { path: 'auction', thought: 'nothing affordable to steal → auction' };
      const appeal = (best.s - misfitValue) + (p.stealBias - 0.5) * 6 + (p.random ? (view.random() - 0.5) * 10 : 0);
      if (appeal > 0 && !chickensOut(view)) {
        return { path: 'steal', giftId: best.id, thought: `steal "${best.g.name}" (${best.g.value.toFixed(1)}) over the Misfit (${misfitValue.toFixed(1)})` };
      }
      return { path: 'auction', thought: `Misfit (${misfitValue.toFixed(1)}) is fine → auction` };
    },

    /** Endgame no-bids duel: pick a defender (loser pays them the 1-chip toll). */
    noBidsDefender(view, candidates) {
      if (!candidates.length) return { playerId: null, thought: 'no one to duel' };
      const scored = candidates.map((id) => [id, targetBonus(view, p, view.players[id]?.giftId ? id : null) + view.random()]);
      scored.sort((a, b) => b[1] - a[1]);
      const id = scored[0][0];
      return { playerId: id, thought: `duel ${view.players[id]?.name}` };
    },

    /** Spectator vote on someone else's duel (flavour — exercises the UI). */
    vote(view) {
      if (view.random() > p.voteRate) return null;
      return { pick: view.random() < 0.5 ? 'challenger' : 'defender' };
    },
  };
}

module.exports = { PERSONAS, DEFAULTS, decide, rng, clamp };
