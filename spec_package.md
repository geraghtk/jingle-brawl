0) Objective

Build a configurable simulation + evaluation system for Jingle Brawl that can:

Simulate full games under the exact rules (with controlled approximations where rules leave degrees of freedom), supporting:

Monte Carlo rollouts (fast baseline)

MCTS over decision points (optional, slower, stronger)

DRL via a Gymnasium-style environment (optional, slower to train, most flexible)

Run parameter studies across:

player counts (e.g., 3–10, with a focus on 7)

economy rules (starting chips, max chip cap, Santa tax threshold/amount, dividend rules on/off, etc.)

strategy profiles (aggressive bidding vs conservative vs opportunistic; learned policies later)

Output a comprehensive report with:

balance stats (win-rate fairness, chip inequality, game length, misfit usage, duel frequency, etc.)

recommended parameter sets per player count (with confidence intervals + sensitivity notes)

1) Definitions and Formal Game Model
1.1 Entities

Player

id

chips (integer)

gift_id or None

in_bag (boolean; true if eligible to be drawn)

derived stats counters (duels won/lost, bids made, reprisal duels, etc.)

Gift

id

base_value (numeric; represents “quality” for scoring)

naughty_level (int, starts 0, increments +1 whenever involved in any duel win or lose, including sealed-bid duels, voluntary challenges, reprisals, endgame steals, and misfit auction duels unless explicitly exempted)

holder_player_id or MISFIT_PILE

Bank (North Pole Fund)

chips_in_bank (for conservation tracking; can be optional if chips are “created” only via dividend—here dividend is paid from bank, so track it)

Bag

multiset of player ids eligible for draw.

Misfit Toy Pile

stack/queue of displaced gifts. Spec: implement as a stack (LIFO) by default, configurable to FIFO.

2) Rules Implementation (Deterministic State Transitions)
2.1 Setup

N players

give each player starting_chips (V2: 10 for ≤10 players, 12 for >10 players)

All players start gift=None, therefore all are in the bag initially.

Wrapped gifts: create G gifts (configurable; default G = N, invariant: gifts = players), each with a base_value sampled from a distribution or from a provided list.

Default Configuration (Jingle Brawl V2 Rules):
```
# Core Economy
INITIAL_CHIPS: 10             # V2: "10 for ≤10 players, 12 for larger groups"
MIN_COST_MODE: LINEAR         # 1 + naughty_level (protects popular gifts)
SANTA_TAX_RULE: FLAT_1        # 1 chip tax if pot >= 3
LOSER_DIVIDEND_RULE: ANY_LOSER # Challenger OR Defender gets +1

# Challenge Rules
REPRISAL_TRIGGER_RULE: ANY_DEFENDER_LOSS  # "Grinch's Gambit or Duel"
REPRISAL_TARGETS_MISFITS: true            # "including Misfits"
REPRISAL_MAX_DEPTH: 2                     # V2: "at most one other Reprisal (2 max)"

# Yield Duel Pot (V2 change)
# YIELD_POT: HALF_ROUNDED_DOWN  # V2: Both challengers pay HALF their bids, rounded DOWN

# Gift Values
GIFT_VALUE_MODEL: CORRELATED   # Base quality + personal taste
TASTE_VARIANCE: 2.0            # Personal taste ±2 around base quality
SEALED_BID_MIN: 1              # Minimum bid (unwrapped gifts only)
SEALED_BID_MAX: null           # No cap

# Reprisal Restriction (V2)
# Reprisal cannot target the gift involved in the duel just lost
```

2.2 Phase 1: Main Game (while wrapped gifts remain)

Step A — Unwrap

Head Elf draws a name from the bag: that player is opener.

Opener opens one wrapped gift: gift becomes “contested gift” with naughty_level=0.

Step B — Sealed Bids

Everyone except the opener may submit a sealed bid b_i where 0 <= b_i <= chips_i.

If no bids, opener may:

Keep the gift (free), OR

Initiate a Voluntary Challenge (see 2.4) by paying min cost to bank.

**Note:** If opener already had a gift and keeps the new one, they must choose which gift to displace to the misfit pile (same choice logic as Step F).

Step C — Determine Challenger

If bids exist, highest bidder is challenger_1.

Tie: run tiebreak duel among tied high bidders:

spec: pairwise bracket or random pairing until one winner remains

no loser dividend for tiebreak duels (per rules)

no pot created in tiebreak unless you explicitly define one (default: no pot, it’s just a selection duel)

Step D — Yield Option

If there is a second-highest bidder (challenger_2), the opener may Yield.

If yield:

duel becomes challenger_1 vs challenger_2

Both challengers pay half their bids, rounded up

If no yield:

duel is challenger_1 vs opener

only challenger_1 pays bid into pot

Step E — Duel Resolution

Duel outcome is determined by the defender’s duel method (see §3).

Update naughty_level += 1 on the contested gift (always; duel involved).

Pot + Santa Tax (only sealed bid duels create pot):

challenger_1 pays bid into pot

if pot >= santa_tax_threshold (default 3), then santa_tax_amount (default 1) goes to bank, remainder to winner

Loser Dividend:

loser gets +1 chip from bank

exceptions: not for misfit duels, yield duels, or tie-break duels (your rules mention yield + tiebreak explicitly; and “no dividend for misfit challenges in either main or end game”)

IMPORTANT: define “yield duel” dividend exception exactly:

default: no dividend for the loser of a yield duel

configurable toggle because this affects balance a lot

Step F — Swap Algorithm

Winner takes the contested gift.

If winner already had a gift, that old gift transfers to loser.

**Misfit Creation with Loser's Choice:** If loser already had a gift (and just received winner's old gift), the loser must displace ONE gift to the misfit pile. The loser CHOOSES which gift to keep:
- Option A: Keep their original gift → Winner's old gift goes to misfit pile
- Option B: Keep the winner's old gift → Their original gift goes to misfit pile

The player whose gift was sent to misfit pile becomes Head Elf.

**Agent Decision Logic for Misfit Choice:**
- Compare personal value of each gift
- Factor in naughty levels (higher naughty = more expensive to reclaim from misfit)
- Score = value - (reclaim_cost × risk_aversion × 0.3)
- Tie-breaker: prefer keeping lower naughty (fresher) gift

Step G — Bag Maintenance

“When someone wins a gift by challenging their name gets removed from the draw bag.”

spec: interpret as: if a player wins a sealed-bid duel or voluntary challenge as the active challenger, remove them from bag (they are no longer giftless / no longer eligible to be drawn).

Any player who ends up with zero gifts is added back to bag.

2.3 Reindeer Reprisal

Trigger: “When the challenging player wins a Voluntary Challenge or a Duel, the loser gets one immediate Reprisal Duel.”

Reprisal is initiated by the loser (call them repriser).

repriser may challenge any gift holder, including misfit pile gifts if any exist.

Cost: pays standard Minimum Cost based on target gift’s naughty level to bank.

Reprisal duel resolves via defender-chosen duel method.

Reprisal cannot trigger another reprisal (depth=1).

Minimum Cost function:

IMPORTANT: The Minimum Cost mechanic serves as "gift protection" - similar to traditional 
white elephant's "3 steal max" rule. High-naughty gifts become progressively more expensive
to challenge, naturally deterring endless stealing of popular items.

Default (REQUIRED for rules compliance):

min_cost = 1 + naughty_level (LINEAR mode)

Examples:
- Naughty 0 → Cost 1 chip (cheap, accessible)
- Naughty 1 → Cost 2 chips
- Naughty 2 → Cost 3 chips  
- Naughty 3 → Cost 4 chips (expensive, protected)
- Naughty 5 → Cost 6 chips (very protected)

Alternative modes (for experimentation only):

- STEEP: min_cost = 2 + naughty_level (faster escalation)
- EXPONENTIAL: min_cost = 2^naughty_level (highly punitive)
- FLAT: min_cost = 1 always (BYPASSES protection mechanic - not recommended)

CRITICAL: Agent bidding logic MUST respect cost-benefit analysis:
- Agents should evaluate: "Is the VALUE GAIN worth the COST?"
- Value-per-chip threshold: ~1.5-2.0 (configurable by aggression)
- High-naughty gifts (3+) should have LOW bid probability due to poor cost-benefit ratio
- This ensures the protection mechanic works as intended

2.4 Voluntary Challenge

If no one bids on newly opened gift:

Opener is safe and may:

keep gift, OR

pay min_cost(target_gift_naughty_level) to bank and challenge a target gift holder.

Voluntary challenge creates no pot.

If voluntary challenge is won by challenger:

apply swap algorithm with the challenged gift being "contested"

then trigger reprisal for the loser (as per reprisal rules)

**Agent Decision Logic for Voluntary Challenges:**

The opener's motivation to challenge is driven by **dissatisfaction with their current gift**:

1. **Dissatisfaction Score** = 10 - (personal value of current gift)
   - Gift worth 2 to opener → dissatisfaction = 8 (very unhappy)
   - Gift worth 8 to opener → dissatisfaction = 2 (satisfied)

2. **Motivation** = base_aggression + (dissatisfaction / 20)
   - Unhappy openers get +0.0 to +0.5 motivation boost
   - Even low-aggression agents will challenge if they hate their gift

3. **Challenge Threshold** is lowered for desperate openers:
   - Normal threshold: score > 3
   - Desperate (dissatisfaction=10): score > 1

4. **Target Selection**: Score = (target_value - current_value) - cost × 0.5
   - Prioritizes high-value targets that are affordable
   - Factors in the gain (how much better is the target?)

**Expected Behavior:**
- Openers with low-value gifts (avg ~3.0) → ~39% initiate voluntary challenge
- Openers with high-value gifts (avg ~7.0) → rarely challenge (satisfied)
- This creates meaningful strategic decisions when nobody wants the opened gift

**Event Logging Note:** A `NO_BIDS` event is emitted when a gift receives zero sealed bids (helps diagnose voluntary funnel).

2.5 Phase 2: Misfit Lottery (Endgame)

Starts when last wrapped gift is opened but names still remain in bag.

Players drawn from bag are “active players” and are currently giftless.

Setup each turn

Head Elf draws top misfit from pile => target_misfit

Active player chooses Path A or Path B:

Path A — Steal (Attack a Player)

Active player challenges any current gift holder.

Cost: pay min_cost(naughty_level(target_gift)) to bank.

Duel:

if active player wins: takes that gift; loser must take target_misfit

if active player loses: active player must take target_misfit

Dividend exception: “No dividend for misfit toy challenges in either main or endgame”

spec: interpret as: any duel where target_misfit is assigned as consequence is treated as a “misfit challenge” and dividend off.

Path B — Auction (Bid on the Misfit)

Bidding opens to everyone (including others in bag).

Scenario 1: No bids

Active player vs Head Elf

If AP wins: gets misfit free

If AP loses: gets misfit and pays 1 chip (“Misfit Toll”) to Head Elf (not bank)

Scenario 2: Bidders exist

Active player vs highest bidder

If AP wins: takes misfit; bidder pays pot (sealed bid pot rules apply? Your text says “Bidder pays Pot.”)

spec default: bidder pays their bid into a pot; apply Santa tax if threshold met; pot awarded to duel winner (AP if AP wins)

If bidder wins:

bidder takes misfit

active player gets bidder’s old gift if bidder had one

if bidder is in bag (giftless), active player gets nothing and must restart turn with next misfit

Dividend rules:

treat auction duels as “misfit duels” => dividend off by default

configurable toggle if you want to explore enabling it

3) Duel System Abstraction (Multimodal)

You said “Define your duel methods” and “Defender decides the type.” That’s a huge design space, so the engine must support pluggable duel models:

3.1 Duel Interface

resolve_duel(state, attacker_id, defender_id, duel_context) -> winner_id

duel_context includes:

duel type: sealed_bid / voluntary / reprisal / endgame_steal / endgame_auction / tiebreak / yield

contested gift id(s), naughty levels, pot size, etc.

3.2 Built-in Duel Models

Implemented models (configurable via DUEL_MODEL):

FAIR_COIN (default): 50/50 random
- Standard for official rules
- Ensures fairness regardless of player state

SKILL_WEIGHTED: Player skill affects win probability
- Each player has latent skill ~ Normal(0, 1)
- Win prob = logistic(skill_attacker - skill_defender + noise)
- Creates skill-based progression

CHIPS_INFLUENCE: Chip count affects win probability
- More chips = higher win probability
- Win prob ≈ chips_attacker / (chips_attacker + chips_defender)
- Captures "economic panic"

DEFENDER_ADVANTAGE: 60/40 in defender's favor
- 40% attacker, 60% defender
- Makes gift defense more reliable

3.3 Defender Choice of Duel Type

Implement a policy hook:
defender_choose_duel_method(state, defender_id, duel_context) -> duel_method

Baseline heuristics:

“Always pick method with highest defender win-prob”

“Random among allowed methods”

“Conservative if gift value high”

4) Player Decision Policies (Agents)

You need strategies for bidding, yielding, choosing targets for voluntary/reprisal/endgame, and path selection.

4.1 Policy API

policy.act(observation) -> action

Actions include:

Sealed bid amount (0..chips)

Yield decision (bool)

Voluntary challenge decision + target selection

Reprisal target selection

Endgame path selection A/B

Endgame target selection for steal

Endgame misfit bid amount (if participating)

4.2 Baseline Heuristic Policies (must-have)

Greedy value: bid/challenge proportional to contested gift value - current gift value, bounded by chips and min cost

Conservative: bids low, avoids costly steals

Bully: bids aggressively early, reprisal targets high-value gifts

Chip-preserver: prioritizes staying above a chip threshold

Random: uniform random legal action (baseline)

4.2.1 Smart Agent Bidding Logic (CRITICAL for valid simulation)

Agents MUST implement cost-benefit analysis that respects the naughty level protection mechanic:

```
def agent_bid_decision(agent, gift):
    min_bid = min_cost(gift)  # 1 + naughty_level
    
    # Can't afford?
    if agent.chips < min_bid:
        return 0
    
    # Calculate value gain
    gain = gift.value - current_gift_value
    
    # COST-BENEFIT ANALYSIS (KEY)
    value_per_chip = gain / min_bid
    threshold = 2.0 - (agent.aggression * 0.8)  # Range: 1.2 to 2.0
    
    # OVERPRICED: Gift costs too much relative to gain
    if value_per_chip < threshold and gain > 0:
        bid_prob = 0.15 * agent.aggression  # Only aggressive players consider
    elif gain <= 0:
        bid_prob = 0.05 * agent.aggression  # Grief bid only
    else:
        # GOOD DEAL: Worth the cost
        bid_prob = 0.4 + 0.3 * agent.aggression
    
    # NAUGHTY LEVEL PROTECTION SIGNAL
    # Even if math works out, high naughty signals "back off"
    if gift.naughty_level >= 4:
        bid_prob *= 0.15  # Very strong deterrent
    elif gift.naughty_level >= 3:
        bid_prob *= 0.35  # Strong deterrent
    elif gift.naughty_level >= 2:
        bid_prob *= 0.55
    elif gift.naughty_level >= 1:
        bid_prob *= 0.75
    
    # Chip reserves, giftless desperation, risk aversion...
    ...
```

This ensures:
- Fresh gifts (naughty 0): ~75% bid rate for good deals
- Naughty 1: ~8% bid rate (threshold kicks in)
- Naughty 2: ~2% bid rate
- Naughty 3+: ~0% bid rate (protection working!)

WITHOUT this logic, agents will incorrectly recommend FLAT cost mode because they
bid blindly regardless of cost, maximizing raw action but breaking the protection mechanic.

4.3 MCTS Layer (optional but requested)

At decision points for a focal player:

Use rollouts with other players modeled by heuristics

Optimize expected final gift value (or utility) given state

Keep it configurable so you can run “MCTS vs heuristics” tournaments

4.4 DRL Environment (optional but requested)

Implement Gymnasium environment:

Observation: structured (player chips, gift values, naughty levels, bag membership, misfit pile top-k summary, whose turn, etc.)

Action space: multi-discrete + masked invalid actions

Reward: final gift value (sparse), plus shaped reward options (e.g., delta in expected gift value)

Provide training script (PPO baseline) and evaluation harness

5) Parameterization for Balance Studies
5.1 Config Schema (YAML/JSON)

Support sweeps over:

player_count

starting_chips

max_chip_cap (optional; enforce chips=min(chips, cap) after each transaction)

Santa tax:

santa_tax_threshold (default 3)

santa_tax_amount (default 1)

Loser dividend toggles:

dividend_enabled_main

dividend_enabled_yield

dividend_enabled_tiebreak

dividend_enabled_misfit

Min cost function:

min_cost_mode = LINEAR (default, REQUIRED) | STEEP | EXPONENTIAL | FLAT

CRITICAL: LINEAR mode is REQUIRED for proper naughty level protection.
FLAT mode bypasses the protection mechanic and should only be used for testing.

Misfit pile order: LIFO/FIFO

Wrapped gifts count: gift_count = N, N+k, or fixed

Gift value distribution:

uniform / normal / discrete list / custom

**Gift Value Model (NEW):**

| Model | Description | Effect |
|-------|-------------|--------|
| `INDEPENDENT` | Each player's valuation is fully random | Every gift has someone who likes it; rare "no bids" |
| `CORRELATED` | Gifts have base quality + personal taste variation | Creates universally bad/good gifts; more no-bid scenarios |

**CORRELATED Model Details:**
- Base quality: sampled from GIFT_VALUE_DIST (e.g., UNIFORM 1-10)
- Personal taste: varies ±TASTE_VARIANCE (default ±2) from base
- Low base (e.g., 2) → all players value it 0-4 → **no bids!**
- High base (e.g., 9) → all players value it 7-10 → **bidding war!**

**Impact on Voluntary Challenges:**
- INDEPENDENT: ~4% no-bid rate, ~0.13 voluntary challenges/game
- CORRELATED: ~16% universally bad gifts, ~0.30 voluntary challenges/game (+135%)

**Switching Models (CLI examples):**
- Use CORRELATED (default): `--gift-value-model CORRELATED --taste-variance 2.0`
- Switch to INDEPENDENT: `--gift-value-model INDEPENDENT`
- Override sealed-bid limits: `--sealed-bid-min 2 --sealed-bid-max 4` (applies only to unwrapped sealed bids)

Policy profiles assigned per player (e.g., all same, or mixed populations)

5.2 Experimental Plan Defaults

Player counts: 3..10

For each count, explore a grid like:

starting chips: {6, 8, 10}

santa tax threshold: {3, 4}

dividend misfit: {off, on} (even if default is off)

min_cost linear slope variants: {1+n, 2+n, 1+2n}

Run K games per config (e.g., 5k–50k depending on speed), with fixed RNG seeds for reproducibility.

6) Outputs: Metrics and “Comprehensive Result”
6.1 Primary Outcome (what “winning” means)

Define and report multiple scoring lenses:

Final gift value per player (primary)

Rank-based finish (1st/2nd/… by gift value)

Optional: if misfit gifts are “bad,” ensure distribution includes low values

6.2 Balance / Fairness Metrics

Per (player_count, parameter_set, policy_profile):

Win-rate by seat/turn order and by “Head Elf” initial role (if any advantage)

Gini coefficient / stddev of final gift value

Gini/stddev of final chip holdings over time (chip inequality)

Probability of being giftless entering endgame

Frequency of:

sealed bid duels

voluntary challenges

reprisal duels

yield usage

misfit pile creation

endgame path A vs B

Average pot size, Santa tax collected, bank chip flow

Average game length: number of unwrapped gifts + number of endgame turns

6.3 Strategy Sensitivity

For each parameter set, run multiple policy populations:

all greedy

all conservative

mixed (e.g., 2 bullies, rest conservative)

learned agent(s) if DRL enabled

Report whether balance holds across populations (robustness)

6.4 Recommendation Engine

Implement a simple optimizer:

Objective: minimize unfairness (e.g., minimize gift-value Gini and seat advantage) subject to:

game length within target range (configurable)

duel rate within target band (so it feels like “brawl”)

Output:

top 3 parameter sets per player count

tradeoff summary (e.g., “fairer but longer games”)

6.5 Reporting Artifacts (must produce)

Machine-readable: CSV/Parquet of per-game logs + per-config aggregates

Human-readable: one HTML or PDF report containing:

tables for top configs by player count

plots: distributions of final gift values, chip trajectories, duel frequencies, seat advantage bars

written recommendations + “why” (key drivers from metrics)

7) Implementation Requirements
7.1 Architecture

core/ game state + rules engine (pure functions where possible)

agents/ heuristic policies + MCTS wrapper + RL adapters

duels/ duel resolvers + defender duel-method selection policy

experiments/ sweeps + parallel runner

analysis/ aggregation + plotting + report builder

configs/ example YAMLs for:

baseline 7-player rules

sweep templates

7.2 Performance & Determinism

Must support running 50k+ games in minutes for Monte Carlo mode (Python with numpy OK; consider numba or Rust if needed, but start Python-first).

Full determinism with seed control:

rng = np.random.Generator(np.random.PCG64(seed))

include seed in every saved row

7.3 Logging

Two levels:

Event log (optional large): every action/duel/payout

Summary log (required): per game + per player outcomes

7.4 Validation / Tests

Unit tests for:

pot + Santa tax math

loser dividend exceptions (tiebreak/yield/misfit)

swap algorithm and misfit creation

bag membership rules (“giftless in bag” invariant)

Property tests:

chip conservation with bank accounting (except where configured caps destroy chips)

no negative chips ever

8) Known Ambiguities (Implement as Configurable Defaults)

The rules text leaves some degrees of freedom; implement toggles and document defaults:

Exact Minimum Cost formula vs naughty level

Whether yield duel increments naughty level (default: yes, because contested gift was in a duel)

Whether reprisal can target misfit pile when pile empty (obviously no) and how it targets misfit gift (challenge “the pile top” vs any misfit)

Whether endgame auction pot uses Santa tax (default: yes if pot exists)

Tiebreak duel bracket method

Whether players are removed from bag on any “gift acquisition” or specifically “wins by challenging” (default: remove when a player wins as challenger; add back if giftless)

9) Deliverables Checklist

Agent must deliver:

Working simulator CLI:

simulate --config configs/baseline_7p.yaml --games 10000 --out results/

sweep --config configs/sweep.yaml --parallel 8 --out results/

Report generator:

report --results results/ --out report.html

Example reports for:

7 players baseline rules

3–10 players recommended parameters per count

Documentation:

README explaining rules mapping + defaults + how to add duel models and policies

If you want, I can also provide a starter config (baseline 7-player), plus a sweep config that explores "Santa tax/no tax", chip caps, and min-cost slopes—so an agent can run it immediately.

---

10) Simulation Results & Optimal Configurations

Based on 1,728 configurations tested with corrected agent logic (50,000+ games):

10.1 Universal Settings (All Player Counts)

These settings are REQUIRED to match official Jingle Brawl rules:

```yaml
MIN_COST_MODE: LINEAR           # 1 + naughty_level (REQUIRED)
LOSER_DIVIDEND_RULE: ANY_LOSER  # Challenger OR Defender gets +1
REPRISAL_TRIGGER_RULE: ANY_DEFENDER_LOSS  # When defender loses any duel
REPRISAL_TARGETS_MISFITS: true
DUEL_MODEL: FAIR_COIN           # 50/50 random
```

10.2 Optimal Chips by Player Count

| Players | Chips | Duels | Reprisals | Duels/Player | Bankruptcy |
|---------|-------|-------|-----------|--------------|------------|
| 3       | 12    | 3.8   | 1.3       | 1.26         | 0.00%      |
| 4       | 12    | 6.0   | 1.8       | 1.49         | 0.00%      |
| 5       | 10    | 8.2   | 2.5       | 1.63         | 0.00%      |
| 6       | 10    | 10.4  | 2.9       | 1.73         | 0.00%      |
| 7       | 10    | 12.6  | 3.4       | 1.80         | 0.00%      |
| 8       | 12    | 15.2  | 3.9       | 1.89         | 0.00%      |
| 9       | 12    | 17.4  | 4.6       | 1.94         | 0.01%      |
| 10      | 12    | 19.8  | 5.0       | 1.98         | 0.00%      |

10.3 Grouped Recommendations

SMALL GROUPS (3-5 players):
- Chips: 10-12
- Santa Tax: FLAT_1 or NONE
- Key insight: Higher chips compensate for fewer interaction opportunities

MEDIUM GROUPS (6-8 players):
- Chips: 8-12 (8 is official, 10 is optimal)
- Santa Tax: FLAT_1 or NONE
- Key insight: Sweet spot for game balance

LARGE GROUPS (9-10 players):
- Chips: 12
- Santa Tax: FLAT_1
- Key insight: More chips sustain longer games

10.4 Official 7-Player Configuration

The official rules (8 chips) work well:

```yaml
# configs/baseline_7p.yaml
NUM_PLAYERS: 7
NUM_GIFTS: 7
INITIAL_CHIPS: 8               # Official rules
MIN_COST_MODE: LINEAR          # 1 + naughty_level
SANTA_TAX_RULE: FLAT_1         # 1 chip if pot >= 3
LOSER_DIVIDEND_RULE: ANY_LOSER
REPRISAL_TRIGGER_RULE: ANY_DEFENDER_LOSS
REPRISAL_TARGETS_MISFITS: true
DUEL_MODEL: FAIR_COIN
```

Expected results:
- Duels/game: 12.6
- Reprisals: 3.5
- Bankruptcy: 0.19%
- Giftless: 0.29%
- Avg Gift Value: 6.78

10.5 Key Findings

1. LINEAR vs FLAT Cost Mode:
   - With proper agent logic, LINEAR and FLAT produce EQUAL action levels (~12.6 duels)
   - LINEAR provides gift protection mechanic (high-naughty gifts left alone)
   - FLAT bypasses protection (agents bid on everything regardless of cost)
   - ALWAYS use LINEAR to match rules intent

2. Loser Dividend Impact:
   - ANY_LOSER: 0.40% bankruptcy (best)
   - DEFENDER_ONLY: 2.47% bankruptcy
   - NONE: 2.98% bankruptcy
   - ANY_LOSER is 7x better at preventing bankruptcy

3. Reprisal Rule Impact:
   - ANY_DEFENDER_LOSS: 11.1 duels, 3.1 reprisals
   - VOLUNTARY_ONLY: 9.1 duels, 0.0 reprisals
   - DISABLED: 9.1 duels, 0.0 reprisals
   - Reprisals add ~2 extra duels per game

4. Correlations:
   - Player count vs Duels: 0.976 (nearly perfect linear)
   - Chips vs Bankruptcy: -0.558 (more chips = less bankruptcy)
   - Duels vs Reprisals: 0.447 (reprisals drive action)

---

11) CLI Usage

```bash
# Run demo game
python jingle_brawl_env.py demo

# Simulate with config file
python jingle_brawl_env.py simulate --config configs/baseline_7p.yaml --games 10000

# Run parameter sweep
python jingle_brawl_env.py sweep --config configs/sweep.yaml --out results.json

# Override specific parameters
python jingle_brawl_env.py simulate --players 7 --chips 8 --seed 42 --games 1000

# Gift value model overrides
python jingle_brawl_env.py simulate --gift-value-model INDEPENDENT --games 1000
python jingle_brawl_env.py simulate --gift-value-model CORRELATED --taste-variance 2.5 --games 1000

# Sealed bid limits (unwrapped gifts only)
python jingle_brawl_env.py simulate --sealed-bid-min 2 --sealed-bid-max 4 --games 1000
```