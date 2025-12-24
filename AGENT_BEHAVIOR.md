# Agent Behavior Model

This document explains how human-like behavior is encoded into the Jingle Brawl simulation agents.

---

## Overview

Agents in the simulation are designed to mimic realistic human behavior in a white elephant gift exchange. Rather than playing "optimally," they exhibit psychological biases and social dynamics observed in real games.

---

## Human Psychology Parameters (Baked-In Defaults)

These values are fixed in the model based on behavioral research and observation of real gift exchanges:

| Parameter | Value | What It Models |
|-----------|-------|----------------|
| `ATTACHMENT_WEIGHT` | 0.5 | Endowment effect strength |
| `ATTACHMENT_GAIN` | 0.3 | How much attachment grows when receiving a gift |
| `ATTACHMENT_DECAY` | 0.1 | How much attachment fades when losing a gift |
| `REVENGE_WEIGHT` | 0.4 | Strength of grudge-based targeting |
| `REVENGE_DECAY` | 0.1 | How quickly grudges fade |
| `VISIBILITY_PENALTY` | 0.03 | Penalty for being seen as a "bully" |
| `LIQUIDITY_RESERVE` | 0.25 | Tendency to keep chips for later |
| `BID_NOISE` | 0.15 | Human imprecision in bidding |
| `BID_QUANTIZATION` | True | Humans prefer round numbers (1, 2, 3...) |

---

## Detailed Behavior Explanations

### 1. Attachment / Endowment Effect

**Real-world behavior:** People value things they own more than identical things they don't own. A gift you've held for 3 rounds "feels" more valuable than one you just got.

**How it's modeled:**
```
effective_value = base_value + (attachment_score × ATTACHMENT_WEIGHT)
```

- When you **receive** a gift: `attachment += ATTACHMENT_GAIN`
- When you **lose** a gift: `attachment -= ATTACHMENT_DECAY`
- When you **defend** successfully: `attachment += 0.1` (pride boost)

**Effect on decisions:**
- Higher attachment → less likely to yield
- Higher attachment → higher bid threshold to give it up
- Agents defend "their" gift more aggressively than pure value would suggest

---

### 2. Grudges & Revenge

**Real-world behavior:** If someone steals your gift, you want payback. Reprisals aren't random—they target the person who wronged you.

**How it's modeled:**
```python
agent.grudge[attacker_id] += 1.0  # When someone takes your gift
```

**Effect on reprisal targeting:**
```
target_score = base_value + (grudge[target] × REVENGE_WEIGHT)
```

Agents preferentially target players who previously attacked them, even if another gift is slightly more valuable.

**Decay:** Grudges fade slightly each turn (`grudge -= REVENGE_DECAY`), modeling "letting it go" over time.

---

### 3. Visibility Penalty (Anti-Bully)

**Real-world behavior:** Players avoid becoming the "table villain." If you've attacked the same person 3 times, others notice and may gang up on you.

**How it's modeled:**
```python
agent.attacks_initiated += 1  # Each time agent challenges someone
bid_prob *= (1 - attacks_initiated × VISIBILITY_PENALTY)
```

**Effect:** Agents who have attacked frequently become more hesitant to attack again. This prevents unrealistic "bully" behavior where one aggressive player dominates.

---

### 4. Liquidity Reserve

**Real-world behavior:** People don't spend all their chips early. They keep reserves for "just in case" and become more willing to spend near the end.

**How it's modeled:**
```python
# Reserve ratio based on game phase
phase = gifts_opened / total_gifts
reserve = LIQUIDITY_RESERVE × (1 - phase)  # Higher early, lower late

# Available chips for bidding
available = chips - (chips × reserve)
```

**Effect:**
- Early game: Agents keep ~25% of chips in reserve
- Late game: Agents spend more freely (reserve → 0)

---

### 5. Bid Noise & Quantization

**Real-world behavior:** Humans don't calculate exact optimal bids. They think "I'll bid 2" or "maybe 3," not "2.47 chips."

**How it's modeled:**
```python
# Add noise
bid = base_bid × (1 + random.uniform(-BID_NOISE, BID_NOISE))

# Quantize to integers (humans prefer round numbers)
if BID_QUANTIZATION:
    bid = round(bid)
```

**Effect:** Bid distributions cluster around 1, 2, 3 rather than spreading evenly. Creates more ties and more realistic bid patterns.

---

### 6. Dissatisfaction-Driven Voluntary Challenges

**Real-world behavior:** If you open a gift nobody wants (including you), you're more motivated to challenge someone else for a better gift.

**How it's modeled:**
```python
dissatisfaction = 10 - personal_value_of_current_gift
motivation = base_aggression + (dissatisfaction / 20)
```

**Effect:**
- Opener with value-2 gift: +0.4 motivation boost
- Opener with value-8 gift: +0.1 motivation boost
- Even passive players will challenge if they hate their gift

---

### 7. Misfit Choice (Loser's Decision)

**Real-world behavior:** When forced to displace a gift, people keep the one they like better, factoring in "how hard would it be to get this back?"

**How it's modeled:**
```python
score_A = value_A - (reclaim_cost_A × risk_aversion × 0.3)
score_B = value_B - (reclaim_cost_B × risk_aversion × 0.3)
keep = A if score_A >= score_B else B
```

**Effect:** Agents keep higher-value gifts, but also consider naughty level (high naughty = expensive to reclaim from misfit pile).

---

## Per-Agent Personality Variation

Each agent is initialized with random personality traits:

| Trait | Range | Effect |
|-------|-------|--------|
| `aggression` | 0.3 – 0.8 | Bid frequency, challenge likelihood |
| `fomo` | 0.2 – 0.7 | Tendency to bid on endgame misfits |
| `risk_aversion` | 0.2 – 0.6 | Chip conservation, bid sizing |
| `spite` | 0.1 – 0.4 | Likelihood of "grief" bids (bidding to deny others) |

This creates a mix of aggressive, conservative, and opportunistic players—like a real game.

---

## Summary: Where Human Behavior Lives

| Decision Point | Human Behavior Encoded |
|----------------|------------------------|
| **Sealed bidding** | Attachment boost, visibility penalty, liquidity reserve, bid noise |
| **Yield decision** | Attachment (don't yield if attached), risk aversion |
| **Voluntary challenge** | Dissatisfaction motivation, target selection by value-cost |
| **Reprisal targeting** | Grudge/revenge priority over pure value |
| **Misfit choice** | Keep higher-value gift, factor reclaim difficulty |
| **Endgame bidding** | FOMO, desperation if giftless |

---

## Validation

Simulations show agents exhibit realistic patterns:
- **Revenge targeting:** ~40% of reprisals target the previous attacker
- **Bid clustering:** 70%+ of bids are exactly 1, 2, or 3
- **Attachment effect:** Agents keep original gift 70% of time in misfit choice
- **Voluntary challenge rate:** Unhappy openers (~value 3) challenge 39% vs happy openers (~value 7) at 10%

---

*These behavioral parameters are fixed in the model. Game balance is tuned via separate parameters (INITIAL_CHIPS, SANTA_TAX_RULE, etc.).*

