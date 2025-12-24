# Self-Play Analysis: Understanding Equilibria in Jingle Brawl

## Executive Summary

Self-play training in Jingle Brawl reveals **two stable equilibria** that agents can converge to, each representing a fundamentally different strategy profile:

| Equilibrium | Duels/Game | Voluntary | Strategy |
|-------------|------------|-----------|----------|
| **Aggressive** | ~14-16 | ~0 | Everyone bids heavily |
| **Passive** | ~3-5 | ~1.5-2.5 | Nobody bids, voluntary challenges only |

This document explains why these equilibria exist, their implications for game design, and how to interpret model results.

---

## The Two Equilibria Explained

### 1. Aggressive Equilibrium (~14 duels, 0 voluntary)

**Behavior**: All agents learn to bid aggressively on every gift.

**Why It's Stable**:
- If everyone bids, not bidding means losing every gift
- Bidding becomes a "mutual assured destruction" scenario
- Any agent who stops bidding gets outcompeted

**Characteristics**:
- High chip throughput (lots of chips changing hands)
- Winner determined largely by luck/coin flips
- Low voluntary challenges (gifts contested via bids)

### 2. Passive Equilibrium (~3 duels, 2+ voluntary)

**Behavior**: All agents learn NOT to bid, relying on voluntary challenges (Grinch's Gambit).

**Why It's Stable**:
- If nobody bids, opener gets gift **free**
- Opener can then use Grinch's Gambit to upgrade strategically
- This is actually **more chip-efficient** than bidding wars
- Any single agent who starts bidding just loses chips while others get free gifts

**Characteristics**:
- Low chip usage (most chips retained)
- Voluntary challenges replace sealed bids
- Higher average gift values (less random redistribution)

---

## Why Both Equilibria Exist (Game Theory)

Jingle Brawl exhibits **multiple Nash equilibria** - stable states where no player benefits from unilaterally changing strategy.

### Nash Equilibrium Analysis

```
Scenario A: Everyone bids
  - If I stop bidding → I get fewer gifts → Worse for me
  - STABLE ✓

Scenario B: Nobody bids
  - If I start bidding → I spend chips, others get free gifts → Worse for me
  - STABLE ✓
```

The passive equilibrium is actually **Pareto-superior** (better for everyone collectively) because:
1. No chips wasted on bidding wars
2. Gifts distributed without costly competition
3. Voluntary challenges allow strategic upgrades

However, it requires **perfect coordination** - one defector ruins it.

---

## Model Comparison

| Model | Typical Result | Equilibrium Found | Why |
|-------|---------------|-------------------|-----|
| **Monte Carlo** | 13 duels, 0.4 vol | Mixed/Realistic | Random strategies don't coordinate |
| **Heuristic** | 15 duels, 0 vol | Aggressive | Hard-coded bidding behavior |
| **DRL** | 14 duels, 0 vol | Aggressive | Trained against diverse opponents |
| **Self-Play** | 3-14 duels (varies) | Either | All agents same, can converge to either |
| **Mixed Self-Play** | 12-15 duels | Aggressive | Forced diversity prevents passive collapse |

### Which Model to Trust?

| Use Case | Recommended Model |
|----------|-------------------|
| **Realistic human games** | Monte Carlo or Mixed Self-Play |
| **Stress testing economy** | Heuristic |
| **Finding theoretical optima** | Self-Play |
| **Balanced analysis** | Mixed Self-Play |

---

## The Mixed Self-Play Solution

To prevent self-play from collapsing to the passive equilibrium, we implemented **Mixed Self-Play** with:

### Key Features

1. **Diverse Population Seeding**
   - 1/3 agents: Aggressive bias (higher bid tendency)
   - 1/3 agents: Neutral (standard)
   - 1/3 agents: Defensive bias (lower bid tendency)

2. **Protected Evolution**
   - At least 2 of each personality type survive each generation
   - Prevents single strategy from dominating

3. **Diversity Bonus**
   - Fitness = gift_value + (duels × 0.3)
   - Rewards agents that create game action

### Results

Mixed Self-Play consistently produces results similar to Monte Carlo (~13 duels, ~0.4 voluntary), making it suitable for realistic game balance analysis.

---

## Implications for Game Design

### The Passive Equilibrium Problem

The existence of a passive equilibrium reveals a potential design issue:
- **If** all players coordinated to not bid, the game would be less exciting
- The Grinch's Gambit (voluntary challenge) mechanism enables this

### Why It's Not a Problem in Practice

1. **Human FOMO**: Real players can't resist bidding on good gifts
2. **Coordination Failure**: Players have different aggression levels
3. **Risk Perception**: Players overestimate loss from not bidding
4. **Social Dynamics**: Bidding is more "fun" than passing

### Design Insight

The voluntary challenge mechanism is crucial:
- Provides a strategic option for unopposed openers
- Creates interesting decisions after "no bids"
- Without it, "no bids" would just end the turn

---

## Experimental Findings

### Breaking the Passive Equilibrium

| Experiment | Duels | Result |
|------------|-------|--------|
| Baseline | 3-14 | Varies by run |
| High Mutation (0.4) | ~16 | Aggressive |
| Aggression Bias | ~5 | Still Passive! |
| Diversity Bonus | ~15 | Aggressive |
| Combined | ~16 | Aggressive |

**Key Finding**: Aggression bias alone **increases passive behavior** because it encourages voluntary challenges over bidding!

---

## Recommendations

### For Game Balance Analysis

Use **Monte Carlo** or **Mixed Self-Play** for realistic results that match human play patterns.

### For Finding Exploits

Use standard **Self-Play** to discover equilibria and potential dominant strategies.

### For Stress Testing

Use **Heuristic** with aggressive agents to test economic limits.

### Configuration

```yaml
# Recommended for balanced analysis
run_monte_carlo: true
run_mixed_self_play: true
run_heuristic: true
run_self_play: false  # Skip unless finding exploits
```

---

## Technical Implementation

### MixedSelfPlayAnalyzer

```python
class MixedSelfPlayAnalyzer:
    """
    Self-play with mixed population to avoid passive equilibrium.
    
    Maintains population diversity by:
    1. Seeding agents with aggression bias
    2. Rewarding action (diversity bonus)
    3. Protecting aggressive agents from extinction
    """
```

### Key Parameters

- `pop_size`: 20 (diverse population)
- `aggressive_ratio`: 1/3 of population
- `diversity_bonus`: 0.3 × duels
- `protected_slots`: 2 per personality type

---

## Conclusion

The dual equilibria in Jingle Brawl self-play are a fascinating example of game-theoretic dynamics. While the passive equilibrium is theoretically optimal for coordinated agents, human games naturally gravitate toward the aggressive equilibrium due to competitive instincts and coordination failures.

For simulation purposes:
- **Mixed Self-Play** best approximates real human games
- **Standard Self-Play** reveals theoretical equilibria
- **Monte Carlo** provides baseline realistic behavior

The game design is sound - the passive equilibrium exists but is unstable in practice among diverse players.

---

*Generated by Jingle Brawl Analysis System v2.0*

