"""
Jingle Brawl Optimization Environment
======================================

Multi-agent simulation with configurable hyperparameters for balance testing.
Includes Deep RL self-play training.

RULES COMPLIANCE (per official spec):
-------------------------------------
- Loser's Dividend: ANY loser (Challenger OR Defender) gets +1 chip
- Exception: NO dividend for misfit duels, yield duels, or tie-break duels
- Santa Tax: If pot >= 3, tax is 1 chip
- Sealed bids create pots; voluntary/reprisal costs go directly to bank
- Naughty Level: +1 per duel involvement, min_cost = naughty_level + 1 (no cap)
- Reprisal: After winning Voluntary Challenge OR Duel, loser gets reprisal
- Endgame: Path A (Steal) or Path B (Auction) with sniper/swapper logic

Configuration Parameters:
- INITIAL_CHIPS (default: 8)
- SANTA_TAX_RULE: FLAT_1, CAP_2, PROGRESSIVE, NONE
- LOSER_DIVIDEND_RULE: DEFENDER_ONLY, ANY_LOSER, NONE
- MIN_COST_MODE: LINEAR, STEEP, EXPONENTIAL
- DUEL_MODEL: FAIR_COIN, SKILL_WEIGHTED, CHIPS_INFLUENCE
- REPRISAL_TARGETS_MISFITS: boolean
- NUM_PLAYERS / NUM_GIFTS
- SEED: For reproducibility
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any, Callable
from collections import defaultdict
import numpy as np
import random
import json
import time
import os
from abc import ABC, abstractmethod

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Optional pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# PyTorch for Deep RL
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class SantaTaxRule(Enum):
    FLAT_1 = "FLAT_1"          # If pot >= 3, tax is 1 (SPEC DEFAULT)
    CAP_2 = "CAP_2"            # Winner takes max 2, rest is tax
    PROGRESSIVE = "PROGRESSIVE" # Tax = pot // 3
    NONE = "NONE"              # No tax


class LoserDividendRule(Enum):
    DEFENDER_ONLY = "DEFENDER_ONLY"  # Only defender gets +1 (SPEC DEFAULT)
    ANY_LOSER = "ANY_LOSER"          # Any loser gets +1
    NONE = "NONE"                     # No dividend


class EndgamePotRule(Enum):
    WINNER_TAKES = "WINNER_TAKES"
    BANK_TAKES = "BANK_TAKES"


class YieldRule(Enum):
    ALLOWED = "ALLOWED"      # Opener can yield if C2 exists
    DISABLED = "DISABLED"    # No yielding


class VoluntaryChallengeRule(Enum):
    ENABLED = "ENABLED"      # Opener can challenge if no bids (SPEC DEFAULT)
    DISABLED = "DISABLED"    # No voluntary challenges


class ReprisalTriggerRule(Enum):
    VOLUNTARY_ONLY = "VOLUNTARY_ONLY"  # Only after voluntary challenge win (SPEC DEFAULT)
    ANY_DEFENDER_LOSS = "ANY_DEFENDER_LOSS"  # Any defender who loses gets reprisal
    ANY_LOSS = "ANY_LOSS"  # Any loser (challenger or defender) gets reprisal
    DISABLED = "DISABLED"  # No reprisals


class MinCostMode(Enum):
    """How minimum challenge cost scales with naughty level."""
    LINEAR = "LINEAR"          # cost = 1 + naughty_level (default)
    STEEP = "STEEP"            # cost = 2 + naughty_level
    EXPONENTIAL = "EXPONENTIAL" # cost = 2^naughty_level (very punitive)
    FLAT = "FLAT"              # cost = 1 always (no scaling)


class DuelModel(Enum):
    """How duel outcomes are determined."""
    FAIR_COIN = "FAIR_COIN"           # 50/50 random (default)
    SKILL_WEIGHTED = "SKILL_WEIGHTED"  # Based on player skill attributes
    CHIPS_INFLUENCE = "CHIPS_INFLUENCE" # Chip count affects win probability
    DEFENDER_ADVANTAGE = "DEFENDER_ADVANTAGE"  # 60/40 in defender's favor


class GiftValueDistribution(Enum):
    """How gift values are assigned."""
    UNIFORM = "UNIFORM"        # Random 1-10 (default)
    NORMAL = "NORMAL"          # Normal(5.5, 2)
    FIXED_LIST = "FIXED_LIST"  # Use provided list
    TIERED = "TIERED"          # 30% low, 40% mid, 30% high


class GiftValueModel(Enum):
    """How player valuations of gifts are determined."""
    INDEPENDENT = "INDEPENDENT"  # Each player's valuation is fully random (old behavior)
    CORRELATED = "CORRELATED"    # Gifts have base quality + personal taste variation


@dataclass
class GameConfig:
    """
    Configuration parameters for the simulation.
    
    Defaults match the official Jingle Brawl V2 ruleset:
    - 10 starting chips (≤10 players), 12 chips (>10 players)
    - Santa Tax: 1 chip if pot >= 3
    - Loser Dividend: ANY loser (Challenger OR Defender) gets +1
    - Reprisal: When Grinch's Gambit OR Duel concludes, loser gets reprisal (max depth 2)
    - Dynamic bag: winners removed, giftless added back
    - Min cost: naughty_level + 1 (no cap)
    """
    # Core rules
    INITIAL_CHIPS: int = 10                                                   # V2: "10 chips for ≤10 players, 12 for larger"
    SANTA_TAX_RULE: SantaTaxRule = SantaTaxRule.FLAT_1                        # "If pot >= 3, one chip tax"
    LOSER_DIVIDEND_RULE: LoserDividendRule = LoserDividendRule.ANY_LOSER      # "Challenger OR Defender gets +1"
    ENDGAME_POT_RULE: EndgamePotRule = EndgamePotRule.WINNER_TAKES
    YIELD_RULE: YieldRule = YieldRule.ALLOWED                                 # "Opener may Yield"
    VOLUNTARY_CHALLENGE_RULE: VoluntaryChallengeRule = VoluntaryChallengeRule.ENABLED
    REPRISAL_TRIGGER_RULE: ReprisalTriggerRule = ReprisalTriggerRule.ANY_DEFENDER_LOSS  # "Voluntary Challenge or Duel"
    REPRISAL_TARGETS_MISFITS: bool = True                                     # "including those in the misfit pile"
    REPRISAL_MAX_DEPTH: int = 2                                               # V2: "at most one other Reprisal (2 max)"
    NUM_PLAYERS: int = 7
    NUM_GIFTS: int = 7    # Invariant: gifts = players
    
    # New configurable options
    MIN_COST_MODE: MinCostMode = MinCostMode.LINEAR                           # cost = 1 + naughty_level
    NAUGHTY_LEVEL_CAP: int = 99                                               # Effectively unlimited
    DUEL_MODEL: DuelModel = DuelModel.FAIR_COIN                               # 50/50 by default
    GIFT_VALUE_DIST: GiftValueDistribution = GiftValueDistribution.UNIFORM    # Random 1-10
    GIFT_VALUE_MODEL: GiftValueModel = GiftValueModel.CORRELATED              # Base quality + personal taste
    TASTE_VARIANCE: float = 2.0                                               # How much personal taste varies (±)
    
    # Human-behavior tuning
    # Human-behavior tuning (baked defaults to mimic real play)
    ATTACHMENT_WEIGHT: float = 0.6                                            # Endowment effect strength
    ATTACHMENT_GAIN: float = 0.5                                              # Attachment gained when keeping/receiving
    ATTACHMENT_DECAY: float = 0.1                                             # Attachment decay on loss/transfer
    REVENGE_WEIGHT: float = 0.6                                               # Grudge weight in target selection
    REVENGE_DECAY: float = 0.1                                                # Grudge decay per turn (not yet applied; hook ready)
    VISIBILITY_PENALTY: float = 0.02                                          # Anti-bully dampening on repeated attacks
    LIQUIDITY_RESERVE: float = 0.30                                           # Fraction of chips kept early
    BID_NOISE: float = 0.12                                                   # Human bid noise
    BID_QUANTIZATION: bool = True                                             # Snap bids to simple ladders
    
    # Sealed bid limits (only for unwrapped gift bidding, not voluntary/reprisal)
    SEALED_BID_MIN: int = 1                                                   # Minimum bid to participate
    SEALED_BID_MAX: Optional[int] = None                                      # Max bid (None = no limit)
    
    # Reproducibility
    SEED: Optional[int] = None                                                # None = random seed
    
    @staticmethod
    def recommended_chips(num_players: int) -> int:
        """V2 SPEC: 10 chips for ≤10 players, 12 chips for >10 players."""
        return 10 if num_players <= 10 else 12
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GameConfig':
        # Auto-scale chips if not specified or set to "AUTO"
        num_players = d.get('NUM_PLAYERS', 7)
        initial_chips = d.get('INITIAL_CHIPS')
        if initial_chips is None or initial_chips == "AUTO":
            initial_chips = cls.recommended_chips(num_players)
        
        return cls(
            INITIAL_CHIPS=initial_chips,
            SANTA_TAX_RULE=SantaTaxRule(d.get('SANTA_TAX_RULE', 'FLAT_1')),
            LOSER_DIVIDEND_RULE=LoserDividendRule(d.get('LOSER_DIVIDEND_RULE', 'ANY_LOSER')),
            ENDGAME_POT_RULE=EndgamePotRule(d.get('ENDGAME_POT_RULE', 'WINNER_TAKES')),
            YIELD_RULE=YieldRule(d.get('YIELD_RULE', 'ALLOWED')),
            VOLUNTARY_CHALLENGE_RULE=VoluntaryChallengeRule(d.get('VOLUNTARY_CHALLENGE_RULE', 'ENABLED')),
            REPRISAL_TRIGGER_RULE=ReprisalTriggerRule(d.get('REPRISAL_TRIGGER_RULE', 'ANY_DEFENDER_LOSS')),
            REPRISAL_TARGETS_MISFITS=d.get('REPRISAL_TARGETS_MISFITS', True),
            REPRISAL_MAX_DEPTH=d.get('REPRISAL_MAX_DEPTH', 1),
            NUM_PLAYERS=d.get('NUM_PLAYERS', 7),
            NUM_GIFTS=d.get('NUM_GIFTS', 7),
            MIN_COST_MODE=MinCostMode(d.get('MIN_COST_MODE', 'LINEAR')),
            NAUGHTY_LEVEL_CAP=d.get('NAUGHTY_LEVEL_CAP', 99),
            DUEL_MODEL=DuelModel(d.get('DUEL_MODEL', 'FAIR_COIN')),
            GIFT_VALUE_DIST=GiftValueDistribution(d.get('GIFT_VALUE_DIST', 'UNIFORM')),
            GIFT_VALUE_MODEL=GiftValueModel(d.get('GIFT_VALUE_MODEL', 'CORRELATED')),
            TASTE_VARIANCE=d.get('TASTE_VARIANCE', 2.0),
            ATTACHMENT_WEIGHT=d.get('ATTACHMENT_WEIGHT', 0.6),
            ATTACHMENT_GAIN=d.get('ATTACHMENT_GAIN', 0.5),
            ATTACHMENT_DECAY=d.get('ATTACHMENT_DECAY', 0.1),
            REVENGE_WEIGHT=d.get('REVENGE_WEIGHT', 0.5),
            REVENGE_DECAY=d.get('REVENGE_DECAY', 0.1),
            VISIBILITY_PENALTY=d.get('VISIBILITY_PENALTY', 0.05),
            LIQUIDITY_RESERVE=d.get('LIQUIDITY_RESERVE', 0.30),
            BID_NOISE=d.get('BID_NOISE', 0.15),
            BID_QUANTIZATION=d.get('BID_QUANTIZATION', True),
            SEALED_BID_MIN=d.get('SEALED_BID_MIN', 1),
            SEALED_BID_MAX=d.get('SEALED_BID_MAX', None),
            SEED=d.get('SEED', None),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> 'GameConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    def to_dict(self) -> dict:
        return {
            'INITIAL_CHIPS': self.INITIAL_CHIPS,
            'SANTA_TAX_RULE': self.SANTA_TAX_RULE.value,
            'LOSER_DIVIDEND_RULE': self.LOSER_DIVIDEND_RULE.value,
            'ENDGAME_POT_RULE': self.ENDGAME_POT_RULE.value,
            'YIELD_RULE': self.YIELD_RULE.value,
            'VOLUNTARY_CHALLENGE_RULE': self.VOLUNTARY_CHALLENGE_RULE.value,
            'REPRISAL_TRIGGER_RULE': self.REPRISAL_TRIGGER_RULE.value,
            'REPRISAL_TARGETS_MISFITS': self.REPRISAL_TARGETS_MISFITS,
            'REPRISAL_MAX_DEPTH': self.REPRISAL_MAX_DEPTH,
            'NUM_PLAYERS': self.NUM_PLAYERS,
            'NUM_GIFTS': self.NUM_GIFTS,
            'MIN_COST_MODE': self.MIN_COST_MODE.value,
            'NAUGHTY_LEVEL_CAP': self.NAUGHTY_LEVEL_CAP,
            'DUEL_MODEL': self.DUEL_MODEL.value,
            'GIFT_VALUE_DIST': self.GIFT_VALUE_DIST.value,
            'GIFT_VALUE_MODEL': self.GIFT_VALUE_MODEL.value,
            'TASTE_VARIANCE': self.TASTE_VARIANCE,
            'ATTACHMENT_WEIGHT': self.ATTACHMENT_WEIGHT,
            'ATTACHMENT_GAIN': self.ATTACHMENT_GAIN,
            'ATTACHMENT_DECAY': self.ATTACHMENT_DECAY,
            'REVENGE_WEIGHT': self.REVENGE_WEIGHT,
            'REVENGE_DECAY': self.REVENGE_DECAY,
            'VISIBILITY_PENALTY': self.VISIBILITY_PENALTY,
            'LIQUIDITY_RESERVE': self.LIQUIDITY_RESERVE,
            'BID_NOISE': self.BID_NOISE,
            'BID_QUANTIZATION': self.BID_QUANTIZATION,
            'SEALED_BID_MIN': self.SEALED_BID_MIN,
            'SEALED_BID_MAX': self.SEALED_BID_MAX,
            'SEED': self.SEED,
        }
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def short_name(self) -> str:
        """Short identifier for this config."""
        reprisal_short = {
            ReprisalTriggerRule.VOLUNTARY_ONLY: "VOL",
            ReprisalTriggerRule.ANY_DEFENDER_LOSS: "DEF",
            ReprisalTriggerRule.ANY_LOSS: "ANY",
            ReprisalTriggerRule.DISABLED: "OFF",
        }
        return f"P{self.NUM_PLAYERS}_C{self.INITIAL_CHIPS}_{self.LOSER_DIVIDEND_RULE.value}_Rep{reprisal_short[self.REPRISAL_TRIGGER_RULE]}"


# =============================================================================
# GAME STATE
# =============================================================================

class TurnPhase(Enum):
    MAIN_GAME = auto()
    MISFIT_LOTTERY = auto()


class DuelType(Enum):
    SEALED_BID = auto()
    TIE_BREAK = auto()
    VOLUNTARY = auto()
    REPRISAL = auto()
    YIELD_DUEL = auto()
    MISFIT_CHALLENGE = auto()
    ENDGAME_STEAL = auto()
    ENDGAME_AUCTION = auto()
    HEAD_ELF_DEFENSE = auto()


# Duels that don't grant dividend (per spec)
NO_DIVIDEND_DUELS = {
    DuelType.TIE_BREAK,
    DuelType.YIELD_DUEL,
    DuelType.MISFIT_CHALLENGE,
    DuelType.HEAD_ELF_DEFENSE,
}


@dataclass
class Gift:
    gift_id: int
    naughty_level: int = 0
    owner_id: Optional[int] = None
    value_scores: Dict[int, float] = field(default_factory=dict)
    is_wrapped: bool = True
    duel_count: int = 0  # Track duels for this gift
    
    def get_value(self, player_id: int) -> float:
        return self.value_scores.get(player_id, 5.0)
    
    def min_cost(self, mode: MinCostMode = MinCostMode.LINEAR, cap: int = 99) -> int:
        """
        Minimum challenge cost based on naughty level.
        
        Modes:
        - LINEAR: 1 + naughty_level (default)
        - STEEP: 2 + naughty_level
        - EXPONENTIAL: 2^naughty_level
        - FLAT: always 1
        """
        if mode == MinCostMode.LINEAR:
            cost = 1 + self.naughty_level
        elif mode == MinCostMode.STEEP:
            cost = 2 + self.naughty_level
        elif mode == MinCostMode.EXPONENTIAL:
            cost = 2 ** self.naughty_level
        elif mode == MinCostMode.FLAT:
            cost = 1
        else:
            cost = 1 + self.naughty_level
        
        return min(cost, cap)


@dataclass
class Agent:
    player_id: int
    chips: int
    gift_id: Optional[int] = None
    in_bag: bool = True
    
    # AI behavior parameters
    valuation_matrix: Dict[int, float] = field(default_factory=dict)
    aggression: float = 0.5      # Bid/challenge tendency
    fomo: float = 0.5            # Tendency to bid on misfits
    risk_aversion: float = 0.5   # Tendency to conserve chips
    spite: float = 0.3           # Tendency to target specific players
    attachment: float = 0.0      # Endowment effect for current gift
    grudge: Dict[int, float] = field(default_factory=dict)  # Targeted revenge pressure
    visibility: float = 0.0      # How much they've been attacking recently
    
    # Tracking
    duels_won: int = 0
    duels_lost: int = 0
    chips_spent: int = 0
    dividends_received: int = 0


@dataclass
class GameState:
    config: GameConfig
    turn_phase: TurnPhase
    turn_index: int
    wrapped_gifts: List[int]
    misfit_pile: List[int]
    bag: List[int]
    head_elf_id: Optional[int]
    gift_registry: Dict[int, Gift]
    agents: Dict[int, Agent]
    bank_chips: int
    reprisal_depth: int = 0  # Current reprisal chain depth (0 = not in reprisal)
    
    # Detailed metrics
    total_duels: int = 0
    duels_by_type: Dict[DuelType, int] = field(default_factory=lambda: defaultdict(int))
    chips_history: List[Dict[int, int]] = field(default_factory=list)
    bankruptcy_events: int = 0
    total_bids: int = 0
    bid_amounts: List[int] = field(default_factory=list)
    yields_count: int = 0
    voluntary_challenges: int = 0
    reprisals_triggered: int = 0
    misfit_creations: int = 0
    dividends_paid: int = 0
    taxes_collected: int = 0
    
    # Event log for analysis
    event_log: List[Dict] = field(default_factory=list)


# =============================================================================
# OBSERVATION SPACE (for RL)
# =============================================================================

@dataclass
class AgentObservation:
    """What an agent can see."""
    self_chips: int
    self_gift_id: Optional[int]
    self_gift_value: float
    self_gift_naughty: int
    market_state: List[Tuple[int, int, int, float]]
    misfit_top_id: Optional[int]
    misfit_top_value: float
    misfit_top_naughty: int
    num_players_with_gifts: int
    num_giftless: int
    is_endgame: bool
    turn_progress: float
    avg_chips_others: float
    num_misfits: int


def get_observation(agent: Agent, state: GameState) -> AgentObservation:
    """Extract observation for an agent."""
    self_gift_value = 0.0
    self_gift_naughty = 0
    if agent.gift_id is not None:
        gift = state.gift_registry[agent.gift_id]
        self_gift_value = gift.get_value(agent.player_id)
        self_gift_naughty = gift.naughty_level
    
    market = []
    for gid, gift in state.gift_registry.items():
        if not gift.is_wrapped and gift.owner_id is not None:
            market.append((gid, gift.owner_id, gift.naughty_level, gift.get_value(agent.player_id)))
    
    misfit_top_id = state.misfit_pile[-1] if state.misfit_pile else None
    misfit_top_value = 0.0
    misfit_top_naughty = 0
    if misfit_top_id is not None:
        misfit = state.gift_registry[misfit_top_id]
        misfit_top_value = misfit.get_value(agent.player_id)
        misfit_top_naughty = misfit.naughty_level
    
    with_gifts = sum(1 for a in state.agents.values() if a.gift_id is not None)
    giftless = len(state.agents) - with_gifts
    
    other_chips = [a.chips for p, a in state.agents.items() if p != agent.player_id]
    avg_chips_others = np.mean(other_chips) if other_chips else 0
    
    return AgentObservation(
        self_chips=agent.chips,
        self_gift_id=agent.gift_id,
        self_gift_value=self_gift_value,
        self_gift_naughty=self_gift_naughty,
        market_state=market,
        misfit_top_id=misfit_top_id,
        misfit_top_value=misfit_top_value,
        misfit_top_naughty=misfit_top_naughty,
        num_players_with_gifts=with_gifts,
        num_giftless=giftless,
        is_endgame=state.turn_phase == TurnPhase.MISFIT_LOTTERY,
        turn_progress=state.turn_index / (state.config.NUM_PLAYERS * 2),
        avg_chips_others=avg_chips_others,
        num_misfits=len(state.misfit_pile),
    )


def obs_to_tensor(obs: AgentObservation, num_players: int) -> np.ndarray:
    """Convert observation to neural network input."""
    features = np.zeros(24, dtype=np.float32)
    
    features[0] = obs.self_chips / 10.0
    features[1] = 1.0 if obs.self_gift_id is not None else 0.0
    features[2] = obs.self_gift_value / 10.0
    features[3] = obs.self_gift_naughty / 3.0
    features[4] = obs.misfit_top_value / 10.0 if obs.misfit_top_id else 0.0
    features[5] = obs.misfit_top_naughty / 3.0 if obs.misfit_top_id else 0.0
    features[6] = obs.num_players_with_gifts / num_players
    features[7] = obs.num_giftless / num_players
    features[8] = 1.0 if obs.is_endgame else 0.0
    features[9] = obs.turn_progress
    features[10] = obs.avg_chips_others / 10.0
    features[11] = obs.num_misfits / num_players
    
    if obs.market_state:
        values = [m[3] for m in obs.market_state]
        naughties = [m[2] for m in obs.market_state]
        features[12] = np.mean(values) / 10.0
        features[13] = np.max(values) / 10.0
        features[14] = np.min(values) / 10.0
        features[15] = np.mean(naughties) / 3.0
        features[16] = np.std(values) / 5.0
    
    if obs.market_state and obs.self_gift_value > 0:
        better = sum(1 for m in obs.market_state if m[3] > obs.self_gift_value)
        features[17] = better / max(1, len(obs.market_state))
    else:
        features[17] = 1.0
    
    features[18] = len(obs.market_state) / num_players if obs.market_state else 0.0
    
    # Chip advantage
    if obs.avg_chips_others > 0:
        features[19] = (obs.self_chips - obs.avg_chips_others) / 10.0
    
    return features


# =============================================================================
# GAME ENGINE
# =============================================================================

class JingleBrawlEnv:
    """
    Jingle Brawl optimization environment with configurable rules.
    
    Features:
    - Configurable min_cost formula (LINEAR, STEEP, EXPONENTIAL, FLAT)
    - Configurable duel model (FAIR_COIN, SKILL_WEIGHTED, CHIPS_INFLUENCE)
    - Deterministic seeding for reproducibility
    - YAML config file support
    """
    
    def __init__(self, config: GameConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.state: Optional[GameState] = None
        
        # Set up random number generator for reproducibility
        if config.SEED is not None:
            self.rng = np.random.Generator(np.random.PCG64(config.SEED))
            random.seed(config.SEED)
        else:
            self.rng = np.random.Generator(np.random.PCG64())
        
        # Player skills for skill-weighted duels (assigned on reset)
        self.player_skills: Dict[int, float] = {}
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def log_event(self, event_type: str, **kwargs):
        """Log event for analysis."""
        if self.state:
            self.state.event_log.append({
                'turn': self.state.turn_index,
                'type': event_type,
                **kwargs
            })
    
    def _assign_base_quality(self) -> float:
        """Assign base quality to a gift based on configured distribution."""
        dist = self.config.GIFT_VALUE_DIST
        
        if dist == GiftValueDistribution.UNIFORM:
            return self.rng.uniform(1, 10)
        elif dist == GiftValueDistribution.NORMAL:
            return max(1, min(10, self.rng.normal(5.5, 2.0)))
        elif dist == GiftValueDistribution.TIERED:
            tier = self.rng.random()
            if tier < 0.3:
                return self.rng.uniform(1, 4)  # Low tier
            elif tier < 0.7:
                return self.rng.uniform(4, 7)  # Mid tier
            else:
                return self.rng.uniform(7, 10)  # High tier
        else:
            return self.rng.uniform(1, 10)
    
    def _assign_player_value(self, base_quality: float) -> float:
        """
        Assign a player's personal value for a gift.
        
        CORRELATED model: base_quality + personal taste variation
        - Creates "universally bad" gifts (low base = everyone dislikes)
        - Creates "universally good" gifts (high base = everyone likes)
        - Personal taste adds variation around the base
        
        INDEPENDENT model: fully random (old behavior)
        - No correlation between player opinions
        - Every gift has someone who likes it
        """
        model = self.config.GIFT_VALUE_MODEL
        
        if model == GiftValueModel.CORRELATED:
            # Personal taste varies within ±TASTE_VARIANCE of base quality
            taste = self.rng.uniform(-self.config.TASTE_VARIANCE, self.config.TASTE_VARIANCE)
            return max(1, min(10, base_quality + taste))
        else:
            # INDEPENDENT: fully random, ignores base quality
            return self._assign_base_quality()
    
    def reset(self) -> GameState:
        """Initialize a new game."""
        # Reset RNG if seeded (for multiple games with same seed pattern)
        if self.config.SEED is not None:
            game_seed = self.config.SEED + hash(time.time()) % 10000
            self.rng = np.random.Generator(np.random.PCG64(game_seed))
        
        # Create gifts and assign base qualities
        gift_registry = {}
        self.gift_base_qualities = {}  # Track base quality for analysis
        for i in range(self.config.NUM_GIFTS):
            gift = Gift(gift_id=i)
            gift_registry[i] = gift
            self.gift_base_qualities[i] = self._assign_base_quality()
        
        # Assign player skills for skill-weighted duels
        self.player_skills = {}
        for i in range(self.config.NUM_PLAYERS):
            self.player_skills[i] = self.rng.normal(0, 1)
        
        # Create agents and assign personal valuations
        agents = {}
        for i in range(self.config.NUM_PLAYERS):
            agent = Agent(
                player_id=i,
                chips=self.config.INITIAL_CHIPS,
                aggression=self.rng.uniform(0.3, 0.8),
                fomo=self.rng.uniform(0.2, 0.7),
                risk_aversion=self.rng.uniform(0.2, 0.6),
                spite=self.rng.uniform(0.1, 0.4),
            )
            for gid in range(self.config.NUM_GIFTS):
                # Use base quality for CORRELATED model
                base_quality = self.gift_base_qualities[gid]
                value = self._assign_player_value(base_quality)
                agent.valuation_matrix[gid] = value
                gift_registry[gid].value_scores[i] = value
            agents[i] = agent
        
        bag = list(range(self.config.NUM_PLAYERS))
        self.rng.shuffle(bag)
        
        self.state = GameState(
            config=self.config,
            turn_phase=TurnPhase.MAIN_GAME,
            turn_index=0,
            wrapped_gifts=list(range(self.config.NUM_GIFTS)),
            misfit_pile=[],
            bag=list(bag),
            head_elf_id=0,
            gift_registry=gift_registry,
            agents=agents,
            bank_chips=0,
        )
        
        wrapped = list(range(self.config.NUM_GIFTS))
        self.rng.shuffle(wrapped)
        self.state.wrapped_gifts = list(wrapped)
        return self.state
    
    # -------------------------------------------------------------------------
    # ECONOMY RULES
    # -------------------------------------------------------------------------
    
    def apply_santa_tax(self, pot: int) -> Tuple[int, int]:
        """Apply Santa Tax based on config. Returns (tax, winner_payout)."""
        if self.config.SANTA_TAX_RULE == SantaTaxRule.FLAT_1:
            # SPEC: If pot >= 3, tax is 1
            if pot >= 3:
                tax = 1
                self.state.taxes_collected += tax
                return (tax, pot - 1)
            return (0, pot)
        elif self.config.SANTA_TAX_RULE == SantaTaxRule.CAP_2:
            if pot > 2:
                tax = pot - 2
                self.state.taxes_collected += tax
                return (tax, 2)
            return (0, pot)
        elif self.config.SANTA_TAX_RULE == SantaTaxRule.PROGRESSIVE:
            tax = pot // 3
            self.state.taxes_collected += tax
            return (tax, pot - tax)
        else:  # NONE
            return (0, pot)
    
    def pay_loser_dividend(self, loser_id: int, duel_type: DuelType, 
                           was_defender: bool) -> int:
        """
        Pay Loser's Dividend based on config.
        SPEC: Only DEFENDER gets +1, and NOT for misfit/tie-break/yield duels.
        """
        # No dividend for certain duel types
        if duel_type in NO_DIVIDEND_DUELS:
            return 0
        
        if self.config.LOSER_DIVIDEND_RULE == LoserDividendRule.NONE:
            return 0
        elif self.config.LOSER_DIVIDEND_RULE == LoserDividendRule.DEFENDER_ONLY:
            if was_defender:
                self.state.agents[loser_id].chips += 1
                self.state.agents[loser_id].dividends_received += 1
                self.state.dividends_paid += 1
                return 1
            return 0
        else:  # ANY_LOSER
            self.state.agents[loser_id].chips += 1
            self.state.agents[loser_id].dividends_received += 1
            self.state.dividends_paid += 1
            return 1
    
    def min_cost(self, gift: Gift) -> int:
        """
        Minimum challenge cost based on gift's naughty level and config.
        
        Uses configurable MIN_COST_MODE:
        - LINEAR: 1 + naughty_level (default)
        - STEEP: 2 + naughty_level  
        - EXPONENTIAL: 2^naughty_level
        - FLAT: always 1
        """
        return gift.min_cost(self.config.MIN_COST_MODE, self.config.NAUGHTY_LEVEL_CAP)
    
    # -------------------------------------------------------------------------
    # SWAP ALGORITHM (per spec)
    # -------------------------------------------------------------------------
    
    def execute_swap(self, winner_id: int, loser_id: int, contested_gift_id: int):
        """
        The Swap Algorithm - handles gift transfers.
        
        1. Winner takes contested gift
        2. If winner had a gift, it goes to loser
        3. If loser had a gift AND received winner's old gift,
           loser's original gift goes to misfit pile
           
        BAG MANAGEMENT:
        - Winner who gains a gift (was giftless) -> removed from bag
        - Loser who becomes giftless -> added back to bag
        """
        winner = self.state.agents[winner_id]
        loser = self.state.agents[loser_id]
        contested = self.state.gift_registry[contested_gift_id]
        
        winner_old_gift_id = winner.gift_id
        loser_old_gift_id = loser.gift_id
        winner_was_giftless = (winner_old_gift_id is None)
        
        # Detach contested gift from current owner
        if contested.owner_id is not None and contested.owner_id != winner_id:
            self.state.agents[contested.owner_id].gift_id = None
            self.state.agents[contested.owner_id].attachment *= (1 - self.config.ATTACHMENT_DECAY)
        
        # Remove from misfit pile if there
        if contested_gift_id in self.state.misfit_pile:
            self.state.misfit_pile.remove(contested_gift_id)
        
        # Winner takes contested gift
        winner.gift_id = contested_gift_id
        contested.owner_id = winner_id
        contested.is_wrapped = False
        # Attachment update for winner: new gift -> reset attachment baseline
        winner.attachment = self.config.ATTACHMENT_GAIN
        
        # Track if loser becomes giftless
        loser_becomes_giftless = False
        
        # Winner's old gift goes to loser
        if winner_old_gift_id is not None and winner_old_gift_id != contested_gift_id:
            old_gift = self.state.gift_registry[winner_old_gift_id]
            
            # Misfit trigger: if loser had a different gift
            if loser_old_gift_id is not None and loser_old_gift_id != winner_old_gift_id:
                # LOSER'S CHOICE: Which gift to displace to misfit pile?
                loser_orig = self.state.gift_registry[loser_old_gift_id]
                received_gift = old_gift  # Winner's old gift that loser is receiving
                
                # Loser chooses which gift to send to misfit pile
                displace_id = self.agent_misfit_choice(loser, loser_orig, received_gift)
                keep_id = loser_old_gift_id if displace_id == winner_old_gift_id else winner_old_gift_id
                
                # Update ownership based on choice
                displaced = self.state.gift_registry[displace_id]
                kept = self.state.gift_registry[keep_id]
                
                displaced.owner_id = None
                loser.gift_id = keep_id
                kept.owner_id = loser_id
                
                self.state.misfit_pile.append(displace_id)
                self.state.head_elf_id = loser_id
                self.state.misfit_creations += 1
                
                choice_desc = "kept original" if keep_id == loser_old_gift_id else "kept received"
                self.log(f"    Misfit: P{loser_id} {choice_desc}, Gift {displace_id} -> pile. Head Elf: P{loser_id}")
                self.log_event('MISFIT_CREATED', gift_id=displace_id, head_elf=loser_id, 
                              loser_choice=choice_desc, kept_gift=keep_id)
                # Attachment updates
                loser.attachment = self.config.ATTACHMENT_GAIN
            else:
                # No misfit scenario - loser just receives winner's old gift
                loser.gift_id = winner_old_gift_id
                old_gift.owner_id = loser_id
                loser.attachment = self.config.ATTACHMENT_GAIN
        elif winner_old_gift_id is None:
            # Winner was giftless - loser keeps their gift or becomes giftless
            if loser_old_gift_id is not None and loser_old_gift_id != contested_gift_id:
                pass  # Loser keeps their gift
            else:
                loser.gift_id = None
                loser_becomes_giftless = True
                loser.attachment = 0.0
        
        # BAG MANAGEMENT
        # If winner was giftless and now has a gift, remove from bag
        if winner_was_giftless and winner.gift_id is not None:
            if winner_id in self.state.bag:
                self.state.bag.remove(winner_id)
                self.log(f"    Bag: P{winner_id} removed (now has gift)")
        
        # If loser becomes giftless, add back to bag
        if loser_becomes_giftless and loser.gift_id is None:
            if loser_id not in self.state.bag:
                self.state.bag.append(loser_id)
                self.log(f"    Bag: P{loser_id} added (now giftless)")
        
        # Increment naughty level
        contested.naughty_level += 1
        contested.duel_count += 1
    
    # -------------------------------------------------------------------------
    # DUEL RESOLUTION
    # -------------------------------------------------------------------------
    
    def resolve_duel(self, attacker_id: int, defender_id: int) -> int:
        """
        Resolve a duel between two players.
        
        Duel models:
        - FAIR_COIN: 50/50 random
        - SKILL_WEIGHTED: Based on player skill attributes (logistic)
        - CHIPS_INFLUENCE: More chips = higher win probability
        - DEFENDER_ADVANTAGE: 60/40 in defender's favor
        """
        model = self.config.DUEL_MODEL
        
        if model == DuelModel.FAIR_COIN:
            return attacker_id if self.rng.random() < 0.5 else defender_id
        
        elif model == DuelModel.SKILL_WEIGHTED:
            # Logistic probability based on skill difference
            skill_a = self.player_skills.get(attacker_id, 0)
            skill_d = self.player_skills.get(defender_id, 0)
            noise = self.rng.normal(0, 0.5)
            # Positive diff = attacker advantage
            diff = skill_a - skill_d + noise
            prob_attacker = 1 / (1 + np.exp(-diff))
            return attacker_id if self.rng.random() < prob_attacker else defender_id
        
        elif model == DuelModel.CHIPS_INFLUENCE:
            # More chips = higher win probability
            chips_a = self.state.agents[attacker_id].chips
            chips_d = self.state.agents[defender_id].chips
            total = chips_a + chips_d + 1  # +1 to avoid division by zero
            prob_attacker = (chips_a + 0.5) / (total + 1)  # Slight floor to prevent 0%
            return attacker_id if self.rng.random() < prob_attacker else defender_id
        
        elif model == DuelModel.DEFENDER_ADVANTAGE:
            # 40% attacker, 60% defender
            return attacker_id if self.rng.random() < 0.4 else defender_id
        
        else:
            # Default to fair coin
            return attacker_id if self.rng.random() < 0.5 else defender_id
    
    def execute_duel(self, challenger_id: int, defender_id: int, 
                     contested_gift_id: int, pot: int, 
                     duel_type: DuelType) -> Tuple[int, int, bool]:
        """
        Execute a duel with all economy effects.
        Returns (winner_id, loser_id, loser_was_defender)
        """
        # Apply tax
        tax, payout = self.apply_santa_tax(pot)
        self.state.bank_chips += tax
        
        # Resolve duel
        winner_id = self.resolve_duel(challenger_id, defender_id)
        loser_id = defender_id if winner_id == challenger_id else challenger_id
        was_defender = (loser_id == defender_id)
        
        # Update agent stats
        self.state.agents[winner_id].duels_won += 1
        self.state.agents[loser_id].duels_lost += 1
        
        # Winner gets payout
        self.state.agents[winner_id].chips += payout
        
        # Loser dividend
        dividend = self.pay_loser_dividend(loser_id, duel_type, was_defender)
        
        self.log(f"  Duel ({duel_type.name}): P{challenger_id} vs P{defender_id} -> Winner: P{winner_id}")
        self.log(f"    Pot={pot}, Tax={tax}, Payout={payout}, Dividend={dividend}")
        
        # Execute swap
        self.execute_swap(winner_id, loser_id, contested_gift_id)
        
        # Track metrics
        self.state.total_duels += 1
        self.state.duels_by_type[duel_type] += 1
        # Visibility / grudge tracking
        self.state.agents[challenger_id].visibility += 1.0
        # Loser holds grudge against winner
        self.state.agents[loser_id].grudge[winner_id] = self.state.agents[loser_id].grudge.get(winner_id, 0.0) + 1.0
        
        # Log event
        self.log_event('DUEL', 
            duel_type=duel_type.name,
            challenger=challenger_id,
            defender=defender_id,
            winner=winner_id,
            pot=pot,
            tax=tax,
            payout=payout,
            dividend=dividend
        )
        
        # Check bankruptcy
        for agent in self.state.agents.values():
            if agent.chips == 0:
                self.state.bankruptcy_events += 1
        
        return winner_id, loser_id, was_defender
    
    def check_reprisal_eligibility(self, loser_id: int, was_defender: bool, 
                                    duel_type: DuelType) -> bool:
        """
        Check if a reprisal should be triggered based on config.
        Supports configurable chaining depth via REPRISAL_MAX_DEPTH.
        """
        # Check depth limit
        if self.state.reprisal_depth >= self.config.REPRISAL_MAX_DEPTH:
            return False  # Max chain depth reached
        
        # Reprisals never trigger from these duel types
        never_trigger_duels = {
            DuelType.TIE_BREAK,
            DuelType.MISFIT_CHALLENGE,
            DuelType.HEAD_ELF_DEFENSE,
            DuelType.ENDGAME_STEAL,
            DuelType.ENDGAME_AUCTION,
        }
        if duel_type in never_trigger_duels:
            return False
        
        # REPRISAL duels can chain if depth < max_depth (checked above)
        # So we don't exclude DuelType.REPRISAL here
        
        rule = self.config.REPRISAL_TRIGGER_RULE
        
        if rule == ReprisalTriggerRule.DISABLED:
            return False
        elif rule == ReprisalTriggerRule.VOLUNTARY_ONLY:
            # Only triggers from voluntary challenges (handled separately)
            return False
        elif rule == ReprisalTriggerRule.ANY_DEFENDER_LOSS:
            return was_defender
        elif rule == ReprisalTriggerRule.ANY_LOSS:
            return True
        
        return False
    
    # -------------------------------------------------------------------------
    # AGENT DECISIONS (Heuristic AI)
    # -------------------------------------------------------------------------
    
    def agent_bid_decision(self, agent: Agent, gift: Gift, is_endgame: bool = False) -> int:
        """
        Smart agent bidding that properly respects the naughty level mechanic.
        
        KEY DESIGN INTENT:
        The naughty level exists to PROTECT popular gifts from endless stealing
        (like traditional white elephant's "3 steal max"). High naughty gifts
        should be progressively less attractive because they're EXPENSIVE.
        
        Considers:
        - Cost-to-value ratio (is this gift WORTH the cost?)
        - Naughty level as "protection" mechanic
        - Chip reserves (don't overspend)
        - Risk aversion and aggression personality
        - SEALED_BID_MIN/MAX limits (for unwrapped gift bidding only)
        """
        min_bid = self.min_cost(gift)
        
        # Apply SEALED_BID_MIN floor (must bid at least this much to participate)
        min_bid = max(min_bid, self.config.SEALED_BID_MIN)
        
        # Can't afford minimum bid
        if agent.chips < min_bid:
            return 0
        
        # Calculate value gain
        value = gift.get_value(agent.player_id)
        current_value = 0
        if agent.gift_id is not None:
            current_value = self.state.gift_registry[agent.gift_id].get_value(agent.player_id)
            # Attachment/endowment effect
            current_value += agent.attachment * self.config.ATTACHMENT_WEIGHT
        gain = value - current_value
        
        # === COST-BENEFIT ANALYSIS (KEY FIX) ===
        # The fundamental question: Is the VALUE GAIN worth the COST?
        # Gift worth 8, naughty=0, cost=1 → ratio 8.0 (great deal!)
        # Gift worth 8, naughty=3, cost=4 → ratio 2.0 (okay deal)
        # Gift worth 8, naughty=5, cost=6 → ratio 1.3 (bad deal - overpriced!)
        
        value_per_chip = gain / min_bid if min_bid > 0 else gain
        
        # Threshold: Need good value-per-chip to consider it worthwhile
        # Aggressive players accept worse deals (lower threshold)
        cost_benefit_threshold = 2.0 - (agent.aggression * 0.8)  # Range: 1.2 to 2.0
        
        # === DECISION BASED ON COST-BENEFIT ===
        if gain <= 0:
            # No value gain - only grief bidders consider it
            bid_prob = agent.aggression * 0.05 * agent.spite
        elif value_per_chip < cost_benefit_threshold:
            # OVERPRICED: Gift costs too much relative to gain
            # This is where naughty level protection kicks in!
            bid_prob = 0.15 * agent.aggression  # Only aggressive players consider
            bid_prob *= max(0.1, value_per_chip / cost_benefit_threshold)  # Scale down further
        else:
            # GOOD DEAL: Worth the cost
            bid_prob = 0.4 + 0.3 * agent.aggression
            bid_prob += 0.2 * min(value_per_chip / 3, 1.0)  # Bonus for great deals
        
        # === NAUGHTY LEVEL AS "PROTECTION" SIGNAL ===
        # Even if math works out, high naughty signals "back off, hot item"
        # This simulates player psychology and provides additional protection
        if gift.naughty_level >= 4:
            bid_prob *= 0.15  # Very strong deterrent - gift is "protected"
        elif gift.naughty_level >= 3:
            bid_prob *= 0.35  # Strong deterrent
        elif gift.naughty_level >= 2:
            bid_prob *= 0.55  # Moderate deterrent
        elif gift.naughty_level >= 1:
            bid_prob *= 0.75  # Slight deterrent
        
        # === CHIP RESERVE CONSIDERATION ===
        chips_after = agent.chips - min_bid
        reserve_ratio = chips_after / self.config.INITIAL_CHIPS
        if reserve_ratio < 0.2:
            bid_prob *= 0.2  # Very strong disincentive - nearly broke
        elif reserve_ratio < 0.4:
            bid_prob *= 0.5  # Moderate disincentive
        
        # === GIFTLESS DESPERATION ===
        # Giftless players are more desperate, but STILL respect costs
        if agent.gift_id is None:
            bid_prob += 0.25
            # Lower threshold slightly when desperate
            if value_per_chip >= cost_benefit_threshold * 0.6:
                bid_prob += 0.15
        
        # Risk aversion affects willingness to pay high costs
        bid_prob *= (1 - agent.risk_aversion * 0.1 * min(min_bid, 6))
        
        # Endgame FOMO
        if is_endgame:
            bid_prob *= (1 + agent.fomo * 0.25)
        
        # Final decision: bid or pass
        if random.random() > bid_prob:
            return 0
        
        # === SMART BID AMOUNT ===
        # Don't just bid max - consider value and cost
        
        # Budget: what can we afford while keeping reserves (phase-aware)?
        phase_factor = len(self.state.wrapped_gifts) / max(1, self.config.NUM_GIFTS)
        target_reserve = int(agent.risk_aversion * self.config.INITIAL_CHIPS * (self.config.LIQUIDITY_RESERVE * phase_factor))
        affordable = max(min_bid, agent.chips - target_reserve)
        
        # Bid based on value gain
        if gain >= 6:  # High value - bid more
            bid = min(min_bid + 2, affordable)
        elif gain >= 3:  # Medium value
            bid = min(min_bid + 1, affordable)
        else:  # Low value - bid minimum
            bid = min_bid
        
        # Apply noise + quantization to mimic human bids
        if self.config.BID_NOISE > 0:
            bid = int(round(bid + self.rng.normal(0, self.config.BID_NOISE)))
        if self.config.BID_QUANTIZATION:
            # Snap to simple levels near min_bid
            ladder = [min_bid + i for i in range(0, 4)]
            bid = min(ladder, key=lambda x: abs(x - bid))
        
        # Apply constraints
        bid = min(bid, agent.chips)  # Never bid more than we have
        if self.config.SEALED_BID_MAX is not None:
            bid = min(bid, self.config.SEALED_BID_MAX)
        bid = max(min_bid, bid)
        
        # Visibility penalty (anti-bully)
        bid_prob *= max(0.3, 1 - agent.visibility * self.config.VISIBILITY_PENALTY)
        
        # Final decision: with new bid
        if random.random() > bid_prob:
            return 0
        
        return bid
    
    def agent_yield_decision(self, agent: Agent, gift: Gift, c2_exists: bool) -> bool:
        """Agent decides whether to yield."""
        if self.config.YIELD_RULE == YieldRule.DISABLED:
            return False
        if not c2_exists:
            return False
        
        # Low aggression players more likely to yield
        yield_prob = (1 - agent.aggression) * 0.4
        
        # High value gifts less likely to yield
        value = gift.get_value(agent.player_id)
        if value > 7:
            yield_prob *= 0.3
        
        return random.random() < yield_prob
    
    def agent_voluntary_target(self, agent: Agent) -> Optional[int]:
        """
        Agent chooses voluntary challenge target.
        
        KEY INSIGHT: If the opener doesn't like their gift, they should be
        MORE motivated to challenge - even if they're not typically aggressive.
        """
        if self.config.VOLUNTARY_CHALLENGE_RULE == VoluntaryChallengeRule.DISABLED:
            return None
        
        # How much does the opener value their current gift?
        current_value = 0
        if agent.gift_id is not None:
            current_value = self.state.gift_registry[agent.gift_id].get_value(agent.player_id)
        
        # Dissatisfaction: 0-10 scale (higher = more unhappy with current gift)
        # A gift worth 2 to them when max is 10 gives dissatisfaction of 8
        dissatisfaction = max(0, 10 - current_value)
        
        # Motivation to challenge = base aggression + dissatisfaction boost
        # Unhappy opener is more motivated even if not naturally aggressive
        motivation = agent.aggression + (dissatisfaction / 20)  # +0.0 to +0.5 boost
        
        if random.random() > motivation:
            return None
        
        best_target = None
        best_score = 0
        
        for pid, other in self.state.agents.items():
            if pid == agent.player_id or other.gift_id is None:
                continue
            gift = self.state.gift_registry[other.gift_id]
            value = gift.get_value(agent.player_id)
            cost = self.min_cost(gift)
            
            if agent.chips < cost:
                continue
                
            # Score = potential gain (target value - current value) minus cost factor
            gain = value - current_value
            score = gain - cost * 0.5
            
            if score > best_score:
                best_score = score
                best_target = pid
        
        # Lower threshold if very dissatisfied (desperate to trade up)
        threshold = max(1, 3 - dissatisfaction / 5)  # 3 -> 1 based on dissatisfaction
        
        return best_target if best_score > threshold else None
    
    def agent_reprisal_target(self, agent: Agent, excluded_gift_id: Optional[int] = None) -> Optional[Tuple[str, int]]:
        """
        Agent chooses reprisal target. Returns ('player', id) or ('misfit', id) or None.
        
        V2 SPEC: "may challenge any gift... except for the gift involved in the duel you just lost"
        """
        if random.random() > agent.aggression * 1.2:  # Reprisal slightly more likely
            return None
        
        best_target = None
        best_value = 0
        target_type = None
        # Grudge bonus: prioritize players with highest grudge
        grudge_bonus = dict(agent.grudge)
        
        # Check players
        for pid, other in self.state.agents.items():
            if pid == agent.player_id or other.gift_id is None:
                continue
            # V2 SPEC: Can't target the gift you just lost the duel for
            if other.gift_id == excluded_gift_id:
                continue
            gift = self.state.gift_registry[other.gift_id]
            value = gift.get_value(agent.player_id)
            cost = self.min_cost(gift)
            if agent.chips >= cost:
                score = value + grudge_bonus.get(pid, 0) * self.config.REVENGE_WEIGHT
                if score > best_value:
                    best_value = score
                    best_target = pid
                    target_type = 'player'
        
        # Check misfits
        if self.config.REPRISAL_TARGETS_MISFITS and self.state.misfit_pile:
            for mid in self.state.misfit_pile:
                # V2 SPEC: Can't target the gift you just lost the duel for
                if mid == excluded_gift_id:
                    continue
                misfit = self.state.gift_registry[mid]
                value = misfit.get_value(agent.player_id)
                cost = self.min_cost(misfit)
                if agent.chips >= cost and value > best_value:
                    best_value = value
                    best_target = mid
                    target_type = 'misfit'
        
        return (target_type, best_target) if best_target is not None else None
    
    def agent_misfit_choice(self, loser: Agent, original_gift: Gift, received_gift: Gift) -> int:
        """
        Loser chooses which gift to displace to the misfit pile.
        
        Called when the loser already had a gift AND receives the winner's old gift.
        The loser must choose one to keep and one to displace.
        
        Returns: gift_id to displace to misfit pile
        """
        orig_value = original_gift.get_value(loser.player_id)
        recv_value = received_gift.get_value(loser.player_id)
        
        # Consider naughty levels - higher naughty = more expensive to reclaim from misfit
        orig_reclaim_cost = self.min_cost(original_gift)
        recv_reclaim_cost = self.min_cost(received_gift)
        
        # Score = value - reclaim_penalty (if it goes to misfit, harder to get back)
        # We want to KEEP the higher score gift
        orig_score = orig_value - (orig_reclaim_cost * loser.risk_aversion * 0.3)
        recv_score = recv_value - (recv_reclaim_cost * loser.risk_aversion * 0.3)
        
        # Tie-breaker: prefer keeping lower naughty (fresher gift)
        if abs(orig_score - recv_score) < 0.5:
            if original_gift.naughty_level <= received_gift.naughty_level:
                return received_gift.gift_id  # Displace received, keep original
            else:
                return original_gift.gift_id  # Displace original, keep received
        
        if orig_score >= recv_score:
            return received_gift.gift_id  # Keep original, displace received
        else:
            return original_gift.gift_id  # Keep received, displace original
    
    def agent_endgame_path(self, agent: Agent, misfit: Gift) -> str:
        """Agent chooses Path A (steal) or Path B (auction)."""
        misfit_value = misfit.get_value(agent.player_id)
        
        best_steal_value = 0
        best_steal_cost = 99
        for pid, other in self.state.agents.items():
            if pid == agent.player_id or other.gift_id is None:
                continue
            gift = self.state.gift_registry[other.gift_id]
            value = gift.get_value(agent.player_id)
            cost = self.min_cost(gift)
            if value > best_steal_value and agent.chips >= cost:
                best_steal_value = value
                best_steal_cost = cost
        
        # Path A if there's a significantly better gift to steal
        if agent.aggression > 0.4 and best_steal_value > misfit_value + 2:
            return 'A'
        
        # Path B (auction) if misfit is decent or no good steals
        return 'B'
    
    # -------------------------------------------------------------------------
    # MAIN PHASE
    # -------------------------------------------------------------------------
    
    def run_main_phase_turn(self, opener_id: int):
        """Run a single main phase turn."""
        opener = self.state.agents[opener_id]
        opener.in_bag = False
        
        # Unwrap gift
        gift_id = self.state.wrapped_gifts.pop()
        gift = self.state.gift_registry[gift_id]
        gift.is_wrapped = False
        gift.naughty_level = 0
        
        self.log(f"\nTurn {self.state.turn_index}: P{opener_id} opens Gift {gift_id} (value={gift.get_value(opener_id):.1f})")
        self.log_event('UNWRAP', opener=opener_id, gift_id=gift_id)
        
        # Bidding (everyone except opener)
        bids = []
        for pid, agent in self.state.agents.items():
            if pid == opener_id:
                continue
            bid = self.agent_bid_decision(agent, gift)
            if bid > 0:
                bids.append((pid, bid))
                self.state.total_bids += 1
                self.state.bid_amounts.append(bid)
        
        bids.sort(key=lambda x: x[1], reverse=True)
        
        if not bids:
            # No bids - opener keeps gift
            self.log(f"  No bids - P{opener_id} keeps gift")
            self.log_event('NO_BIDS', opener=opener_id, gift_id=gift_id)
            old_gift_id = opener.gift_id
            opener_was_giftless = (old_gift_id is None)
            
            if old_gift_id is not None:
                # OPENER'S CHOICE: Which gift to displace to misfit pile?
                old_gift = self.state.gift_registry[old_gift_id]
                new_gift = gift
                
                # Opener chooses which gift to keep
                displace_id = self.agent_misfit_choice(opener, old_gift, new_gift)
                keep_id = old_gift_id if displace_id == gift_id else gift_id
                
                # Update ownership based on choice
                displaced = self.state.gift_registry[displace_id]
                kept = self.state.gift_registry[keep_id]
                
                displaced.owner_id = None
                opener.gift_id = keep_id
                kept.owner_id = opener_id
                
                self.state.misfit_pile.append(displace_id)
                self.state.head_elf_id = opener_id
                self.state.misfit_creations += 1
                
                choice_desc = "kept old gift" if keep_id == old_gift_id else "kept new gift"
                self.log(f"    Misfit: P{opener_id} {choice_desc}, Gift {displace_id} -> pile. Head Elf: P{opener_id}")
                self.log_event('MISFIT_CREATED', gift_id=displace_id, head_elf=opener_id,
                              opener_choice=choice_desc, kept_gift=keep_id)
            else:
                # Opener was giftless - just takes the new gift
                opener.gift_id = gift_id
                gift.owner_id = opener_id
            
            # BAG: If opener was giftless and now has gift, ensure they're out of bag
            # (They were already removed when drawn, but this is defensive)
            if opener_was_giftless and opener_id in self.state.bag:
                self.state.bag.remove(opener_id)
            
            # Voluntary challenge opportunity
            if self.config.VOLUNTARY_CHALLENGE_RULE == VoluntaryChallengeRule.ENABLED:
                target_id = self.agent_voluntary_target(opener)
                if target_id is not None:
                    self.state.voluntary_challenges += 1
                    self.run_voluntary_challenge(opener_id, target_id)
        else:
            # Handle ties with tie-break duel
            c1_id, c1_bid = bids[0]
            c2_id, c2_bid = None, 0
            
            ties = [b for b in bids if b[1] == c1_bid]
            if len(ties) > 1:
                winner = self.resolve_duel(ties[0][0], ties[1][0])
                c1_id = winner
                self.state.total_duels += 1
                self.state.duels_by_type[DuelType.TIE_BREAK] += 1
                self.log(f"  Tie-break: P{ties[0][0]} vs P{ties[1][0]} -> P{winner}")
                remaining = [b for b in bids if b[0] != winner]
                if remaining:
                    c2_id, c2_bid = remaining[0]
            elif len(bids) > 1:
                c2_id, c2_bid = bids[1]
            
            # Yield decision - must check BEFORE paying to calculate pot correctly
            if c2_id is not None and self.agent_yield_decision(opener, gift, True):
                # V2 SPEC: Yield duel - both challengers pay HALF their bids, rounded DOWN
                c1_half = c1_bid // 2
                c2_half = c2_bid // 2
                
                self.state.agents[c1_id].chips -= c1_half
                self.state.agents[c1_id].chips_spent += c1_half
                self.state.agents[c2_id].chips -= c2_half
                self.state.agents[c2_id].chips_spent += c2_half
                pot = c1_half + c2_half
                
                self.log(f"  P{opener_id} YIELDS to P{c2_id}")
                self.log(f"    C1 pays {c1_half} (half of {c1_bid}), C2 pays {c2_half} (half of {c2_bid})")
                self.state.yields_count += 1
                self.log_event('YIELD', opener=opener_id, c1=c1_id, c2=c2_id, 
                              c1_paid=c1_half, c2_paid=c2_half, pot=pot)
                winner_id, loser_id, was_defender = self.execute_duel(c1_id, c2_id, gift_id, pot, DuelType.YIELD_DUEL)
                # Check for reprisal on yield duel
                if self.check_reprisal_eligibility(loser_id, was_defender, DuelType.YIELD_DUEL):
                    self.state.reprisals_triggered += 1
                    self.run_reprisal(loser_id, excluded_gift_id=gift_id)
            else:
                # V2 SPEC: Normal duel - Challenger 1 pays their FULL bid into the pot
                self.state.agents[c1_id].chips -= c1_bid
                self.state.agents[c1_id].chips_spent += c1_bid
                pot = c1_bid
                
                self.log(f"  P{opener_id} FIGHTS P{c1_id} (bid={c1_bid})")
                winner_id, loser_id, was_defender = self.execute_duel(c1_id, opener_id, gift_id, pot, DuelType.SEALED_BID)
                # Check for reprisal on sealed bid duel
                if self.check_reprisal_eligibility(loser_id, was_defender, DuelType.SEALED_BID):
                    self.state.reprisals_triggered += 1
                    self.run_reprisal(loser_id, excluded_gift_id=gift_id)
        
        # Snapshot chips
        self.state.chips_history.append({p: a.chips for p, a in self.state.agents.items()})
    
    def run_voluntary_challenge(self, ap_id: int, target_id: int):
        """Run voluntary challenge with reprisal."""
        ap = self.state.agents[ap_id]
        target = self.state.agents[target_id]
        
        if target.gift_id is None:
            return
        
        gift = self.state.gift_registry[target.gift_id]
        cost = self.min_cost(gift)
        
        if ap.chips < cost:
            return
        
        # SPEC: Voluntary challenge cost goes directly to bank, no pot
        ap.chips -= cost
        ap.chips_spent += cost
        self.state.bank_chips += cost
        
        self.log(f"  VOLUNTARY: P{ap_id} challenges P{target_id} (cost={cost} to bank)")
        self.log_event('VOLUNTARY_CHALLENGE', challenger=ap_id, target=target_id, cost=cost)
        
        # Pot is 0 for voluntary challenges (cost went to bank)
        winner_id, loser_id, was_defender = self.execute_duel(ap_id, target_id, target.gift_id, 0, DuelType.VOLUNTARY)
        
        # Check reprisal based on config
        should_reprisal = False
        if self.config.REPRISAL_TRIGGER_RULE == ReprisalTriggerRule.VOLUNTARY_ONLY:
            # SPEC default: Reprisal only if challenger (AP) wins voluntary challenge
            should_reprisal = (winner_id == ap_id)
        else:
            # Other rules: check based on who lost
            should_reprisal = self.check_reprisal_eligibility(loser_id, was_defender, DuelType.VOLUNTARY)
        
        if should_reprisal:
            self.state.reprisals_triggered += 1
            self.run_reprisal(loser_id, excluded_gift_id=target.gift_id)
    
    def run_reprisal(self, avenger_id: int, excluded_gift_id: Optional[int] = None):
        """
        Run reprisal duel. Supports chained reprisals based on REPRISAL_MAX_DEPTH.
        - Depth 1 = current behavior (no chaining)
        - Depth 2+ = loser of reprisal can get their own reprisal
        
        V2 SPEC: excluded_gift_id = gift the avenger just lost, cannot be targeted
        """
        # Increment depth
        self.state.reprisal_depth += 1
        current_depth = self.state.reprisal_depth
        
        avenger = self.state.agents[avenger_id]
        
        target = self.agent_reprisal_target(avenger, excluded_gift_id)
        if target is None:
            self.state.reprisal_depth -= 1
            return
        
        target_type, target_id = target
        
        if target_type == 'player':
            other = self.state.agents[target_id]
            if other.gift_id is None:
                self.state.reprisal_depth -= 1
                return
            
            gift = self.state.gift_registry[other.gift_id]
            cost = self.min_cost(gift)
            if avenger.chips < cost:
                self.state.reprisal_depth -= 1
                return
            
            # SPEC: Reprisal cost goes to bank
            avenger.chips -= cost
            avenger.chips_spent += cost
            self.state.bank_chips += cost
            
            depth_str = f" (depth {current_depth})" if self.config.REPRISAL_MAX_DEPTH > 1 else ""
            self.log(f"    REPRISAL{depth_str}: P{avenger_id} challenges P{target_id}")
            self.log_event('REPRISAL', avenger=avenger_id, target=target_id, target_type='player', depth=current_depth)
            contested_gift_id = other.gift_id
            winner_id, loser_id, was_defender = self.execute_duel(avenger_id, target_id, contested_gift_id, 0, DuelType.REPRISAL)
            
            # Check for chained reprisal (if depth allows)
            if self.check_reprisal_eligibility(loser_id, was_defender, DuelType.REPRISAL):
                self.state.reprisals_triggered += 1
                self.run_reprisal(loser_id, excluded_gift_id=contested_gift_id)
        
        elif target_type == 'misfit':
            misfit = self.state.gift_registry[target_id]
            cost = self.min_cost(misfit)
            if avenger.chips < cost:
                self.state.reprisal_depth -= 1
                return
            
            avenger.chips -= cost
            avenger.chips_spent += cost
            self.state.bank_chips += cost
            
            depth_str = f" (depth {current_depth})" if self.config.REPRISAL_MAX_DEPTH > 1 else ""
            self.log(f"    REPRISAL{depth_str}: P{avenger_id} challenges Misfit {target_id}")
            self.log_event('REPRISAL', avenger=avenger_id, target=target_id, target_type='misfit', depth=current_depth)
            
            if self.state.head_elf_id is not None:
                _, _, _ = self.execute_duel(avenger_id, self.state.head_elf_id, target_id, 0, DuelType.MISFIT_CHALLENGE)
            else:
                self.state.misfit_pile.remove(target_id)
                avenger.gift_id = target_id
                misfit.owner_id = avenger_id
        
        # Decrement depth
        self.state.reprisal_depth -= 1
    
    # -------------------------------------------------------------------------
    # ENDGAME (MISFIT LOTTERY)
    # -------------------------------------------------------------------------
    
    def run_endgame_turn(self, ap_id: int) -> bool:
        """Run endgame turn. Returns True if completed."""
        ap = self.state.agents[ap_id]
        ap.in_bag = False
        
        if not self.state.misfit_pile:
            return True
        
        misfit_id = self.state.misfit_pile[-1]
        misfit = self.state.gift_registry[misfit_id]
        
        self.log(f"\nEndgame: P{ap_id} targets Misfit {misfit_id}")
        self.log_event('ENDGAME_TURN', player=ap_id, misfit=misfit_id)
        
        path = self.agent_endgame_path(ap, misfit)
        
        if path == 'A':
            return self.run_endgame_steal(ap_id, misfit_id)
        else:
            return self.run_endgame_auction(ap_id, misfit_id)
    
    def run_endgame_steal(self, ap_id: int, misfit_id: int) -> bool:
        """Path A: Steal from a player."""
        ap = self.state.agents[ap_id]
        misfit = self.state.gift_registry[misfit_id]
        
        # Find best target
        targets = []
        for p, a in self.state.agents.items():
            if p != ap_id and a.gift_id is not None:
                gift = self.state.gift_registry[a.gift_id]
                cost = self.min_cost(gift)
                if ap.chips >= cost:
                    targets.append((p, a, gift, cost))
        
        if not targets:
            # Just take misfit
            self.state.misfit_pile.pop()
            ap.gift_id = misfit_id
            misfit.owner_id = ap_id
            return True
        
        # Choose best value target affordable
        target_id, target, gift, cost = max(targets, key=lambda x: x[2].get_value(ap_id))
        
        # SPEC: Steal cost goes to bank
        ap.chips -= cost
        ap.chips_spent += cost
        self.state.bank_chips += cost
        
        self.log(f"  PATH A (STEAL): P{ap_id} vs P{target_id}")
        self.log_event('ENDGAME_STEAL', attacker=ap_id, defender=target_id)
        
        winner_id = self.resolve_duel(ap_id, target_id)
        self.state.total_duels += 1
        self.state.duels_by_type[DuelType.ENDGAME_STEAL] += 1
        
        self.state.misfit_pile.pop()
        
        if winner_id == ap_id:
            # AP takes gift, defender takes misfit
            victim_gift_id = target.gift_id
            victim_gift = self.state.gift_registry[victim_gift_id]
            
            ap.gift_id = victim_gift_id
            victim_gift.owner_id = ap_id
            victim_gift.naughty_level += 1
            
            target.gift_id = misfit_id
            misfit.owner_id = target_id
            
            # BAG: AP got gift, remove from bag if there
            if ap_id in self.state.bag:
                self.state.bag.remove(ap_id)
            self.log(f"    AP wins! Takes Gift {victim_gift_id}, P{target_id} gets Misfit")
        else:
            # AP takes misfit
            ap.gift_id = misfit_id
            misfit.owner_id = ap_id
            # BAG: AP got gift (misfit), remove from bag if there
            if ap_id in self.state.bag:
                self.state.bag.remove(ap_id)
            self.log(f"    P{target_id} defends! AP takes Misfit")
        
        return True
    
    def run_endgame_auction(self, ap_id: int, misfit_id: int) -> bool:
        """Path B: Auction for misfit."""
        ap = self.state.agents[ap_id]
        misfit = self.state.gift_registry[misfit_id]
        
        # Collect bids from everyone (including players in bag - "snipers")
        bids = []
        for pid, agent in self.state.agents.items():
            if pid == ap_id:
                continue
            # FOMO affects bidding
            if random.random() < agent.fomo:
                bid = self.agent_bid_decision(agent, misfit, is_endgame=True)
                if bid > 0:
                    bids.append((pid, bid, agent.in_bag))  # Track if sniper
        
        bids.sort(key=lambda x: x[1], reverse=True)
        
        if not bids:
            # SPEC: No bids - duel Head Elf
            return self.run_endgame_no_bids(ap_id, misfit_id)
        
        bidder_id, bid_amount, is_sniper = bids[0]
        bidder = self.state.agents[bidder_id]
        
        # Bidder pays into pot
        bidder.chips -= bid_amount
        bidder.chips_spent += bid_amount
        
        self.log(f"  PATH B (AUCTION): P{ap_id} vs P{bidder_id} {'[Sniper]' if is_sniper else '[Swapper]'}")
        self.log_event('ENDGAME_AUCTION', active_player=ap_id, bidder=bidder_id, bid=bid_amount, is_sniper=is_sniper)
        
        winner_id = self.resolve_duel(ap_id, bidder_id)
        self.state.total_duels += 1
        self.state.duels_by_type[DuelType.ENDGAME_AUCTION] += 1
        
        self.state.misfit_pile.pop()
        
        if winner_id == ap_id:
            # SPEC: AP wins - takes misfit, bidder pays pot
            ap.gift_id = misfit_id
            misfit.owner_id = ap_id
            self.state.bank_chips += bid_amount  # Pot goes to bank
            # BAG: AP got gift, remove from bag
            if ap_id in self.state.bag:
                self.state.bag.remove(ap_id)
            self.log(f"    AP wins! Takes Misfit, bidder paid {bid_amount}")
            return True
        else:
            # SPEC: Bidder wins
            if is_sniper or bidder.gift_id is None:
                # Sniper was giftless - they take misfit, AP retries
                bidder.gift_id = misfit_id
                misfit.owner_id = bidder_id
                bidder.in_bag = False
                # Remove bidder from bag if they were in it
                if bidder_id in self.state.bag:
                    self.state.bag.remove(bidder_id)
                self.log(f"    Sniper wins! Takes Misfit, AP must retry")
                return False  # AP retries with next misfit
            else:
                # Swapper had a gift - they take misfit, AP gets their old gift
                old_gift_id = bidder.gift_id
                old_gift = self.state.gift_registry[old_gift_id]
                
                bidder.gift_id = misfit_id
                misfit.owner_id = bidder_id
                
                ap.gift_id = old_gift_id
                old_gift.owner_id = ap_id
                # Remove AP from bag if they were in it (now have gift)
                if ap_id in self.state.bag:
                    self.state.bag.remove(ap_id)
                self.log(f"    Swapper wins! Takes Misfit, AP gets Gift {old_gift_id}")
                return True
    
    def agent_endgame_defender_choice(self, ap: Agent, misfit: Gift) -> Optional[int]:
        """
        V2 SPEC: In endgame Path B with no bids, active player picks a defender.
        
        Strategy: Pick a defender the agent thinks they can beat (low chips, favorable grudge)
        or pick a defender who has a gift they'd want if they lose anyway.
        """
        possible_defenders = []
        for pid, other in self.state.agents.items():
            if pid == ap.player_id:
                continue
            if pid in self.state.bag:  # Giftless players can't defend
                continue
            possible_defenders.append(pid)
        
        if not possible_defenders:
            return None
        
        # Score each defender: prefer players we have grudges against, or weak opponents
        best_defender = None
        best_score = float('-inf')
        
        for pid in possible_defenders:
            other = self.state.agents[pid]
            
            # Score factors:
            # - Low chips = easier target
            # - High grudge = revenge motivation
            # - We don't want them much = less painful if we lose
            grudge = ap.grudge.get(pid, 0)
            chip_factor = 10 - other.chips  # Prefer low-chip opponents
            
            score = grudge * self.config.REVENGE_WEIGHT + chip_factor * 0.5
            
            # Random tiebreaker
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_defender = pid
        
        return best_defender
    
    def run_endgame_no_bids(self, ap_id: int, misfit_id: int) -> bool:
        """V2 SPEC: Endgame with no bids - active player picks a defender."""
        ap = self.state.agents[ap_id]
        misfit = self.state.gift_registry[misfit_id]
        
        self.state.misfit_pile.pop()
        
        # V2 SPEC: Active player picks a defender
        chosen_defender_id = self.agent_endgame_defender_choice(ap, misfit)
        
        if chosen_defender_id is None:
            # No valid defenders - AP gets misfit free
            ap.gift_id = misfit_id
            misfit.owner_id = ap_id
            if ap_id in self.state.bag:
                self.state.bag.remove(ap_id)
            self.log(f"  No bids, no defenders - P{ap_id} gets Misfit FREE")
            return True
        
        self.log(f"  No bids - P{ap_id} chooses P{chosen_defender_id} as defender")
        
        winner_id = self.resolve_duel(ap_id, chosen_defender_id)
        self.state.total_duels += 1
        self.state.duels_by_type[DuelType.HEAD_ELF_DEFENSE] += 1
        
        # V2 SPEC: AP always gets misfit, but pays toll if loses
        ap.gift_id = misfit_id
        misfit.owner_id = ap_id
        
        # BAG: AP got gift, remove from bag
        if ap_id in self.state.bag:
            self.state.bag.remove(ap_id)
        
        if winner_id == ap_id:
            self.log(f"    AP wins! Gets Misfit FREE")
        else:
            # V2 SPEC: Pay 1 chip toll to the CHOSEN defender
            if ap.chips >= 1:
                ap.chips -= 1
                self.state.agents[chosen_defender_id].chips += 1
                self.log(f"    AP loses! Pays 1 chip toll to P{chosen_defender_id}")
            else:
                self.log(f"    AP loses but broke - no toll paid")
        
        return True
    
    # -------------------------------------------------------------------------
    # MAIN GAME LOOP
    # -------------------------------------------------------------------------
    
    def run_game(self) -> GameState:
        """Run a complete game."""
        if self.state is None:
            self.reset()
        
        self.log("=" * 60)
        self.log(f"JINGLE BRAWL - {self.config.NUM_PLAYERS} players, {self.config.NUM_GIFTS} gifts")
        self.log(f"Config: Tax={self.config.SANTA_TAX_RULE.value}, Dividend={self.config.LOSER_DIVIDEND_RULE.value}")
        self.log("=" * 60)
        
        # Main phase
        while self.state.wrapped_gifts:
            if not self.state.bag:
                # Refill bag with players who haven't gone
                self.state.bag = [p for p in self.state.agents.keys() 
                                  if self.state.agents[p].in_bag]
                random.shuffle(self.state.bag)
            
            if not self.state.bag:
                break
            
            opener_id = self.state.bag.pop()
            self.state.turn_index += 1
            self.run_main_phase_turn(opener_id)
        
        # Endgame
        self.log("\n" + "=" * 60)
        self.log("ENDGAME: MISFIT LOTTERY")
        self.log("=" * 60)
        
        self.state.turn_phase = TurnPhase.MISFIT_LOTTERY
        sniper_retry_count = defaultdict(int)
        
        while self.state.bag and self.state.misfit_pile:
            ap_id = self.state.bag.pop()
            self.state.turn_index += 1
            
            # Infinite loop protection (mercy rule)
            if sniper_retry_count[ap_id] >= 3:
                misfit_id = self.state.misfit_pile.pop()
                misfit = self.state.gift_registry[misfit_id]
                self.state.agents[ap_id].gift_id = misfit_id
                misfit.owner_id = ap_id
                self.log(f"  Mercy rule: P{ap_id} takes Misfit {misfit_id}")
                continue
            
            completed = self.run_endgame_turn(ap_id)
            if not completed:
                sniper_retry_count[ap_id] += 1
                self.state.bag.insert(0, ap_id)
        
        # Final cleanup - distribute remaining misfits
        while self.state.misfit_pile:
            giftless = [p for p, a in self.state.agents.items() if a.gift_id is None]
            if giftless:
                pid = giftless[0]
                misfit_id = self.state.misfit_pile.pop()
                self.state.agents[pid].gift_id = misfit_id
                self.state.gift_registry[misfit_id].owner_id = pid
            else:
                break
        
        self.log("\n" + "=" * 60)
        giftless = sum(1 for a in self.state.agents.values() if a.gift_id is None)
        self.log(f"GAME END - Giftless: {giftless}")
        self.log("=" * 60)
        
        return self.state


# =============================================================================
# METRICS & ANALYSIS
# =============================================================================

@dataclass
class GameMetrics:
    """Comprehensive success metrics for a game."""
    # Core metrics
    velocity: float           # Total duels
    liquidity: float          # Avg chips at endgame
    hoarding: float           # % with max chips + bad gifts
    bankruptcy: float         # % bankruptcy events per duel
    duration: int             # Total turns
    giftless_rate: float      # % players without gifts
    avg_gift_value: float     # Avg final gift value
    
    # Economy metrics
    avg_bid: float            # Average bid amount
    total_bids: int           # Number of bids placed
    taxes_collected: int      # Total Santa Tax
    dividends_paid: int       # Total dividends
    bank_final: int           # Final bank balance
    
    # Engagement metrics
    yields: int               # Number of yields
    voluntary_challenges: int # Number of voluntary challenges
    reprisals: int           # Number of reprisals triggered
    misfit_count: int        # Final misfits
    
    # Duel breakdown
    sealed_bid_duels: int
    tie_break_duels: int
    voluntary_duels: int
    reprisal_duels: int
    yield_duels: int
    misfit_duels: int
    endgame_steals: int
    endgame_auctions: int
    head_elf_duels: int
    
    def to_dict(self) -> dict:
        return {
            'velocity': self.velocity,
            'liquidity': self.liquidity,
            'hoarding': self.hoarding,
            'bankruptcy': self.bankruptcy,
            'duration': self.duration,
            'giftless_rate': self.giftless_rate,
            'avg_gift_value': self.avg_gift_value,
            'avg_bid': self.avg_bid,
            'total_bids': self.total_bids,
            'taxes_collected': self.taxes_collected,
            'dividends_paid': self.dividends_paid,
            'bank_final': self.bank_final,
            'yields': self.yields,
            'voluntary_challenges': self.voluntary_challenges,
            'reprisals': self.reprisals,
            'misfit_count': self.misfit_count,
            'sealed_bid_duels': self.sealed_bid_duels,
            'tie_break_duels': self.tie_break_duels,
            'voluntary_duels': self.voluntary_duels,
            'reprisal_duels': self.reprisal_duels,
            'yield_duels': self.yield_duels,
            'misfit_duels': self.misfit_duels,
            'endgame_steals': self.endgame_steals,
            'endgame_auctions': self.endgame_auctions,
            'head_elf_duels': self.head_elf_duels,
        }


def compute_metrics(state: GameState) -> GameMetrics:
    """Compute comprehensive metrics from game state."""
    num_players = state.config.NUM_PLAYERS
    
    # Velocity
    velocity = state.total_duels
    
    # Liquidity
    if state.chips_history:
        endgame_idx = min(state.config.NUM_GIFTS, len(state.chips_history) - 1)
        endgame_chips = state.chips_history[endgame_idx]
        liquidity = np.mean(list(endgame_chips.values()))
    else:
        liquidity = state.config.INITIAL_CHIPS
    
    # Hoarding
    max_chips = state.config.INITIAL_CHIPS + 5
    hoarding_count = 0
    for agent in state.agents.values():
        if agent.chips >= max_chips * 0.8:
            if agent.gift_id is not None:
                gift = state.gift_registry[agent.gift_id]
                value = gift.get_value(agent.player_id)
                if value < 4:
                    hoarding_count += 1
    hoarding = hoarding_count / num_players
    
    # Bankruptcy
    bankruptcy = state.bankruptcy_events / max(1, state.total_duels)
    
    # Duration
    duration = state.turn_index
    
    # Giftless
    giftless = sum(1 for a in state.agents.values() if a.gift_id is None)
    giftless_rate = giftless / num_players
    
    # Avg gift value
    values = []
    for agent in state.agents.values():
        if agent.gift_id is not None:
            gift = state.gift_registry[agent.gift_id]
            values.append(gift.get_value(agent.player_id))
    avg_gift_value = np.mean(values) if values else 0.0
    
    # Economy
    avg_bid = np.mean(state.bid_amounts) if state.bid_amounts else 0.0
    
    return GameMetrics(
        velocity=velocity,
        liquidity=liquidity,
        hoarding=hoarding,
        bankruptcy=bankruptcy,
        duration=duration,
        giftless_rate=giftless_rate,
        avg_gift_value=avg_gift_value,
        avg_bid=avg_bid,
        total_bids=state.total_bids,
        taxes_collected=state.taxes_collected,
        dividends_paid=state.dividends_paid,
        bank_final=state.bank_chips,
        yields=state.yields_count,
        voluntary_challenges=state.voluntary_challenges,
        reprisals=state.reprisals_triggered,
        misfit_count=len(state.misfit_pile),
        sealed_bid_duels=state.duels_by_type[DuelType.SEALED_BID],
        tie_break_duels=state.duels_by_type[DuelType.TIE_BREAK],
        voluntary_duels=state.duels_by_type[DuelType.VOLUNTARY],
        reprisal_duels=state.duels_by_type[DuelType.REPRISAL],
        yield_duels=state.duels_by_type[DuelType.YIELD_DUEL],
        misfit_duels=state.duels_by_type[DuelType.MISFIT_CHALLENGE],
        endgame_steals=state.duels_by_type[DuelType.ENDGAME_STEAL],
        endgame_auctions=state.duels_by_type[DuelType.ENDGAME_AUCTION],
        head_elf_duels=state.duels_by_type[DuelType.HEAD_ELF_DEFENSE],
    )


# =============================================================================
# LARGE-SCALE SIMULATION
# =============================================================================

def run_simulation_batch(config: GameConfig, num_games: int, 
                         progress_interval: int = 100) -> List[GameMetrics]:
    """Run a batch of games and collect metrics."""
    metrics_list = []
    start_time = time.time()
    
    for i in range(num_games):
        env = JingleBrawlEnv(config)
        env.reset()
        state = env.run_game()
        metrics = compute_metrics(state)
        metrics_list.append(metrics)
        
        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{num_games} games ({rate:.1f} games/sec)")
    
    return metrics_list


def aggregate_metrics(metrics_list: List[GameMetrics]) -> dict:
    """Aggregate metrics across multiple games."""
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].to_dict().keys()
    result = {}
    
    for key in keys:
        values = [m.to_dict()[key] for m in metrics_list]
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)
        result[f'{key}_min'] = np.min(values)
        result[f'{key}_max'] = np.max(values)
    
    return result


def run_permutation_test(configs: List[GameConfig], num_games: int = 100) -> List[dict]:
    """Run games with different configs and compare metrics."""
    results = []
    
    for config in configs:
        print(f"\nTesting: {config.short_name()}")
        
        metrics_list = run_simulation_batch(config, num_games, progress_interval=50)
        
        agg = aggregate_metrics(metrics_list)
        agg['config'] = config.short_name()
        agg['config_dict'] = config.to_dict()
        results.append(agg)
        
        print(f"  Velocity: {agg['velocity_mean']:.1f} ± {agg['velocity_std']:.1f}")
        print(f"  Liquidity: {agg['liquidity_mean']:.2f}")
        print(f"  Bankruptcy: {agg['bankruptcy_mean']:.2%}")
        print(f"  Reprisals: {agg['reprisals_mean']:.1f}")
    
    return results


# =============================================================================
# TEST SCENARIOS (based on raw rules)
# =============================================================================

def run_stress_tests(num_games: int = 200):
    """Run comprehensive stress tests based on the raw rules."""
    print("=" * 70)
    print("STRESS TESTS (7 players)")
    print("=" * 70)
    
    # 1. DIVIDEND MECHANICS TEST
    print("\n1. LOSER'S DIVIDEND MECHANICS")
    print("-" * 50)
    print("Testing: Defender-only vs Any-loser vs None")
    
    for rule in LoserDividendRule:
        config = GameConfig(
            LOSER_DIVIDEND_RULE=rule,
            NUM_PLAYERS=7,
            NUM_GIFTS=7,
        )
        metrics = run_simulation_batch(config, num_games, progress_interval=0)
        agg = aggregate_metrics(metrics)
        print(f"  {rule.value}:")
        print(f"    Dividends paid: {agg['dividends_paid_mean']:.1f}")
        print(f"    Avg final chips: {agg['liquidity_mean']:.2f}")
        print(f"    Bankruptcies: {agg['bankruptcy_mean']:.2%}")
    
    # 2. SANTA TAX VARIATIONS
    print("\n2. SANTA TAX VARIATIONS")
    print("-" * 50)
    
    for rule in SantaTaxRule:
        config = GameConfig(
            SANTA_TAX_RULE=rule,
            NUM_PLAYERS=7,
            NUM_GIFTS=7,
        )
        metrics = run_simulation_batch(config, num_games, progress_interval=0)
        agg = aggregate_metrics(metrics)
        print(f"  {rule.value}:")
        print(f"    Taxes collected: {agg['taxes_collected_mean']:.1f}")
        print(f"    Bank final: {agg['bank_final_mean']:.1f}")
        print(f"    Avg bid: {agg['avg_bid_mean']:.2f}")
    
    # 3. INITIAL CHIPS SENSITIVITY
    print("\n3. INITIAL CHIPS SENSITIVITY")
    print("-" * 50)
    
    for chips in [3, 5, 7, 10]:
        config = GameConfig(
            INITIAL_CHIPS=chips,
            NUM_PLAYERS=7,
            NUM_GIFTS=7,
        )
        metrics = run_simulation_batch(config, num_games, progress_interval=0)
        agg = aggregate_metrics(metrics)
        print(f"  {chips} starting chips:")
        print(f"    Duels: {agg['velocity_mean']:.1f}")
        print(f"    Voluntary challenges: {agg['voluntary_challenges_mean']:.1f}")
        print(f"    Reprisals: {agg['reprisals_mean']:.1f}")
    
    # 4. VOLUNTARY CHALLENGE IMPACT
    print("\n4. VOLUNTARY CHALLENGE & REPRISAL TEST")
    print("-" * 50)
    
    config_enabled = GameConfig(
        VOLUNTARY_CHALLENGE_RULE=VoluntaryChallengeRule.ENABLED,
        REPRISAL_TARGETS_MISFITS=True,
        NUM_PLAYERS=7,
        NUM_GIFTS=7,
    )
    config_disabled = GameConfig(
        VOLUNTARY_CHALLENGE_RULE=VoluntaryChallengeRule.DISABLED,
        REPRISAL_TARGETS_MISFITS=True,
        NUM_PLAYERS=7,
        NUM_GIFTS=7,
    )
    
    m_enabled = run_simulation_batch(config_enabled, num_games, progress_interval=0)
    m_disabled = run_simulation_batch(config_disabled, num_games, progress_interval=0)
    
    agg_en = aggregate_metrics(m_enabled)
    agg_dis = aggregate_metrics(m_disabled)
    
    print(f"  Voluntary ENABLED:")
    print(f"    Total duels: {agg_en['velocity_mean']:.1f}")
    print(f"    Voluntary challenges: {agg_en['voluntary_challenges_mean']:.1f}")
    print(f"    Reprisals: {agg_en['reprisals_mean']:.1f}")
    print(f"  Voluntary DISABLED:")
    print(f"    Total duels: {agg_dis['velocity_mean']:.1f}")
    print(f"    Voluntary challenges: {agg_dis['voluntary_challenges_mean']:.1f}")
    print(f"    Reprisals: {agg_dis['reprisals_mean']:.1f}")
    
    # 5. YIELD RULE IMPACT
    print("\n5. YIELD RULE TEST")
    print("-" * 50)
    
    config_yield = GameConfig(YIELD_RULE=YieldRule.ALLOWED, NUM_PLAYERS=7, NUM_GIFTS=7)
    config_no_yield = GameConfig(YIELD_RULE=YieldRule.DISABLED, NUM_PLAYERS=7, NUM_GIFTS=7)
    
    m_yield = run_simulation_batch(config_yield, num_games, progress_interval=0)
    m_no_yield = run_simulation_batch(config_no_yield, num_games, progress_interval=0)
    
    agg_y = aggregate_metrics(m_yield)
    agg_ny = aggregate_metrics(m_no_yield)
    
    print(f"  Yield ALLOWED:")
    print(f"    Yields: {agg_y['yields_mean']:.1f}")
    print(f"    Yield duels: {agg_y['yield_duels_mean']:.1f}")
    print(f"  Yield DISABLED:")
    print(f"    Yields: {agg_ny['yields_mean']:.1f}")
    print(f"    Yield duels: {agg_ny['yield_duels_mean']:.1f}")
    
    # 6. NAUGHTY LEVEL CAP
    print("\n6. NAUGHTY LEVEL CAP TEST")
    print("-" * 50)
    
    for cap in [2, 3, 5]:
        config = GameConfig(NAUGHTY_LEVEL_CAP=cap, NUM_PLAYERS=7, NUM_GIFTS=7)
        metrics = run_simulation_batch(config, num_games, progress_interval=0)
        agg = aggregate_metrics(metrics)
        print(f"  Cap at {cap}:")
        print(f"    Avg bid: {agg['avg_bid_mean']:.2f}")
        print(f"    Duels: {agg['velocity_mean']:.1f}")
    
    # 7. ENDGAME MECHANICS
    print("\n7. ENDGAME STRESS TEST")
    print("-" * 50)
    
    config = GameConfig(NUM_PLAYERS=7, NUM_GIFTS=7)
    metrics = run_simulation_batch(config, num_games, progress_interval=0)
    agg = aggregate_metrics(metrics)
    
    print(f"  Endgame Steals: {agg['endgame_steals_mean']:.1f}")
    print(f"  Endgame Auctions: {agg['endgame_auctions_mean']:.1f}")
    print(f"  Head Elf Duels: {agg['head_elf_duels_mean']:.1f}")
    print(f"  Final Misfits: {agg['misfit_count_mean']:.1f}")
    print(f"  Giftless Rate: {agg['giftless_rate_mean']:.2%}")
    
    # 8. INFINITE LOOP CHECK
    print("\n8. INFINITE LOOP PROTECTION CHECK")
    print("-" * 50)
    
    max_turns = []
    for _ in range(num_games):
        env = JingleBrawlEnv(GameConfig(NUM_PLAYERS=7, NUM_GIFTS=7))
        state = env.run_game()
        max_turns.append(state.turn_index)
    
    print(f"  Max turns observed: {max(max_turns)}")
    print(f"  Avg turns: {np.mean(max_turns):.1f}")
    print(f"  Infinite loop protection: {'✅ PASSED' if max(max_turns) < 50 else '❌ FAILED'}")


# =============================================================================
# DEEP RL TRAINING
# =============================================================================

if TORCH_AVAILABLE:
    
    class PolicyNet(nn.Module):
        def __init__(self, input_size=24, hidden=128):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 64)
            self.bid_head = nn.Linear(64, 4)  # 0,1,2,3
            self.yield_head = nn.Linear(64, 2)
            self.path_head = nn.Linear(64, 2)  # A, B
            self.target_head = nn.Linear(64, 8)  # Target selection
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return {
                'bid': F.softmax(self.bid_head(x), dim=-1),
                'yield': F.softmax(self.yield_head(x), dim=-1),
                'path': F.softmax(self.path_head(x), dim=-1),
                'target': F.softmax(self.target_head(x), dim=-1),
            }


class DeepRLTrainer:
    """Self-play deep RL trainer."""
    
    def __init__(self, config: GameConfig, lr: float = 1e-3):
        self.config = config
        self.lr = lr
        
        if TORCH_AVAILABLE:
            self.policy = PolicyNet()
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.metrics_history = []
        self.best_velocity = 0
        self.best_config = None
    
    def compute_reward(self, agent: Agent, state: GameState) -> float:
        """Reward function."""
        reward = 0.0
        
        if agent.gift_id is not None:
            gift = state.gift_registry[agent.gift_id]
            value = gift.get_value(agent.player_id)
            
            if value > 7:
                reward += 10
            elif value > 5:
                reward += 5
            else:
                reward -= 2
        else:
            reward -= 10  # Strong penalty for giftless
        
        # Chip bonus
        reward += agent.chips * 0.1
        
        # Duel participation bonus
        reward += (agent.duels_won + agent.duels_lost) * 0.5
        
        return reward
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 100):
        """Train via self-play."""
        print(f"Training for {num_episodes} episodes...")
        print(f"PyTorch available: {TORCH_AVAILABLE}")
        print(f"Config: {self.config.NUM_PLAYERS} players, {self.config.NUM_GIFTS} gifts")
        
        for ep in range(num_episodes):
            env = JingleBrawlEnv(self.config)
            state = env.run_game()
            metrics = compute_metrics(state)
            
            # Track best velocity
            if metrics.velocity > self.best_velocity:
                self.best_velocity = metrics.velocity
                self.best_config = self.config.to_dict()
            
            self.metrics_history.append({
                'episode': ep,
                'velocity': metrics.velocity,
                'liquidity': metrics.liquidity,
                'avg_gift_value': metrics.avg_gift_value,
                'reprisals': metrics.reprisals,
                'voluntary': metrics.voluntary_challenges,
                'giftless_rate': metrics.giftless_rate,
            })
            
            if (ep + 1) % eval_interval == 0:
                recent = self.metrics_history[-eval_interval:]
                avg_vel = np.mean([m['velocity'] for m in recent])
                avg_val = np.mean([m['avg_gift_value'] for m in recent])
                avg_rep = np.mean([m['reprisals'] for m in recent])
                print(f"Episode {ep+1}: Velocity={avg_vel:.1f}, AvgGift={avg_val:.2f}, Reprisals={avg_rep:.1f}")
        
        print(f"\nBest velocity: {self.best_velocity:.1f}")
        return self.metrics_history


# =============================================================================
# CLI INTERFACE
# =============================================================================

def cli_simulate(config: GameConfig, num_games: int = 1000, output: str = None):
    """Run simulation with given config."""
    print("=" * 70)
    print("JINGLE BRAWL SIMULATION")
    print(f"Config: {config.short_name()}")
    print(f"Games: {num_games}")
    print("=" * 70)
    
    results = []
    for i in range(num_games):
        env = JingleBrawlEnv(config)
        state = env.run_game()
        metrics = compute_metrics(state)
        results.append(metrics.to_dict())
        
        if (i + 1) % (num_games // 10) == 0:
            print(f"  Progress: {i+1}/{num_games}")
    
    # Aggregate results
    agg = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        agg[f'{key}_mean'] = np.mean(values)
        agg[f'{key}_std'] = np.std(values)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Duels per game:  {agg['velocity_mean']:.1f} ± {agg['velocity_std']:.1f}")
    print(f"  Reprisals:       {agg['reprisals_mean']:.1f}")
    print(f"  Voluntary:       {agg['voluntary_challenges_mean']:.1f}")
    print(f"  Avg gift value:  {agg['avg_gift_value_mean']:.2f}")
    print(f"  Bankruptcy rate: {agg['bankruptcy_mean']:.1%}")
    print(f"  Avg final chips: {agg['liquidity_mean']:.1f}")
    
    # Save if output specified
    if output:
        with open(output, 'w') as f:
            json.dump({
                'config': config.to_dict(),
                'num_games': num_games,
                'aggregated': agg,
                'raw': results
            }, f, indent=2)
        print(f"\n✅ Results saved to {output}")
    
    return agg


def cli_sweep(sweep_config: dict, games_per_config: int = 500, output: str = None):
    """Run parameter sweep."""
    print("=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)
    
    from itertools import product
    
    base = sweep_config.get('base', {})
    sweeps = sweep_config.get('sweeps', {})
    
    # Generate all combinations
    param_names = list(sweeps.keys())
    param_values = [sweeps[k] for k in param_names]
    
    all_results = []
    total_configs = 1
    for v in param_values:
        total_configs *= len(v)
    
    print(f"Testing {total_configs} configurations with {games_per_config} games each...")
    
    config_idx = 0
    for combo in product(*param_values):
        config_dict = {**base}
        for name, value in zip(param_names, combo):
            config_dict[name] = value
        
        # Ensure NUM_GIFTS matches NUM_PLAYERS
        if 'NUM_PLAYERS' in config_dict:
            config_dict['NUM_GIFTS'] = config_dict['NUM_PLAYERS']
        
        config = GameConfig.from_dict(config_dict)
        config_idx += 1
        print(f"\n[{config_idx}/{total_configs}] {config.short_name()}")
        
        metrics_list = []
        for _ in range(games_per_config):
            env = JingleBrawlEnv(config)
            state = env.run_game()
            metrics = compute_metrics(state)
            metrics_list.append(metrics.to_dict())
        
        agg = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            agg[f'{key}_mean'] = np.mean(values)
            agg[f'{key}_std'] = np.std(values)
        
        all_results.append({
            'config': config.to_dict(),
            'aggregated': agg,
        })
        
        print(f"  Duels: {agg['velocity_mean']:.1f}, Bankruptcy: {agg['bankruptcy_mean']:.1%}")
    
    # Save results
    if output:
        with open(output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Sweep results saved to {output}")
    
    return all_results


def cli_demo(verbose: bool = True):
    """Run a single demo game."""
    print("=" * 70)
    print("JINGLE BRAWL DEMO GAME")
    print("=" * 70)
    
    config = GameConfig(NUM_PLAYERS=7, NUM_GIFTS=7)
    env = JingleBrawlEnv(config, verbose=verbose)
    state = env.run_game()
    metrics = compute_metrics(state)
    
    print(f"\n--- GAME SUMMARY ---")
    print(f"Total Duels: {metrics.velocity}")
    print(f"  Sealed Bids: {metrics.sealed_bid_duels}")
    print(f"  Voluntary: {metrics.voluntary_duels}")
    print(f"  Reprisals: {metrics.reprisal_duels}")
    print(f"  Yields: {metrics.yield_duels}")
    print(f"  Endgame: {metrics.endgame_steals + metrics.endgame_auctions + metrics.head_elf_duels}")
    print(f"Avg Gift Value: {metrics.avg_gift_value:.2f}")
    print(f"Final Bank: {metrics.bank_final}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Jingle Brawl Optimization Environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo game (verbose)
  python jingle_brawl_env.py demo
  
  # Simulate with config file
  python jingle_brawl_env.py simulate --config configs/baseline_7p.yaml --games 1000
  
  # Run parameter sweep
  python jingle_brawl_env.py sweep --config configs/sweep.yaml --out results/sweep.json
  
  # Run stress tests
  python jingle_brawl_env.py stress --games 500
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a demo game')
    demo_parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('--config', '-c', type=str, help='YAML config file')
    sim_parser.add_argument('--games', '-n', type=int, default=1000, help='Number of games')
    sim_parser.add_argument('--out', '-o', type=str, help='Output JSON file')
    sim_parser.add_argument('--players', '-p', type=int, help='Override player count')
    sim_parser.add_argument('--chips', type=int, help='Override initial chips')
    sim_parser.add_argument('--seed', '-s', type=int, help='Random seed')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweep')
    sweep_parser.add_argument('--config', '-c', type=str, help='Sweep config YAML')
    sweep_parser.add_argument('--games', '-n', type=int, default=500, help='Games per config')
    sweep_parser.add_argument('--out', '-o', type=str, help='Output JSON file')
    
    # Stress test command
    stress_parser = subparsers.add_parser('stress', help='Run stress tests')
    stress_parser.add_argument('--games', '-n', type=int, default=200, help='Games per test')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train DRL agents')
    train_parser.add_argument('--episodes', '-e', type=int, default=1000, help='Training episodes')
    train_parser.add_argument('--config', '-c', type=str, help='YAML config file')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        cli_demo(verbose=not args.quiet)
    
    elif args.command == 'simulate':
        # Load config
        if args.config:
            config = GameConfig.from_yaml(args.config)
        else:
            config = GameConfig()
        
        # Apply overrides
        if args.players:
            config.NUM_PLAYERS = args.players
            config.NUM_GIFTS = args.players
        if args.chips:
            config.INITIAL_CHIPS = args.chips
        if args.seed:
            config.SEED = args.seed
        
        cli_simulate(config, num_games=args.games, output=args.out)
    
    elif args.command == 'sweep':
        if not args.config:
            print("Error: --config required for sweep")
            return
        
        if not YAML_AVAILABLE:
            print("Error: PyYAML not installed. Run: pip install pyyaml")
            return
        
        with open(args.config) as f:
            sweep_config = yaml.safe_load(f)
        
        cli_sweep(sweep_config, games_per_config=args.games, output=args.out)
    
    elif args.command == 'stress':
        run_stress_tests(num_games=args.games)
    
    elif args.command == 'train':
        if args.config:
            config = GameConfig.from_yaml(args.config)
        else:
            config = GameConfig()
        
        trainer = DeepRLTrainer(config)
        trainer.train(num_episodes=args.episodes)
    
    else:
        # Default: run demo + stress tests
        cli_demo(verbose=True)
        run_stress_tests(num_games=200)


if __name__ == "__main__":
    main()
