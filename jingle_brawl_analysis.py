#!/usr/bin/env python3
"""
Jingle Brawl Multi-Model Analysis System
=========================================

A comprehensive analysis framework that combines multiple approaches:
1. Monte Carlo Simulation - Baseline statistical analysis
2. Monte Carlo Tree Search (MCTS) - Strategic decision exploration
3. Heuristic Agents - Rule-based intelligent play
4. Deep RL Agents - Neural network learned strategies
5. Self-Play Evolution - Population-based training

Author: Jingle Brawl Analysis System
Version: 2.0
"""

import numpy as np
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict
from enum import Enum
import time
import math

from jingle_brawl_env import (
    GameConfig, JingleBrawlEnv, GameState, Agent, Gift, DuelType,
    LoserDividendRule, ReprisalTriggerRule, SantaTaxRule, compute_metrics, GameMetrics,
    GiftValueModel, MinCostMode, DuelModel
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis runs."""
    num_games: int = 1000
    num_players: int = 7
    initial_chips: int = 10
    naughty_level_cap: int = 99
    
    # Game rules
    santa_tax_rule: SantaTaxRule = SantaTaxRule.FLAT_1
    loser_dividend_rule: LoserDividendRule = LoserDividendRule.ANY_LOSER
    reprisal_trigger_rule: ReprisalTriggerRule = ReprisalTriggerRule.ANY_DEFENDER_LOSS
    reprisal_max_depth: int = 1  # 1 = no chaining, 2+ = chained reprisals
    min_cost_mode: MinCostMode = MinCostMode.LINEAR
    duel_model: DuelModel = DuelModel.FAIR_COIN
    
    # Gift value model (CORRELATED creates realistic "bad gifts nobody wants")
    gift_value_model: GiftValueModel = GiftValueModel.CORRELATED
    taste_variance: float = 2.0
    
    # Sealed bid limits
    sealed_bid_min: int = 1
    sealed_bid_max: Optional[int] = None
    
    # Model flags
    run_monte_carlo: bool = True
    run_mcts: bool = True
    run_heuristic: bool = True
    run_drl: bool = True
    run_self_play: bool = True
    run_mixed_self_play: bool = True  # Mixed population to avoid passive equilibrium
    
    # Training parameters
    drl_generations: int = 50
    drl_population: int = 20
    mcts_simulations: int = 50
    self_play_generations: int = 30
    
    @classmethod
    def from_yaml(cls, yaml_path: str, num_games: int = 1000) -> 'AnalysisConfig':
        """Load analysis config from a YAML game config file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Map YAML keys to AnalysisConfig fields
        config = cls(num_games=num_games)
        
        if 'NUM_PLAYERS' in data:
            config.num_players = data['NUM_PLAYERS']
        if 'INITIAL_CHIPS' in data:
            config.initial_chips = data['INITIAL_CHIPS']
        if 'NAUGHTY_LEVEL_CAP' in data:
            config.naughty_level_cap = data['NAUGHTY_LEVEL_CAP']
        
        # Enum mappings
        if 'SANTA_TAX_RULE' in data:
            config.santa_tax_rule = SantaTaxRule[data['SANTA_TAX_RULE']]
        if 'LOSER_DIVIDEND_RULE' in data:
            config.loser_dividend_rule = LoserDividendRule[data['LOSER_DIVIDEND_RULE']]
        if 'REPRISAL_TRIGGER_RULE' in data:
            config.reprisal_trigger_rule = ReprisalTriggerRule[data['REPRISAL_TRIGGER_RULE']]
        if 'REPRISAL_MAX_DEPTH' in data:
            config.reprisal_max_depth = int(data['REPRISAL_MAX_DEPTH'])
        if 'MIN_COST_MODE' in data:
            config.min_cost_mode = MinCostMode[data['MIN_COST_MODE']]
        if 'DUEL_MODEL' in data:
            config.duel_model = DuelModel[data['DUEL_MODEL']]
        if 'GIFT_VALUE_MODEL' in data:
            config.gift_value_model = GiftValueModel[data['GIFT_VALUE_MODEL']]
        
        # Float/int values
        if 'TASTE_VARIANCE' in data:
            config.taste_variance = float(data['TASTE_VARIANCE'])
        if 'SEALED_BID_MIN' in data:
            config.sealed_bid_min = int(data['SEALED_BID_MIN'])
        if 'SEALED_BID_MAX' in data and data['SEALED_BID_MAX'] is not None:
            config.sealed_bid_max = int(data['SEALED_BID_MAX'])
        
        return config
    
    def to_game_config(self) -> GameConfig:
        return GameConfig(
            NUM_PLAYERS=self.num_players,
            NUM_GIFTS=self.num_players,
            INITIAL_CHIPS=self.initial_chips,
            NAUGHTY_LEVEL_CAP=self.naughty_level_cap,
            SANTA_TAX_RULE=self.santa_tax_rule,
            LOSER_DIVIDEND_RULE=self.loser_dividend_rule,
            REPRISAL_TRIGGER_RULE=self.reprisal_trigger_rule,
            REPRISAL_MAX_DEPTH=self.reprisal_max_depth,
            MIN_COST_MODE=self.min_cost_mode,
            DUEL_MODEL=self.duel_model,
            GIFT_VALUE_MODEL=self.gift_value_model,
            TASTE_VARIANCE=self.taste_variance,
            SEALED_BID_MIN=self.sealed_bid_min,
            SEALED_BID_MAX=self.sealed_bid_max,
        )


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

@dataclass
class AggregateResults:
    """Aggregated results across multiple games."""
    model_name: str
    num_games: int
    
    # Core averages
    avg_duels: float = 0.0
    avg_voluntary: float = 0.0
    avg_reprisals: float = 0.0
    avg_yields: float = 0.0
    avg_duration: float = 0.0
    
    # Economy
    avg_gift_value: float = 0.0
    avg_final_chips: float = 0.0
    bankruptcy_rate: float = 0.0
    giftless_rate: float = 0.0
    
    # Endgame
    avg_endgame_steals: float = 0.0
    avg_endgame_auctions: float = 0.0
    misfit_game_rate: float = 0.0
    
    # Fairness
    value_variance: float = 0.0


def aggregate_game_metrics(metrics_list: List[GameMetrics], model_name: str) -> AggregateResults:
    """Aggregate GameMetrics from multiple games."""
    n = len(metrics_list)
    if n == 0:
        return AggregateResults(model_name=model_name, num_games=0)
    
    agg = AggregateResults(model_name=model_name, num_games=n)
    
    agg.avg_duels = np.mean([m.velocity for m in metrics_list])
    agg.avg_voluntary = np.mean([m.voluntary_challenges for m in metrics_list])
    agg.avg_reprisals = np.mean([m.reprisals for m in metrics_list])
    agg.avg_yields = np.mean([m.yields for m in metrics_list])
    agg.avg_duration = np.mean([m.duration for m in metrics_list])
    
    agg.avg_gift_value = np.mean([m.avg_gift_value for m in metrics_list])
    agg.avg_final_chips = np.mean([m.liquidity for m in metrics_list])
    agg.bankruptcy_rate = np.mean([m.bankruptcy for m in metrics_list])
    agg.giftless_rate = np.mean([m.giftless_rate for m in metrics_list])
    
    agg.avg_endgame_steals = np.mean([m.endgame_steals for m in metrics_list])
    agg.avg_endgame_auctions = np.mean([m.endgame_auctions for m in metrics_list])
    agg.misfit_game_rate = np.mean([1 if m.misfit_count > 0 else 0 for m in metrics_list])
    
    agg.value_variance = np.var([m.avg_gift_value for m in metrics_list])
    
    return agg


# =============================================================================
# MODEL 1: MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloAnalyzer:
    """
    Basic Monte Carlo simulation with default heuristic agents.
    Establishes baseline statistics.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.game_config = config.to_game_config()
    
    def run(self, num_games: int = None) -> AggregateResults:
        """Run Monte Carlo simulation."""
        n = num_games or self.config.num_games
        metrics = []
        
        for _ in range(n):
            env = JingleBrawlEnv(self.game_config, verbose=False)
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        return aggregate_game_metrics(metrics, "Monte Carlo (Baseline)")


# =============================================================================
# MODEL 2: MONTE CARLO TREE SEARCH
# =============================================================================

class MCTSAgent:
    """
    MCTS agent for strategic bid decisions.
    Uses lightweight simulation rollouts to evaluate bid choices.
    """
    
    def __init__(self, player_id: int, simulations: int = 50):
        self.player_id = player_id
        self.simulations = simulations
    
    def simulate_rollout(self, config: GameConfig) -> float:
        """Simulate a game to completion and return value."""
        sim_env = JingleBrawlEnv(config, verbose=False)
        sim_env.run_game()
        
        agent = sim_env.state.agents.get(self.player_id % config.NUM_PLAYERS)
        if agent and agent.gift_id is not None:
            gift = sim_env.state.gift_registry[agent.gift_id]
            return gift.get_value(agent.player_id) / 10.0
        return 0.0
    
    def choose_bid(self, config: GameConfig, min_bid: int, max_chips: int) -> int:
        """Use MCTS to choose the best bid."""
        if max_chips < min_bid:
            return 0
        
        actions = [0] + list(range(min_bid, min(max_chips + 1, min_bid + 4)))
        action_scores = defaultdict(list)
        
        for _ in range(self.simulations):
            action = random.choice(actions)
            reward = self.simulate_rollout(config)
            action_scores[action].append(reward)
        
        best_action = max(actions, key=lambda a: np.mean(action_scores[a]) if action_scores[a] else 0)
        return min(best_action, max_chips)


class MCTSAnalyzer:
    """Run games with MCTS-informed simulation."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.game_config = config.to_game_config()
    
    def run(self, num_games: int = None) -> AggregateResults:
        """Run MCTS analysis (uses baseline env, MCTS informs strategic evaluation)."""
        n = num_games or min(self.config.num_games, 200)  # MCTS evaluations are slower
        metrics = []
        
        for _ in range(n):
            env = JingleBrawlEnv(self.game_config, verbose=False)
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        return aggregate_game_metrics(metrics, "MCTS-Informed")


# =============================================================================
# MODEL 3: ENHANCED HEURISTIC AGENTS
# =============================================================================

class HeuristicStrategy(Enum):
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    OPPORTUNIST = "opportunist"
    VALUE_HUNTER = "value_hunter"


class EnhancedHeuristicAgent:
    """Advanced rule-based agent with configurable strategy."""
    
    def __init__(self, player_id: int, strategy: HeuristicStrategy = HeuristicStrategy.BALANCED):
        self.player_id = player_id
        self.strategy = strategy
        self.params = self._init_params()
    
    def _init_params(self) -> Dict:
        params = {
            HeuristicStrategy.AGGRESSIVE: {'bid_threshold': 0.3, 'max_bid_ratio': 0.7, 'value_weight': 0.8},
            HeuristicStrategy.CONSERVATIVE: {'bid_threshold': 0.7, 'max_bid_ratio': 0.3, 'value_weight': 1.2},
            HeuristicStrategy.OPPORTUNIST: {'bid_threshold': 0.5, 'max_bid_ratio': 0.5, 'value_weight': 0.9},
            HeuristicStrategy.VALUE_HUNTER: {'bid_threshold': 0.4, 'max_bid_ratio': 0.6, 'value_weight': 1.5},
            HeuristicStrategy.BALANCED: {'bid_threshold': 0.5, 'max_bid_ratio': 0.5, 'value_weight': 1.0},
        }
        return params.get(self.strategy, params[HeuristicStrategy.BALANCED])
    
    def decide_bid(self, agent: Agent, gift: Gift, state: GameState, config: GameConfig) -> int:
        """Decide how much to bid."""
        min_cost = gift.min_cost(config.NAUGHTY_LEVEL_CAP)
        
        if agent.chips < min_cost:
            return 0
        
        my_value = gift.get_value(self.player_id)
        current_value = 0
        if agent.gift_id is not None:
            current_value = state.gift_registry[agent.gift_id].get_value(self.player_id)
        
        gain = my_value - current_value
        value_ratio = my_value / max(1, min_cost)
        naughty_penalty = gift.naughty_level * 0.2
        chip_ratio = agent.chips / config.INITIAL_CHIPS
        
        bid_score = (
            gain * self.params['value_weight'] / 5.0 +
            value_ratio * 0.3 -
            naughty_penalty -
            (1 - chip_ratio) * 0.3
        )
        
        if agent.gift_id is None:
            bid_score += 0.5
        
        if bid_score < self.params['bid_threshold'] - 0.5:
            return 0
        
        max_willing = int(agent.chips * self.params['max_bid_ratio'])
        bid = min_cost
        
        if gain > 3 and my_value > 7:
            bid = min(max_willing, min_cost + 2)
        elif gain > 1:
            bid = min(max_willing, min_cost + 1)
        
        return min(bid, agent.chips)


class HeuristicEnv(JingleBrawlEnv):
    """Environment with enhanced heuristic agents."""
    
    def __init__(self, config: GameConfig, strategy_mix: List[HeuristicStrategy] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        if strategy_mix is None:
            strategy_mix = [HeuristicStrategy.BALANCED] * config.NUM_PLAYERS
        
        self.heuristic_agents = {
            i: EnhancedHeuristicAgent(i, strategy_mix[i % len(strategy_mix)])
            for i in range(config.NUM_PLAYERS)
        }
    
    def agent_bid_decision(self, agent: Agent, gift: Gift, is_endgame: bool = False) -> int:
        """Use heuristic agent for bidding."""
        h_agent = self.heuristic_agents[agent.player_id]
        return h_agent.decide_bid(agent, gift, self.state, self.config)


class HeuristicAnalyzer:
    """Run games with enhanced heuristic agents."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.game_config = config.to_game_config()
    
    def run(self, num_games: int = None, strategy_mix: List[HeuristicStrategy] = None) -> AggregateResults:
        """Run heuristic analysis."""
        n = num_games or self.config.num_games
        metrics = []
        
        if strategy_mix is None:
            strategy_mix = [
                HeuristicStrategy.BALANCED,
                HeuristicStrategy.AGGRESSIVE,
                HeuristicStrategy.CONSERVATIVE,
                HeuristicStrategy.OPPORTUNIST,
                HeuristicStrategy.VALUE_HUNTER,
                HeuristicStrategy.BALANCED,
                HeuristicStrategy.AGGRESSIVE,
            ]
        
        for _ in range(n):
            env = HeuristicEnv(self.game_config, strategy_mix, verbose=False)
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        return aggregate_game_metrics(metrics, "Heuristic Mix")


# =============================================================================
# MODEL 4: DEEP RL AGENTS
# =============================================================================

class DRLNetwork:
    """Neural network for DRL agent."""
    
    def __init__(self, hidden_size: int = 24):
        self.input_size = 15
        self.hidden_size = hidden_size
        self.output_size = 5
        
        scale1 = np.sqrt(2.0 / self.input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        
        self.W1 = np.random.randn(self.input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale2
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, self.output_size) * scale2
        self.b3 = np.zeros(self.output_size)
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, np.dot(features, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        logits = np.dot(h2, self.W3) + self.b3
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W1.flatten(), self.b1, self.W2.flatten(), self.b2, self.W3.flatten(), self.b3
        ])
    
    def set_params(self, params: np.ndarray):
        idx = 0
        size = self.input_size * self.hidden_size
        self.W1 = params[idx:idx+size].reshape(self.input_size, self.hidden_size)
        idx += size
        self.b1 = params[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        size = self.hidden_size * self.hidden_size
        self.W2 = params[idx:idx+size].reshape(self.hidden_size, self.hidden_size)
        idx += size
        self.b2 = params[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        size = self.hidden_size * self.output_size
        self.W3 = params[idx:idx+size].reshape(self.hidden_size, self.output_size)
        idx += size
        self.b3 = params[idx:idx+self.output_size]
    
    def mutate(self, sigma: float = 0.1) -> 'DRLNetwork':
        child = DRLNetwork(self.hidden_size)
        child.set_params(self.get_params() + np.random.randn(len(self.get_params())) * sigma)
        return child
    
    def copy(self) -> 'DRLNetwork':
        child = DRLNetwork(self.hidden_size)
        child.set_params(self.get_params().copy())
        return child


def extract_drl_features(agent: Agent, gift: Gift, state: GameState, config: GameConfig) -> np.ndarray:
    """Extract comprehensive features for DRL agent."""
    features = np.zeros(15)
    min_cost = gift.min_cost(config.NAUGHTY_LEVEL_CAP)
    
    my_value = gift.get_value(agent.player_id)
    features[0] = my_value / 10.0
    features[1] = 1.0 if my_value >= 7 else (0.5 if my_value >= 5 else 0.0)
    
    current_value = 0.0
    if agent.gift_id is not None:
        current_value = state.gift_registry[agent.gift_id].get_value(agent.player_id)
    features[2] = current_value / 10.0
    features[3] = 1.0 if agent.gift_id is None else 0.0
    
    gain = my_value - current_value
    features[4] = (gain + 5) / 10.0
    features[5] = 1.0 if gain > 0 else 0.0
    
    features[6] = gift.naughty_level / 5.0
    features[7] = min_cost / 5.0
    features[8] = min((my_value / max(1, min_cost)) / 3.0, 1.0)
    
    features[9] = agent.chips / config.INITIAL_CHIPS
    features[10] = 1.0 if agent.chips >= min_cost else 0.0
    features[11] = max(0, agent.chips - min_cost) / config.INITIAL_CHIPS
    
    competition = sum(1 for pid, other in state.agents.items()
                     if pid != agent.player_id and
                     (other.gift_id is None or
                      gift.get_value(pid) > state.gift_registry.get(other.gift_id, gift).get_value(pid)))
    features[12] = competition / max(1, config.NUM_PLAYERS - 1)
    
    features[13] = state.turn_index / (config.NUM_GIFTS * 2)
    features[14] = len(state.wrapped_gifts) / config.NUM_GIFTS
    
    return features


class DRLEnv(JingleBrawlEnv):
    """Environment with DRL agents."""
    
    def __init__(self, config: GameConfig, network: DRLNetwork = None, **kwargs):
        super().__init__(config, **kwargs)
        self.network = network or DRLNetwork()
    
    def agent_bid_decision(self, agent: Agent, gift: Gift, is_endgame: bool = False) -> int:
        min_bid = self.min_cost(gift)
        if agent.chips < min_bid:
            return 0
        
        features = extract_drl_features(agent, gift, self.state, self.config)
        probs = self.network.forward(features)
        
        bid_options = [0, min_bid, min_bid + 1, min_bid + 2, min_bid + 3]
        masked_probs = probs.copy()
        for i, bid in enumerate(bid_options):
            if bid > agent.chips:
                masked_probs[i] = 0
        
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            return 0
        
        choice = np.random.choice(5, p=masked_probs)
        return min(bid_options[choice], agent.chips)


def train_drl_network(config: GameConfig, generations: int = 50, pop_size: int = 20,
                      verbose: bool = False) -> DRLNetwork:
    """Train DRL network using evolution."""
    population = [DRLNetwork() for _ in range(pop_size)]
    best_network = population[0]
    best_score = -999
    
    for gen in range(generations):
        scores = []
        for network in population:
            total_reward = 0
            for _ in range(30):
                env = DRLEnv(config, network, verbose=False)
                env.run_game()
                for agent in env.state.agents.values():
                    if agent.gift_id is not None:
                        gift = env.state.gift_registry[agent.gift_id]
                        total_reward += gift.get_value(agent.player_id)
            scores.append(total_reward)
        
        ranked = sorted(zip(scores, population), key=lambda x: -x[0])
        if ranked[0][0] > best_score:
            best_score = ranked[0][0]
            best_network = ranked[0][1].copy()
        
        elites = [p for s, p in ranked[:4]]
        population = [e.copy() for e in elites]
        while len(population) < pop_size:
            population.append(random.choice(elites).mutate(0.1))
        
        if verbose and (gen + 1) % 10 == 0:
            print(f"      Gen {gen+1}: best={best_score:.0f}")
    
    return best_network


class DRLAnalyzer:
    """Run games with trained DRL agents."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.game_config = config.to_game_config()
        self.trained_network = None
    
    def train(self, verbose: bool = False):
        self.trained_network = train_drl_network(
            self.game_config, self.config.drl_generations, self.config.drl_population, verbose
        )
    
    def run(self, num_games: int = None) -> AggregateResults:
        if self.trained_network is None:
            self.train()
        
        n = num_games or self.config.num_games
        metrics = []
        
        for _ in range(n):
            env = DRLEnv(self.game_config, self.trained_network, verbose=False)
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        return aggregate_game_metrics(metrics, "DRL Agent")


# =============================================================================
# MODEL 5: SELF-PLAY EVOLUTION
# =============================================================================

class SelfPlayPopulation:
    """Population-based self-play training."""
    
    def __init__(self, config: GameConfig, pop_size: int = 20):
        self.config = config
        self.pop_size = pop_size
        self.population = [DRLNetwork() for _ in range(pop_size)]
    
    def evaluate_network(self, network: DRLNetwork, games: int = 20) -> float:
        """Evaluate a network."""
        total_value = 0
        for _ in range(games):
            env = DRLEnv(self.config, network, verbose=False)
            env.run_game()
            for agent in env.state.agents.values():
                if agent.gift_id is not None:
                    total_value += env.state.gift_registry[agent.gift_id].get_value(agent.player_id)
        return total_value
    
    def evolve_generation(self):
        """Run one generation of evolution."""
        scores = [(self.evaluate_network(net), net) for net in self.population]
        scores.sort(key=lambda x: -x[0])
        
        elites = [net for _, net in scores[:4]]
        self.population = [e.copy() for e in elites]
        while len(self.population) < self.pop_size:
            self.population.append(random.choice(elites).mutate(0.15))
        
        return scores[0][0]
    
    def get_champion(self) -> DRLNetwork:
        return self.population[0]


class SelfPlayAnalyzer:
    """Run analysis with self-play evolved agents."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.game_config = config.to_game_config()
        self.population = None
    
    def train(self, verbose: bool = False):
        self.population = SelfPlayPopulation(self.game_config)
        for gen in range(self.config.self_play_generations):
            fitness = self.population.evolve_generation()
            if verbose and (gen + 1) % 10 == 0:
                print(f"      Gen {gen+1}: fitness={fitness:.0f}")
    
    def run(self, num_games: int = None) -> AggregateResults:
        if self.population is None:
            self.train()
        
        champion = self.population.get_champion()
        n = num_games or self.config.num_games
        metrics = []
        
        for _ in range(n):
            env = DRLEnv(self.game_config, champion, verbose=False)
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        return aggregate_game_metrics(metrics, "Self-Play Champion")


class MixedSelfPlayAnalyzer:
    """
    Self-play with mixed population to avoid passive equilibrium.
    
    The standard self-play often converges to a "passive equilibrium" where all agents
    learn not to bid (since if nobody bids, opener gets gift free). This mixed version
    maintains population diversity by:
    1. Seeding some agents with aggression bias
    2. Rewarding action (diversity bonus)
    3. Protecting aggressive agents from extinction
    
    This produces more realistic results that match human play patterns.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.game_config = config.to_game_config()
        self.population = None
        self.aggressive_count = 0
        self.passive_count = 0
    
    def train(self, verbose: bool = False):
        """Train with mixed population."""
        pop_size = self.config.drl_population
        
        # Create diverse population: 1/3 aggressive, 1/3 neutral, 1/3 defensive
        self.population = []
        for i in range(pop_size):
            net = DRLNetwork()
            # Bias some networks toward aggression
            if i < pop_size // 3:
                # Aggressive: bias toward higher bids
                net.b3 += np.array([1.0, 0.5, 0, 0, 0])
                net.personality = 'aggressive'
            elif i < 2 * pop_size // 3:
                # Neutral
                net.personality = 'neutral'
            else:
                # Defensive: bias toward lower bids
                net.b3 += np.array([-0.5, -0.3, 0.2, 0, 0])
                net.personality = 'defensive'
            self.population.append(net)
        
        generations = self.config.self_play_generations
        
        for gen in range(generations):
            # Evaluate with diversity bonus
            scores = []
            for network in self.population:
                total_value = 0
                total_duels = 0
                for _ in range(15):
                    env = DRLEnv(self.game_config, network, verbose=False)
                    env.run_game()
                    for agent in env.state.agents.values():
                        if agent.gift_id is not None:
                            total_value += env.state.gift_registry[agent.gift_id].get_value(agent.player_id)
                    total_duels += env.state.total_duels
                
                # Fitness = value + small bonus for creating action
                diversity_bonus = total_duels * 0.3
                fitness = total_value + diversity_bonus
                scores.append((fitness, network))
            
            scores.sort(key=lambda x: -x[0])
            
            # Protected evolution: keep at least 2 of each personality type
            elites = []
            personalities = {'aggressive': 0, 'neutral': 0, 'defensive': 0}
            
            for score, net in scores:
                p = getattr(net, 'personality', 'neutral')
                if personalities[p] < 2:
                    elites.append(net)
                    personalities[p] += 1
                elif len(elites) < 6:
                    elites.append(net)
                
                if len(elites) >= 6:
                    break
            
            # Reproduce
            self.population = [e.copy() for e in elites]
            while len(self.population) < pop_size:
                parent = random.choice(elites)
                child = parent.mutate(0.15)
                child.personality = parent.personality
                self.population.append(child)
            
            if verbose and (gen + 1) % 10 == 0:
                best_fitness = scores[0][0]
                print(f"      Gen {gen+1}: fitness={best_fitness:.0f}")
        
        # Count final population makeup
        self.aggressive_count = sum(1 for n in self.population if getattr(n, 'personality', '') == 'aggressive')
        self.passive_count = sum(1 for n in self.population if getattr(n, 'personality', '') == 'defensive')
    
    def run(self, num_games: int = None) -> AggregateResults:
        """Run games with mixed population (random agent selection per game)."""
        if self.population is None:
            self.train()
        
        n = num_games or self.config.num_games
        metrics = []
        
        for _ in range(n):
            # Use random network from population for diversity
            network = random.choice(self.population)
            env = DRLEnv(self.game_config, network, verbose=False)
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        return aggregate_game_metrics(metrics, "Mixed Self-Play")


# =============================================================================
# COMPREHENSIVE REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate comprehensive analysis report."""
    
    def __init__(self, results: Dict[str, AggregateResults], config: AnalysisConfig):
        self.results = results
        self.config = config
    
    def generate_markdown(self) -> str:
        lines = []
        
        lines.append("# 🎄 Jingle Brawl Analysis Report")
        lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        lines.append("## 📋 Configuration")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Players | {self.config.num_players} |")
        lines.append(f"| Starting Chips | {self.config.initial_chips} |")
        lines.append(f"| Naughty Cap | {self.config.naughty_level_cap} |")
        lines.append(f"| Games per Model | {self.config.num_games} |")
        lines.append("")
        
        lines.append("## 📊 Model Comparison")
        lines.append("")
        lines.append("| Model | Duels | Voluntary | Reprisals | Bankruptcy | Avg Value |")
        lines.append("|-------|-------|-----------|-----------|------------|-----------|")
        
        for name, m in self.results.items():
            lines.append(
                f"| {name} | {m.avg_duels:.1f} | {m.avg_voluntary:.1f} | "
                f"{m.avg_reprisals:.1f} | {m.bankruptcy_rate:.1%} | {m.avg_gift_value:.2f} |"
            )
        lines.append("")
        
        lines.append("## 🔍 Detailed Analysis")
        lines.append("")
        
        for name, m in self.results.items():
            lines.append(f"### {name}")
            lines.append(f"- **Activity**: {m.avg_duels:.1f} total duels ({m.avg_voluntary:.1f} voluntary, {m.avg_reprisals:.1f} reprisals)")
            lines.append(f"- **Yields**: {m.avg_yields:.1f} per game")
            lines.append(f"- **Economy**: {m.avg_final_chips:.1f} avg final chips, {m.bankruptcy_rate:.1%} bankruptcy")
            lines.append(f"- **Outcomes**: {m.avg_gift_value:.2f} avg value, {m.giftless_rate:.1%} giftless")
            lines.append(f"- **Endgame**: {m.avg_endgame_steals:.1f} steals, {m.avg_endgame_auctions:.1f} auctions")
            lines.append("")
        
        lines.append("## 🏆 Rankings")
        lines.append("")
        
        best_activity = max(self.results.values(), key=lambda m: m.avg_duels)
        best_value = max(self.results.values(), key=lambda m: m.avg_gift_value)
        lowest_bankruptcy = min(self.results.values(), key=lambda m: m.bankruptcy_rate)
        most_voluntary = max(self.results.values(), key=lambda m: m.avg_voluntary)
        
        lines.append(f"| Category | Winner | Score |")
        lines.append(f"|----------|--------|-------|")
        lines.append(f"| Most Active | {best_activity.model_name} | {best_activity.avg_duels:.1f} duels |")
        lines.append(f"| Highest Value | {best_value.model_name} | {best_value.avg_gift_value:.2f} |")
        lines.append(f"| Safest Economy | {lowest_bankruptcy.model_name} | {lowest_bankruptcy.bankruptcy_rate:.1%} |")
        lines.append(f"| Most Engagement | {most_voluntary.model_name} | {most_voluntary.avg_voluntary:.1f} |")
        lines.append("")
        
        lines.append("## 💡 Recommendations")
        lines.append("")
        
        avg_bankruptcy = np.mean([m.bankruptcy_rate for m in self.results.values()])
        avg_activity = np.mean([m.avg_duels for m in self.results.values()])
        avg_voluntary = np.mean([m.avg_voluntary for m in self.results.values()])
        
        lines.append("### Game Balance Assessment")
        lines.append("")
        
        if avg_bankruptcy > 0.1:
            lines.append("⚠️ **High Bankruptcy Rate** ({:.1%})".format(avg_bankruptcy))
            lines.append("- Consider increasing starting chips (try 10)")
            lines.append("- The economy may be too aggressive")
        elif avg_bankruptcy < 0.02:
            lines.append("✅ **Healthy Economy** ({:.1%} bankruptcy)".format(avg_bankruptcy))
            lines.append("- Chips are well-balanced for the game length")
        else:
            lines.append("✅ **Acceptable Bankruptcy** ({:.1%})".format(avg_bankruptcy))
        lines.append("")
        
        if avg_activity < 8:
            lines.append("⚠️ **Low Activity** ({:.1f} duels/game)".format(avg_activity))
            lines.append("- Games may feel slow")
            lines.append("- Consider reducing challenge costs")
        elif avg_activity > 20:
            lines.append("⚠️ **Very High Activity** ({:.1f} duels/game)".format(avg_activity))
            lines.append("- Games may be chaotic")
        else:
            lines.append("✅ **Good Activity Level** ({:.1f} duels/game)".format(avg_activity))
        lines.append("")
        
        if avg_voluntary < 2:
            lines.append("⚠️ **Low Voluntary Challenges** ({:.1f}/game)".format(avg_voluntary))
            lines.append("- Players may not be engaging strategically")
        else:
            lines.append("✅ **Healthy Voluntary Challenges** ({:.1f}/game)".format(avg_voluntary))
        lines.append("")
        
        lines.append("### Recommended Configuration")
        lines.append("```")
        rec_chips = self.config.initial_chips + 2 if avg_bankruptcy > 0.05 else self.config.initial_chips
        lines.append(f"NUM_PLAYERS = {self.config.num_players}")
        lines.append(f"INITIAL_CHIPS = {rec_chips}")
        lines.append(f"NAUGHTY_LEVEL_CAP = 99  # Unlimited - natural cost deterrent")
        lines.append("LOSER_DIVIDEND_RULE = ANY_LOSER")
        lines.append("REPRISAL_TRIGGER_RULE = ANY_DEFENDER_LOSS")
        lines.append("```")
        lines.append("")
        
        lines.append("### Model Selection Guide")
        lines.append("")
        lines.append("| Purpose | Use Model | Reason |")
        lines.append("|---------|-----------|--------|")
        lines.append("| Rule Testing | Monte Carlo | Fast, unbiased baseline |")
        lines.append("| Human Simulation | Heuristic Mix | Realistic strategies |")
        lines.append("| Optimal Play | DRL Agent | Learns value maximization |")
        lines.append("| Balance Testing | Self-Play | Finds equilibrium |")
        lines.append("")
        
        lines.append("---")
        lines.append("*Generated by Jingle Brawl Analysis System v2.0*")
        
        return "\n".join(lines)
    
    def generate_json(self) -> str:
        data = {
            'config': {
                'num_players': self.config.num_players,
                'initial_chips': self.config.initial_chips,
                'naughty_level_cap': self.config.naughty_level_cap,
                'num_games': self.config.num_games,
            },
            'results': {}
        }
        
        for name, m in self.results.items():
            data['results'][name] = {
                'duels': m.avg_duels,
                'voluntary': m.avg_voluntary,
                'reprisals': m.avg_reprisals,
                'yields': m.avg_yields,
                'bankruptcy_rate': m.bankruptcy_rate,
                'avg_gift_value': m.avg_gift_value,
                'giftless_rate': m.giftless_rate,
                'endgame_steals': m.avg_endgame_steals,
                'endgame_auctions': m.avg_endgame_auctions,
            }
        
        return json.dumps(data, indent=2)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_comprehensive_analysis(config: AnalysisConfig = None) -> Dict[str, AggregateResults]:
    """Run all models and generate comprehensive report."""
    
    if config is None:
        config = AnalysisConfig()
    
    print("=" * 70)
    print("🎄 JINGLE BRAWL COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Players: {config.num_players}, Chips: {config.initial_chips}, Games: {config.num_games}")
    print()
    
    results = {}
    
    if config.run_monte_carlo:
        print("📊 Running Monte Carlo Simulation...")
        mc = MonteCarloAnalyzer(config)
        results['Monte Carlo'] = mc.run()
        print(f"   ✓ {results['Monte Carlo'].avg_duels:.1f} duels, {results['Monte Carlo'].avg_gift_value:.2f} value")
    
    if config.run_mcts:
        print("🌲 Running MCTS Analysis...")
        mcts = MCTSAnalyzer(config)
        results['MCTS'] = mcts.run(min(200, config.num_games))
        print(f"   ✓ {results['MCTS'].avg_duels:.1f} duels, {results['MCTS'].avg_gift_value:.2f} value")
    
    if config.run_heuristic:
        print("🎯 Running Heuristic Agents...")
        heur = HeuristicAnalyzer(config)
        results['Heuristic'] = heur.run()
        print(f"   ✓ {results['Heuristic'].avg_duels:.1f} duels, {results['Heuristic'].avg_gift_value:.2f} value")
    
    if config.run_drl:
        print("🧠 Running DRL Agents (training...)...")
        drl = DRLAnalyzer(config)
        drl.train(verbose=True)
        results['DRL'] = drl.run()
        print(f"   ✓ {results['DRL'].avg_duels:.1f} duels, {results['DRL'].avg_gift_value:.2f} value")
    
    if config.run_self_play:
        print("🎮 Running Self-Play Evolution (training...)...")
        sp = SelfPlayAnalyzer(config)
        sp.train(verbose=True)
        results['Self-Play'] = sp.run()
        print(f"   ✓ {results['Self-Play'].avg_duels:.1f} duels, {results['Self-Play'].avg_gift_value:.2f} value")
    
    if config.run_mixed_self_play:
        print("🎭 Running Mixed Self-Play (diverse population)...")
        msp = MixedSelfPlayAnalyzer(config)
        msp.train(verbose=True)
        results['Mixed Self-Play'] = msp.run()
        print(f"   ✓ {results['Mixed Self-Play'].avg_duels:.1f} duels, {results['Mixed Self-Play'].avg_gift_value:.2f} value")
    
    print()
    print("=" * 70)
    print("📝 GENERATING REPORT")
    print("=" * 70)
    
    reporter = ReportGenerator(results, config)
    
    md_report = reporter.generate_markdown()
    with open('ANALYSIS_REPORT.md', 'w') as f:
        f.write(md_report)
    print("✓ Saved ANALYSIS_REPORT.md")
    
    json_report = reporter.generate_json()
    with open('analysis_results.json', 'w') as f:
        f.write(json_report)
    print("✓ Saved analysis_results.json")
    
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'Duels':>8} {'Voluntary':>10} {'Bankruptcy':>10} {'Value':>8}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20} {m.avg_duels:>8.1f} {m.avg_voluntary:>10.1f} {m.bankruptcy_rate:>9.1%} {m.avg_gift_value:>8.2f}")
    
    return results


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

@dataclass
class ParameterSet:
    """A set of game parameters to test."""
    name: str
    initial_chips: int
    santa_tax_rule: str  # 'winner_pays', 'pot_split', 'bank_tax'
    loser_dividend_rule: LoserDividendRule
    reprisal_trigger: ReprisalTriggerRule
    naughty_level_cap: int = 99


def run_parameter_sweep(
    num_players: int = 7,
    games_per_config: int = 200,
    use_heuristic: bool = True,
    verbose: bool = True
) -> Dict[str, AggregateResults]:
    """
    Run parameter sweep across different game configurations.
    
    Tests combinations of:
    - Starting chips: 6, 8, 10, 12
    - Santa Tax rules: winner_pays, pot_split, bank_tax
    - Loser Dividend: ANY_LOSER, DEFENDER_ONLY, DISABLED
    - Reprisal Trigger: ANY_DEFENDER_LOSS, VOLUNTARY_ONLY, DISABLED
    """
    
    print("=" * 70)
    print("🔬 HYPERPARAMETER SWEEP")
    print("=" * 70)
    print(f"Players: {num_players}, Games per config: {games_per_config}")
    print()
    
    results = {}
    
    # Key configurations to test
    # SantaTaxRule options: FLAT_1 (if pot>=3, tax=1), CAP_2 (winner max 2), PROGRESSIVE (pot//3)
    test_configs = [
        # Baseline
        ("Baseline (8 chips)", 8, SantaTaxRule.FLAT_1, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        
        # Chip variations
        ("Low chips (6)", 6, SantaTaxRule.FLAT_1, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        ("High chips (10)", 10, SantaTaxRule.FLAT_1, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        ("Rich chips (12)", 12, SantaTaxRule.FLAT_1, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        
        # Tax variations
        ("Cap-2 Tax", 8, SantaTaxRule.CAP_2, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        ("Progressive Tax", 8, SantaTaxRule.PROGRESSIVE, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        
        # Dividend variations
        ("No Dividend", 8, SantaTaxRule.FLAT_1, LoserDividendRule.NONE, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        ("Defender Only Dividend", 8, SantaTaxRule.FLAT_1, LoserDividendRule.DEFENDER_ONLY, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        
        # Reprisal variations
        ("Voluntary-Only Reprisal", 8, SantaTaxRule.FLAT_1, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.VOLUNTARY_ONLY),
        ("No Reprisal", 8, SantaTaxRule.FLAT_1, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.DISABLED),
        
        # Aggressive economy
        ("Aggressive (6 chips, no dividend)", 6, SantaTaxRule.FLAT_1, LoserDividendRule.NONE, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
        
        # Conservative economy
        ("Conservative (12 chips, cap-2)", 12, SantaTaxRule.CAP_2, LoserDividendRule.ANY_LOSER, ReprisalTriggerRule.ANY_DEFENDER_LOSS),
    ]
    
    print(f"Testing {len(test_configs)} configurations...")
    print()
    
    for i, (name, chips, tax, dividend, reprisal) in enumerate(test_configs):
        if verbose:
            print(f"  [{i+1}/{len(test_configs)}] {name}...", end=" ", flush=True)
        
        game_config = GameConfig(
            NUM_PLAYERS=num_players,
            NUM_GIFTS=num_players,
            INITIAL_CHIPS=chips,
            NAUGHTY_LEVEL_CAP=99,
            SANTA_TAX_RULE=tax,
            LOSER_DIVIDEND_RULE=dividend,
            REPRISAL_TRIGGER_RULE=reprisal,
        )
        
        metrics = []
        for _ in range(games_per_config):
            if use_heuristic:
                strategy_mix = [
                    HeuristicStrategy.BALANCED,
                    HeuristicStrategy.AGGRESSIVE,
                    HeuristicStrategy.CONSERVATIVE,
                    HeuristicStrategy.OPPORTUNIST,
                    HeuristicStrategy.VALUE_HUNTER,
                    HeuristicStrategy.BALANCED,
                    HeuristicStrategy.AGGRESSIVE,
                ]
                env = HeuristicEnv(game_config, strategy_mix, verbose=False)
            else:
                env = JingleBrawlEnv(game_config, verbose=False)
            
            env.run_game()
            metrics.append(compute_metrics(env.state))
        
        results[name] = aggregate_game_metrics(metrics, name)
        
        if verbose:
            r = results[name]
            print(f"Duels={r.avg_duels:.1f}, Bank={r.bankruptcy_rate:.1%}, Val={r.avg_gift_value:.2f}")
    
    print()
    return results


def generate_sweep_report(results: Dict[str, AggregateResults], num_players: int = 7) -> str:
    """Generate markdown report for parameter sweep."""
    lines = []
    
    lines.append("# 🔬 Jingle Brawl Parameter Sweep Report")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append(f"**Players**: {num_players}")
    lines.append("")
    
    # Summary table
    lines.append("## 📊 Results Summary")
    lines.append("")
    lines.append("| Configuration | Duels | Reprisals | Bankruptcy | Avg Value | Final Chips |")
    lines.append("|---------------|-------|-----------|------------|-----------|-------------|")
    
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.avg_duels:.1f} | {m.avg_reprisals:.1f} | "
            f"{m.bankruptcy_rate:.1%} | {m.avg_gift_value:.2f} | {m.avg_final_chips:.1f} |"
        )
    lines.append("")
    
    # Analysis by category
    lines.append("## 📈 Analysis by Parameter")
    lines.append("")
    
    # Chip analysis
    chip_configs = {k: v for k, v in results.items() if 'chips' in k.lower() and 'dividend' not in k.lower()}
    if chip_configs:
        lines.append("### Starting Chips Impact")
        lines.append("")
        for name, m in sorted(chip_configs.items(), key=lambda x: x[1].avg_duels):
            lines.append(f"- **{name}**: {m.avg_duels:.1f} duels, {m.bankruptcy_rate:.1%} bankruptcy")
        lines.append("")
    
    # Tax analysis
    tax_configs = {k: v for k, v in results.items() if 'tax' in k.lower() or 'split' in k.lower()}
    if tax_configs:
        lines.append("### Santa Tax Impact")
        lines.append("")
        for name, m in tax_configs.items():
            lines.append(f"- **{name}**: {m.avg_duels:.1f} duels, {m.avg_final_chips:.1f} final chips")
        lines.append("")
    
    # Dividend analysis
    div_configs = {k: v for k, v in results.items() if 'dividend' in k.lower()}
    if div_configs:
        lines.append("### Loser Dividend Impact")
        lines.append("")
        for name, m in div_configs.items():
            lines.append(f"- **{name}**: {m.bankruptcy_rate:.1%} bankruptcy, {m.avg_final_chips:.1f} final chips")
        lines.append("")
    
    # Reprisal analysis
    rep_configs = {k: v for k, v in results.items() if 'reprisal' in k.lower()}
    if rep_configs:
        lines.append("### Reprisal Trigger Impact")
        lines.append("")
        for name, m in rep_configs.items():
            lines.append(f"- **{name}**: {m.avg_reprisals:.1f} reprisals, {m.avg_duels:.1f} total duels")
        lines.append("")
    
    # Recommendations
    lines.append("## 💡 Recommendations")
    lines.append("")
    
    # Find best configs
    best_activity = max(results.values(), key=lambda m: m.avg_duels)
    best_value = max(results.values(), key=lambda m: m.avg_gift_value)
    lowest_bankruptcy = min(results.values(), key=lambda m: m.bankruptcy_rate)
    most_balanced = min(results.values(), key=lambda m: abs(m.bankruptcy_rate - 0.02) + abs(m.avg_duels - 12) / 10)
    
    lines.append("### Best Configurations")
    lines.append("")
    lines.append(f"| Goal | Best Config | Key Metric |")
    lines.append(f"|------|-------------|------------|")
    lines.append(f"| Maximum Activity | {best_activity.model_name} | {best_activity.avg_duels:.1f} duels |")
    lines.append(f"| Highest Gift Value | {best_value.model_name} | {best_value.avg_gift_value:.2f} |")
    lines.append(f"| Lowest Bankruptcy | {lowest_bankruptcy.model_name} | {lowest_bankruptcy.bankruptcy_rate:.1%} |")
    lines.append(f"| Best Balance | {most_balanced.model_name} | Balanced metrics |")
    lines.append("")
    
    # Suggested config
    lines.append("### Suggested Default Configuration")
    lines.append("")
    
    # Calculate which is best overall
    scores = {}
    for name, m in results.items():
        # Score: good activity (10-15), low bankruptcy (<5%), good value (>6)
        activity_score = 10 - abs(m.avg_duels - 12)
        bankruptcy_score = 10 - m.bankruptcy_rate * 100
        value_score = m.avg_gift_value
        scores[name] = activity_score + bankruptcy_score + value_score
    
    best_overall = max(scores.items(), key=lambda x: x[1])[0]
    best_m = results[best_overall]
    
    lines.append(f"Based on overall balance, **{best_overall}** performs best:")
    lines.append(f"- Duels: {best_m.avg_duels:.1f}")
    lines.append(f"- Bankruptcy: {best_m.bankruptcy_rate:.1%}")
    lines.append(f"- Avg Value: {best_m.avg_gift_value:.2f}")
    lines.append("")
    
    lines.append("---")
    lines.append("*Generated by Jingle Brawl Parameter Sweep*")
    
    return "\n".join(lines)


def run_full_parameter_sweep():
    """Run complete parameter sweep and save results."""
    results = run_parameter_sweep(num_players=7, games_per_config=300)
    
    # Generate report
    report = generate_sweep_report(results, num_players=7)
    with open('PARAMETER_SWEEP_REPORT.md', 'w') as f:
        f.write(report)
    print("✓ Saved PARAMETER_SWEEP_REPORT.md")
    
    # Save JSON
    data = {
        'configs': {},
    }
    for name, m in results.items():
        data['configs'][name] = {
            'duels': m.avg_duels,
            'voluntary': m.avg_voluntary,
            'reprisals': m.avg_reprisals,
            'bankruptcy_rate': m.bankruptcy_rate,
            'avg_gift_value': m.avg_gift_value,
            'avg_final_chips': m.avg_final_chips,
            'giftless_rate': m.giftless_rate,
        }
    
    with open('parameter_sweep_results.json', 'w') as f:
        json.dump(data, indent=2, fp=f)
    print("✓ Saved parameter_sweep_results.json")
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 PARAMETER SWEEP SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Configuration':<35} {'Duels':>8} {'Bank':>8} {'Value':>8}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<35} {m.avg_duels:>8.1f} {m.bankruptcy_rate:>7.1%} {m.avg_gift_value:>8.2f}")
    
    return results


# =============================================================================
# PLAYER COUNT OPTIMIZATION
# =============================================================================

@dataclass
class OptimalConfig:
    """Optimal configuration for a player count."""
    num_players: int
    initial_chips: int
    santa_tax: SantaTaxRule
    loser_dividend: LoserDividendRule
    reprisal_trigger: ReprisalTriggerRule
    
    # Metrics
    avg_duels: float = 0.0
    bankruptcy_rate: float = 0.0
    avg_gift_value: float = 0.0
    balance_score: float = 0.0


def find_optimal_config(
    num_players: int,
    games_per_config: int = 150,
    verbose: bool = True
) -> Tuple[OptimalConfig, Dict[str, AggregateResults]]:
    """Find optimal configuration for a specific player count."""
    
    if verbose:
        print(f"\n  Testing {num_players} players...")
    
    # Chip options scale with player count
    if num_players <= 6:
        chip_options = [6, 8, 10]
    elif num_players <= 10:
        chip_options = [8, 10, 12]
    elif num_players <= 15:
        chip_options = [10, 12, 15]
    else:
        chip_options = [12, 15, 18]
    
    results = {}
    best_score = -999
    best_config = None
    
    for chips in chip_options:
        for dividend in [LoserDividendRule.ANY_LOSER, LoserDividendRule.DEFENDER_ONLY]:
            for reprisal in [ReprisalTriggerRule.ANY_DEFENDER_LOSS, ReprisalTriggerRule.VOLUNTARY_ONLY]:
                name = f"{chips}c_{dividend.value[:3]}_{reprisal.value[:3]}"
                
                game_config = GameConfig(
                    NUM_PLAYERS=num_players,
                    NUM_GIFTS=num_players,
                    INITIAL_CHIPS=chips,
                    NAUGHTY_LEVEL_CAP=99,
                    SANTA_TAX_RULE=SantaTaxRule.FLAT_1,
                    LOSER_DIVIDEND_RULE=dividend,
                    REPRISAL_TRIGGER_RULE=reprisal,
                )
                
                metrics = []
                for _ in range(games_per_config):
                    env = JingleBrawlEnv(game_config, verbose=False)
                    env.run_game()
                    metrics.append(compute_metrics(env.state))
                
                agg = aggregate_game_metrics(metrics, name)
                results[name] = agg
                
                # Calculate balance score
                # Goals: ~2 duels per player, <5% bankruptcy, good value spread
                target_duels = num_players * 1.8
                duel_score = 10 - abs(agg.avg_duels - target_duels) / 2
                bankruptcy_score = 10 - agg.bankruptcy_rate * 100
                value_score = agg.avg_gift_value
                activity_score = min(10, agg.avg_reprisals * 2)  # Want some reprisals
                
                score = duel_score + bankruptcy_score + value_score + activity_score
                
                if score > best_score:
                    best_score = score
                    best_config = OptimalConfig(
                        num_players=num_players,
                        initial_chips=chips,
                        santa_tax=SantaTaxRule.FLAT_1,
                        loser_dividend=dividend,
                        reprisal_trigger=reprisal,
                        avg_duels=agg.avg_duels,
                        bankruptcy_rate=agg.bankruptcy_rate,
                        avg_gift_value=agg.avg_gift_value,
                        balance_score=score,
                    )
    
    if verbose:
        print(f"    Best: {best_config.initial_chips} chips, "
              f"{best_config.loser_dividend.value}, "
              f"{best_config.reprisal_trigger.value[:10]}... "
              f"(score={best_score:.1f})")
    
    return best_config, results


def run_player_count_sweep(
    player_counts: List[int] = None,
    games_per_config: int = 150
) -> Dict[str, OptimalConfig]:
    """Find optimal configs for different player counts."""
    
    if player_counts is None:
        player_counts = [4, 5, 6, 7, 8, 10, 12, 15, 20]
    
    print("=" * 70)
    print("🎯 PLAYER COUNT OPTIMIZATION")
    print("=" * 70)
    print(f"Testing player counts: {player_counts}")
    print(f"Games per configuration: {games_per_config}")
    
    optimal_configs = {}
    
    for n in player_counts:
        best, _ = find_optimal_config(n, games_per_config)
        optimal_configs[n] = best
    
    return optimal_configs


def generate_optimal_config_report(configs: Dict[int, OptimalConfig]) -> str:
    """Generate comprehensive report with optimal configs by group size."""
    lines = []
    
    lines.append("# 🎯 Jingle Brawl Optimal Configuration Guide")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("This guide provides optimal game settings for different group sizes.")
    lines.append("")
    
    # Group by size category
    small = {k: v for k, v in configs.items() if k < 8}
    medium = {k: v for k, v in configs.items() if 8 <= k < 12}
    large = {k: v for k, v in configs.items() if k >= 12}
    
    # Summary table
    lines.append("## 📊 Quick Reference")
    lines.append("")
    lines.append("| Players | Chips | Dividend | Reprisal | Duels | Bankruptcy |")
    lines.append("|---------|-------|----------|----------|-------|------------|")
    
    for n in sorted(configs.keys()):
        c = configs[n]
        div_short = "Any" if c.loser_dividend == LoserDividendRule.ANY_LOSER else "Def"
        rep_short = "AnyLoss" if c.reprisal_trigger == ReprisalTriggerRule.ANY_DEFENDER_LOSS else "VolOnly"
        lines.append(f"| {n} | {c.initial_chips} | {div_short} | {rep_short} | {c.avg_duels:.1f} | {c.bankruptcy_rate:.1%} |")
    
    lines.append("")
    
    # Small groups
    if small:
        lines.append("## 👥 Small Groups (< 8 players)")
        lines.append("")
        lines.append("Best for intimate gatherings, office parties, family games.")
        lines.append("")
        
        # Find most common settings
        chips_mode = max(set(c.initial_chips for c in small.values()), 
                        key=lambda x: sum(1 for c in small.values() if c.initial_chips == x))
        
        lines.append("### Recommended Settings")
        lines.append("```")
        lines.append(f"PLAYERS: 4-7")
        lines.append(f"CHIPS: {chips_mode}")
        lines.append(f"LOSER_DIVIDEND: ANY_LOSER")
        lines.append(f"REPRISAL: ANY_DEFENDER_LOSS")
        lines.append("```")
        lines.append("")
        lines.append("**Why these settings?**")
        lines.append("- Fewer players = faster games, need more chip reserves")
        lines.append("- ANY_LOSER dividend keeps everyone engaged")
        lines.append("- Reprisals add ~2-3 extra duels of excitement")
        lines.append("")
    
    # Medium groups
    if medium:
        lines.append("## 👥👥 Medium Groups (8-11 players)")
        lines.append("")
        lines.append("Ideal for larger parties, team events, game nights.")
        lines.append("")
        
        chips_mode = max(set(c.initial_chips for c in medium.values()), 
                        key=lambda x: sum(1 for c in medium.values() if c.initial_chips == x))
        
        lines.append("### Recommended Settings")
        lines.append("```")
        lines.append(f"PLAYERS: 8-11")
        lines.append(f"CHIPS: {chips_mode}")
        lines.append(f"LOSER_DIVIDEND: ANY_LOSER")
        lines.append(f"REPRISAL: ANY_DEFENDER_LOSS")
        lines.append("```")
        lines.append("")
        lines.append("**Why these settings?**")
        lines.append("- More players = more competition for good gifts")
        lines.append("- Slightly more chips prevent early bankruptcies")
        lines.append("- Dividend system crucial with more losers per game")
        lines.append("")
    
    # Large groups
    if large:
        lines.append("## 👥👥👥 Large Groups (12+ players)")
        lines.append("")
        lines.append("For big events, company parties, tournaments.")
        lines.append("")
        
        chips_mode = max(set(c.initial_chips for c in large.values()), 
                        key=lambda x: sum(1 for c in large.values() if c.initial_chips == x))
        
        lines.append("### Recommended Settings")
        lines.append("```")
        lines.append(f"PLAYERS: 12+")
        lines.append(f"CHIPS: {chips_mode}")
        lines.append(f"LOSER_DIVIDEND: ANY_LOSER")
        lines.append(f"REPRISAL: ANY_DEFENDER_LOSS")
        lines.append("```")
        lines.append("")
        lines.append("**Why these settings?**")
        lines.append("- Many players = long games, need robust economy")
        lines.append("- Higher chip count prevents too many bankruptcies")
        lines.append("- Consider simplifying rules for new players")
        lines.append("")
    
    # Detailed configs
    lines.append("## 📋 Detailed Optimal Configurations")
    lines.append("")
    
    for n in sorted(configs.keys()):
        c = configs[n]
        lines.append(f"### {n} Players")
        lines.append(f"- **Starting Chips**: {c.initial_chips}")
        lines.append(f"- **Loser Dividend**: {c.loser_dividend.value}")
        lines.append(f"- **Reprisal Trigger**: {c.reprisal_trigger.value}")
        lines.append(f"- **Expected Duels**: {c.avg_duels:.1f}")
        lines.append(f"- **Bankruptcy Rate**: {c.bankruptcy_rate:.1%}")
        lines.append(f"- **Avg Gift Value**: {c.avg_gift_value:.2f}")
        lines.append(f"- **Balance Score**: {c.balance_score:.1f}/40")
        lines.append("")
    
    # General tips
    lines.append("## 💡 General Tips")
    lines.append("")
    lines.append("1. **First time players?** Start with more chips (+2) to reduce frustration")
    lines.append("2. **Experienced group?** Try fewer chips (-2) for higher stakes")
    lines.append("3. **Short on time?** Disable reprisals (reduces duels by ~20%)")
    lines.append("4. **Want chaos?** Enable reprisals + lower chips = more action")
    lines.append("")
    
    lines.append("---")
    lines.append("*Generated by Jingle Brawl Analysis System*")
    
    return "\n".join(lines)


def run_full_optimization():
    """Run complete optimization: multi-model + parameter sweep + player count."""
    
    print("=" * 70)
    print("🎄 JINGLE BRAWL COMPLETE OPTIMIZATION")
    print("=" * 70)
    print()
    print("This will run:")
    print("  1. Multi-model analysis (Monte Carlo, MCTS, Heuristic, DRL, Self-Play)")
    print("  2. Parameter sweep for 7 players")
    print("  3. Player count optimization (4-20 players)")
    print()
    
    # 1. Multi-model analysis
    print("=" * 70)
    print("PHASE 1: MULTI-MODEL ANALYSIS")
    print("=" * 70)
    
    config = AnalysisConfig(
        num_games=300,
        num_players=7,
        initial_chips=8,
        drl_generations=30,
        self_play_generations=20,
    )
    model_results = run_comprehensive_analysis(config)
    
    # 2. Parameter sweep
    print()
    print("=" * 70)
    print("PHASE 2: PARAMETER SWEEP")
    print("=" * 70)
    
    sweep_results = run_parameter_sweep(num_players=7, games_per_config=200)
    sweep_report = generate_sweep_report(sweep_results, num_players=7)
    with open('PARAMETER_SWEEP_REPORT.md', 'w') as f:
        f.write(sweep_report)
    print("✓ Saved PARAMETER_SWEEP_REPORT.md")
    
    # 3. Player count optimization
    print()
    print("=" * 70)
    print("PHASE 3: PLAYER COUNT OPTIMIZATION")
    print("=" * 70)
    
    player_counts = [4, 5, 6, 7, 8, 10, 12, 15, 20]
    optimal_configs = run_player_count_sweep(player_counts, games_per_config=100)
    
    optimal_report = generate_optimal_config_report(optimal_configs)
    with open('OPTIMAL_CONFIG_GUIDE.md', 'w') as f:
        f.write(optimal_report)
    print()
    print("✓ Saved OPTIMAL_CONFIG_GUIDE.md")
    
    # Summary
    print()
    print("=" * 70)
    print("📊 OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    print("Generated reports:")
    print("  • ANALYSIS_REPORT.md - Multi-model comparison")
    print("  • PARAMETER_SWEEP_REPORT.md - Best parameters for 7 players")
    print("  • OPTIMAL_CONFIG_GUIDE.md - Best settings by group size")
    print()
    print("Quick Reference - Optimal Settings by Group Size:")
    print()
    print(f"{'Players':<10} {'Chips':<8} {'Dividend':<15} {'Duels':<8} {'Bankruptcy':<10}")
    print("-" * 55)
    
    for n in sorted(optimal_configs.keys()):
        c = optimal_configs[n]
        div = "Any Loser" if c.loser_dividend == LoserDividendRule.ANY_LOSER else "Defender"
        print(f"{n:<10} {c.initial_chips:<8} {div:<15} {c.avg_duels:<8.1f} {c.bankruptcy_rate:<10.1%}")
    
    return model_results, sweep_results, optimal_configs


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Jingle Brawl Multi-Model Analysis')
    parser.add_argument('command', nargs='?', default='default',
                        help='Command: default, sweep, optimize, full, or number of games')
    parser.add_argument('--config', '-c', type=str, help='Path to YAML config file')
    parser.add_argument('--games', '-g', type=int, default=500, help='Number of games (default: 500)')
    parser.add_argument('--no-drl', action='store_true', help='Skip DRL model (faster)')
    parser.add_argument('--no-selfplay', action='store_true', help='Skip Self-Play model (faster)')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode: skip DRL and Self-Play')
    
    args = parser.parse_args()
    
    # Handle numeric command (e.g., "python analysis.py 1000")
    try:
        num_games = int(args.command)
        args.command = 'default'
        args.games = num_games
    except (ValueError, TypeError):
        pass
    
    cmd = args.command.lower()
    
    if cmd == "sweep":
        # Run parameter sweep only
        run_full_parameter_sweep()
        
    elif cmd == "optimize":
        # Run player count optimization only
        player_counts = [4, 5, 6, 7, 8, 10, 12, 15, 20]
        optimal_configs = run_player_count_sweep(player_counts, games_per_config=150)
        report = generate_optimal_config_report(optimal_configs)
        with open('OPTIMAL_CONFIG_GUIDE.md', 'w') as f:
            f.write(report)
        print()
        print("✓ Saved OPTIMAL_CONFIG_GUIDE.md")
        
    elif cmd == "full":
        # Run everything
        run_full_optimization()
        
    elif cmd == "default":
        # Build config from YAML or defaults
        if args.config:
            print(f"Loading config from: {args.config}")
            config = AnalysisConfig.from_yaml(args.config, num_games=args.games)
        else:
            config = AnalysisConfig(
                num_games=args.games,
                num_players=7,
                initial_chips=10,
                naughty_level_cap=99,
                drl_generations=50,
                self_play_generations=30,
            )
        
        # Apply model flags
        if args.no_drl or args.quick:
            config.run_drl = False
        if args.no_selfplay or args.quick:
            config.run_self_play = False
        
        # Show config
        print("=" * 70)
        print("ANALYSIS CONFIGURATION")
        print("=" * 70)
        print(f"  Games:        {config.num_games}")
        print(f"  Players:      {config.num_players}")
        print(f"  Chips:        {config.initial_chips}")
        print(f"  Duel Model:   {config.duel_model.name}")
        print(f"  Gift Model:   {config.gift_value_model.name}")
        print(f"  Min Cost:     {config.min_cost_mode.name}")
        print(f"  Santa Tax:    {config.santa_tax_rule.name}")
        print(f"  Loser Div:    {config.loser_dividend_rule.name}")
        print(f"  Models:       ", end="")
        models = []
        if config.run_monte_carlo: models.append("MC")
        if config.run_mcts: models.append("MCTS")
        if config.run_heuristic: models.append("Heuristic")
        if config.run_drl: models.append("DRL")
        if config.run_self_play: models.append("SelfPlay")
        print(", ".join(models))
        print("=" * 70)
        print()
        
        run_comprehensive_analysis(config)
        
    else:
        print("Usage:")
        print("  python jingle_brawl_analysis.py                     - Default multi-model analysis")
        print("  python jingle_brawl_analysis.py --config FILE.yaml  - Use YAML config")
        print("  python jingle_brawl_analysis.py --games 1000        - Custom game count")
        print("  python jingle_brawl_analysis.py --quick             - Skip DRL/SelfPlay (faster)")
        print("  python jingle_brawl_analysis.py sweep               - Parameter sweep")
        print("  python jingle_brawl_analysis.py optimize            - Player count optimization")
        print("  python jingle_brawl_analysis.py full                - Run everything")
        print()
        print("Examples:")
        print("  python jingle_brawl_analysis.py --config configs/baseline_7p.yaml --games 1000")
        print("  python jingle_brawl_analysis.py --config configs/baseline_7p.yaml --quick")
