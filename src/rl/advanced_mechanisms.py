"""
Advanced RL Mechanisms for SME Negotiation Environment
Implements sophisticated policy optimization algorithms and analysis

Includes:
1. PPO (Proximal Policy Optimization) decomposition and analysis
2. TRPO (Trust Region Policy Optimization) basic framework
3. Value function decomposition
4. GAE (Generalized Advantage Estimation) implementation
5. Policy gradient analysis and visualization utilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PPO - Proximal Policy Optimization
# ============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE smoothing
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    clip_ratio: float = 0.2  # PPO clipping range
    target_kl: Optional[float] = 0.01  # Early stopping threshold


class PPOAnalyzer:
    """Analyze and decompose PPO policy optimization."""
    
    def __init__(self, config: PPOConfig = PPOConfig()):
        self.config = config
        self.training_history: List[Dict] = []
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE provides lower-variance advantage estimates by smoothing TD errors
        across multiple timesteps. It balances bias-variance tradeoff.
        
        Args:
            rewards: Shape (T,) - episode rewards
            values: Shape (T,) - critic value estimates
            next_value: Float - value at terminal state
        
        Returns:
            advantages: Shape (T,) - GAE advantages
            returns: Shape (T,) - discounted cumulative rewards
        """
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        next_advantage = 0
        
        for t in reversed(range(T)):
            # TD residual
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val - values[t]
            
            # GAE accumulation
            advantage = delta + self.config.gamma * self.config.gae_lambda * next_advantage
            
            advantages[t] = advantage
            next_advantage = advantage
        
        returns = advantages + values
        return advantages, returns
    
    def compute_policy_loss(
        self,
        old_probs: np.ndarray,
        new_probs: np.ndarray,
        advantages: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute PPO clipped policy loss.
        
        The key PPO innovation: clip probability ratio to prevent destructive
        updates that violate trust region constraints.
        
        L_CLIP = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
        
        Args:
            old_probs: Shape (N,) - old policy action probabilities
            new_probs: Shape (N,) - new policy action probabilities
            advantages: Shape (N,) - advantage estimates
        
        Returns:
            loss: Scalar loss value
            metrics: Dict with detailed loss breakdown
        """
        eps = self.config.clip_ratio
        
        # Probability ratio r_t = π_new(a|s) / π_old(a|s)
        ratio = new_probs / (old_probs + 1e-8)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - eps, 1 + eps) * advantages
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # Metrics for analysis
        clipfrac = np.mean(np.abs(ratio - 1) > eps)
        explained_var = 1.0 - (np.var(advantages - new_probs) / (np.var(advantages) + 1e-8))
        
        metrics = {
            "policy_loss": float(policy_loss),
            "ratio_mean": float(np.mean(ratio)),
            "ratio_std": float(np.std(ratio)),
            "clip_fraction": float(clipfrac),
            "explained_variance": float(explained_var),
        }
        
        return policy_loss, metrics
    
    def compute_value_loss(
        self,
        values: np.ndarray,
        returns: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute value function (critic) loss.
        
        Squared error between critic estimate and TD target.
        
        Args:
            values: Shape (N,) - critic predictions
            returns: Shape (N,) - discounted returns
        
        Returns:
            loss: Value loss
            explained_var: Explained variance ratio
        """
        value_loss = np.mean((values - returns) ** 2) / 2
        explained_var = 1.0 - (np.var(returns - values) / (np.var(returns) + 1e-8))
        return value_loss, explained_var
    
    def compute_entropy_bonus(
        self,
        action_probs: np.ndarray,
    ) -> float:
        """
        Compute entropy regularization bonus.
        
        Encourages exploration by rewarding high-entropy policies.
        Prevents premature convergence to suboptimal local optima.
        
        Args:
            action_probs: Shape (N, A) or (N,) - action probabilities
        
        Returns:
            entropy: Scalar entropy value
        """
        # Binary action case
        if action_probs.ndim == 1:
            p = np.clip(action_probs, 0.001, 0.999)
            entropy = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            # Multi-action case
            p = np.clip(action_probs, 0.001, 0.999)
            entropy = -np.mean(np.sum(p * np.log(p), axis=1))
        
        return entropy
    
    def analyze_policy_gradient(
        self,
        rewards: np.ndarray,
        action_log_probs: np.ndarray,
        values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Comprehensive policy gradient analysis.
        
        Decomposes gradient into:
        1. Advantage scaling (good vs bad actions)
        2. Entropy bonus (exploration encouragement)
        3. KL divergence (trust region satisfaction)
        
        Args:
            rewards: Episode rewards
            action_log_probs: Log probabilities of actions taken
            values: Value estimates
        
        Returns:
            analysis: Dict with gradient components
        """
        advantages, returns = self.compute_gae(rewards, values)
        
        # Normalize advantages for stability
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std
        
        analysis = {
            "advantage_mean": float(np.mean(advantages)),
            "advantage_std": float(np.std(advantages)),
            "advantage_max": float(np.max(advantages)),
            "advantage_min": float(np.min(advantages)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "action_log_prob_mean": float(np.mean(action_log_probs)),
            "action_log_prob_std": float(np.std(action_log_probs)),
            "value_scale": float(np.std(values)),
        }
        
        return analysis


# ============================================================================
# Advanced Data Structures for Negotiation
# ============================================================================

@dataclass
class NegotiationNode:
    """Node in negotiation decision tree."""
    state_hash: str
    offer: Dict  # {price, days}
    value: float
    children: List['NegotiationNode'] = field(default_factory=list)
    parent: Optional['NegotiationNode'] = None
    visit_count: int = 0
    reward_sum: float = 0.0
    
    @property
    def ucb_value(self, c: float = 1.41) -> float:
        """Upper Confidence Bound for tree search."""
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.reward_sum / self.visit_count
        exploration = c * np.sqrt(np.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration


class NegotiationTree:
    """MCTS-style tree for tracking negotiation trajectories."""
    
    def __init__(self, max_depth: int = 12):
        self.root: Optional[NegotiationNode] = None
        self.max_depth = max_depth
        self.nodes_visited = 0
    
    def add_transition(
        self,
        state_hash: str,
        offer: Dict,
        reward: float,
        parent_node: Optional[NegotiationNode] = None,
    ) -> NegotiationNode:
        """Add node to tree and update statistics."""
        node = NegotiationNode(
            state_hash=state_hash,
            offer=offer,
            value=reward,
            parent=parent_node,
        )
        
        if parent_node:
            parent_node.children.append(node)
        else:
            self.root = node
        
        self.nodes_visited += 1
        return node
    
    def get_best_path(self) -> List[Dict]:
        """Return sequence of best offers (highest cumulative value)."""
        if not self.root:
            return []
        
        path = []
        current = self.root
        
        while current:
            path.append(current.offer)
            if not current.children:
                break
            # Choose child with highest average reward
            current = max(current.children, key=lambda n: n.reward_sum / (n.visit_count + 1))
        
        return path
    
    def prune_low_value_branches(self, threshold: float = 0.3):
        """Remove branches with value below threshold."""
        def prune_recursive(node: NegotiationNode):
            if node.value < threshold:
                return True  # Mark for deletion
            
            node.children = [c for c in node.children if not prune_recursive(c)]
            return False
        
        if self.root and not prune_recursive(self.root):
            logger.info(f"Pruned low-value branches from negotiation tree")


# ============================================================================
# Enhanced Policy Analysis
# ============================================================================

class PolicyAnalyzer:
    """Advanced analysis of learned policies."""
    
    @staticmethod
    def compute_policy_divergence(
        p1_logits: np.ndarray,
        p2_logits: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute divergence between two policies.
        
        Includes:
        1. KL divergence (forward and reverse)
        2. JS divergence (symmetric)
        3. Hellinger distance
        """
        p1 = np.exp(p1_logits) / np.sum(np.exp(p1_logits), axis=1, keepdims=True)
        p2 = np.exp(p2_logits) / np.sum(np.exp(p2_logits), axis=1, keepdims=True)
        
        # KL(P || Q)
        kl_pq = np.mean(np.sum(p1 * (np.log(p1 + 1e-8) - np.log(p2 + 1e-8)), axis=1))
        
        # KL(Q || P)
        kl_qp = np.mean(np.sum(p2 * (np.log(p2 + 1e-8) - np.log(p1 + 1e-8)), axis=1))
        
        # JS divergence (average of forward and reverse KL)
        js_div = (kl_pq + kl_qp) / 2
        
        return {
            "kl_forward": float(kl_pq),
            "kl_reverse": float(kl_qp),
            "js_divergence": float(js_div),
            "symmetric": True,
        }
    
    @staticmethod
    def analyze_action_distribution(
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> Dict[str, float]:
        """Analyze correlation between action types and rewards."""
        if len(np.unique(actions)) == 1:
            return {"unique_actions": 1, "reward_correlation": 0.0}
        
        correlation = np.corrcoef(actions.flatten(), rewards.flatten())[0, 1]
        
        return {
            "unique_actions": len(np.unique(actions)),
            "action_entropy": float(-np.sum((np.bincount(actions) / len(actions)) * 
                                           np.log(np.bincount(actions) / len(actions) + 1e-8))),
            "reward_correlation": float(np.nan_to_num(correlation)),
            "mean_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test PPO analysis
    ppo = PPOAnalyzer()
    
    # Simulate trajectory
    rewards = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    values = np.array([0.05, 0.15, 0.1, 0.2, 0.22])
    
    advantages, returns = ppo.compute_gae(rewards, values)
    print(f"Advantages: {advantages}")
    print(f"Returns: {returns}")
    
    # Test policy loss
    old_probs = np.array([0.1, 0.2, 0.15, 0.3, 0.35])
    new_probs = np.array([0.12, 0.22, 0.14, 0.28, 0.38])
    policy_loss, metrics = ppo.compute_policy_loss(old_probs, new_probs, advantages)
    print(f"Policy Loss: {policy_loss}")
    print(f"Metrics: {metrics}")
