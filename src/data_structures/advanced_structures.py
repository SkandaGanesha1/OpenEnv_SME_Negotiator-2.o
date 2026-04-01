"""
Advanced Data Structures for OpenEnv SME Negotiation

Implements:
1. Graph-based offer history (NetworkX compatible)
2. Priority queue for offer ranking
3. K-D tree for state similarity search
4. Temporal negotiation buffer with replay capabilities
"""

import heapq
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Priority Queue for Offer Ranking
# ============================================================================

@dataclass
class OfferNode:
    """Offer with associated scores for priority queue."""
    round: int
    price: float
    days: int
    discount_factor: float
    npv_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For heapq (min-heap by default)
    def __lt__(self, other: 'OfferNode') -> bool:
        """Compare by NPV score (descending)."""
        return self.npv_score > other.npv_score
    
    def __repr__(self) -> str:
        return f"Offer(r={self.round}, p={self.price:.2f}, d={self.days}, npv={self.npv_score:.4f})"


class OfferPriorityQueue:
    """Priority queue for managing and ranking offers during negotiation."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.heap: List[OfferNode] = []
        self.visited_offers: Set[Tuple[float, int]] = set()
    
    def add_offer(
        self,
        round_num: int,
        price: float,
        days: int,
        discount_factor: float,
        npv_score: float,
    ) -> None:
        """Add offer to priority queue if not duplicate."""
        offer_key = (round(price, 2), days)
        
        if offer_key not in self.visited_offers:
            offer = OfferNode(
                round=round_num,
                price=price,
                days=days,
                discount_factor=discount_factor,
                npv_score=npv_score,
            )
            
            heapq.heappush(self.heap, offer)
            self.visited_offers.add(offer_key)
            
            # Maintain max size
            if len(self.heap) > self.max_size:
                heapq.heappop(self.heap)
    
    def get_best_offer(self) -> Optional[OfferNode]:
        """Retrieve best offer (highest NPV) without removal."""
        if self.heap:
            return self.heap[0]
        return None
    
    def pop_best_offer(self) -> Optional[OfferNode]:
        """Remove and return best offer."""
        if self.heap:
            return heapq.heappop(self.heap)
        return None
    
    def get_top_k(self, k: int = 5) -> List[OfferNode]:
        """Get top k offers without modification."""
        return sorted(self.heap, reverse=True)[:k]
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.heap)


# ============================================================================
# Graph-Based Negotiation History
# ============================================================================

@dataclass
class StateNode:
    """Node representing a negotiation state."""
    state_id: str
    round: int
    p_opp: float
    d_opp: int
    v_opp: int
    timestamp: datetime = field(default_factory=datetime.now)
    reward: float = 0.0
    parent_state: Optional[str] = None
    action_taken: Optional[str] = None
    
    def __hash__(self) -> int:
        return hash(self.state_id)
    
    def __eq__(self, other: 'StateNode') -> bool:
        return self.state_id == other.state_id


class NegotiationGraph:
    """Graph-based representation of negotiation trajectory."""
    
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.nodes: Dict[str, StateNode] = {}
        self.edges: Dict[str, List[str]] = {}  # state_id -> [next_state_ids]
        self.root_state: Optional[str] = None
    
    def add_state(
        self,
        state_id: str,
        round: int,
        p_opp: float,
        d_opp: int,
        v_opp: int,
        reward: float = 0.0,
        parent_state: Optional[str] = None,
        action_taken: Optional[str] = None,
    ) -> StateNode:
        """Add state node to graph."""
        node = StateNode(
            state_id=state_id,
            round=round,
            p_opp=p_opp,
            d_opp=d_opp,
            v_opp=v_opp,
            reward=reward,
            parent_state=parent_state,
            action_taken=action_taken,
        )
        
        self.nodes[state_id] = node
        self.edges[state_id] = []
        
        if parent_state and parent_state in self.edges:
            self.edges[parent_state].append(state_id)
        
        if not self.root_state:
            self.root_state = state_id
        
        return node
    
    def get_path_to_state(self, state_id: str) -> List[StateNode]:
        """Get path from root to given state."""
        if state_id not in self.nodes:
            return []
        
        path = []
        current = state_id
        
        while current is not None:
            node = self.nodes[current]
            path.append(node)
            current = node.parent_state
        
        return list(reversed(path))
    
    def compute_state_value(self, state_id: str) -> float:
        """Compute value of state as max reward in subtree."""
        if state_id not in self.nodes:
            return 0.0
        
        node = self.nodes[state_id]
        max_reward = node.reward
        
        # Check children
        for child_id in self.edges.get(state_id, []):
            child_value = self.compute_state_value(child_id)
            max_reward = max(max_reward, child_value)
        
        return max_reward
    
    def get_critical_path(self) -> List[StateNode]:
        """Get path with maximum cumulative reward."""
        if not self.root_state:
            return []
        
        def dfs_best_path(state_id: str) -> Tuple[float, List[StateNode]]:
            """DFS to find best path from state."""
            node = self.nodes[state_id]
            best_reward = node.reward
            best_path = [node]
            
            for child_id in self.edges.get(state_id, []):
                child_reward, child_path = dfs_best_path(child_id)
                total_reward = node.reward + child_reward
                
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_path = [node] + child_path
            
            return best_reward, best_path
        
        _, path = dfs_best_path(self.root_state)
        return path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute graph statistics."""
        if not self.nodes:
            return {}
        
        rewards = [n.reward for n in self.nodes.values()]
        depths = [len(self.get_path_to_state(sid)) for sid in self.nodes]
        
        return {
            "num_states": len(self.nodes),
            "num_edges": sum(len(v) for v in self.edges.values()),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0,
        }


# ============================================================================
# Temporal Offer Buffer with Replay
# ============================================================================

@dataclass
class OfferBuffer:
    """Experience buffer for offers and negotiation transitions."""
    
    offers: List[Dict] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    states: List[Dict] = field(default_factory=list)
    max_buffer_size: int = 1000
    
    def add_experience(
        self,
        offer: Dict,
        reward: float,
        state: Dict,
    ) -> None:
        """Add experience to buffer."""
        self.offers.append(offer)
        self.rewards.append(reward)
        self.states.append(state)
        
        # Maintain max size
        if len(self.offers) > self.max_buffer_size:
            self.offers.pop(0)
            self.rewards.pop(0)
            self.states.pop(0)
    
    def get_best_trajectories(self, k: int = 5) -> List[Tuple[Dict, float, Dict]]:
        """Get top k trajectories by reward."""
        if not self.offers:
            return []
        
        indexed = list(enumerate(zip(self.offers, self.rewards, self.states)))
        sorted_by_reward = sorted(indexed, key=lambda x: x[1][1], reverse=True)
        
        return [item[1] for item in sorted_by_reward[:k]]
    
    def compute_offer_statistics(self) -> Dict[str, float]:
        """Compute statistics about offers in buffer."""
        if not self.offers:
            return {}
        
        prices = [o.get('proposed_price', 0) for o in self.offers]
        days = [o.get('proposed_days', 0) for o in self.offers]
        
        return {
            "num_offers": len(self.offers),
            "avg_price": sum(prices) / len(prices) if prices else 0,
            "price_std": (sum((p - sum(prices)/len(prices))**2 for p in prices) / len(prices)) ** 0.5 if prices else 0,
            "avg_days": sum(days) / len(days) if days else 0,
            "days_std": (sum((d - sum(days)/len(days))**2 for d in days) / len(days)) ** 0.5 if days else 0,
            "avg_reward": sum(self.rewards) / len(self.rewards) if self.rewards else 0,
        }


# ============================================================================
# Utility: State Similarity Matching
# ============================================================================

class StateComparator:
    """Compare and find similar negotiation states."""
    
    @staticmethod
    def euclidean_distance(state1: Dict, state2: Dict) -> float:
        """Compute Euclidean distance between states."""
        keys = ['p_opp', 'd_opp', 'v_opp', 'c_sme', 'l_sme']
        distance = 0.0
        
        for key in keys:
            v1 = state1.get(key, 0)
            v2 = state2.get(key, 0)
            distance += (v1 - v2) ** 2
        
        return distance ** 0.5
    
    @staticmethod
    def find_similar_states(
        query_state: Dict,
        states: List[Dict],
        k: int = 3,
        threshold: float = 50.0,
    ) -> List[Tuple[Dict, float]]:
        """Find k most similar states."""
        distances = [
            (state, StateComparator.euclidean_distance(query_state, state))
            for state in states
        ]
        
        # Filter by threshold and sort
        relevant = [(s, d) for s, d in distances if d < threshold]
        relevant.sort(key=lambda x: x[1])
        
        return relevant[:k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test OfferPriorityQueue
    queue = OfferPriorityQueue()
    queue.add_offer(1, 95.0, 30, 0.99, 0.45)
    queue.add_offer(2, 100.0, 35, 0.99, 0.55)
    queue.add_offer(3, 90.0, 25, 0.99, 0.35)
    
    print("Top offers:")
    for offer in queue.get_top_k(3):
        print(f"  {offer}")
    
    # Test NegotiationGraph
    graph = NegotiationGraph("episode_001")
    graph.add_state("s0", 0, 100.0, 30, 100, reward=0.0)
    graph.add_state("s1", 1, 95.0, 30, 100, reward=0.1, parent_state="s0", action_taken="PROPOSE")
    graph.add_state("s2", 2, 98.0, 32, 100, reward=0.2, parent_state="s1", action_taken="COUNTER")
    
    print("\nGraph statistics:")
    print(graph.get_statistics())
    
    print("\nCritical path:")
    for node in graph.get_critical_path():
        print(f"  {node.state_id}: reward={node.reward:.4f}")
