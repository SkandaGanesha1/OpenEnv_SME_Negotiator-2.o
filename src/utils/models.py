"""Data models for SME Negotiation environment."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from pydantic import ConfigDict, Field
from openenv.core.env_server.types import Action, Observation


class ActionType(Enum):
    """Negotiation action types."""
    PROPOSE = "PROPOSE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class OfferStatus(Enum):
    """Status of an offer."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"


@dataclass
class NegotiationTerms:
    """Contract terms for negotiation - used for final terms."""
    final_price: float  # Final agreed price (₹/unit)
    final_days: int     # Final agreed payment days
    final_volume: int   # Final agreed volume (units)
    treds_utilized: bool = False  # Whether TReDS was used
    
    # Keep backward compatibility with simple terms too
    price: Optional[float] = None  # ₹/unit
    days: Optional[int] = None     # Payment terms (days)
    volume: Optional[int] = None   # Units to be supplied
    
    def __post_init__(self):
        """Set simple terms if final terms are provided."""
        if self.price is None and self.final_price is not None:
            self.price = self.final_price
        if self.days is None and self.final_days is not None:
            self.days = self.final_days
        if self.volume is None and self.final_volume is not None:
            self.volume = self.final_volume

    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible model_dump method for serialization."""
        return self.to_dict()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'final_price': float(self.final_price),
            'final_days': int(self.final_days),
            'final_volume': int(self.final_volume),
            'treds_utilized': bool(self.treds_utilized),
            'price': float(self.price) if self.price else None,
            'days': int(self.days) if self.days else None,
            'volume': int(self.volume) if self.volume else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NegotiationTerms':
        """Create from dictionary."""
        return cls(
            final_price=float(data['final_price']),
            final_days=int(data['final_days']),
            final_volume=int(data['final_volume']),
            treds_utilized=bool(data.get('treds_utilized', False)),
            price=float(data['price']) if data.get('price') else None,
            days=int(data['days']) if data.get('days') else None,
            volume=int(data['volume']) if data.get('volume') else None
        )


class NegotiationAction(Action):
    """Action taken in negotiation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: str  # "PROPOSE", "ACCEPT", "REJECT"
    proposed_price: Optional[float] = None  # Proposed price (₹/unit)
    proposed_days: Optional[int] = None     # Proposed payment terms (days)
    request_treds: bool = False              # Request TReDS financing
    justification: Optional[str] = None      # Text justification
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def validate_action(self) -> Tuple[bool, str]:
        """
        Validate the action.
        
        Returns:
            (is_valid, message)
        """
        if self.action_type == "PROPOSE":
            if self.proposed_price is None:
                return False, "PROPOSE action requires proposed_price"
            if self.proposed_days is None:
                return False, "PROPOSE action requires proposed_days"
            if self.proposed_price <= 0:
                return False, "proposed_price must be positive"
            if self.proposed_days <= 0:
                return False, "proposed_days must be positive"
            if self.justification:
                words = len(self.justification.split())
                if words > 500:
                    return False, f"Justification exceeds 500 words ({words} words)"
        elif self.action_type == "ACCEPT":
            if self.proposed_price is None:
                return False, "ACCEPT action requires proposed_price"
            if self.proposed_days is None:
                return False, "ACCEPT action requires proposed_days"
        elif self.action_type == "REJECT":
            # REJECT can be sent without additional parameters
            pass
        else:
            return False, f"Unknown action_type: {self.action_type}"
        
        return True, "Valid action"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NegotiationAction':
        """Create from dictionary."""
        return cls(
            action_type=data['action_type'],
            proposed_price=float(data['proposed_price']) if data.get('proposed_price') else None,
            proposed_days=int(data['proposed_days']) if data.get('proposed_days') else None,
            request_treds=bool(data.get('request_treds', False)),
            justification=data.get('justification')
        )


@dataclass
class OfferRecord:
    """Record of an offer made in negotiation."""
    round: int
    proposed_price: float
    proposed_days: int
    request_treds: bool
    justification: str
    party: str  # "buyer" or "sme"

    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible model_dump method for serialization."""
        return self.to_dict()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'round': int(self.round),
            'proposed_price': float(self.proposed_price),
            'proposed_days': int(self.proposed_days),
            'request_treds': bool(self.request_treds),
            'justification': self.justification,
            'party': self.party
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OfferRecord':
        """Create from dictionary."""
        return cls(
            round=int(data['round']),
            proposed_price=float(data['proposed_price']),
            proposed_days=int(data['proposed_days']),
            request_treds=bool(data['request_treds']),
            justification=data['justification'],
            party=data['party']
        )


class NegotiationState(Observation):
    """Current state of negotiation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = ""
    t_elapsed: int = 0  # Elapsed rounds
    t_max: int = 10     # Maximum rounds allowed
    episode_seed: int = 0  # Random seed for episode reproducibility
    
    # Current opponent offers
    p_opp: float = 100.0  # Opponent's current price (₹/unit)
    d_opp: int = 30       # Opponent's current payment days
    v_opp: int = 100      # Volume in units
    treds_opp: bool = False  # Opponent requesting TReDS
    
    # SME state
    c_sme: float = 50.0   # SME cost per unit
    l_sme: float = 100.0  # SME liquidity
    
    # Discount rate for TReDS
    r_discount: float = 0.08  # 8% discount rate
    
    # Negotiation history
    history: List[OfferRecord] = Field(default_factory=list)
    
    # Status
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __iter__(self):
        """Backwards compatibility for callers unpacking step() into 4 values."""

        yield self
        yield float(self.reward)
        yield bool(self.done)
        yield dict(self.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            't_elapsed': int(self.t_elapsed),
            't_max': int(self.t_max),
            'episode_seed': int(self.episode_seed),
            'p_opp': float(self.p_opp),
            'd_opp': int(self.d_opp),
            'v_opp': int(self.v_opp),
            'treds_opp': bool(self.treds_opp),
            'c_sme': float(self.c_sme),
            'l_sme': float(self.l_sme),
            'r_discount': float(self.r_discount),
            'done': bool(self.done),
            'reward': float(self.reward),
            'metadata': dict(self.metadata),
            'history': [o.to_dict() for o in self.history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NegotiationState':
        """Create state from dictionary."""
        history = []
        if 'history' in data:
            history = [OfferRecord.from_dict(o) for o in data['history']]
        
        return cls(
            task_id=data.get('task_id', ''),
            t_elapsed=int(data.get('t_elapsed', 0)),
            t_max=int(data.get('t_max', 10)),
            episode_seed=int(data.get('episode_seed', 0)),
            p_opp=float(data.get('p_opp', 100.0)),
            d_opp=int(data.get('d_opp', 30)),
            v_opp=int(data.get('v_opp', 100)),
            treds_opp=bool(data.get('treds_opp', False)),
            c_sme=float(data.get('c_sme', 50.0)),
            l_sme=float(data.get('l_sme', 100.0)),
            r_discount=float(data.get('r_discount', 0.08)),
            done=bool(data.get('done', False)),
            reward=float(data.get('reward', 0.0)),
            metadata=dict(data.get('metadata', {})),
            history=history
        )


@dataclass
class EpisodeResult:
    """Result of a complete negotiation episode."""
    success: bool  # Whether the agent successfully negotiated a deal
    terms: Optional[NegotiationTerms] = None  # Final negotiated terms
    score: float = 0.0  # Deterministic score (0-1 range)
    npv_base: float = 0.0  # Base NPV calculation
    final_utility: float = 0.0  # Final utility score
    round_completed: int = 0  # Number of rounds completed
    u_max: float = 0.0  # Maximum possible utility
    u_min: float = 0.0  # Minimum possible utility (break-even)
    treds_utilized: bool = False  # Whether TReDS was used
    deal_reached: bool = False
    final_price: Optional[float] = None
    final_days: Optional[int] = None
    total_rounds: int = 0
    total_reward: float = 0.0
    normalized_reward: float = 0.0
    reason: str = ""
    failure_reason: Optional[str] = None  # Reason for failure if applicable
    
    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible model_dump method for serialization."""
        return self.to_dict()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': bool(self.success),
            'terms': self.terms.to_dict() if self.terms else None,
            'score': float(self.score),
            'npv_base': float(self.npv_base),
            'final_utility': float(self.final_utility),
            'round_completed': int(self.round_completed),
            'u_max': float(self.u_max),
            'u_min': float(self.u_min),
            'treds_utilized': bool(self.treds_utilized),
            'deal_reached': bool(self.deal_reached),
            'final_price': float(self.final_price) if self.final_price else None,
            'final_days': int(self.final_days) if self.final_days else None,
            'total_rounds': int(self.total_rounds),
            'total_reward': float(self.total_reward),
            'normalized_reward': float(self.normalized_reward),
            'reason': self.reason,
            'failure_reason': self.failure_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeResult':
        """Create from dictionary."""
        terms = None
        if data.get('terms'):
            terms = NegotiationTerms.from_dict(data['terms'])
        
        return cls(
            success=bool(data['success']),
            terms=terms,
            score=float(data.get('score', 0.0)),
            npv_base=float(data.get('npv_base', 0.0)),
            final_utility=float(data.get('final_utility', 0.0)),
            round_completed=int(data.get('round_completed', 0)),
            u_max=float(data.get('u_max', 0.0)),
            u_min=float(data.get('u_min', 0.0)),
            treds_utilized=bool(data.get('treds_utilized', False)),
            deal_reached=bool(data.get('deal_reached', False)),
            final_price=float(data['final_price']) if data.get('final_price') else None,
            final_days=int(data['final_days']) if data.get('final_days') else None,
            total_rounds=int(data.get('total_rounds', 0)),
            total_reward=float(data.get('total_reward', 0.0)),
            normalized_reward=float(data.get('normalized_reward', 0.0)),
            reason=data.get('reason', ''),
            failure_reason=data.get('failure_reason')
        )

