"""Typed OpenEnv models for the SME negotiation environment."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import ConfigDict, Field
from openenv.core import Action, Observation, State


class NegotiationAction(Action):
    """Action the SME agent can take each round."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: Literal["propose", "accept", "reject"] = "propose"
    price: float = Field(..., description="Proposed price in ₹/unit", ge=0)
    payment_days: int = Field(..., description="Proposed payment days", ge=0)
    use_treds: bool = Field(False, description="Whether to propose TReDS financing")
    reason: Optional[str] = Field(None, description="Agent's reasoning (optional)")


class NegotiationObservation(Observation):
    """What the agent sees after each step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    round_number: int
    max_rounds: int
    buyer_price: float
    buyer_days: int
    buyer_accepted: bool
    negotiation_done: bool
    cost_threshold: float
    liquidity_threshold: int
    volume: int
    difficulty: str
    price_score: float
    days_score: float
    treds_bonus: float
    step_reward: float
    message: str


class NegotiationState(State):
    """Full episode state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    seed: int
    difficulty: str
    step_count: int
    max_steps: int
    deal_reached: bool
    final_price: Optional[float]
    final_days: Optional[int]
    treds_used: bool
    cumulative_reward: float
    buyer_price: float
    buyer_days: int
    cost_threshold: float
    liquidity_threshold: int
    volume: int
    message: str
