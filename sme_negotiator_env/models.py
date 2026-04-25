"""Canonical typed models for the SME negotiation environment.

These classes define the stable JSON contracts used by the OpenEnv server and
client. Stage 0 intentionally preserves the existing field names and validation
rules so the wire format remains unchanged.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from openenv.core import Action, Observation, State


class SMEAccountState(BaseModel):
    """Persistent SME-level state used by the multi-agent liquidity world.

    This model is not part of the Stage 0 single-deal OpenEnv contract. It is
    additive Stage 1 structure for tracking one SME across deals and later
    long-horizon scenarios.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sme_id: str
    cash_balance: float
    supplier_payment_days: int = Field(..., ge=0)
    credit_limit: float = Field(..., ge=0.0)
    current_utilization: float = Field(..., ge=0.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    required_minimum_cash: float = Field(0.0, ge=0.0)
    defaulted: bool = False
    missed_supplier_payment: bool = False


class BuyerState(BaseModel):
    """State for a buyer agent in the Stage 1 world model.

    Buyers are modeled as other agents in the environment rather than the
    external learner. Stage 1 keeps them scripted, with room for richer behavior
    in later stages.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    buyer_id: str
    demand_level: float = Field(..., ge=0.0)
    budget_per_period: float = Field(..., ge=0.0)
    default_tendency: float = Field(..., ge=0.0, le=1.0)
    baseline_payment_days: int = Field(..., ge=0)


class FinancierState(BaseModel):
    """State for the financier or TReDS-like counterparty in the world model.

    This actor is internal to the multi-agent world and is not the external
    learning agent.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    financier_id: str
    available_capital: float = Field(..., ge=0.0)
    risk_appetite: float = Field(..., ge=0.0, le=1.0)
    base_interest_rate: float = Field(..., ge=0.0, le=1.0)


class WorldSnapshot(BaseModel):
    """Aggregate world metrics captured when a macro period closes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    period_index: int = Field(..., ge=0)
    total_cash_balance: float
    defaulted_sme_count: int = Field(..., ge=0)
    open_deal_count: int = Field(..., ge=0)
    resolved_deal_count: int = Field(..., ge=0)
    average_payment_days: float = Field(..., ge=0.0)
    total_penalty_exposure: float = Field(..., ge=0.0)


class DealState(BaseModel):
    """Serializable macro-level record for one purchase-order negotiation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    deal_id: str
    sme_id: str
    buyer_id: str
    status: str = "open"
    created_period: int = Field(..., ge=0)
    agreement_period: Optional[int] = Field(None, ge=0)
    invoice_amount: float = Field(0.0, ge=0.0)
    supplier_payment_amount: float = Field(0.0, ge=0.0)
    agreed_price: Optional[float] = Field(None, ge=0.0)
    agreed_payment_days: Optional[int] = Field(None, ge=0)
    financed: bool = False
    finance_rate: float = Field(0.0, ge=0.0, le=1.0)
    supplier_due_period: Optional[int] = Field(None, ge=0)
    buyer_due_period: Optional[int] = Field(None, ge=0)
    supplier_paid: bool = False
    settled: bool = False
    failed: bool = False
    volume: int = Field(0, ge=0)
    dynamic_discounting: bool = False
    dynamic_discount_annual_rate: float = Field(0.0, ge=0.0, le=0.95)
    late_payment_penalty_agreed: bool = False
    financing_principal: float = Field(0.0, ge=0.0)
    accrued_interest: float = Field(0.0, ge=0.0)
    initial_funding_applied: bool = False


class CashflowProjection(BaseModel):
    """Deterministic macro cash projection returned by ``simulate_cashflow``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    period_balances: list[float] = Field(default_factory=list)
    period_defaults: list[bool] = Field(default_factory=list)
    period_penalties: list[float] = Field(default_factory=list)


class HistoryEvent(BaseModel):
    """Compact serializable event shown in liquidity observations and state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    actor: Literal["SME", "BUYER", "SYSTEM", "TOOL", "FINANCIER"]
    event_type: str
    period_index: int = Field(..., ge=0)
    deal_id: Optional[str] = None
    summary: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional["ToolResultEnvelope"] = None


class ToolResultEnvelope(BaseModel):
    """Normalized provenance envelope returned for every Theme 3.1 tool call."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source: Literal["deterministic", "live"] = "deterministic"
    backend_name: str
    request_id: str
    latency_ms: int = Field(0, ge=0)
    cache_hit: bool = False
    stale: bool = False
    normalized_payload: dict[str, Any] = Field(default_factory=dict)
    requested_source: Literal["deterministic", "live"] = "deterministic"
    fallback_reason: Optional[str] = None


class RewardComponentReport(BaseModel):
    """Pure additive reward report for one deal trajectory or episode slice."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    verifiable_reward: float = 0.0
    shaping_gap: float = 0.0
    shaping_days: float = 0.0
    shaping_alignment: float = 0.0
    shaping_total: float = 0.0
    tool_bonus: float = 0.0
    total_reward: float = 0.0
    npv_delta_vs_baseline: float = 0.0
    success_no_default_positive_npv: bool = False
    lambda_shaping: float = Field(0.1, ge=0.0)


class ToolCallRecord(BaseModel):
    """Serializable record for one deterministic Stage 4 tool invocation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step_index: int = Field(..., ge=0)
    period_index: int = Field(..., ge=0)
    deal_id: Optional[str] = None
    tool_name: Literal["QUERY_TREDS", "CHECK_COMPLIANCE", "RUN_CASHFLOW_SIM"]
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: ToolResultEnvelope
    context_fingerprint: str


class WorldState(BaseModel):
    """World-level state introduced in Stage 1.

    ``WorldState`` holds multiple SMEs, buyers, and an optional financier. It
    is internal structure for the new liquidity environment and is intended to
    support later long-horizon and world-modeling stages.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    smes: list[SMEAccountState]
    buyers: list[BuyerState]
    financier: Optional[FinancierState] = None
    legal_max_payment_days: int = Field(45, ge=0)
    baseline_discount_rate: float = Field(0.0, ge=0.0, le=1.0)
    reward_lambda_shaping: float = Field(0.1, ge=0.0)
    current_period: int = Field(0, ge=0)
    total_periods: int = Field(1, ge=1)
    episode_step: int = Field(0, ge=0)
    history: list[WorldSnapshot] = Field(default_factory=list)
    deals: list[DealState] = Field(default_factory=list)


class NegotiationAction(Action):
    """Public action contract sent by the SME agent on each ``step()`` call.

    This is the agent-controlled input to the environment. It is serialized over
    the OpenEnv HTTP/WebSocket boundary, so field names and validation behavior
    are part of the baseline interface and should remain stable.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: Literal[
        "propose", "accept", "reject", "simulate_plan", "advance_period", "tool",
        # Multi-agent Theme #1 action types
        "invoke_regulator",
        "request_financier_auction",
        "propose_split_deal",
        "signal_distress",
    ] = "propose"
    price: float = Field(0, description="Proposed price in ₹/unit", ge=0)
    payment_days: int = Field(0, description="Proposed payment days", ge=0)
    use_treds: bool = Field(False, description="Whether to propose TReDS financing")
    reason: Optional[str] = Field(None, description="Agent's reasoning (optional)")
    deal_id: Optional[str] = Field(None, description="Target deal for liquidity-environment actions")
    simulation_plan: Optional[dict[str, Any]] = Field(
        None,
        description="Pure planning payload used when action_type='simulate_plan'.",
    )
    simulation_horizon: Optional[int] = Field(
        None,
        ge=1,
        description="Optional macro horizon for deterministic plan simulation.",
    )
    tool_name: Optional[Literal["QUERY_TREDS", "CHECK_COMPLIANCE", "RUN_CASHFLOW_SIM"]] = Field(
        None,
        description="Deterministic tool name used when action_type='tool'.",
    )
    tool_args: Optional[dict[str, Any]] = Field(
        None,
        description="JSON-serializable deterministic tool arguments.",
    )
    # Medium / hard task extensions
    propose_late_payment_penalty_clause: bool = Field(
        False,
        description="If true, SME requests a contractual late-payment penalty (medium task).",
    )
    propose_dynamic_discounting: bool = Field(
        False,
        description="If true, SME proposes dynamic discounting for early payment (hard task).",
    )
    dynamic_discount_annual_rate: float = Field(
        0.0,
        ge=0.0,
        le=0.95,
        description="Annualized discount for early payment as a fraction (e.g. 0.08 = 8%).",
    )
    # Multi-agent Theme #1 extensions — all optional with defaults
    split_deal_buyer_a_days: Optional[int] = Field(
        None, ge=0, description="Days offered to buyer_a in a split-deal proposal."
    )
    split_deal_buyer_b_days: Optional[int] = Field(
        None, ge=0, description="Days offered to buyer_b in a split-deal proposal."
    )
    split_deal_buyer_a_price: Optional[float] = Field(
        None, ge=0.0, description="Price offered to buyer_a in a split-deal proposal."
    )
    split_deal_buyer_b_price: Optional[float] = Field(
        None, ge=0.0, description="Price offered to buyer_b in a split-deal proposal."
    )
    distress_disclosure_level: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Level of SME financial distress disclosed when action_type='signal_distress'."
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_tool_mode(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        action_type = str(payload.get("action_type", "propose")).lower()
        payload["action_type"] = action_type
        if action_type == "tool":
            if not payload.get("tool_name"):
                raise ValueError("tool_name is required when action_type='tool'")
            if payload.get("tool_args") is None:
                payload["tool_args"] = {}
        return payload


class NegotiationObservation(Observation):
    """Public observation returned from ``reset()`` and ``step()``.

    Observations expose the buyer's current counter-offer, task context, and the
    OpenEnv reward/done/metadata fields. They are a filtered view of the
    negotiation rather than the full internal episode state stored on the
    environment.
    """

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
    # Financial / task context
    task_name: str = ""
    sme_monthly_revenue: float = 0.0
    working_capital_gap: float = 0.0
    interest_rate_annual: float = 0.0
    buyer_power_score: float = 0.0
    secondary_buyer_power: Optional[float] = None
    current_payment_terms_days: int = 0
    sme_supplier_payment_days: int = 0


class LiquidityObservation(NegotiationObservation):
    """Stage 1 observation for the new world-level liquidity environment.

    This extends the baseline single-deal observation with explicit agent
    identity while leaving ``NegotiationObservation`` unchanged for the current
    OpenEnv contract.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    agent_type: str = "SME"
    agent_id: str = "sme_0"
    current_actor: str = "SME"
    active_deal_id: Optional[str] = None
    open_deal_ids: list[str] = Field(default_factory=list)
    resolved_deal_ids: list[str] = Field(default_factory=list)
    current_period: int = 0
    total_periods: int = 1
    episode_step: int = 0
    simulation_projection: Optional[CashflowProjection] = None
    projected_balances: Optional[list[float]] = None
    projected_defaults: Optional[list[bool]] = None
    projected_penalties: Optional[list[float]] = None
    last_tool_name: Optional[str] = None
    last_tool_args: Optional[dict[str, Any]] = None
    last_tool_result: Optional[ToolResultEnvelope] = None
    reward_component_report: Optional[RewardComponentReport] = None
    history: list[HistoryEvent] = Field(default_factory=list)


class NegotiationState(State):
    """Internal full episode state used by the simulator and terminal graders.

    ``NegotiationState`` is the environment-owned source of truth for the full
    negotiation state. It includes bookkeeping and task-specific financial
    fields that are not all exposed directly to the agent. Terminal graders
    consume this state to compute deterministic final scores.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    seed: int
    difficulty: str
    task_name: str = ""
    step_count: int
    max_steps: int
    negotiation_round: int = 0
    max_rounds: int = 0
    deal_reached: bool
    final_price: Optional[float]
    final_days: Optional[int]
    treds_used: bool
    cumulative_reward: float
    buyer_price: float
    buyer_days: int
    initial_buyer_days: int
    cost_threshold: float
    liquidity_threshold: int
    volume: int
    message: str
    # Financial realism
    sme_monthly_revenue: float = Field(..., ge=0.0)
    current_payment_terms_days: int = Field(..., ge=0)
    sme_supplier_payment_days: int = Field(..., ge=0)
    interest_rate_annual: float = Field(0.22, ge=0.0, le=1.0)
    buyer_power_score: float = Field(0.5, ge=0.0, le=1.0)
    secondary_buyer_power: Optional[float] = Field(None, ge=0.0, le=1.0)
    agreed_terms: Optional[int] = Field(None, ge=0)
    late_payment_penalty_agreed: bool = False
    dynamic_discounting_agreed: bool = False
    agreed_dynamic_discount_annual: float = Field(0.0, ge=0.0, le=0.95)
    sme_id: str = "sme_0"
    buyer_id: str = "buyer_0"
    financier_id: Optional[str] = None
    deal_id: Optional[str] = None

    @computed_field
    @property
    def working_capital_gap(self) -> float:
        """INR tied up between supplier cash-out and buyer cash-in (simplified annualized gap)."""
        return self.sme_monthly_revenue * (
            self.current_payment_terms_days - self.sme_supplier_payment_days
        ) / 365.0


class LiquidityEnvironmentState(State):
    """Serializable state snapshot for ``SMELiquidityEnvironment``.

    The state exposes world metadata, the active negotiation, and the last
    financier quote while keeping the existing Stage 0 state model unchanged.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    seed: int
    task_name: str
    difficulty: str
    step_count: int
    max_steps: int
    current_actor_type: str = "SME"
    current_actor_id: str = "sme_0"
    active_sme_id: str = "sme_0"
    active_buyer_id: str = "buyer_0"
    world_state: WorldState
    current_negotiation: NegotiationState
    active_deal_id: Optional[str] = None
    current_negotiations: dict[str, NegotiationState] = Field(default_factory=dict)
    deal_trajectories: dict[str, list[NegotiationState]] = Field(default_factory=dict)
    trajectory: list[NegotiationState] = Field(default_factory=list)
    shaping_rewards: list[float] = Field(default_factory=list)
    cumulative_rl_reward: float = 0.0
    latest_verifiable_reward: Optional[float] = None
    legacy_last_reward: Optional[float] = None
    last_financier_quote_rate: Optional[float] = None
    resolved_deal_ids: list[str] = Field(default_factory=list)
    last_simulation_result: Optional[CashflowProjection] = None
    tool_history: list[ToolCallRecord] = Field(default_factory=list)
    last_tool_call: Optional[ToolCallRecord] = None
    history_tail: list[HistoryEvent] = Field(default_factory=list)
    pending_tool_bonus: float = 0.0
    active_reward_component_report: Optional[RewardComponentReport] = None
    tool_call_count: int = Field(0, ge=0)
    tool_effective_count: int = Field(0, ge=0)
    duplicate_tool_count: int = Field(0, ge=0)
    invalid_action_count: int = Field(0, ge=0)
    stall_step_count: int = Field(0, ge=0)
    terminated_by_step_cap: bool = False
    episode_step_cap: int = Field(0, ge=0)
    tool_backend_mode: Literal["deterministic", "live"] = "deterministic"


def default_negotiation_state(
    *,
    episode_id: str,
    seed: int,
    difficulty: str,
    task_name: str,
    max_steps: int,
    max_rounds: int,
    buyer_price: float,
    buyer_days: int,
    initial_buyer_days: int,
    cost_threshold: float,
    liquidity_threshold: int,
    volume: int,
    sme_monthly_revenue: float = 500_000.0,
    current_payment_terms_days: int = 90,
    sme_supplier_payment_days: int = 30,
    interest_rate_annual: float = 0.22,
    buyer_power_score: float = 0.4,
    secondary_buyer_power: Optional[float] = None,
    sme_id: str = "sme_0",
    buyer_id: str = "buyer_0",
    financier_id: Optional[str] = None,
    deal_id: Optional[str] = None,
    message: str = "",
) -> NegotiationState:
    """Construct the canonical internal state for a fresh negotiation episode.

    The factory centralizes the default shape of ``NegotiationState`` so the
    environment can build a complete internal state object without changing the
    public action/observation contracts. The return value is the authoritative
    internal state; observations are derived from it later in the environment.
    """
    return NegotiationState(
        episode_id=episode_id,
        seed=seed,
        difficulty=difficulty,
        task_name=task_name,
        step_count=0,
        max_steps=max_steps,
        negotiation_round=0,
        max_rounds=max_rounds,
        deal_reached=False,
        final_price=None,
        final_days=None,
        treds_used=False,
        cumulative_reward=0.0,
        buyer_price=buyer_price,
        buyer_days=buyer_days,
        initial_buyer_days=initial_buyer_days,
        cost_threshold=cost_threshold,
        liquidity_threshold=liquidity_threshold,
        volume=volume,
        message=message,
        sme_monthly_revenue=sme_monthly_revenue,
        current_payment_terms_days=current_payment_terms_days,
        sme_supplier_payment_days=sme_supplier_payment_days,
        interest_rate_annual=interest_rate_annual,
        buyer_power_score=buyer_power_score,
        secondary_buyer_power=secondary_buyer_power,
        agreed_terms=None,
        late_payment_penalty_agreed=False,
        dynamic_discounting_agreed=False,
        agreed_dynamic_discount_annual=0.0,
        sme_id=sme_id,
        buyer_id=buyer_id,
        financier_id=financier_id,
        deal_id=deal_id,
    )


HistoryEvent.model_rebuild()
ToolCallRecord.model_rebuild()
LiquidityObservation.model_rebuild()
LiquidityEnvironmentState.model_rebuild()


# ======================================================================= #
# Multi-Agent Theme #1 data contracts                                       #
# These classes are defined here to avoid circular imports with the        #
# sme_negotiator_env.agents sub-package.                                   #
# ======================================================================= #

class RegulatoryWarning(BaseModel):
    """A live MSMED Act / Section 43B(h) warning for one deal-buyer pair."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    deal_id: str
    buyer_id: str
    violation_type: Literal[
        "exceeds_45_days",
        "no_penalty_clause",
        "discount_rate_cap",
        "tax_deduction_at_risk",
    ]
    penalty_exposure_inr: float = Field(0.0, ge=0.0)
    section_reference: str = ""
    is_active: bool = True
    issued_period: int = Field(0, ge=0)
    issued_by_sme: bool = False


class FinancierBid(BaseModel):
    """One financing bid from a competitive financier in the auction."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    financier_id: str
    financier_type: Literal["treds_platform", "bank", "nbfc", "microfinance"]
    annual_rate: float = Field(ge=0.0, le=1.0)
    max_tenor_days: int = Field(ge=0)
    advance_amount: float = Field(ge=0.0)
    approval_probability: float = Field(0.9, ge=0.0, le=1.0)
    conditions: list[str] = Field(default_factory=list)


class OpponentSignal(BaseModel):
    """A structured signal emitted by a non-SME agent during a negotiation step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sender_id: str
    sender_type: Literal["buyer", "financier", "regulator", "coalition"]
    signal_type: Literal[
        "counter_offer",
        "coalition_formed",
        "coalition_dissolved",
        "financier_bid",
        "regulatory_warning",
        "defection_intent",
        "distress_ack",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)
    round_number: int = Field(0, ge=0)


class CoalitionStatus(BaseModel):
    """Serializable snapshot of the current buyer coalition lifecycle state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    is_active: bool
    buyer_ids: list[str] = Field(default_factory=list)
    formed_at_round: Optional[int] = Field(None, ge=0)
    joint_demand_days: Optional[int] = Field(None, ge=0)
    defection_risk: float = Field(0.0, ge=0.0, le=1.0)


class MultiAgentObservation(LiquidityObservation):
    """Theme #1 observation extending LiquidityObservation with multi-agent signals.

    All new fields have safe defaults so existing code that consumes
    LiquidityObservation (or NegotiationObservation) continues to work
    without modification.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    opponent_signals: list[OpponentSignal] = Field(default_factory=list)
    coalition_status: Optional[CoalitionStatus] = None
    regulatory_warnings: list[RegulatoryWarning] = Field(default_factory=list)
    financier_bids: list[FinancierBid] = Field(default_factory=list)
    sme_belief_estimate: dict[str, Any] = Field(default_factory=dict)
    social_welfare_score: float = Field(0.0, ge=0.0, le=1.0)
    buyer_surplus_estimate: float = Field(0.0, ge=0.0, le=1.0)


MultiAgentObservation.model_rebuild()
