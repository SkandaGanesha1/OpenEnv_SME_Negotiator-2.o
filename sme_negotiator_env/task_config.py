"""Task definitions for the SME negotiation hackathon gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Optional


@dataclass(frozen=True)
class TaskConfig:
    """Static configuration for one evaluation task."""

    name: str
    description: str
    difficulty: str
    # Negotiation / buyer dynamics (existing sim)
    initial_buyer_price: float
    initial_buyer_days: int
    max_rounds: int
    volume: int
    cost_threshold: float
    liquidity_threshold: int
    concede_low: float
    concede_high: float
    day_floor: int
    day_step_low: int
    day_step_high: int
    # Financial / strategic (exposed in state & observation)
    sme_monthly_revenue: float
    current_payment_terms_days: int
    sme_supplier_payment_days: int
    interest_rate_annual: float
    buyer_power_score: float
    secondary_buyer_power: Optional[float]
    negotiation_round_start: int
    # Which grader key in ``sme_negotiator_env.graders.TASK_GRADERS``
    grader_id: str
    context_note: str = ""
    minimum_cash_buffer_ratio: float = 0.25
    initial_cash_balance_ratio: float = 0.5
    credit_limit_multiplier: float = 1.5
    primary_buyer_default_tendency: float = 0.15
    secondary_buyer_default_tendency: Optional[float] = None
    financier_capital_multiplier: float = 10.0
    financier_risk_appetite: float = 0.6
    legal_max_payment_days: int = 45
    reward_lambda_shaping: float = 0.1


def default_task_config(task_id: str) -> TaskConfig:
    """Factory: realistic defaults per task."""
    if task_id == "payment-terms-easy":
        return TaskConfig(
            name="payment-terms-easy",
            description="Single SME vs cooperative buyer: reduce payment terms from 90d toward 60d or better.",
            difficulty="easy",
            initial_buyer_price=100.0,
            initial_buyer_days=90,
            max_rounds=10,
            volume=1000,
            cost_threshold=80.0,
            liquidity_threshold=60,
            concede_low=0.012,
            concede_high=0.035,
            day_floor=50,
            day_step_low=2,
            day_step_high=5,
            sme_monthly_revenue=500_000.0,
            current_payment_terms_days=90,
            sme_supplier_payment_days=30,
            interest_rate_annual=0.22,
            buyer_power_score=0.3,
            secondary_buyer_power=None,
            negotiation_round_start=0,
            grader_id="payment-terms-easy",
            context_note="Single buyer; target payment terms 90→≤60 days.",
        )
    if task_id == "payment-terms-medium":
        return TaskConfig(
            name="payment-terms-medium",
            description=(
                "SME under working-capital stress: tighten 60 to 45 days or better and agree a late payment "
                "penalty clause with a neutral buyer."
            ),
            difficulty="medium",
            initial_buyer_price=100.0,
            initial_buyer_days=60,
            max_rounds=12,
            volume=1000,
            cost_threshold=80.0,
            liquidity_threshold=45,
            concede_low=0.006,
            concede_high=0.018,
            day_floor=35,
            day_step_low=2,
            day_step_high=5,
            sme_monthly_revenue=500_000.0,
            # Status-quo receivable horizon for WCGap (days); stress vs 60d buyer opening
            current_payment_terms_days=75,
            sme_supplier_payment_days=30,
            interest_rate_annual=0.22,
            buyer_power_score=0.6,
            secondary_buyer_power=None,
            negotiation_round_start=0,
            grader_id="payment-terms-medium",
            context_note=(
                "Long receivable vs short supplier pay drives working-capital stress (see working_capital_gap). "
                "Negotiate ≤45 days where possible and a late payment penalty clause for partial credit."
            ),
        )
    if task_id == "payment-terms-hard":
        return TaskConfig(
            name="payment-terms-hard",
            description=(
                "Two-buyer consortium (hostile): negotiate dynamic discounting for faster payment; "
                "reward from NPV of financing vs status quo."
            ),
            difficulty="hard",
            initial_buyer_price=96.0,
            initial_buyer_days=100,
            max_rounds=16,
            volume=5000,
            cost_threshold=78.0,
            liquidity_threshold=55,
            concede_low=0.003,
            concede_high=0.009,
            day_floor=45,
            day_step_low=1,
            day_step_high=3,
            sme_monthly_revenue=600_000.0,
            current_payment_terms_days=120,
            sme_supplier_payment_days=25,
            interest_rate_annual=0.22,
            buyer_power_score=0.85,
            secondary_buyer_power=0.82,
            negotiation_round_start=0,
            grader_id="payment-terms-hard",
            context_note="Consortium of two buyers; leverage is high — use dynamic discounting.",
        )
    if task_id == "liquidity-stress-medium":
        return TaskConfig(
            name="liquidity-stress-medium",
            description=(
                "Stage 2 liquidity stress: a thin-cash SME faces a primary slow payer, a riskier secondary buyer, "
                "and modest financing capacity."
            ),
            difficulty="medium",
            initial_buyer_price=99.0,
            initial_buyer_days=75,
            max_rounds=14,
            volume=1500,
            cost_threshold=82.0,
            liquidity_threshold=40,
            concede_low=0.005,
            concede_high=0.014,
            day_floor=35,
            day_step_low=2,
            day_step_high=5,
            sme_monthly_revenue=400_000.0,
            current_payment_terms_days=85,
            sme_supplier_payment_days=25,
            interest_rate_annual=0.24,
            buyer_power_score=0.72,
            secondary_buyer_power=0.76,
            negotiation_round_start=0,
            grader_id="payment-terms-medium",
            context_note=(
                "Thin cash buffers and tighter supplier cycles make solvency fragile; the secondary buyer is riskier "
                "and financing capacity is limited."
            ),
            # Headroom calibrated so a competent agent (negotiates short tenor +
            # uses TReDS responsibly) can finish without default while a naive
            # always-accept-on-buyer-floor policy still risks tripping the
            # solvency constraint. Without this, supplier payments alone
            # exceed initial cash + financier advance and every rollout
            # terminates with verifiable_reward = 0.
            initial_cash_balance_ratio=0.6,
            credit_limit_multiplier=1.5,
            minimum_cash_buffer_ratio=0.20,
            primary_buyer_default_tendency=0.25,
            secondary_buyer_default_tendency=0.35,
            financier_capital_multiplier=4.0,
            financier_risk_appetite=0.45,
            legal_max_payment_days=45,
            reward_lambda_shaping=0.1,
        )
    if task_id == "liquidity-correlation-hard":
        return TaskConfig(
            name="liquidity-correlation-hard",
            description=(
                "Stage 2 correlated-risk hard mode: both buyers are slow and risky, the SME has very thin buffers, "
                "and the financier cannot absorb every invoice."
            ),
            difficulty="hard",
            initial_buyer_price=96.0,
            initial_buyer_days=95,
            max_rounds=18,
            volume=6000,
            cost_threshold=79.0,
            liquidity_threshold=35,
            concede_low=0.002,
            concede_high=0.008,
            day_floor=30,
            day_step_low=1,
            day_step_high=3,
            sme_monthly_revenue=450_000.0,
            current_payment_terms_days=105,
            sme_supplier_payment_days=20,
            interest_rate_annual=0.28,
            buyer_power_score=0.90,
            secondary_buyer_power=0.88,
            negotiation_round_start=0,
            grader_id="payment-terms-hard",
            context_note=(
                "Correlated buyer risk, tight supplier terms, and limited financing force solvency and NPV trade-offs."
            ),
            # Hard task: narrow but achievable solvency margin. Initial cash
            # large enough to cover one supplier cycle so a plan-aware agent
            # can earn non-zero verifiable reward without the rollout always
            # ending in default. Financier capital is still scarce so the
            # agent must prefer dynamic discounting + TReDS selectively.
            initial_cash_balance_ratio=0.65,
            credit_limit_multiplier=1.2,
            minimum_cash_buffer_ratio=0.25,
            primary_buyer_default_tendency=0.38,
            secondary_buyer_default_tendency=0.42,
            financier_capital_multiplier=2.0,
            financier_risk_appetite=0.35,
            legal_max_payment_days=45,
            reward_lambda_shaping=0.35,
        )
    raise KeyError(f"Unknown task_id: {task_id}")


# Aliases for older client / notebook code
_LEGACY_ALIASES: Dict[str, str] = {
    "payment-term-negotiation": "payment-terms-medium",
    "early-payment-discount": "payment-terms-easy",
    "treds-enrollment": "payment-terms-hard",
}

TASK_REGISTRY: Final[Dict[str, TaskConfig]] = {
    "payment-terms-easy": default_task_config("payment-terms-easy"),
    "payment-terms-medium": default_task_config("payment-terms-medium"),
    "payment-terms-hard": default_task_config("payment-terms-hard"),
    "liquidity-stress-medium": default_task_config("liquidity-stress-medium"),
    "liquidity-correlation-hard": default_task_config("liquidity-correlation-hard"),
}


def resolve_task_id(requested: Optional[str], *, difficulty: Optional[str] = None) -> str:
    """Map kwargs / legacy names to a canonical task id."""
    if requested:
        rid = str(requested).strip()
        if rid in _LEGACY_ALIASES:
            return _LEGACY_ALIASES[rid]
        if rid in TASK_REGISTRY:
            return rid
    if difficulty:
        d = str(difficulty).lower()
        if d in ("easy", "e"):
            return "payment-terms-easy"
        if d in ("medium", "m"):
            return "payment-terms-medium"
        if d in ("hard", "h"):
            return "payment-terms-hard"
    return "payment-terms-medium"
