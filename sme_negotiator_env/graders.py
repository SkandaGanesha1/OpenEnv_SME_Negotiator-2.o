"""Task-specific reward functions over :class:`NegotiationState`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sme_negotiator_env.models import NegotiationState


_STRICT_EPS = 1e-6


def _strict_unit_interval(score: float) -> float:
    """Map any score into the strict open interval (0, 1)."""
    return float(min(1.0 - _STRICT_EPS, max(_STRICT_EPS, float(score))))


def compute_financing_npv_vs_status_quo(state: "NegotiationState") -> float:
    """NPV improvement (INR) from a dynamic discounting arrangement vs status quo.

    Status quo: one month of revenue is received after ``initial_buyer_days`` (buyer offer at reset).
    Arrangement: same nominal revenue is received earlier and/or with an explicit annualized
    discount for faster payment. Uses ``interest_rate_annual`` as the daily discount rate.
    """
    r = float(state.interest_rate_annual)
    F = float(state.sme_monthly_revenue)
    d0 = int(state.initial_buyer_days)
    d1 = int(state.agreed_terms) if state.agreed_terms is not None else d0
    disc = float(state.agreed_dynamic_discount_annual) if state.dynamic_discounting_agreed else 0.0

    def pv(days: int, face: float) -> float:
        return face / math.pow(1.0 + r, max(days, 0) / 365.0)

    pv_old = pv(d0, F)
    pv_new = pv(d1, F * (1.0 - min(max(disc, 0.0), 0.95)))
    return float(pv_new - pv_old)


def grade_task_payment_terms_easy(state: "NegotiationState") -> float:
    """Easy: cooperative buyer; success = agreed payment days at or below task liquidity threshold."""
    if not state.deal_reached or state.agreed_terms is None:
        return _strict_unit_interval(0.0)
    d = int(state.agreed_terms)
    cap = int(state.liquidity_threshold)
    if d <= cap:
        return _strict_unit_interval(1.0)
    if d <= cap + 15:
        return _strict_unit_interval(0.5)
    return _strict_unit_interval(0.0)


def grade_task_payment_terms_medium(state: "NegotiationState") -> float:
    """Medium: primary success = agreed days at or below liquidity threshold (45d); bonus tier with penalty clause."""
    if not state.deal_reached or state.agreed_terms is None:
        return _strict_unit_interval(0.0)
    d = int(state.agreed_terms)
    cap = int(state.liquidity_threshold)
    if d <= cap:
        return _strict_unit_interval(1.0)
    if d <= cap + 7 and state.late_payment_penalty_agreed:
        return _strict_unit_interval(0.5)
    return _strict_unit_interval(0.0)


def grade_task_dynamic_discounting_hard(state: "NegotiationState") -> float:
    """Hard: two-buyer pressure; reward from NPV of financing vs status quo."""
    if not state.deal_reached:
        return _strict_unit_interval(0.0)
    if not state.dynamic_discounting_agreed:
        return _strict_unit_interval(0.0)
    delta = compute_financing_npv_vs_status_quo(state)
    scale = max(abs(state.working_capital_gap) * 0.05, 1.0)
    return _strict_unit_interval(float(max(0.0, min(1.0, delta / scale))))


TASK_GRADERS = {
    "payment-terms-easy": grade_task_payment_terms_easy,
    "payment-terms-medium": grade_task_payment_terms_medium,
    "payment-terms-hard": grade_task_dynamic_discounting_hard,
}
