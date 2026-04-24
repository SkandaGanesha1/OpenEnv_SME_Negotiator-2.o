"""Deterministic reward helpers and compatibility graders for the SME environment.

Round 1 compatibility:
- ``grade_task_payment_terms_easy``
- ``grade_task_payment_terms_medium``
- ``grade_task_dynamic_discounting_hard``

These legacy terminal graders remain numerically unchanged for the existing
``payment-terms-*`` tasks. Their legacy dense shaping counterpart still lives in
``server.environment.SMENegotiatorEnvironment._compute_reward``.

Stage 2 additions in this module are deterministic RLVR-style helpers:
- ``compute_verifiable_reward``
- ``compute_shaping_rewards``
- ``compute_total_sme_reward``
- ``compute_npv_delta_vs_baseline``
- ``compute_rubric_score`` (external-only stub)

Stage 4 adds a liquidity-only helper:
- ``compute_tool_use_bonus``

This helper stays separate from ``compute_shaping_rewards`` so the existing
Stage 2 reward semantics remain unchanged.
"""

from __future__ import annotations

import math
from typing import Callable

from sme_negotiator_env.models import (
    BuyerState,
    NegotiationState,
    SMEAccountState,
    ToolCallRecord,
    WorldState,
)

_STRICT_EPS = 1e-3


def _strict_unit_interval(score: float) -> float:
    """Map any numeric legacy terminal score into the strict open interval ``(0, 1)``."""
    value = float(score)
    if not math.isfinite(value):
        return _STRICT_EPS
    return float(min(1.0 - _STRICT_EPS, max(_STRICT_EPS, value)))


def _clip(value: float, lower: float, upper: float) -> float:
    """Clip a numeric value into a closed interval."""
    return float(min(upper, max(lower, value)))


def _effective_receivable_days(state: NegotiationState) -> int:
    """Return the effective receivable horizon used by Stage 2 reward logic."""
    if state.agreed_terms is not None:
        return int(state.agreed_terms)
    return int(state.buyer_days)


def _effective_working_capital_gap(state: NegotiationState) -> float:
    """Estimate the working-capital gap using the effective receivable horizon."""
    return float(state.sme_monthly_revenue) * max(
        0,
        _effective_receivable_days(state) - int(state.sme_supplier_payment_days),
    ) / 365.0


def _receivable_payable_mismatch(state: NegotiationState) -> float:
    """Return the absolute mismatch between receivable and payable cycles."""
    return float(abs(_effective_receivable_days(state) - int(state.sme_supplier_payment_days)))


def _compliance_distance(state: NegotiationState, legal_max_payment_days: int) -> float:
    """Return a non-negative distance from legal/payment-term compliance."""
    effective_days = _effective_receivable_days(state)
    if effective_days <= int(legal_max_payment_days):
        return 0.0

    excess_days = max(0, effective_days - int(legal_max_payment_days))
    if excess_days <= 7 and state.late_payment_penalty_agreed:
        return 0.0
    return float(excess_days)


def _lookup_sme(world_state: WorldState, sme_id: str) -> SMEAccountState:
    """Find the active SME state for reward computation."""
    for sme in world_state.smes:
        if sme.sme_id == sme_id:
            return sme
    raise KeyError(f"Unknown SME id in world_state: {sme_id}")


def _lookup_buyer(world_state: WorldState, buyer_id: str) -> BuyerState:
    """Find the active buyer state for reward computation."""
    for buyer in world_state.buyers:
        if buyer.buyer_id == buyer_id:
            return buyer
    raise KeyError(f"Unknown buyer id in world_state: {buyer_id}")


def _contract_price(state: NegotiationState, *, baseline: bool) -> float:
    """Return the price used for baseline or actual contract valuation."""
    if baseline:
        return float(state.buyer_price)
    if state.final_price is not None:
        return float(state.final_price)
    return float(state.buyer_price)


def _contract_days(state: NegotiationState, *, baseline: bool) -> int:
    """Return the receivable days used for baseline or actual contract valuation."""
    if baseline:
        return int(state.initial_buyer_days)
    return _effective_receivable_days(state)


def _npv_from_state(state: NegotiationState, *, baseline: bool, discount_rate: float) -> float:
    """Compute a simplified NPV for the SME contract under baseline or actual terms."""
    price = _contract_price(state, baseline=baseline)
    days = _contract_days(state, baseline=baseline)
    gross_inflow = price * float(state.volume)

    if baseline:
        discount_fraction = 0.0
    elif state.dynamic_discounting_agreed:
        discount_fraction = float(state.agreed_dynamic_discount_annual)
    else:
        discount_fraction = 0.0

    net_inflow = gross_inflow * (1.0 - _clip(discount_fraction, 0.0, 0.95))
    pv = net_inflow / math.pow(1.0 + float(discount_rate), max(days, 0) / 365.0)

    financing_cost = 0.0
    if (not baseline) and state.treds_used:
        financing_cost = (
            _effective_working_capital_gap(state)
            * float(discount_rate)
            * max(0, days - int(state.sme_supplier_payment_days))
            / 365.0
        )

    return float(pv - financing_cost)


def compute_financing_npv_vs_status_quo(state: NegotiationState) -> float:
    """Pure helper for the legacy hard-task compatibility grader.

    Returns the NPV improvement (INR) from a dynamic discounting arrangement
    versus the status quo.

    Status quo:
        One month of revenue is received after ``initial_buyer_days``.

    Arrangement:
        The same nominal revenue is received earlier and/or with an explicit
        annualized discount for faster payment.

    Uses ``interest_rate_annual`` as the annual discount rate. The function is a
    deterministic transformation of the provided ``NegotiationState``.
    """
    r = float(state.interest_rate_annual)
    F = float(state.sme_monthly_revenue)
    d0 = int(state.initial_buyer_days)
    d1 = int(state.agreed_terms) if state.agreed_terms is not None else d0
    disc = (
        float(state.agreed_dynamic_discount_annual)
        if state.dynamic_discounting_agreed
        else 0.0
    )

    def pv(days: int, face: float) -> float:
        return face / math.pow(1.0 + r, max(days, 0) / 365.0)

    pv_old = pv(d0, F)
    pv_new = pv(d1, F * (1.0 - min(max(disc, 0.0), 0.95)))
    return float(pv_new - pv_old)


def grade_task_payment_terms_easy(state: NegotiationState) -> float:
    """Legacy Round 1 terminal score for ``payment-terms-easy``.

    This compatibility scalar rewards a final deal at or below the liquidity
    threshold and returns strict-open-interval scores for existing evaluation
    pipelines. The logic is fully verifiable and deterministic, but it is still
    a task-specific compatibility grader rather than the Stage 2 RL reward.
    """
    if not state.deal_reached or state.agreed_terms is None:
        return _strict_unit_interval(0.0)

    d = int(state.agreed_terms)
    cap = int(state.liquidity_threshold)

    if d <= cap:
        return _strict_unit_interval(1.0)

    if d <= cap + 15:
        return _strict_unit_interval(0.5)

    return _strict_unit_interval(0.0)


def grade_task_payment_terms_medium(state: NegotiationState) -> float:
    """Legacy Round 1 terminal score for ``payment-terms-medium``.

    This compatibility grader remains deterministic and verifiable: it checks
    the final payment days against the liquidity threshold and grants partial
    credit when a slightly slower deal includes a late-payment penalty clause.
    """
    if not state.deal_reached or state.agreed_terms is None:
        return _strict_unit_interval(0.0)

    d = int(state.agreed_terms)
    cap = int(state.liquidity_threshold)

    if d <= cap:
        return _strict_unit_interval(1.0)

    if d <= cap + 7 and state.late_payment_penalty_agreed:
        return _strict_unit_interval(0.5)

    return _strict_unit_interval(0.0)


def grade_task_dynamic_discounting_hard(state: NegotiationState) -> float:
    """Legacy Round 1 terminal score for ``payment-terms-hard``.

    This compatibility grader is deterministic and grounded in NPV improvement
    under dynamic discounting. It is still separate from the Stage 2 dense
    shaping and RL reward decomposition.
    """
    if not state.deal_reached:
        return _strict_unit_interval(0.0)

    if not state.dynamic_discounting_agreed:
        return _strict_unit_interval(0.0)

    delta = compute_financing_npv_vs_status_quo(state)
    scale = max(abs(float(state.working_capital_gap)) * 0.05, 1.0)

    raw_score = delta / scale
    raw_score = max(0.0, min(1.0, raw_score))

    return _strict_unit_interval(raw_score)


def compute_verifiable_reward(world_state: WorldState, trajectory: list[NegotiationState]) -> float:
    """Compute the deterministic terminal RL reward for a liquidity episode.

    The reward is a pure function of ``world_state`` and ``trajectory`` and
    combines four verifiable criteria:
    - SME solvency / absence of default
    - liquidity buffer adequacy
    - NPV improvement versus the baseline contract
    - legal/payment-term compliance
    """
    if not trajectory:
        return 0.0

    final_state = trajectory[-1]
    sme = _lookup_sme(world_state, final_state.sme_id)
    _lookup_buyer(world_state, final_state.buyer_id)

    default_flag = bool(
        sme.defaulted
        or float(sme.cash_balance) < 0.0
        or float(sme.current_utilization) > float(sme.credit_limit)
        or sme.missed_supplier_payment
    )
    if default_flag:
        return 0.0

    solvency_score = 1.0
    liquidity_score = _clip(
        float(sme.cash_balance) / max(float(sme.required_minimum_cash), 1.0),
        0.0,
        1.0,
    )

    discount_rate = (
        float(world_state.baseline_discount_rate)
        if float(world_state.baseline_discount_rate) > 0.0
        else float(final_state.interest_rate_annual)
    )
    baseline_state = trajectory[0]
    npv_baseline = _npv_from_state(baseline_state, baseline=True, discount_rate=discount_rate)
    npv_actual = _npv_from_state(final_state, baseline=False, discount_rate=discount_rate)
    npv_score = _clip(
        0.5
        + 0.5
        * (npv_actual - npv_baseline)
        / max(abs(npv_baseline), float(sme.required_minimum_cash), 1.0),
        0.0,
        1.0,
    )

    effective_days = _effective_receivable_days(final_state)
    if effective_days <= int(world_state.legal_max_payment_days):
        compliance_score = 1.0
    elif effective_days <= int(world_state.legal_max_payment_days) + 7 and final_state.late_payment_penalty_agreed:
        compliance_score = 0.5
    else:
        compliance_score = 0.0

    reward = (
        0.35 * solvency_score
        + 0.20 * liquidity_score
        + 0.35 * npv_score
        + 0.10 * compliance_score
    )
    return round(_clip(reward, 0.0, 1.0), 6)


def compute_shaping_rewards(trajectory: list[NegotiationState]) -> list[float]:
    """Compute dense deterministic shaping rewards over a trajectory.

    Each per-step value depends only on consecutive state pairs and reflects
    progress in:
    - reducing the effective working-capital gap
    - moving payment days toward a fair/legal target range
    - aligning receivable and payable cycles
    """
    if len(trajectory) < 2:
        return []

    shaping: list[float] = []
    for prev_state, next_state in zip(trajectory, trajectory[1:]):
        prev_gap = _effective_working_capital_gap(prev_state)
        next_gap = _effective_working_capital_gap(next_state)
        gap_term = _clip(
            (prev_gap - next_gap)
            / max(prev_gap, float(prev_state.sme_monthly_revenue) * 0.25, 1.0),
            -1.0,
            1.0,
        )

        target_days = min(int(prev_state.liquidity_threshold), 45)
        prev_days_delta = abs(_effective_receivable_days(prev_state) - target_days)
        next_days_delta = abs(_effective_receivable_days(next_state) - target_days)
        days_term = _clip(
            (prev_days_delta - next_days_delta) / max(target_days, 1),
            -1.0,
            1.0,
        )

        prev_mismatch = _receivable_payable_mismatch(prev_state)
        next_mismatch = _receivable_payable_mismatch(next_state)
        alignment_term = _clip(
            (prev_mismatch - next_mismatch) / max(int(prev_state.sme_supplier_payment_days), 1),
            -1.0,
            1.0,
        )

        reward_t = round(0.5 * gap_term + 0.3 * days_term + 0.2 * alignment_term, 6)
        shaping.append(reward_t)

    return shaping


def compute_total_sme_reward(
    world_state: WorldState,
    trajectory: list[NegotiationState],
    lambda_shaping: float = 0.1,
) -> float:
    """Compute the total Stage 2 RL reward for the SME agent."""
    terminal = compute_verifiable_reward(world_state, trajectory)
    shaping = compute_shaping_rewards(trajectory)
    return float(terminal + float(lambda_shaping) * sum(shaping))


def compute_npv_delta_vs_baseline(world_state: WorldState, trajectory: list[NegotiationState]) -> float:
    """Return deterministic NPV uplift versus the baseline trajectory state."""
    if not trajectory:
        return 0.0

    final_state = trajectory[-1]
    discount_rate = (
        float(world_state.baseline_discount_rate)
        if float(world_state.baseline_discount_rate) > 0.0
        else float(final_state.interest_rate_annual)
    )
    baseline_state = trajectory[0]
    npv_baseline = _npv_from_state(baseline_state, baseline=True, discount_rate=discount_rate)
    npv_actual = _npv_from_state(final_state, baseline=False, discount_rate=discount_rate)
    return round(float(npv_actual - npv_baseline), 6)


def compute_tool_use_bonus(
    *,
    latest_tool_call: ToolCallRecord | None,
    current_deal_id: str | None,
    current_step_index: int,
    base_shaping_reward: float,
    previous_state: NegotiationState | None,
    next_state: NegotiationState | None,
    legal_max_payment_days: int,
    pending_tool_bonus: float = 0.0,
) -> float:
    """Compute the tiny deterministic Stage 4 tool bonus for liquidity episodes.

    The function is bounded by design and never replaces the underlying reward
    objective:
    - ``QUERY_TREDS`` / ``RUN_CASHFLOW_SIM`` can contribute ``+0.01`` if the
      next negotiation transition on the same deal has positive base shaping.
    - ``CHECK_COMPLIANCE`` can contribute ``+0.005`` if the next negotiation
      transition reduces compliance distance.
    - repeated identical tool calls can contribute ``-0.005`` via
      ``pending_tool_bonus`` from the environment.
    """
    bonus = float(pending_tool_bonus)
    if latest_tool_call is None or current_deal_id is None:
        return round(_clip(bonus, -0.005, 0.01), 6)
    if latest_tool_call.deal_id != current_deal_id:
        return round(_clip(bonus, -0.005, 0.01), 6)
    if int(current_step_index) - int(latest_tool_call.step_index) > 2:
        return round(_clip(bonus, -0.005, 0.01), 6)

    tool_name = str(latest_tool_call.tool_name)
    if tool_name in {"QUERY_TREDS", "RUN_CASHFLOW_SIM"}:
        if float(base_shaping_reward) > 0.0:
            bonus += 0.01
    elif tool_name == "CHECK_COMPLIANCE" and previous_state is not None and next_state is not None:
        previous_distance = _compliance_distance(previous_state, legal_max_payment_days)
        next_distance = _compliance_distance(next_state, legal_max_payment_days)
        if next_distance < previous_distance or (previous_distance > 0.0 and next_distance == 0.0):
            bonus += 0.005

    return round(_clip(bonus, -0.005, 0.01), 6)


def compute_rubric_score(episode_log: str) -> dict[str, float]:
    """External-only Rubrics-as-Rewards hook for RL training scripts.

    The environment core never calls this function from ``step()`` or the
    deterministic graders. Training-side orchestration can override or wrap this
    stub with LLM-judge logic later.
    """
    raise NotImplementedError("Rubric-based scoring is implemented in RL training scripts.")


# Canonical mapping from task ids to deterministic legacy terminal graders.
TASK_GRADERS: dict[str, Callable[[NegotiationState], float]] = {
    "payment-terms-easy": grade_task_payment_terms_easy,
    "payment-terms-medium": grade_task_payment_terms_medium,
    "payment-terms-hard": grade_task_dynamic_discounting_hard,
}
