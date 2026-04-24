"""Deterministic macro cashflow simulation helpers for the liquidity environment."""

from __future__ import annotations

import math

from sme_negotiator_env.models import BuyerState, CashflowProjection, DealState, SMEAccountState, WorldSnapshot, WorldState

_PERIOD_DAYS = 30


def _periods_from_days(days: int) -> int:
    return max(1, int(math.ceil(max(int(days), 0) / _PERIOD_DAYS)))


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _lookup_sme(world_state: WorldState, sme_id: str) -> SMEAccountState:
    for sme in world_state.smes:
        if sme.sme_id == sme_id:
            return sme
    raise KeyError(f"Unknown SME id: {sme_id}")


def _lookup_buyer(world_state: WorldState, buyer_id: str) -> BuyerState:
    for buyer in world_state.buyers:
        if buyer.buyer_id == buyer_id:
            return buyer
    raise KeyError(f"Unknown buyer id: {buyer_id}")


def _effective_payment_days(world_state: WorldState, deal: DealState) -> int:
    if deal.agreed_payment_days is not None:
        return int(deal.agreed_payment_days)
    return int(_lookup_buyer(world_state, deal.buyer_id).baseline_payment_days)


def _invoice_amount_from_decision(deal: DealState, decision: dict[str, object]) -> float:
    price = float(decision.get("price", deal.agreed_price or 0.0))
    volume = max(int(deal.volume), 1)
    discount_rate = float(decision.get("dynamic_discount_annual_rate", deal.dynamic_discount_annual_rate))
    if bool(decision.get("propose_dynamic_discounting", False)):
        return round(max(0.0, price * volume * (1.0 - _clip(discount_rate, 0.0, 0.95))), 2)
    return round(max(0.0, price * volume), 2)


def _deal_penalty_exposure(world_state: WorldState, deal: DealState) -> float:
    if deal.status != "agreed" or deal.agreed_payment_days is None:
        return 0.0

    excess_days = max(0, int(deal.agreed_payment_days) - int(world_state.legal_max_payment_days))
    if excess_days <= 0:
        return 0.0

    if excess_days <= 7 and deal.late_payment_penalty_agreed:
        return 0.0

    return round(
        float(deal.invoice_amount)
        * 0.01
        * excess_days
        / max(float(world_state.legal_max_payment_days), 1.0),
        2,
    )


def _apply_initial_financing(world_state: WorldState, deal: DealState) -> None:
    if not deal.financed or deal.initial_funding_applied or world_state.financier is None:
        return

    financier = world_state.financier
    sme = _lookup_sme(world_state, deal.sme_id)
    first_period_fee = round(float(deal.invoice_amount) * float(deal.finance_rate) / 12.0, 2)
    requested_advance = max(0.0, float(deal.invoice_amount) - first_period_fee)
    actual_advance = round(min(float(financier.available_capital), requested_advance), 2)

    if actual_advance <= 0.0:
        deal.financed = False
        deal.finance_rate = 0.0
        return

    financier.available_capital = round(float(financier.available_capital) - actual_advance, 2)
    sme.cash_balance = round(float(sme.cash_balance) + actual_advance, 2)
    deal.financing_principal = actual_advance
    deal.initial_funding_applied = True


def _recompute_world_metrics(world_state: WorldState) -> None:
    for sme in world_state.smes:
        open_gap = 0.0
        buyer_default_pressure = 0.0
        deal_count = 0

        for deal in world_state.deals:
            if deal.sme_id != sme.sme_id or deal.failed or deal.settled:
                continue

            buyer = _lookup_buyer(world_state, deal.buyer_id)
            effective_days = _effective_payment_days(world_state, deal)
            base_amount = float(deal.invoice_amount)
            open_gap += base_amount * max(0, effective_days - int(sme.supplier_payment_days)) / 365.0
            buyer_default_pressure += float(buyer.default_tendency)
            deal_count += 1

        sme.current_utilization = round(open_gap, 2)
        pressure = min(1.0, float(sme.current_utilization) / max(float(sme.credit_limit), 1.0))
        buyer_component = buyer_default_pressure / deal_count if deal_count else float(sme.risk_score)
        sme.risk_score = round(_clip(0.5 * buyer_component + 0.5 * pressure, 0.0, 1.0), 6)
        sme.missed_supplier_payment = bool(float(sme.cash_balance) < float(sme.required_minimum_cash))
        sme.defaulted = bool(
            float(sme.cash_balance) < 0.0
            or float(sme.current_utilization) > float(sme.credit_limit)
            or sme.missed_supplier_payment
        )


def apply_plan_to_world_state(world_state: WorldState, plan: dict[str, object] | None) -> WorldState:
    """Return a copy of ``world_state`` with plan decisions deterministically applied."""

    if not plan:
        return world_state.model_copy(deep=True)

    copied = world_state.model_copy(deep=True)
    deal_decisions = plan.get("deal_decisions", {})
    financing = plan.get("financing", {})

    if not isinstance(deal_decisions, dict):
        deal_decisions = {}
    if not isinstance(financing, dict):
        financing = {}

    for deal in copied.deals:
        raw_decision = deal_decisions.get(deal.deal_id)
        if not isinstance(raw_decision, dict):
            continue

        decision = str(raw_decision.get("decision", "continue")).lower()
        if decision == "reject" and deal.status == "open":
            deal.status = "rejected"
            deal.failed = True
            continue

        if decision != "accept" or deal.status != "open":
            continue

        sme = _lookup_sme(copied, deal.sme_id)
        deal.agreement_period = int(copied.current_period)
        deal.agreed_price = float(raw_decision.get("price", deal.agreed_price or 0.0))
        deal.agreed_payment_days = int(
            raw_decision.get(
                "payment_days",
                deal.agreed_payment_days if deal.agreed_payment_days is not None else _lookup_buyer(copied, deal.buyer_id).baseline_payment_days,
            )
        )
        deal.dynamic_discounting = bool(raw_decision.get("propose_dynamic_discounting", False))
        deal.dynamic_discount_annual_rate = float(
            raw_decision.get("dynamic_discount_annual_rate", deal.dynamic_discount_annual_rate)
        )
        deal.late_payment_penalty_agreed = bool(raw_decision.get("late_payment_penalty_agreed", False))
        deal.invoice_amount = _invoice_amount_from_decision(deal, raw_decision)
        if deal.supplier_payment_amount <= 0.0:
            deal.supplier_payment_amount = round(float(deal.invoice_amount) * 0.8, 2)
        deal.status = "agreed"
        deal.financed = bool(financing.get(deal.deal_id, raw_decision.get("use_treds", False)))
        deal.finance_rate = float(copied.financier.base_interest_rate if copied.financier is not None else deal.finance_rate)
        deal.supplier_due_period = int(copied.current_period) + _periods_from_days(int(sme.supplier_payment_days))
        deal.buyer_due_period = int(copied.current_period) + _periods_from_days(int(deal.agreed_payment_days))
        _apply_initial_financing(copied, deal)

    _recompute_world_metrics(copied)
    return copied


def advance_world_state(world_state: WorldState) -> WorldState:
    """Advance a copied world by one deterministic macro period."""

    copied = world_state.model_copy(deep=True)
    next_period = int(copied.current_period) + 1
    total_penalty = 0.0

    for deal in copied.deals:
        if deal.status == "agreed" and deal.financed and not deal.settled and deal.initial_funding_applied:
            interest = round(float(deal.financing_principal) * float(deal.finance_rate) / 12.0, 2)
            if interest:
                sme = _lookup_sme(copied, deal.sme_id)
                sme.cash_balance = round(float(sme.cash_balance) - interest, 2)
                deal.accrued_interest = round(float(deal.accrued_interest) + interest, 2)

        if deal.status != "agreed":
            continue

        if deal.supplier_due_period == next_period and not deal.supplier_paid:
            sme = _lookup_sme(copied, deal.sme_id)
            sme.cash_balance = round(float(sme.cash_balance) - float(deal.supplier_payment_amount), 2)
            deal.supplier_paid = True

        if deal.buyer_due_period == next_period and not deal.settled:
            if deal.financed and copied.financier is not None:
                copied.financier.available_capital = round(
                    float(copied.financier.available_capital) + float(deal.financing_principal),
                    2,
                )
            else:
                sme = _lookup_sme(copied, deal.sme_id)
                sme.cash_balance = round(float(sme.cash_balance) + float(deal.invoice_amount), 2)
            deal.settled = True
            deal.status = "settled"

        total_penalty += _deal_penalty_exposure(copied, deal)

    _recompute_world_metrics(copied)
    copied.history.append(
        WorldSnapshot(
            period_index=int(copied.current_period),
            total_cash_balance=round(sum(float(sme.cash_balance) for sme in copied.smes), 2),
            defaulted_sme_count=sum(1 for sme in copied.smes if sme.defaulted),
            open_deal_count=sum(1 for deal in copied.deals if deal.status == "open"),
            resolved_deal_count=sum(1 for deal in copied.deals if deal.status != "open"),
            average_payment_days=round(
                sum(_effective_payment_days(copied, deal) for deal in copied.deals if not deal.failed)
                / max(sum(1 for deal in copied.deals if not deal.failed), 1),
                2,
            ),
            total_penalty_exposure=round(total_penalty, 2),
        )
    )
    copied.current_period = next_period
    return copied


def simulate_cashflow(
    world_state: WorldState,
    plan: dict[str, object] | None,
    horizon: int,
) -> CashflowProjection:
    """Pure deterministic cashflow simulation over a macro horizon."""

    projected = apply_plan_to_world_state(world_state, plan)
    requested_horizon = int(horizon)
    advance_periods = requested_horizon
    if isinstance(plan, dict) and plan.get("advance_periods") is not None:
        advance_periods = min(requested_horizon, max(int(plan["advance_periods"]), 0))

    balances: list[float] = []
    defaults: list[bool] = []
    penalties: list[float] = []

    for _ in range(max(advance_periods, 0)):
        projected = advance_world_state(projected)
        balances.append(round(sum(float(sme.cash_balance) for sme in projected.smes), 2))
        defaults.append(any(bool(sme.defaulted) for sme in projected.smes))
        penalties.append(
            round(
                sum(_deal_penalty_exposure(projected, deal) for deal in projected.deals if deal.status == "agreed"),
                2,
            )
        )

    return CashflowProjection(
        period_balances=balances,
        period_defaults=defaults,
        period_penalties=penalties,
    )
