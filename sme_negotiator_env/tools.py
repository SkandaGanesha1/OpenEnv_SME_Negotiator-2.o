"""Deterministic Stage 4 enterprise tools for the liquidity environment."""

from __future__ import annotations

from typing import Any, Optional

from sme_negotiator_env.models import BuyerState, DealState, NegotiationState, SMEAccountState, WorldState
from sme_negotiator_env.simulation import simulate_cashflow


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _lookup_deal(world_state: WorldState, deal_id: str) -> DealState:
    for deal in world_state.deals:
        if deal.deal_id == deal_id:
            return deal
    raise KeyError(f"Unknown deal_id: {deal_id}")


def _lookup_sme(world_state: WorldState, sme_id: str) -> SMEAccountState:
    for sme in world_state.smes:
        if sme.sme_id == sme_id:
            return sme
    raise KeyError(f"Unknown sme_id: {sme_id}")


def _lookup_buyer(world_state: WorldState, buyer_id: str) -> BuyerState:
    for buyer in world_state.buyers:
        if buyer.buyer_id == buyer_id:
            return buyer
    raise KeyError(f"Unknown buyer_id: {buyer_id}")


def _resolve_identifier(identifier: str | None, *, alias: str) -> str:
    if not identifier:
        raise ValueError(f"{alias} is required")
    return str(identifier)


def query_treds(world_state: WorldState, invoice_id: str) -> dict[str, Any]:
    """Return deterministic TReDS-like quotes for a deal-backed invoice."""
    deal_id = _resolve_identifier(invoice_id, alias="invoice_id")
    deal = _lookup_deal(world_state, deal_id)
    sme = _lookup_sme(world_state, deal.sme_id)
    buyer = _lookup_buyer(world_state, deal.buyer_id)
    financier = world_state.financier

    invoice_amount = float(
        deal.invoice_amount
        or ((deal.agreed_price or 0.0) * max(int(deal.volume), 1))
    )
    base = (
        float(financier.base_interest_rate)
        if financier is not None
        else max(float(world_state.baseline_discount_rate), 0.12)
    )
    risk_markup = 0.04 * float(sme.risk_score) + 0.03 * float(buyer.default_tendency)
    util_markup = 0.02 * min(1.0, float(sme.current_utilization) / max(float(sme.credit_limit), 1.0))
    appetite_discount = 0.01 * (float(financier.risk_appetite) if financier is not None else 0.5)
    base_quote = _clip(base + risk_markup + util_markup - appetite_discount, 0.05, 0.60)
    available_capital = float(financier.available_capital) if financier is not None else 0.0

    quote_options: list[dict[str, Any]] = []
    for index, tenor_days in enumerate((15, 30, 45)):
        annual_discount_rate = round(_clip(base_quote + 0.005 * index, 0.05, 0.60), 6)
        discount_fee = round(invoice_amount * annual_discount_rate * tenor_days / 365.0, 2)
        advance_amount = round(max(0.0, invoice_amount - discount_fee), 2)
        available = bool(available_capital >= advance_amount and advance_amount > 0.0)
        quote_options.append(
            {
                "tenor_days": tenor_days,
                "annual_discount_rate": annual_discount_rate,
                "discount_fee": discount_fee,
                "advance_amount": advance_amount,
                "available": available,
            }
        )

    available_options = [quote for quote in quote_options if quote["available"]]
    recommended_tenor = (
        min(available_options, key=lambda option: (option["annual_discount_rate"], option["tenor_days"]))["tenor_days"]
        if available_options
        else quote_options[0]["tenor_days"]
    )
    return {
        "invoice_id": deal_id,
        "quote_options": quote_options,
        "recommended_tenor_days": int(recommended_tenor),
        "available_capital": round(available_capital, 2),
    }


def check_compliance(
    world_state: WorldState,
    contract_id: str,
    negotiation_state: Optional[NegotiationState] = None,
) -> dict[str, Any]:
    """Return a deterministic compliance verdict for a deal or in-flight contract."""
    deal_id = _resolve_identifier(contract_id, alias="contract_id")
    deal = _lookup_deal(world_state, deal_id)
    financier = world_state.financier

    effective_days = (
        int(deal.agreed_payment_days)
        if deal.agreed_payment_days is not None
        else int(
            negotiation_state.agreed_terms
            if negotiation_state is not None and negotiation_state.agreed_terms is not None
            else (
                negotiation_state.buyer_days
                if negotiation_state is not None
                else _lookup_buyer(world_state, deal.buyer_id).baseline_payment_days
            )
        )
    )
    late_payment_penalty = bool(
        deal.late_payment_penalty_agreed
        or (negotiation_state.late_payment_penalty_agreed if negotiation_state is not None else False)
    )
    dynamic_discounting = bool(
        deal.dynamic_discounting
        or (negotiation_state.dynamic_discounting_agreed if negotiation_state is not None else False)
    )
    dynamic_discount_rate = float(
        deal.dynamic_discount_annual_rate
        or (
            negotiation_state.agreed_dynamic_discount_annual
            if negotiation_state is not None
            else 0.0
        )
    )

    legal_max = int(world_state.legal_max_payment_days)
    policy_cap = max(
        0.30,
        (float(financier.base_interest_rate) + 0.10) if financier is not None else 0.30,
    )
    violated_clauses: list[str] = []

    if effective_days > legal_max:
        if effective_days <= legal_max + 7 and late_payment_penalty:
            pass
        else:
            violated_clauses.append("payment_days_exceed_legal_max")

    if legal_max < effective_days <= legal_max + 7 and not late_payment_penalty:
        violated_clauses.append("grace_extension_requires_penalty_clause")

    if dynamic_discounting and dynamic_discount_rate > policy_cap:
        violated_clauses.append("dynamic_discount_rate_exceeds_policy_cap")

    is_compliant = not violated_clauses
    if is_compliant:
        explanation = "Contract terms are compliant with the deterministic Stage 4 policy checks."
    else:
        explanation = "Contract terms violate deterministic Stage 4 policy checks."

    return {
        "contract_id": deal_id,
        "is_compliant": is_compliant,
        "violated_clauses": violated_clauses,
        "checked_terms": {
            "payment_days": int(effective_days),
            "late_payment_penalty_agreed": late_payment_penalty,
            "dynamic_discounting_agreed": dynamic_discounting,
            "dynamic_discount_annual_rate": round(dynamic_discount_rate, 6),
            "policy_cap": round(policy_cap, 6),
        },
        "explanation": explanation,
    }


def run_cashflow_sim(
    world_state: WorldState,
    plan: dict[str, Any] | None,
    horizon: int,
) -> dict[str, Any]:
    """Run the deterministic Stage 3 cashflow simulator and summarize results."""
    projection = simulate_cashflow(world_state, plan or {}, horizon=horizon)
    ending_balance = projection.period_balances[-1] if projection.period_balances else round(
        sum(float(sme.cash_balance) for sme in world_state.smes),
        2,
    )
    return {
        "period_balances": list(projection.period_balances),
        "period_defaults": list(projection.period_defaults),
        "period_penalties": list(projection.period_penalties),
        "ending_balance": round(float(ending_balance), 2),
        "any_default": any(bool(flag) for flag in projection.period_defaults),
        "total_penalty_exposure": round(sum(float(value) for value in projection.period_penalties), 2),
    }
