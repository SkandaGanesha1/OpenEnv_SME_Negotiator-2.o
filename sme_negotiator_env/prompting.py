"""Pure prompt and action helpers shared by inference and RL tooling."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from sme_negotiator_env.models import LiquidityObservation, NegotiationAction

MAX_REASON_CHARS = 160


def clip_ascii_text(value: Any, max_len: int) -> str:
    """Return a bounded ASCII-safe string representation."""
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "~"


def observation_to_dict(observation: Any) -> Dict[str, Any]:
    """Convert an observation-like object into a plain dictionary."""
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return dict(observation)
    return dict(observation)


def format_observation_text(
    observation: Any,
    *,
    role: str = "sme",
    persona: Optional[str] = None,
    opponent_context: Optional[dict[str, Any]] = None,
) -> str:
    """Format legacy or liquidity observations into a prompt-friendly string."""
    obs = observation_to_dict(observation)
    normalized_role = str(role or "sme").strip().lower()
    msg = str(obs.get("message") or "").strip()
    lines: list[str] = []
    lines.append(f"Role={normalized_role.upper()}")
    if persona:
        lines.append(f"Persona={persona}")
    if msg:
        clipped = f"{msg[:900]}..." if len(msg) > 900 else msg
        lines.append(f"EnvMessage={clipped}")

    if normalized_role == "buyer":
        base = (
            f"Round={obs.get('round_number')} | Task={obs.get('task_name')} | "
            f"CurrentOfferPrice={obs.get('buyer_price')} | CurrentOfferDays={obs.get('buyer_days')} | "
            f"SupplierCostFloor={obs.get('cost_threshold')} | SupplierLiquidityTarget={obs.get('liquidity_threshold')} | "
            f"SupplierWCGap={obs.get('working_capital_gap')} | SupplierPayDays={obs.get('sme_supplier_payment_days')} | "
            f"BuyerPower={obs.get('buyer_power_score')}"
        )
    elif normalized_role == "financier":
        base = (
            f"Round={obs.get('round_number')} | Task={obs.get('task_name')} | "
            f"InvoiceBuyerPrice={obs.get('buyer_price')} | InvoiceBuyerDays={obs.get('buyer_days')} | "
            f"SMERevenue={obs.get('sme_monthly_revenue')} | SMEWCGap={obs.get('working_capital_gap')} | "
            f"InterestAnnual={obs.get('interest_rate_annual')} | BuyerPower={obs.get('buyer_power_score')}"
        )
    else:
        base = (
            f"Round={obs.get('round_number')} | Task={obs.get('task_name')} | "
            f"BuyerPrice={obs.get('buyer_price')} | BuyerDays={obs.get('buyer_days')} | "
            f"LiquidityThreshold={obs.get('liquidity_threshold')} | CostThreshold={obs.get('cost_threshold')} | "
            f"MonthlyRevenueINR={obs.get('sme_monthly_revenue')} | WCGap={obs.get('working_capital_gap')} | "
            f"SupplierPayDays={obs.get('sme_supplier_payment_days')} | InterestAnnual={obs.get('interest_rate_annual')} | "
            f"BuyerPower={obs.get('buyer_power_score')}"
        )
    lines.append(base)

    if "current_period" in obs or "active_deal_id" in obs:
        lines.append(
            f"Macro: active_deal={obs.get('active_deal_id')} | open_deals={obs.get('open_deal_ids')} | "
            f"resolved_deals={obs.get('resolved_deal_ids')} | current_period={obs.get('current_period')} / "
            f"{obs.get('total_periods')} | episode_step={obs.get('episode_step')}"
        )

    last_tool_name = obs.get("last_tool_name")
    if last_tool_name:
        lines.append(
            "LastTool: "
            f"name={last_tool_name} | args={json.dumps(obs.get('last_tool_args'), ensure_ascii=True, sort_keys=True)} | "
            f"result={json.dumps(obs.get('last_tool_result'), ensure_ascii=True, sort_keys=True)}"
        )

    history = obs.get("history") or []
    if history:
        compact_history: list[str] = []
        for event in history[-3:]:
            if hasattr(event, "model_dump"):
                item = event.model_dump()
            else:
                item = dict(event)
            compact_history.append(
                f"{item.get('period_index')}:{item.get('actor')}:{item.get('event_type')}:{item.get('summary')}"
            )
        lines.append("HistoryTail=" + " || ".join(compact_history))

    if obs.get("done") is True:
        lines.append(
            f"Terminal: reward={obs.get('reward')} | negotiation_done={obs.get('negotiation_done')} | "
            f"buyer_accepted={obs.get('buyer_accepted')}"
        )

    if opponent_context:
        lines.append(
            "OpponentContext="
            + json.dumps(opponent_context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )

    return "\n".join(lines)


def normalize_action_type(action_type: Any) -> str:
    """Normalize action types from legacy or prompt variants to repo schema."""
    value = str(action_type or "propose").strip().lower()
    if value == "negotiate":
        return "propose"
    if value == "simulate_plan":
        return "simulate_plan"
    if value == "advance_period":
        return "advance_period"
    if value == "tool":
        return "tool"
    if value in {"propose", "accept", "reject"}:
        return value
    return "propose"


def action_payload_to_model_action(
    action_payload: Dict[str, Any],
    observation: Optional[Any] = None,
    *,
    default_reason: str = "Model-selected action",
) -> NegotiationAction:
    """Convert a normalized payload dict into the canonical action model."""
    action_type = normalize_action_type(action_payload.get("action_type", "propose"))
    obs = observation_to_dict(observation) if observation is not None else {}

    price = float(action_payload.get("price", obs.get("buyer_price", 0.0)))
    payment_days = int(action_payload.get("payment_days", obs.get("buyer_days", 0)))
    use_treds = bool(action_payload.get("use_treds", False))
    reason = clip_ascii_text(action_payload.get("reason", default_reason), MAX_REASON_CHARS)

    return NegotiationAction(
        action_type=action_type,
        price=round(price, 2),
        payment_days=payment_days,
        use_treds=use_treds,
        reason=reason,
        deal_id=action_payload.get("deal_id"),
        simulation_plan=action_payload.get("simulation_plan"),
        simulation_horizon=action_payload.get("simulation_horizon"),
        tool_name=action_payload.get("tool_name"),
        tool_args=action_payload.get("tool_args"),
        propose_late_payment_penalty_clause=bool(action_payload.get("propose_late_payment_penalty_clause", False)),
        propose_dynamic_discounting=bool(action_payload.get("propose_dynamic_discounting", False)),
        dynamic_discount_annual_rate=float(action_payload.get("dynamic_discount_annual_rate", 0.0)),
    )


def conservative_default_action(observation: Optional[Any] = None) -> NegotiationAction:
    """Return a conservative fallback action for offline parsing failures."""
    obs = observation_to_dict(observation) if observation is not None else {}
    buyer_price = float(obs.get("buyer_price", 0.0))
    buyer_days = int(obs.get("buyer_days", 30))
    cost = float(obs.get("cost_threshold", 0.0))
    liquidity = int(obs.get("liquidity_threshold", max(0, buyer_days - 5)))
    return NegotiationAction(
        action_type="propose",
        price=round(max(cost, buyer_price * 0.99), 2),
        payment_days=max(0, min(buyer_days, liquidity)),
        use_treds=bool(buyer_days > liquidity + 20),
        reason="Default action after failed parse",
    )
