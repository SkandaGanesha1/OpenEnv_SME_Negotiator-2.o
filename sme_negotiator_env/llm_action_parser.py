"""Parse free-form LLM text into a typed NegotiationAction (fallback when JSON fails)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Union

from sme_negotiator_env.models import NegotiationAction

logger = logging.getLogger(__name__)

_DAYS_PATTERNS = [
    re.compile(r"(?i)(?:payment\s*)?(?:terms?|days?)\s*(?:of|:|=)?\s*(\d{1,4})\s*(?:days?|d\b)"),
    re.compile(r"(?i)(?:propose|proposing|offer|accept)\s+(?:up\s*to\s+)?(\d{1,4})\s*(?:days?|day\b)"),
    re.compile(r"(?i)(\d{1,4})\s*(?:days?|day)\s+(?:payment|terms?)"),
    re.compile(r"(?i)(?:in|within|at)\s+(\d{1,4})\s*(?:days?|calendar\s*days?)"),
    re.compile(r"(?i)\b(\d{1,4})\s*DDS\b"),
]

_PRICE_PATTERNS = [
    re.compile(r"(?i)(?:price|₹|rs\.?|inr)\s*[:\s]*(\d+(?:\.\d{1,2})?)"),
    re.compile(r"(?i)(\d+(?:\.\d{1,2})?)\s*(?:per\s*unit|/unit|₹)"),
]


def _default_action_from_observation(obs: Dict[str, Any]) -> NegotiationAction:
    """Conservative propose when parsing fails completely."""
    buyer_price = float(obs.get("buyer_price", 100.0))
    buyer_days = int(obs.get("buyer_days", 45))
    cost = float(obs.get("cost_threshold", 80.0))
    liq = int(obs.get("liquidity_threshold", 30))
    return NegotiationAction(
        action_type="propose",
        price=round(max(cost + 1.0, buyer_price * 0.99), 2),
        payment_days=int(max(liq, buyer_days - 5)),
        use_treds=bool(buyer_days > liq + 20),
        reason="Default action after failed LLM parse",
    )


def _infer_action_type(text: str) -> str:
    t = text.lower().strip()
    if re.search(r"\breject\b", t) and "reject" in t[:120]:
        return "reject"
    if re.search(r"\baccept\b", t) and "accept" in t[:120]:
        return "accept"
    return "propose"


def _extract_days(text: str) -> Optional[int]:
    for pat in _DAYS_PATTERNS:
        m = pat.search(text)
        if m:
            d = int(m.group(1))
            if 0 <= d <= 3650:
                return d
    # Last resort: first plausible standalone integer 7–999 in negotiation context
    for m in re.finditer(r"\b(\d{2,3})\b", text):
        v = int(m.group(1))
        if 7 <= v <= 400:
            return v
    return None


def _extract_price(text: str, obs: Dict[str, Any]) -> float:
    for pat in _PRICE_PATTERNS:
        m = pat.search(text)
        if m:
            return round(float(m.group(1)), 2)
    return float(obs.get("buyer_price", 100.0))


def parse_llm_text_to_negotiation_action(
    raw_text: str,
    observation: Union[Dict[str, Any], Any],
    *,
    allow_json: bool = True,
) -> NegotiationAction:
    """Turn LLM output (JSON or prose) into NegotiationAction.

    1. If ``allow_json``, try strict JSON / fenced JSON first.
    2. Else extract payment days (regex), price, and action keywords.
    3. On failure, log a warning and return a safe default proposal.
    """
    obs: Dict[str, Any]
    if hasattr(observation, "model_dump"):
        obs = observation.model_dump()
    elif isinstance(observation, dict):
        obs = dict(observation)
    else:
        obs = dict(observation)

    text = (raw_text or "").strip()
    if not text:
        logger.warning("Empty LLM output; using default NegotiationAction")
        return _default_action_from_observation(obs)

    if allow_json:
        candidate = text
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            candidate = re.sub(r"(?i)^json\s*", "", candidate).strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return _dict_to_action(data, obs)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    try:
        action_type = _infer_action_type(text)
        days = _extract_days(text)
        price = _extract_price(text, obs)

        if days is None:
            days = int(obs.get("buyer_days", 45))

        if action_type == "reject":
            return NegotiationAction(
                action_type="reject",
                price=round(float(obs.get("buyer_price", price)), 2),
                payment_days=int(obs.get("buyer_days", days)),
                use_treds=False,
                reason="Parsed reject from prose",
            )

        if action_type == "accept":
            return NegotiationAction(
                action_type="accept",
                price=round(float(obs.get("buyer_price", price)), 2),
                payment_days=int(obs.get("buyer_days", days)),
                use_treds=False,
                reason="Parsed accept from prose",
            )

        use_treds = bool(re.search(r"(?i)\btreds\b|TReDS", text))
        return NegotiationAction(
            action_type="propose",
            price=round(max(float(obs.get("cost_threshold", 0)), price), 2),
            payment_days=max(0, int(days)),
            use_treds=use_treds,
            reason="Parsed propose from prose",
        )
    except Exception as exc:
        logger.warning("LLM action parse failed (%s); using default. Snippet: %r", exc, text[:240])
        return _default_action_from_observation(obs)


def _dict_to_action(data: Dict[str, Any], obs: Dict[str, Any]) -> NegotiationAction:
    action_type = str(data.get("action_type", "propose")).lower()
    if action_type not in {"propose", "accept", "reject", "simulate_plan", "advance_period", "tool"}:
        action_type = "propose"
    price = float(data.get("price", obs.get("buyer_price", 100.0)))
    payment_days = int(data.get("payment_days", obs.get("buyer_days", 45)))
    return NegotiationAction(
        action_type=action_type,
        price=round(price, 2),
        payment_days=payment_days,
        use_treds=bool(data.get("use_treds", False)),
        reason=str(data.get("reason") or "") or None,
        tool_name=data.get("tool_name"),
        tool_args=data.get("tool_args"),
    )
