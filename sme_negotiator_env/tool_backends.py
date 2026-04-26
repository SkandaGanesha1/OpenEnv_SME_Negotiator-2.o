"""Hybrid deterministic/live-backed tool adapters for Theme 3.1 workflows."""

from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Any, Literal, Optional, Protocol

from sme_negotiator_env.models import NegotiationState, ToolResultEnvelope, WorldState
from sme_negotiator_env.tools import check_compliance, query_treds, run_cashflow_sim


ToolName = Literal["QUERY_TREDS", "CHECK_COMPLIANCE", "RUN_CASHFLOW_SIM"]


class LiveToolAdapter(Protocol):
    """Callable signature for optional live tool adapters."""

    def __call__(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        negotiation_state: Optional[NegotiationState] = None,
    ) -> dict[str, Any]:
        ...


def _stable_request_id(
    *,
    tool_name: ToolName,
    tool_args: dict[str, Any],
    context_fingerprint: str,
) -> str:
    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "context_fingerprint": context_fingerprint,
        },
        ensure_ascii=True,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _json_round_trip(payload: Any) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str))
    if not isinstance(normalized, dict):
        raise TypeError("Tool payload must normalize to a JSON object.")
    return normalized


class BaseToolBackend:
    """Base helper for deterministic and live-backed tool adapters."""

    tool_name: ToolName
    backend_name: str
    source: Literal["deterministic", "live"]

    def invoke(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        context_fingerprint: str,
        negotiation_state: Optional[NegotiationState] = None,
        cache_hit: bool = False,
        stale: bool = False,
    ) -> ToolResultEnvelope:
        start = perf_counter()
        payload = self._execute(
            world_state=world_state,
            tool_args=tool_args,
            negotiation_state=negotiation_state,
        )
        raw_latency_ms = int(max(0.0, (perf_counter() - start) * 1000.0))
        latency_ms = 0 if self.source == "deterministic" else raw_latency_ms
        normalized_payload = self.normalize_payload(payload)
        return ToolResultEnvelope(
            source=self.source,
            backend_name=self.backend_name,
            request_id=_stable_request_id(
                tool_name=self.tool_name,
                tool_args=tool_args,
                context_fingerprint=context_fingerprint,
            ),
            latency_ms=latency_ms,
            cache_hit=bool(cache_hit),
            stale=bool(stale),
            normalized_payload=normalized_payload,
            requested_source=self.source,
        )

    def _execute(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        negotiation_state: Optional[NegotiationState] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _json_round_trip(payload)


class DeterministicTredsBackend(BaseToolBackend):
    tool_name = "QUERY_TREDS"
    backend_name = "DeterministicTredsBackend"
    source = "deterministic"

    def _execute(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        negotiation_state: Optional[NegotiationState] = None,
    ) -> dict[str, Any]:
        invoice_id = str(tool_args.get("invoice_id") or tool_args.get("deal_id") or "")
        return query_treds(world_state, invoice_id)

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = _json_round_trip(payload)
        return {
            "invoice_id": str(data.get("invoice_id", "")),
            "quote_options": [
                {
                    "tenor_days": int(option.get("tenor_days", 0)),
                    "annual_discount_rate": float(option.get("annual_discount_rate", 0.0)),
                    "discount_fee": float(option.get("discount_fee", 0.0)),
                    "advance_amount": float(option.get("advance_amount", 0.0)),
                    "available": bool(option.get("available", False)),
                }
                for option in list(data.get("quote_options", []))
            ],
            "recommended_tenor_days": int(data.get("recommended_tenor_days", 0)),
            "available_capital": float(data.get("available_capital", 0.0)),
        }


class DeterministicComplianceBackend(BaseToolBackend):
    tool_name = "CHECK_COMPLIANCE"
    backend_name = "DeterministicComplianceBackend"
    source = "deterministic"

    def _execute(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        negotiation_state: Optional[NegotiationState] = None,
    ) -> dict[str, Any]:
        contract_id = str(tool_args.get("contract_id") or tool_args.get("deal_id") or "")
        return check_compliance(
            world_state,
            contract_id,
            negotiation_state=negotiation_state,
        )

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = _json_round_trip(payload)
        checked_terms = dict(data.get("checked_terms") or {})
        return {
            "contract_id": str(data.get("contract_id", "")),
            "is_compliant": bool(data.get("is_compliant", False)),
            "violated_clauses": [str(item) for item in list(data.get("violated_clauses", []))],
            "checked_terms": {
                "payment_days": int(checked_terms.get("payment_days", 0)),
                "late_payment_penalty_agreed": bool(checked_terms.get("late_payment_penalty_agreed", False)),
                "dynamic_discounting_agreed": bool(checked_terms.get("dynamic_discounting_agreed", False)),
                "dynamic_discount_annual_rate": float(checked_terms.get("dynamic_discount_annual_rate", 0.0)),
                "policy_cap": float(checked_terms.get("policy_cap", 0.0)),
            },
            "explanation": str(data.get("explanation", "")),
        }


class DeterministicCashflowBackend(BaseToolBackend):
    tool_name = "RUN_CASHFLOW_SIM"
    backend_name = "DeterministicCashflowBackend"
    source = "deterministic"

    def _execute(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        negotiation_state: Optional[NegotiationState] = None,
    ) -> dict[str, Any]:
        plan = tool_args.get("plan") if isinstance(tool_args.get("plan"), dict) else {}
        horizon = int(tool_args.get("horizon")) if tool_args.get("horizon") is not None else 0
        return run_cashflow_sim(world_state, plan, horizon=max(0, horizon))

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = _json_round_trip(payload)
        return {
            "period_balances": [float(item) for item in list(data.get("period_balances", []))],
            "period_defaults": [bool(item) for item in list(data.get("period_defaults", []))],
            "period_penalties": [float(item) for item in list(data.get("period_penalties", []))],
            "ending_balance": float(data.get("ending_balance", 0.0)),
            "any_default": bool(data.get("any_default", False)),
            "total_penalty_exposure": float(data.get("total_penalty_exposure", 0.0)),
        }


class LiveFallbackToolBackend(BaseToolBackend):
    """Optional live-backed adapter with deterministic fallback."""

    def __init__(
        self,
        *,
        adapter: Optional[LiveToolAdapter],
        fallback_backend: BaseToolBackend,
        backend_name: str,
    ) -> None:
        self._adapter = adapter
        self._fallback_backend = fallback_backend
        self.backend_name = backend_name
        self.tool_name = fallback_backend.tool_name
        self.source = "live"

    def _execute(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        negotiation_state: Optional[NegotiationState] = None,
    ) -> dict[str, Any]:
        if self._adapter is None:
            raise RuntimeError("No live adapter configured.")
        payload = self._adapter(
            world_state=world_state,
            tool_args=tool_args,
            negotiation_state=negotiation_state,
        )
        if not isinstance(payload, dict):
            raise TypeError("Live adapter must return a dict payload.")
        return payload

    def invoke(
        self,
        *,
        world_state: WorldState,
        tool_args: dict[str, Any],
        context_fingerprint: str,
        negotiation_state: Optional[NegotiationState] = None,
        cache_hit: bool = False,
        stale: bool = False,
    ) -> ToolResultEnvelope:
        try:
            return super().invoke(
                world_state=world_state,
                tool_args=tool_args,
                context_fingerprint=context_fingerprint,
                negotiation_state=negotiation_state,
                cache_hit=cache_hit,
                stale=stale,
            )
        except Exception as exc:
            fallback = self._fallback_backend.invoke(
                world_state=world_state,
                tool_args=tool_args,
                context_fingerprint=context_fingerprint,
                negotiation_state=negotiation_state,
                cache_hit=cache_hit,
                stale=stale,
            )
            return fallback.model_copy(
                update={
                    "requested_source": "live",
                    "fallback_reason": str(exc),
                }
            )


class LiveTredsBackend(LiveFallbackToolBackend):
    def __init__(self, adapter: Optional[LiveToolAdapter], fallback_backend: Optional[BaseToolBackend] = None) -> None:
        super().__init__(
            adapter=adapter,
            fallback_backend=fallback_backend or DeterministicTredsBackend(),
            backend_name="LiveTredsBackend",
        )


class LiveComplianceBackend(LiveFallbackToolBackend):
    def __init__(self, adapter: Optional[LiveToolAdapter], fallback_backend: Optional[BaseToolBackend] = None) -> None:
        super().__init__(
            adapter=adapter,
            fallback_backend=fallback_backend or DeterministicComplianceBackend(),
            backend_name="LiveComplianceBackend",
        )


class LiveCashflowBackend(LiveFallbackToolBackend):
    def __init__(self, adapter: Optional[LiveToolAdapter], fallback_backend: Optional[BaseToolBackend] = None) -> None:
        super().__init__(
            adapter=adapter,
            fallback_backend=fallback_backend or DeterministicCashflowBackend(),
            backend_name="LiveCashflowBackend",
        )


def build_tool_backend_registry(
    *,
    mode: Literal["deterministic", "live"] = "deterministic",
    live_adapters: Optional[dict[ToolName, LiveToolAdapter]] = None,
) -> dict[ToolName, BaseToolBackend]:
    """Build the canonical Theme 3.1 tool registry for the liquidity environment."""

    deterministic: dict[ToolName, BaseToolBackend] = {
        "QUERY_TREDS": DeterministicTredsBackend(),
        "CHECK_COMPLIANCE": DeterministicComplianceBackend(),
        "RUN_CASHFLOW_SIM": DeterministicCashflowBackend(),
    }
    if mode != "live":
        return deterministic

    adapters = live_adapters or {}
    return {
        "QUERY_TREDS": LiveTredsBackend(adapters.get("QUERY_TREDS"), deterministic["QUERY_TREDS"]),
        "CHECK_COMPLIANCE": LiveComplianceBackend(
            adapters.get("CHECK_COMPLIANCE"),
            deterministic["CHECK_COMPLIANCE"],
        ),
        "RUN_CASHFLOW_SIM": LiveCashflowBackend(
            adapters.get("RUN_CASHFLOW_SIM"),
            deterministic["RUN_CASHFLOW_SIM"],
        ),
    }
