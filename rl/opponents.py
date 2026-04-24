"""Training-side opponent policies and snapshot management for Stage 6."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Optional

from sme_negotiator_env.llm_action_parser import parse_llm_text_to_negotiation_action
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation
from sme_negotiator_env.prompting import (
    action_payload_to_model_action,
    conservative_default_action,
    format_observation_text,
)


class TextPolicy(ABC):
    """Abstract policy interface for buyer-like textual counterparties."""

    @abstractmethod
    def act(self, observation: NegotiationObservation) -> NegotiationAction:
        """Return a structured action for the given observation."""

    @property
    def policy_id(self) -> str:
        """Stable identifier for logging."""
        return type(self).__name__


@dataclass(frozen=True)
class FinancierQuote:
    """Deterministic financing decision returned by a financier policy."""

    approved: bool
    annual_rate: float
    approved_amount: float
    reason: str = ""


class FinancierPolicy(ABC):
    """Abstract policy interface for financing quote generation."""

    @abstractmethod
    def act(self, observation: NegotiationObservation) -> FinancierQuote:
        """Return a financing decision for the current negotiation context."""

    @property
    def policy_id(self) -> str:
        """Stable identifier for logging."""
        return type(self).__name__


def _extract_json_suffix(message: str, prefix: str) -> dict[str, Any]:
    marker = f"{prefix}="
    if marker not in message:
        return {}
    suffix = message.split(marker, 1)[1].strip()
    try:
        payload = json.loads(suffix)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


class HeuristicBuyerPolicy(TextPolicy):
    """Buyer policy that reproduces the Stage 0–5 counter-offer heuristic."""

    def act(self, observation: NegotiationObservation) -> NegotiationAction:
        context = _extract_json_suffix(str(observation.message or ""), "BUYER_CONTEXT")
        proposed_price = float(context.get("proposed_price", observation.buyer_price))
        current_price = float(context.get("current_buyer_price", observation.buyer_price))
        current_days = int(context.get("current_buyer_days", observation.buyer_days))
        buyer_min_days_floor = int(context.get("buyer_min_days_floor", current_days))
        cost_threshold = float(context.get("cost_threshold", observation.cost_threshold))
        price_drop = float(context.get("price_drop", 0.0))
        day_drop = max(1, int(context.get("day_drop", 1)))

        next_price = round(max(cost_threshold, min(current_price, proposed_price) - price_drop), 2)
        next_days = max(buyer_min_days_floor, current_days - day_drop)
        return NegotiationAction(
            action_type="propose",
            price=next_price,
            payment_days=int(next_days),
            use_treds=bool(context.get("treds_used", False)),
            reason="Heuristic buyer counter-offer",
        )

    @property
    def policy_id(self) -> str:
        return "heuristic_buyer"


class HeuristicFinancierPolicy(FinancierPolicy):
    """Financier policy matching the Stage 0–5 quote heuristic."""

    def act(self, observation: NegotiationObservation) -> FinancierQuote:
        context = _extract_json_suffix(str(observation.message or ""), "FINANCIER_CONTEXT")
        rate = float(context.get("heuristic_rate", observation.interest_rate_annual))
        approved_amount = float(context.get("requested_amount", 0.0))
        available_capital = float(context.get("available_capital", approved_amount))
        return FinancierQuote(
            approved=bool(approved_amount > 0.0 and available_capital > 0.0),
            annual_rate=round(rate, 6),
            approved_amount=round(min(approved_amount, available_capital), 2),
            reason="Heuristic financier quote",
        )

    @property
    def policy_id(self) -> str:
        return "heuristic_financier"


class SnapshotLLMBuyerPolicy(TextPolicy):
    """Frozen LLM snapshot acting from the buyer viewpoint."""

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 128) -> None:
        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._tokenizer = None

    @property
    def policy_id(self) -> str:
        return f"snapshot_buyer:{Path(self.model_path).name}"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load snapshot buyer policies. Install optional RL dependencies."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def act(self, observation: NegotiationObservation) -> NegotiationAction:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        prompt = format_observation_text(observation, role="buyer")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            return parse_llm_text_to_negotiation_action(text, observation, allow_json=True)
        except Exception:
            return conservative_default_action(observation)


class SnapshotLLMFinancierPolicy(FinancierPolicy):
    """Frozen LLM snapshot acting from the financier viewpoint."""

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 128) -> None:
        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._tokenizer = None

    @property
    def policy_id(self) -> str:
        return f"snapshot_financier:{Path(self.model_path).name}"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load snapshot financier policies. Install optional RL dependencies."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def act(self, observation: NegotiationObservation) -> FinancierQuote:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        prompt = format_observation_text(observation, role="financier")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        payload: dict[str, Any]
        try:
            payload = json.loads(text.strip())
            if not isinstance(payload, dict):
                raise ValueError("Financier snapshot output must decode to a JSON object.")
        except Exception:
            payload = {}
        approved = bool(payload.get("approved", payload.get("approve_financing", False)))
        annual_rate = float(payload.get("annual_rate", observation.interest_rate_annual))
        approved_amount = float(payload.get("approved_amount", 0.0))
        return FinancierQuote(
            approved=approved,
            annual_rate=round(annual_rate, 6),
            approved_amount=round(max(0.0, approved_amount), 2),
            reason=str(payload.get("reason", "Snapshot financier decision"))[:160],
        )


@dataclass
class OpponentPolicyManager:
    """Manage a rolling zoo of frozen snapshot opponents plus heuristics."""

    snapshots_dir: Path
    zoo_size: int = 5
    latest_weight: float = 0.6
    older_weight: float = 0.3
    heuristic_weight: float = 0.1
    snapshot_paths: list[Path] = field(default_factory=list)
    _buyer_cache: dict[str, TextPolicy] = field(default_factory=dict, init=False, repr=False)
    _financier_cache: dict[str, FinancierPolicy] = field(default_factory=dict, init=False, repr=False)

    def register_snapshot(self, snapshot_path: str | Path) -> None:
        """Add a new snapshot and prune the zoo to the configured size."""
        path = Path(snapshot_path)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        if path not in self.snapshot_paths:
            self.snapshot_paths.append(path)
        self.snapshot_paths = sorted(self.snapshot_paths, key=lambda item: item.name)[-self.zoo_size :]

    def snapshot_ids(self) -> list[str]:
        """Return stable ids for all managed snapshots."""
        return [path.name for path in self.snapshot_paths]

    def _sample_snapshot_path(self, seed: int) -> Optional[Path]:
        if not self.snapshot_paths:
            return None
        rng = Random(int(seed))
        draw = rng.random()
        if draw < self.heuristic_weight:
            return None
        if len(self.snapshot_paths) == 1 or draw < self.heuristic_weight + self.latest_weight:
            return self.snapshot_paths[-1]
        older = self.snapshot_paths[:-1]
        return older[rng.randrange(len(older))] if older else self.snapshot_paths[-1]

    def _buyer_policy_for_snapshot(self, snapshot_path: Optional[Path]) -> TextPolicy:
        if snapshot_path is None:
            return HeuristicBuyerPolicy()
        cache_key = str(snapshot_path)
        if cache_key not in self._buyer_cache:
            self._buyer_cache[cache_key] = SnapshotLLMBuyerPolicy(snapshot_path)
        return self._buyer_cache[cache_key]

    def _financier_policy_for_snapshot(self, snapshot_path: Optional[Path]) -> FinancierPolicy:
        if snapshot_path is None:
            return HeuristicFinancierPolicy()
        cache_key = str(snapshot_path)
        if cache_key not in self._financier_cache:
            self._financier_cache[cache_key] = SnapshotLLMFinancierPolicy(snapshot_path)
        return self._financier_cache[cache_key]

    def sample_policies(self, *, seed: int) -> tuple[TextPolicy, FinancierPolicy, str, str]:
        """Sample buyer and financier policies deterministically from the zoo."""
        buyer_snapshot = self._sample_snapshot_path(seed + 17)
        financier_snapshot = self._sample_snapshot_path(seed + 29)
        buyer_policy = self._buyer_policy_for_snapshot(buyer_snapshot)
        financier_policy = self._financier_policy_for_snapshot(financier_snapshot)
        return buyer_policy, financier_policy, buyer_policy.policy_id, financier_policy.policy_id
