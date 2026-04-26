"""OpenEnv client for the SME negotiation environment."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from openenv.core import GenericEnvClient
from openenv.core.client_types import StepResult

from .models import NegotiationAction, NegotiationObservation, NegotiationState


def choose_action(observation: NegotiationObservation, round_number: int) -> NegotiationAction:
    """State-aware heuristic policy that reads the current observation."""

    buyer_price = float(observation.buyer_price)
    buyer_days = int(observation.buyer_days)
    cost = float(observation.cost_threshold)
    liquidity = int(observation.liquidity_threshold)
    difficulty = str(getattr(observation, "difficulty", "") or "").lower()

    if difficulty == "easy":
        target_days = liquidity
        target_price = round(cost + 10.0, 2)
        if round_number > 0:
            return NegotiationAction(
                action_type="accept",
                price=target_price,
                payment_days=target_days,
                use_treds=False,
                reason="Close on the previously proposed liquidity-safe terms.",
            )
        if buyer_days <= target_days and buyer_price >= cost + 1.0:
            return NegotiationAction(
                action_type="accept",
                price=buyer_price,
                payment_days=buyer_days,
                use_treds=False,
                reason="Accept cooperative-buyer terms inside the liquidity threshold.",
            )
        return NegotiationAction(
            action_type="propose",
            price=target_price,
            payment_days=target_days,
            use_treds=False,
            reason="Easy task target: settle at or below the liquidity threshold.",
        )

    if difficulty == "hard":
        target_days = min(45, liquidity)
        target_price = round(cost + 10.0, 2)
        if round_number > 0:
            return NegotiationAction(
                action_type="accept",
                price=target_price,
                payment_days=target_days,
                use_treds=True,
                propose_dynamic_discounting=True,
                dynamic_discount_annual_rate=0.02,
                reason="Close on previously proposed dynamic-discounting terms.",
            )
        if buyer_days <= target_days and buyer_price >= cost + 1.0:
            return NegotiationAction(
                action_type="accept",
                price=buyer_price,
                payment_days=buyer_days,
                use_treds=True,
                propose_dynamic_discounting=True,
                dynamic_discount_annual_rate=0.02,
                reason="Accept only after dynamic-discounting terms improve NPV versus status quo.",
            )
        return NegotiationAction(
            action_type="propose",
            price=target_price,
            payment_days=target_days,
            use_treds=True,
            propose_dynamic_discounting=True,
            dynamic_discount_annual_rate=0.02,
            reason="Hard task target: dynamic discounting with faster payment and positive margin.",
        )

    if round_number == 0:
        target_price = round(cost + 10.0, 2)
        target_days = max(0, liquidity)
        return NegotiationAction(
            action_type="propose",
            price=target_price,
            payment_days=target_days,
            use_treds=False,
            propose_late_payment_penalty_clause=True,
            reason="Opening offer anchored slightly below the buyer's ask",
        )

    if round_number == 1:
        return NegotiationAction(
            action_type="accept",
            price=round(cost + 10.0, 2),
            payment_days=max(0, liquidity),
            use_treds=False,
            propose_late_payment_penalty_clause=True,
            reason="Close on previously proposed liquidity-safe terms with penalty protection.",
        )

    if buyer_price > cost + 5:
        target_price = round(max(cost + 2.0, buyer_price * 1.02), 2)
        target_days = max(liquidity, buyer_days - 5)
        return NegotiationAction(
            action_type="propose",
            price=target_price,
            payment_days=target_days,
            use_treds=False,
            propose_late_payment_penalty_clause=True,
            reason="Countering above cost with a modest price improvement",
        )

    if buyer_price > cost and buyer_days <= liquidity:
        return NegotiationAction(
            action_type="accept",
            price=buyer_price,
            payment_days=buyer_days,
            use_treds=False,
            propose_late_payment_penalty_clause=True,
            reason="Current offer is viable within liquidity threshold",
        )

    if buyer_days > liquidity and round_number >= 2:
        target_price = round(max(cost + 2.0, buyer_price * 1.01), 2)
        target_days = max(0, liquidity)
        return NegotiationAction(
            action_type="propose",
            price=target_price,
            payment_days=target_days,
            use_treds=True,
            propose_late_payment_penalty_clause=True,
            reason="TReDS-enabled proposal to compress payment terms",
        )

    target_price = round(max(cost + 2.0, buyer_price * 1.015), 2)
    target_days = max(liquidity, buyer_days - 5)
    return NegotiationAction(
        action_type="propose",
        price=target_price,
        payment_days=target_days,
        use_treds=buyer_days > liquidity,
        propose_late_payment_penalty_clause=True,
        reason="Incremental counter-offer based on observed buyer state",
    )


class SMENegotiatorEnv(GenericEnvClient):
    """Typed OpenEnv client for the SME negotiation environment."""

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        provider: Optional[Any] = None,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            provider=provider,
            mode=mode,
        )
        self._last_observation: Optional[NegotiationObservation] = None

    def _step_payload(self, action: NegotiationAction) -> Dict[str, Any]:
        if hasattr(action, "model_dump"):
            return action.model_dump()
        if isinstance(action, dict):
            return dict(action)
        return dict(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[NegotiationObservation]:
        observation_payload = payload.get("observation", payload)
        if isinstance(observation_payload, NegotiationObservation):
            observation = observation_payload
        else:
            observation = NegotiationObservation(**observation_payload)

        result = StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )
        self._last_observation = observation
        return result

    def _parse_state(self, payload: Dict[str, Any]) -> NegotiationState:
        if isinstance(payload, NegotiationState):
            return payload
        return NegotiationState(**payload)

    async def negotiate(self, max_rounds: Optional[int] = None) -> StepResult[NegotiationObservation]:
        """Run a full negotiation episode with the built-in heuristic policy."""

        result = await self.reset()
        observation = result.observation
        round_number = 0

        while not result.done and (max_rounds is None or round_number < max_rounds):
            action = choose_action(observation, round_number)
            result = await self.step(action)
            observation = result.observation
            round_number += 1

        return result
