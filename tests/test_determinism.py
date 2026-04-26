"""Stage 7 determinism tests for the liquidity environment."""

from __future__ import annotations

import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.environment import SMELiquidityEnvironment
from sme_negotiator_env.models import LiquidityObservation
from sme_negotiator_env.models import NegotiationAction


def _normalize_payload(value):
    if isinstance(value, dict):
        return {
            key: _normalize_payload(item)
            for key, item in value.items()
            if key != "message"
        }
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"Episode reset @ [^)]+", "Episode reset @ <normalized>", value)
    return value


def _fixed_policy_action(env: SMELiquidityEnvironment, observation: LiquidityObservation) -> NegotiationAction:
    assert env.state is not None
    if observation.open_deal_ids:
        deal_id = observation.open_deal_ids[0]
        negotiation = env.state.current_negotiations[deal_id]
        return NegotiationAction(
            action_type="accept",
            deal_id=deal_id,
            price=float(negotiation.buyer_price),
            payment_days=int(negotiation.buyer_days),
            use_treds=bool(negotiation.buyer_days > negotiation.liquidity_threshold),
            reason="Deterministic test acceptance",
        )
    return NegotiationAction(action_type="advance_period")


def _run_episode(seed: int) -> tuple[list[dict[str, object]], list[float], list[bool], dict[str, object]]:
    env = SMELiquidityEnvironment(total_periods=2)
    observation = env.reset(seed=seed, difficulty="hard", task_name="liquidity-correlation-hard")

    observations = [observation.model_dump()]
    rewards: list[float] = []
    done_trace = [bool(observation.done)]

    for _ in range(12):
        if env.state is None or done_trace[-1]:
            break
        action = _fixed_policy_action(env, observation)
        next_observation = env.step(action)
        observations.append(next_observation.model_dump())
        rewards.append(float(next_observation.reward))
        done_trace.append(bool(next_observation.done))
        observation = next_observation
        if next_observation.done:
            break

    assert env.state is not None
    return (
        _normalize_payload(observations),
        rewards,
        done_trace,
        _normalize_payload(env.state.model_dump()),
    )


def test_liquidity_episode_is_deterministic_for_same_seed() -> None:
    first = _run_episode(42)
    second = _run_episode(42)

    assert first == second


def test_liquidity_episode_changes_for_different_seed() -> None:
    first = _run_episode(42)
    second = _run_episode(43)

    assert first != second
