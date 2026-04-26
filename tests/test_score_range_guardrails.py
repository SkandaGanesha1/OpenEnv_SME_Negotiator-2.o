"""Regression tests to guarantee all task scores remain in the strict open interval (0, 1)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.environment import _strict_unit_interval as env_strict_unit_interval
from server.sme_environment import SMENegotiatorEnvironment
from sme_negotiator_env.graders import _strict_unit_interval as grader_strict_unit_interval
from sme_negotiator_env.models import NegotiationAction, default_negotiation_state


def _assert_open_interval(value: float) -> None:
    score = float(value)
    assert 0.0 < score < 1.0, f"score must be in (0, 1), got {score}"


def test_strict_interval_helpers_handle_boundaries_and_non_finite() -> None:
    candidates = [0.0, 1.0, -1.0, 2.0, 0.5, float("nan"), float("inf"), -float("inf")]
    for raw in candidates:
        g = grader_strict_unit_interval(raw)
        e = env_strict_unit_interval(raw)
        _assert_open_interval(g)
        _assert_open_interval(e)


def test_task_graders_non_deal_state_still_in_open_interval() -> None:
    from sme_negotiator_env.graders import TASK_GRADERS
    from sme_negotiator_env.task_config import TASK_REGISTRY

    tc = TASK_REGISTRY["payment-terms-medium"]
    state = default_negotiation_state(
        episode_id="guardrail-medium-42",
        seed=42,
        difficulty=tc.difficulty,
        task_name=tc.name,
        max_steps=tc.max_rounds,
        max_rounds=tc.max_rounds,
        buyer_price=tc.initial_buyer_price,
        buyer_days=tc.initial_buyer_days,
        initial_buyer_days=tc.initial_buyer_days,
        cost_threshold=tc.cost_threshold,
        liquidity_threshold=tc.liquidity_threshold,
        volume=tc.volume,
        sme_monthly_revenue=tc.sme_monthly_revenue,
        current_payment_terms_days=tc.current_payment_terms_days,
        sme_supplier_payment_days=tc.sme_supplier_payment_days,
        interest_rate_annual=tc.interest_rate_annual,
        buyer_power_score=tc.buyer_power_score,
        secondary_buyer_power=tc.secondary_buyer_power,
    )
    state.deal_reached = False
    for name, grader_fn in TASK_GRADERS.items():
        score = float(grader_fn(state))
        _assert_open_interval(score)


def test_terminal_reward_paths_always_in_open_interval() -> None:
    for difficulty in ("easy", "medium", "hard"):
        for seed in (1000, 1001, 1002):
            env = SMENegotiatorEnvironment()
            obs = env.reset(seed=seed, difficulty=difficulty)

            reject_obs = env.step(
                NegotiationAction(
                    action_type="reject",
                    price=obs.buyer_price,
                    payment_days=obs.buyer_days,
                    use_treds=False,
                )
            )
            _assert_open_interval(float(reject_obs.reward))

            env = SMENegotiatorEnvironment()
            obs = env.reset(seed=seed, difficulty=difficulty)
            invalid_accept_obs = env.step(
                NegotiationAction(
                    action_type="accept",
                    price=obs.buyer_price + 10.0,
                    payment_days=obs.buyer_days + 10,
                    use_treds=False,
                )
            )
            _assert_open_interval(float(invalid_accept_obs.reward))

            env = SMENegotiatorEnvironment()
            obs = env.reset(seed=seed, difficulty=difficulty)
            valid_accept_obs = env.step(
                NegotiationAction(
                    action_type="accept",
                    price=obs.buyer_price,
                    payment_days=obs.buyer_days,
                    use_treds=False,
                )
            )
            _assert_open_interval(float(valid_accept_obs.reward))
            assert math.isfinite(float(valid_accept_obs.reward))
