"""Deterministic tests for the Stage 2 RLVR reward helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sme_negotiator_env.graders import (
    compute_rubric_score,
    compute_shaping_rewards,
    compute_total_sme_reward,
    compute_verifiable_reward,
)
from sme_negotiator_env.models import BuyerState, NegotiationState, SMEAccountState, WorldState, default_negotiation_state


def _base_state() -> NegotiationState:
    return default_negotiation_state(
        episode_id="stage2-test",
        seed=11,
        difficulty="medium",
        task_name="liquidity-stress-medium",
        max_steps=12,
        max_rounds=12,
        buyer_price=100.0,
        buyer_days=90,
        initial_buyer_days=90,
        cost_threshold=80.0,
        liquidity_threshold=45,
        volume=1000,
        sme_monthly_revenue=400_000.0,
        current_payment_terms_days=90,
        sme_supplier_payment_days=30,
        interest_rate_annual=0.24,
        buyer_power_score=0.72,
        secondary_buyer_power=0.76,
        message="test trajectory",
    )


def _world_state(
    *,
    cash_balance: float = 150_000.0,
    required_minimum_cash: float = 100_000.0,
    current_utilization: float = 50_000.0,
    credit_limit: float = 200_000.0,
    legal_max_payment_days: int = 45,
    defaulted: bool = False,
    missed_supplier_payment: bool = False,
) -> WorldState:
    return WorldState(
        smes=[
            SMEAccountState(
                sme_id="sme_0",
                cash_balance=cash_balance,
                supplier_payment_days=30,
                credit_limit=credit_limit,
                current_utilization=current_utilization,
                risk_score=0.4,
                required_minimum_cash=required_minimum_cash,
                defaulted=defaulted,
                missed_supplier_payment=missed_supplier_payment,
            )
        ],
        buyers=[
            BuyerState(
                buyer_id="buyer_0",
                demand_level=1.0,
                budget_per_period=120_000.0,
                default_tendency=0.2,
                baseline_payment_days=90,
            )
        ],
        financier=None,
        legal_max_payment_days=legal_max_payment_days,
        baseline_discount_rate=0.0,
        reward_lambda_shaping=0.1,
    )


def _deal_state(*, price: float, days: int, late_penalty: bool = False) -> NegotiationState:
    return _base_state().model_copy(
        update={
            "deal_reached": True,
            "buyer_price": price,
            "buyer_days": days,
            "final_price": price,
            "final_days": days,
            "agreed_terms": days,
            "late_payment_penalty_agreed": late_penalty,
            "step_count": 3,
            "negotiation_round": 3,
        }
    )


def test_compute_verifiable_reward_is_deterministic() -> None:
    world_state = _world_state()
    trajectory = [_base_state(), _deal_state(price=103.0, days=45)]

    reward_a = compute_verifiable_reward(world_state, trajectory)
    reward_b = compute_verifiable_reward(world_state, trajectory)

    assert reward_a == reward_b


def test_compute_verifiable_reward_returns_zero_on_default_or_missed_supplier_payment() -> None:
    trajectory = [_base_state(), _deal_state(price=100.0, days=45)]

    defaulted_world = _world_state(defaulted=True)
    missed_payment_world = _world_state(missed_supplier_payment=True)

    assert compute_verifiable_reward(defaulted_world, trajectory) == 0.0
    assert compute_verifiable_reward(missed_payment_world, trajectory) == 0.0


def test_compute_verifiable_reward_increases_with_higher_final_cash() -> None:
    trajectory = [_base_state(), _deal_state(price=100.0, days=45)]

    low_cash_world = _world_state(cash_balance=50_000.0, required_minimum_cash=100_000.0)
    high_cash_world = _world_state(cash_balance=100_000.0, required_minimum_cash=100_000.0)

    assert compute_verifiable_reward(high_cash_world, trajectory) > compute_verifiable_reward(low_cash_world, trajectory)


def test_compute_verifiable_reward_improves_when_npv_beats_baseline() -> None:
    world_state = _world_state(legal_max_payment_days=120)
    baseline_state = _base_state()
    weak_deal = _deal_state(price=95.0, days=90)
    strong_deal = _deal_state(price=105.0, days=30)

    weak_reward = compute_verifiable_reward(world_state, [baseline_state, weak_deal])
    strong_reward = compute_verifiable_reward(world_state, [baseline_state, strong_deal])

    assert strong_reward > weak_reward


def test_compute_verifiable_reward_drops_when_payment_days_break_legal_cap() -> None:
    world_state = _world_state(legal_max_payment_days=45)
    baseline_state = _base_state()
    compliant = _deal_state(price=101.0, days=45)
    non_compliant = _deal_state(price=101.0, days=60)

    assert compute_verifiable_reward(world_state, [baseline_state, compliant]) > compute_verifiable_reward(
        world_state,
        [baseline_state, non_compliant],
    )


def test_compute_shaping_rewards_length_and_directionality() -> None:
    start = _base_state()
    improved = start.model_copy(
        update={
            "buyer_days": 50,
            "agreed_terms": 50,
            "step_count": 1,
            "negotiation_round": 1,
        }
    )
    regressed = start.model_copy(
        update={
            "buyer_days": 105,
            "agreed_terms": 105,
            "step_count": 1,
            "negotiation_round": 1,
        }
    )

    improving_rewards = compute_shaping_rewards([start, improved, _deal_state(price=102.0, days=40)])
    regressing_reward = compute_shaping_rewards([start, regressed])[0]

    assert len(improving_rewards) == 2
    assert improving_rewards[0] > 0.0
    assert regressing_reward <= 0.0


def test_compute_total_sme_reward_matches_terminal_plus_weighted_shaping_sum() -> None:
    world_state = _world_state()
    trajectory = [_base_state(), _deal_state(price=103.0, days=45)]
    lambda_shaping = 0.2

    total_reward = compute_total_sme_reward(world_state, trajectory, lambda_shaping=lambda_shaping)
    expected = compute_verifiable_reward(world_state, trajectory) + lambda_shaping * sum(
        compute_shaping_rewards(trajectory)
    )

    assert total_reward == pytest.approx(expected)


def test_compute_rubric_score_is_external_only_stub() -> None:
    with pytest.raises(NotImplementedError):
        compute_rubric_score("episode transcript")
