"""Stage 7 reward-ordering sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sme_negotiator_env.graders import (
    compute_reward_component_report,
    compute_total_sme_reward,
    compute_verifiable_reward,
)
from sme_negotiator_env.models import (
    BuyerState,
    FinancierState,
    SMEAccountState,
    WorldState,
    default_negotiation_state,
)


def _base_world_state() -> WorldState:
    return WorldState(
        smes=[
            SMEAccountState(
                sme_id="sme_0",
                cash_balance=250_000.0,
                supplier_payment_days=30,
                credit_limit=400_000.0,
                current_utilization=40_000.0,
                risk_score=0.3,
                required_minimum_cash=100_000.0,
                defaulted=False,
                missed_supplier_payment=False,
            )
        ],
        buyers=[
            BuyerState(
                buyer_id="buyer_0",
                demand_level=1.0,
                budget_per_period=1_000_000.0,
                default_tendency=0.1,
                baseline_payment_days=90,
            )
        ],
        financier=FinancierState(
            financier_id="financier_0",
            available_capital=1_000_000.0,
            risk_appetite=0.7,
            base_interest_rate=0.12,
        ),
        legal_max_payment_days=45,
        baseline_discount_rate=0.12,
        reward_lambda_shaping=0.1,
        current_period=0,
        total_periods=1,
        episode_step=0,
        history=[],
        deals=[],
    )


def _baseline_state():
    return default_negotiation_state(
        episode_id="reward-test",
        seed=1,
        difficulty="hard",
        task_name="liquidity-correlation-hard",
        max_steps=4,
        max_rounds=4,
        buyer_price=95.0,
        buyer_days=90,
        initial_buyer_days=90,
        cost_threshold=70.0,
        liquidity_threshold=45,
        volume=1_000,
        sme_monthly_revenue=500_000.0,
        current_payment_terms_days=90,
        sme_supplier_payment_days=30,
        interest_rate_annual=0.12,
        buyer_power_score=0.4,
        buyer_id="buyer_0",
        sme_id="sme_0",
        financier_id="financier_0",
        deal_id="deal_reward_test",
        message="baseline",
    )


def test_verifiable_reward_prefers_shorter_higher_value_terms() -> None:
    world_state = _base_world_state()
    baseline = _baseline_state()
    bad_final = baseline.model_copy(
        update={
            "deal_reached": True,
            "final_price": 88.0,
            "final_days": 75,
            "agreed_terms": 75,
            "buyer_days": 75,
            "late_payment_penalty_agreed": False,
            "message": "bad outcome",
        }
    )
    good_final = baseline.model_copy(
        update={
            "deal_reached": True,
            "final_price": 95.0,
            "final_days": 45,
            "agreed_terms": 45,
            "buyer_days": 45,
            "late_payment_penalty_agreed": False,
            "message": "good outcome",
        }
    )

    reward_bad = compute_verifiable_reward(world_state, [baseline, bad_final])
    reward_good = compute_verifiable_reward(world_state, [baseline, good_final])

    assert reward_good > reward_bad


def test_verifiable_reward_prefers_compliance_repair_via_penalty_clause() -> None:
    world_state = _base_world_state()
    baseline = _baseline_state()
    non_compliant = baseline.model_copy(
        update={
            "deal_reached": True,
            "final_price": 94.0,
            "final_days": 50,
            "agreed_terms": 50,
            "buyer_days": 50,
            "late_payment_penalty_agreed": False,
            "message": "non compliant",
        }
    )
    repaired = baseline.model_copy(
        update={
            "deal_reached": True,
            "final_price": 94.0,
            "final_days": 50,
            "agreed_terms": 50,
            "buyer_days": 50,
            "late_payment_penalty_agreed": True,
            "message": "repaired compliance",
        }
    )

    reward_non_compliant = compute_verifiable_reward(world_state, [baseline, non_compliant])
    reward_repaired = compute_verifiable_reward(world_state, [baseline, repaired])

    assert reward_repaired > reward_non_compliant


def test_reward_component_report_reconstructs_scalar_total_with_bounded_tool_bonus() -> None:
    world_state = _base_world_state()
    baseline = _baseline_state()
    improved = baseline.model_copy(
        update={
            "deal_reached": True,
            "final_price": 95.0,
            "final_days": 45,
            "agreed_terms": 45,
            "buyer_days": 45,
            "late_payment_penalty_agreed": False,
            "message": "improved",
        }
    )

    report = compute_reward_component_report(
        world_state,
        [baseline, improved],
        lambda_shaping=0.1,
        tool_bonus=0.01,
    )
    expected_total = round(compute_total_sme_reward(world_state, [baseline, improved], lambda_shaping=0.1) + 0.01, 6)

    assert report.total_reward == expected_total
    assert report.verifiable_reward == compute_verifiable_reward(world_state, [baseline, improved])
    assert abs(report.tool_bonus) <= 0.01
    assert report.success_no_default_positive_npv is True
