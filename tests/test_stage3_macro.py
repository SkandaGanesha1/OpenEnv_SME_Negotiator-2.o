"""Stage 3 macro-level tests: pure simulation functions, flat projection fields, and env invariants.

These tests are focused on the new structures introduced in Stage 3:
  - WorldSnapshot aggregate logging
  - CashflowProjection parallel-list contract
  - simulate_cashflow() pure-function guarantees
  - advance_world_state() deterministic period mechanics
  - NegotiationAction optional price/payment_days for non-negotiation action types
  - LiquidityObservation flat projection field exposure
  - SMELiquidityEnvironment configurable total_periods
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sme_negotiator_env.models import (
    BuyerState,
    CashflowProjection,
    DealState,
    FinancierState,
    NegotiationAction,
    SMEAccountState,
    WorldSnapshot,
    WorldState,
)
from sme_negotiator_env.simulation import (
    advance_world_state,
    apply_plan_to_world_state,
    simulate_cashflow,
)
from server.sme_environment import SMELiquidityEnvironment


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _minimal_world_state(*, cash: float = 200_000.0, total_periods: int = 3) -> WorldState:
    return WorldState(
        smes=[
            SMEAccountState(
                sme_id="sme_0",
                cash_balance=cash,
                supplier_payment_days=30,
                credit_limit=500_000.0,
                current_utilization=0.0,
                risk_score=0.3,
                required_minimum_cash=50_000.0,
            )
        ],
        buyers=[
            BuyerState(
                buyer_id="buyer_0",
                demand_level=1.0,
                budget_per_period=150_000.0,
                default_tendency=0.1,
                baseline_payment_days=60,
            )
        ],
        current_period=0,
        total_periods=total_periods,
    )


def _world_with_agreed_deal(
    *,
    cash: float = 100_000.0,
    invoice_amount: float = 50_000.0,
    supplier_payment_amount: float = 30_000.0,
    finance_rate: float = 0.24,
    financed: bool = False,
    buyer_due_period: int = 1,
    supplier_due_period: int = 1,
) -> WorldState:
    ws = _minimal_world_state(cash=cash)
    ws.financier = FinancierState(
        financier_id="financier_0",
        available_capital=500_000.0,
        risk_appetite=0.6,
        base_interest_rate=0.22,
    )
    deal = DealState(
        deal_id="deal_test",
        sme_id="sme_0",
        buyer_id="buyer_0",
        status="agreed",
        created_period=0,
        invoice_amount=invoice_amount,
        supplier_payment_amount=supplier_payment_amount,
        agreed_price=100.0,
        agreed_payment_days=60,
        financed=financed,
        finance_rate=finance_rate,
        supplier_due_period=supplier_due_period,
        buyer_due_period=buyer_due_period,
        volume=500,
        financing_principal=invoice_amount if financed else 0.0,
        initial_funding_applied=financed,
    )
    ws.deals.append(deal)
    return ws


def _plan_action(plan: dict, *, horizon: int = 2) -> NegotiationAction:
    return NegotiationAction(
        action_type="simulate_plan",
        simulation_plan=plan,
        simulation_horizon=horizon,
    )


def _advance_action() -> NegotiationAction:
    return NegotiationAction(action_type="advance_period")


# ---------------------------------------------------------------------------
# WorldSnapshot model contract
# ---------------------------------------------------------------------------


def test_world_snapshot_has_all_required_fields() -> None:
    snap = WorldSnapshot(
        period_index=3,
        total_cash_balance=500_000.0,
        defaulted_sme_count=0,
        open_deal_count=2,
        resolved_deal_count=1,
        average_payment_days=45.0,
        total_penalty_exposure=0.0,
    )
    assert snap.period_index == 3
    assert snap.total_cash_balance == 500_000.0
    assert snap.defaulted_sme_count == 0
    assert snap.open_deal_count == 2
    assert snap.resolved_deal_count == 1
    assert snap.average_payment_days == 45.0
    assert snap.total_penalty_exposure == 0.0


# ---------------------------------------------------------------------------
# CashflowProjection parallel-list contract
# ---------------------------------------------------------------------------


def test_cashflow_projection_lists_are_parallel_and_correctly_typed() -> None:
    ws = _minimal_world_state()
    proj = simulate_cashflow(ws, {}, horizon=3)

    assert len(proj.period_balances) == 3
    assert len(proj.period_defaults) == 3
    assert len(proj.period_penalties) == 3
    assert all(isinstance(b, float) for b in proj.period_balances)
    assert all(isinstance(d, bool) for d in proj.period_defaults)
    assert all(isinstance(p, float) for p in proj.period_penalties)


def test_cashflow_projection_zero_horizon_returns_empty_lists() -> None:
    ws = _minimal_world_state()
    proj = simulate_cashflow(ws, {}, horizon=0)

    assert proj.period_balances == []
    assert proj.period_defaults == []
    assert proj.period_penalties == []


# ---------------------------------------------------------------------------
# simulate_cashflow() pure-function guarantees
# ---------------------------------------------------------------------------


def test_simulate_cashflow_is_pure_no_side_effects_on_original() -> None:
    ws = _minimal_world_state(cash=200_000.0)
    original_cash = ws.smes[0].cash_balance
    original_period = ws.current_period

    simulate_cashflow(ws, {}, horizon=3)

    assert ws.smes[0].cash_balance == original_cash
    assert ws.current_period == original_period


def test_simulate_cashflow_identical_on_repeated_calls() -> None:
    ws = _minimal_world_state()
    plan = {"advance_periods": 2}

    first = simulate_cashflow(ws, plan, horizon=2)
    second = simulate_cashflow(ws, plan, horizon=2)

    assert first.model_dump() == second.model_dump()


def test_simulate_cashflow_with_accept_plan_differs_from_empty_plan() -> None:
    ws = _minimal_world_state(cash=100_000.0)
    # Add an open deal that can be accepted in the plan
    ws.deals.append(
        DealState(
            deal_id="deal_open_0",
            sme_id="sme_0",
            buyer_id="buyer_0",
            status="open",
            created_period=0,
            invoice_amount=50_000.0,
            supplier_payment_amount=30_000.0,
            volume=500,
        )
    )

    accept_plan = {
        "deal_decisions": {
            "deal_open_0": {
                "decision": "accept",
                "price": 100.0,
                "payment_days": 30,
                "use_treds": False,
            }
        }
    }
    empty_plan = {}

    proj_accept = simulate_cashflow(ws, accept_plan, horizon=2)
    proj_empty = simulate_cashflow(ws, empty_plan, horizon=2)

    # Accepting a deal changes future balances (inflows arrive later)
    assert proj_accept.period_balances != proj_empty.period_balances


# ---------------------------------------------------------------------------
# advance_world_state() deterministic period mechanics
# ---------------------------------------------------------------------------


def test_advance_world_state_increments_period_counter_correctly() -> None:
    ws = _minimal_world_state()
    assert ws.current_period == 0

    for expected_period in range(1, 4):
        ws = advance_world_state(ws)
        assert ws.current_period == expected_period


def test_advance_world_state_logs_snapshot_per_advance() -> None:
    ws = _minimal_world_state()
    assert len(ws.history) == 0

    ws = advance_world_state(ws)
    assert len(ws.history) == 1
    assert ws.history[0].period_index == 0

    ws = advance_world_state(ws)
    assert len(ws.history) == 2
    assert ws.history[1].period_index == 1


def test_advance_world_state_snapshot_total_cash_matches_sme_sum() -> None:
    ws = _minimal_world_state(cash=200_000.0)
    ws = advance_world_state(ws)

    snapshot_cash = ws.history[0].total_cash_balance
    actual_cash = sum(sme.cash_balance for sme in ws.smes)
    assert snapshot_cash == actual_cash


def test_advance_world_state_accrues_financing_interest() -> None:
    ws = _world_with_agreed_deal(
        cash=100_000.0,
        invoice_amount=50_000.0,
        financed=True,
        finance_rate=0.24,
        buyer_due_period=3,
        supplier_due_period=3,
    )
    cash_before = ws.smes[0].cash_balance
    ws_after = advance_world_state(ws)

    # Expected monthly interest: 50_000 * 0.24 / 12 = 1_000
    expected_interest = round(50_000.0 * 0.24 / 12.0, 2)
    assert ws_after.deals[0].accrued_interest == expected_interest
    assert ws_after.smes[0].cash_balance == round(cash_before - expected_interest, 2)


def test_advance_world_state_settles_buyer_payment_on_due_period() -> None:
    ws = _world_with_agreed_deal(
        cash=50_000.0,
        invoice_amount=40_000.0,
        financed=False,
        buyer_due_period=1,
        supplier_due_period=3,
    )
    cash_before = ws.smes[0].cash_balance
    ws_after = advance_world_state(ws)

    # Buyer payment arrives: SME cash should increase by invoice_amount
    assert ws_after.deals[0].settled is True
    assert ws_after.deals[0].status == "settled"
    assert ws_after.smes[0].cash_balance == round(cash_before + 40_000.0, 2)


def test_advance_world_state_deducts_supplier_payment_on_due_period() -> None:
    ws = _world_with_agreed_deal(
        cash=100_000.0,
        supplier_payment_amount=30_000.0,
        financed=False,
        supplier_due_period=1,
        buyer_due_period=3,
    )
    cash_before = ws.smes[0].cash_balance
    ws_after = advance_world_state(ws)

    assert ws_after.deals[0].supplier_paid is True
    assert ws_after.smes[0].cash_balance == round(cash_before - 30_000.0, 2)


def test_advance_world_state_does_not_mutate_original() -> None:
    ws = _minimal_world_state(cash=200_000.0)
    cash_before = ws.smes[0].cash_balance
    period_before = ws.current_period

    advance_world_state(ws)

    assert ws.smes[0].cash_balance == cash_before
    assert ws.current_period == period_before


# ---------------------------------------------------------------------------
# NegotiationAction: optional price / payment_days
# ---------------------------------------------------------------------------


def test_negotiation_action_price_and_payment_days_default_to_zero() -> None:
    action_sim = NegotiationAction(action_type="simulate_plan", simulation_horizon=2)
    assert action_sim.price == 0.0
    assert action_sim.payment_days == 0

    action_adv = NegotiationAction(action_type="advance_period")
    assert action_adv.price == 0.0
    assert action_adv.payment_days == 0


def test_negotiation_action_explicit_values_still_accepted() -> None:
    action = NegotiationAction(action_type="propose", price=95.0, payment_days=45)
    assert action.price == 95.0
    assert action.payment_days == 45


# ---------------------------------------------------------------------------
# LiquidityObservation: flat projection fields
# ---------------------------------------------------------------------------


def test_flat_projection_fields_none_before_any_simulation() -> None:
    env = SMELiquidityEnvironment(total_periods=3)
    obs = env.reset(seed=42, difficulty="medium")

    assert obs.simulation_projection is None
    assert obs.projected_balances is None
    assert obs.projected_defaults is None
    assert obs.projected_penalties is None


def test_flat_projection_fields_populated_after_simulate_plan() -> None:
    env = SMELiquidityEnvironment(total_periods=6)
    obs = env.reset(seed=77, difficulty="medium")
    assert env.state is not None

    deal_id = obs.open_deal_ids[0]
    plan = {
        "deal_decisions": {
            deal_id: {"decision": "accept", "price": 94.0, "payment_days": 45}
        },
        "advance_periods": 2,
    }
    sim_obs = env.step(_plan_action(plan, horizon=2))

    assert sim_obs.simulation_projection is not None
    assert sim_obs.projected_balances is not None
    assert sim_obs.projected_defaults is not None
    assert sim_obs.projected_penalties is not None
    # Flat fields must mirror the nested CashflowProjection
    assert sim_obs.projected_balances == sim_obs.simulation_projection.period_balances
    assert sim_obs.projected_defaults == sim_obs.simulation_projection.period_defaults
    assert sim_obs.projected_penalties == sim_obs.simulation_projection.period_penalties


def test_flat_projection_fields_none_after_advance_period_without_prior_simulation() -> None:
    env = SMELiquidityEnvironment(total_periods=3)
    env.reset(seed=31, difficulty="easy")

    adv_obs = env.step(_advance_action())

    # No simulate_plan was called before advance, so flat fields are None
    assert adv_obs.projected_balances is None
    assert adv_obs.projected_defaults is None
    assert adv_obs.projected_penalties is None


# ---------------------------------------------------------------------------
# SMELiquidityEnvironment: configurable total_periods
# ---------------------------------------------------------------------------


def test_configurable_total_periods_flows_through_to_observation_and_world() -> None:
    for n in (1, 3, 6, 12):
        env = SMELiquidityEnvironment(total_periods=n)
        obs = env.reset(seed=1)
        assert obs.total_periods == n, f"Expected total_periods={n}, got {obs.total_periods}"
        assert env.state is not None
        assert env.state.world_state.total_periods == n
        assert env.state.world_state.current_period == 0
