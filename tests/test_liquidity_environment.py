"""Tests for the Stage 3 long-horizon liquidity environment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.sme_environment import SMELiquidityEnvironment, SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation
from sme_negotiator_env.simulation import simulate_cashflow


def _observation_contract_keys() -> list[str]:
    return [
        "done",
        "reward",
        "metadata",
        "round_number",
        "max_rounds",
        "buyer_price",
        "buyer_days",
        "buyer_accepted",
        "negotiation_done",
        "cost_threshold",
        "liquidity_threshold",
        "volume",
        "difficulty",
        "price_score",
        "days_score",
        "treds_bonus",
        "step_reward",
        "message",
        "task_name",
        "sme_monthly_revenue",
        "working_capital_gap",
        "interest_rate_annual",
        "buyer_power_score",
        "secondary_buyer_power",
        "current_payment_terms_days",
        "sme_supplier_payment_days",
    ]


def _plan_action(plan: dict[str, object], *, horizon: int = 2, deal_id: str | None = None) -> NegotiationAction:
    return NegotiationAction(
        action_type="simulate_plan",
        price=0.0,
        payment_days=0,
        use_treds=False,
        deal_id=deal_id,
        simulation_plan=plan,
        simulation_horizon=horizon,
    )


def _advance_action() -> NegotiationAction:
    return NegotiationAction(
        action_type="advance_period",
        price=0.0,
        payment_days=0,
        use_treds=False,
    )


def test_negotiation_observation_contract_remains_unchanged() -> None:
    assert list(NegotiationObservation.model_fields) == _observation_contract_keys()


def test_liquidity_reset_exposes_macro_state_and_initial_deals() -> None:
    env = SMELiquidityEnvironment(total_periods=6)

    observation = env.reset(seed=42, difficulty="medium")

    assert observation.agent_type == "SME"
    assert observation.agent_id == "sme_0"
    assert observation.current_actor == "SME"
    assert observation.current_period == 0
    assert observation.total_periods == 6
    assert observation.episode_step == 0
    assert observation.simulation_projection is None
    assert env.state is not None
    assert env.state.world_state.current_period == 0
    assert env.state.world_state.total_periods == 6
    assert len(env.state.world_state.deals) == 2
    assert len(env.state.current_negotiations) == 2
    assert len(observation.open_deal_ids) == 2
    assert env.state.active_deal_id is not None
    assert len(env.state.trajectory) == 1


def test_liquidity_reset_honors_total_periods_parameter() -> None:
    short_env = SMELiquidityEnvironment(total_periods=1)
    long_env = SMELiquidityEnvironment(total_periods=6)

    short_obs = short_env.reset(seed=7, difficulty="easy")
    long_obs = long_env.reset(seed=7, difficulty="easy")

    assert short_obs.total_periods == 1
    assert long_obs.total_periods == 6
    assert short_env.state is not None and long_env.state is not None
    assert short_env.state.world_state.current_period == 0
    assert long_env.state.world_state.current_period == 0


def test_liquidity_simulate_plan_is_deterministic_and_non_mutating() -> None:
    env1 = SMELiquidityEnvironment(total_periods=6)
    env2 = SMELiquidityEnvironment(total_periods=6)

    obs1 = env1.reset(seed=77, difficulty="hard")
    obs2 = env2.reset(seed=77, difficulty="hard")
    assert env1.state is not None and env2.state is not None

    target_deal = obs1.open_deal_ids[0]
    before_cash = env1.state.world_state.smes[0].cash_balance
    before_period = env1.state.world_state.current_period
    before_deals = env1.state.world_state.deals.model_copy(deep=True) if hasattr(env1.state.world_state.deals, "model_copy") else None
    plan = {
        "deal_decisions": {
            target_deal: {
                "decision": "accept",
                "price": 94.0,
                "payment_days": 45,
                "use_treds": True,
            }
        },
        "financing": {target_deal: True},
        "advance_periods": 2,
    }

    sim1 = env1.step(_plan_action(plan, horizon=2, deal_id=target_deal))
    sim2 = env2.step(_plan_action(plan, horizon=2, deal_id=target_deal))

    assert sim1.reward == 0.0
    assert sim1.done is False
    assert sim1.simulation_projection is not None
    assert sim1.simulation_projection.model_dump() == sim2.simulation_projection.model_dump()
    assert env1.state is not None
    assert env1.state.world_state.current_period == before_period
    assert env1.state.world_state.smes[0].cash_balance == before_cash
    assert [deal.model_dump() for deal in env1.state.world_state.deals] == [deal.model_dump() for deal in env2.state.world_state.deals]


def test_simulate_cashflow_is_pure_for_fixed_world_state() -> None:
    env = SMELiquidityEnvironment(total_periods=3)
    obs = env.reset(seed=12, difficulty="medium")
    assert env.state is not None

    deal_id = obs.open_deal_ids[0]
    plan = {
        "deal_decisions": {
            deal_id: {
                "decision": "accept",
                "price": 96.0,
                "payment_days": 45,
                "use_treds": False,
            }
        },
        "financing": {deal_id: False},
        "advance_periods": 2,
    }

    first = simulate_cashflow(env.state.world_state, plan, horizon=2)
    second = simulate_cashflow(env.state.world_state, plan, horizon=2)

    assert first.model_dump() == second.model_dump()


def test_explicit_deal_id_routes_micro_step_to_requested_negotiation() -> None:
    env = SMELiquidityEnvironment(total_periods=6)
    observation = env.reset(seed=55, difficulty="medium")
    assert env.state is not None

    first_deal, second_deal = observation.open_deal_ids
    second_buyer_id = env.state.current_negotiations[second_deal].buyer_id

    stepped = env.step(
        NegotiationAction(
            action_type="propose",
            deal_id=second_deal,
            price=95.0,
            payment_days=50,
            use_treds=True,
        )
    )

    assert stepped.active_deal_id == second_deal
    assert env.state is not None
    assert env.state.active_deal_id == second_deal
    assert env.state.active_buyer_id == second_buyer_id
    assert len(env.state.deal_trajectories[second_deal]) == 2
    assert len(env.state.deal_trajectories[first_deal]) == 1
    assert stepped.metadata.get("legacy_inner_reward") is not None


def test_advance_period_spawns_new_purchase_orders_deterministically() -> None:
    env = SMELiquidityEnvironment(total_periods=2)
    env.reset(seed=31, difficulty="easy")
    assert env.state is not None

    advanced = env.step(_advance_action())

    assert advanced.done is False
    assert advanced.current_period == 1
    assert env.state is not None
    assert env.state.world_state.current_period == 1
    assert env.state.world_state.history[-1].period_index == 0
    assert len(env.state.world_state.deals) == 4
    assert len(advanced.open_deal_ids) >= 2


def test_final_macro_closeout_emits_terminal_reward_only_at_episode_end() -> None:
    env = SMELiquidityEnvironment(total_periods=1)
    observation = env.reset(seed=42, difficulty="medium")
    assert env.state is not None

    first_deal = observation.open_deal_ids[0]
    reject_obs = env.step(
        NegotiationAction(
            action_type="reject",
            deal_id=first_deal,
            price=95.0,
            payment_days=45,
            use_treds=False,
        )
    )

    assert reject_obs.done is False
    assert reject_obs.negotiation_done is True
    assert reject_obs.metadata.get("latest_verifiable_reward") is None

    terminal_obs = env.step(_advance_action())

    assert terminal_obs.done is True
    assert terminal_obs.metadata.get("latest_verifiable_reward") is not None
    assert terminal_obs.reward == terminal_obs.metadata.get("latest_verifiable_reward")
    assert terminal_obs.metadata.get("reward_mode") == "stage3_long_horizon"


def test_liquidity_step_cap_terminates_non_progress_tool_spam_deterministically() -> None:
    def _run(seed: int) -> tuple[int, object]:
        env = SMELiquidityEnvironment(total_periods=1)
        observation = env.reset(seed=seed, difficulty="medium")
        steps = 0
        while not observation.done and steps < 64:
            deal_id = observation.active_deal_id or observation.open_deal_ids[0]
            observation = env.step(
                NegotiationAction(
                    action_type="tool",
                    deal_id=deal_id,
                    tool_name="QUERY_TREDS",
                    tool_args={"invoice_id": deal_id, "deal_id": deal_id},
                )
            )
            steps += 1
        assert env.state is not None
        assert env.state.terminated_by_step_cap is True
        return steps, observation

    steps_a, final_a = _run(91)
    steps_b, final_b = _run(91)

    assert final_a.done is True
    assert final_a.metadata["terminated_by_step_cap"] is True
    assert steps_a == steps_b
    assert final_a.reward == final_b.reward


def test_simulation_projection_matches_manual_period_advance_for_balance_and_defaults() -> None:
    env = SMELiquidityEnvironment(total_periods=2)
    observation = env.reset(seed=21, difficulty="medium")
    assert env.state is not None

    plan = {
        "deal_decisions": {},
        "financing": {},
        "advance_periods": 1,
    }
    projected = simulate_cashflow(env.state.world_state, plan, horizon=1)

    advanced = env.step(_advance_action())

    assert advanced.current_period == 1
    assert env.state is not None
    snapshot = env.state.world_state.history[-1]
    assert projected.period_balances[0] == snapshot.total_cash_balance
    assert projected.period_defaults[0] == (snapshot.defaulted_sme_count > 0)
    assert projected.period_penalties[0] == snapshot.total_penalty_exposure


def test_legacy_environment_rejects_stage3_action_types_deterministically() -> None:
    env = SMENegotiatorEnvironment()
    env.reset(seed=42, difficulty="easy")

    result = env.step(
        NegotiationAction(
            action_type="simulate_plan",
            price=0.0,
            payment_days=0,
            use_treds=False,
            simulation_plan={},
            simulation_horizon=1,
        )
    )

    assert result.done is True
    assert result.metadata.get("termination_reason") == "unsupported_action_type"
    assert 0.0 < result.reward < 1.0


def test_sme_negotiator_environment_matches_frozen_stage0_trace() -> None:
    expected = {
        "easy": (97.11, 87, 0.1448, "partial_progress:improvement=0.471|days_baseline=0.200|price_baseline=0.975"),
        "medium": (99.01, 57, 0.1682, "partial_progress:improvement=0.601|days_baseline=0.400|price_baseline=0.975"),
        "hard": (95.2, 99, 0.1368, "partial_progress:improvement=0.427|days_baseline=0.133|price_baseline=0.972"),
    }

    for difficulty, seed in (("easy", 42), ("medium", 43), ("hard", 44)):
        env = SMENegotiatorEnvironment()
        obs0 = env.reset(seed=seed, difficulty=difficulty)
        obs1 = env.step(
            NegotiationAction(
                action_type="propose",
                price=round(max(obs0.cost_threshold + 1.0, obs0.buyer_price - 0.5), 2),
                payment_days=max(obs0.liquidity_threshold, obs0.buyer_days - 6),
                use_treds=bool(obs0.buyer_days > obs0.liquidity_threshold + 10),
            )
        )
        expected_price, expected_days, expected_reward, expected_branch = expected[difficulty]
        assert obs1.buyer_price == expected_price
        assert obs1.buyer_days == expected_days
        assert obs1.reward == expected_reward
        assert obs1.metadata.get("reward_branch") == expected_branch
