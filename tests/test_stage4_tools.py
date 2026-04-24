"""Stage 4 deterministic tool-using liquidity environment tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference

from server.sme_environment import SMELiquidityEnvironment, SMENegotiatorEnvironment
from sme_negotiator_env.llm_action_parser import parse_llm_text_to_negotiation_action
from sme_negotiator_env.models import NegotiationAction
from sme_negotiator_env.simulation import simulate_cashflow
from sme_negotiator_env.tools import check_compliance, query_treds, run_cashflow_sim


def _tool_action(
    tool_name: str,
    *,
    deal_id: str | None = None,
    tool_args: dict[str, object] | None = None,
) -> NegotiationAction:
    payload = dict(tool_args or {})
    if deal_id is not None and "deal_id" not in payload:
        payload["deal_id"] = deal_id
    return NegotiationAction(
        action_type="tool",
        tool_name=tool_name,
        tool_args=payload,
    )


def _proposal_action(deal_id: str) -> NegotiationAction:
    return NegotiationAction(
        action_type="propose",
        deal_id=deal_id,
        price=95.0,
        payment_days=45,
        use_treds=True,
    )


def test_tool_action_requires_tool_name() -> None:
    with pytest.raises(ValidationError):
        NegotiationAction(action_type="tool")


def test_non_tool_action_ignores_missing_tool_fields() -> None:
    action = NegotiationAction(action_type="propose", price=90.0, payment_days=45, use_treds=False)
    assert action.action_type == "propose"
    assert action.tool_name is None


def test_pure_tools_are_deterministic_and_wrapper_matches_simulator() -> None:
    env = SMELiquidityEnvironment(total_periods=3)
    observation = env.reset(seed=22, difficulty="medium")
    assert env.state is not None

    deal_id = observation.open_deal_ids[0]
    world_state = env.state.world_state
    quote_a = query_treds(world_state, deal_id)
    quote_b = query_treds(world_state, deal_id)
    assert quote_a == quote_b

    rates = [option["annual_discount_rate"] for option in quote_a["quote_options"]]
    assert rates == sorted(rates)
    assert all(0.05 <= rate <= 0.60 for rate in rates)

    bad_contract = env.state.current_negotiations[deal_id].model_copy(
        update={
            "buyer_days": 60,
            "agreed_terms": 60,
            "dynamic_discounting_agreed": True,
            "agreed_dynamic_discount_annual": 0.5,
            "late_payment_penalty_agreed": False,
        }
    )
    compliance = check_compliance(world_state, deal_id, negotiation_state=bad_contract)
    assert compliance["is_compliant"] is False
    assert "payment_days_exceed_legal_max" in compliance["violated_clauses"]
    assert "dynamic_discount_rate_exceeds_policy_cap" in compliance["violated_clauses"]

    plan = {
        "deal_decisions": {
            deal_id: {
                "decision": "accept",
                "price": 95.0,
                "payment_days": 45,
                "use_treds": True,
            }
        },
        "financing": {deal_id: True},
        "advance_periods": 2,
    }
    wrapped = run_cashflow_sim(world_state, plan, horizon=2)
    projected = simulate_cashflow(world_state, plan, horizon=2)
    assert wrapped["period_balances"] == projected.period_balances
    assert wrapped["period_defaults"] == projected.period_defaults
    assert wrapped["period_penalties"] == projected.period_penalties


def test_tool_action_is_non_mutating_and_populates_liquidity_observation() -> None:
    env = SMELiquidityEnvironment(total_periods=4)
    observation = env.reset(seed=41, difficulty="medium")
    assert env.state is not None

    deal_id = observation.open_deal_ids[0]
    before_cash = env.state.world_state.smes[0].cash_balance
    before_deals = [deal.model_dump() for deal in env.state.world_state.deals]
    before_trajectory_lengths = {key: len(value) for key, value in env.state.deal_trajectories.items()}
    before_resolved = list(env.state.resolved_deal_ids)

    tool_observation = env.step(
        _tool_action(
            "QUERY_TREDS",
            deal_id=deal_id,
            tool_args={"invoice_id": deal_id},
        )
    )

    assert tool_observation.done is False
    assert tool_observation.reward == 0.0
    assert tool_observation.last_tool_name == "QUERY_TREDS"
    assert tool_observation.last_tool_args == {"invoice_id": deal_id, "deal_id": deal_id}
    assert tool_observation.last_tool_result is not None
    assert tool_observation.history[-1].event_type == "tool_call"
    assert env.state is not None
    assert env.state.world_state.smes[0].cash_balance == before_cash
    assert [deal.model_dump() for deal in env.state.world_state.deals] == before_deals
    assert {key: len(value) for key, value in env.state.deal_trajectories.items()} == before_trajectory_lengths
    assert env.state.resolved_deal_ids == before_resolved


def test_run_cashflow_sim_tool_matches_simulate_plan_projection() -> None:
    env_tool = SMELiquidityEnvironment(total_periods=3)
    env_plan = SMELiquidityEnvironment(total_periods=3)
    obs_tool = env_tool.reset(seed=77, difficulty="hard")
    obs_plan = env_plan.reset(seed=77, difficulty="hard")
    assert env_tool.state is not None and env_plan.state is not None

    deal_id = obs_tool.open_deal_ids[0]
    plan = {
        "deal_decisions": {
            deal_id: {
                "decision": "accept",
                "price": 94.0,
                "payment_days": 45,
                "use_treds": True,
            }
        },
        "financing": {deal_id: True},
        "advance_periods": 2,
    }

    tool_obs = env_tool.step(
        _tool_action(
            "RUN_CASHFLOW_SIM",
            deal_id=deal_id,
            tool_args={"plan": plan, "horizon": 2},
        )
    )
    plan_obs = env_plan.step(
        NegotiationAction(
            action_type="simulate_plan",
            deal_id=deal_id,
            simulation_plan=plan,
            simulation_horizon=2,
        )
    )

    assert tool_obs.simulation_projection is not None
    assert plan_obs.simulation_projection is not None
    assert tool_obs.simulation_projection.model_dump() == plan_obs.simulation_projection.model_dump()


def test_query_treds_bonus_is_tiny_and_positive_for_improving_next_step() -> None:
    env_plain = SMELiquidityEnvironment(total_periods=3)
    env_tool = SMELiquidityEnvironment(total_periods=3)
    base_obs = env_plain.reset(seed=88, difficulty="medium")
    tool_obs = env_tool.reset(seed=88, difficulty="medium")

    plain_reward = env_plain.step(_proposal_action(base_obs.open_deal_ids[0]))
    env_tool.step(_tool_action("QUERY_TREDS", deal_id=tool_obs.open_deal_ids[0], tool_args={"invoice_id": tool_obs.open_deal_ids[0]}))
    improved_reward = env_tool.step(_proposal_action(tool_obs.open_deal_ids[0]))

    assert improved_reward.metadata["tool_bonus_applied"] == pytest.approx(0.01)
    assert improved_reward.reward == pytest.approx(plain_reward.reward + 0.01)


def test_compliance_bonus_is_smaller_and_triggered_by_legal_progress() -> None:
    env_plain = SMELiquidityEnvironment(total_periods=3)
    env_tool = SMELiquidityEnvironment(total_periods=3)
    base_obs = env_plain.reset(seed=99, difficulty="medium")
    tool_obs = env_tool.reset(seed=99, difficulty="medium")

    plain_reward = env_plain.step(_proposal_action(base_obs.open_deal_ids[0]))
    env_tool.step(_tool_action("CHECK_COMPLIANCE", deal_id=tool_obs.open_deal_ids[0], tool_args={"contract_id": tool_obs.open_deal_ids[0]}))
    improved_reward = env_tool.step(_proposal_action(tool_obs.open_deal_ids[0]))

    assert improved_reward.metadata["tool_bonus_applied"] == pytest.approx(0.005)
    assert improved_reward.reward == pytest.approx(plain_reward.reward + 0.005)


def test_duplicate_tool_calls_only_apply_bounded_spam_penalty() -> None:
    env_single = SMELiquidityEnvironment(total_periods=3)
    env_spam = SMELiquidityEnvironment(total_periods=3)
    single_obs = env_single.reset(seed=123, difficulty="medium")
    spam_obs = env_spam.reset(seed=123, difficulty="medium")
    deal_id = single_obs.open_deal_ids[0]

    env_single.step(_tool_action("QUERY_TREDS", deal_id=deal_id, tool_args={"invoice_id": deal_id}))
    single_reward = env_single.step(_proposal_action(deal_id))

    env_spam.step(_tool_action("QUERY_TREDS", deal_id=deal_id, tool_args={"invoice_id": deal_id}))
    spam_flag_obs = env_spam.step(_tool_action("QUERY_TREDS", deal_id=deal_id, tool_args={"invoice_id": deal_id}))
    spam_reward = env_spam.step(_proposal_action(deal_id))

    assert spam_flag_obs.metadata["pending_tool_bonus"] == pytest.approx(-0.005)
    assert spam_flag_obs.metadata["tool_spam_flag"] is True
    assert spam_reward.metadata["tool_bonus_applied"] == pytest.approx(0.005)
    assert spam_reward.reward == pytest.approx(single_reward.reward - 0.005)


def test_legacy_environment_rejects_tool_action_deterministically() -> None:
    env = SMENegotiatorEnvironment()
    env.reset(seed=7, difficulty="easy")

    observation = env.step(
        NegotiationAction(
            action_type="tool",
            tool_name="QUERY_TREDS",
            tool_args={"invoice_id": "deal_x"},
        )
    )

    assert observation.done is True
    assert observation.reward > 0.0
    assert observation.metadata["termination_reason"] == "unsupported_action_type"


def test_parser_and_inference_accept_structured_tool_actions() -> None:
    env = SMELiquidityEnvironment(total_periods=2)
    observation = env.reset(seed=55, difficulty="medium")
    raw_json = '{"action_type":"tool","tool_name":"QUERY_TREDS","tool_args":{"deal_id":"%s"}}' % observation.open_deal_ids[0]

    parsed = parse_llm_text_to_negotiation_action(raw_json, observation, allow_json=True)
    coerced = inference._to_model_action(
        {
            "action_type": "tool",
            "tool_name": "CHECK_COMPLIANCE",
            "tool_args": {"contract_id": observation.open_deal_ids[0]},
        },
        observation,
    )

    assert parsed.action_type == "tool"
    assert parsed.tool_name == "QUERY_TREDS"
    assert parsed.tool_args == {"deal_id": observation.open_deal_ids[0]}
    assert coerced.action_type == "tool"
    assert coerced.tool_name == "CHECK_COMPLIANCE"
    assert coerced.tool_args == {"contract_id": observation.open_deal_ids[0]}
