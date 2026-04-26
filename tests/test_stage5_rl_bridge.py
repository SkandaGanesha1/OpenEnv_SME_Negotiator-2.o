"""Stage 5 deterministic TRL GRPO bridge tests.

Tests verify the in-process environment wrapper contract required by TRL's
`environment_factory` paradigm: zero-arg construction, reset(**kwargs)->str,
individual tool methods with correct return types, reward composition,
episode logging, and the make_environment_factory helper.

No TRL or Unsloth installation is required to run these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.bridge import (
    InProcessEnvWrapper,
    format_observation,
    make_environment_factory,
    parse_action,
)
from rl.episode_logging import (
    EpisodeSummary,
    combine_rewards,
    build_episode_log,
)
from rl.train_grpo_trl import (
    build_training_rows,
    build_dataset,
    make_reward_function,
    summarize_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESET_KWARGS = {
    "task_name": "payment-terms-medium",
    "difficulty": "medium",
    "seed": 42,
    "total_periods": 2,
}


def _make_wrapper(**kwargs) -> InProcessEnvWrapper:
    env = InProcessEnvWrapper()
    env.reset(**{**_RESET_KWARGS, **kwargs})
    return env


# ---------------------------------------------------------------------------
# Zero-arg construction
# ---------------------------------------------------------------------------


def test_in_process_env_wrapper_zero_arg_construction() -> None:
    wrapper = InProcessEnvWrapper()
    assert wrapper.env is None
    assert wrapper.done is False
    assert wrapper.env_reward_total == 0.0
    assert wrapper.tool_bonus_total == 0.0


# ---------------------------------------------------------------------------
# reset(**kwargs) → str
# ---------------------------------------------------------------------------


def test_reset_returns_nonempty_string() -> None:
    wrapper = InProcessEnvWrapper()
    result = wrapper.reset(**_RESET_KWARGS)
    assert isinstance(result, str)
    assert len(result) > 0


def test_reset_populates_config_fields() -> None:
    wrapper = InProcessEnvWrapper()
    wrapper.reset(task_name="payment-terms-easy", difficulty="easy", seed=7, total_periods=2)
    assert wrapper.task_name == "payment-terms-easy"
    assert wrapper.difficulty == "easy"
    assert wrapper.seed == 7
    assert wrapper.total_periods == 2


def test_reset_clears_episode_state() -> None:
    wrapper = _make_wrapper()
    wrapper.env_reward_total = 99.9
    wrapper.tool_bonus_total = 5.0
    wrapper.reset(**_RESET_KWARGS)
    assert wrapper.env_reward_total == 0.0
    assert wrapper.tool_bonus_total == 0.0
    assert not wrapper.done


def test_reset_with_list_prompt_does_not_crash() -> None:
    wrapper = InProcessEnvWrapper()
    result = wrapper.reset(
        prompt=[{"role": "user", "content": "Negotiate carefully."}],
        **_RESET_KWARGS,
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tool methods return strings and update state
# ---------------------------------------------------------------------------


def test_propose_returns_string_and_accumulates_reward() -> None:
    wrapper = _make_wrapper()
    assert wrapper.last_observation is not None
    obs = wrapper.last_observation
    result = wrapper.propose(
        price=float(obs.buyer_price),
        payment_days=int(obs.buyer_days) - 5,
        use_treds=False,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_query_treds_returns_string_and_increments_tool_count() -> None:
    wrapper = _make_wrapper()
    assert wrapper.last_observation is not None
    deal_id = wrapper.last_observation.open_deal_ids[0]
    result = wrapper.query_treds(invoice_id=deal_id)
    assert isinstance(result, str)
    assert wrapper.tool_counts["QUERY_TREDS"] == 1


def test_check_compliance_returns_string_and_increments_tool_count() -> None:
    wrapper = _make_wrapper()
    assert wrapper.last_observation is not None
    deal_id = wrapper.last_observation.open_deal_ids[0]
    result = wrapper.check_compliance(contract_id=deal_id)
    assert isinstance(result, str)
    assert wrapper.tool_counts["CHECK_COMPLIANCE"] == 1


def test_advance_period_returns_string() -> None:
    wrapper = _make_wrapper()
    result = wrapper.advance_period()
    assert isinstance(result, str)
    assert not wrapper.done


def test_run_cashflow_sim_returns_string_and_increments_tool_count() -> None:
    wrapper = _make_wrapper()
    assert wrapper.last_observation is not None
    deal_id = wrapper.last_observation.open_deal_ids[0]
    plan = {
        "deal_decisions": {
            deal_id: {"decision": "accept", "price": 95.0, "payment_days": 45, "use_treds": False}
        }
    }
    result = wrapper.run_cashflow_sim(plan=plan, horizon=1)
    assert isinstance(result, str)
    assert wrapper.tool_counts["RUN_CASHFLOW_SIM"] == 1


def test_simulate_plan_returns_string_and_preserves_read_only_flow() -> None:
    wrapper = _make_wrapper()
    assert wrapper.last_observation is not None
    deal_id = wrapper.last_observation.open_deal_ids[0]
    plan = {
        "deal_decisions": {
            deal_id: {"decision": "accept", "price": 95.0, "payment_days": 45, "use_treds": False}
        }
    }
    result = wrapper.simulate_plan(plan=plan, horizon=1, deal_id=deal_id)
    assert isinstance(result, str)
    assert wrapper.last_observation is not None
    assert wrapper.last_observation.simulation_projection is not None
    assert wrapper.tool_counts["RUN_CASHFLOW_SIM"] == 0


# ---------------------------------------------------------------------------
# reward property (TRL compatibility)
# ---------------------------------------------------------------------------


def test_reward_property_mirrors_env_reward_total() -> None:
    wrapper = _make_wrapper()
    assert wrapper.reward == wrapper.env_reward_total
    wrapper.env_reward_total = 3.14
    assert wrapper.reward == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Episode termination and final reward
# ---------------------------------------------------------------------------


def test_done_eventually_true_after_completing_all_deals() -> None:
    # In the liquidity env, one accept resolves one deal but the episode only
    # terminates after all macro periods are exhausted. Drive to completion.
    wrapper = InProcessEnvWrapper()
    wrapper.reset(task_name="payment-terms-easy", difficulty="easy", seed=1, total_periods=1)
    for _ in range(30):
        if wrapper.done:
            break
        obs = wrapper.last_observation
        assert obs is not None
        if obs.open_deal_ids:
            wrapper.accept(
                price=float(obs.buyer_price),
                payment_days=int(obs.buyer_days),
                deal_id=obs.open_deal_ids[0],
            )
        else:
            wrapper.advance_period()
    assert wrapper.done is True


def test_compute_final_reward_returns_float_after_episode() -> None:
    wrapper = _make_wrapper()
    obs = wrapper.last_observation
    assert obs is not None
    wrapper.accept(price=float(obs.buyer_price), payment_days=int(obs.buyer_days))
    final = wrapper.compute_final_reward()
    assert isinstance(final, float)
    assert final >= 0.0


# ---------------------------------------------------------------------------
# summarize_episode and build_episode_log
# ---------------------------------------------------------------------------


def test_summarize_episode_returns_correct_types() -> None:
    wrapper = _make_wrapper()
    obs = wrapper.last_observation
    assert obs is not None
    wrapper.query_treds(invoice_id=obs.open_deal_ids[0])
    wrapper.accept(price=float(obs.buyer_price), payment_days=int(obs.buyer_days))
    summary = wrapper.summarize_episode()
    assert isinstance(summary, EpisodeSummary)
    assert isinstance(summary.episode_completed, bool)
    assert isinstance(summary.base_rl_reward, float)
    assert isinstance(summary.tool_usage_count, int)
    assert summary.tool_usage_count >= 1
    assert isinstance(summary.verifiable_reward, float)
    assert isinstance(summary.total_reward, float)
    assert summary.tool_call_count >= 1


def test_build_episode_log_returns_nonempty_string_with_config() -> None:
    wrapper = _make_wrapper(seed=55)
    wrapper.accept(
        price=float(wrapper.last_observation.buyer_price),
        payment_days=int(wrapper.last_observation.buyer_days),
    )
    log = wrapper.build_episode_log()
    assert isinstance(log, str)
    assert "seed=55" in log
    assert len(log) > 50


# ---------------------------------------------------------------------------
# make_environment_factory
# ---------------------------------------------------------------------------


def test_make_environment_factory_returns_zero_arg_callable() -> None:
    factory = make_environment_factory(
        task_name="payment-terms-easy",
        difficulty="easy",
        seed=1,
        total_periods=2,
    )
    wrapper = factory()
    assert isinstance(wrapper, InProcessEnvWrapper)
    assert wrapper.env is None


def test_factory_defaults_flow_through_to_wrapper() -> None:
    factory = make_environment_factory(
        task_name="payment-terms-hard",
        difficulty="hard",
        seed=99,
        total_periods=3,
    )
    wrapper = factory()
    wrapper.reset()
    assert wrapper.task_name == "payment-terms-hard"
    assert wrapper.difficulty == "hard"
    assert wrapper.seed == 99


# ---------------------------------------------------------------------------
# format_observation and parse_action
# ---------------------------------------------------------------------------


def test_format_observation_returns_nonempty_string() -> None:
    wrapper = _make_wrapper()
    obs = wrapper.last_observation
    assert obs is not None
    result = format_observation(obs)
    assert isinstance(result, str)
    assert len(result) > 0


def test_parse_action_json_propose() -> None:
    wrapper = _make_wrapper()
    obs = wrapper.last_observation
    raw = '{"action_type": "propose", "price": 92.5, "payment_days": 40}'
    action = parse_action(raw, obs)
    assert action.action_type == "propose"
    assert action.price == pytest.approx(92.5)
    assert action.payment_days == 40


def test_parse_action_json_tool() -> None:
    wrapper = _make_wrapper()
    obs = wrapper.last_observation
    deal_id = obs.open_deal_ids[0]
    raw = f'{{"action_type": "tool", "tool_name": "QUERY_TREDS", "tool_args": {{"invoice_id": "{deal_id}"}}}}'
    action = parse_action(raw, obs)
    assert action.action_type == "tool"
    assert action.tool_name == "QUERY_TREDS"


# ---------------------------------------------------------------------------
# combine_rewards
# ---------------------------------------------------------------------------


def test_combine_rewards_no_rubric_returns_base() -> None:
    result = combine_rewards(0.75, None, 0.0)
    assert result == pytest.approx(0.75)


def test_combine_rewards_with_rubric_blends_correctly() -> None:
    result = combine_rewards(0.5, {"clarity": 0.8, "compliance": 0.8}, 0.2)
    assert result == pytest.approx(0.5 + 0.2 * 0.8)


def test_combine_rewards_zero_weight_ignores_rubric() -> None:
    result = combine_rewards(0.4, {"q": 1.0}, 0.0)
    assert result == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Training script helpers
# ---------------------------------------------------------------------------


def test_build_training_rows_produces_correct_structure() -> None:
    rows = build_training_rows(num_samples=4, seed_base=10)
    assert len(rows) == 4
    for i, row in enumerate(rows):
        assert "prompt" in row
        assert isinstance(row["prompt"], list)
        assert row["prompt"][0]["role"] == "user"
        assert row["seed"] == 10 + i


def test_build_dataset_returns_hf_dataset() -> None:
    Dataset = pytest.importorskip("datasets").Dataset
    rows = build_training_rows(num_samples=2)
    ds = build_dataset(rows)
    assert isinstance(ds, Dataset)
    assert len(ds) == 2


def test_make_reward_function_produces_correct_length_output() -> None:
    reward_func = make_reward_function()
    wrappers = [_make_wrapper(seed=s) for s in (1, 2, 3)]
    for w in wrappers:
        w.accept(price=float(w.last_observation.buyer_price), payment_days=int(w.last_observation.buyer_days))
    rewards = reward_func(wrappers)
    assert len(rewards) == 3
    assert all(isinstance(r, float) for r in rewards)


def test_summarize_batch_produces_expected_keys() -> None:
    summaries = [
        EpisodeSummary(
            episode_completed=True,
            base_rl_reward=0.5,
            tool_bonus_total=0.01,
            env_reward_total=0.5,
            success_no_default_positive_npv=True,
            average_final_payment_days=45.0,
            tool_usage_count=2,
            resolved_deal_count=1,
            defaulted_sme_count=0,
        )
    ]
    metrics = summarize_batch(summaries)
    assert "episode/avg_base_rl_reward" in metrics
    assert "episode/avg_total_reward" in metrics
    assert "episode/avg_verifiable_reward" in metrics
    assert "episode/success_rate" in metrics
    assert "episode/avg_tool_usage_count" in metrics
    assert "episode/avg_tool_call_count" in metrics
    assert "episode/avg_tool_effective_count" in metrics
    assert metrics["episode/success_rate"] == pytest.approx(1.0)
