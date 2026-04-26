"""Stage 5 bridge tests for the in-process liquidity training wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.bridge import format_observation, make_environment_factory, parse_action


def test_format_observation_includes_macro_fields_and_tool_summaries() -> None:
    wrapper_cls = make_environment_factory(total_periods=2)
    wrapper = wrapper_cls()
    wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1000, total_periods=2)
    assert wrapper.last_observation is not None

    deal_id = wrapper.last_observation.open_deal_ids[0]
    wrapper.query_treds(invoice_id=deal_id)
    assert wrapper.last_observation is not None

    formatted = format_observation(wrapper.last_observation)
    assert "Macro:" in formatted
    assert "LastTool:" in formatted
    assert "HistoryTail=" in formatted


def test_parse_action_handles_tool_json_and_strict_json_fallback() -> None:
    action = parse_action(
        '{"action_type":"TOOL","tool_name":"QUERY_TREDS","tool_args":{"deal_id":"deal_x"}}'
    )
    assert action.action_type == "tool"
    assert action.tool_name == "QUERY_TREDS"
    assert action.tool_args == {"deal_id": "deal_x"}

    fallback = parse_action("not valid json")
    assert fallback.action_type == "propose"


def test_make_environment_factory_returns_zero_arg_wrapper_and_reset_uses_row_fields() -> None:
    wrapper_cls = make_environment_factory()
    wrapper = wrapper_cls()

    text = wrapper.reset(
        prompt=[{"role": "user", "content": "Train me"}],
        task_name="liquidity-correlation-hard",
        difficulty="hard",
        seed=1007,
        total_periods=4,
    )

    assert isinstance(text, str)
    assert wrapper.task_name == "liquidity-correlation-hard"
    assert wrapper.difficulty == "hard"
    assert wrapper.seed == 1007
    assert wrapper.total_periods == 4
    assert wrapper.env is not None
    assert wrapper.env.state is not None
    assert wrapper.env.state.world_state.total_periods == 4


def test_public_tool_methods_return_strings_and_update_wrapper_state() -> None:
    wrapper_cls = make_environment_factory(total_periods=2)
    wrapper = wrapper_cls()
    wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1010, total_periods=2)
    assert wrapper.last_observation is not None

    deal_id = wrapper.last_observation.open_deal_ids[0]
    result = wrapper.query_treds(invoice_id=deal_id)

    assert isinstance(result, str)
    assert wrapper.last_observation is not None
    assert wrapper.last_observation.last_tool_name == "QUERY_TREDS"
    assert wrapper.tool_counts["QUERY_TREDS"] == 1


def test_post_terminal_tool_call_raises_value_error() -> None:
    wrapper_cls = make_environment_factory(total_periods=1)
    wrapper = wrapper_cls()
    wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1011, total_periods=1)
    wrapper.advance_period()

    with pytest.raises(ValueError, match="Episode already completed."):
        wrapper.advance_period()


def test_compute_final_reward_is_deterministic_for_fixed_seed() -> None:
    wrapper_cls = make_environment_factory(total_periods=1)
    wrapper_a = wrapper_cls()
    wrapper_b = wrapper_cls()

    wrapper_a.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1020, total_periods=1)
    wrapper_b.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1020, total_periods=1)

    assert wrapper_a.compute_final_reward() == wrapper_b.compute_final_reward()


def test_bridge_smoke_episode_runs_in_process_without_model_download() -> None:
    wrapper_cls = make_environment_factory(total_periods=1)
    wrapper = wrapper_cls()
    first_prompt = wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1030, total_periods=1)
    assert isinstance(first_prompt, str)
    assert wrapper.last_observation is not None

    deal_id = wrapper.last_observation.open_deal_ids[0]
    tool_prompt = wrapper.run_cashflow_sim(plan={"deal_decisions": {}, "financing": {}}, horizon=1, deal_id=deal_id)
    final_prompt = wrapper.advance_period()

    assert isinstance(tool_prompt, str)
    assert isinstance(final_prompt, str)
    assert wrapper.done is True
