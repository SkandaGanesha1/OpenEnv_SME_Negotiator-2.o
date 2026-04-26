"""Fast tests for inference mode selection and liquidity visibility helpers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference
from sme_negotiator_env.models import NegotiationAction


def test_runtime_banner_matches_selected_mode() -> None:
    assert inference._runtime_banner("legacy") == "LEGACY_LIVE_BASELINE"
    assert inference._runtime_banner("liquidity") == "LIQUIDITY_IN_PROCESS_ADVANCED"


def test_inference_mode_defaults_and_normalization(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_ENV_MODE", raising=False)
    monkeypatch.delenv("INFERENCE_REWARD_MODE", raising=False)
    monkeypatch.delenv("INFERENCE_AGENT_MODE", raising=False)
    assert inference._inference_env_mode() == "liquidity"
    assert inference._inference_reward_mode() == "legacy+shadow_rlvr"
    assert inference._inference_agent_mode() == "router"

    monkeypatch.setenv("INFERENCE_ENV_MODE", "LIQUIDITY")
    monkeypatch.setenv("INFERENCE_REWARD_MODE", "legacy+full_debug")
    monkeypatch.setenv("INFERENCE_AGENT_MODE", "HEURISTIC")
    assert inference._inference_env_mode() == "liquidity"
    assert inference._inference_reward_mode() == "legacy+full_debug"
    assert inference._inference_agent_mode() == "heuristic"


def test_token_resolution_prefers_hf_then_api_key_then_openai() -> None:
    assert inference._resolve_router_token({"HF_TOKEN": "hf-token", "API_KEY": "api-key", "OPENAI_API_KEY": "openai-key"}) == "hf-token"
    assert inference._resolve_router_token({"HF_TOKEN": "", "API_KEY": "api-key", "OPENAI_API_KEY": "openai-key"}) == "api-key"
    assert inference._resolve_router_token({"OPENAI_API_KEY": "openai-key"}) == "openai-key"


def test_localhost_client_key_falls_back_to_not_needed() -> None:
    assert inference._resolve_openai_client_key("http://127.0.0.1:11434/v1", {}) == "not-needed"
    assert inference._resolve_openai_client_key("https://router.huggingface.co/v1", {}) == ""


def test_liquidity_task_mapping_uses_hard_default() -> None:
    assert inference._liquidity_task_for_difficulty("HARD") == "liquidity-correlation-hard"
    assert inference._liquidity_task_for_difficulty("MEDIUM") == "liquidity-stress-medium"


def test_liquidity_bridge_exposes_tool_and_macro_actions() -> None:
    async def _exercise_bridge() -> tuple[object, object, object]:
        bridge = inference.InProcessLiquidityBridge()
        reset_result = await bridge.reset(seed=1000, difficulty="hard", task_name="liquidity-correlation-hard")
        tool_result = await bridge.step(
            NegotiationAction(
                action_type="tool",
                deal_id=reset_result.observation.active_deal_id,
                tool_name="QUERY_TREDS",
                tool_args={
                    "invoice_id": reset_result.observation.active_deal_id,
                    "deal_id": reset_result.observation.active_deal_id,
                },
            )
        )
        period_result = await bridge.step(NegotiationAction(action_type="advance_period"))
        return reset_result.observation, tool_result.observation, period_result.observation

    reset_observation, tool_observation, period_observation = asyncio.run(_exercise_bridge())

    assert reset_observation.open_deal_ids
    assert tool_observation.last_tool_name == "QUERY_TREDS"
    assert tool_observation.metadata["reward_mode"] == "stage3_long_horizon"
    assert period_observation.current_period >= reset_observation.current_period


def test_liquidity_bridge_simulate_plan_uses_public_wrapper_method(monkeypatch) -> None:
    async def _exercise_bridge() -> bool:
        bridge = inference.InProcessLiquidityBridge()
        await bridge.reset(seed=1000, difficulty="hard", task_name="liquidity-correlation-hard")
        called = {"value": False}
        original = bridge._wrapper.simulate_plan

        def _wrapped(*args, **kwargs):
            called["value"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bridge._wrapper, "simulate_plan", _wrapped)
        await bridge.step(
            NegotiationAction(
                action_type="simulate_plan",
                deal_id=bridge._wrapper.last_observation.active_deal_id,
                simulation_plan={"advance_periods": 1},
                simulation_horizon=1,
            )
        )
        return bool(called["value"])

    assert asyncio.run(_exercise_bridge()) is True


def test_safe_liquidity_fallback_uses_real_advanced_actions() -> None:
    bridge = inference.InProcessLiquidityBridge()
    reset_result = asyncio.run(
        bridge.reset(seed=1000, difficulty="hard", task_name="liquidity-correlation-hard")
    )

    fallback_action = inference._safe_liquidity_fallback_action(reset_result.observation)

    assert fallback_action.action_type in {"tool", "accept", "advance_period"}


def test_terminal_reward_line_marks_source() -> None:
    line = inference._format_terminal_reward_line(
        verifiable_reward=0.625,
        final_score=0.625,
        success=True,
        source="shadow_rlvr",
    )
    assert "[TERMINAL_REWARD]" in line
    assert "source=shadow_rlvr" in line
    assert "verifiable=0.6250" in line
    assert "success=true" in line


def test_verifiable_reward_breakdown_line_formats_components() -> None:
    line = inference._format_verifiable_reward_breakdown_line(
        {
            "solvency": 1.0,
            "liquidity": 0.4,
            "npv": 0.25,
            "compliance": 0.8,
            "total": 0.62,
        }
    )
    assert line is not None
    assert "[VERIFIABLE_REWARD]" in line
    assert "solvency=1.0000" in line
    assert "total=0.6200" in line


def test_verifiable_reward_breakdown_line_marks_unavailable_without_fake_zeroes() -> None:
    line = inference._format_verifiable_reward_breakdown_line(None, canonical_total=0.7465)
    assert line == "[VERIFIABLE_REWARD] total=0.7465 breakdown=unavailable"


def test_ascii_sparkline_uses_ascii_only() -> None:
    sparkline = inference._ascii_sparkline([0.0, 0.5, 1.0])
    assert sparkline.isascii()


def test_auto_advance_liquidity_guardrail_detects_empty_period() -> None:
    assert inference._should_auto_advance_liquidity_period(
        {"open_deal_ids": [], "current_period": 1, "total_periods": 3}
    ) is True
    assert inference._should_auto_advance_liquidity_period(
        {"open_deal_ids": ["deal-1"], "current_period": 1, "total_periods": 3}
    ) is False
    assert inference._should_auto_advance_liquidity_period(
        {"open_deal_ids": [], "current_period": 3, "total_periods": 3}
    ) is False
    assert inference._should_auto_advance_liquidity_period(
        {"open_deal_ids": [], "current_period": 1, "total_periods": 3},
        done=True,
    ) is False


def test_run_liquidity_episode_auto_advances_without_calling_llm(monkeypatch) -> None:
    class _FakeLiquidityEnv:
        def __init__(self) -> None:
            self.actions: list[str] = []

        async def reset(self, **kwargs):
            return SimpleNamespace(
                done=False,
                reward=0.0,
                observation={
                    "open_deal_ids": [],
                    "resolved_deal_ids": [],
                    "active_deal_id": None,
                    "current_period": 0,
                    "total_periods": 2,
                    "reward": 0.0,
                    "metadata": {},
                },
            )

        async def step(self, action):
            self.actions.append(action.action_type)
            return SimpleNamespace(
                done=True,
                reward=0.4,
                observation={
                    "open_deal_ids": [],
                    "resolved_deal_ids": ["deal_p0_buyer_0_0"],
                    "active_deal_id": None,
                    "current_period": 2,
                    "total_periods": 2,
                    "reward": 0.4,
                    "done": True,
                    "metadata": {
                        "reward_breakdown": {
                            "solvency": 1.0,
                            "liquidity": 0.5,
                            "npv": 0.4,
                            "compliance": 1.0,
                            "total": 0.68,
                        }
                    },
                },
            )

        def summarize_episode(self):
            return SimpleNamespace(
                base_rl_reward=0.4,
                verifiable_reward=0.4,
                total_reward=0.4,
                tool_bonus_total=0.0,
                env_reward_total=0.4,
                success_no_default_positive_npv=True,
                average_final_payment_days=45.0,
                tool_usage_count=0,
                tool_call_count=0,
                tool_effective_count=0,
                duplicate_tool_count=0,
                invalid_action_count=0,
                stall_step_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
                terminated_by_step_cap=False,
            )

        def build_episode_log(self) -> str:
            return "fake-log"

    def _unexpected_llm_call(*args, **kwargs):
        raise AssertionError("LLM should not be called when no open deals remain.")

    monkeypatch.setattr(inference, "get_liquidity_agent_action", _unexpected_llm_call)
    result = asyncio.run(inference.run_liquidity_episode(_FakeLiquidityEnv(), "MEDIUM", 7))

    assert result["steps"] == 1
    assert result["success"] is True
    assert result["final_score"] > 0.0


def test_liquidity_summary_aggregation_adds_expected_keys() -> None:
    metrics = inference._aggregate_liquidity_episode_summaries(
        [
            {
                "episode_summary": {
                    "verifiable_reward": 0.5,
                    "tool_bonus_total": 0.01,
                    "tool_call_count": 2,
                    "tool_effective_count": 1,
                    "average_final_payment_days": 45.0,
                    "resolved_deal_count": 3,
                    "defaulted_sme_count": 0,
                    "terminated_by_step_cap": False,
                }
            },
            {
                "episode_summary": {
                    "verifiable_reward": 0.7,
                    "tool_bonus_total": 0.03,
                    "tool_call_count": 4,
                    "tool_effective_count": 3,
                    "average_final_payment_days": 35.0,
                    "resolved_deal_count": 4,
                    "defaulted_sme_count": 1,
                    "terminated_by_step_cap": True,
                }
            },
        ]
    )

    assert metrics["avg_verifiable_reward"] == 0.6
    assert metrics["avg_tool_bonus"] == 0.02
    assert metrics["avg_tool_call_count"] == 3.0
    assert metrics["avg_tool_effective_count"] == 2.0
    assert metrics["avg_final_payment_days"] == 40.0
    assert metrics["avg_resolved_deal_count"] == 3.5
    assert metrics["default_rate"] == 0.5
    assert metrics["timeout_or_stepcap_rate"] == 0.5


def test_run_liquidity_episode_heuristic_mode_skips_router(monkeypatch) -> None:
    class _FakeLiquidityEnv:
        def __init__(self) -> None:
            self.actions: list[object] = []

        async def reset(self, **kwargs):
            return SimpleNamespace(
                done=False,
                reward=0.0,
                observation={
                    "open_deal_ids": ["deal-1"],
                    "resolved_deal_ids": [],
                    "active_deal_id": "deal-1",
                    "current_period": 0,
                    "total_periods": 1,
                    "buyer_price": 96.0,
                    "buyer_days": 80,
                    "liquidity_threshold": 45,
                    "cost_threshold": 82.0,
                    "metadata": {},
                },
            )

        async def step(self, action):
            self.actions.append(action)
            return SimpleNamespace(
                done=True,
                reward=0.25,
                observation={
                    "open_deal_ids": [],
                    "resolved_deal_ids": ["deal-1"],
                    "active_deal_id": None,
                    "current_period": 1,
                    "total_periods": 1,
                    "buyer_price": 96.0,
                    "buyer_days": 80,
                    "liquidity_threshold": 45,
                    "cost_threshold": 82.0,
                    "done": True,
                    "metadata": {
                        "reward_breakdown": {
                            "solvency": 1.0,
                            "liquidity": 0.2,
                            "npv": 0.1,
                            "compliance": 1.0,
                            "total": 0.32,
                        },
                        "termination_reason": "episode_complete",
                        "defaulted_sme_count": 0,
                    },
                },
            )

        def summarize_episode(self):
            return SimpleNamespace(
                base_rl_reward=0.25,
                verifiable_reward=0.25,
                total_reward=0.25,
                tool_bonus_total=0.0,
                env_reward_total=0.25,
                success_no_default_positive_npv=True,
                average_final_payment_days=45.0,
                tool_usage_count=1,
                tool_call_count=1,
                tool_effective_count=1,
                duplicate_tool_count=0,
                invalid_action_count=0,
                stall_step_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
                terminated_by_step_cap=False,
            )

        def build_episode_log(self) -> str:
            return "fake-trained-log"

    def _unexpected_llm_call(*args, **kwargs):
        raise AssertionError("Router path should not run in heuristic mode.")

    monkeypatch.setenv("INFERENCE_AGENT_MODE", "heuristic")
    monkeypatch.setattr(inference, "get_liquidity_agent_action", _unexpected_llm_call)
    result = asyncio.run(inference.run_liquidity_episode(_FakeLiquidityEnv(), "HARD", 7))

    assert result["steps"] == 1
    assert result["termination_reason"] == "episode_complete"


def test_run_liquidity_episode_logs_period_and_terminal_fields(monkeypatch, capsys) -> None:
    class _FakeLiquidityEnv:
        async def reset(self, **kwargs):
            return SimpleNamespace(
                done=False,
                reward=0.0,
                observation={
                    "open_deal_ids": [],
                    "resolved_deal_ids": [],
                    "active_deal_id": None,
                    "current_period": 0,
                    "total_periods": 2,
                    "reward": 0.0,
                    "metadata": {},
                },
            )

        async def step(self, action):
            return SimpleNamespace(
                done=True,
                reward=0.4,
                observation={
                    "open_deal_ids": [],
                    "resolved_deal_ids": ["deal-1"],
                    "active_deal_id": None,
                    "current_period": 1,
                    "total_periods": 2,
                    "reward": 0.4,
                    "done": True,
                    "metadata": {
                        "termination_reason": "macro_horizon_end",
                        "defaulted_sme_count": 1,
                        "resolved_deal_count": 1,
                        "reward_breakdown": {
                            "solvency": 1.0,
                            "liquidity": 0.5,
                            "npv": 0.4,
                            "compliance": 1.0,
                            "total": 0.68,
                        },
                    },
                },
            )

        def summarize_episode(self):
            return SimpleNamespace(
                base_rl_reward=0.4,
                verifiable_reward=0.4,
                total_reward=0.4,
                tool_bonus_total=0.0,
                env_reward_total=0.4,
                success_no_default_positive_npv=False,
                average_final_payment_days=45.0,
                tool_usage_count=0,
                tool_call_count=0,
                tool_effective_count=0,
                duplicate_tool_count=0,
                invalid_action_count=0,
                stall_step_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=1,
                terminated_by_step_cap=False,
            )

        def build_episode_log(self) -> str:
            return "fake-log"

    asyncio.run(inference.run_liquidity_episode(_FakeLiquidityEnv(), "MEDIUM", 7))
    captured = capsys.readouterr().out

    assert "[PERIOD_SUMMARY]" in captured
    assert "step_reward_raw=0.4000" in captured
    assert "termination_reason=macro_horizon_end" in captured
    assert "defaulted_sme_count=1" in captured
    assert "breakdown=unavailable" in captured


def test_router_message_builder_bounds_history(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_HISTORY_MAX_TURNS", "2")
    monkeypatch.setenv("INFERENCE_HISTORY_MAX_CHARS", "120")
    monkeypatch.setenv("INFERENCE_HISTORY_SUMMARY_MAX_CHARS", "80")
    history = [
        {"role": "user", "content": "u1 " * 40},
        {"role": "assistant", "content": '{"action_type":"propose","price":91.0,"payment_days":60}'},
        {"role": "user", "content": "u2 " * 40},
        {"role": "assistant", "content": '{"action_type":"propose","price":89.0,"payment_days":55}'},
        {"role": "user", "content": "u3 " * 40},
        {"role": "assistant", "content": '{"action_type":"tool","tool_name":"QUERY_TREDS"}'},
    ]

    messages = inference._build_liquidity_router_messages(
        history=history,
        history_summary="older summary " * 20,
        task_name="liquidity-stress-medium",
        observation={"buyer_price": 95.0, "buyer_days": 75, "liquidity_threshold": 45},
    )

    assert messages[0]["role"] == "system"
    assert len(messages) <= 7
    assert messages[1]["content"].startswith("Rolling prior context summary:\n")
    assert len(messages[1]["content"].split("\n", 1)[1]) <= 80
