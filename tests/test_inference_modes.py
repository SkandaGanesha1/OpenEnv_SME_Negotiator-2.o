"""Fast tests for inference mode selection and liquidity visibility helpers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
    assert inference._inference_env_mode() == "liquidity"
    assert inference._inference_reward_mode() == "legacy+shadow_rlvr"

    monkeypatch.setenv("INFERENCE_ENV_MODE", "LIQUIDITY")
    monkeypatch.setenv("INFERENCE_REWARD_MODE", "legacy+full_debug")
    assert inference._inference_env_mode() == "liquidity"
    assert inference._inference_reward_mode() == "legacy+full_debug"


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


def test_ascii_sparkline_uses_ascii_only() -> None:
    sparkline = inference._ascii_sparkline([0.0, 0.5, 1.0])
    assert sparkline.isascii()


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
    assert metrics["timeout_or_stepcap_rate"] == 0.5
