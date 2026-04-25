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


def test_safe_liquidity_fallback_uses_real_advanced_actions() -> None:
    bridge = inference.InProcessLiquidityBridge()
    reset_result = asyncio.run(
        bridge.reset(seed=1000, difficulty="hard", task_name="liquidity-correlation-hard")
    )

    fallback_action = inference._safe_liquidity_fallback_action(reset_result.observation)

    assert fallback_action.action_type in {"tool", "accept", "advance_period"}
