"""Stage 5 deterministic RL metric tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.bridge import make_environment_factory
from rl.episode_logging import combine_rewards
from sme_negotiator_env.graders import compute_npv_delta_vs_baseline, compute_total_sme_reward


def test_weighted_final_reward_matches_manual_aggregation() -> None:
    wrapper_cls = make_environment_factory(total_periods=2)
    wrapper = wrapper_cls()
    wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1040, total_periods=2)
    assert wrapper.env is not None
    assert wrapper.env.state is not None

    state = wrapper.env.state
    world_state = state.world_state
    deal_map = {deal.deal_id: deal for deal in world_state.deals}
    weighted = 0.0
    total_weight = 0.0
    for deal_id, trajectory in state.deal_trajectories.items():
        if not trajectory:
            continue
        deal = deal_map.get(deal_id)
        weight = float(deal.invoice_amount) if deal is not None and float(deal.invoice_amount) > 0.0 else 1.0
        weighted += weight * compute_total_sme_reward(
            world_state,
            trajectory,
            lambda_shaping=float(world_state.reward_lambda_shaping),
        )
        total_weight += weight
    manual_reward = round((weighted / total_weight) + wrapper.tool_bonus_total, 6)

    assert wrapper.compute_final_reward() == manual_reward


def test_compute_npv_delta_vs_baseline_is_deterministic() -> None:
    wrapper_cls = make_environment_factory(total_periods=2)
    wrapper = wrapper_cls()
    wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1041, total_periods=2)
    assert wrapper.env is not None
    assert wrapper.env.state is not None

    trajectory = next(iter(wrapper.env.state.deal_trajectories.values()))
    first = compute_npv_delta_vs_baseline(wrapper.env.state.world_state, trajectory)
    second = compute_npv_delta_vs_baseline(wrapper.env.state.world_state, trajectory)
    assert first == second


def test_success_flag_flips_for_default_vs_positive_npv(monkeypatch) -> None:
    wrapper_cls = make_environment_factory(total_periods=1)
    wrapper = wrapper_cls()
    wrapper.reset(task_name="liquidity-correlation-hard", difficulty="hard", seed=1042, total_periods=1)
    assert wrapper.env is not None
    assert wrapper.env.state is not None

    monkeypatch.setattr(wrapper, "_weighted_deal_metrics", lambda: (0.2, 1.0, [30.0]))
    summary = wrapper.summarize_episode()
    assert summary.success_no_default_positive_npv is True

    wrapper.env.state.world_state.smes[0].defaulted = True
    summary_after_default = wrapper.summarize_episode()
    assert summary_after_default.success_no_default_positive_npv is False


def test_combine_rewards_is_zero_overlay_when_disabled_and_additive_when_enabled() -> None:
    assert combine_rewards(0.4, None, 0.5) == pytest.approx(0.4)
    assert combine_rewards(0.4, {"a": 0.2, "b": 0.6}, 0.5) == pytest.approx(0.6)
