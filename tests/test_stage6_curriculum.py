"""Stage 6 curriculum and variance tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.curriculum import CurriculumManager
from server.environment import SMELiquidityEnvironment


def test_curriculum_does_not_promote_before_window_fills() -> None:
    manager = CurriculumManager(window_size=3, reward_threshold=0.6, max_default_rate=0.2)
    manager.record_episode(0.9, False)
    manager.record_episode(0.8, False)

    assert manager.maybe_advance_level() is False
    assert manager.current_level() == 0


def test_curriculum_promotes_clears_window_and_respects_cooldown() -> None:
    manager = CurriculumManager(window_size=3, reward_threshold=0.6, max_default_rate=0.2, cooldown_windows=1)
    for _ in range(3):
        manager.record_episode(0.9, False)

    assert manager.maybe_advance_level() is True
    assert manager.current_level() == 1
    assert list(manager.recent_rewards) == []
    assert list(manager.recent_failures) == []

    for _ in range(3):
        manager.record_episode(0.95, False)
    assert manager.maybe_advance_level() is False
    assert manager.current_level() == 1

    for _ in range(3):
        manager.record_episode(0.95, False)
    assert manager.maybe_advance_level() is True
    assert manager.current_level() == 2


def test_curriculum_does_not_promote_when_default_rate_is_too_high() -> None:
    manager = CurriculumManager(window_size=4, reward_threshold=0.6, max_default_rate=0.2)
    manager.record_episode(0.9, False)
    manager.record_episode(0.9, True)
    manager.record_episode(0.9, False)
    manager.record_episode(0.9, False)

    assert manager.maybe_advance_level() is False
    assert manager.current_level() == 0


def test_zero_variance_reproduces_default_liquidity_world_state() -> None:
    default_env = SMELiquidityEnvironment(total_periods=2)
    explicit_zero_env = SMELiquidityEnvironment(total_periods=2, buyer_variance=0.0, financier_variance=0.0)

    default_env.reset(seed=1300, difficulty="hard", task_name="liquidity-correlation-hard")
    explicit_zero_env.reset(seed=1300, difficulty="hard", task_name="liquidity-correlation-hard")

    assert default_env.state is not None
    assert explicit_zero_env.state is not None
    assert default_env.state.world_state.model_dump() == explicit_zero_env.state.world_state.model_dump()
