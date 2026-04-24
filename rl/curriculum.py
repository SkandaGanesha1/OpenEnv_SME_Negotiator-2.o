"""Adaptive curriculum helpers for Stage 6 training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    """Deterministic environment-complexity configuration."""

    total_periods: int
    buyer_variance: float
    financier_variance: float


DEFAULT_CURRICULUM_LEVELS: list[DifficultyConfig] = [
    DifficultyConfig(total_periods=1, buyer_variance=0.0, financier_variance=0.0),
    DifficultyConfig(total_periods=2, buyer_variance=0.1, financier_variance=0.05),
    DifficultyConfig(total_periods=4, buyer_variance=0.2, financier_variance=0.1),
    DifficultyConfig(total_periods=6, buyer_variance=0.3, financier_variance=0.15),
]


class CurriculumManager:
    """Moving-window curriculum scheduler with a promotion cooldown."""

    def __init__(
        self,
        levels: list[DifficultyConfig] | None = None,
        *,
        window_size: int = 100,
        reward_threshold: float = 0.6,
        max_default_rate: float = 0.2,
        cooldown_windows: int = 1,
    ) -> None:
        self.levels = list(levels or DEFAULT_CURRICULUM_LEVELS)
        self.current_level_idx = 0
        self.window_size = max(1, int(window_size))
        self.reward_threshold = float(reward_threshold)
        self.max_default_rate = float(max_default_rate)
        self.cooldown_windows = max(0, int(cooldown_windows))
        self.recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self.recent_failures: deque[bool] = deque(maxlen=self.window_size)
        self._cooldown_remaining = 0

    def record_episode(self, reward: float, defaulted: bool) -> None:
        """Append one completed episode outcome to the moving window."""
        self.recent_rewards.append(float(reward))
        self.recent_failures.append(bool(defaulted))

    def current_config(self) -> DifficultyConfig:
        """Return the active difficulty level."""
        return self.levels[self.current_level_idx]

    def current_level(self) -> int:
        """Return the active level index."""
        return int(self.current_level_idx)

    def metrics(self) -> dict[str, float]:
        """Expose current moving-window summary metrics."""
        if not self.recent_rewards:
            return {"avg_reward": 0.0, "default_rate": 0.0}
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        default_rate = sum(1.0 for value in self.recent_failures if value) / len(self.recent_failures)
        return {"avg_reward": avg_reward, "default_rate": default_rate}

    def maybe_advance_level(self) -> bool:
        """Promote the curriculum level when reward and failure thresholds are met."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False
        if len(self.recent_rewards) < self.window_size:
            return False
        if self.current_level_idx >= len(self.levels) - 1:
            return False

        metrics = self.metrics()
        if metrics["avg_reward"] <= self.reward_threshold:
            return False
        if metrics["default_rate"] >= self.max_default_rate:
            return False

        self.current_level_idx += 1
        self.recent_rewards.clear()
        self.recent_failures.clear()
        self._cooldown_remaining = self.cooldown_windows
        return True
