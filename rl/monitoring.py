"""Training monitoring utilities for the SME negotiator RL loop.

Components:
- RewardMonitorCallback: logs per-component reward columns and detects reward hacking
- GenerationSampler:     writes full episode transcripts to JSONL for human inspection
- SuccessRateTracker:    rolling window success and timeout rate
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from .bridge import get_episode_log, get_episode_summary

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:  # pragma: no cover - optional in lightweight utility contexts
    class _TrainerCallbackBase:  # type: ignore[too-many-ancestors]
        """Fallback base when transformers is unavailable."""

        pass


# ======================================================================= #
# SuccessRateTracker                                                        #
# ======================================================================= #

class SuccessRateTracker:
    """Rolling-window success and timeout rate tracker.

    Parameters
    ----------
    window_size:
        Number of recent episodes to keep in each rolling window.
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = int(window_size)
        self._successes: deque[bool] = deque(maxlen=self._window_size)
        self._timeouts: deque[bool] = deque(maxlen=self._window_size)

    def record(
        self,
        deal_reached: bool,
        agreed_days: int,
        threshold: int,
        *,
        timed_out: bool = False,
    ) -> None:
        """Record one episode outcome."""
        success = bool(deal_reached) and int(agreed_days) <= int(threshold)
        self._successes.append(success)
        self._timeouts.append(bool(timed_out))

    @property
    def success_rate(self) -> float:
        if not self._successes:
            return 0.0
        return round(mean(float(s) for s in self._successes), 4)

    @property
    def timeout_rate(self) -> float:
        if not self._timeouts:
            return 0.0
        return round(mean(float(t) for t in self._timeouts), 4)

    def reset(self) -> None:
        self._successes.clear()
        self._timeouts.clear()


# ======================================================================= #
# GenerationSampler                                                         #
# ======================================================================= #

class GenerationSampler:
    """Write full episode transcripts to JSONL for human inspection.

    Call ``sample_and_write`` periodically during training to capture
    actual model generations alongside their reward breakdowns.
    """

    def __init__(
        self,
        output_file: Path = Path("logs/sampled_generations.jsonl"),
        n_samples: int = 3,
    ) -> None:
        self._output_file = Path(output_file)
        self._n_samples = int(n_samples)

    def sample_and_write(
        self,
        environments: list[Any],
        step: int,
    ) -> None:
        """Write up to n_samples episode transcripts at the given training step."""
        if not environments:
            return
        self._output_file.parent.mkdir(parents=True, exist_ok=True)

        samples = environments[: self._n_samples]
        records = []
        for i, env in enumerate(samples):
            try:
                episode_log = get_episode_log(env)
                bd = None
                if hasattr(env, "reward_breakdown"):
                    try:
                        bd = env.reward_breakdown.to_dict()
                    except Exception:
                        bd = {"total": float(getattr(env, "reward", 0.0))}

                summary = {}
                try:
                    s = get_episode_summary(env)
                    if hasattr(s, "__dict__"):
                        summary = vars(s)
                    elif hasattr(s, "model_dump"):
                        summary = s.model_dump()
                except Exception:
                    pass

                records.append({
                    "training_step": int(step),
                    "sample_index": i,
                    "episode_log": episode_log,
                    "reward_breakdown": bd,
                    "episode_summary": summary,
                })
            except Exception as exc:
                records.append({
                    "training_step": int(step),
                    "sample_index": i,
                    "error": str(exc),
                })

        with self._output_file.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# ======================================================================= #
# RewardMonitorCallback                                                     #
# ======================================================================= #

class RewardMonitorCallback(_TrainerCallbackBase):
    """TrainerCallback that logs per-component reward columns and detects reward hacking.

    Compatible with both HuggingFace TrainerCallback (on_log / on_step_end)
    and direct use without a trainer.

    Reward hacking alert: if format_reward_mean > hack_alert_threshold AND
    outcome_reward_mean < 0.2, a WARNING is emitted because the model may
    be gaming the format signal without learning the task.

    Parameters
    ----------
    log_every_n_steps:
        Frequency for flushing component means to logs.
    generation_sample_every_n_steps:
        Frequency for writing episode transcript samples.
    generation_sample_file:
        Path to the JSONL file where transcripts are written.
    hack_alert_threshold:
        format_reward threshold above which the hacking check fires.
    """

    def __init__(
        self,
        log_every_n_steps: int = 10,
        generation_sample_every_n_steps: int = 50,
        generation_sample_file: Path = Path("logs/sampled_generations.jsonl"),
        hack_alert_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self._log_every = int(log_every_n_steps)
        self._sample_every = int(generation_sample_every_n_steps)
        self._hack_threshold = float(hack_alert_threshold)
        self._sampler = GenerationSampler(
            output_file=Path(generation_sample_file),
            n_samples=3,
        )
        self.success_tracker = SuccessRateTracker(window_size=20)

        # Rolling accumulator for per-component means between log flushes
        self._outcome_rewards: list[float] = []
        self._format_rewards: list[float] = []
        self._process_rewards: list[float] = []
        self._anti_hack_penalties: list[float] = []
        self._current_environments: list[Any] = []

    # ------------------------------------------------------------------ #
    # Public update method (call from reward functions if not using TRL)   #
    # ------------------------------------------------------------------ #

    def record_batch(self, environments: list[Any]) -> None:
        """Record reward component values from a batch of environments."""
        self._current_environments = list(environments)
        for env in environments:
            try:
                bd = env.reward_breakdown
                self._outcome_rewards.append(float(bd.terminal_component()))
                self._format_rewards.append(float(bd.format_compliance))
                self._process_rewards.append(float(bd.process_component()))
                self._anti_hack_penalties.append(float(bd.penalty_total()))
            except Exception:
                self._outcome_rewards.append(float(getattr(env, "reward", 0.0)))
                self._format_rewards.append(0.0)
                self._process_rewards.append(0.0)
                self._anti_hack_penalties.append(0.0)

    def _flush_component_means(self, step: int) -> dict[str, float]:
        """Compute and clear the rolling component buffers."""
        metrics: dict[str, float] = {}
        if self._outcome_rewards:
            metrics["reward_components/outcome_reward_mean"] = round(mean(self._outcome_rewards), 4)
        if self._format_rewards:
            metrics["reward_components/format_reward_mean"] = round(mean(self._format_rewards), 4)
        if self._process_rewards:
            metrics["reward_components/process_reward_mean"] = round(mean(self._process_rewards), 4)
        if self._anti_hack_penalties:
            metrics["reward_components/anti_hack_penalty_mean"] = round(mean(self._anti_hack_penalties), 4)
        metrics["reward_components/success_rate"] = self.success_tracker.success_rate
        metrics["reward_components/timeout_rate"] = self.success_tracker.timeout_rate

        # Reward hacking detector
        fmt_mean = metrics.get("reward_components/format_reward_mean", 0.0)
        out_mean = metrics.get("reward_components/outcome_reward_mean", 1.0)
        if fmt_mean > self._hack_threshold and out_mean < 0.2:
            warnings.warn(
                f"[RewardMonitor step={step}] Reward hacking detected: "
                f"format_reward_mean={fmt_mean:.3f} > {self._hack_threshold} "
                f"but outcome_reward_mean={out_mean:.3f} < 0.2. "
                "Model may be gaming format without learning the task.",
                stacklevel=2,
            )
            logger.warning(
                "Reward hacking detected at step %d: format=%.3f outcome=%.3f",
                step, fmt_mean, out_mean,
            )
            metrics["reward_components/hacking_alert"] = 1.0
        else:
            metrics["reward_components/hacking_alert"] = 0.0

        self._outcome_rewards.clear()
        self._format_rewards.clear()
        self._process_rewards.clear()
        self._anti_hack_penalties.clear()
        return metrics

    # ------------------------------------------------------------------ #
    # HuggingFace TrainerCallback interface                                #
    # ------------------------------------------------------------------ #

    def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """No-op init hook so Trainer can always register this callback safely."""
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Called by HuggingFace Trainer after each optimizer step."""
        global_step = int(getattr(state, "global_step", 0) or 0)

        # Sample generations periodically
        if global_step > 0 and global_step % self._sample_every == 0:
            if self._current_environments:
                self._sampler.sample_and_write(self._current_environments, step=global_step)

        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called by HuggingFace Trainer at logging steps."""
        global_step = int(getattr(state, "global_step", 0) or 0)
        if logs is None:
            return control

        if global_step % max(1, self._log_every) == 0:
            component_metrics = self._flush_component_means(global_step)
            logs.update(component_metrics)

        return control
