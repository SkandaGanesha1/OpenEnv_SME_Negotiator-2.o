"""
Convenience wrappers around the existing graders for use in the Gradio UI.

Delegates all actual computation to ``sme_negotiator_env.graders`` — this
module only provides a clean interface and formatting helpers.
"""
from __future__ import annotations

from typing import Any

from config import GRADER_WEIGHTS


class RewardEngine:
    """Thin wrapper around the existing grader functions."""

    # ── per-step helpers ─────────────────────────────────────────────────────

    def extract_step_reward(self, obs_dict: dict[str, Any]) -> float:
        """Pull the per-step shaping reward from a raw observation dict."""
        return float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))

    def is_done(self, obs_dict: dict[str, Any]) -> bool:
        return bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))

    def buyer_accepted(self, obs_dict: dict[str, Any]) -> bool:
        return bool(obs_dict.get("buyer_accepted", False))

    # ── terminal score ───────────────────────────────────────────────────────

    def compute_terminal_score(
        self,
        task_id: str,
        state_kwargs: dict[str, Any],
    ) -> float | None:
        """
        Compute the benchmark terminal score for a completed episode.

        Args:
            task_id: e.g. "payment-terms-easy"
            state_kwargs: keyword arguments forwarded to ``NegotiationState()``.

        Returns:
            Score in [0, 1], or ``None`` if the grader is unavailable.
        """
        try:
            from sme_negotiator_env.graders import TASK_GRADERS
            from sme_negotiator_env.models import NegotiationState
        except ImportError:
            return None

        grader = TASK_GRADERS.get(task_id)
        if grader is None:
            return None

        try:
            dummy = NegotiationState(**state_kwargs)
            return float(grader(dummy))
        except Exception:
            return None

    # ── display helpers ──────────────────────────────────────────────────────

    def score_bar(self, score: float, width: int = 10) -> str:
        """ASCII progress bar for a 0–1 score."""
        filled = max(0, min(width, round(float(score) * width)))
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {score:.4f}"

    def formula_string(self) -> str:
        """Human-readable grading formula."""
        parts = [f"{w:.2f} × {k}" for k, w in GRADER_WEIGHTS.items()]
        return "score = " + " + ".join(parts)

    def reward_label(self, reward: float) -> str:
        """Short signed label for a reward value."""
        sign = "▲" if reward >= 0 else "▼"
        return f"{sign} {reward:+.4f}"

    def episode_summary(
        self,
        step_rewards: list[float],
        cum_reward: float,
        step_num: int,
    ) -> dict[str, float | int]:
        """Aggregate stats for an ended episode."""
        return {
            "steps": step_num,
            "cumulative_reward": round(cum_reward, 4),
            "mean_step_reward": round(cum_reward / max(step_num, 1), 4),
            "best_step": round(max(step_rewards, default=0.0), 4),
            "worst_step": round(min(step_rewards, default=0.0), 4),
        }
