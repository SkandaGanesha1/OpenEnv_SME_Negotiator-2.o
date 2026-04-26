"""
In-memory episode state manager for one Gradio browser session.

``SessionStore`` is designed to be stored directly in a ``gr.State`` component.
It replaces the raw ``dict`` that was previously passed around as Gradio state.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional


class SessionStore:
    """
    Manages the full lifecycle of a negotiation episode for one browser session.

    Attributes are read/written by ``reset_episode()`` and ``submit_step()``
    in ``app.py``.  Gradio serialises and passes the object automatically —
    no special pickling is needed.
    """

    def __init__(self) -> None:
        # Live environment handle (SMENegotiatorEnvironment instance or None)
        self.env: Any = None

        # Episode metadata
        self.episode_id: str = ""
        self.task_label: str = ""
        self.seed: int = 42

        # Running counters
        self.cum_rew: float = 0.0
        self.step_num: int = 0

        # Chatbot message history  — list of [user_str, assistant_str] pairs
        self.messages: list[list[str]] = []

        # Last observation payload (for JSON viewer)
        self.last_payload: dict[str, Any] = {
            "reward": 0.0, "done": False, "info": {},
        }

        # Internal step-reward list for episode analytics
        self._step_rewards: list[float] = []

    # ── episode lifecycle ─────────────────────────────────────────────────────

    def start_episode(self, env: Any, task_label: str, seed: int) -> str:
        """
        Initialise a new episode.  Clears all previous state.

        Returns:
            A short hex ``episode_id`` for logging.
        """
        self.env = env
        self.task_label = task_label
        self.seed = seed
        self.episode_id = uuid.uuid4().hex[:8]
        self.messages = []
        self.cum_rew = 0.0
        self.step_num = 0
        self._step_rewards = []
        self.last_payload = {"reward": 0.0, "done": False, "info": {}}
        return self.episode_id

    def record_step(
        self,
        *,
        user_message: str,
        assistant_message: str,
        reward: float,
        done: bool,
        obs_dict: dict[str, Any],
    ) -> None:
        """Append one completed negotiation turn."""
        self.step_num += 1
        self.cum_rew += reward
        self._step_rewards.append(reward)
        self.messages = self.messages + [[user_message, assistant_message]]
        self.last_payload = {
            "reward": reward,
            "done": done,
            "info": {
                k: obs_dict.get(k)
                for k in (
                    "buyer_price", "buyer_days", "round_number",
                    "buyer_accepted", "negotiation_done", "message",
                )
            },
        }

    def clear(self) -> None:
        """Full reset (same as constructing a new SessionStore)."""
        self.__init__()  # type: ignore[misc]

    # ── read helpers ──────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True once ``start_episode`` has been called."""
        return self.env is not None

    @property
    def chatbot_messages(self) -> list[list[str]]:
        return self.messages

    def summary(self) -> dict[str, Any]:
        """Aggregate stats for the current episode."""
        return {
            "episode_id": self.episode_id,
            "task": self.task_label,
            "seed": self.seed,
            "steps": self.step_num,
            "cumulative_reward": round(self.cum_rew, 4),
            "mean_step_reward": round(
                self.cum_rew / max(self.step_num, 1), 4
            ),
            "best_step_reward": round(max(self._step_rewards, default=0.0), 4),
            "worst_step_reward": round(min(self._step_rewards, default=0.0), 4),
        }
