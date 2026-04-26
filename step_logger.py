"""
Structured JSON step logger for SME Negotiator episodes.

Each event is emitted as a single-line JSON record to both stdout and
(if writable) a local log file.  On read-only filesystems (e.g. HF Spaces
Docker), the file handler is silently skipped and only the console is used.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional


class StepLogger:
    """Emits structured JSON log lines for every negotiation event."""

    def __init__(self, log_file: str = "negotiation_steps.log") -> None:
        self.logger = logging.getLogger("sme_negotiator.steps")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            # Console handler — always available
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(sh)

            # File handler — best-effort (may fail on read-only HF filesystem)
            try:
                log_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), log_file
                )
                fh = logging.FileHandler(log_path, encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(fh)
            except OSError:
                pass  # Silently fall back to console-only

    # ── internal ─────────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _emit(self, entry: dict[str, Any]) -> None:
        self.logger.info(json.dumps(entry, default=str))

    # ── public API ────────────────────────────────────────────────────────────

    def log_reset(
        self,
        *,
        episode_id: str,
        task_label: str,
        seed: int,
    ) -> None:
        """Log the start of a new episode."""
        self._emit({
            "event": "reset",
            "ts": self._now(),
            "episode_id": episode_id,
            "task": task_label,
            "seed": seed,
        })

    def log_step(
        self,
        *,
        episode_id: str,
        step: int,
        action_type: str,
        price: float,
        payment_days: int,
        use_treds: bool,
        reward: float,
        cum_reward: float,
        done: bool,
        buyer_price: Optional[float] = None,
        buyer_days: Optional[int] = None,
        round_number: Optional[int] = None,
    ) -> None:
        """Log one completed negotiation step."""
        self._emit({
            "event": "step",
            "ts": self._now(),
            "episode_id": episode_id,
            "step": step,
            "action_type": action_type,
            "price": round(price, 2),
            "payment_days": payment_days,
            "use_treds": use_treds,
            "reward": round(reward, 4),
            "cum_reward": round(cum_reward, 4),
            "done": done,
            "buyer_price": round(buyer_price, 2) if buyer_price is not None else None,
            "buyer_days": buyer_days,
            "round_number": round_number,
        })

    def log_validation_error(
        self,
        *,
        episode_id: str,
        step: int,
        error: str,
        action_type: str,
        price: float,
        payment_days: int,
    ) -> None:
        """Log a pre-flight validation failure (action not sent to env)."""
        self._emit({
            "event": "validation_error",
            "ts": self._now(),
            "episode_id": episode_id,
            "step": step,
            "error": error,
            "action_type": action_type,
            "price": round(price, 2),
            "payment_days": payment_days,
        })

    def log_episode_end(
        self,
        *,
        episode_id: str,
        task_label: str,
        steps: int,
        total_reward: float,
    ) -> None:
        """Log episode completion summary."""
        self._emit({
            "event": "episode_end",
            "ts": self._now(),
            "episode_id": episode_id,
            "task": task_label,
            "steps": steps,
            "total_reward": round(total_reward, 4),
        })
