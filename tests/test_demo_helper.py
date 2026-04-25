"""Stage 7 smoke tests for the notebook-facing RL demo helper."""

from __future__ import annotations

import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.demo import run_heuristic_episode


def _normalize_transcript(text: str) -> str:
    return re.sub(r"Episode reset @ [^\n]+", "Episode reset @ <normalized>", text)


def test_run_heuristic_episode_is_stable_and_download_free() -> None:
    first = run_heuristic_episode(seed=77, total_periods=2, task_name="liquidity-correlation-hard")
    second = run_heuristic_episode(seed=77, total_periods=2, task_name="liquidity-correlation-hard")

    assert first["total_reward"] == second["total_reward"]
    assert first["steps"] == second["steps"]
    assert first["done"] == second["done"]
    assert _normalize_transcript(first["transcript"]) == _normalize_transcript(second["transcript"])
    assert isinstance(first["total_reward"], float)
    assert isinstance(first["transcript"], str)
    assert len(first["transcript"]) > 0
    assert first["summary"] is not None
    assert "verifiable_reward" in first["summary"]
    assert "tool_call_count" in first["summary"]
    assert "terminated_by_step_cap" in first["summary"]
