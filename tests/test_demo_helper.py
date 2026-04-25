"""Stage 7 smoke tests for the notebook-facing RL demo helper."""

from __future__ import annotations

import sys
from pathlib import Path
import re
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rl.demo as demo
from rl.demo import run_heuristic_episode, run_policy_episode


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


def test_run_heuristic_episode_summary_allows_attribute_access() -> None:
    run = run_heuristic_episode(seed=19, total_periods=2, task_name="liquidity-correlation-hard")
    summary = run["summary"]

    assert isinstance(summary.success_no_default_positive_npv, bool)
    assert isinstance(summary.total_reward, float)


def test_run_policy_episode_trained_uses_wrapper_path_and_surfaces_raw_text(monkeypatch) -> None:
    state = {"factory_called": False, "accept_calls": 0}

    class _Obs:
        def __init__(self, *, done: bool, reward: float, buyer_days: int = 40) -> None:
            self.done = done
            self.reward = reward
            self.metadata = {}
            self.active_deal_id = "deal-1"
            self.open_deal_ids = ["deal-1"] if not done else []
            self.buyer_price = 95.0
            self.buyer_days = buyer_days
            self.cost_threshold = 82.0
            self.liquidity_threshold = 35
            self.current_period = 0 if not done else 1
            self.total_periods = 1

        def model_dump(self):
            return {
                "done": self.done,
                "reward": self.reward,
                "metadata": self.metadata,
                "active_deal_id": self.active_deal_id,
                "open_deal_ids": list(self.open_deal_ids),
                "buyer_price": self.buyer_price,
                "buyer_days": self.buyer_days,
                "cost_threshold": self.cost_threshold,
                "liquidity_threshold": self.liquidity_threshold,
                "current_period": self.current_period,
                "total_periods": self.total_periods,
            }

    class _FakeWrapper:
        def __init__(self) -> None:
            self.last_observation = _Obs(done=False, reward=0.0)
            self.done = False

        def reset(self, **kwargs):
            self.last_observation = _Obs(done=False, reward=0.0)
            self.done = False
            return "obs-reset"

        def accept(self, **kwargs):
            state["accept_calls"] += 1
            self.last_observation = _Obs(done=True, reward=0.25, buyer_days=int(kwargs.get("payment_days", 40)))
            self.done = True
            return "obs-accept"

    def _fake_factory(**kwargs):
        state["factory_called"] = True
        return _FakeWrapper

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, checkpoint_path):
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, checkpoint_path):
            return cls()

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModelForCausalLM = _FakeModel

    fake_peft = types.ModuleType("peft")

    class _FakePeftConfig:
        @staticmethod
        def from_pretrained(path):
            raise RuntimeError("not a peft checkpoint")

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model, path):
            return model

    fake_peft.PeftConfig = _FakePeftConfig
    fake_peft.PeftModel = _FakePeftModel

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(demo, "configure_rollout_tokenizer", lambda tokenizer: tokenizer)
    monkeypatch.setattr(demo, "make_environment_factory", _fake_factory)
    monkeypatch.setattr(
        demo,
        "_generate_completion_turn",
        lambda *args, **kwargs: {
            "prompt_ids": [1],
            "completion_ids": [2],
            "logprobs": [-0.1],
            "text": "<think>Accept the buyer terms at 40 days</think> accept 40 days",
        },
    )

    transcript = run_policy_episode(
        policy="trained",
        seed=77,
        total_periods=1,
        task_name="liquidity-correlation-hard",
        difficulty="hard",
        checkpoint_path=str(PROJECT_ROOT / "outputs" / "fake-checkpoint"),
        max_steps=2,
    )

    assert state["factory_called"] is True
    assert state["accept_calls"] == 1
    assert "MODEL 1 :: valid_json=false" in transcript
    assert "raw=<think>Accept the buyer terms at 40 days</think> accept 40 days" in transcript
    assert "STEP 1 :: action[" in transcript
