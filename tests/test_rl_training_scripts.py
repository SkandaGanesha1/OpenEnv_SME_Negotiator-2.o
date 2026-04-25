"""Stage 5 dry-run and helper tests for RL training scripts."""

from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.curriculum import CurriculumManager
from rl.opponents import OpponentPolicyManager
from rl.episode_logging import EpisodeSummary
from rl.train_grpo_trl import (
    _ensure_grpo_response_schema,
    _render_chat_prompt,
    _run_single_rollout_sample,
    _score_prompt_completion_via_environment,
    _strict_json_payload,
    EpisodeSummaryBuffer,
    PendingRolloutBuffer,
    create_trainer,
    build_arg_parser,
    build_environment_factory,
    build_grpo_config_kwargs,
    build_metrics_callback,
    build_rollout_func,
    build_training_session,
    build_snapshot_callback,
    build_training_rows,
    configure_tokenizer,
    main as trl_main,
    make_training_args,
    make_reward_function,
    prepare_model_for_grpo,
    _require_rollout_func_support,
    run_training_session,
    save_training_session,
)
from rl.monitoring import RewardMonitorCallback
from rl.train_grpo_unsloth import (
    build_grpo_config_kwargs as unsloth_build_grpo_config_kwargs,
    main as unsloth_main,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "<eos>"


class _DummyEnv:
    current_persona = None

    def compute_final_reward(self) -> float:
        return 0.75

    def build_episode_log(self) -> str:
        return "episode-log"

    def summarize_episode(self) -> EpisodeSummary:
        return EpisodeSummary(
            episode_completed=True,
            base_rl_reward=0.7,
            tool_bonus_total=0.05,
            env_reward_total=0.75,
            success_no_default_positive_npv=True,
            average_final_payment_days=30.0,
            tool_usage_count=2,
            resolved_deal_count=2,
            defaulted_sme_count=0,
        )


class _DummyModelSaver:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_text("model", encoding="utf-8")


class _DummyTokenizerSaver:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("tokenizer", encoding="utf-8")


def _workspace_tmp_dir(name: str) -> Path:
    path = PROJECT_ROOT / ".test_tmp" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_notebook_source() -> str:
    notebook_path = PROJECT_ROOT / "notebooks" / "colab_grpo_sme_liquidity.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))


def test_dry_run_works_for_trl_and_unsloth(capsys) -> None:
    assert trl_main(["--dry-run", "--num-samples", "2"]) == 0
    captured = capsys.readouterr()
    assert "liquidity-correlation-hard" in captured.out

    assert unsloth_main(["--dry-run", "--num-samples", "2"]) == 0
    captured = capsys.readouterr()
    assert "liquidity-correlation-hard" in captured.out


def test_dry_run_supports_stage6_flags(capsys) -> None:
    assert (
        trl_main(
            [
                "--dry-run",
                "--num-samples",
                "2",
                "--enable-self-play",
                "--enable-rubrics",
                "--persona-mode",
                "fixed",
                "--build-preference-dataset",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"self_play"' in captured.out
    assert '"rubrics"' in captured.out

    assert (
        unsloth_main(
            [
                "--dry-run",
                "--num-samples",
                "2",
                "--enable-self-play",
                "--enable-rubrics",
                "--persona-mode",
                "fixed",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"self_play"' in captured.out
    assert '"curriculum"' in captured.out


def test_dataset_builder_emits_expected_columns_and_seed_range() -> None:
    rows = build_training_rows(num_samples=3, seed_base=1000)
    assert len(rows) == 3
    assert rows[0]["seed"] == 1000
    assert rows[2]["seed"] == 1002
    assert set(rows[0]) == {"prompt", "task_name", "difficulty", "seed", "total_periods"}


def test_configure_tokenizer_enforces_left_padding_and_pad_token_fallback() -> None:
    tokenizer = configure_tokenizer(_DummyTokenizer())
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"


def test_ensure_grpo_response_schema_adds_fallback_schema_when_missing() -> None:
    tokenizer = _ensure_grpo_response_schema(_DummyTokenizer())
    assert getattr(tokenizer, "response_schema", None) is not None
    assert tokenizer.response_schema["properties"]["content"]["type"] == "string"


def test_render_chat_prompt_disables_thinking_when_chat_template_supports_it() -> None:
    captured: dict[str, object] = {}

    class _TemplateTokenizer(_DummyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "rendered-prompt"

    prompt = _render_chat_prompt(
        _TemplateTokenizer(),
        [{"role": "user", "content": "Return JSON only."}],
    )

    assert prompt == "rendered-prompt"
    assert captured["kwargs"]["tokenize"] is False
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert captured["kwargs"]["enable_thinking"] is False


def test_reward_function_reads_environments_and_returns_one_scalar_per_env() -> None:
    reward_func = make_reward_function(
        rubric_scorer=lambda episode_log: {"rubric": 0.5},
        rubric_weight=0.2,
    )
    rewards = reward_func([_DummyEnv(), _DummyEnv()])
    assert rewards == [0.85, 0.85]


def test_reward_function_accepts_keyword_only_trl_call_shape() -> None:
    reward_func = make_reward_function()

    rewards = reward_func(
        prompts=["prompt-a", "prompt-b"],
        completions=["{}", '{"action_type":"accept"}'],
        environments=[_DummyEnv(), _DummyEnv()],
    )

    assert len(rewards) == 2
    assert all(isinstance(value, float) for value in rewards)


def test_reward_function_uses_pending_rollout_buffer_for_keyword_only_calls() -> None:
    pending_buffer = PendingRolloutBuffer()
    reward_func = make_reward_function(pending_rollout_buffer=pending_buffer)

    pending_buffer.extend(
        [
            {
                "prompt": "prompt-a",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.4,
                    verifiable_reward=0.4,
                    total_reward=0.4,
                    tool_bonus_total=0.0,
                    env_reward_total=0.4,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=40.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-a",
                "raw_completion_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "completion_signature_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "reward_std": 0.2,
                "reward_mean": 0.45,
                "unique_action_count": 2.0,
                "unique_completion_count": 2.0,
                "invalid_parse_fraction": 0.0,
                "identical_terminal_fraction": 0.0,
                "termination_reason": "done",
            },
            {
                "prompt": "prompt-b",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.6,
                    verifiable_reward=0.6,
                    total_reward=0.6,
                    tool_bonus_total=0.0,
                    env_reward_total=0.6,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=35.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-b",
                "raw_completion_text": '{"action_type":"propose","price":97.0,"payment_days":35}',
                "completion_signature_text": '{"action_type":"propose","price":97.0,"payment_days":35}',
                "reward_std": 0.2,
                "reward_mean": 0.45,
                "unique_action_count": 2.0,
                "unique_completion_count": 2.0,
                "invalid_parse_fraction": 0.0,
                "identical_terminal_fraction": 0.0,
                "termination_reason": "done",
            },
        ]
    )

    rewards = reward_func(
        prompts=["prompt-a", "prompt-b"],
        completions=[
            '{"action_type":"accept","price":95.0,"payment_days":40}',
            '{"action_type":"propose","price":97.0,"payment_days":35}',
        ],
    )

    assert rewards == [0.4, 0.6]
    assert pending_buffer.items == []


def test_reward_function_matches_pending_rollouts_by_prompt_completion_signature_when_reordered() -> None:
    pending_buffer = PendingRolloutBuffer()
    reward_func = make_reward_function(pending_rollout_buffer=pending_buffer)
    pending_buffer.extend(
        [
            {
                "prompt": [{"role": "user", "content": "prompt-a"}],
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.4,
                    verifiable_reward=0.4,
                    total_reward=0.4,
                    tool_bonus_total=0.0,
                    env_reward_total=0.4,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=40.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-a",
                "raw_completion_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "completion_signature_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "reward_std": 0.2,
            },
            {
                "prompt": [{"role": "user", "content": "prompt-b"}],
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.6,
                    verifiable_reward=0.6,
                    total_reward=0.6,
                    tool_bonus_total=0.0,
                    env_reward_total=0.6,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=35.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-b",
                "raw_completion_text": '{"action_type":"propose","price":97.0,"payment_days":35}',
                "completion_signature_text": '{"action_type":"propose","price":97.0,"payment_days":35}',
                "reward_std": 0.2,
            },
        ]
    )

    rewards = reward_func(
        prompts=[
            [{"role": "user", "content": "prompt-b"}],
            [{"role": "user", "content": "prompt-a"}],
        ],
        completions=[
            '{"action_type":"propose","price":97.0,"payment_days":35}',
            '{"action_type":"accept","price":95.0,"payment_days":40}',
        ],
    )

    assert rewards == [0.6, 0.4]
    assert pending_buffer.items == []


def test_reward_function_preserves_duplicate_signature_queue_order() -> None:
    pending_buffer = PendingRolloutBuffer()
    reward_func = make_reward_function(pending_rollout_buffer=pending_buffer)
    pending_buffer.extend(
        [
            {
                "prompt": "prompt-a",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.4,
                    verifiable_reward=0.4,
                    total_reward=0.4,
                    tool_bonus_total=0.0,
                    env_reward_total=0.4,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=40.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-a",
                "raw_completion_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "completion_signature_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "reward_std": 0.2,
            },
            {
                "prompt": "prompt-a",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.6,
                    verifiable_reward=0.6,
                    total_reward=0.6,
                    tool_bonus_total=0.0,
                    env_reward_total=0.6,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=40.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-b",
                "raw_completion_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "completion_signature_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "reward_std": 0.2,
            },
        ]
    )

    rewards = reward_func(
        prompts=["prompt-a", "prompt-a"],
        completions=[
            '{"action_type":"accept","price":95.0,"payment_days":40}',
            '{"action_type":"accept","price":95.0,"payment_days":40}',
        ],
    )

    assert rewards == [0.4, 0.6]
    assert pending_buffer.items == []


def test_reward_function_reports_bridge_miss_and_returns_strict_invalid_reward() -> None:
    pending_buffer = PendingRolloutBuffer()
    reward_func = make_reward_function(pending_rollout_buffer=pending_buffer)
    pending_buffer.extend(
        [
            {
                "prompt": "prompt-a",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.4,
                    verifiable_reward=0.4,
                    total_reward=0.4,
                    tool_bonus_total=0.0,
                    env_reward_total=0.4,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=40.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-a",
                "raw_completion_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "completion_signature_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "reward_std": 0.2,
            }
        ]
    )

    rewards = reward_func(
        prompts=["prompt-b", "prompt-c"],
        completions=[
            "<think>not json</think>",
            '{"action_type":"proposal","price":95.0,"payment_days":40}',
        ],
    )

    assert rewards[0] < rewards[1] < 0.0
    assert reward_func.bridge_diagnostics["bridge_miss_count"] == 2
    assert len(pending_buffer.items) == 1


def test_reward_function_uses_fifo_compatibility_fallback_after_signature_miss() -> None:
    pending_buffer = PendingRolloutBuffer()
    reward_func = make_reward_function(pending_rollout_buffer=pending_buffer)
    pending_buffer.extend(
        [
            {
                "prompt": "prompt-a",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.4,
                    verifiable_reward=0.4,
                    total_reward=0.4,
                    tool_bonus_total=0.0,
                    env_reward_total=0.4,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=40.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-a",
                "raw_completion_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "completion_signature_text": '{"action_type":"accept","price":95.0,"payment_days":40}',
                "reward_std": 0.2,
            },
            {
                "prompt": "prompt-b",
                "episode_summary": EpisodeSummary(
                    episode_completed=True,
                    base_rl_reward=0.6,
                    verifiable_reward=0.6,
                    total_reward=0.6,
                    tool_bonus_total=0.0,
                    env_reward_total=0.6,
                    success_no_default_positive_npv=True,
                    average_final_payment_days=35.0,
                    tool_usage_count=1,
                    resolved_deal_count=1,
                    defaulted_sme_count=0,
                ),
                "episode_log": "log-b",
                "raw_completion_text": '{"action_type":"propose","price":97.0,"payment_days":35}',
                "completion_signature_text": '{"action_type":"propose","price":97.0,"payment_days":35}',
                "reward_std": 0.2,
            },
        ]
    )

    rewards = reward_func(
        prompts=["different-a", "different-b"],
        completions=[
            '{"action_type":"accept","price":95.0,"payment_days":40}',
            '{"action_type":"propose","price":97.0,"payment_days":35}',
        ],
    )

    assert rewards == [0.4, 0.6]
    assert reward_func.bridge_diagnostics["bridge_miss_count"] == 0
    assert pending_buffer.items == []


def test_reward_function_uses_prompt_env_fallback_when_pending_buffer_is_empty(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    pending_buffer = PendingRolloutBuffer()

    def _fake_score(prompt, completion, *, fallback_args, env_factory):
        is_valid = '"action_type":"propose"' in str(completion)
        reward = 0.6 if is_valid else 0.65
        return {
            "episode_summary": EpisodeSummary(
                episode_completed=is_valid,
                base_rl_reward=reward,
                verifiable_reward=reward,
                total_reward=reward,
                tool_bonus_total=0.0,
                env_reward_total=reward,
                success_no_default_positive_npv=is_valid,
                average_final_payment_days=35.0,
                tool_usage_count=1,
                resolved_deal_count=1 if is_valid else 0,
                defaulted_sme_count=0,
            ),
            "episode_log": f"log::{prompt}",
            "raw_completion_text": str(completion),
            "termination_reason": "done" if is_valid else "prompt_env_invalid_first_action",
            "invalid_parse_fraction": 0.0 if is_valid else 1.0,
            "contract_score": 0.95 if is_valid else 0.65,
        }

    monkeypatch.setattr(trl_script, "_score_prompt_completion_via_environment", _fake_score)
    reward_func = make_reward_function(
        pending_rollout_buffer=pending_buffer,
        prompt_env_factory=lambda: object(),
        prompt_env_args=make_training_args(),
    )

    rewards = reward_func(
        prompts=["prompt-a", "prompt-b"],
        completions=[
            '{"action_type":"propose","price":95.0,"payment_days":40}',
            '{"action_type":"proposal","price":95.0,"payment_days":40}',
        ],
    )

    assert rewards[0] > 0.6
    assert rewards[1] < 0.0
    assert reward_func.bridge_diagnostics["bridge_miss_count"] == 2
    assert reward_func.bridge_diagnostics["prompt_env_fallback_count"] == 2


def test_score_prompt_completion_via_environment_short_circuits_invalid_first_action(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    class _Wrapper:
        def __init__(self) -> None:
            self.done = False
            self.last_observation = SimpleNamespace(metadata={})

        def reset(self, **kwargs):
            self.done = False
            self.last_observation = SimpleNamespace(metadata={})
            return "reset"

        def summarize_episode(self) -> EpisodeSummary:
            return EpisodeSummary(
                episode_completed=False,
                base_rl_reward=0.8,
                verifiable_reward=0.8,
                total_reward=0.8,
                tool_bonus_total=0.0,
                env_reward_total=0.8,
                success_no_default_positive_npv=True,
                average_final_payment_days=30.0,
                tool_usage_count=1,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            )

        def build_episode_log(self) -> str:
            return "wrapper-log"

    monkeypatch.setattr(trl_script, "_extract_training_row_from_prompt", lambda prompt, fallback_args: {"prompt": prompt})
    monkeypatch.setattr(
        trl_script,
        "_parse_action_and_validity",
        lambda raw_text, observation, strict_json=True: (
            SimpleNamespace(model_dump=lambda exclude_none=True: {"action_type": "advance_period"}),
            False,
        ),
    )

    def _fail_execute_action(wrapper, action):
        raise AssertionError("invalid first action should not be executed in prompt-env fallback")

    monkeypatch.setattr(trl_script, "execute_action", _fail_execute_action)

    result = _score_prompt_completion_via_environment(
        "prompt-a",
        '{"action_type":"proposal","price":95.0,"payment_days":40}',
        fallback_args=make_training_args(),
        env_factory=_Wrapper,
    )

    assert result["termination_reason"] == "prompt_env_invalid_first_action"
    assert result["episode_summary"].verifiable_reward < 0.0
    assert result["invalid_parse_fraction"] == 1.0
    assert result["parsed_actions"] == []


def test_reward_function_falls_back_safely_for_non_environment_prompt_inputs() -> None:
    reward_func = make_reward_function()

    rewards = reward_func(
        prompts=[[{"role": "user", "content": "hello"}], [{"role": "user", "content": "world"}]],
        completions=["{}", '{"action_type":"accept"}'],
    )

    assert len(rewards) == 2
    assert all(isinstance(value, float) for value in rewards)


def test_reward_function_penalizes_invalid_json_in_episode_payloads() -> None:
    reward_func = make_reward_function()
    summary = EpisodeSummary(
        episode_completed=True,
        base_rl_reward=0.5,
        verifiable_reward=0.5,
        total_reward=0.5,
        tool_bonus_total=0.0,
        env_reward_total=0.5,
        success_no_default_positive_npv=True,
        average_final_payment_days=40.0,
        tool_usage_count=1,
        resolved_deal_count=1,
        defaulted_sme_count=0,
    )

    rewards = reward_func(
        [None, None],
        episode_summaries=[summary, summary],
        raw_completion_texts=[
            '{"action_type":"accept","price":95.0,"payment_days":40}',
            "<think>I'll think about it first</think>",
        ],
        env_reward_std=[0.0, 0.0],
        invalid_parse_fraction=[0.0, 1.0],
    )

    assert rewards[0] > rewards[1]


def test_reward_function_penalizes_prompt_env_incomplete_episode_payloads() -> None:
    reward_func = make_reward_function()
    summary = EpisodeSummary(
        episode_completed=False,
        base_rl_reward=0.7,
        verifiable_reward=0.7,
        total_reward=0.7,
        tool_bonus_total=0.0,
        env_reward_total=0.7,
        success_no_default_positive_npv=True,
        average_final_payment_days=35.0,
        tool_usage_count=1,
        resolved_deal_count=1,
        defaulted_sme_count=0,
    )

    rewards = reward_func(
        [None, None],
        episode_summaries=[summary, summary],
        raw_completion_texts=[
            '{"action_type":"propose","price":95.0,"payment_days":40}',
            '{"action_type":"propose","price":95.0,"payment_days":40}',
        ],
        env_reward_std=[0.0, 0.0],
        invalid_parse_fraction=[0.0, 0.0],
        termination_reasons=["done", "prompt_env_incomplete"],
        contract_score=[0.95, 0.95],
    )

    assert rewards[0] > rewards[1]
    assert rewards[1] <= 0.5


def test_reward_function_completion_text_provides_per_group_variance() -> None:
    """GRPO loss collapses to 0 when every completion in a group scores
    identically. The reward function must therefore add a bounded text-derived
    signal so distinct model outputs always produce distinct rewards even when
    the deterministic env trajectory is constant across the group."""
    reward_func = make_reward_function()
    completions = [
        "",
        "I will think about it.",
        '{"action_type":"propose","price":95.0,"payment_days":45,"reason":"close gap"}',
        '{"action_type":"accept","deal_id":"d1","reason":"agreed"}',
    ]
    envs = [_DummyEnv() for _ in completions]
    rewards = reward_func(envs, completions=completions)
    assert len(rewards) == len(completions)
    # All rewards must differ by more than floating noise so GRPO advantage > 0.
    assert len(set(round(r, 4) for r in rewards)) >= 3
    # Bounded: contract shaping must stay small relative to the environment reward.
    base = max(rewards) - min(rewards)
    assert 0.0 < base <= 0.18


def test_strict_json_payload_rejects_near_schema_action_type_variants() -> None:
    assert _strict_json_payload('{"action_type":"proposed","price":100.0,"payment_days":30}') is None
    assert _strict_json_payload('{"action_type":"proposal","price":100.0,"payment_days":30}') is None
    assert _strict_json_payload('{"action_type":"proposals","price":100.0,"payment_days":30}') is None
    assert _strict_json_payload('{"action_type":"proposer","price":100.0,"payment_days":30}') is None


def test_strict_json_payload_requires_exact_field_types_for_contract_actions() -> None:
    assert _strict_json_payload('{"action_type":"propose","price":"0.01","payment_days":"15"}') is None
    assert _strict_json_payload('{"action_type":"tool","tool":"QUERY_TREDS"}') is None
    assert _strict_json_payload('{"action_type":"simulate_plan","simulation_plan":"simulate_two_periods"}') is None
    assert (
        _strict_json_payload(
            '{"action_type":"propose","price":95.0,"payment_days":40,"reason":"close gap"}'
        )
        == {
            "action_type": "propose",
            "price": 95.0,
            "payment_days": 40,
            "reason": "close gap",
        }
    )


def test_snapshot_callback_registers_and_prunes_opponent_zoo() -> None:
    tmp_path = _workspace_tmp_dir("snapshot_callback")
    manager = OpponentPolicyManager(snapshots_dir=tmp_path / "snapshots", zoo_size=2)
    callback = build_snapshot_callback(
        object,
        opponent_manager=manager,
        interval=100,
        output_dir=str(tmp_path),
    )

    for step in (100, 200, 300):
        callback.on_step_end(
            None,
            SimpleNamespace(global_step=step),
            SimpleNamespace(),
            model=_DummyModelSaver(),
            processing_class=_DummyTokenizerSaver(),
        )

    assert manager.snapshot_ids() == [
        "sme_policy_step_000200",
        "sme_policy_step_000300",
    ]


def test_environment_factory_picks_up_curriculum_config_and_opponents() -> None:
    tmp_path = _workspace_tmp_dir("environment_factory")
    curriculum = CurriculumManager(window_size=1)
    curriculum.record_episode(0.9, False)
    assert curriculum.maybe_advance_level() is True

    opponent_manager = OpponentPolicyManager(snapshots_dir=tmp_path / "snapshots", zoo_size=2)
    args = Namespace(
        task_name="liquidity-correlation-hard",
        difficulty="hard",
        total_periods=3,
        seed_base=1000,
        persona_mode="off",
        persona_name=None,
    )
    env_factory = build_environment_factory(
        args,
        curriculum=curriculum,
        opponent_manager=opponent_manager,
    )

    wrapper = env_factory()
    wrapper.reset(
        prompt=[{"role": "user", "content": "Train me"}],
        task_name="liquidity-correlation-hard",
        difficulty="hard",
        seed=1000,
        total_periods=99,
    )

    assert wrapper.total_periods == curriculum.current_config().total_periods
    assert wrapper.buyer_variance == curriculum.current_config().buyer_variance
    assert wrapper.financier_variance == curriculum.current_config().financier_variance
    assert wrapper.curriculum_level == curriculum.current_level()
    assert wrapper.opponent_manager is opponent_manager


def test_build_grpo_config_kwargs_uses_training_log_backend_env(monkeypatch) -> None:
    args = build_arg_parser().parse_args([])

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "tensorboard")
    kwargs = build_grpo_config_kwargs(args)
    assert kwargs["report_to"] == "tensorboard"
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["generation_batch_size"] == kwargs["num_generations"]
    assert kwargs["save_only_model"] is True
    assert kwargs["save_steps"] == 1000

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "unsupported")
    assert build_grpo_config_kwargs(args)["report_to"] == "none"


def test_make_training_args_applies_notebook_overrides_without_mutating_parser_defaults() -> None:
    defaults = make_training_args()
    updated = make_training_args(
        model_name="Qwen/Qwen3-0.6B",
        total_periods=2,
        num_samples=32,
        max_episode_steps=12,
        use_vllm=False,
    )

    assert defaults.total_periods == 3
    assert defaults.num_samples == 64
    assert defaults.max_episode_steps == 24
    assert updated.model_name == "Qwen/Qwen3-0.6B"
    assert updated.total_periods == 2
    assert updated.num_samples == 32
    assert updated.max_episode_steps == 12
    assert updated.use_vllm is False


def test_unsloth_config_uses_training_log_backend_env(monkeypatch) -> None:
    args = build_arg_parser().parse_args([])

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "wandb")
    assert unsloth_build_grpo_config_kwargs(args)["report_to"] == "wandb"

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "unsupported")
    assert unsloth_build_grpo_config_kwargs(args)["report_to"] == "none"


def test_build_rollout_func_duplicates_each_prompt_by_num_generations(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    prompt = build_training_rows(num_samples=1)[0]["prompt"]
    args = build_arg_parser().parse_args([])
    call_counter = {"value": 0}

    def _fake_run_single_rollout_sample(prompt, *, model, tokenizer, rollout_args, env_factory):
        index = call_counter["value"]
        call_counter["value"] += 1
        return {
            "prompt_ids": [1, 2],
            "completion_ids": [10 + index],
            "logprobs": [-0.1],
            "episode_summary": EpisodeSummary(
                episode_completed=True,
                base_rl_reward=0.3 + index * 0.1,
                verifiable_reward=0.3 + index * 0.1,
                total_reward=0.3 + index * 0.1,
                tool_bonus_total=0.0,
                env_reward_total=0.3 + index * 0.1,
                success_no_default_positive_npv=True,
                average_final_payment_days=40.0,
                tool_usage_count=1,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            ),
            "episode_log": f"log-{index}",
            "reward_breakdown": {"total": 0.3 + index * 0.1},
            "termination_reason": "done",
            "parsed_actions": [{"action_type": "accept", "price": 95.0, "payment_days": 40 + index}],
            "raw_completion_text": f'{{"action_type":"accept","price":95.0,"payment_days":{40 + index}}}',
            "invalid_parse_fraction": 0.0,
        }

    monkeypatch.setattr(trl_script, "_run_single_rollout_sample", _fake_run_single_rollout_sample)
    rollout_func = build_rollout_func(args, tokenizer=_DummyTokenizer(), curriculum=None, opponent_manager=None)
    trainer = SimpleNamespace(
        model=object(),
        args=SimpleNamespace(num_generations=3),
        processing_class=_DummyTokenizer(),
    )

    result = rollout_func([prompt], trainer)

    assert len(result["prompt_ids"]) == 3
    assert len(result["completion_ids"]) == 3
    assert len(result["episode_summaries"]) == 3
    assert all(std > 0.0 for std in result["env_reward_std"])
    assert all(count >= 1.0 for count in result["unique_action_count"])


def test_reward_function_uses_epsilon_tiebreaker_when_group_std_is_zero() -> None:
    reward_func = make_reward_function()
    summary = EpisodeSummary(
        episode_completed=True,
        base_rl_reward=0.5,
        verifiable_reward=0.5,
        total_reward=0.5,
        tool_bonus_total=0.0,
        env_reward_total=0.5,
        success_no_default_positive_npv=True,
        average_final_payment_days=40.0,
        tool_usage_count=1,
        resolved_deal_count=1,
        defaulted_sme_count=0,
    )
    rewards = reward_func(
        [None, None],
        episode_summaries=[summary, summary],
        raw_completion_texts=[
            '{"action_type":"accept","price":95.0,"payment_days":40,"reason":"good"}',
            "",
        ],
        env_reward_std=[0.0, 0.0],
    )
    assert rewards[0] > rewards[1]


def test_reward_function_ignores_epsilon_tiebreaker_when_env_variance_exists() -> None:
    reward_func = make_reward_function()
    summaries = [
        EpisodeSummary(
            episode_completed=True,
            base_rl_reward=0.4,
            verifiable_reward=0.4,
            total_reward=0.4,
            tool_bonus_total=0.0,
            env_reward_total=0.4,
            success_no_default_positive_npv=True,
            average_final_payment_days=40.0,
            tool_usage_count=1,
            resolved_deal_count=1,
            defaulted_sme_count=0,
        ),
        EpisodeSummary(
            episode_completed=True,
            base_rl_reward=0.6,
            verifiable_reward=0.6,
            total_reward=0.6,
            tool_bonus_total=0.0,
            env_reward_total=0.6,
            success_no_default_positive_npv=True,
            average_final_payment_days=40.0,
            tool_usage_count=1,
            resolved_deal_count=1,
            defaulted_sme_count=0,
        ),
    ]
    rewards = reward_func(
        [None, None],
        episode_summaries=summaries,
        raw_completion_texts=["{}", '{"action_type":"accept","price":95.0,"payment_days":40}'],
        env_reward_std=[0.2, 0.2],
    )
    assert rewards == [0.4, 0.6]


def test_run_single_rollout_sample_steps_environment_with_structured_actions(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    actions = [
        '{"action_type":"accept","price":95.0,"payment_days":40}',
        '{"action_type":"advance_period"}',
    ]

    def _fake_generate_completion_turn(model, tokenizer, messages, *, max_new_tokens, temperature, top_p):
        text = actions.pop(0)
        return {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4, 5],
            "logprobs": [-0.1, -0.2],
            "text": text,
        }

    monkeypatch.setattr(trl_script, "_generate_completion_turn", _fake_generate_completion_turn)
    args = build_arg_parser().parse_args(["--max-episode-steps", "2"])
    prompt = build_training_rows(num_samples=1, total_periods=1)[0]["prompt"]

    class _Obs:
        def __init__(self, *, done: bool, reward: float, metadata: dict[str, object], buyer_days: int = 40) -> None:
            self.done = done
            self.reward = reward
            self.metadata = metadata
            self.active_deal_id = "deal-1"
            self.open_deal_ids = ["deal-1"] if not done else []
            self.buyer_price = 95.0
            self.buyer_days = buyer_days
            self.current_period = 0
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
                "current_period": self.current_period,
                "total_periods": self.total_periods,
            }

    class _FakeWrapper:
        def __init__(self) -> None:
            self.done = False
            self.env_reward_total = 0.0
            self.last_observation = _Obs(done=False, reward=0.0, metadata={})

        def reset(self, **kwargs):
            self.done = False
            self.last_observation = _Obs(done=False, reward=0.0, metadata={})
            return "obs-reset"

        def accept(self, **kwargs):
            self.last_observation = _Obs(done=False, reward=0.2, metadata={}, buyer_days=40)
            self.env_reward_total += 0.2
            return "obs-accept"

        def advance_period(self):
            self.done = True
            self.last_observation = _Obs(
                done=True,
                reward=0.5,
                metadata={"termination_reason": "episode_complete", "reward_breakdown": {"total": 0.7}},
                buyer_days=40,
            )
            self.env_reward_total += 0.5
            return "obs-done"

        def summarize_episode(self):
            return EpisodeSummary(
                episode_completed=True,
                base_rl_reward=0.7,
                verifiable_reward=0.7,
                total_reward=0.7,
                tool_bonus_total=0.0,
                env_reward_total=0.7,
                success_no_default_positive_npv=True,
                average_final_payment_days=40.0,
                tool_usage_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            )

        def build_episode_log(self):
            return "fake-episode-log"

    env_factory = lambda: _FakeWrapper()

    result = _run_single_rollout_sample(
        prompt,
        model=object(),
        tokenizer=_DummyTokenizer(),
        rollout_args=args,
        env_factory=env_factory,
    )

    assert result["episode_summary"].episode_completed is True
    assert result["parsed_actions"]
    assert '"action_type"' in result["raw_completion_text"]
    assert result["termination_reason"]


def test_run_single_rollout_sample_marks_non_json_invalid_and_uses_conservative_fallback(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    def _fake_generate_completion_turn(model, tokenizer, messages, *, max_new_tokens, temperature, top_p):
        return {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4, 5],
            "logprobs": [-0.1, -0.2],
            "text": "<think>I should reason this out first</think>",
        }

    monkeypatch.setattr(trl_script, "_generate_completion_turn", _fake_generate_completion_turn)
    args = build_arg_parser().parse_args(["--max-episode-steps", "1"])
    prompt = build_training_rows(num_samples=1, total_periods=1)[0]["prompt"]

    class _Obs:
        def __init__(self, *, done: bool, reward: float, metadata: dict[str, object], buyer_days: int = 40) -> None:
            self.done = done
            self.reward = reward
            self.metadata = metadata
            self.active_deal_id = "deal-1"
            self.open_deal_ids = ["deal-1"] if not done else []
            self.buyer_price = 95.0
            self.buyer_days = buyer_days
            self.cost_threshold = 82.0
            self.liquidity_threshold = 35
            self.current_period = 0
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
            self.done = False
            self.env_reward_total = 0.0
            self.last_observation = _Obs(done=False, reward=0.0, metadata={})

        def reset(self, **kwargs):
            self.done = False
            self.last_observation = _Obs(done=False, reward=0.0, metadata={})
            return "obs-reset"

        def propose(self, **kwargs):
            self.done = True
            self.last_observation = _Obs(
                done=True,
                reward=0.1,
                metadata={"termination_reason": "fallback_propose", "reward_breakdown": {"total": 0.1}},
                buyer_days=int(kwargs.get("payment_days", 35)),
            )
            self.env_reward_total += 0.1
            return "obs-propose"

        def summarize_episode(self):
            return EpisodeSummary(
                episode_completed=True,
                base_rl_reward=0.1,
                verifiable_reward=0.1,
                total_reward=0.1,
                tool_bonus_total=0.0,
                env_reward_total=0.1,
                success_no_default_positive_npv=True,
                average_final_payment_days=35.0,
                tool_usage_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            )

        def build_episode_log(self):
            return "fake-episode-log"

    result = _run_single_rollout_sample(
        prompt,
        model=object(),
        tokenizer=_DummyTokenizer(),
        rollout_args=args,
        env_factory=lambda: _FakeWrapper(),
    )

    assert result["invalid_parse_fraction"] == 1.0
    assert result["parsed_actions"][0]["action_type"] == "propose"
    assert result["parsed_actions"][0]["reason"] == "Default action after failed parse"
    assert "think" in result["raw_completion_text"]


def test_prepare_model_for_grpo_refuses_missing_peft(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    def _missing_peft():
        raise ImportError("missing peft")

    monkeypatch.setattr(trl_script, "_require_peft_components", _missing_peft)
    try:
        prepare_model_for_grpo(object())
    except ImportError as exc:
        assert "missing peft" in str(exc)
    else:
        raise AssertionError("prepare_model_for_grpo should require peft")


def test_prepare_model_for_grpo_disables_incompatible_torchao_and_retries(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    calls = {"count": 0, "disabled": False}

    class _FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def _fake_disable_torchao() -> None:
        calls["disabled"] = True

    def _fake_get_peft_model(model, config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ImportError("Found an incompatible version of torchao")
        return {"wrapped": model, "config": config}

    monkeypatch.setattr(trl_script, "_torchao_version", lambda: "0.10.0")
    monkeypatch.setattr(trl_script, "_disable_torchao_in_peft", _fake_disable_torchao)
    monkeypatch.setattr(trl_script, "_require_peft_components", lambda: (_FakeLoraConfig, _fake_get_peft_model))

    model = prepare_model_for_grpo(object())

    assert calls["disabled"] is True
    assert calls["count"] == 2
    assert isinstance(model, dict)


def test_trl_main_passes_rollout_func_and_not_environment_factory(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    captured: dict[str, object] = {}

    monkeypatch.setattr(trl_script, "build_curriculum_manager_from_args", lambda args: None)
    monkeypatch.setattr(trl_script, "build_opponent_manager_from_args", lambda args: None)
    monkeypatch.setattr(trl_script, "load_rubric_scorer", lambda spec, enable_rubrics=False: None)
    monkeypatch.setattr(trl_script, "build_dataset", lambda rows: rows)
    monkeypatch.setattr(trl_script, "build_metrics_callback", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        trl_script,
        "load_training_model_and_tokenizer",
        lambda model_name: (object(), _ensure_grpo_response_schema(_DummyTokenizer())),
    )

    fake_transformers = types.ModuleType("transformers")

    class _FakeTrainerCallback:
        pass

    fake_transformers.TrainerCallback = _FakeTrainerCallback

    fake_trl = types.ModuleType("trl")

    class _FakeGRPOConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeGRPOTrainer:
        def __init__(self, rollout_func=None, **kwargs) -> None:
            captured.update(kwargs)
            captured["rollout_func"] = rollout_func

        def train(self) -> None:
            captured["trained"] = True

        def save_model(self, output_dir: str) -> None:
            captured["saved_to"] = output_dir

    fake_trl.GRPOConfig = _FakeGRPOConfig
    fake_trl.GRPOTrainer = _FakeGRPOTrainer

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    assert trl_main(["--num-samples", "1"]) == 0
    assert callable(captured["rollout_func"])
    assert "environment_factory" not in captured
    assert captured["trained"] is True


def test_build_training_session_returns_canonical_explicit_rollout_bundle(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    monkeypatch.setattr(trl_script, "build_curriculum_manager_from_args", lambda args: None)
    monkeypatch.setattr(trl_script, "build_opponent_manager_from_args", lambda args: None)
    monkeypatch.setattr(trl_script, "load_rubric_scorer", lambda spec, enable_rubrics=False: None)
    monkeypatch.setattr(trl_script, "build_dataset", lambda rows: rows)
    monkeypatch.setattr(
        trl_script,
        "load_training_model_and_tokenizer",
        lambda model_name: (object(), _ensure_grpo_response_schema(_DummyTokenizer())),
    )
    monkeypatch.setattr(trl_script, "build_metrics_callback", lambda *args, **kwargs: object())

    fake_transformers = types.ModuleType("transformers")

    class _FakeTrainerCallback:
        pass

    fake_transformers.TrainerCallback = _FakeTrainerCallback

    fake_trl = types.ModuleType("trl")

    class _FakeGRPOConfig:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.kwargs = kwargs

    class _FakeGRPOTrainer:
        def __init__(self, rollout_func=None, **kwargs) -> None:
            self.kwargs = kwargs

    fake_trl.GRPOConfig = _FakeGRPOConfig
    fake_trl.GRPOTrainer = _FakeGRPOTrainer

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    session = build_training_session(
        make_training_args(
            num_samples=1,
            output_dir=str(_workspace_tmp_dir("training_session")),
        )
    )

    assert callable(session["rollout_func"])
    assert callable(session["reward_funcs"])
    assert session["summary_buffer"].__class__ is EpisodeSummaryBuffer
    assert session["training_args"].save_only_model is True
    assert getattr(session["tokenizer"], "response_schema", None) is not None
    assert session["final_checkpoint_path"].name == "final-grpo-model"
    assert set(session["trainer_kwargs"]) >= {
        "model",
        "processing_class",
        "reward_funcs",
        "train_dataset",
        "args",
        "rollout_func",
        "callbacks",
    }


def test_create_trainer_and_run_training_session_use_canonical_helper_path(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        trl_script,
        "build_training_session",
        lambda args: {
            "trainer_kwargs": {"model": object(), "args": object(), "rollout_func": object()},
            "tokenizer": _DummyTokenizerSaver(),
            "final_checkpoint_path": _workspace_tmp_dir("run_training_session") / "final-grpo-model",
        },
    )

    class _FakeTrainer:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def train(self) -> None:
            captured["trained"] = True

        def save_model(self, output_dir: str) -> None:
            captured["saved_to"] = output_dir

    monkeypatch.setattr(trl_script, "create_trainer", lambda session: _FakeTrainer(**session["trainer_kwargs"]))

    result = run_training_session(make_training_args())

    assert captured["trained"] is True
    assert "checkpoint_path" in result
    assert result["session"]["final_checkpoint_path"].name == "final-grpo-model"


def test_require_rollout_func_support_rejects_older_trl_signature() -> None:
    class _OldTrainer:
        def __init__(self, model=None, args=None, callbacks=None) -> None:
            self.model = model
            self.args = args
            self.callbacks = callbacks

    try:
        _require_rollout_func_support(_OldTrainer)
    except ImportError as exc:
        assert "rollout_func" in str(exc)
    else:
        raise AssertionError("Older TRL signatures should be rejected before model loading.")


def test_metrics_callback_saves_reward_curve_without_matplotlib_dependency(monkeypatch) -> None:
    tmp_path = _workspace_tmp_dir("metrics_callback")
    summary_buffer = EpisodeSummaryBuffer()
    callback = build_metrics_callback(
        summary_buffer,
        object,
        curriculum=None,
        build_preference_dataset=False,
        scorer=None,
        output_dir=str(tmp_path),
    )

    def _fake_save(reward_curve, success_curve, *, output_dir):
        path = Path(output_dir) / "reward_curve.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr("rl.train_grpo_trl._save_reward_curve_plot", _fake_save)

    control = SimpleNamespace()
    args = SimpleNamespace(output_dir=str(tmp_path))
    for index, reward in enumerate((0.2, 0.5), start=1):
        summary_buffer.append(
            EpisodeSummary(
                episode_completed=True,
                base_rl_reward=reward,
                verifiable_reward=reward,
                total_reward=reward,
                tool_bonus_total=0.01,
                env_reward_total=reward,
                success_no_default_positive_npv=True,
                average_final_payment_days=40.0,
                tool_usage_count=1,
                resolved_deal_count=2,
                defaulted_sme_count=0,
            )
        )
        callback.on_log(args, SimpleNamespace(global_step=index), control, logs={})

    callback.on_train_end(args, SimpleNamespace(global_step=2), control)


def test_reward_monitor_callback_exposes_trainer_lifecycle_methods() -> None:
    callback = RewardMonitorCallback()

    assert hasattr(callback, "on_init_end")
    assert callable(callback.on_init_end)
    assert hasattr(callback, "on_log")


def test_colab_notebook_defaults_to_qwen3_thin_wrapper() -> None:
    source = _load_notebook_source()

    assert 'MODEL_NAME = "Qwen/Qwen3-0.6B"' in source
    assert "build_training_session" in source
    assert 'trl==0.29.0' in source


def test_colab_notebook_drops_legacy_generate_rollout_completions_and_environment_factory() -> None:
    source = _load_notebook_source()

    assert "generate_rollout_completions" not in source
    assert "environment_factory=" not in source


def test_colab_notebook_requires_peft_and_no_inline_training_loop_fallback() -> None:
    source = _load_notebook_source()

    assert "full-parameter training fallback" not in source
    assert "def rollout_func(" not in source
    assert "build_training_session" in source


def test_colab_notebook_baseline_handles_dict_or_attr_summary_shape() -> None:
    source = _load_notebook_source()

    assert "_baseline_success" in source
    assert "summary.get(\"success_no_default_positive_npv\", False)" in source
    assert "run[\"summary\"].success_no_default_positive_npv" not in source


def test_colab_notebook_uses_shared_dashboard_helper_and_saved_artifacts() -> None:
    source = _load_notebook_source()

    assert 'RUN_PROFILE = "submission"' in source
    assert "save_training_dashboard" in source
    assert 'training_dashboard.png' in source
    assert 'policy_comparison.png' in source
    assert 'eval_summary.json' in source


def test_colab_notebook_evaluates_base_and_trained_policies() -> None:
    source = _load_notebook_source()

    assert 'policy="base"' in source
    assert 'policy="trained"' in source
    assert "evaluate_before_after_policies" in source
    assert "submission_ready" in source


def test_import_trl_grpo_symbols_installs_optional_stubs_on_demand(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script
    import builtins

    real_import = builtins.__import__
    state = {"stub_visible": False, "attempts": 0}

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "trl" and "GRPOConfig" in fromlist and "GRPOTrainer" in fromlist:
            state["attempts"] += 1
            if "mergekit" not in sys.modules:
                raise RuntimeError("No module named 'mergekit'")
            if "llm_blender" not in sys.modules:
                raise RuntimeError("No module named 'llm_blender'")
            if "weave" not in sys.modules:
                raise RuntimeError("No module named 'weave'")
            state["stub_visible"] = True
            module = types.ModuleType("trl")
            module.GRPOConfig = type("GRPOConfig", (), {})
            module.GRPOTrainer = type("GRPOTrainer", (), {})
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    for module_name in ("mergekit", "mergekit.config", "mergekit.merge", "llm_blender", "weave"):
        sys.modules.pop(module_name, None)

    GRPOConfig, GRPOTrainer = trl_script._import_trl_grpo_symbols()

    assert GRPOConfig.__name__ == "GRPOConfig"
    assert GRPOTrainer.__name__ == "GRPOTrainer"
    assert state["stub_visible"] is True
    assert state["attempts"] >= 4
    assert "weave.trace.context" in sys.modules
    assert hasattr(sys.modules["weave"], "__path__")
    assert callable(getattr(sys.modules["weave.trace.context"], "weave_client_context", None))


def test_metrics_callback_plot_failures_do_not_raise(monkeypatch) -> None:
    tmp_path = _workspace_tmp_dir("metrics_callback_failure")
    summary_buffer = EpisodeSummaryBuffer()
    callback = build_metrics_callback(
        summary_buffer,
        object,
        curriculum=None,
        build_preference_dataset=False,
        scorer=None,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        "rl.train_grpo_trl._save_reward_curve_plot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )

    control = SimpleNamespace()
    args = SimpleNamespace(output_dir=str(tmp_path))
    for index, reward in enumerate((0.2, 0.4), start=1):
        summary_buffer.append(
            EpisodeSummary(
                episode_completed=True,
                base_rl_reward=reward,
                verifiable_reward=reward,
                total_reward=reward,
                tool_bonus_total=0.0,
                env_reward_total=reward,
                success_no_default_positive_npv=True,
                average_final_payment_days=45.0,
                tool_usage_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            )
        )
        callback.on_log(args, SimpleNamespace(global_step=index), control, logs={})

    callback.on_train_end(args, SimpleNamespace(global_step=2), control)
