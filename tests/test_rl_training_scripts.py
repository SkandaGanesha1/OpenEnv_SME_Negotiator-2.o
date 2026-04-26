"""Stage 5 dry-run and helper tests for RL training scripts."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.curriculum import CurriculumManager
from rl.opponents import OpponentPolicyManager
from rl.episode_logging import EpisodeSummary
from rl.bridge import InProcessEnvWrapper, get_exposed_environment_method_names
from rl.train_grpo_trl import (
    _align_trl_environment_batch,
    _build_environment_tool_dicts,
    _infer_expected_environment_batch_size,
    _default_vllm_max_model_length,
    _ensure_grpo_response_schema,
    _generate_completion_turn,
    _render_chat_prompt,
    _strip_training_row_metadata,
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
    _require_environment_factory_support,
    _require_rollout_func_support,
    run_training_session,
    save_training_session,
)
from rl.monitoring import RewardMonitorCallback
from rl.train_grpo_unsloth import (
    build_grpo_config_kwargs as unsloth_build_grpo_config_kwargs,
    main as unsloth_main,
)
from rl.train_grpo_liquidity import (
    build_canonical_training_args as build_liquidity_canonical_training_args,
    main as liquidity_main,
    make_training_args as make_liquidity_training_args,
    resolve_profile_config as resolve_liquidity_profile_config,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        return {"input_ids": [[10, 11, 12]]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "decoded"


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


class _NoGenerateModel:
    device = None

    def generate(self, **kwargs):
        raise AssertionError("training model.generate should not be used when vLLM LLM is available")


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


def test_simple_script_dry_run_works(capsys) -> None:
    assert liquidity_main(["--dry-run", "--profile", "tiny"]) == 0
    captured = capsys.readouterr()
    assert '"mode": "dry-run"' in captured.out
    assert '"smoke_test"' in captured.out
    assert '"profile": "tiny"' in captured.out


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


def test_simple_script_profile_defaults_and_overrides_resolve_cleanly() -> None:
    defaults = make_liquidity_training_args()
    resolved_defaults = resolve_liquidity_profile_config(defaults)
    assert resolved_defaults["num_samples"] == 8
    assert resolved_defaults["max_steps"] == 4
    canonical_defaults = build_liquidity_canonical_training_args(defaults)
    assert canonical_defaults.use_vllm is True
    assert canonical_defaults.vllm_gpu_memory_utilization == 0.5
    assert canonical_defaults.vllm_max_model_length is None

    updated = make_liquidity_training_args(profile="standard", num_samples=20, learning_rate=1e-5)
    resolved_updated = resolve_liquidity_profile_config(updated)
    canonical = build_liquidity_canonical_training_args(updated)

    assert resolved_updated["num_samples"] == 20
    assert resolved_updated["learning_rate"] == 1e-5
    assert canonical.num_samples == 20
    assert canonical.learning_rate == 1e-5
    assert canonical.max_episode_steps == 12


def test_default_vllm_max_model_length_matches_small_grpo_recipe() -> None:
    assert _default_vllm_max_model_length(max_prompt_length=1024, max_completion_length=256) == 2048
    assert _default_vllm_max_model_length(max_prompt_length=2048, max_completion_length=512) == 2816


def test_generate_completion_turn_uses_vllm_llm_when_available(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeVllmLLM:
        def generate(self, prompts, sampling_params=None, use_tqdm=False):
            assert prompts == ["user: hi"]
            assert sampling_params.kwargs["max_tokens"] == 32
            assert sampling_params.kwargs["temperature"] == 0.7
            return [
                SimpleNamespace(
                    prompt_token_ids=[10, 11, 12],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[21, 22],
                            text='{"action_type":"advance_period"}',
                            logprobs=[{21: -0.2}, {22: -0.3}],
                        )
                    ],
                )
            ]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(trl_script, "_render_chat_prompt", lambda tokenizer, messages: "user: hi")

    turn = _generate_completion_turn(
        _NoGenerateModel(),
        _DummyTokenizer(),
        [{"role": "user", "content": "hi"}],
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        vllm_llm=_FakeVllmLLM(),
    )

    assert turn["prompt_ids"] == [10, 11, 12]
    assert turn["completion_ids"] == [21, 22]
    assert turn["logprobs"] == [-0.2, -0.3]
    assert turn["text"] == '{"action_type":"advance_period"}'


def test_strip_training_row_metadata_removes_embedded_row_lines() -> None:
    messages = [
        {
            "role": "user",
            "content": '[TRAINING_ROW] {"seed":1000}\n/no_think\nUse JSON only.',
        }
    ]

    cleaned = _strip_training_row_metadata(messages)

    assert cleaned == [{"role": "user", "content": "/no_think\nUse JSON only."}]


def test_build_training_rows_can_skip_embedded_row_metadata() -> None:
    rows = build_training_rows(num_samples=1, include_row_metadata=False)

    assert rows[0]["prompt"][0]["content"] == rows[0]["prompt"][0]["content"].replace("[TRAINING_ROW]", "")
    assert "[TRAINING_ROW]" not in rows[0]["prompt"][0]["content"]


def test_infer_expected_environment_batch_size_handles_dict_and_list_inputs() -> None:
    assert _infer_expected_environment_batch_size({"prompt": ["a", "b", "c"]}) == 3
    assert _infer_expected_environment_batch_size([{"prompt": "a"}, {"prompt": "b"}]) == 2
    assert _infer_expected_environment_batch_size({"task_name": ["a", "b"]}) == 2


def test_align_trl_environment_batch_resizes_environment_and_tool_lists() -> None:
    class _Env:
        def reset(self, **kwargs):
            return "obs"

        def propose(self, price: float, payment_days: int) -> str:
            return "ok"

    created = {"count": 0}

    def _factory():
        created["count"] += 1
        return _Env()

    trainer = SimpleNamespace(
        environments=[_factory()],
        _sync_tool_dicts=[{"propose": trainer} for trainer in []],
        _async_tool_dicts=[{}],
        _canonical_environment_factory=_factory,
    )
    trainer._sync_tool_dicts = [{"propose": trainer.environments[0].propose}]

    _align_trl_environment_batch(trainer, 3)

    assert len(trainer.environments) == 3
    assert len(trainer._sync_tool_dicts) == 3
    assert len(trainer._async_tool_dicts) == 3
    assert created["count"] == 3


def test_liquidity_runner_rejects_legacy_backend_for_notebook_training() -> None:
    import rl.train_grpo_liquidity as liquidity_script

    with pytest.raises(RuntimeError, match="runtime_backend='environment'"):
        liquidity_script.run_training(
            liquidity_script.make_training_args(
                profile="tiny",
                output_dir=str(_workspace_tmp_dir("liquidity_legacy_backend")),
                skip_smoke_test=True,
                runtime_backend="legacy",
            )
        )


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
    summary_buffer = EpisodeSummaryBuffer()
    reward_func = make_reward_function(
        rubric_scorer=lambda episode_log: {"rubric": 0.5},
        rubric_weight=0.2,
        summary_buffer=summary_buffer,
    )
    rewards = reward_func([_DummyEnv(), _DummyEnv()])
    assert rewards == [0.85, 0.85]
    assert len(summary_buffer.export_records()) == 2


def test_strict_json_payload_extracts_embedded_object_after_think_tags() -> None:
    payload = _strict_json_payload(
        '<think>Reason about the contract first</think> {"action_type":"propose","price":1000,"payment_days":30}'
    )

    assert payload == {"action_type": "propose", "price": 1000, "payment_days": 30}


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
    reward_func = make_reward_function(runtime_backend="legacy", pending_rollout_buffer=pending_buffer)

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
    reward_func = make_reward_function(runtime_backend="legacy", pending_rollout_buffer=pending_buffer)
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
    reward_func = make_reward_function(runtime_backend="legacy", pending_rollout_buffer=pending_buffer)
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
    reward_func = make_reward_function(runtime_backend="legacy", pending_rollout_buffer=pending_buffer)
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
    reward_func = make_reward_function(runtime_backend="legacy", pending_rollout_buffer=pending_buffer)
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
        runtime_backend="legacy",
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


def test_score_prompt_completion_via_environment_relaxes_non_json_but_executable_actions() -> None:
    class _Obs:
        def __init__(self, *, done: bool, reward: float, metadata: dict[str, object], buyer_days: int = 40) -> None:
            self.done = done
            self.reward = reward
            self.metadata = metadata
            self.active_deal_id = "deal-1"
            self.open_deal_ids = ["deal-1"] if not done else []
            self.buyer_price = 95.0
            self.buyer_days = buyer_days
            self.cost_threshold = 80.0
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

    class _Wrapper:
        def __init__(self) -> None:
            self.done = False
            self.last_observation = _Obs(done=False, reward=0.0, metadata={})

        def reset(self, **kwargs):
            self.done = False
            self.last_observation = _Obs(done=False, reward=0.0, metadata={})
            return "obs-reset"

        def propose(self, **kwargs):
            self.done = True
            self.last_observation = _Obs(
                done=True,
                reward=0.3,
                metadata={"termination_reason": "episode_complete"},
                buyer_days=int(kwargs.get("payment_days", 30)),
            )
            return "obs-propose"

        def summarize_episode(self):
            return EpisodeSummary(
                episode_completed=True,
                base_rl_reward=0.3,
                verifiable_reward=0.3,
                total_reward=0.3,
                tool_bonus_total=0.0,
                env_reward_total=0.3,
                success_no_default_positive_npv=True,
                average_final_payment_days=30.0,
                tool_usage_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            )

        def build_episode_log(self):
            return "wrapper-log"

    result = _score_prompt_completion_via_environment(
        "prompt-a",
        '{"action_type":"proposal","price":1000,"payment_days":30}',
        fallback_args=make_training_args(),
        env_factory=_Wrapper,
    )

    assert result["termination_reason"] == "prompt_env_relaxed_episode_complete"
    assert result["invalid_parse_fraction"] == 1.0
    assert result["episode_summary"].total_reward == 0.3
    assert result["parsed_actions"][0]["action_type"] == "propose"


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


def test_in_process_env_wrapper_public_tool_methods_are_typed_and_documented() -> None:
    for method_name in (
        "reset",
        "propose",
        "accept",
        "reject",
        "advance_period",
        "query_treds",
        "check_compliance",
        "run_cashflow_sim",
        "simulate_plan",
        "summarize_episode",
        "build_episode_log",
    ):
        method = getattr(InProcessEnvWrapper, method_name)
        annotations = getattr(method, "__annotations__", {})
        assert annotations, f"{method_name} should expose type hints for tool/schema inference"
        assert method.__doc__, f"{method_name} should expose a docstring for tool/schema inference"


def test_build_environment_tool_dicts_uses_canonical_allowlist_only() -> None:
    class _ExposedEnv:
        def propose(self):  # pragma: no cover - invoked via reflection only
            return "propose"

        def accept(self):
            return "accept"

        def reject(self):
            return "reject"

        def advance_period(self):
            return "advance"

        def query_treds(self):
            return "treds"

        def check_compliance(self):
            return "compliance"

        def run_cashflow_sim(self):
            return "sim"

        def simulate_plan(self):
            return "plan"

        def build_episode_log(self):
            return "helper"

        def compute_final_reward(self):
            return 0.5

        def summarize_episode(self):
            return {}

    sync_tools, async_tools = _build_environment_tool_dicts(_ExposedEnv())

    assert async_tools == {}
    assert list(sync_tools) == list(get_exposed_environment_method_names())
    assert "build_episode_log" not in sync_tools
    assert "compute_final_reward" not in sync_tools
    assert "summarize_episode" not in sync_tools


def test_build_grpo_config_kwargs_uses_training_log_backend_env(monkeypatch) -> None:
    args = build_arg_parser().parse_args([])

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "tensorboard")
    kwargs = build_grpo_config_kwargs(args)
    assert kwargs["report_to"] == "tensorboard"
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 1.0
    assert "generation_batch_size" not in kwargs
    assert kwargs["save_only_model"] is True
    assert kwargs["save_steps"] == 1000

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "unsupported")
    assert build_grpo_config_kwargs(args)["report_to"] == "none"


def test_build_grpo_config_kwargs_only_passes_generation_batch_size_when_explicit() -> None:
    args = build_arg_parser().parse_args(["--generation-batch-size", "16"])

    kwargs = build_grpo_config_kwargs(args)

    assert kwargs["generation_batch_size"] == 16


def test_build_grpo_config_kwargs_sets_notebook_safe_vllm_limits() -> None:
    args = build_arg_parser().parse_args(
        [
            "--use-vllm",
            "--vllm-gpu-memory-utilization",
            "0.45",
        ]
    )

    kwargs = build_grpo_config_kwargs(args)

    assert kwargs["vllm_gpu_memory_utilization"] == 0.45
    assert kwargs["vllm_max_model_length"] == 2048
    assert kwargs["vllm_importance_sampling_correction"] is False


def test_build_grpo_config_kwargs_plumbs_scale_rewards_and_completion_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--scale-rewards",
            "none",
            "--mask-truncated-completions",
            "--no-log-completions",
        ]
    )

    kwargs = build_grpo_config_kwargs(args)

    assert kwargs["scale_rewards"] == "none"
    assert kwargs["mask_truncated_completions"] is True
    assert kwargs["log_completions"] is False


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


def test_liquidity_canonical_args_force_environment_backend() -> None:
    canonical_args = build_liquidity_canonical_training_args(
        make_liquidity_training_args(runtime_backend="legacy")
    )

    assert canonical_args.runtime_backend == "environment"
    assert canonical_args.gradient_accumulation_steps == 4


def test_build_grpo_config_kwargs_rejects_invalid_implicit_generation_batch_size() -> None:
    args = build_arg_parser().parse_args(
        [
            "--num-generations",
            "4",
            "--gradient-accumulation-steps",
            "2",
            "--per-device-train-batch-size",
            "1",
        ]
    )

    with pytest.raises(ValueError, match="incompatible generation_batch_size"):
        build_grpo_config_kwargs(args)


def test_patch_additional_chat_templates_404_returns_empty_template_list(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    import huggingface_hub.utils as hf_hub_utils

    error_type = getattr(hf_hub_utils, "RemoteEntryNotFoundError", None) or getattr(
        hf_hub_utils, "EntryNotFoundError", None
    )
    assert error_type is not None

    def _raising_list_repo_templates(*args, **kwargs):
        raise error_type("missing additional_chat_templates")

    fake_transformers_pkg = types.ModuleType("transformers")
    fake_transformers_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_transformers_utils_pkg = types.ModuleType("transformers.utils")
    fake_transformers_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_transformers_hub = types.ModuleType("transformers.utils.hub")
    fake_transformers_hub.list_repo_templates = _raising_list_repo_templates
    fake_tokenization_utils_base = types.ModuleType("transformers.tokenization_utils_base")
    fake_tokenization_utils_base.list_repo_templates = _raising_list_repo_templates

    fake_transformers_pkg.utils = fake_transformers_utils_pkg
    fake_transformers_pkg.tokenization_utils_base = fake_tokenization_utils_base
    fake_transformers_utils_pkg.hub = fake_transformers_hub

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers_pkg)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_transformers_utils_pkg)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", fake_transformers_hub)
    monkeypatch.setitem(sys.modules, "transformers.tokenization_utils_base", fake_tokenization_utils_base)

    trl_script._patch_additional_chat_templates_404()

    assert fake_transformers_hub.list_repo_templates() == []
    assert fake_tokenization_utils_base.list_repo_templates() == []


def test_patch_vllm_notebook_stdout_replaces_suppressors_and_sets_debug(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    class _StdoutWithoutFileno:
        def fileno(self):
            raise OSError("no fileno")

    fake_system_utils = types.ModuleType("vllm.utils.system_utils")
    fake_parallel_state = types.ModuleType("vllm.distributed.parallel_state")

    @contextlib.contextmanager
    def _original_suppress_stdout():
        raise AssertionError("original suppress_stdout should be replaced")
        yield

    fake_system_utils.suppress_stdout = _original_suppress_stdout
    fake_parallel_state.suppress_stdout = _original_suppress_stdout

    monkeypatch.setattr(sys, "stdout", _StdoutWithoutFileno())
    monkeypatch.setitem(sys.modules, "vllm.utils.system_utils", fake_system_utils)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", fake_parallel_state)
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)

    trl_script._patch_vllm_notebook_stdout()

    assert os.environ["VLLM_LOGGING_LEVEL"] == "DEBUG"
    with fake_system_utils.suppress_stdout():
        pass
    with fake_parallel_state.suppress_stdout():
        pass


def test_patch_vllm_attention_backend_prefers_triton_on_pre_ampere(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda index=0: (7, 5),
        )
    )

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    trl_script._patch_vllm_attention_backend()

    assert os.environ["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"


def test_patch_vllm_attention_backend_respects_existing_override(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda index=0: (7, 5),
        )
    )

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    trl_script._patch_vllm_attention_backend()

    assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"


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
    assert result["logprobs"] == [[[-0.1]], [[-0.1]], [[-0.1]]]
    assert result["logprob_token_ids"] == [[[10]], [[11]], [[12]]]
    assert result["termination_reasons"] == ["done", "done", "done"]
    assert len(result["raw_completion_texts"]) == 3


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
    assert result["prompt_ids"] == [10, 11, 12]
    assert result["completion_ids"] == [10, 11, 12]
    assert '"action_type":"accept"' in result["completion_signature_text"]
    assert '"action_type":"advance_period"' in result["completion_signature_text"]
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


def test_trl_main_passes_environment_factory_and_not_rollout_func(monkeypatch) -> None:
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
        def __init__(self, environment_factory=None, **kwargs) -> None:
            captured.update(kwargs)
            captured["environment_factory"] = environment_factory

        def train(self) -> None:
            captured["trained"] = True

        def save_model(self, output_dir: str) -> None:
            captured["saved_to"] = output_dir

    fake_trl.GRPOConfig = _FakeGRPOConfig
    fake_trl.GRPOTrainer = _FakeGRPOTrainer

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    assert trl_main(["--num-samples", "1"]) == 0
    assert callable(captured["environment_factory"])
    assert "rollout_func" not in captured
    assert captured["trained"] is True


def test_build_training_session_returns_canonical_environment_bundle(monkeypatch) -> None:
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
        def __init__(self, environment_factory=None, **kwargs) -> None:
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

    assert session["runtime_backend"] == "environment"
    assert session["rollout_func"] is None
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
        "environment_factory",
        "callbacks",
    }


def test_create_trainer_and_run_training_session_use_canonical_helper_path(monkeypatch) -> None:
    import rl.train_grpo_trl as trl_script

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        trl_script,
        "build_training_session",
        lambda args: {
            "runtime_backend": "environment",
            "trainer_kwargs": {"model": object(), "args": object(), "environment_factory": object()},
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
    assert result["runtime_backend"] == "environment"
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


def test_require_environment_factory_support_rejects_older_trl_signature() -> None:
    class _OldTrainer:
        def __init__(self, model=None, args=None, callbacks=None) -> None:
            self.model = model
            self.args = args
            self.callbacks = callbacks

    with pytest.raises(ImportError, match="environment_factory"):
        _require_environment_factory_support(_OldTrainer)


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


def test_colab_notebook_describes_environment_native_training_path() -> None:
    source = _load_notebook_source()

    assert "generate_rollout_completions" not in source
    assert "environment-native trainer" in source
    assert "explicit-rollout trainer" not in source


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


def test_simple_script_main_writes_manifest_with_artifact_paths(monkeypatch, capsys) -> None:
    import rl.train_grpo_liquidity as liquidity_script

    tmp_path = _workspace_tmp_dir("simple_training_script")
    checkpoint_path = tmp_path / "final-grpo-model"
    reward_curve_path = tmp_path / "reward_curve.png"
    trainer_reward_log_path = tmp_path / "reward_log.json"
    episode_reward_log_path = tmp_path / "episode_reward_log.json"
    training_dashboard_path = tmp_path / "training_dashboard.png"
    manifest_path = tmp_path / "run_manifest.json"

    monkeypatch.setattr(
        liquidity_script,
        "save_training_dashboard",
        lambda trainer, *, output_dir: {
            "training_dashboard_path": str(training_dashboard_path),
            "reward_curve_path": str(reward_curve_path),
            "reward_log_path": str(trainer_reward_log_path),
            "history_points": 3,
            "zero_variance_warning": False,
            "training_trustworthy": True,
            "trust_failures": [],
            "trust_metrics": {
                "training_trustworthy": True,
                "median_reward_std": 0.2,
                "median_unique_completion_count": 3.0,
                "median_identical_terminal_fraction": 0.5,
                "mean_total_reward": 0.2,
                "mean_verifiable_reward": 0.2,
            },
        },
    )
    monkeypatch.setattr(liquidity_script, "_require_vllm_installed", lambda: None)
    monkeypatch.setattr(
        liquidity_script,
        "run_canonical_training_session",
        lambda args: {
            "trainer": SimpleNamespace(state=SimpleNamespace(log_history=[])),
            "checkpoint_path": checkpoint_path,
            "runtime_backend": "environment",
            "exposed_tool_names": list(get_exposed_environment_method_names()),
            "episode_reward_history": [
                {
                    "episode": 1,
                    "training_reward": 0.25,
                    "total_reward": 0.2,
                    "verifiable_reward": 0.2,
                    "base_rl_reward": 0.2,
                    "success_no_default_positive_npv": True,
                }
            ],
        },
    )
    monkeypatch.setattr(
        liquidity_script,
        "evaluate_before_after_policies",
        lambda **kwargs: {
            "summary": {
                "metadata": {
                    "trained_beats_base_without_extra_defaults": True,
                    "submission_ready": True,
                },
                "policies": {
                    "base": {"mean_verifiable_reward": 0.2, "default_rate": 0.1},
                    "trained": {"mean_verifiable_reward": 0.3, "default_rate": 0.1},
                    "heuristic": {"mean_verifiable_reward": 0.25, "default_rate": 0.1},
                },
            },
            "eval_summary_path": str(tmp_path / "eval_summary.json"),
            "policy_comparison_path": str(tmp_path / "policy_comparison.png"),
        },
    )
    monkeypatch.setattr(liquidity_script, "plot_rewards", lambda reward_log, output_path: Path(output_path))
    monkeypatch.setattr(
        liquidity_script,
        "save_run_manifest",
        lambda manifest, *, output_dir, filename="run_manifest.json": str(manifest_path),
    )

    assert liquidity_main(["--profile", "tiny", "--output-dir", str(tmp_path), "--skip-smoke-test"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["training"]["checkpoint_path"] == str(checkpoint_path.resolve())
    assert payload["training"]["reward_log_path"] == str(episode_reward_log_path.resolve())
    assert payload["training"]["episode_reward_log_path"] == str(episode_reward_log_path.resolve())
    assert payload["training"]["trainer_reward_log_path"] == str(trainer_reward_log_path.resolve())
    assert payload["training"]["runtime_backend"] == "environment"
    assert payload["training"]["environment_backend_valid"] is True
    assert payload["training"]["training_trustworthy"] is True
    assert payload["training"]["trust_failures"] == []
    assert payload["training"]["exposed_tool_names"] == list(get_exposed_environment_method_names())
    assert payload["training"]["trust_metrics"]["median_reward_std"] == 0.2
    assert payload["training"]["median_reward_std"] == 0.2
    assert payload["training"]["median_unique_completion_count"] == 3.0
    assert payload["eval_summary"]["trained_beats_base_without_extra_defaults"] is True
    assert set(payload["eval_summary"]["tasks"]) == {
        "liquidity-stress-medium",
        "liquidity-correlation-hard",
    }
    assert payload["manifest_path"] == str(manifest_path)


def test_simple_script_main_strict_trustworthiness_returns_nonzero_on_failure(monkeypatch) -> None:
    import rl.train_grpo_liquidity as liquidity_script

    monkeypatch.setattr(
        liquidity_script,
        "run_training",
        lambda args: {
            "training": {
                "training_trustworthy": False,
                "trust_failures": ["zero_variance_warning"],
            }
        },
    )

    assert liquidity_main(["--strict-trustworthiness"]) == 2


def test_simple_script_run_training_rejects_non_environment_backend(monkeypatch) -> None:
    import rl.train_grpo_liquidity as liquidity_script

    tmp_path = _workspace_tmp_dir("simple_training_backend_failure")

    with pytest.raises(RuntimeError, match="runtime_backend='environment'"):
        liquidity_script.run_training(
            liquidity_script.make_training_args(
                profile="tiny",
                output_dir=str(tmp_path),
                skip_smoke_test=True,
                runtime_backend="legacy",
            )
        )
