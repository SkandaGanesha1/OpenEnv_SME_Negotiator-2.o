#!/usr/bin/env python3
"""Canonical TRL GRPO training entrypoint for the liquidity environment.

Install with optional extras such as:
    pip install -e ".[rl]"
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Optional

from rl.bridge import (
    build_action_contract_text,
    execute_action,
    format_observation,
    make_environment_factory,
    parse_action,
)
from rl.curriculum import CurriculumManager, DEFAULT_CURRICULUM_LEVELS
from rl.episode_logging import EpisodeSummary, combine_rewards
from rl.opponents import OpponentPolicyManager
from rl.rubrics import persona_reward
from rl.self_rewarding_dpo import build_preference_examples, write_preference_dataset
from sme_negotiator_env.prompting import conservative_default_action

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = (
    "/no_think\n"
    "You are an SME treasury agent operating a deterministic long-horizon liquidity workflow. "
    "Maximize the final verifiable reward, preserve positive NPV, use tools only when they improve the current state, "
    "and finish the episode without default. "
    + build_action_contract_text()
)
TRAINING_ROW_PREFIX = "[TRAINING_ROW]"
ROLL_OUT_USER_TURN = (
    "/no_think\n"
    "Current observation:\n{observation}\n\n"
    "Choose the single best next action for this exact state. "
    + build_action_contract_text()
)


def _training_log_backend(env: Optional[dict[str, str]] = None) -> str:
    source = os.environ if env is None else env
    value = str(source.get("TRAINING_LOG_BACKEND", "none") or "none").strip().lower()
    if value in {"wandb", "tensorboard", "none"}:
        return value
    return "none"


def _save_reward_curve_plot(
    reward_curve: list[float],
    success_curve: list[float],
    *,
    output_dir: str,
) -> Path:
    if len(reward_curve) < 2:
        raise ValueError("Need at least two reward points to save a curve.")

    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / "reward_curve.png"

    fig, (ax_reward, ax_success) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax_reward.plot(reward_curve, label="avg_total_reward")
    ax_reward.set_ylabel("Avg Total Reward")
    ax_reward.legend()

    ax_success.plot(success_curve, label="success_rate", color="green")
    ax_success.set_ylabel("Success Rate")
    ax_success.set_xlabel("Log Step")
    ax_success.legend()

    fig.suptitle("SME Negotiator - GRPO Training Curves")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=120)
    plt.close(fig)
    return figure_path


@dataclass
class EpisodeSummaryBuffer:
    """Mutable buffer shared by reward functions, callbacks, and DPO export."""

    items: list[EpisodeSummary] = field(default_factory=list)
    episode_logs: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        summary: EpisodeSummary,
        episode_log: Optional[str] = None,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> None:
        self.items.append(summary)
        if episode_log is not None:
            self.episode_logs.append(episode_log)
        if diagnostics is not None:
            self.diagnostics.append(dict(diagnostics))

    def drain(self) -> tuple[list[EpisodeSummary], list[str], list[dict[str, Any]]]:
        summaries = list(self.items)
        episode_logs = list(self.episode_logs)
        diagnostics = list(self.diagnostics)
        self.items.clear()
        self.episode_logs.clear()
        self.diagnostics.clear()
        return summaries, episode_logs, diagnostics


@dataclass
class PendingRolloutBuffer:
    """Bridge rollout_func outputs into reward_func calls using signatures first."""

    items: list[dict[str, Any]] = field(default_factory=list)
    signature_queues: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def extend(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            stored = dict(record)
            self.items.append(stored)
            signature = _pending_rollout_record_signature(stored)
            if signature is not None:
                self.signature_queues.setdefault(signature, []).append(stored)

    def take(self, count: int) -> Optional[list[dict[str, Any]]]:
        if count <= 0:
            return []
        if len(self.items) < count:
            return None
        batch = self.items[:count]
        del self.items[:count]
        for record in batch:
            self._discard_signature_record(record)
        return batch

    def take_by_signature(self, prompts: list[Any], completions: list[Any]) -> Optional[list[dict[str, Any]]]:
        if not prompts or len(prompts) != len(completions):
            return None

        requested_signatures: list[str] = []
        requested_counts: Counter[str] = Counter()
        for prompt, completion in zip(prompts, completions):
            signature = _build_pending_rollout_signature(prompt, completion)
            if signature is None:
                return None
            requested_signatures.append(signature)
            requested_counts[signature] += 1

        for signature, count in requested_counts.items():
            if len(self.signature_queues.get(signature, [])) < count:
                return None

        batch: list[dict[str, Any]] = []
        for signature in requested_signatures:
            record = self.signature_queues[signature].pop(0)
            if not self.signature_queues[signature]:
                del self.signature_queues[signature]
            self._remove_from_items(record)
            batch.append(record)
        return batch

    def _remove_from_items(self, record: dict[str, Any]) -> None:
        for index, candidate in enumerate(self.items):
            if candidate is record:
                del self.items[index]
                break

    def _discard_signature_record(self, record: dict[str, Any]) -> None:
        signature = _pending_rollout_record_signature(record)
        if signature is None:
            return
        queue = self.signature_queues.get(signature)
        if not queue:
            return
        for index, candidate in enumerate(queue):
            if candidate is record:
                del queue[index]
                break
        if not queue:
            del self.signature_queues[signature]


def _build_training_prompt_content(
    *,
    prompt: str,
    task_name: str,
    difficulty: str,
    seed: int,
    total_periods: int,
) -> str:
    row_metadata = {
        "task_name": str(task_name),
        "difficulty": str(difficulty),
        "seed": int(seed),
        "total_periods": int(total_periods),
    }
    return "\n".join(
        [
            f"{TRAINING_ROW_PREFIX} {json.dumps(row_metadata, sort_keys=True, ensure_ascii=True)}",
            str(prompt),
        ]
    )


def build_training_rows(
    *,
    prompt: str = DEFAULT_PROMPT,
    task_name: str = "liquidity-correlation-hard",
    difficulty: str = "hard",
    total_periods: int = 3,
    num_samples: int = 64,
    seed_base: int = 1000,
) -> list[dict[str, Any]]:
    """Build deterministic conversational rows for GRPO/OpenEnv training."""
    rows: list[dict[str, Any]] = []
    for index in range(int(num_samples)):
        seed = int(seed_base) + index
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": _build_training_prompt_content(
                            prompt=prompt,
                            task_name=task_name,
                            difficulty=difficulty,
                            seed=seed,
                            total_periods=total_periods,
                        ),
                    }
                ],
                "task_name": str(task_name),
                "difficulty": str(difficulty),
                "seed": seed,
                "total_periods": int(total_periods),
            }
        )
    return rows


def build_dataset(rows: list[dict[str, Any]]):
    """Create a Hugging Face dataset from row dictionaries."""
    from datasets import Dataset

    return Dataset.from_list(rows)


def configure_tokenizer(tokenizer: Any) -> Any:
    """Set left padding and a pad-token fallback for GRPO generation."""
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    return tokenizer


def _ensure_grpo_response_schema(tokenizer: Any) -> Any:
    """Attach a response schema when TRL provides one."""
    if getattr(tokenizer, "response_schema", None) is not None:
        return tokenizer

    try:
        from trl.chat_template_utils import add_response_schema

        return add_response_schema(tokenizer)
    except (ImportError, AttributeError):
        pass

    try:
        from trl.chat_template_utils import qwen3_schema

        tokenizer.response_schema = qwen3_schema
    except (ImportError, AttributeError):
        tokenizer.response_schema = {
            "x-regex": r"^(?P<content>.*?)(?:<\|im_end\|>|$)",
            "type": "object",
            "properties": {
                "role": {"const": "assistant"},
                "content": {"type": "string"},
            },
        }
    return tokenizer


def configure_rollout_tokenizer(tokenizer: Any) -> Any:
    """Apply tokenizer settings required for explicit multi-step rollouts."""
    tokenizer = _ensure_grpo_response_schema(configure_tokenizer(tokenizer))
    setattr(tokenizer, "training_chat_template", None)
    try:
        from trl.chat_template_utils import get_training_chat_template

        chat_template = get_training_chat_template(tokenizer)
        if chat_template is not None:
            tokenizer.training_chat_template = chat_template
    except (ImportError, AttributeError):
        pass
    return tokenizer


def _cuda_training_dtype() -> Any:
    """Return a safe dtype for CUDA training when torch is available."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _require_peft_components() -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for canonical GRPO training. Install optional RL extras with: pip install -e \".[rl]\""
        ) from exc
    return LoraConfig, get_peft_model


def _parse_version_components(raw: str) -> tuple[int, ...]:
    """Return numeric version components for a best-effort comparison."""
    parts = re.findall(r"\d+", str(raw))
    return tuple(int(part) for part in parts[:4])


def _torchao_version() -> Optional[str]:
    """Return the installed torchao version when present."""
    try:
        return importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return None


def _disable_torchao_in_peft() -> None:
    """Force PEFT to treat torchao as unavailable for LoRA injection."""
    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available = lambda: False
    except ImportError:
        pass

    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass


def _maybe_disable_incompatible_torchao() -> None:
    """Disable torchao integration when an unsupported version is installed."""
    version = _torchao_version()
    if version is None:
        return
    if _parse_version_components(version) < (0, 16, 0):
        print(
            f"[train_grpo_trl] Disabling incompatible torchao {version}; PEFT LoRA requires >= 0.16.0.",
            flush=True,
        )
        _disable_torchao_in_peft()


def prepare_model_for_grpo(model: Any) -> Any:
    """Apply canonical training settings and attach LoRA adapters."""
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    LoraConfig, get_peft_model = _require_peft_components()
    _maybe_disable_incompatible_torchao()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    try:
        model = get_peft_model(model, lora_config)
    except ImportError as exc:
        if "torchao" not in str(exc).lower():
            raise
        _disable_torchao_in_peft()
        model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def load_training_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load the canonical model/tokenizer pair for explicit rollout training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = configure_rollout_tokenizer(AutoTokenizer.from_pretrained(model_name))
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    torch_dtype = _cuda_training_dtype()
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return prepare_model_for_grpo(model), tokenizer


def _default_fake_rubric_scorer(episode_log: str) -> dict[str, float]:
    # Replaced by rule-based scorer from self_rewarding_dpo; kept as thin wrapper
    # for backward compatibility with any callers that reference this symbol directly.
    from rl.self_rewarding_dpo import build_rule_based_rubric_scorer

    return build_rule_based_rubric_scorer()(episode_log)


def load_rubric_scorer(spec: Optional[str], *, enable_rubrics: bool = False):
    """Load a provider-agnostic rubric scorer from an import path."""
    if not enable_rubrics:
        return None
    if not spec:
        return _default_fake_rubric_scorer
    module_name, _, attr_name = str(spec).partition(":")
    if not module_name or not attr_name:
        raise ValueError("judge_scorer must use the format 'module.submodule:callable_name'.")
    module = importlib.import_module(module_name)
    scorer = getattr(module, attr_name)
    if not callable(scorer):
        raise TypeError("Resolved judge scorer is not callable.")
    return scorer


def summarize_batch(summaries: list[EpisodeSummary]) -> dict[str, float]:
    """Aggregate episode summaries into rolling scalar metrics."""
    if not summaries:
        return {}
    return {
        "episode/avg_base_rl_reward": mean(item.base_rl_reward for item in summaries),
        "episode/avg_verifiable_reward": mean(item.verifiable_reward for item in summaries),
        "episode/avg_total_reward": mean(item.total_reward for item in summaries),
        "episode/avg_tool_bonus": mean(item.tool_bonus_total for item in summaries),
        "episode/success_rate": mean(1.0 if item.success_no_default_positive_npv else 0.0 for item in summaries),
        "episode/avg_final_payment_days": mean(item.average_final_payment_days for item in summaries),
        "episode/avg_tool_usage_count": mean(float(item.tool_usage_count) for item in summaries),
        "episode/avg_tool_call_count": mean(float(item.tool_call_count) for item in summaries),
        "episode/avg_tool_effective_count": mean(float(item.tool_effective_count) for item in summaries),
        "episode/avg_resolved_deal_count": mean(float(item.resolved_deal_count) for item in summaries),
        "episode/timeout_or_stepcap_rate": mean(1.0 if item.terminated_by_step_cap else 0.0 for item in summaries),
        "episode/avg_curriculum_level": mean(float(item.curriculum_level) for item in summaries),
    }


def summarize_rollout_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate explicit rollout diagnostics for logging."""
    if not diagnostics:
        return {}
    metrics: dict[str, float] = {}
    numeric_keys = (
        "reward_mean",
        "reward_std",
        "unique_completion_count",
        "unique_action_count",
        "invalid_parse_fraction",
        "identical_terminal_fraction",
    )
    for key in numeric_keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in diagnostics if key in item]
        if values:
            metrics[f"rollout/{key}"] = mean(values)
    return metrics


def _coerce_completion_text(completion: Any) -> str:
    """Return a flat string for a TRL completion (chat list or raw string)."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for message in completion:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict):
                            parts.append(str(chunk.get("text", "")))
                        else:
                            parts.append(str(chunk))
                else:
                    parts.append(str(content))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(completion)


def _normalize_signature_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _coerce_prompt_signature_text(prompt: Any) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return _normalize_signature_text(prompt)
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                role = str(message.get("role", "user"))
                content = message.get("content", "")
                parts.append(f"{role}: {_coerce_completion_text(content)}")
            else:
                parts.append(str(message))
        return _normalize_signature_text("\n".join(parts))
    if isinstance(prompt, dict):
        if "prompt" in prompt:
            return _coerce_prompt_signature_text(prompt.get("prompt"))
        if "messages" in prompt:
            return _coerce_prompt_signature_text(prompt.get("messages"))
        role = str(prompt.get("role", "user"))
        return _normalize_signature_text(f"{role}: {_coerce_completion_text(prompt.get('content', ''))}")
    return _normalize_signature_text(prompt)


def _build_pending_rollout_signature(prompt: Any, completion: Any) -> Optional[str]:
    prompt_text = _coerce_prompt_signature_text(prompt)
    completion_text = _normalize_signature_text(_coerce_completion_text(completion))
    if not prompt_text or not completion_text:
        return None
    return json.dumps(
        {"prompt": prompt_text, "completion": completion_text},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _pending_rollout_record_signature(record: dict[str, Any]) -> Optional[str]:
    completion = record.get("completion_signature_text", record.get("raw_completion_text", ""))
    return _build_pending_rollout_signature(record.get("prompt"), completion)


def _strict_invalid_reward(text: str) -> float:
    if _strict_json_payload(text) is not None:
        return 0.0
    return -0.01


def _completion_format_score(text: str) -> float:
    """Bounded format-quality score in [0, 1] that reads the model output text."""
    raw = (text or "").strip()
    if not raw:
        return 0.0

    score = 0.0
    lowered = raw.lower()
    if raw.startswith("{") and raw.rstrip().endswith("}"):
        score += 0.30
    if '"action_type"' in raw:
        score += 0.15
    for verb in ("propose", "accept", "reject", "advance_period", "tool", "simulate_plan"):
        if verb in lowered:
            score += 0.05
            break
    for key in ("payment_days", "use_treds", "price", "tool_name"):
        if key in lowered:
            score += 0.07
    if '"reason"' in raw or "reason=" in lowered:
        score += 0.10
    length = len(raw)
    if 50 <= length <= 600:
        score += 0.10
    elif 10 <= length < 50:
        score += 0.05

    return float(min(1.0, max(0.0, score)))


def make_reward_function(
    *,
    rubric_scorer=None,
    rubric_weight: float = 0.0,
    summary_buffer: Optional[EpisodeSummaryBuffer] = None,
    pending_rollout_buffer: Optional[PendingRolloutBuffer] = None,
) -> Callable[[list[Any]], list[float]]:
    """Build the TRL reward function for both explicit rollouts and env wrappers."""
    warned_non_environment_inputs = False
    warned_pending_bridge_miss = False
    warned_pending_fifo_fallback = False
    bridge_diagnostics = {"bridge_miss_count": 0}

    def _reward_from_episode_payloads(
        episode_summaries_value: list[EpisodeSummary],
        *,
        completions_value: Optional[list[Any]] = None,
        episode_logs_value: Optional[list[str]] = None,
        raw_completion_texts_value: Optional[list[str]] = None,
        env_reward_std_value: Optional[list[float]] = None,
        reward_mean_value: Optional[list[float]] = None,
        unique_action_count_value: Optional[list[float]] = None,
        unique_completion_count_value: Optional[list[float]] = None,
        invalid_parse_fraction_value: Optional[list[float]] = None,
        identical_terminal_fraction_value: Optional[list[float]] = None,
        termination_reasons_value: Optional[list[str]] = None,
    ) -> list[float]:
        rewards: list[float] = []
        texts = raw_completion_texts_value or [
            _coerce_completion_text(completion) for completion in (completions_value or [None] * len(episode_summaries_value))
        ]
        logs = episode_logs_value or [""] * len(episode_summaries_value)
        reward_std_values = env_reward_std_value or [0.0] * len(episode_summaries_value)
        reward_mean_values = reward_mean_value or [0.0] * len(episode_summaries_value)
        unique_action_values = unique_action_count_value or [0.0] * len(episode_summaries_value)
        unique_completion_values = unique_completion_count_value or [0.0] * len(episode_summaries_value)
        invalid_values = invalid_parse_fraction_value or [0.0] * len(episode_summaries_value)
        identical_terminal_values = identical_terminal_fraction_value or [0.0] * len(episode_summaries_value)
        termination_values = termination_reasons_value or [""] * len(episode_summaries_value)

        for index, summary in enumerate(episode_summaries_value):
            verifiable_reward = float(getattr(summary, "verifiable_reward", 0.0) or 0.0)
            total_reward = float(getattr(summary, "total_reward", verifiable_reward) or verifiable_reward)
            base_reward = verifiable_reward + 0.05 * max(0.0, total_reward - verifiable_reward)
            base_reward -= 0.05 * float(invalid_values[index] or 0.0)

            if float(reward_std_values[index] or 0.0) <= 1e-8:
                base_reward += 0.01 * _completion_format_score(texts[index])

            final_reward = base_reward
            if rubric_scorer is not None and float(rubric_weight) > 0.0:
                rubric_scores = rubric_scorer(logs[index])
                if getattr(summary, "persona_name", None):
                    final_reward = base_reward + float(rubric_weight) * mean(
                        float(value) for value in rubric_scores.values()
                    )
                else:
                    final_reward = combine_rewards(base_reward, rubric_scores, rubric_weight)
            rewards.append(round(final_reward, 6))

            if summary_buffer is not None:
                summary_buffer.append(
                    summary,
                    logs[index],
                    diagnostics={
                        "reward_mean": float(reward_mean_values[index] or 0.0),
                        "reward_std": float(reward_std_values[index] or 0.0),
                        "unique_action_count": float(unique_action_values[index] or 0.0),
                        "unique_completion_count": float(unique_completion_values[index] or 0.0),
                        "invalid_parse_fraction": float(invalid_values[index] or 0.0),
                        "identical_terminal_fraction": float(identical_terminal_values[index] or 0.0),
                        "termination_reason": termination_values[index],
                        "sample_completion": texts[index],
                    },
                )
        return rewards

    def reward_func(
        inputs: Optional[list[Any]] = None,
        prompts: Optional[list[Any]] = None,
        completions: Optional[list[Any]] = None,
        episode_summaries: Optional[list[EpisodeSummary]] = None,
        episode_logs: Optional[list[str]] = None,
        raw_completion_texts: Optional[list[str]] = None,
        env_reward_std: Optional[list[float]] = None,
        reward_mean: Optional[list[float]] = None,
        unique_action_count: Optional[list[float]] = None,
        unique_completion_count: Optional[list[float]] = None,
        invalid_parse_fraction: Optional[list[float]] = None,
        identical_terminal_fraction: Optional[list[float]] = None,
        termination_reasons: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []

        if episode_summaries is not None:
            return _reward_from_episode_payloads(
                episode_summaries,
                completions_value=completions,
                episode_logs_value=episode_logs,
                raw_completion_texts_value=raw_completion_texts,
                env_reward_std_value=env_reward_std,
                reward_mean_value=reward_mean,
                unique_action_count_value=unique_action_count,
                unique_completion_count_value=unique_completion_count,
                invalid_parse_fraction_value=invalid_parse_fraction,
                identical_terminal_fraction_value=identical_terminal_fraction,
                termination_reasons_value=termination_reasons,
            )

        batch_count = len(completions or prompts or inputs or kwargs.get("environments") or [])
        if pending_rollout_buffer is not None and batch_count > 0:
            prompt_batch = prompts or kwargs.get("prompts") or kwargs.get("inputs")
            completion_batch = completions or kwargs.get("completions")
            pending_batch = None
            signature_available = bool(prompt_batch) and bool(completion_batch) and len(prompt_batch) == len(completion_batch)
            if signature_available:
                pending_batch = pending_rollout_buffer.take_by_signature(list(prompt_batch), list(completion_batch))
                if pending_batch is None:
                    pending_batch = pending_rollout_buffer.take(batch_count)
                    if pending_batch is not None:
                        nonlocal warned_pending_fifo_fallback
                        if not warned_pending_fifo_fallback:
                            print(
                                "[train_grpo_trl][WARN] reward_func could not signature-match pending rollout rewards; "
                                "using FIFO compatibility fallback for this TRL build.",
                                flush=True,
                            )
                            warned_pending_fifo_fallback = True
                    else:
                        nonlocal warned_pending_bridge_miss
                        bridge_diagnostics["bridge_miss_count"] = int(bridge_diagnostics["bridge_miss_count"]) + batch_count
                        if not warned_pending_bridge_miss:
                            print(
                                "[train_grpo_trl][WARN] reward_func could not match pending rollout rewards "
                                "for the current prompt/completion batch; using strict-invalid fallback reward.",
                                flush=True,
                            )
                            warned_pending_bridge_miss = True
                        return [round(_strict_invalid_reward(_coerce_completion_text(completion)), 6) for completion in completion_batch]
            else:
                pending_batch = pending_rollout_buffer.take(batch_count)
            if pending_batch is not None:
                return _reward_from_episode_payloads(
                    [record["episode_summary"] for record in pending_batch],
                    completions_value=completions,
                    episode_logs_value=[str(record.get("episode_log", "")) for record in pending_batch],
                    raw_completion_texts_value=[str(record.get("raw_completion_text", "")) for record in pending_batch],
                    env_reward_std_value=[float(record.get("reward_std", 0.0) or 0.0) for record in pending_batch],
                    reward_mean_value=[float(record.get("reward_mean", 0.0) or 0.0) for record in pending_batch],
                    unique_action_count_value=[float(record.get("unique_action_count", 0.0) or 0.0) for record in pending_batch],
                    unique_completion_count_value=[
                        float(record.get("unique_completion_count", 0.0) or 0.0) for record in pending_batch
                    ],
                    invalid_parse_fraction_value=[
                        float(record.get("invalid_parse_fraction", 0.0) or 0.0) for record in pending_batch
                    ],
                    identical_terminal_fraction_value=[
                        float(record.get("identical_terminal_fraction", 0.0) or 0.0) for record in pending_batch
                    ],
                    termination_reasons_value=[str(record.get("termination_reason", "")) for record in pending_batch],
                )

        resolved_inputs = inputs
        if resolved_inputs is None:
            resolved_inputs = kwargs.get("environments")
        if resolved_inputs is None:
            resolved_inputs = kwargs.get("inputs")
        if resolved_inputs is None:
            resolved_inputs = prompts

        environments = list(resolved_inputs or [])
        completions = completions or [None] * len(environments)
        env_records: list[dict[str, Any]] = []
        env_base_rewards: list[float] = []
        for env, completion in zip(environments, completions):
            nonlocal warned_non_environment_inputs
            inner_env = getattr(env, "env", env)
            world_state = getattr(inner_env, "_world_state", None)
            trajectory = getattr(env, "_trajectory_states", [])
            verifiable = None
            completion_text = _coerce_completion_text(completion)

            # Some TRL builds call the reward function with prompt/chat payloads instead of
            # the original environment wrappers. Fall back to a tiny format-only reward
            # instead of crashing, while our explicit rollout buffer path handles the
            # canonical environment-based signal.
            if not (
                hasattr(env, "build_episode_log")
                or hasattr(env, "compute_final_reward")
                or hasattr(env, "summarize_episode")
            ):
                if not warned_non_environment_inputs:
                    print(
                        "[train_grpo_trl][WARN] reward_func received non-environment inputs from TRL; "
                        "falling back to completion-format reward for this batch.",
                        flush=True,
                    )
                    warned_non_environment_inputs = True
                rewards.append(round(_strict_invalid_reward(completion_text), 6))
                continue

            if world_state is not None and trajectory:
                try:
                    from sme_negotiator_env.graders import compute_verifiable_reward  # type: ignore[import]

                    verifiable = float(compute_verifiable_reward(world_state, trajectory))
                except Exception:
                    verifiable = None
            if verifiable is None:
                try:
                    base_reward = float(env.compute_final_reward())
                except Exception:
                    base_reward = float(getattr(env, "reward", 0.0))
            else:
                shaping_signal = 0.0
                try:
                    summary = env.summarize_episode()
                    env_total = float(getattr(summary, "env_reward_total", 0.0) or 0.0)
                    shaping_signal = max(0.0, env_total - verifiable)
                except Exception:
                    pass
                base_reward = verifiable + 0.1 * shaping_signal

            env_records.append(
                {
                    "env": env,
                    "completion_text": completion_text,
                    "base_reward": float(base_reward),
                }
            )
            env_base_rewards.append(float(base_reward))

        env_reward_std_value = pstdev(env_base_rewards) if len(env_base_rewards) > 1 else 0.0
        for record in env_records:
            env = record["env"]
            completion_text = str(record["completion_text"])
            base_reward = float(record["base_reward"])
            if env_reward_std_value <= 1e-8:
                base_reward += 0.1 * _completion_format_score(completion_text)

            final_reward = base_reward
            episode_log = env.build_episode_log()
            if rubric_scorer is not None and float(rubric_weight) > 0.0:
                rubric_scores = rubric_scorer(episode_log)
                if getattr(env, "current_persona", None) is not None:
                    final_reward += float(rubric_weight) * persona_reward(env.current_persona, rubric_scores)
                else:
                    final_reward = combine_rewards(base_reward, rubric_scores, rubric_weight)
            rewards.append(round(final_reward, 6))
            if summary_buffer is not None:
                try:
                    summary_buffer.append(env.summarize_episode(), episode_log)
                except Exception:
                    pass
        return rewards

    setattr(reward_func, "bridge_diagnostics", bridge_diagnostics)
    return reward_func


def build_curriculum_manager_from_args(args: argparse.Namespace) -> CurriculumManager:
    """Construct the Stage 6 curriculum manager from CLI arguments."""
    return CurriculumManager(
        levels=DEFAULT_CURRICULUM_LEVELS,
        window_size=args.curriculum_window_size,
        reward_threshold=args.curriculum_reward_threshold,
        max_default_rate=args.curriculum_max_default_rate,
        cooldown_windows=1,
    )


def build_opponent_manager_from_args(args: argparse.Namespace) -> Optional[OpponentPolicyManager]:
    """Construct the snapshot-opponent manager when self-play is enabled."""
    if not args.enable_self_play:
        return None
    return OpponentPolicyManager(
        snapshots_dir=Path(args.output_dir) / "snapshots",
        zoo_size=args.opponent_zoo_size,
    )


def build_environment_factory(
    args: argparse.Namespace,
    *,
    curriculum: Optional[CurriculumManager],
    opponent_manager: Optional[OpponentPolicyManager],
):
    """Build a zero-arg in-process environment factory."""

    def environment_factory():
        if curriculum is not None:
            difficulty_config = curriculum.current_config()
            total_periods = difficulty_config.total_periods
            buyer_variance = difficulty_config.buyer_variance
            financier_variance = difficulty_config.financier_variance
            curriculum_level = curriculum.current_level()
        else:
            total_periods = args.total_periods
            buyer_variance = 0.0
            financier_variance = 0.0
            curriculum_level = 0
        wrapper_cls = make_environment_factory(
            task_name=args.task_name,
            difficulty=args.difficulty,
            total_periods=total_periods,
            seed=args.seed_base,
            prompt=DEFAULT_PROMPT,
            buyer_variance=buyer_variance,
            financier_variance=financier_variance,
            curriculum_level=curriculum_level,
            lock_curriculum_config=curriculum is not None,
            persona_mode=args.persona_mode,
            persona_name=args.persona_name,
            opponent_manager=opponent_manager,
        )
        return wrapper_cls()

    return environment_factory


def _coerce_prompt_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if isinstance(message, dict):
                messages.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                messages.append({"role": "user", "content": str(message)})
        return messages
    return [{"role": "user", "content": str(prompt or "")}]


def _extract_training_row_from_prompt(prompt: Any, fallback_args: Any) -> dict[str, Any]:
    messages = _coerce_prompt_messages(prompt)
    row = {
        "prompt": messages,
        "task_name": str(getattr(fallback_args, "task_name", "liquidity-correlation-hard")),
        "difficulty": str(getattr(fallback_args, "difficulty", "hard")),
        "seed": int(getattr(fallback_args, "seed_base", 1000)),
        "total_periods": int(getattr(fallback_args, "total_periods", 3)),
    }
    for message in messages:
        for line in str(message.get("content", "")).splitlines():
            if not line.startswith(TRAINING_ROW_PREFIX):
                continue
            payload = json.loads(line[len(TRAINING_ROW_PREFIX) :].strip())
            if isinstance(payload, dict):
                row.update(payload)
            return row
    return row


def _build_observation_message(observation_text: str) -> str:
    return ROLL_OUT_USER_TURN.format(observation=observation_text)


def _render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        training_chat_template = getattr(tokenizer, "training_chat_template", None)
        try:
            return str(tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs))
        except TypeError:
            pass
        if training_chat_template is not None:
            try:
                return str(
                    tokenizer.apply_chat_template(
                        messages,
                        chat_template=training_chat_template,
                        enable_thinking=False,
                        **kwargs,
                    )
                )
            except TypeError:
                try:
                    return str(
                        tokenizer.apply_chat_template(
                            messages,
                            chat_template=training_chat_template,
                            chat_template_kwargs={"enable_thinking": False},
                            **kwargs,
                        )
                    )
                except TypeError:
                    pass
        try:
            return str(tokenizer.apply_chat_template(messages, chat_template_kwargs={"enable_thinking": False}, **kwargs))
        except TypeError:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _move_inputs_to_model_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    params = inspect.signature(fn).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()):
        return dict(kwargs), []
    valid = set(params) - {"self"}
    unsupported = sorted(key for key in kwargs if key not in valid)
    return {key: value for key, value in kwargs.items() if key in valid}, unsupported


def _token_ids_to_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [int(token) for token in data[0]]
        if isinstance(data, list):
            return [int(token) for token in data]
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [int(token) for token in value[0]]
        return [int(token) for token in value]
    return []


def _compute_generated_logprobs(output_scores: Any, completion_ids: list[int]) -> list[float]:
    if not output_scores or not completion_ids:
        return []
    try:
        import torch
    except ImportError:
        return [0.0 for _ in completion_ids]

    values: list[float] = []
    for index, token_id in enumerate(completion_ids):
        if index >= len(output_scores):
            break
        logits = output_scores[index]
        if hasattr(logits, "dim") and int(logits.dim()) > 1:
            logits = logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        values.append(float(log_probs[int(token_id)].item()))
    return values


def _generate_completion_turn(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt_text = _render_chat_prompt(tokenizer, messages)
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    tokenized = _move_inputs_to_model_device(tokenized, model)
    prompt_ids = _token_ids_to_list(tokenized.get("input_ids"))
    use_sampling = float(temperature) > 0.0
    generate_kwargs = {
        **tokenized,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": use_sampling,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if use_sampling:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generate_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = int(tokenizer.eos_token_id)

    outputs = model.generate(**generate_kwargs)
    sequence_ids = _token_ids_to_list(getattr(outputs, "sequences", None))
    completion_ids = sequence_ids[len(prompt_ids) :] if len(sequence_ids) >= len(prompt_ids) else []
    text = tokenizer.decode(completion_ids, skip_special_tokens=True) if completion_ids else ""
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": _compute_generated_logprobs(getattr(outputs, "scores", None), completion_ids),
        "text": str(text),
    }


def _strict_json_payload(raw_text: str) -> Optional[dict[str, Any]]:
    candidate = str(raw_text or "").strip()
    if not candidate.startswith("{") or not candidate.rstrip().endswith("}"):
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "action_type" not in parsed:
        return None
    return parsed


def _parse_action_and_validity(raw_text: str, observation: Any, *, strict_json: bool = True) -> tuple[Any, bool]:
    payload = _strict_json_payload(raw_text)
    if payload is not None:
        return parse_action(json.dumps(payload, sort_keys=True, ensure_ascii=True), observation), True
    if strict_json:
        return conservative_default_action(observation), False
    return parse_action(str(raw_text or "").strip(), observation), False


def _run_single_rollout_sample(
    prompt: Any,
    *,
    model: Any,
    tokenizer: Any,
    rollout_args: Any,
    env_factory: Callable[[], Any],
) -> dict[str, Any]:
    row = _extract_training_row_from_prompt(prompt, rollout_args)
    wrapper = env_factory()
    reset_text = wrapper.reset(**row)
    messages = copy.deepcopy(_coerce_prompt_messages(row["prompt"]))
    messages.append({"role": "user", "content": _build_observation_message(reset_text)})

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    raw_turn_texts: list[str] = []
    parsed_actions: list[dict[str, Any]] = []
    valid_json_steps = 0

    max_episode_steps = int(getattr(rollout_args, "max_episode_steps", 24) or 24)
    max_new_tokens = int(getattr(rollout_args, "max_completion_length", 256) or 256)
    temperature = float(getattr(rollout_args, "temperature", 1.0) or 1.0)
    top_p = float(getattr(rollout_args, "top_p", 1.0) or 1.0)

    termination_reason = "rollout_step_cap"
    for _ in range(max_episode_steps):
        turn = _generate_completion_turn(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        prompt_ids.extend(list(turn["prompt_ids"]))
        completion_ids.extend(list(turn["completion_ids"]))
        logprobs.extend(list(turn["logprobs"]))
        raw_text = str(turn["text"])
        raw_turn_texts.append(raw_text)

        observation = getattr(wrapper, "last_observation", None)
        action, was_valid_json = _parse_action_and_validity(raw_text, observation, strict_json=True)
        if was_valid_json:
            valid_json_steps += 1
        parsed_actions.append(action.model_dump(exclude_none=True))

        messages.append({"role": "assistant", "content": raw_text})
        try:
            execute_action(wrapper, action)
        except Exception as exc:
            termination_reason = f"action_error:{exc}"
            break

        latest_observation = getattr(wrapper, "last_observation", None)
        if latest_observation is not None:
            latest_metadata = latest_observation.metadata or {}
            termination_reason = str(latest_metadata.get("termination_reason", termination_reason))
        if bool(getattr(wrapper, "done", False)):
            break
        if latest_observation is None:
            termination_reason = "missing_observation"
            break
        messages.append({"role": "user", "content": _build_observation_message(format_observation(latest_observation))})

    summary = wrapper.summarize_episode()
    episode_log = wrapper.build_episode_log()
    last_observation = getattr(wrapper, "last_observation", None)
    reward_breakdown = {}
    if last_observation is not None:
        reward_breakdown = dict((last_observation.metadata or {}).get("reward_breakdown", {}) or {})
        metadata_reason = str((last_observation.metadata or {}).get("termination_reason", "") or "")
        if metadata_reason:
            termination_reason = metadata_reason

    strict_json_fraction = valid_json_steps / max(1, len(raw_turn_texts))
    completion_signature_text = (
        str(tokenizer.decode(completion_ids, skip_special_tokens=True))
        if completion_ids and hasattr(tokenizer, "decode")
        else "\n".join(raw_turn_texts)
    )
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs or [0.0 for _ in completion_ids],
        "episode_summary": summary,
        "episode_log": episode_log,
        "reward_breakdown": reward_breakdown,
        "termination_reason": termination_reason,
        "parsed_actions": parsed_actions,
        "prompt": copy.deepcopy(prompt),
        "raw_completion_text": "\n".join(raw_turn_texts),
        "completion_signature_text": completion_signature_text,
        "invalid_parse_fraction": round(1.0 - strict_json_fraction, 6),
    }


def _compute_group_rollout_diagnostics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "unique_completion_count": 0.0,
            "unique_action_count": 0.0,
            "invalid_parse_fraction": 0.0,
            "identical_terminal_fraction": 0.0,
        }
    rewards = [float(sample["episode_summary"].verifiable_reward) for sample in samples]
    completion_signatures = {str(sample["raw_completion_text"]) for sample in samples}
    action_signatures = {
        json.dumps(sample["parsed_actions"], sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        for sample in samples
    }
    terminal_signatures = [
        json.dumps(
            {
                "termination_reason": str(sample["termination_reason"]),
                "verifiable_reward": round(float(sample["episode_summary"].verifiable_reward), 6),
                "resolved_deal_count": int(sample["episode_summary"].resolved_deal_count),
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        for sample in samples
    ]
    modal_terminal_count = max(Counter(terminal_signatures).values()) if terminal_signatures else 0
    return {
        "reward_mean": round(mean(rewards), 6),
        "reward_std": round(pstdev(rewards) if len(rewards) > 1 else 0.0, 6),
        "unique_completion_count": float(len(completion_signatures)),
        "unique_action_count": float(len(action_signatures)),
        "invalid_parse_fraction": round(
            mean(float(sample["invalid_parse_fraction"]) for sample in samples),
            6,
        ),
        "identical_terminal_fraction": round(modal_terminal_count / max(1, len(samples)), 6),
    }


def build_rollout_func(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    curriculum: Optional[CurriculumManager],
    opponent_manager: Optional[OpponentPolicyManager],
    pending_rollout_buffer: Optional[PendingRolloutBuffer] = None,
) -> Callable[..., dict[str, Any]]:
    """Build a TRL rollout_func for explicit environment stepping."""
    env_factory = build_environment_factory(args, curriculum=curriculum, opponent_manager=opponent_manager)

    def rollout_func(prompts: list[Any], *context: Any, **kwargs: Any) -> dict[str, Any]:
        trainer = None
        rollout_args: Any = args
        processing_class = tokenizer
        model = kwargs.get("model")

        if context:
            candidate = context[0]
            if hasattr(candidate, "model") and hasattr(candidate, "args"):
                trainer = candidate
            else:
                rollout_args = candidate
                if len(context) > 1:
                    processing_class = context[1]
        if trainer is None:
            trainer = kwargs.get("trainer")
        if trainer is not None:
            rollout_args = getattr(trainer, "args", rollout_args)
            processing_class = getattr(trainer, "processing_class", processing_class)
            model = getattr(trainer, "model", model)
        if model is None:
            raise ValueError("rollout_func could not resolve the active model.")

        num_generations = int(getattr(rollout_args, "num_generations", 1) or 1)
        samples: list[dict[str, Any]] = []
        for prompt in prompts:
            group = [
                _run_single_rollout_sample(
                    prompt,
                    model=model,
                    tokenizer=processing_class,
                    rollout_args=rollout_args,
                    env_factory=env_factory,
                )
                for _ in range(num_generations)
            ]
            group_metrics = _compute_group_rollout_diagnostics(group)
            for sample in group:
                sample.update(group_metrics)
                samples.append(sample)

        if pending_rollout_buffer is not None and samples:
            pending_rollout_buffer.extend(samples)

        return {
            "prompt_ids": [list(sample["prompt_ids"]) for sample in samples],
            "completion_ids": [list(sample["completion_ids"]) for sample in samples],
            "logprobs": [list(sample["logprobs"]) for sample in samples],
            "episode_summaries": [sample["episode_summary"] for sample in samples],
            "episode_logs": [sample["episode_log"] for sample in samples],
            "reward_breakdowns": [sample["reward_breakdown"] for sample in samples],
            "termination_reasons": [sample["termination_reason"] for sample in samples],
            "parsed_actions": [sample["parsed_actions"] for sample in samples],
            "raw_completion_texts": [sample["raw_completion_text"] for sample in samples],
            "reward_mean": [float(sample["reward_mean"]) for sample in samples],
            "env_reward_std": [float(sample["reward_std"]) for sample in samples],
            "unique_action_count": [float(sample["unique_action_count"]) for sample in samples],
            "unique_completion_count": [float(sample["unique_completion_count"]) for sample in samples],
            "invalid_parse_fraction": [float(sample["invalid_parse_fraction"]) for sample in samples],
            "identical_terminal_fraction": [float(sample["identical_terminal_fraction"]) for sample in samples],
        }

    return rollout_func


def build_metrics_callback(
    summary_buffer: EpisodeSummaryBuffer,
    trainer_callback_base: type[Any],
    *,
    curriculum: Optional[CurriculumManager],
    build_preference_dataset: bool,
    scorer,
    output_dir: str,
):
    """Create a TrainerCallback that logs metrics and updates the curriculum."""

    class RollingEpisodeMetricsCallback(trainer_callback_base):
        """Attach summarized episode metrics, diagnostics, and curriculum promotion."""

        def __init__(self) -> None:
            super().__init__()
            self.reward_curve: list[float] = []
            self.success_curve: list[float] = []
            self.zero_variance_streak = 0

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            summaries, episode_logs, diagnostics = summary_buffer.drain()
            if logs is None:
                return control

            if summaries:
                logs.update(summarize_batch(summaries))
                if "episode/avg_total_reward" in logs:
                    self.reward_curve.append(float(logs["episode/avg_total_reward"]))
                    self.success_curve.append(float(logs.get("episode/success_rate", 0.0)))
                if curriculum is not None:
                    for summary in summaries:
                        curriculum.record_episode(summary.base_rl_reward, summary.defaulted_sme_count > 0)
                    promoted = curriculum.maybe_advance_level()
                    logs["curriculum/current_level"] = curriculum.current_level()
                    logs["curriculum/promoted"] = 1.0 if promoted else 0.0
                if build_preference_dataset and scorer is not None and episode_logs:
                    examples = build_preference_examples(
                        episode_logs,
                        scorer,
                        seed=int(getattr(state, "global_step", 0) or 0) + 1000,
                    )
                    output_path = (
                        Path(output_dir)
                        / "preferences"
                        / f"step_{int(getattr(state, 'global_step', 0)):06d}.jsonl"
                    )
                    write_preference_dataset(output_path, examples)
                    logs["preferences/written"] = float(len(examples))

            if diagnostics:
                logs.update(summarize_rollout_diagnostics(diagnostics))
                reward_std = float(logs.get("rollout/reward_std", 0.0) or 0.0)
                if reward_std <= 1e-8:
                    self.zero_variance_streak += 1
                else:
                    self.zero_variance_streak = 0
                if self.zero_variance_streak >= 2:
                    sample = diagnostics[0]
                    print(
                        "[TRAINING][WARN] rollout reward variance is zero for consecutive logs. "
                        f"sample_completion={json.dumps(str(sample.get('sample_completion', '')))} "
                        f"termination_reason={json.dumps(str(sample.get('termination_reason', '')))}",
                        flush=True,
                    )
            return control

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
            if len(self.reward_curve) < 2:
                return control
            try:
                figure_path = _save_reward_curve_plot(
                    self.reward_curve,
                    self.success_curve,
                    output_dir=str(args.output_dir),
                )
                print(f"[TRAINING] Reward curve saved to {figure_path}", flush=True)
            except Exception as exc:
                print(f"[TRAINING] Could not save reward curve: {exc}", flush=True)
            return control

    return RollingEpisodeMetricsCallback()


def build_snapshot_callback(
    trainer_callback_base: type[Any],
    *,
    opponent_manager: Optional[OpponentPolicyManager],
    interval: int,
    output_dir: str,
):
    """Create a callback that periodically snapshots the SME policy for self-play."""
    snapshot_interval = max(1, int(interval))
    snapshots_dir = Path(output_dir) / "snapshots"

    class SnapshotOpponentCallback(trainer_callback_base):
        """Persist periodic model snapshots and register them with the opponent zoo."""

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            global_step = int(getattr(state, "global_step", 0) or 0)
            if opponent_manager is None or global_step <= 0 or global_step % snapshot_interval != 0:
                return control
            snapshot_path = snapshots_dir / f"sme_policy_step_{global_step:06d}"
            model = kwargs.get("model")
            processing_class = kwargs.get("processing_class")
            if model is not None and hasattr(model, "save_pretrained"):
                snapshot_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(snapshot_path)
                if processing_class is not None and hasattr(processing_class, "save_pretrained"):
                    processing_class.save_pretrained(snapshot_path)
                opponent_manager.register_snapshot(snapshot_path)
            return control

    return SnapshotOpponentCallback()


def build_grpo_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI args into a GRPOConfig kwargs dictionary."""
    per_device_train_batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1)
    gradient_accumulation_steps = int(getattr(args, "gradient_accumulation_steps", 4) or 4)
    num_generations = int(getattr(args, "num_generations", 4) or 4)
    generation_batch_size = int(getattr(args, "generation_batch_size", 0) or num_generations)
    learning_rate = float(getattr(args, "learning_rate", 5e-6) or 5e-6)
    temperature = float(getattr(args, "temperature", 1.0) or 1.0)
    top_p = float(getattr(args, "top_p", 1.0) or 1.0)
    max_prompt_length = int(getattr(args, "max_prompt_length", 1024) or 1024)
    max_completion_length = int(getattr(args, "max_completion_length", 256) or 256)
    logging_steps = int(getattr(args, "logging_steps", 1) or 1)
    save_steps = int(getattr(args, "save_steps", 1000) or 1000)
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "generation_batch_size": generation_batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "top_p": top_p,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_only_model": True,
        "report_to": _training_log_backend(),
        "use_vllm": bool(args.use_vllm),
    }
    if getattr(args, "max_steps", None) is not None:
        kwargs["max_steps"] = int(args.max_steps)
    if bool(args.use_vllm):
        kwargs["vllm_mode"] = args.vllm_mode
    return kwargs


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the TRL training script."""
    parser = argparse.ArgumentParser(description="Train GRPO on the SME liquidity environment with TRL.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--task-name", default="liquidity-correlation-hard")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--total-periods", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/grpo_sme_liquidity_trl")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--rubric-weight", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-self-play", action="store_true")
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--opponent-zoo-size", type=int, default=5)
    parser.add_argument("--curriculum-window-size", type=int, default=100)
    parser.add_argument("--curriculum-reward-threshold", type=float, default=0.6)
    parser.add_argument("--curriculum-max-default-rate", type=float, default=0.2)
    parser.add_argument("--enable-rubrics", action="store_true")
    parser.add_argument("--judge-scorer", default=None)
    parser.add_argument("--persona-mode", choices=("off", "fixed", "per_episode"), default="off")
    parser.add_argument("--persona-name", default=None)
    parser.add_argument("--build-preference-dataset", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=24)
    return parser


def make_training_args(**overrides: Any) -> argparse.Namespace:
    """Return parser-backed training args with optional attribute overrides."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    valid_keys = {
        action.dest
        for action in parser._actions
        if getattr(action, "dest", None) not in {None, "help"}
    }
    unknown = sorted(set(overrides) - valid_keys)
    if unknown:
        raise KeyError(f"Unknown training arg overrides: {unknown}")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _resolve_training_components(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve shared runtime components used by dry-runs and real training."""
    curriculum = build_curriculum_manager_from_args(args)
    opponent_manager = build_opponent_manager_from_args(args)
    rubric_scorer = load_rubric_scorer(args.judge_scorer, enable_rubrics=args.enable_rubrics)
    rows = build_training_rows(
        task_name=args.task_name,
        difficulty=args.difficulty,
        total_periods=args.total_periods,
        num_samples=args.num_samples,
        seed_base=args.seed_base,
    )
    env_factory = build_environment_factory(
        args,
        curriculum=curriculum,
        opponent_manager=opponent_manager,
    )
    return {
        "curriculum": curriculum,
        "opponent_manager": opponent_manager,
        "rubric_scorer": rubric_scorer,
        "rows": rows,
        "env_factory": env_factory,
    }


def _install_optional_dependency_stub(module_name: str) -> bool:
    """Install a minimal shim for optional TRL imports we do not use."""
    def _new_stub_module(name: str, *, is_package: bool = False) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__file__ = f"<optional-stub:{name}>"
        if is_package:
            module.__path__ = []  # type: ignore[attr-defined]
        return module

    if module_name == "mergekit":
        if "mergekit" in sys.modules:
            return True

        mergekit_module = _new_stub_module("mergekit", is_package=True)
        config_module = _new_stub_module("mergekit.config")
        merge_module = _new_stub_module("mergekit.merge")

        class MergeConfiguration:  # pragma: no cover - compatibility shim only
            pass

        class MergeOptions:  # pragma: no cover - compatibility shim only
            pass

        def run_merge(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - compatibility shim only
            raise RuntimeError(
                "mergekit compatibility stub was invoked. This training path does not support model merging."
            )

        config_module.MergeConfiguration = MergeConfiguration
        merge_module.MergeOptions = MergeOptions
        merge_module.run_merge = run_merge
        mergekit_module.config = config_module
        mergekit_module.merge = merge_module

        sys.modules["mergekit"] = mergekit_module
        sys.modules["mergekit.config"] = config_module
        sys.modules["mergekit.merge"] = merge_module
        return True

    if module_name == "llm_blender":
        if "llm_blender" not in sys.modules:
            sys.modules["llm_blender"] = _new_stub_module("llm_blender")
        return True

    if module_name == "weave":
        if "weave" in sys.modules:
            return True

        weave_module = _new_stub_module("weave", is_package=True)
        trace_module = _new_stub_module("weave.trace", is_package=True)
        context_module = _new_stub_module("weave.trace.context")

        class EvaluationLogger:  # pragma: no cover - compatibility shim only
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class _WeaveClientContext:  # pragma: no cover - compatibility shim only
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        def weave_client_context(*args: Any, **kwargs: Any) -> _WeaveClientContext:
            return _WeaveClientContext()

        weave_module.EvaluationLogger = EvaluationLogger
        trace_module.context = context_module
        context_module.weave_client_context = weave_client_context

        sys.modules["weave"] = weave_module
        sys.modules["weave.trace"] = trace_module
        sys.modules["weave.trace.context"] = context_module
        return True

    if module_name == "comet_ml":
        if module_name not in sys.modules:
            sys.modules[module_name] = _new_stub_module(module_name)
        return True

    return False


def _clear_trl_import_cache() -> None:
    """Clear TRL submodules that may cache failed optional imports."""
    for module_name in (
        "trl.mergekit_utils",
        "trl.trainer.callbacks",
        "trl.trainer.judges",
        "trl.trainer.grpo_trainer",
    ):
        sys.modules.pop(module_name, None)


def _extract_missing_module_name(exc: Exception) -> Optional[str]:
    """Extract the missing dependency name from an import exception."""
    match = re.search(r"No module named '([^']+)'", str(exc))
    if match is None:
        return None
    missing = str(match.group(1) or "").strip()
    return missing.split(".", 1)[0] if missing else None


def _import_trl_grpo_symbols() -> tuple[Any, Any]:
    """Import GRPOConfig and GRPOTrainer with a clearer dependency error."""
    last_exc: Optional[Exception] = None
    for _ in range(8):
        try:
            from trl import GRPOConfig, GRPOTrainer
        except Exception as exc:  # pragma: no cover - import path depends on installed TRL build
            last_exc = exc
            missing_module = _extract_missing_module_name(exc)
            if missing_module and _install_optional_dependency_stub(missing_module):
                _clear_trl_import_cache()
                continue
            raise
        else:
            return GRPOConfig, GRPOTrainer

    if last_exc is not None:
        missing_module = _extract_missing_module_name(last_exc)
        if missing_module is not None:
            raise ImportError(
                "TRL GRPO import failed because this TRL build eagerly imports optional dependencies. "
                f"A compatibility stub for '{missing_module}' was installed, but the import still failed. "
                "Restart the runtime and rerun the notebook install cell."
            ) from last_exc
        raise ImportError(
            "TRL GRPO import failed even after installing compatibility stubs for optional dependencies. "
            "Restart the runtime and rerun the notebook install cell."
        ) from last_exc
    raise ImportError("TRL GRPO import failed for an unknown reason.")


def _trl_version() -> Optional[str]:
    """Return the installed TRL version when available."""
    try:
        return importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_rollout_func_support(grpo_trainer_cls: Any) -> None:
    """Fail fast when the installed TRL build does not support explicit rollouts."""
    parameters = inspect.signature(grpo_trainer_cls.__init__).parameters
    if "rollout_func" in parameters:
        return

    version = _trl_version() or "unknown"
    raise ImportError(
        "Installed TRL version does not support the canonical explicit-rollout training path: "
        f"GRPOTrainer.__init__ has no 'rollout_func' parameter (detected trl=={version}). "
        "Install a rollout-capable TRL release such as trl>=0.25.1, then restart the runtime and rerun the install cell."
    )


def build_training_session(args: argparse.Namespace) -> dict[str, Any]:
    """Build the canonical explicit-rollout GRPO training bundle."""
    components = _resolve_training_components(args)

    from transformers import TrainerCallback
    GRPOConfig, GRPOTrainer = _import_trl_grpo_symbols()
    _require_rollout_func_support(GRPOTrainer)

    dataset = build_dataset(components["rows"])
    model, tokenizer = load_training_model_and_tokenizer(args.model_name)
    summary_buffer = EpisodeSummaryBuffer()
    pending_rollout_buffer = PendingRolloutBuffer()
    reward_func = make_reward_function(
        rubric_scorer=components["rubric_scorer"],
        rubric_weight=args.rubric_weight,
        summary_buffer=summary_buffer,
        pending_rollout_buffer=pending_rollout_buffer,
    )
    rollout_func = build_rollout_func(
        args,
        tokenizer=tokenizer,
        curriculum=components["curriculum"],
        opponent_manager=components["opponent_manager"],
        pending_rollout_buffer=pending_rollout_buffer,
    )

    grpo_kwargs = build_grpo_config_kwargs(args)
    grpo_kwargs["reward_weights"] = [1.0]
    grpo_kwargs["log_completions"] = True
    config_kwargs, unsupported_config = _filter_kwargs_for_callable(GRPOConfig.__init__, grpo_kwargs)
    if unsupported_config:
        print(
            f"[train_grpo_trl] Dropping unsupported GRPOConfig args for this TRL version: {unsupported_config}",
            flush=True,
        )
    training_args = GRPOConfig(**config_kwargs)

    try:
        from rl.monitoring import RewardMonitorCallback

        monitoring_cb = RewardMonitorCallback(
            log_every_n_steps=10,
            generation_sample_every_n_steps=50,
        )
    except ImportError:
        monitoring_cb = None

    callbacks = [
        build_metrics_callback(
            summary_buffer,
            TrainerCallback,
            curriculum=components["curriculum"],
            build_preference_dataset=args.build_preference_dataset,
            scorer=components["rubric_scorer"] or _default_fake_rubric_scorer,
            output_dir=args.output_dir,
        )
    ]
    if args.enable_self_play:
        callbacks.append(
            build_snapshot_callback(
                TrainerCallback,
                opponent_manager=components["opponent_manager"],
                interval=args.snapshot_interval,
                output_dir=args.output_dir,
            )
        )
    if monitoring_cb is not None:
        callbacks.append(monitoring_cb)

    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": reward_func,
        "train_dataset": dataset,
        "args": training_args,
        "rollout_func": rollout_func,
        "callbacks": callbacks,
    }
    filtered_trainer_kwargs, unsupported_trainer = _filter_kwargs_for_callable(GRPOTrainer.__init__, trainer_kwargs)
    if unsupported_trainer:
        raise TypeError(
            "Installed TRL version does not support the canonical explicit-rollout training path. "
            f"Unsupported trainer args: {unsupported_trainer}"
        )

    final_checkpoint_path = Path(args.output_dir) / "final-grpo-model"
    return {
        **components,
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "training_args": training_args,
        "reward_funcs": reward_func,
        "rollout_func": rollout_func,
        "callbacks": callbacks,
        "summary_buffer": summary_buffer,
        "pending_rollout_buffer": pending_rollout_buffer,
        "final_checkpoint_path": final_checkpoint_path,
        "grpo_config_kwargs": config_kwargs,
        "trainer_kwargs": filtered_trainer_kwargs,
    }


def create_trainer(session: dict[str, Any]) -> Any:
    """Instantiate GRPOTrainer from a freshly built training session."""
    _, GRPOTrainer = _import_trl_grpo_symbols()
    return GRPOTrainer(**dict(session["trainer_kwargs"]))


def save_training_session(session: dict[str, Any], trainer: Any) -> Path:
    """Save the final model/tokenizer checkpoint for a completed session."""
    checkpoint_path = Path(session["final_checkpoint_path"])
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(checkpoint_path))
    tokenizer = session.get("tokenizer")
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path


def run_training_session(args: argparse.Namespace) -> dict[str, Any]:
    """Build, train, and save a canonical GRPO session in one call."""
    session = build_training_session(args)
    trainer = create_trainer(session)
    trainer.train()
    checkpoint_path = save_training_session(session, trainer)
    return {
        "session": session,
        "trainer": trainer,
        "checkpoint_path": checkpoint_path,
    }


def print_dry_run_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    env_factory,
    *,
    curriculum: Optional[CurriculumManager],
    opponent_manager: Optional[OpponentPolicyManager],
    rubric_scorer,
) -> None:
    """Print the resolved training configuration without loading a model."""
    preview_wrapper = env_factory()
    preview_text = preview_wrapper.reset(**rows[0]) if rows else ""
    summary = {
        "mode": "dry-run",
        "model_name": args.model_name,
        "task_name": args.task_name,
        "difficulty": args.difficulty,
        "total_periods": args.total_periods,
        "num_samples": args.num_samples,
        "seed_range": [args.seed_base, args.seed_base + max(args.num_samples - 1, 0)],
        "output_dir": args.output_dir,
        "use_vllm": bool(args.use_vllm),
        "vllm_mode": args.vllm_mode,
        "grpo_config": build_grpo_config_kwargs(args),
        "rollout": {
            "max_episode_steps": int(getattr(args, "max_episode_steps", 24) or 24),
            "strict_action_contract": build_action_contract_text(),
        },
        "first_row": rows[0] if rows else None,
        "observation_preview": preview_text,
        "curriculum": {
            "enabled": True,
            "current_level": curriculum.current_level() if curriculum is not None else 0,
            "current_config": curriculum.current_config().__dict__ if curriculum is not None else None,
        },
        "self_play": {
            "enabled": bool(args.enable_self_play),
            "snapshot_interval": args.snapshot_interval,
            "opponent_zoo_size": args.opponent_zoo_size,
            "snapshot_ids": opponent_manager.snapshot_ids() if opponent_manager is not None else [],
        },
        "rubrics": {
            "enabled": bool(args.enable_rubrics),
            "persona_mode": args.persona_mode,
            "persona_name": args.persona_name,
            "judge_scorer": args.judge_scorer,
            "resolved_scorer": None if rubric_scorer is None else getattr(rubric_scorer, "__name__", str(rubric_scorer)),
        },
    }
    print(json.dumps(summary, indent=2, default=str))


def main(argv: Optional[list[str]] = None) -> int:
    """Run GRPO training or print a dry-run summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    components = _resolve_training_components(args)

    if args.dry_run:
        print_dry_run_summary(
            args,
            components["rows"],
            components["env_factory"],
            curriculum=components["curriculum"],
            opponent_manager=components["opponent_manager"],
            rubric_scorer=components["rubric_scorer"],
        )
        return 0

    result = run_training_session(args)
    checkpoint_path = Path(result["checkpoint_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
