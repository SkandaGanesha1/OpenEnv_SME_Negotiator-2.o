#!/usr/bin/env python3
"""Canonical TRL GRPO training entrypoint for the liquidity environment.

Install with optional extras such as:
    pip install -e ".[rl]"
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import importlib.metadata
import inspect
import json
import math
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
    SUPPORTED_ACTION_TYPES,
    SUPPORTED_TOOL_NAMES,
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
from sme_negotiator_env.models import NegotiationAction
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
ROLLOUT_SYSTEM_PROMPT = (
    "You are an SME treasury agent operating a deterministic long-horizon liquidity workflow. "
    "Choose the single best next action for the exact current state. "
    "Maximize the final verifiable reward, preserve positive NPV, use tools only when they improve the state, "
    "and finish the episode without default. "
    + build_action_contract_text()
)
ROLL_OUT_USER_TURN = (
    "Current observation:\n{observation}"
)
TRAIN_GRPO_TRL_BRIDGE_REVISION = "2026-04-26c"
_CANONICAL_ACTION_KEYS = frozenset(
    {
        "action_type",
        "price",
        "payment_days",
        "use_treds",
        "reason",
        "deal_id",
        "simulation_plan",
        "simulation_horizon",
        "tool_name",
        "tool_args",
        "propose_late_payment_penalty_clause",
        "propose_dynamic_discounting",
        "dynamic_discount_annual_rate",
        "split_deal_buyer_a_days",
        "split_deal_buyer_b_days",
        "split_deal_buyer_a_price",
        "split_deal_buyer_b_price",
        "distress_disclosure_level",
    }
)
_CANONICAL_BOOL_KEYS = frozenset(
    {
        "use_treds",
        "propose_late_payment_penalty_clause",
        "propose_dynamic_discounting",
    }
)
_CANONICAL_NUMERIC_KEYS = frozenset(
    {
        "price",
        "dynamic_discount_annual_rate",
        "split_deal_buyer_a_price",
        "split_deal_buyer_b_price",
    }
)
_CANONICAL_INT_KEYS = frozenset(
    {
        "payment_days",
        "simulation_horizon",
        "split_deal_buyer_a_days",
        "split_deal_buyer_b_days",
    }
)


def _training_log_backend(env: Optional[dict[str, str]] = None) -> str:
    source = os.environ if env is None else env
    value = str(source.get("TRAINING_LOG_BACKEND", "none") or "none").strip().lower()
    if value in {"wandb", "tensorboard", "none"}:
        return value
    return "none"


def _default_vllm_max_model_length(*, max_prompt_length: int, max_completion_length: int) -> int:
    """Return a notebook-safe vLLM context window for colocated GRPO runs.

    vLLM defaults to the model's full configured context window when this is
    unset, which is too large for small GPUs like a T4 in notebook runtimes.
    The runtime only needs enough room for the prompt, the generated
    completion, and a modest buffer.
    """

    required_tokens = int(max_prompt_length) + int(max_completion_length) + 256
    return max(2048, required_tokens)


def _patch_additional_chat_templates_404() -> None:
    """Treat missing additional chat templates as an empty directory.

    Some `transformers`/`huggingface_hub` runtime combinations raise
    `RemoteEntryNotFoundError` when a model repo does not contain the optional
    `additional_chat_templates/` directory. Tokenizer loading should interpret
    that case as "no extra templates", not as a fatal error.
    """

    try:
        import huggingface_hub.utils as hf_hub_utils
        import transformers.tokenization_utils_base as tokenization_utils_base
        import transformers.utils.hub as transformers_hub
    except Exception:
        return

    original = getattr(transformers_hub.list_repo_templates, "__wrapped_original__", None)
    if original is None:
        original = transformers_hub.list_repo_templates

    error_types = tuple(
        error_type
        for error_type in (
            getattr(hf_hub_utils, "RemoteEntryNotFoundError", None),
            getattr(hf_hub_utils, "EntryNotFoundError", None),
        )
        if error_type is not None
    )
    if not error_types:
        return

    def _safe_list_repo_templates(*args: Any, **kwargs: Any) -> list[str]:
        try:
            return original(*args, **kwargs)
        except error_types:
            return []

    setattr(_safe_list_repo_templates, "__wrapped_original__", original)
    transformers_hub.list_repo_templates = _safe_list_repo_templates
    tokenization_utils_base.list_repo_templates = _safe_list_repo_templates


def _stdout_supports_fileno() -> bool:
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    try:
        stream.fileno()
    except Exception:
        return False
    return True


def _patch_vllm_notebook_stdout() -> None:
    """Prevent vLLM stdout-suppression helpers from crashing in notebooks.

    Some notebook runtimes expose `sys.stdout` as an object without a usable
    `fileno()`, which breaks vLLM's `suppress_stdout()` helper during engine
    initialization. In that environment we bypass the suppression context and
    enable DEBUG logging so vLLM also avoids its own suppression fast path.
    """

    if _stdout_supports_fileno():
        return

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

    @contextlib.contextmanager
    def _passthrough_suppress_stdout():
        yield

    for module_name in ("vllm.utils.system_utils", "vllm.distributed.parallel_state"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        setattr(module, "suppress_stdout", _passthrough_suppress_stdout)


def _patch_vllm_attention_backend() -> None:
    """Prefer a non-FlashInfer backend on older notebook GPUs unless overridden."""

    if os.environ.get("VLLM_ATTENTION_BACKEND"):
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return
    if int(major) < 8:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")


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
    episode_records: list[dict[str, Any]] = field(default_factory=list)
    episode_count: int = 0

    def append(
        self,
        summary: EpisodeSummary,
        episode_log: Optional[str] = None,
        diagnostics: Optional[dict[str, Any]] = None,
        training_reward: Optional[float] = None,
    ) -> None:
        self.items.append(summary)
        if episode_log is not None:
            self.episode_logs.append(episode_log)
        diagnostics_copy = None if diagnostics is None else dict(diagnostics)
        if diagnostics_copy is not None:
            self.diagnostics.append(diagnostics_copy)
        self.episode_count += 1
        self.episode_records.append(
            _build_episode_reward_record(
                episode=self.episode_count,
                summary=summary,
                diagnostics=diagnostics_copy,
                training_reward=training_reward,
            )
        )

    def drain(self) -> tuple[list[EpisodeSummary], list[str], list[dict[str, Any]]]:
        summaries = list(self.items)
        episode_logs = list(self.episode_logs)
        diagnostics = list(self.diagnostics)
        self.items.clear()
        self.episode_logs.clear()
        self.diagnostics.clear()
        return summaries, episode_logs, diagnostics

    def export_records(self) -> list[dict[str, Any]]:
        """Return all persisted episode reward records collected during training."""
        return [dict(record) for record in self.episode_records]


def _build_episode_reward_record(
    *,
    episode: int,
    summary: EpisodeSummary,
    diagnostics: Optional[dict[str, Any]],
    training_reward: Optional[float],
) -> dict[str, Any]:
    record = {
        "episode": int(episode),
        "step": int(episode),
        "training_reward": float(
            training_reward
            if training_reward is not None
            else getattr(summary, "total_reward", 0.0) or 0.0
        ),
        "total_reward": float(getattr(summary, "total_reward", 0.0) or 0.0),
        "verifiable_reward": float(getattr(summary, "verifiable_reward", 0.0) or 0.0),
        "base_rl_reward": float(getattr(summary, "base_rl_reward", 0.0) or 0.0),
        "tool_bonus_total": float(getattr(summary, "tool_bonus_total", 0.0) or 0.0),
        "env_reward_total": float(getattr(summary, "env_reward_total", 0.0) or 0.0),
        "success_no_default_positive_npv": bool(
            getattr(summary, "success_no_default_positive_npv", False)
        ),
        "average_final_payment_days": float(
            getattr(summary, "average_final_payment_days", 0.0) or 0.0
        ),
        "tool_usage_count": int(getattr(summary, "tool_usage_count", 0) or 0),
        "tool_call_count": int(getattr(summary, "tool_call_count", 0) or 0),
        "tool_effective_count": int(getattr(summary, "tool_effective_count", 0) or 0),
        "resolved_deal_count": int(getattr(summary, "resolved_deal_count", 0) or 0),
        "defaulted_sme_count": int(getattr(summary, "defaulted_sme_count", 0) or 0),
        "terminated_by_step_cap": bool(getattr(summary, "terminated_by_step_cap", False)),
    }
    if diagnostics:
        for key in (
            "reward_mean",
            "reward_std",
            "unique_action_count",
            "unique_completion_count",
            "invalid_parse_fraction",
            "identical_terminal_fraction",
            "contract_score",
        ):
            if key in diagnostics:
                record[key] = float(diagnostics.get(key, 0.0) or 0.0)
        termination_reason = str(diagnostics.get("termination_reason", "") or "").strip()
        if termination_reason:
            record["termination_reason"] = termination_reason
    return record


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


def resolve_training_precision_kwargs() -> dict[str, bool]:
    """Return explicit GRPO mixed-precision flags for the active hardware.

    Some TRL / Transformers builds default to `bf16=True` on GPU-backed
    TrainingArguments, which fails on Colab T4-class hardware. We set the
    precision flags explicitly so notebook and CLI runs stay aligned.
    """
    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {}

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
    }


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

    _patch_additional_chat_templates_404()
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
        "contract_score",
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


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_integer_like_json_number(value: Any) -> bool:
    return _is_json_number(value) and float(value).is_integer()


def _coerce_observation_dict(observation: Any) -> dict[str, Any]:
    if observation is None:
        return {}
    if isinstance(observation, dict):
        return dict(observation)
    if hasattr(observation, "model_dump"):
        dumped = observation.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    result: dict[str, Any] = {}
    for key in (
        "buyer_price",
        "buyer_days",
        "cost_threshold",
        "liquidity_threshold",
        "current_period",
        "total_periods",
        "done",
    ):
        if hasattr(observation, key):
            result[key] = getattr(observation, key)
    return result


def _has_only_canonical_action_keys(payload: dict[str, Any]) -> bool:
    return all(str(key) in _CANONICAL_ACTION_KEYS for key in payload)


def _validate_canonical_optional_fields(payload: dict[str, Any]) -> bool:
    if not _has_only_canonical_action_keys(payload):
        return False

    for key in _CANONICAL_BOOL_KEYS:
        if key in payload and not isinstance(payload.get(key), bool):
            return False

    for key in _CANONICAL_NUMERIC_KEYS:
        if key in payload and payload.get(key) is not None and not _is_json_number(payload.get(key)):
            return False

    for key in _CANONICAL_INT_KEYS:
        if key in payload and payload.get(key) is not None and not _is_integer_like_json_number(payload.get(key)):
            return False

    if "reason" in payload and payload.get("reason") is not None and not isinstance(payload.get("reason"), str):
        return False
    if "deal_id" in payload and payload.get("deal_id") is not None and not isinstance(payload.get("deal_id"), str):
        return False
    if "tool_args" in payload and payload.get("tool_args") is not None and not isinstance(payload.get("tool_args"), dict):
        return False
    if "simulation_plan" in payload and payload.get("simulation_plan") is not None and not isinstance(
        payload.get("simulation_plan"), dict
    ):
        return False
    if "tool_name" in payload and payload.get("tool_name") is not None and payload.get("tool_name") not in SUPPORTED_TOOL_NAMES:
        return False
    if "distress_disclosure_level" in payload and payload.get("distress_disclosure_level") not in {None, "low", "medium", "high"}:
        return False
    if "dynamic_discount_annual_rate" in payload and payload.get("dynamic_discount_annual_rate") is not None:
        rate = float(payload["dynamic_discount_annual_rate"])
        if rate < 0.0 or rate > 0.95:
            return False
    if "simulation_horizon" in payload and payload.get("simulation_horizon") is not None:
        if int(float(payload["simulation_horizon"])) < 1:
            return False
    return True


def _action_contract_score(payload: dict[str, Any], observation: Any = None) -> float:
    action_type = str(payload.get("action_type", "") or "")
    obs = _coerce_observation_dict(observation)

    if action_type in {"reject", "advance_period"}:
        return 0.7

    if action_type == "tool":
        score = 0.7
        if payload.get("tool_args"):
            score += 0.05
        if bool(obs.get("done", False)):
            score -= 0.15
        return float(min(1.0, max(0.0, score)))

    if action_type == "simulate_plan":
        score = 0.68
        if payload.get("simulation_plan"):
            score += 0.08
        if payload.get("simulation_horizon") is not None:
            score += 0.04
        return float(min(1.0, max(0.0, score)))

    if action_type not in {"propose", "accept"}:
        return 0.0

    price = float(payload.get("price", 0.0) or 0.0)
    payment_days = int(float(payload.get("payment_days", 0) or 0))
    score = 0.35

    if 0.01 <= price <= 100000.0:
        score += 0.08
    if 0 <= payment_days <= 365:
        score += 0.08

    buyer_price = obs.get("buyer_price")
    if _is_json_number(buyer_price):
        gap_ratio = abs(price - float(buyer_price)) / max(abs(float(buyer_price)), 1.0)
        score += max(0.0, 0.2 * (1.0 - min(gap_ratio, 1.5) / 1.5))

    cost_threshold = obs.get("cost_threshold")
    if _is_json_number(cost_threshold) and price < float(cost_threshold):
        score -= min(0.25, (float(cost_threshold) - price) / max(abs(float(cost_threshold)), 1.0))

    buyer_days = obs.get("buyer_days")
    if _is_integer_like_json_number(buyer_days):
        buyer_days_value = int(float(buyer_days))
        day_gap = abs(payment_days - buyer_days_value)
        score += max(0.0, 0.15 * (1.0 - min(day_gap, 120) / 120))

    liquidity_threshold = obs.get("liquidity_threshold")
    if _is_integer_like_json_number(liquidity_threshold):
        liquidity_value = int(float(liquidity_threshold))
        if payment_days <= max(liquidity_value, 0):
            score += 0.05
        if payment_days > max(liquidity_value + 120, 365):
            score -= 0.18
        if bool(payload.get("use_treds", False)) and _is_integer_like_json_number(buyer_days):
            if int(float(buyer_days)) > liquidity_value + 15:
                score += 0.05

    return float(min(1.0, max(0.0, score)))


def _strict_invalid_reward(text: str) -> float:
    return round(-0.08 + 0.08 * _completion_format_score(text), 6)


def _completion_format_score(text: str, observation: Any = None) -> float:
    """Bounded format-quality score in [0, 1] that reads the model output text."""
    raw = (text or "").strip()
    if not raw:
        return 0.0

    strict_payload = _strict_json_payload(raw)
    if strict_payload is not None:
        return float(min(1.0, 0.7 + 0.3 * _action_contract_score(strict_payload, observation)))

    score = 0.0
    lowered = raw.lower()
    if raw.startswith("{") and raw.rstrip().endswith("}"):
        score += 0.12
    if '"action_type"' in raw:
        score += 0.08
    for verb in SUPPORTED_ACTION_TYPES:
        if verb in lowered:
            score += 0.04
            break
    for key in ("price", "payment_days", "tool_name", "tool_args", "simulation_plan"):
        if key in lowered:
            score += 0.04
    if '"tool"' in raw and '"tool_name"' not in raw:
        score -= 0.08
    if "<think>" in lowered or "</think>" in lowered:
        score -= 0.15
    for invalid_variant in ("proposed", "proposal", "proposals", "proposer"):
        if f'"action_type":"{invalid_variant}"' in lowered.replace(" ", ""):
            score -= 0.08
    length = len(raw)
    if 20 <= length <= 600:
        score += 0.05
    elif 1 <= length < 20:
        score += 0.02

    return float(min(1.0, max(0.0, score)))


def make_reward_function(
    *,
    rubric_scorer=None,
    rubric_weight: float = 0.0,
    summary_buffer: Optional[EpisodeSummaryBuffer] = None,
    pending_rollout_buffer: Optional[PendingRolloutBuffer] = None,
    prompt_env_factory: Optional[Callable[[], Any]] = None,
    prompt_env_args: Any = None,
) -> Callable[[list[Any]], list[float]]:
    """Build the TRL reward function for both explicit rollouts and env wrappers."""
    warned_non_environment_inputs = False
    warned_pending_bridge_miss = False
    warned_pending_fifo_fallback = False
    warned_prompt_env_fallback = False
    bridge_diagnostics = {
        "bridge_miss_count": 0,
        "fifo_fallback_count": 0,
        "signature_match_count": 0,
        "prompt_env_fallback_count": 0,
    }
    bridge_debug_state = {"reward_calls": 0}

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
        contract_score_value: Optional[list[float]] = None,
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
        contract_values = contract_score_value or [_completion_format_score(text) for text in texts]

        for index, summary in enumerate(episode_summaries_value):
            verifiable_reward = float(getattr(summary, "verifiable_reward", 0.0) or 0.0)
            total_reward = float(getattr(summary, "total_reward", verifiable_reward) or verifiable_reward)
            base_reward = verifiable_reward + 0.05 * max(0.0, total_reward - verifiable_reward)
            contract_score = float(contract_values[index] or 0.0)
            invalid_fraction = float(invalid_values[index] or 0.0)
            termination_reason = str(termination_values[index] or "")
            termination_lower = termination_reason.lower()
            reward_std_scalar = float(reward_std_values[index] or 0.0)
            if reward_std_scalar <= 1e-8 or invalid_fraction > 0.0:
                base_reward += 0.12 * (contract_score - 0.5)
            base_reward -= 0.04 * invalid_fraction

            if reward_std_scalar <= 1e-8:
                base_reward += 0.02 * (contract_score - 0.5)

            if termination_lower == "prompt_env_invalid_first_action":
                invalid_cap = -0.35 + 0.06 * max(-0.5, min(0.5, contract_score - 0.5))
                base_reward = min(base_reward, invalid_cap)
            elif termination_lower.startswith("prompt_env_") and any(
                marker in termination_lower for marker in ("ongoing", "incomplete", "step_cap", "missing_observation")
            ):
                unresolved_cap = 0.45 + 0.08 * max(-0.5, min(0.5, contract_score - 0.5))
                base_reward = min(base_reward - 0.18, unresolved_cap)
            elif termination_lower == "ongoing":
                base_reward -= 0.12

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
                        "termination_reason": termination_reason,
                        "contract_score": contract_score,
                        "sample_completion": texts[index],
                    },
                    training_reward=final_reward,
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
        contract_score: Optional[list[float]] = None,
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
                contract_score_value=contract_score,
            )

        batch_count = len(completions or prompts or inputs or kwargs.get("environments") or [])
        if pending_rollout_buffer is not None and batch_count > 0:
            bridge_debug_state["reward_calls"] = int(bridge_debug_state["reward_calls"]) + 1
            prompt_batch = prompts or kwargs.get("prompts") or kwargs.get("inputs")
            completion_batch = completions or kwargs.get("completions")
            pending_batch = None
            pending_before = len(pending_rollout_buffer.items)
            bridge_mode = "unavailable"
            signature_available = bool(prompt_batch) and bool(completion_batch) and len(prompt_batch) == len(completion_batch)
            if signature_available:
                pending_batch = pending_rollout_buffer.take_by_signature(list(prompt_batch), list(completion_batch))
                if pending_batch is not None:
                    bridge_diagnostics["signature_match_count"] = int(bridge_diagnostics["signature_match_count"]) + batch_count
                    bridge_mode = "signature"
                else:
                    pending_batch = pending_rollout_buffer.take(batch_count)
                    if pending_batch is not None:
                        bridge_diagnostics["fifo_fallback_count"] = int(bridge_diagnostics["fifo_fallback_count"]) + batch_count
                        bridge_mode = "fifo"
                        nonlocal warned_pending_fifo_fallback
                        if not warned_pending_fifo_fallback:
                            print(
                                f"[train_grpo_trl][bridge:{TRAIN_GRPO_TRL_BRIDGE_REVISION}][WARN] "
                                "reward_func could not signature-match pending rollout rewards; "
                                "using FIFO compatibility fallback for this TRL build.",
                                flush=True,
                            )
                            warned_pending_fifo_fallback = True
                    else:
                        bridge_mode = "miss"
                        nonlocal warned_pending_bridge_miss
                        bridge_diagnostics["bridge_miss_count"] = int(bridge_diagnostics["bridge_miss_count"]) + batch_count
                        if not warned_pending_bridge_miss:
                            print(
                                f"[train_grpo_trl][bridge:{TRAIN_GRPO_TRL_BRIDGE_REVISION}][WARN] "
                                "reward_func could not match pending rollout rewards "
                                "for the current prompt/completion batch; using strict-invalid fallback reward.",
                                flush=True,
                            )
                            warned_pending_bridge_miss = True
                        if int(bridge_debug_state["reward_calls"]) <= 6:
                            print(
                                f"[train_grpo_trl][bridge:{TRAIN_GRPO_TRL_BRIDGE_REVISION}] "
                                f"reward_call={bridge_debug_state['reward_calls']} batch_count={batch_count} "
                                f"pending_before={pending_before} pending_after={len(pending_rollout_buffer.items)} "
                                f"signature_available={signature_available} bridge_mode={bridge_mode}",
                                flush=True,
                            )
                        if prompt_env_factory is not None and prompt_env_args is not None and prompt_batch and completion_batch:
                            bridge_diagnostics["prompt_env_fallback_count"] = int(
                                bridge_diagnostics["prompt_env_fallback_count"]
                            ) + batch_count
                            nonlocal warned_prompt_env_fallback
                            if not warned_prompt_env_fallback:
                                print(
                                    f"[train_grpo_trl][bridge:{TRAIN_GRPO_TRL_BRIDGE_REVISION}][WARN] "
                                    "pending rollout buffer is empty for this TRL build; "
                                    "scoring prompt/completion batches via prompt-derived environment fallback.",
                                    flush=True,
                                )
                                warned_prompt_env_fallback = True
                            simulated_batch = [
                                _score_prompt_completion_via_environment(
                                    prompt,
                                    completion,
                                    fallback_args=prompt_env_args,
                                    env_factory=prompt_env_factory,
                                )
                                for prompt, completion in zip(prompt_batch, completion_batch)
                            ]
                            return _reward_from_episode_payloads(
                                [record["episode_summary"] for record in simulated_batch],
                                completions_value=list(completion_batch),
                                episode_logs_value=[str(record.get("episode_log", "")) for record in simulated_batch],
                                raw_completion_texts_value=[
                                    str(record.get("raw_completion_text", "")) for record in simulated_batch
                                ],
                                invalid_parse_fraction_value=[
                                    float(record.get("invalid_parse_fraction", 0.0) or 0.0) for record in simulated_batch
                                ],
                                termination_reasons_value=[
                                    str(record.get("termination_reason", "")) for record in simulated_batch
                                ],
                                contract_score_value=[
                                    float(record.get("contract_score", 0.0) or 0.0) for record in simulated_batch
                                ],
                            )
                        return [round(_strict_invalid_reward(_coerce_completion_text(completion)), 6) for completion in completion_batch]
            else:
                pending_batch = pending_rollout_buffer.take(batch_count)
                if pending_batch is not None:
                    bridge_diagnostics["fifo_fallback_count"] = int(bridge_diagnostics["fifo_fallback_count"]) + batch_count
                    bridge_mode = "fifo_no_signature"
                else:
                    bridge_mode = "missing_no_signature"
            if int(bridge_debug_state["reward_calls"]) <= 6:
                print(
                    f"[train_grpo_trl][bridge:{TRAIN_GRPO_TRL_BRIDGE_REVISION}] "
                    f"reward_call={bridge_debug_state['reward_calls']} batch_count={batch_count} "
                    f"pending_before={pending_before} pending_after={len(pending_rollout_buffer.items)} "
                    f"signature_available={signature_available} bridge_mode={bridge_mode}",
                    flush=True,
                )
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
                    contract_score_value=[
                        float(record.get("contract_score", _completion_format_score(str(record.get("raw_completion_text", "")))) or 0.0)
                        for record in pending_batch
                    ],
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
                    summary_buffer.append(env.summarize_episode(), episode_log, training_reward=final_reward)
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


def _strip_training_row_metadata(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        content = str(message.get("content", "") or "")
        kept_lines = [line for line in content.splitlines() if not line.startswith(TRAINING_ROW_PREFIX)]
        cleaned_content = "\n".join(kept_lines).strip()
        if not cleaned_content:
            continue
        cleaned.append(
            {
                "role": str(message.get("role", "user")),
                "content": cleaned_content,
            }
        )
    return cleaned


def _build_rollout_turn_messages(
    *,
    observation_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROLLOUT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_observation_message(observation_text)},
    ]


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


def _tokenize_text_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    try:
        tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    except TypeError:
        tokenized = tokenizer(text, return_tensors="pt")
    return _token_ids_to_list(tokenized.get("input_ids"))


def _serialize_rollout_actions(parsed_actions: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for action in parsed_actions:
        if not isinstance(action, dict):
            continue
        lines.append(json.dumps(action, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
    return "\n".join(lines)


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


def _coerce_vllm_sampled_logprobs(raw_logprobs: Any, completion_ids: list[int]) -> list[float]:
    if not raw_logprobs or not completion_ids:
        return []

    values: list[float] = []
    for token_id, item in zip(completion_ids, raw_logprobs):
        chosen = None
        if isinstance(item, dict):
            chosen = item.get(token_id)
            if chosen is None:
                chosen = item.get(str(token_id))
            if chosen is None and item:
                chosen = next(iter(item.values()))
        else:
            chosen = item

        if hasattr(chosen, "logprob"):
            values.append(float(getattr(chosen, "logprob")))
        elif chosen is None:
            values.append(0.0)
        else:
            try:
                values.append(float(chosen))
            except Exception:
                values.append(0.0)
    return values


def _format_rollout_logprobs_for_trl(logprobs: list[float], completion_ids: list[int]) -> list[list[float]]:
    values = list(logprobs) if logprobs else [0.0 for _ in completion_ids]
    if len(values) < len(completion_ids):
        values.extend([0.0 for _ in range(len(completion_ids) - len(values))])
    return [[float(value)] for value in values[: len(completion_ids)]]


def _format_rollout_logprob_token_ids_for_trl(completion_ids: list[int]) -> list[list[int]]:
    return [[int(token_id)] for token_id in completion_ids]


def _generate_completion_turn_with_vllm(
    vllm_llm: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompt_text = _render_chat_prompt(tokenizer, messages)
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = _token_ids_to_list(tokenized.get("input_ids"))
    use_sampling = float(temperature) > 0.0

    sampling_params = SamplingParams(
        n=1,
        max_tokens=int(max_new_tokens),
        temperature=float(temperature) if use_sampling else 0.0,
        top_p=float(top_p) if use_sampling else 1.0,
        logprobs=0,
    )
    outputs = vllm_llm.generate([prompt_text], sampling_params=sampling_params, use_tqdm=False)
    if not outputs:
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": [],
            "logprobs": [],
            "text": "",
        }

    request_output = outputs[0]
    request_prompt_ids = _token_ids_to_list(getattr(request_output, "prompt_token_ids", None))
    if request_prompt_ids:
        prompt_ids = request_prompt_ids
    candidates = list(getattr(request_output, "outputs", []) or [])
    if not candidates:
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": [],
            "logprobs": [],
            "text": "",
        }
    candidate = candidates[0]
    completion_ids = _token_ids_to_list(getattr(candidate, "token_ids", None))
    text = str(getattr(candidate, "text", "") or "")
    logprobs = _coerce_vllm_sampled_logprobs(getattr(candidate, "logprobs", None), completion_ids)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": text,
    }


def _generate_completion_turn(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    vllm_llm: Any = None,
) -> dict[str, Any]:
    if vllm_llm is not None:
        return _generate_completion_turn_with_vllm(
            vllm_llm,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

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


def _extract_embedded_json_object(raw_text: str) -> Optional[str]:
    """Extract the first balanced JSON object from mixed model output."""
    text = re.sub(r"<think>.*?</think>", " ", str(raw_text or ""), flags=re.DOTALL | re.IGNORECASE).strip()
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strict_json_payload(raw_text: str) -> Optional[dict[str, Any]]:
    candidate = str(raw_text or "").strip()
    if not candidate.startswith("{") or not candidate.rstrip().endswith("}"):
        candidate = str(_extract_embedded_json_object(candidate) or "").strip()
        if not candidate:
            return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "action_type" not in parsed:
        return None
    if not _validate_canonical_optional_fields(parsed):
        return None

    action_type = parsed.get("action_type")
    if not isinstance(action_type, str) or action_type not in SUPPORTED_ACTION_TYPES:
        return None

    if action_type in {"propose", "accept"}:
        price = parsed.get("price")
        payment_days = parsed.get("payment_days")
        if not _is_json_number(price) or float(price) < 0.0:
            return None
        if not _is_integer_like_json_number(payment_days) or int(float(payment_days)) < 0:
            return None
        return parsed

    if action_type == "tool":
        if parsed.get("tool_name") not in SUPPORTED_TOOL_NAMES:
            return None
        if "tool_args" not in parsed or not isinstance(parsed.get("tool_args"), dict):
            return None
        return parsed

    if action_type == "simulate_plan":
        if not isinstance(parsed.get("simulation_plan"), dict):
            return None
        horizon = parsed.get("simulation_horizon")
        if horizon is not None and (not _is_integer_like_json_number(horizon) or int(float(horizon)) < 1):
            return None
        return parsed

    return parsed


def _parse_action_and_validity(raw_text: str, observation: Any, *, strict_json: bool = True) -> tuple[Any, bool]:
    payload = _strict_json_payload(raw_text)
    if payload is not None:
        return parse_action(json.dumps(payload, sort_keys=True, ensure_ascii=True), observation), True
    if strict_json:
        return conservative_default_action(observation), False
    return parse_action(str(raw_text or "").strip(), observation), False


def _followup_policy_action(observation: Any, *, step_index: int) -> NegotiationAction:
    obs = _coerce_observation_dict(observation)
    open_deal_ids = list(obs.get("open_deal_ids") or [])
    active_deal_id = obs.get("active_deal_id") or (open_deal_ids[0] if open_deal_ids else None)

    if not open_deal_ids:
        return NegotiationAction(action_type="advance_period")

    buyer_price = float(obs.get("buyer_price", 0.0) or 0.0)
    buyer_days = int(obs.get("buyer_days", 0) or 0)
    cost_threshold = float(obs.get("cost_threshold", 0.0) or 0.0)
    liquidity_threshold = int(obs.get("liquidity_threshold", buyer_days) or buyer_days)
    use_treds = bool(buyer_days > liquidity_threshold + 10)

    if step_index >= 1 and buyer_days <= liquidity_threshold and buyer_price >= cost_threshold:
        return NegotiationAction(
            action_type="accept",
            deal_id=active_deal_id,
            price=buyer_price,
            payment_days=buyer_days,
            use_treds=use_treds,
            reason="Deterministic follow-up policy acceptance",
        )

    fallback = conservative_default_action(observation)
    if getattr(fallback, "deal_id", None) is None and active_deal_id is not None:
        payload = fallback.model_dump(exclude_none=True)
        payload["deal_id"] = str(active_deal_id)
        return NegotiationAction(**payload)
    return fallback


def _prompt_env_penalty_summary(
    *,
    reward: float,
    invalid_action_count: int = 0,
    terminated_by_step_cap: bool = False,
) -> EpisodeSummary:
    penalty_reward = float(reward)
    return EpisodeSummary(
        episode_completed=False,
        base_rl_reward=penalty_reward,
        verifiable_reward=penalty_reward,
        total_reward=penalty_reward,
        tool_bonus_total=0.0,
        env_reward_total=penalty_reward,
        success_no_default_positive_npv=False,
        average_final_payment_days=0.0,
        tool_usage_count=0,
        resolved_deal_count=0,
        defaulted_sme_count=0,
        invalid_action_count=int(invalid_action_count),
        terminated_by_step_cap=bool(terminated_by_step_cap),
    )


def _score_prompt_completion_via_environment(
    prompt: Any,
    completion: Any,
    *,
    fallback_args: Any,
    env_factory: Callable[[], Any],
) -> dict[str, Any]:
    row = _extract_training_row_from_prompt(prompt, fallback_args)
    wrapper = env_factory()
    wrapper.reset(**row)

    completion_text = _coerce_completion_text(completion)
    first_observation = getattr(wrapper, "last_observation", None)
    first_contract_score = _completion_format_score(completion_text, first_observation)
    first_action, was_valid_json = _parse_action_and_validity(completion_text, first_observation, strict_json=True)

    if not was_valid_json:
        relaxed_action, _ = _parse_action_and_validity(completion_text, first_observation, strict_json=False)
        try:
            execute_action(wrapper, relaxed_action)
        except Exception:
            return {
                "episode_summary": _prompt_env_penalty_summary(reward=-0.4, invalid_action_count=1),
                "episode_log": "Prompt-env fallback rejected an invalid first action before stepping the environment.",
                "raw_completion_text": completion_text,
                "termination_reason": "prompt_env_invalid_first_action",
                "invalid_parse_fraction": 1.0,
                "contract_score": round(first_contract_score, 6),
                "parsed_actions": [],
            }
        final_reason = "prompt_env_relaxed_first_action"
        final_observation = getattr(wrapper, "last_observation", None)
        if final_observation is not None:
            metadata_reason = str((final_observation.metadata or {}).get("termination_reason", "") or "")
            if metadata_reason:
                final_reason = f"prompt_env_relaxed_{metadata_reason}"
        return {
            "episode_summary": wrapper.summarize_episode(),
            "episode_log": wrapper.build_episode_log(),
            "raw_completion_text": completion_text,
            "termination_reason": final_reason,
            "invalid_parse_fraction": 1.0,
            "contract_score": round(first_contract_score, 6),
            "parsed_actions": [relaxed_action.model_dump(exclude_none=True)],
        }

    termination_reason = "prompt_env_fallback"
    action_history = [first_action.model_dump(exclude_none=True)]
    try:
        execute_action(wrapper, first_action)
    except Exception as exc:
        termination_reason = f"prompt_env_action_error:{exc}"
    else:
        max_episode_steps = int(getattr(fallback_args, "max_episode_steps", 24) or 24)
        followup_step_limit = max(0, min(3, max_episode_steps - 1))
        for step_index in range(1, followup_step_limit + 1):
            if bool(getattr(wrapper, "done", False)):
                break
            observation = getattr(wrapper, "last_observation", None)
            if observation is None:
                termination_reason = "prompt_env_missing_observation"
                break
            next_action = _followup_policy_action(observation, step_index=step_index)
            action_history.append(next_action.model_dump(exclude_none=True))
            try:
                execute_action(wrapper, next_action)
            except Exception as exc:
                termination_reason = f"prompt_env_followup_error:{exc}"
                break
        else:
            if not bool(getattr(wrapper, "done", False)):
                termination_reason = "prompt_env_incomplete"

    final_observation = getattr(wrapper, "last_observation", None)
    if final_observation is not None:
        metadata_reason = str((final_observation.metadata or {}).get("termination_reason", "") or "")
        if metadata_reason:
            if metadata_reason.startswith("prompt_env_"):
                termination_reason = metadata_reason
            elif bool(getattr(wrapper, "done", False)):
                termination_reason = metadata_reason
            else:
                termination_reason = f"prompt_env_{metadata_reason}"
    if not bool(getattr(wrapper, "done", False)) and termination_reason == "prompt_env_fallback":
        termination_reason = "prompt_env_incomplete"

    return {
        "episode_summary": wrapper.summarize_episode(),
        "episode_log": wrapper.build_episode_log(),
        "raw_completion_text": completion_text,
        "termination_reason": termination_reason,
        "invalid_parse_fraction": 0.0 if was_valid_json else 1.0,
        "contract_score": round(first_contract_score, 6),
        "parsed_actions": action_history,
    }


def _run_single_rollout_sample(
    prompt: Any,
    *,
    model: Any,
    tokenizer: Any,
    rollout_args: Any,
    env_factory: Callable[[], Any],
    vllm_llm: Any = None,
) -> dict[str, Any]:
    row = _extract_training_row_from_prompt(prompt, rollout_args)
    wrapper = env_factory()
    reset_text = wrapper.reset(**row)
    current_observation_text = str(reset_text)
    training_prompt_messages = _build_rollout_turn_messages(observation_text=current_observation_text)
    training_prompt_ids = _tokenize_text_without_special_tokens(
        tokenizer,
        _render_chat_prompt(tokenizer, training_prompt_messages),
    )

    completion_ids: list[int] = []
    raw_turn_texts: list[str] = []
    parsed_actions: list[dict[str, Any]] = []
    valid_json_steps = 0
    contract_scores: list[float] = []

    max_episode_steps = int(getattr(rollout_args, "max_episode_steps", 24) or 24)
    max_new_tokens = int(getattr(rollout_args, "max_completion_length", 256) or 256)
    temperature = float(getattr(rollout_args, "temperature", 1.0) or 1.0)
    top_p = float(getattr(rollout_args, "top_p", 1.0) or 1.0)

    termination_reason = "rollout_step_cap"
    for _ in range(max_episode_steps):
        turn_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if vllm_llm is not None:
            turn_kwargs["vllm_llm"] = vllm_llm
        turn = _generate_completion_turn(
            model,
            tokenizer,
            _build_rollout_turn_messages(observation_text=current_observation_text),
            **turn_kwargs,
        )
        raw_text = str(turn["text"])
        raw_turn_texts.append(raw_text)

        observation = getattr(wrapper, "last_observation", None)
        contract_scores.append(_completion_format_score(raw_text, observation))
        action, was_valid_json = _parse_action_and_validity(raw_text, observation, strict_json=True)
        if was_valid_json:
            valid_json_steps += 1
        parsed_actions.append(action.model_dump(exclude_none=True))
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
        current_observation_text = format_observation(latest_observation)

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
    completion_signature_text = _serialize_rollout_actions(parsed_actions)
    if not completion_signature_text:
        completion_signature_text = "\n".join(raw_turn_texts)
    completion_ids = _tokenize_text_without_special_tokens(tokenizer, completion_signature_text)
    return {
        "prompt_ids": training_prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": [0.0 for _ in completion_ids],
        "episode_summary": summary,
        "episode_log": episode_log,
        "reward_breakdown": reward_breakdown,
        "termination_reason": termination_reason,
        "parsed_actions": parsed_actions,
        "prompt": copy.deepcopy(prompt),
        "raw_completion_text": "\n".join(raw_turn_texts),
        "completion_signature_text": completion_signature_text,
        "invalid_parse_fraction": round(1.0 - strict_json_fraction, 6),
        "contract_score": round(mean(contract_scores), 6) if contract_scores else 0.0,
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
        vllm_generation = None if trainer is None else getattr(trainer, "vllm_generation", None)
        vllm_llm = None if vllm_generation is None else getattr(vllm_generation, "llm", None)

        num_generations = int(getattr(rollout_args, "num_generations", 1) or 1)
        samples: list[dict[str, Any]] = []
        for prompt in prompts:
            sample_kwargs = {
                "model": model,
                "tokenizer": processing_class,
                "rollout_args": rollout_args,
                "env_factory": env_factory,
            }
            if vllm_llm is not None:
                sample_kwargs["vllm_llm"] = vllm_llm
            group = [
                _run_single_rollout_sample(
                    prompt,
                    **sample_kwargs,
                )
                for _ in range(num_generations)
            ]
            group_metrics = _compute_group_rollout_diagnostics(group)
            for sample in group:
                sample.update(group_metrics)
                samples.append(sample)

        if pending_rollout_buffer is not None and samples:
            pending_before = len(pending_rollout_buffer.items)
            pending_rollout_buffer.extend(samples)
            if len(samples) and len(prompts) and len(samples) <= 12:
                print(
                    f"[train_grpo_trl][bridge:{TRAIN_GRPO_TRL_BRIDGE_REVISION}] "
                    f"rollout_prompts={len(prompts)} samples_added={len(samples)} "
                    f"pending_before={pending_before} pending_after={len(pending_rollout_buffer.items)}",
                    flush=True,
                )

        return {
            "prompt_ids": [list(sample["prompt_ids"]) for sample in samples],
            "completion_ids": [list(sample["completion_ids"]) for sample in samples],
            "logprobs": [
                _format_rollout_logprobs_for_trl(
                    list(sample["logprobs"]),
                    list(sample["completion_ids"]),
                )
                for sample in samples
            ],
            "logprob_token_ids": [
                _format_rollout_logprob_token_ids_for_trl(
                    list(sample["completion_ids"]),
                )
                for sample in samples
            ],
            "episode_summaries": [sample["episode_summary"] for sample in samples],
            "episode_logs": [sample["episode_log"] for sample in samples],
            "termination_reasons": [sample["termination_reason"] for sample in samples],
            "raw_completion_texts": [sample["raw_completion_text"] for sample in samples],
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
    kwargs.update(resolve_training_precision_kwargs())
    if getattr(args, "max_steps", None) is not None:
        kwargs["max_steps"] = int(args.max_steps)
    if bool(args.use_vllm):
        kwargs["vllm_mode"] = args.vllm_mode
        kwargs["vllm_importance_sampling_correction"] = False
        kwargs["vllm_gpu_memory_utilization"] = float(
            getattr(args, "vllm_gpu_memory_utilization", 0.5) or 0.5
        )
        vllm_max_model_length = getattr(args, "vllm_max_model_length", None)
        kwargs["vllm_max_model_length"] = int(
            vllm_max_model_length
            if vllm_max_model_length is not None
            else _default_vllm_max_model_length(
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
            )
        )
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
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-max-model-length", type=int, default=None)
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
    print(
        f"[train_grpo_trl] Bridge revision {TRAIN_GRPO_TRL_BRIDGE_REVISION} "
        "(strict-contract-v2, reward-shaping-v2, bridge-debug-v1).",
        flush=True,
    )
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
        prompt_env_factory=components["env_factory"],
        prompt_env_args=args,
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
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    if bool(getattr(session.get("training_args"), "use_vllm", False)):
        _patch_vllm_attention_backend()
        _patch_vllm_notebook_stdout()
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
    reward_func = session.get("reward_funcs")
    bridge_diagnostics = dict(getattr(reward_func, "bridge_diagnostics", {}) or {})
    summary_buffer = session.get("summary_buffer")
    episode_reward_history = (
        summary_buffer.export_records() if isinstance(summary_buffer, EpisodeSummaryBuffer) else []
    )
    return {
        "session": session,
        "trainer": trainer,
        "checkpoint_path": checkpoint_path,
        "bridge_diagnostics": bridge_diagnostics,
        "episode_reward_history": episode_reward_history,
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
