"""Notebook-facing helpers for tiny in-process liquidity demos."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from rl.bridge import format_observation, make_environment_factory, parse_action
from rl.train_grpo_trl import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    _training_log_backend,
    EpisodeSummaryBuffer,
    build_dataset,
    build_metrics_callback,
    build_training_rows,
    configure_tokenizer,
    make_reward_function,
)
from sme_negotiator_env.models import LiquidityObservation, NegotiationAction


def _ensure_grpo_response_schema(tokenizer: Any) -> Any:
    """Set a tool-response schema for TRL versions that require one."""
    if getattr(tokenizer, "response_schema", None) is not None:
        return tokenizer

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


def _cuda_training_dtype() -> Any:
    """Return the lowest-risk CUDA dtype for Colab-class demo training."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_demo_model(model_cls: Any, model_name: str) -> Any:
    """Load the demo model with memory-conscious defaults."""
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    torch_dtype = _cuda_training_dtype()
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    try:
        return model_cls.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("low_cpu_mem_usage", None)
        return model_cls.from_pretrained(model_name, **model_kwargs)


def _prepare_demo_model_for_training(model: Any, *, use_lora: bool) -> Any:
    """Apply memory-saving training settings for tiny GRPO demos."""
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if not use_lora:
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("[demo_train_grpo] peft is not installed; falling back to full-parameter training.")
        return model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
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
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def make_demo_action(observation: LiquidityObservation) -> NegotiationAction:
    """Return a deterministic baseline action for notebook demos."""
    if not observation.open_deal_ids:
        return NegotiationAction(action_type="advance_period")

    deal_id = observation.active_deal_id or observation.open_deal_ids[0]
    use_treds = bool(observation.buyer_days > observation.liquidity_threshold)
    if observation.last_tool_name != "QUERY_TREDS" and use_treds:
        return NegotiationAction(
            action_type="tool",
            deal_id=deal_id,
            tool_name="QUERY_TREDS",
            tool_args={"invoice_id": deal_id, "deal_id": deal_id},
        )

    return NegotiationAction(
        action_type="accept",
        deal_id=deal_id,
        price=float(observation.buyer_price),
        payment_days=int(observation.buyer_days),
        use_treds=use_treds,
        reason="Deterministic demo policy acceptance",
    )


def _action_to_text(action: NegotiationAction) -> str:
    payload = action.model_dump(exclude_none=True)
    parts = [f"{key}={value}" for key, value in payload.items()]
    return ", ".join(parts)


def run_heuristic_episode(
    *,
    seed: int = 0,
    total_periods: int = 2,
    task_name: str = "liquidity-correlation-hard",
    difficulty: str = "hard",
    max_steps: int = 20,
) -> dict[str, Any]:
    """Run one deterministic heuristic episode for notebook baselines."""
    wrapper_cls = make_environment_factory(
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        seed=seed,
        prompt=DEFAULT_PROMPT,
    )
    wrapper = wrapper_cls()
    wrapper.reset(
        seed=seed,
        difficulty=difficulty,
        task_name=task_name,
        total_periods=total_periods,
        prompt=DEFAULT_PROMPT,
    )
    observation = wrapper.last_observation
    assert observation is not None
    total_reward = float(wrapper.env_reward_total)
    transcript_lines = [f"RESET :: {format_observation(observation)}"]

    for step_index in range(max_steps):
        if observation.done:
            break
        action = make_demo_action(observation)
        if action.action_type == "tool":
            wrapper.query_treds(invoice_id=action.deal_id or "")
        elif action.action_type == "accept":
            wrapper.accept(
                price=float(action.price),
                payment_days=int(action.payment_days),
                use_treds=bool(action.use_treds),
                deal_id=action.deal_id,
                reason=action.reason,
            )
        elif action.action_type == "advance_period":
            wrapper.advance_period()
        else:
            raise ValueError(f"Unexpected heuristic demo action: {action.action_type!r}")
        observation = wrapper.last_observation
        assert observation is not None
        total_reward = float(wrapper.env_reward_total)
        transcript_lines.append(
            f"STEP {step_index + 1} :: action[{_action_to_text(action)}] :: reward={float(observation.reward):.6f}"
        )
        transcript_lines.append(f"OBS {step_index + 1} :: {format_observation(observation)}")
        if observation.done:
            break

    episode_summary = wrapper.summarize_episode()
    summary = {
        "base_rl_reward": float(episode_summary.base_rl_reward),
        "verifiable_reward": float(episode_summary.verifiable_reward),
        "total_reward": float(episode_summary.total_reward),
        "tool_bonus_total": float(episode_summary.tool_bonus_total),
        "env_reward_total": float(episode_summary.env_reward_total),
        "success_no_default_positive_npv": bool(episode_summary.success_no_default_positive_npv),
        "average_final_payment_days": float(episode_summary.average_final_payment_days),
        "tool_usage_count": int(episode_summary.tool_usage_count),
        "tool_call_count": int(episode_summary.tool_call_count),
        "tool_effective_count": int(episode_summary.tool_effective_count),
        "duplicate_tool_count": int(episode_summary.duplicate_tool_count),
        "invalid_action_count": int(episode_summary.invalid_action_count),
        "stall_step_count": int(episode_summary.stall_step_count),
        "resolved_deal_count": int(episode_summary.resolved_deal_count),
        "defaulted_sme_count": int(episode_summary.defaulted_sme_count),
        "terminated_by_step_cap": bool(episode_summary.terminated_by_step_cap),
    }

    return {
        "seed": seed,
        "total_reward": round(total_reward, 6),
        "steps": max(0, len(transcript_lines) // 2),
        "done": bool(observation.done),
        "transcript": "\n".join(transcript_lines),
        "summary": summary,
    }


def demo_train_grpo(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    steps: int = 10,
    total_periods: int = 2,
    task_name: str = "liquidity-correlation-hard",
    difficulty: str = "hard",
    num_samples: int = 8,
    seed_base: int = 1000,
    output_dir: str = "outputs/grpo_sme_liquidity_demo",
    use_lora: bool = True,
    max_completion_length: int = 256,
) -> dict[str, Any]:
    """Run a tiny in-process GRPO demo and return reward history."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError("TRL demo training requires the optional RL dependencies: pip install -e \".[rl]\"") from exc

    class HistoryCallback(TrainerCallback):
        """Collect logged reward summaries for plotting."""

        def __init__(self) -> None:
            self.steps: list[int] = []
            self.avg_reward: list[float] = []

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            reward_key = None
            if logs and "episode/avg_total_reward" in logs:
                reward_key = "episode/avg_total_reward"
            elif logs and "episode/avg_base_rl_reward" in logs:
                reward_key = "episode/avg_base_rl_reward"
            if logs and reward_key is not None:
                self.steps.append(int(getattr(state, "global_step", 0) or 0))
                self.avg_reward.append(float(logs[reward_key]))
            return control

    rows = build_training_rows(
        prompt=DEFAULT_PROMPT,
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        num_samples=num_samples,
        seed_base=seed_base,
    )
    dataset = build_dataset(rows)
    wrapper_cls = make_environment_factory(
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        seed=seed_base,
        prompt=DEFAULT_PROMPT,
    )
    summary_buffer = EpisodeSummaryBuffer()
    reward_func = make_reward_function(summary_buffer=summary_buffer)
    output_path = Path(output_dir)

    tokenizer = _ensure_grpo_response_schema(configure_tokenizer(AutoTokenizer.from_pretrained(model_name)))
    model = _prepare_demo_model_for_training(
        _load_demo_model(AutoModelForCausalLM, model_name),
        use_lora=use_lora,
    )
    metrics_callback = build_metrics_callback(
        summary_buffer,
        TrainerCallback,
        curriculum=None,
        build_preference_dataset=False,
        scorer=None,
        output_dir=str(output_path),
    )
    history_callback = HistoryCallback()
    grpo_kwargs = {
        "output_dir": str(output_path),
        "remove_unused_columns": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "num_generations": 2,
        "generation_batch_size": 2,
        "learning_rate": 5e-6,
        "max_prompt_length": 512,
        "max_completion_length": max(64, int(max_completion_length)),
        "max_steps": max(1, int(steps)),
        "logging_steps": 1,
        "save_steps": max(1, int(steps)),
        "report_to": _training_log_backend(),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    torch_dtype = _cuda_training_dtype()
    if torch_dtype is not None:
        dtype_name = str(torch_dtype)
        grpo_kwargs["bf16"] = dtype_name == "torch.bfloat16"
        grpo_kwargs["fp16"] = dtype_name == "torch.float16"
    valid_grpo_keys = set(inspect.signature(GRPOConfig.__init__).parameters) - {"self"}
    unsupported = sorted(key for key in grpo_kwargs if key not in valid_grpo_keys)
    if unsupported:
        print(f"[demo_train_grpo] Dropping unsupported GRPOConfig args for this TRL version: {unsupported}")
    training_args = GRPOConfig(**{key: value for key, value in grpo_kwargs.items() if key in valid_grpo_keys})

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=wrapper_cls,
        callbacks=[metrics_callback, history_callback],
    )
    trainer.train()

    checkpoint_path = output_path / "final-demo-model"
    trainer.save_model(str(checkpoint_path))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(checkpoint_path))

    return {
        "steps": history_callback.steps,
        "avg_reward": history_callback.avg_reward,
        "output_dir": str(output_path),
        "checkpoint_path": str(checkpoint_path),
    }


def run_policy_episode(
    *,
    policy: str = "heuristic",
    seed: int = 123,
    total_periods: int = 2,
    task_name: str = "liquidity-correlation-hard",
    difficulty: str = "hard",
    checkpoint_path: Optional[str] = None,
    max_steps: int = 20,
) -> str:
    """Run one heuristic or model-driven episode and return a compact transcript."""
    if policy == "heuristic":
        return str(
            run_heuristic_episode(
                seed=seed,
                total_periods=total_periods,
                task_name=task_name,
                difficulty=difficulty,
                max_steps=max_steps,
            )["transcript"]
        )

    if policy != "trained":
        raise ValueError("policy must be either 'heuristic' or 'trained'.")
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required when policy='trained'.")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Running a trained policy requires transformers to be installed.") from exc

    checkpoint = Path(checkpoint_path)
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(checkpoint_path))
    try:
        from peft import PeftConfig, PeftModel
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    else:
        try:
            peft_config = PeftConfig.from_pretrained(str(checkpoint))
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        else:
            base_model_name = str(getattr(peft_config, "base_model_name_or_path", "") or DEFAULT_MODEL_NAME)
            model = PeftModel.from_pretrained(
                _load_demo_model(AutoModelForCausalLM, base_model_name),
                str(checkpoint),
            )
    if hasattr(model, "eval"):
        model.eval()
    env = SMELiquidityEnvironment(total_periods=total_periods)
    observation = env.reset(seed=seed, difficulty=difficulty, task_name=task_name)
    transcript_lines = [f"RESET :: {format_observation(observation)}"]

    for step_index in range(max_steps):
        if observation.done:
            break
        prompt = format_observation(observation)
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=160)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        action = parse_action(text, observation)
        observation = env.step(action)
        transcript_lines.append(
            f"STEP {step_index + 1} :: action[{_action_to_text(action)}] :: reward={float(observation.reward):.6f}"
        )
        transcript_lines.append(f"OBS {step_index + 1} :: {format_observation(observation)}")
        if observation.done:
            break

    return "\n".join(transcript_lines)
