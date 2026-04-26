"""Notebook-facing helpers for tiny in-process liquidity demos."""

from __future__ import annotations

import base64
import copy
import inspect
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from rl.bridge import (
    SUPPORTED_ACTION_TYPES,
    SUPPORTED_TOOL_NAMES,
    execute_action,
    format_observation,
    make_environment_factory,
)
from rl.train_grpo_trl import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    _build_observation_message,
    _coerce_prompt_messages,
    _generate_completion_turn,
    _parse_action_and_validity,
    _training_log_backend,
    EpisodeSummaryBuffer,
    build_dataset,
    build_metrics_callback,
    build_training_rows,
    configure_rollout_tokenizer,
    make_reward_function,
    resolve_training_precision_kwargs,
)
from sme_negotiator_env.models import LiquidityObservation, NegotiationAction


class SummaryView(dict[str, Any]):
    """Dict-like summary that also supports attribute access.

    Notebook code sometimes uses `summary.foo` while tests and JSON paths use
    `summary["foo"]`. This view supports both patterns safely.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


_TRAINING_DASHBOARD_GROUPS = (
    ("Reward and Success", "Episode metric", ("episode/avg_total_reward", "episode/success_rate")),
    (
        "Rollout Diversity",
        "Std / Count",
        ("rollout/reward_std", "rollout/unique_action_count", "rollout/unique_completion_count"),
    ),
    (
        "Format Quality",
        "Fraction",
        ("rollout/invalid_parse_fraction", "rollout/identical_terminal_fraction"),
    ),
)
_POLICY_COMPARISON_METRICS = (
    ("mean_total_reward", "Mean Total Reward", True),
    ("mean_verifiable_reward", "Mean Verifiable Reward", True),
    ("success_rate", "Success Rate", True),
    ("mean_final_payment_days", "Mean Final Payment Days", False),
    ("default_rate", "Default Rate", False),
    ("timeout_or_stepcap_rate", "Timeout / Step-Cap Rate", False),
)
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=="
)
_SUSPICIOUS_COMPLETION_PATTERNS = (
    "globals(",
    "__dict__",
    "os.environ",
    "sys.modules",
    "subprocess",
    "eval(",
    "exec(",
    "import os",
    "import sys",
)


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
        model_kwargs["dtype"] = torch_dtype
    try:
        return model_cls.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            try:
                return model_cls.from_pretrained(model_name, **model_kwargs)
            except TypeError:
                pass
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
    """Return a deterministic liquidity-aware baseline action for notebook demos."""
    if not observation.open_deal_ids:
        return NegotiationAction(action_type="advance_period")

    deal_id = observation.active_deal_id or observation.open_deal_ids[0]
    buyer_days = int(observation.buyer_days)
    liquidity_threshold = int(observation.liquidity_threshold)
    cost_threshold = float(observation.cost_threshold)
    target_days = max(15, min(liquidity_threshold, int(observation.sme_supplier_payment_days) + 15))
    target_price = round(max(float(observation.buyer_price), cost_threshold + 3.0), 2)
    use_treds = bool(buyer_days > int(observation.sme_supplier_payment_days) + 10)

    if observation.last_tool_name != "RUN_CASHFLOW_SIM" and buyer_days > liquidity_threshold:
        return NegotiationAction(
            action_type="tool",
            deal_id=deal_id,
            tool_name="RUN_CASHFLOW_SIM",
            tool_args={
                "deal_id": deal_id,
                "plan": {
                    "deal_decisions": {
                        deal_id: {
                            "decision": "accept",
                            "price": target_price,
                            "payment_days": target_days,
                            "use_treds": use_treds,
                            "late_payment_penalty_agreed": True,
                            "propose_dynamic_discounting": True,
                            "dynamic_discount_annual_rate": 0.12,
                        }
                    }
                },
            },
        )

    if buyer_days > liquidity_threshold:
        return NegotiationAction(
            action_type="propose",
            deal_id=deal_id,
            price=target_price,
            payment_days=target_days,
            use_treds=use_treds,
            propose_late_payment_penalty_clause=True,
            propose_dynamic_discounting=True,
            dynamic_discount_annual_rate=0.12,
            reason="Reduce payment days toward liquidity threshold while preserving positive margin.",
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


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PLACEHOLDER_PNG)


def _episode_summary_view(episode_summary: Any) -> SummaryView:
    return SummaryView(
        {
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
    )


def _policy_episode_result(
    *,
    seed: int,
    transcript_lines: list[str],
    done: bool,
    summary: SummaryView,
    steps: int,
    policy: str,
    completion_records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "policy": policy,
        "total_reward": round(float(summary["total_reward"]), 6),
        "steps": int(steps),
        "done": bool(done),
        "transcript": "\n".join(transcript_lines),
        "summary": summary,
        "completion_records": list(completion_records or []),
    }


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
    transcript_lines = [f"RESET :: {format_observation(observation)}"]

    for step_index in range(max_steps):
        if observation.done:
            break
        action = make_demo_action(observation)
        if action.action_type == "tool":
            if action.tool_name == "RUN_CASHFLOW_SIM":
                wrapper.run_cashflow_sim(
                    plan=action.tool_args.get("plan") if action.tool_args else None,
                    horizon=action.simulation_horizon,
                    deal_id=action.deal_id,
                )
            else:
                wrapper.query_treds(invoice_id=action.deal_id or "")
        elif action.action_type == "propose":
            wrapper.propose(
                price=float(action.price),
                payment_days=int(action.payment_days),
                use_treds=bool(action.use_treds),
                deal_id=action.deal_id,
                reason=action.reason,
                propose_late_payment_penalty_clause=bool(action.propose_late_payment_penalty_clause),
                propose_dynamic_discounting=bool(action.propose_dynamic_discounting),
                dynamic_discount_annual_rate=float(action.dynamic_discount_annual_rate),
            )
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
        transcript_lines.append(
            f"STEP {step_index + 1} :: action[{_action_to_text(action)}] :: reward={float(observation.reward):.6f}"
        )
        transcript_lines.append(f"OBS {step_index + 1} :: {format_observation(observation)}")
        if observation.done:
            break

    episode_summary = wrapper.summarize_episode()
    summary = _episode_summary_view(episode_summary)

    return _policy_episode_result(
        seed=seed,
        transcript_lines=transcript_lines,
        done=bool(observation.done),
        summary=summary,
        steps=max(0, len(transcript_lines) // 2),
        policy="heuristic",
        completion_records=[],
    )


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

    tokenizer = configure_rollout_tokenizer(AutoTokenizer.from_pretrained(model_name))
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
        "num_generations": 4,
        "generation_batch_size": 4,
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
    grpo_kwargs.update(resolve_training_precision_kwargs())
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


def _coerce_history_entries(trainer_or_history: Any) -> list[dict[str, Any]]:
    if isinstance(trainer_or_history, list):
        return [entry for entry in trainer_or_history if isinstance(entry, dict)]
    state = getattr(trainer_or_history, "state", None)
    log_history = getattr(state, "log_history", []) if state is not None else []
    return [entry for entry in log_history if isinstance(entry, dict)]


def _metric_series(history: list[dict[str, Any]], name: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for entry in history:
        if name in entry and "step" in entry:
            steps.append(int(entry["step"]))
            values.append(float(entry[name]))
    return steps, values


def _write_reward_log(history: list[dict[str, Any]], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_path


def plot_rewards(reward_log_path: str | Path, output_path: str | Path) -> Path:
    """Render reward and success curves from a saved trainer log history file."""
    reward_log = Path(reward_log_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        history = json.loads(reward_log.read_text(encoding="utf-8"))
    except Exception:
        _write_placeholder_png(destination)
        return destination

    if not isinstance(history, list):
        _write_placeholder_png(destination)
        return destination

    reward_steps, reward_values = _metric_series(history, "episode/avg_total_reward")
    if not reward_values:
        reward_steps, reward_values = _metric_series(history, "episode/avg_base_rl_reward")
    success_steps, success_values = _metric_series(history, "episode/success_rate")

    if not reward_values and not success_values:
        _write_placeholder_png(destination)
        return destination

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_placeholder_png(destination)
        return destination

    fig, (ax_reward, ax_success) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    if reward_values:
        ax_reward.plot(reward_steps, reward_values, label="avg_total_reward")
        ax_reward.legend()
    else:
        ax_reward.text(0.5, 0.5, "No reward entries found.", ha="center", va="center", transform=ax_reward.transAxes)
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.3)

    if success_values:
        ax_success.plot(success_steps, success_values, label="success_rate", color="green")
        ax_success.legend()
    else:
        ax_success.text(0.5, 0.5, "No success-rate entries found.", ha="center", va="center", transform=ax_success.transAxes)
    ax_success.set_ylabel("Success Rate")
    ax_success.set_xlabel("Training step")
    ax_success.grid(True, alpha=0.3)

    fig.suptitle("SME Negotiator Reward Curve")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_training_dashboard(trainer_or_history: Any, *, output_dir: str) -> dict[str, Any]:
    """Save a readable multi-metric dashboard for notebook and README use."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = _coerce_history_entries(trainer_or_history)
    metrics: dict[str, float | None] = {}
    for _, _, group in _TRAINING_DASHBOARD_GROUPS:
        for metric_name in group:
            _, values = _metric_series(history, metric_name)
            metrics[metric_name] = values[-1] if values else None

    figure_path = output_path / "training_dashboard.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_placeholder_png(figure_path)
    else:
        fig, axes = plt.subplots(len(_TRAINING_DASHBOARD_GROUPS), 1, figsize=(11, 10), sharex=True)
        if len(_TRAINING_DASHBOARD_GROUPS) == 1:
            axes = [axes]

        for axis, (title, ylabel, group) in zip(axes, _TRAINING_DASHBOARD_GROUPS):
            plotted = False
            for metric_name in group:
                steps, values = _metric_series(history, metric_name)
                if values:
                    axis.plot(steps, values, label=metric_name)
                    plotted = True
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if plotted:
                axis.legend()
            else:
                axis.text(0.5, 0.5, "No logged data for this panel.", ha="center", va="center", transform=axis.transAxes)

        for axis in axes:
            axis.set_xlabel("Training step")
        fig.suptitle("SME Negotiator Training Dashboard")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        fig.savefig(figure_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    reward_log_path = _write_reward_log(history, output_path=output_path / "reward_log.json")
    reward_curve_path = output_path / "reward_curve.png"
    try:
        plot_rewards(reward_log_path, reward_curve_path)
    except Exception:
        _write_placeholder_png(reward_curve_path)

    zero_variance_streak = 0
    zero_variance_warning = False
    for entry in history:
        reward_std = float(entry.get("rollout/reward_std", 0.0) or 0.0)
        if reward_std <= 1e-8:
            zero_variance_streak += 1
        else:
            zero_variance_streak = 0
        if zero_variance_streak >= 2:
            zero_variance_warning = True
            break

    return {
        "training_dashboard_path": str(figure_path.resolve()),
        "reward_curve_path": str(reward_curve_path.resolve()),
        "reward_log_path": str(reward_log_path.resolve()),
        "metrics": metrics,
        "history_points": len(history),
        "zero_variance_warning": zero_variance_warning,
    }


def _load_policy_model(
    *,
    policy: str,
    model_name: str,
    checkpoint_path: Optional[str],
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Running a model policy requires transformers to be installed.") from exc

    if policy == "base":
        tokenizer = configure_rollout_tokenizer(AutoTokenizer.from_pretrained(model_name))
        model = _load_demo_model(AutoModelForCausalLM, model_name)
    elif policy == "trained":
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required when policy='trained'.")
        checkpoint = Path(checkpoint_path)
        tokenizer = configure_rollout_tokenizer(AutoTokenizer.from_pretrained(checkpoint_path))
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
                base_model_name = str(getattr(peft_config, "base_model_name_or_path", "") or model_name)
                model = PeftModel.from_pretrained(
                    _load_demo_model(AutoModelForCausalLM, base_model_name),
                    str(checkpoint),
                )
    else:
        raise ValueError("policy must be one of 'heuristic', 'base', or 'trained'.")

    if hasattr(model, "eval"):
        model.eval()
    return model, tokenizer


def run_policy_episode_report(
    *,
    policy: str = "heuristic",
    seed: int = 123,
    total_periods: int = 2,
    task_name: str = "liquidity-correlation-hard",
    difficulty: str = "hard",
    checkpoint_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_steps: int = 20,
) -> dict[str, Any]:
    """Run one heuristic, base-model, or trained-model episode and return a compact report."""
    if policy == "heuristic":
        return run_heuristic_episode(
            seed=seed,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            max_steps=max_steps,
        )

    model, tokenizer = _load_policy_model(
        policy=policy,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )
    rows = build_training_rows(
        prompt=DEFAULT_PROMPT,
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        num_samples=1,
        seed_base=seed,
    )
    row = rows[0]
    wrapper_cls = make_environment_factory(
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        seed=seed,
        prompt=row["prompt"],
    )
    wrapper = wrapper_cls()
    wrapper.reset(
        seed=seed,
        difficulty=difficulty,
        task_name=task_name,
        total_periods=total_periods,
        prompt=row["prompt"],
    )
    observation = wrapper.last_observation
    assert observation is not None
    messages = copy.deepcopy(_coerce_prompt_messages(row["prompt"]))
    messages.append({"role": "user", "content": _build_observation_message(format_observation(observation))})
    transcript_lines = [f"RESET :: {format_observation(observation)}"]
    completion_records: list[dict[str, Any]] = []
    executed_steps = 0

    for step_index in range(max_steps):
        if observation.done:
            break
        turn = _generate_completion_turn(
            model,
            tokenizer,
            messages,
            max_new_tokens=160,
            temperature=0.0,
            top_p=1.0,
        )
        text = str(turn["text"])
        action, was_valid_json = _parse_action_and_validity(text, observation, strict_json=False)
        completion_records.append(
            {
                "step": int(step_index + 1),
                "valid_json": bool(was_valid_json),
                "raw_text": text,
                "action_type": str(getattr(action, "action_type", "") or ""),
                "tool_name": str(getattr(action, "tool_name", "") or ""),
            }
        )
        transcript_lines.append(f"MODEL {step_index + 1} :: valid_json={str(was_valid_json).lower()} :: raw={text}")
        messages.append({"role": "assistant", "content": text})
        try:
            execute_action(wrapper, action)
        except Exception as exc:
            transcript_lines.append(f"ERROR {step_index + 1} :: action_error={exc}")
            executed_steps = step_index + 1
            break
        observation = wrapper.last_observation
        assert observation is not None
        executed_steps = step_index + 1
        transcript_lines.append(
            f"STEP {step_index + 1} :: action[{_action_to_text(action)}] :: reward={float(observation.reward):.6f}"
        )
        transcript_lines.append(f"OBS {step_index + 1} :: {format_observation(observation)}")
        if observation.done:
            break
        messages.append({"role": "user", "content": _build_observation_message(format_observation(observation))})

    summary = _episode_summary_view(wrapper.summarize_episode())
    return _policy_episode_result(
        seed=seed,
        transcript_lines=transcript_lines,
        done=bool(getattr(observation, "done", False)),
        summary=summary,
        steps=executed_steps,
        policy=policy,
        completion_records=completion_records,
    )


def _aggregate_policy_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    if not runs:
        return {
            "episode_count": 0.0,
            "mean_total_reward": 0.0,
            "mean_verifiable_reward": 0.0,
            "success_rate": 0.0,
            "mean_final_payment_days": 0.0,
            "default_rate": 0.0,
            "timeout_or_stepcap_rate": 0.0,
        }

    summaries = [run["summary"] for run in runs]
    return {
        "episode_count": float(len(runs)),
        "mean_total_reward": round(mean(float(run["total_reward"]) for run in runs), 6),
        "mean_verifiable_reward": round(mean(float(summary["verifiable_reward"]) for summary in summaries), 6),
        "success_rate": round(
            mean(1.0 if bool(summary["success_no_default_positive_npv"]) else 0.0 for summary in summaries),
            6,
        ),
        "mean_final_payment_days": round(mean(float(summary["average_final_payment_days"]) for summary in summaries), 6),
        "default_rate": round(
            mean(1.0 if int(summary["defaulted_sme_count"]) > 0 else 0.0 for summary in summaries),
            6,
        ),
        "timeout_or_stepcap_rate": round(
            mean(1.0 if bool(summary["terminated_by_step_cap"]) else 0.0 for summary in summaries),
            6,
        ),
    }


def _slugify_artifact_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    return slug or "artifact"


def inspect_policy_reports(
    reports: list[dict[str, Any]],
    *,
    policy_label: str,
) -> dict[str, Any]:
    """Inspect sampled policy generations for malformed or suspicious behavior."""
    completion_records = [
        dict(record)
        for report in reports
        for record in report.get("completion_records", [])
        if isinstance(record, dict)
    ]
    completion_texts = [str(record.get("raw_text", "") or "").strip() for record in completion_records]
    action_types = [str(record.get("action_type", "") or "").strip() for record in completion_records]
    tool_names = [str(record.get("tool_name", "") or "").strip() for record in completion_records if record.get("tool_name")]
    invalid_count = sum(1 for record in completion_records if not bool(record.get("valid_json", False)))
    unsupported_action_count = sum(1 for action_type in action_types if action_type and action_type not in SUPPORTED_ACTION_TYPES)
    unsupported_tool_count = sum(1 for tool_name in tool_names if tool_name not in SUPPORTED_TOOL_NAMES)
    suspicious_hits = [
        text
        for text in completion_texts
        if any(pattern in text.lower() for pattern in _SUSPICIOUS_COMPLETION_PATTERNS)
    ]
    duplicate_tool_abuse_count = sum(max(0, count - 1) for count in Counter(tool_names).values())
    repeated_completion_fraction = 0.0
    if completion_texts:
        repeated_completion_fraction = round(
            max(0.0, 1.0 - (len(set(completion_texts)) / float(len(completion_texts)))),
            6,
        )
    warnings: list[str] = []
    if invalid_count > 0:
        warnings.append("Invalid JSON completions detected.")
    if unsupported_action_count > 0:
        warnings.append("Unsupported action schema drift detected.")
    if unsupported_tool_count > 0:
        warnings.append("Unsupported tool usage detected.")
    if duplicate_tool_abuse_count > 0:
        warnings.append("Repeated tool calls suggest possible tool abuse.")
    if repeated_completion_fraction >= 0.5 and completion_texts:
        warnings.append("Low completion diversity detected.")
    if suspicious_hits:
        warnings.append("Suspicious shortcut or environment-abuse patterns detected.")

    return {
        "policy_label": str(policy_label),
        "episode_count": int(len(reports)),
        "completion_count": int(len(completion_records)),
        "invalid_parse_fraction": round(
            float(invalid_count) / float(len(completion_records)) if completion_records else 0.0,
            6,
        ),
        "repeated_completion_fraction": repeated_completion_fraction,
        "unsupported_action_count": int(unsupported_action_count),
        "unsupported_tool_count": int(unsupported_tool_count),
        "duplicate_tool_abuse_count": int(duplicate_tool_abuse_count),
        "suspicious_pattern_count": int(len(suspicious_hits)),
        "suspicious_examples": suspicious_hits[:3],
        "warnings": warnings,
        "flagged": bool(warnings),
    }


def inspect_policy_checkpoint(
    *,
    output_dir: str,
    policy_label: str,
    seeds: list[int],
    total_periods: int,
    task_name: str,
    difficulty: str,
    policy: str = "trained",
    checkpoint_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_steps: int = 20,
) -> dict[str, Any]:
    """Run fixed-seed inspection episodes and save a JSON inspection summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports = [
        run_policy_episode_report(
            policy=policy,
            seed=seed,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            max_steps=max_steps,
        )
        for seed in seeds
    ]
    inspection = inspect_policy_reports(reports, policy_label=policy_label)
    payload = {
        "policy_label": str(policy_label),
        "policy": str(policy),
        "inspection": inspection,
        "reports": reports,
    }
    inspection_path = output_path / f"{_slugify_artifact_label(policy_label)}_inspection.json"
    inspection_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "inspection": inspection,
        "reports": reports,
        "inspection_path": str(inspection_path.resolve()),
    }


def save_run_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: str,
    filename: str = "run_manifest.json",
) -> str:
    """Persist a canonical manifest for notebook, README, or judge consumption."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / filename
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return str(manifest_path.resolve())


def evaluate_policy_checkpoints(
    *,
    output_dir: str,
    seeds: list[int],
    total_periods: int,
    task_name: str,
    difficulty: str,
    checkpoint_paths: dict[str, str],
    model_name: str = DEFAULT_MODEL_NAME,
    max_steps: int = 20,
    include_base: bool = True,
    include_heuristic: bool = False,
    artifact_prefix: str = "checkpoint_policy",
) -> dict[str, Any]:
    """Evaluate multiple trained checkpoints on the same seed panel."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runs_by_label: dict[str, list[dict[str, Any]]] = {}
    transcript_paths: dict[str, str] = {}
    if include_base:
        runs_by_label["base"] = [
            run_policy_episode_report(
                policy="base",
                seed=seed,
                total_periods=total_periods,
                task_name=task_name,
                difficulty=difficulty,
                model_name=model_name,
                max_steps=max_steps,
            )
            for seed in seeds
        ]
    for label, checkpoint_path in checkpoint_paths.items():
        runs_by_label[str(label)] = [
            run_policy_episode_report(
                policy="trained",
                seed=seed,
                total_periods=total_periods,
                task_name=task_name,
                difficulty=difficulty,
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                max_steps=max_steps,
            )
            for seed in seeds
        ]
    heuristic_reference = None
    if include_heuristic:
        heuristic_reference = run_policy_episode_report(
            policy="heuristic",
            seed=seeds[0] if seeds else 0,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            max_steps=max_steps,
        )

    comparison = {label: _aggregate_policy_runs(runs) for label, runs in runs_by_label.items()}

    figure_path = output_path / f"{artifact_prefix}_comparison.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_placeholder_png(figure_path)
    else:
        labels = list(comparison)
        metric_names = [metric_name for metric_name, _, _ in _POLICY_COMPARISON_METRICS]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes_flat = list(axes.flatten())
        for axis, (metric_name, title, higher_is_better) in zip(axes_flat, _POLICY_COMPARISON_METRICS):
            values = [comparison[label][metric_name] for label in labels]
            axis.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(labels)])
            axis.set_title(f"{title}{'' if higher_is_better else ' (lower is better)'}")
            axis.set_ylabel(title)
            axis.grid(True, axis="y", alpha=0.3)
            axis.tick_params(axis="x", rotation=15)
        fig.suptitle("Checkpoint Policy Comparison")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(figure_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    for label, runs in runs_by_label.items():
        transcript_path = output_path / f"{_slugify_artifact_label(label)}_transcript.txt"
        transcript_path.write_text(str(runs[0]["transcript"]) if runs else "", encoding="utf-8")
        transcript_paths[label] = str(transcript_path.resolve())

    heuristic_transcript_path = None
    if heuristic_reference is not None:
        heuristic_path = output_path / "heuristic_reference_transcript.txt"
        heuristic_path.write_text(str(heuristic_reference["transcript"]), encoding="utf-8")
        heuristic_transcript_path = str(heuristic_path.resolve())

    summary = {
        "metadata": {
            "task_name": task_name,
            "difficulty": difficulty,
            "total_periods": int(total_periods),
            "max_steps": int(max_steps),
            "model_name": model_name,
            "seeds": [int(seed) for seed in seeds],
            "labels": list(comparison),
        },
        "policies": comparison,
        "artifacts": {
            "comparison_path": str(figure_path.resolve()),
            "transcript_paths": transcript_paths,
            "heuristic_transcript_path": heuristic_transcript_path,
        },
    }
    summary_path = output_path / f"{artifact_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "summary_path": str(summary_path.resolve()),
        "comparison_path": str(figure_path.resolve()),
        "heuristic_transcript": str(heuristic_reference["transcript"]) if heuristic_reference is not None else "",
    }


def evaluate_before_after_policies(
    *,
    output_dir: str,
    seeds: list[int],
    total_periods: int,
    task_name: str,
    difficulty: str,
    checkpoint_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_steps: int = 20,
    include_heuristic: bool = True,
) -> dict[str, Any]:
    """Evaluate base vs trained policies on a fixed seed panel and save judge-ready artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_runs = [
        run_policy_episode_report(
            policy="base",
            seed=seed,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            model_name=model_name,
            max_steps=max_steps,
        )
        for seed in seeds
    ]
    trained_runs = [
        run_policy_episode_report(
            policy="trained",
            seed=seed,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            max_steps=max_steps,
        )
        for seed in seeds
    ]

    heuristic_reference = None
    if include_heuristic:
        heuristic_reference = run_policy_episode_report(
            policy="heuristic",
            seed=seeds[0] if seeds else 0,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            max_steps=max_steps,
        )

    comparison = {
        "base": _aggregate_policy_runs(base_runs),
        "trained": _aggregate_policy_runs(trained_runs),
    }
    primary_metrics = ("success_rate", "mean_verifiable_reward", "mean_total_reward")
    trained_primary_wins = sum(
        1 for metric_name in primary_metrics if comparison["trained"][metric_name] > comparison["base"][metric_name]
    )
    submission_ready = trained_primary_wins >= 2

    figure_path = output_path / "policy_comparison.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _write_placeholder_png(figure_path)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes_flat = list(axes.flatten())
        labels = ["Base model", "Trained checkpoint"]
        colors = ["#4C72B0", "#55A868"]
        for axis, (metric_name, title, higher_is_better) in zip(axes_flat, _POLICY_COMPARISON_METRICS):
            values = [comparison["base"][metric_name], comparison["trained"][metric_name]]
            axis.bar(labels, values, color=colors)
            axis.set_title(f"{title}{'' if higher_is_better else ' (lower is better)'}")
            axis.set_ylabel(title)
            axis.grid(True, axis="y", alpha=0.3)
        fig.suptitle("Base Model vs Trained Checkpoint")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(figure_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    base_transcript_path = output_path / "base_model_transcript.txt"
    trained_transcript_path = output_path / "trained_model_transcript.txt"
    base_transcript_path.write_text(str(base_runs[0]["transcript"]) if base_runs else "", encoding="utf-8")
    trained_transcript_path.write_text(str(trained_runs[0]["transcript"]) if trained_runs else "", encoding="utf-8")
    heuristic_transcript_path = None
    if heuristic_reference is not None:
        heuristic_transcript_path = output_path / "heuristic_reference_transcript.txt"
        heuristic_transcript_path.write_text(str(heuristic_reference["transcript"]), encoding="utf-8")

    eval_summary = {
        "metadata": {
            "task_name": task_name,
            "difficulty": difficulty,
            "total_periods": int(total_periods),
            "max_steps": int(max_steps),
            "model_name": model_name,
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "seeds": [int(seed) for seed in seeds],
            "submission_ready": submission_ready,
            "trained_primary_wins": int(trained_primary_wins),
        },
        "policies": comparison,
        "artifacts": {
            "policy_comparison_path": str(figure_path.resolve()),
            "base_transcript_path": str(base_transcript_path.resolve()),
            "trained_transcript_path": str(trained_transcript_path.resolve()),
            "heuristic_transcript_path": str(heuristic_transcript_path.resolve()) if heuristic_transcript_path else None,
        },
    }
    summary_path = output_path / "eval_summary.json"
    summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    return {
        "summary": eval_summary,
        "eval_summary_path": str(summary_path.resolve()),
        "policy_comparison_path": str(figure_path.resolve()),
        "base_transcript": str(base_runs[0]["transcript"]) if base_runs else "",
        "trained_transcript": str(trained_runs[0]["transcript"]) if trained_runs else "",
        "heuristic_transcript": str(heuristic_reference["transcript"]) if heuristic_reference is not None else "",
    }


def run_policy_episode(
    *,
    policy: str = "heuristic",
    seed: int = 123,
    total_periods: int = 2,
    task_name: str = "liquidity-correlation-hard",
    difficulty: str = "hard",
    checkpoint_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_steps: int = 20,
) -> str:
    """Run one policy episode and return the transcript only."""
    return str(
        run_policy_episode_report(
            policy=policy,
            seed=seed,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            max_steps=max_steps,
        )["transcript"]
    )
