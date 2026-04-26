#!/usr/bin/env python3
"""Optional Unsloth-accelerated GRPO training entrypoint.

Install with optional extras such as:
    pip install -e ".[rl,rl-unsloth]"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from rl.train_grpo_trl import (
    DEFAULT_PROMPT,
    _filter_kwargs_for_callable,
    _import_trl_grpo_symbols,
    _training_log_backend,
    EpisodeSummaryBuffer,
    build_curriculum_manager_from_args,
    build_dataset,
    build_environment_factory,
    build_metrics_callback,
    build_opponent_manager_from_args,
    build_snapshot_callback,
    build_training_rows,
    configure_tokenizer,
    load_rubric_scorer,
    print_dry_run_summary,
    resolve_training_precision_kwargs,
)
from rl.reward_functions import make_all_reward_funcs
from rl.train_grpo_trl import make_reward_function

DEFAULT_UNSLOTH_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Unsloth GRPO script."""
    parser = argparse.ArgumentParser(description="Train GRPO on the SME liquidity environment with Unsloth.")
    parser.add_argument("--model-name", default=DEFAULT_UNSLOTH_MODEL)
    parser.add_argument("--task-name", default="liquidity-correlation-hard")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--total-periods", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/grpo_sme_liquidity_unsloth")
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
    return parser


def build_grpo_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI args into an Unsloth-friendly GRPOConfig kwargs dictionary."""
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 6,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "max_prompt_length": 512,
        "max_completion_length": 1536,
        "max_steps": 250,
        "logging_steps": 1,
        "report_to": _training_log_backend(),
        "use_vllm": bool(args.use_vllm),
    }
    kwargs.update(resolve_training_precision_kwargs())
    if bool(args.use_vllm):
        kwargs["vllm_mode"] = args.vllm_mode
    return kwargs


def make_training_args(**overrides: Any) -> argparse.Namespace:
    """Return parser-backed Unsloth args with optional attribute overrides."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    valid_keys = {
        action.dest
        for action in parser._actions
        if getattr(action, "dest", None) not in {None, "help"}
    }
    unknown = sorted(set(overrides) - valid_keys)
    if unknown:
        raise KeyError(f"Unknown Unsloth training arg overrides: {unknown}")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_training_session(args: argparse.Namespace) -> dict[str, Any]:
    """Train and save an Unsloth GRPO session, returning notebook-friendly artifacts."""
    curriculum = build_curriculum_manager_from_args(args)
    opponent_manager = build_opponent_manager_from_args(args)
    rubric_scorer = load_rubric_scorer(args.judge_scorer, enable_rubrics=args.enable_rubrics)
    rows = build_training_rows(
        prompt=DEFAULT_PROMPT,
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

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed. Install optional extras with: pip install -e \".[rl,rl-unsloth]\""
        ) from exc

    from transformers import TrainerCallback

    GRPOConfig, GRPOTrainer = _import_trl_grpo_symbols()

    dataset = build_dataset(rows)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
    )
    tokenizer = configure_tokenizer(tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    summary_buffer = EpisodeSummaryBuffer()
    hybrid_reward_func = make_reward_function(
        rubric_scorer=rubric_scorer,
        rubric_weight=args.rubric_weight,
        summary_buffer=summary_buffer,
    )
    reward_funcs = [hybrid_reward_func]
    reward_weights = [1.0]
    callbacks = [
        build_metrics_callback(
            summary_buffer,
            TrainerCallback,
            curriculum=curriculum,
            build_preference_dataset=args.build_preference_dataset,
            scorer=rubric_scorer,
            output_dir=args.output_dir,
        )
    ]
    if args.enable_self_play:
        callbacks.append(
            build_snapshot_callback(
                TrainerCallback,
                opponent_manager=opponent_manager,
                interval=args.snapshot_interval,
                output_dir=args.output_dir,
            )
        )
    grpo_kwargs = build_grpo_config_kwargs(args)
    grpo_kwargs["reward_weights"] = reward_weights
    grpo_kwargs["log_completions"] = True
    filtered_grpo_kwargs, unsupported_grpo = _filter_kwargs_for_callable(GRPOConfig.__init__, grpo_kwargs)
    if unsupported_grpo:
        print(
            f"[train_grpo_unsloth] Dropping unsupported GRPOConfig args for this TRL version: {unsupported_grpo}",
            flush=True,
        )
    training_args = GRPOConfig(**filtered_grpo_kwargs)

    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": reward_funcs,
        "train_dataset": dataset,
        "args": training_args,
        "environment_factory": env_factory,
        "callbacks": callbacks,
    }
    filtered_trainer_kwargs, unsupported_trainer = _filter_kwargs_for_callable(GRPOTrainer.__init__, trainer_kwargs)
    if unsupported_trainer:
        raise TypeError(
            "Installed TRL version does not support the current Unsloth training path. "
            f"Unsupported trainer args: {unsupported_trainer}"
        )

    trainer = GRPOTrainer(**filtered_trainer_kwargs)
    trainer.train()

    checkpoint_path = Path(args.output_dir) / "final-grpo-model"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(checkpoint_path))
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(str(checkpoint_path))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(checkpoint_path))

    return {
        "session": {
            "rows": rows,
            "dataset": dataset,
            "summary_buffer": summary_buffer,
            "final_checkpoint_path": checkpoint_path,
            "training_args": training_args,
            "callbacks": callbacks,
        },
        "trainer": trainer,
        "checkpoint_path": checkpoint_path,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Run optional Unsloth GRPO training or print a dry-run summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    curriculum = build_curriculum_manager_from_args(args)
    opponent_manager = build_opponent_manager_from_args(args)
    rubric_scorer = load_rubric_scorer(args.judge_scorer, enable_rubrics=args.enable_rubrics)
    rows = build_training_rows(
        prompt=DEFAULT_PROMPT,
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

    if args.dry_run:
        print_dry_run_summary(
            args,
            rows,
            env_factory,
            curriculum=curriculum,
            opponent_manager=opponent_manager,
            rubric_scorer=rubric_scorer,
        )
        return 0

    run_training_session(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
