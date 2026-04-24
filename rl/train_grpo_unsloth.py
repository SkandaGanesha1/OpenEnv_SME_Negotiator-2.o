#!/usr/bin/env python3
"""Optional Unsloth-accelerated GRPO training entrypoint.

Install with optional extras such as:
    pip install -e ".[rl,rl-unsloth]"
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from rl.train_grpo_trl import (
    DEFAULT_PROMPT,
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
    make_reward_function,
    print_dry_run_summary,
)

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
        "report_to": "none",
        "use_vllm": bool(args.use_vllm),
    }
    if bool(args.use_vllm):
        kwargs["vllm_mode"] = args.vllm_mode
    return kwargs


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

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed. Install optional extras with: pip install -e \".[rl,rl-unsloth]\""
        ) from exc

    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

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
    reward_func = make_reward_function(
        rubric_scorer=rubric_scorer,
        rubric_weight=args.rubric_weight,
        summary_buffer=summary_buffer,
    )
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
    training_args = GRPOConfig(**build_grpo_config_kwargs(args))

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=env_factory,
        callbacks=callbacks,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
