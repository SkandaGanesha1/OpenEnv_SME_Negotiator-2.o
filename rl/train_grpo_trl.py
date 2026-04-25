#!/usr/bin/env python3
"""Canonical TRL GRPO training entrypoint for the liquidity environment.

Install with optional extras such as:
    pip install -e ".[rl]"
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

from rl.bridge import NegotiatorEnvFactory, make_environment_factory
from rl.curriculum import CurriculumManager, DEFAULT_CURRICULUM_LEVELS
from rl.episode_logging import EpisodeSummary, combine_rewards
from rl.opponents import OpponentPolicyManager
from rl.reward_functions import make_all_reward_funcs
from rl.rubrics import persona_reward
from rl.self_rewarding_dpo import build_preference_examples, write_preference_dataset

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = (
    "You are an SME treasury agent operating a long-horizon liquidity workflow. "
    "Use explicit tools when useful, negotiate responsibly across multiple deals, "
    "advance macro periods when appropriate, and finish the episode without default."
)


@dataclass
class EpisodeSummaryBuffer:
    """Mutable buffer shared by reward functions, callbacks, and DPO export."""

    items: list[EpisodeSummary] = field(default_factory=list)
    episode_logs: list[str] = field(default_factory=list)

    def append(self, summary: EpisodeSummary, episode_log: Optional[str] = None) -> None:
        self.items.append(summary)
        if episode_log is not None:
            self.episode_logs.append(episode_log)

    def drain(self) -> tuple[list[EpisodeSummary], list[str]]:
        summaries = list(self.items)
        episode_logs = list(self.episode_logs)
        self.items.clear()
        self.episode_logs.clear()
        return summaries, episode_logs


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
    return [
        {
            "prompt": [{"role": "user", "content": prompt}],
            "task_name": str(task_name),
            "difficulty": str(difficulty),
            "seed": int(seed_base) + index,
            "total_periods": int(total_periods),
        }
        for index in range(int(num_samples))
    ]


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


def make_reward_function(
    *,
    rubric_scorer=None,
    rubric_weight: float = 0.0,
    summary_buffer: Optional[EpisodeSummaryBuffer] = None,
) -> Callable[[list[Any]], list[float]]:
    """Build the TRL reward function that reads deterministic env state."""

    def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for env in environments:
            # Prefer the verifiable reward (solvency+liquidity+NPV+compliance) when
            # world state is available; fall back to the environment's own scorer.
            inner_env = getattr(env, "env", env)
            world_state = getattr(inner_env, "_world_state", None)
            trajectory = getattr(env, "_trajectory_states", [])
            if world_state is not None and trajectory:
                try:
                    from sme_negotiator_env.graders import compute_verifiable_reward  # type: ignore[import]
                    base_reward = float(compute_verifiable_reward(world_state, trajectory))
                except Exception:
                    base_reward = float(env.compute_final_reward())
            else:
                base_reward = float(env.compute_final_reward())
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
                summary_buffer.append(env.summarize_episode(), episode_log)
        return rewards

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
    """Build a zero-arg TRL environment factory that reads live curriculum state."""

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


def build_metrics_callback(
    summary_buffer: EpisodeSummaryBuffer,
    trainer_callback_base: type[Any],
    *,
    curriculum: Optional[CurriculumManager],
    build_preference_dataset: bool,
    scorer,
    output_dir: str,
):
    """Create a TrainerCallback that logs rolling episode means and updates the curriculum."""

    class RollingEpisodeMetricsCallback(trainer_callback_base):
        """Attach summarized episode metrics, curriculum promotion, and optional preference export."""

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            summaries, episode_logs = summary_buffer.drain()
            if summaries and logs is not None:
                logs.update(summarize_batch(summaries))
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
                    output_path = Path(output_dir) / "preferences" / f"step_{int(getattr(state, 'global_step', 0)):06d}.jsonl"
                    write_preference_dataset(output_path, examples)
                    logs["preferences/written"] = float(len(examples))
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
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 4,
        "learning_rate": 5e-6,
        "max_prompt_length": 512,
        "max_completion_length": 1536,
        "logging_steps": 1,
        "save_steps": 50,
        "report_to": "none",
        "use_vllm": bool(args.use_vllm),
    }
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


def print_dry_run_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    env_factory,
    *,
    curriculum: Optional[CurriculumManager],
    opponent_manager: Optional[OpponentPolicyManager],
    rubric_scorer,
) -> None:
    """Print the resolved Stage 6 training configuration without loading a model."""
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

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    dataset = build_dataset(rows)
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(args.model_name))
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    summary_buffer = EpisodeSummaryBuffer()

    # Split reward functions: [outcome, format, process, anti_hack]
    # Weights:               [1.0,     0.3,    0.2,     1.0]
    reward_funcs, reward_weights = make_all_reward_funcs(
        rubric_scorer=rubric_scorer,
        rubric_weight=args.rubric_weight,
        summary_buffer=summary_buffer,
    )

    grpo_kwargs = build_grpo_config_kwargs(args)
    grpo_kwargs["reward_weights"] = reward_weights
    grpo_kwargs["log_completions"] = True   # enables generation inspection
    training_args = GRPOConfig(**grpo_kwargs)

    # Build monitoring callback (reward hacking detector + per-component logs)
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
            curriculum=curriculum,
            build_preference_dataset=args.build_preference_dataset,
            scorer=rubric_scorer or _default_fake_rubric_scorer,
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
    if monitoring_cb is not None:
        callbacks.append(monitoring_cb)

    # NegotiatorEnvFactory is the drop-in environment_factory for GRPOTrainer.
    # It exposes propose_terms / accept_offer / reject_offer / use_tool /
    # advance_period as tools and surfaces reward_breakdown for split reward fns.
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        args=training_args,
        environment_factory=NegotiatorEnvFactory,
        callbacks=callbacks,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
