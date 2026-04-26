#!/usr/bin/env python3
"""Notebook-facing GRPO helpers for the SME liquidity environment.

This module mirrors the role that ``kube_sre_gym.train`` plays in the
reference notebook: it is the reusable, environment-specific source of truth
for notebook orchestration and the simple CLI training entrypoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rl.bridge import get_exposed_environment_method_names
from rl.demo import (
    evaluate_before_after_policies,
    plot_rewards,
    save_run_manifest,
    save_training_dashboard,
    summarize_training_trustworthiness,
)
from rl.train_grpo_trl import (
    DEFAULT_MODEL_NAME,
    build_environment_factory,
    build_training_rows,
    create_trainer as create_canonical_trainer,
    build_training_session as build_canonical_training_session,
    make_training_args as make_trl_training_args,
    run_training_session as run_canonical_training_session,
)

DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_TASK_NAME = "liquidity-correlation-hard"
DEFAULT_DIFFICULTY = "hard"
DEFAULT_TOTAL_PERIODS = 2

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "tiny": {
        "num_samples": 8,
        "max_steps": 4,
        "max_episode_steps": 8,
        "num_generations": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
    },
    "small": {
        "num_samples": 16,
        "max_steps": 8,
        "max_episode_steps": 10,
        "num_generations": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
    },
    "standard": {
        "num_samples": 32,
        "max_steps": 20,
        "max_episode_steps": 12,
        "num_generations": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
    },
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-") or "run"


def _default_output_dir(*, profile: str, model_name: str, root: str = DEFAULT_OUTPUT_ROOT) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(root) / f"grpo-sme-liquidity-{profile}-{_slugify(model_name)}-{timestamp}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a notebook-like CLI for canonical liquidity GRPO training."""
    parser = argparse.ArgumentParser(description="Run a simple GRPO training session for SME liquidity.")
    parser.add_argument("--profile", choices=tuple(_PROFILE_DEFAULTS), default="tiny")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY)
    parser.add_argument("--total-periods", type=int, default=DEFAULT_TOTAL_PERIODS)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--rubric-weight", type=float, default=0.0)
    parser.add_argument("--runtime-backend", choices=("environment", "legacy"), default="environment")
    parser.add_argument("--use-vllm", dest="use_vllm", action="store_true")
    parser.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-max-model-length", type=int, default=None)
    parser.add_argument("--scale-rewards", default="none")
    parser.add_argument(
        "--mask-truncated-completions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--log-completions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strict-trustworthiness", action="store_true")
    parser.add_argument("--eval-num-seeds", type=int, default=4)
    parser.add_argument("--eval-seed-offset", type=int, default=10000)
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(use_vllm=True)
    return parser


def make_training_args(**overrides: Any) -> argparse.Namespace:
    """Return parser-backed liquidity training args with optional overrides."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    valid_keys = {
        action.dest
        for action in parser._actions
        if getattr(action, "dest", None) not in {None, "help"}
    }
    unknown = sorted(set(overrides) - valid_keys)
    if unknown:
        raise KeyError(f"Unknown liquidity training arg overrides: {unknown}")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def resolve_profile_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge explicit CLI overrides onto a named liquidity training profile."""
    defaults = dict(_PROFILE_DEFAULTS[str(args.profile)])
    return {
        "profile": str(args.profile),
        "num_samples": int(args.num_samples if args.num_samples is not None else defaults["num_samples"]),
        "max_steps": int(args.max_steps if args.max_steps is not None else defaults["max_steps"]),
        "max_episode_steps": int(
            args.max_episode_steps if args.max_episode_steps is not None else defaults["max_episode_steps"]
        ),
        "num_generations": int(args.num_generations if args.num_generations is not None else defaults["num_generations"]),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else defaults["gradient_accumulation_steps"]
        ),
        "learning_rate": float(args.learning_rate if args.learning_rate is not None else defaults["learning_rate"]),
    }


def build_canonical_training_args(args: argparse.Namespace) -> argparse.Namespace:
    """Translate the notebook-friendly CLI into canonical TRL training args."""
    profile = resolve_profile_config(args)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(
        profile=profile["profile"],
        model_name=args.model_name,
    )
    return make_trl_training_args(
        model_name=args.model_name,
        task_name=args.task_name,
        difficulty=args.difficulty,
        total_periods=args.total_periods,
        seed_base=args.seed_base,
        output_dir=str(output_dir),
        num_samples=profile["num_samples"],
        max_steps=profile["max_steps"],
        max_episode_steps=profile["max_episode_steps"],
        num_generations=profile["num_generations"],
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        learning_rate=profile["learning_rate"],
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        use_vllm=bool(args.use_vllm),
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=args.mask_truncated_completions,
        log_completions=args.log_completions,
        rubric_weight=args.rubric_weight,
        runtime_backend="environment",
    )


def build_run_plan(args: argparse.Namespace, canonical_args: argparse.Namespace) -> dict[str, Any]:
    """Build a compact JSON-serializable run plan for notebook display and manifests."""
    profile = resolve_profile_config(args)
    output_dir = Path(canonical_args.output_dir)
    return {
        "profile": profile["profile"],
        "task": {
            "model_name": canonical_args.model_name,
            "task_name": canonical_args.task_name,
            "difficulty": canonical_args.difficulty,
            "total_periods": int(canonical_args.total_periods),
            "seed_base": int(canonical_args.seed_base),
        },
        "training": {
            "runtime_backend": str(canonical_args.runtime_backend),
            "num_samples": int(canonical_args.num_samples),
            "max_steps": int(canonical_args.max_steps) if canonical_args.max_steps is not None else None,
            "max_episode_steps": int(canonical_args.max_episode_steps),
            "num_generations": int(canonical_args.num_generations),
            "gradient_accumulation_steps": int(canonical_args.gradient_accumulation_steps),
            "learning_rate": float(canonical_args.learning_rate),
            "max_completion_length": int(canonical_args.max_completion_length),
            "use_vllm": bool(canonical_args.use_vllm),
            "vllm_mode": str(canonical_args.vllm_mode),
            "vllm_gpu_memory_utilization": float(canonical_args.vllm_gpu_memory_utilization),
            "vllm_max_model_length": None
            if getattr(canonical_args, "vllm_max_model_length", None) is None
            else int(canonical_args.vllm_max_model_length),
            "scale_rewards": getattr(canonical_args, "scale_rewards", None),
            "mask_truncated_completions": getattr(canonical_args, "mask_truncated_completions", None),
            "log_completions": bool(getattr(canonical_args, "log_completions", True)),
            "rubric_weight": float(canonical_args.rubric_weight),
            "strict_trustworthiness": bool(args.strict_trustworthiness),
            "eval_num_seeds": int(args.eval_num_seeds),
            "eval_seed_offset": int(args.eval_seed_offset),
        },
        "paths": {
            "output_dir": str(output_dir.resolve()),
            "final_checkpoint": str((output_dir / "final-grpo-model").resolve()),
            "reward_log": str((output_dir / "episode_reward_log.json").resolve()),
            "episode_reward_log": str((output_dir / "episode_reward_log.json").resolve()),
            "trainer_reward_log": str((output_dir / "reward_log.json").resolve()),
            "reward_curve": str((output_dir / "reward_curve.png").resolve()),
            "training_dashboard": str((output_dir / "training_dashboard.png").resolve()),
            "run_manifest": str((output_dir / "run_manifest.json").resolve()),
        },
    }


def _write_episode_reward_log(records: list[dict[str, Any]], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def _held_out_eval_seeds(*, seed_offset: int, num_seeds: int) -> list[int]:
    count = max(1, int(num_seeds))
    base = int(seed_offset)
    return [base + index for index in range(count)]


def _run_trustworthiness_evaluations(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    model_name: str,
    eval_num_seeds: int,
    eval_seed_offset: int,
) -> dict[str, Any]:
    panels = (
        ("liquidity-stress-medium", "medium", 3),
        ("liquidity-correlation-hard", "hard", 2),
    )
    by_task: dict[str, Any] = {}
    seed_panels: dict[str, list[int]] = {}
    all_passed = True
    for index, (task_name, difficulty, total_periods) in enumerate(panels):
        seeds = _held_out_eval_seeds(
            seed_offset=eval_seed_offset + index * max(100, eval_num_seeds),
            num_seeds=eval_num_seeds,
        )
        seed_panels[task_name] = seeds
        result = evaluate_before_after_policies(
            output_dir=str(output_dir / task_name),
            seeds=seeds,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_path=str(checkpoint_path),
            model_name=model_name,
            max_steps=24,
            include_heuristic=True,
        )
        summary = dict(result["summary"])
        metadata = dict(summary.get("metadata", {}) or {})
        panel_passed = bool(metadata.get("trained_beats_base_without_extra_defaults", False))
        all_passed = all_passed and panel_passed
        by_task[task_name] = {
            "difficulty": difficulty,
            "total_periods": total_periods,
            "seeds": seeds,
            "trained_beats_base_without_extra_defaults": panel_passed,
            "summary": summary,
            "eval_summary_path": result["eval_summary_path"],
            "policy_comparison_path": result["policy_comparison_path"],
        }
    return {
        "tasks": by_task,
        "seed_panels": seed_panels,
        "trained_beats_base_without_extra_defaults": all_passed,
    }


def _build_training_trust_report(
    *,
    runtime_backend: str,
    dashboard: dict[str, Any],
    episode_reward_history: list[dict[str, Any]],
    exposed_tool_names: list[str],
    eval_summary: dict[str, Any],
) -> dict[str, Any]:
    expected_tools = list(get_exposed_environment_method_names())
    unexpected_tools = sorted(set(exposed_tool_names) - set(expected_tools))
    missing_tools = sorted(set(expected_tools) - set(exposed_tool_names))
    helper_tools = sorted(
        tool_name
        for tool_name in exposed_tool_names
        if tool_name in {"build_episode_log", "compute_final_reward", "summarize_episode"}
    )
    trust_metrics = dict(dashboard.get("trust_metrics", {}) or {})
    if not trust_metrics:
        trust_metrics = summarize_training_trustworthiness([])
    reward_shaping_gap = round(
        trust_metrics.get("mean_total_reward", 0.0) - trust_metrics.get("mean_verifiable_reward", 0.0),
        6,
    )
    trust_metrics["reward_shaping_gap"] = reward_shaping_gap
    trust_metrics["exposed_tool_count"] = float(len(exposed_tool_names))
    trust_metrics["episode_reward_history_count"] = float(len(episode_reward_history))
    trust_metrics["held_out_eval_passed"] = 1.0 if bool(
        eval_summary.get("trained_beats_base_without_extra_defaults", False)
    ) else 0.0

    trust_failures = list(dashboard.get("trust_failures", []) or [])
    if runtime_backend != "environment":
        trust_failures.append("runtime_backend_not_environment")
    if helper_tools:
        trust_failures.append("internal_helper_tools_exposed")
    if unexpected_tools:
        trust_failures.append("unexpected_tool_exposure")
    if missing_tools:
        trust_failures.append("missing_canonical_tools")
    if not bool(eval_summary.get("trained_beats_base_without_extra_defaults", False)):
        trust_failures.append("held_out_eval_no_improvement")
    if reward_shaping_gap > 0.25 and trust_metrics.get("mean_verifiable_reward", 0.0) <= 0.0:
        trust_failures.append("reward_shaping_dominates_verifiable_reward")

    deduped_failures = list(dict.fromkeys(str(item) for item in trust_failures))
    return {
        "training_trustworthy": not deduped_failures,
        "trust_failures": deduped_failures,
        "trust_metrics": trust_metrics,
        "unexpected_tools": unexpected_tools,
        "missing_tools": missing_tools,
        "helper_tools": helper_tools,
        "expected_tools": expected_tools,
    }


def _validate_canonical_backend(args: argparse.Namespace) -> None:
    runtime_backend = str(getattr(args, "runtime_backend", "environment") or "environment")
    if runtime_backend == "environment":
        return
    raise RuntimeError(
        "Notebook/simple liquidity training only supports `runtime_backend='environment'`. "
        "The legacy rollout bridge remains an internal escape hatch and is not a supported canonical backend."
    )


def _require_vllm_installed() -> None:
    try:
        importlib.import_module("vllm")
    except Exception as exc:
        raise RuntimeError(
            "This notebook-facing GRPO path requires the `vllm` package. Rerun the notebook install cell, which "
            "should install `trl[vllm]` and `vllm`, then restart the runtime before training again."
        ) from exc


def smoke_test_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Run a cheap reset-only smoke test against the in-process environment."""
    rows = build_training_rows(
        task_name=args.task_name,
        difficulty=args.difficulty,
        total_periods=args.total_periods,
        num_samples=1,
        seed_base=args.seed_base,
    )
    env_factory = build_environment_factory(
        args,
        curriculum=None,
        opponent_manager=None,
    )
    wrapper = env_factory()
    preview_text = wrapper.reset(**rows[0])
    observation = getattr(wrapper, "last_observation", None)
    return {
        "seed": int(rows[0]["seed"]),
        "task_name": str(rows[0]["task_name"]),
        "difficulty": str(rows[0]["difficulty"]),
        "total_periods": int(rows[0]["total_periods"]),
        "observation_preview": str(preview_text),
        "buyer_price": None if observation is None else float(getattr(observation, "buyer_price", 0.0) or 0.0),
        "buyer_days": None if observation is None else int(getattr(observation, "buyer_days", 0) or 0),
        "liquidity_threshold": None
        if observation is None
        else int(getattr(observation, "liquidity_threshold", 0) or 0),
        "open_deal_ids": []
        if observation is None
        else [str(item) for item in list(getattr(observation, "open_deal_ids", []) or [])],
    }


def build_training_session(args: argparse.Namespace) -> dict[str, Any]:
    """Build a canonical liquidity GRPO session using notebook-friendly args."""
    _validate_canonical_backend(args)
    canonical_args = build_canonical_training_args(args)
    _validate_canonical_backend(canonical_args)
    if bool(canonical_args.use_vllm):
        _require_vllm_installed()
    return build_canonical_training_session(canonical_args)


def build_trainer(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    """Return a freshly built canonical liquidity training session and trainer."""
    session = build_training_session(args)
    return session, create_canonical_trainer(session)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run canonical liquidity GRPO training and save standard artifacts."""
    _validate_canonical_backend(args)
    canonical_args = build_canonical_training_args(args)
    _validate_canonical_backend(canonical_args)
    if bool(canonical_args.use_vllm):
        _require_vllm_installed()
    output_dir = Path(canonical_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke = None if args.skip_smoke_test else smoke_test_environment(canonical_args)
    result = run_canonical_training_session(canonical_args)
    episode_reward_log_path = _write_episode_reward_log(
        list(result.get("episode_reward_history", [])),
        output_path=output_dir / "episode_reward_log.json",
    )
    dashboard = save_training_dashboard(result["trainer"], output_dir=str(output_dir))
    reward_curve_path = Path(dashboard["reward_curve_path"])
    trainer_reward_log_path = Path(dashboard["reward_log_path"])
    plot_rewards(episode_reward_log_path, reward_curve_path)
    eval_summary = _run_trustworthiness_evaluations(
        output_dir=output_dir / "held_out_eval",
        checkpoint_path=Path(result["checkpoint_path"]),
        model_name=str(canonical_args.model_name),
        eval_num_seeds=int(args.eval_num_seeds),
        eval_seed_offset=int(args.eval_seed_offset),
    )
    exposed_tool_names = [str(name) for name in list(result.get("exposed_tool_names", []))]
    trust_report = _build_training_trust_report(
        runtime_backend=str(result.get("runtime_backend", canonical_args.runtime_backend)),
        dashboard=dashboard,
        episode_reward_history=list(result.get("episode_reward_history", [])),
        exposed_tool_names=exposed_tool_names,
        eval_summary=eval_summary,
    )

    manifest = {
        "run_type": "simple_grpo_training",
        "plan": build_run_plan(args, canonical_args),
        "smoke_test": smoke,
        "training": {
            "runtime_backend": str(result.get("runtime_backend", canonical_args.runtime_backend)),
            "environment_backend_valid": bool(
                str(result.get("runtime_backend", canonical_args.runtime_backend)) == "environment"
            ),
            "checkpoint_path": str(Path(result["checkpoint_path"]).resolve()),
            "reward_log_path": str(episode_reward_log_path.resolve()),
            "episode_reward_log_path": str(episode_reward_log_path.resolve()),
            "trainer_reward_log_path": str(trainer_reward_log_path.resolve()),
            "reward_curve_path": str(reward_curve_path.resolve()),
            "training_dashboard_path": str(Path(dashboard["training_dashboard_path"]).resolve()),
            "history_points": int(dashboard["history_points"]),
            "zero_variance_warning": bool(dashboard["zero_variance_warning"]),
            "training_trustworthy": bool(trust_report["training_trustworthy"]),
            "trust_failures": list(trust_report["trust_failures"]),
            "trust_metrics": dict(trust_report["trust_metrics"]),
            "median_reward_std": float(trust_report["trust_metrics"].get("median_reward_std", 0.0) or 0.0),
            "median_unique_completion_count": float(
                trust_report["trust_metrics"].get("median_unique_completion_count", 0.0) or 0.0
            ),
            "median_identical_terminal_fraction": float(
                trust_report["trust_metrics"].get("median_identical_terminal_fraction", 0.0) or 0.0
            ),
            "exposed_tool_names": exposed_tool_names,
        },
        "eval_summary": eval_summary,
    }
    manifest_path = save_run_manifest(manifest, output_dir=str(output_dir))
    manifest["manifest_path"] = manifest_path
    return manifest


def run_simple_training(args: argparse.Namespace) -> dict[str, Any]:
    """Backward-compatible alias for notebook and script callers."""
    return run_training(args)


def main(argv: Optional[list[str]] = None) -> int:
    """Run or preview the liquidity GRPO training entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    canonical_args = build_canonical_training_args(args)
    plan = build_run_plan(args, canonical_args)
    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "plan": plan,
        }
        if not args.skip_smoke_test:
            payload["smoke_test"] = smoke_test_environment(canonical_args)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    manifest = run_training(args)
    print(json.dumps(manifest, indent=2, default=str))
    if args.strict_trustworthiness and not bool(manifest["training"].get("training_trustworthy", False)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
