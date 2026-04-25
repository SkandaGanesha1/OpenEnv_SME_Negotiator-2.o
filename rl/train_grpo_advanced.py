#!/usr/bin/env python3
"""Advanced notebook-first GRPO orchestration for the SME liquidity environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from rl.demo import (
    evaluate_before_after_policies,
    evaluate_policy_checkpoints,
    inspect_policy_checkpoint,
    save_run_manifest,
    save_training_dashboard,
)
from rl.train_grpo_trl import DEFAULT_MODEL_NAME, make_training_args as make_trl_training_args
from rl.train_grpo_trl import run_training_session as run_trl_training_session
from rl.train_grpo_unsloth import DEFAULT_UNSLOTH_MODEL, make_training_args as make_unsloth_training_args
from rl.train_grpo_unsloth import run_training_session as run_unsloth_training_session

DEFAULT_ADVANCED_OUTPUT_DIR = "outputs/grpo_sme_liquidity_advanced"
DEFAULT_TASK_NAME = "liquidity-correlation-hard"
DEFAULT_DIFFICULTY = "hard"
DEFAULT_TOTAL_PERIODS = 2

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "tiny": {
        "eval_seeds": [1000, 1001],
        "inspection_seeds": [1000, 1001],
        "eval_max_steps": 10,
        "tiny_trl": {
            "num_samples": 8,
            "max_steps": 4,
            "max_episode_steps": 8,
            "learning_rate": 5e-6,
        },
        "tiny_unsloth": {
            "num_samples": 8,
            "build_preference_dataset": False,
        },
        "main_trl": {
            "num_samples": 12,
            "max_steps": 8,
            "max_episode_steps": 10,
            "learning_rate": 5e-6,
        },
    },
    "stabilize": {
        "eval_seeds": [1000, 1001, 1002],
        "inspection_seeds": [1000, 1001, 1002],
        "eval_max_steps": 12,
        "tiny_trl": {
            "num_samples": 10,
            "max_steps": 6,
            "max_episode_steps": 10,
            "learning_rate": 5e-6,
        },
        "tiny_unsloth": {
            "num_samples": 10,
            "build_preference_dataset": False,
        },
        "main_trl": {
            "num_samples": 20,
            "max_steps": 16,
            "max_episode_steps": 12,
            "learning_rate": 5e-6,
        },
    },
    "submission": {
        "eval_seeds": [1000, 1001, 1002, 1003],
        "inspection_seeds": [1000, 1001, 1002, 1003],
        "eval_max_steps": 12,
        "tiny_trl": {
            "num_samples": 12,
            "max_steps": 6,
            "max_episode_steps": 10,
            "learning_rate": 5e-6,
        },
        "tiny_unsloth": {
            "num_samples": 12,
            "build_preference_dataset": False,
        },
        "main_trl": {
            "num_samples": 32,
            "max_steps": 25,
            "max_episode_steps": 12,
            "learning_rate": 5e-6,
        },
    },
}


def build_advanced_run_plan(
    *,
    profile: str,
    output_dir: str = DEFAULT_ADVANCED_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    unsloth_model_name: str = DEFAULT_UNSLOTH_MODEL,
    task_name: str = DEFAULT_TASK_NAME,
    difficulty: str = DEFAULT_DIFFICULTY,
    total_periods: int = DEFAULT_TOTAL_PERIODS,
) -> dict[str, Any]:
    """Build a decision-complete run plan for the advanced pipeline."""
    if profile not in _PROFILE_DEFAULTS:
        raise KeyError(f"Unsupported advanced profile: {profile!r}")

    root = Path(output_dir)
    profile_config = dict(_PROFILE_DEFAULTS[profile])
    plan = {
        "profile": str(profile),
        "output_dir": str(root.resolve()),
        "task": {
            "task_name": task_name,
            "difficulty": difficulty,
            "total_periods": int(total_periods),
            "model_name": model_name,
            "unsloth_model_name": unsloth_model_name,
        },
        "phases": [
            "tiny_trl_run",
            "tiny_unsloth_run",
            "tiny_policy_comparison",
            "hacking_inspection",
            "curriculum_decision",
            "main_trl_run",
            "before_after_evaluation",
            "manifest_export",
        ],
        "paths": {
            "root": str(root.resolve()),
            "tiny_trl": str((root / "tiny_trl").resolve()),
            "tiny_unsloth": str((root / "tiny_unsloth").resolve()),
            "reward_log": str((root / "reward_log.json").resolve()),
            "reward_curve": str((root / "reward_curve.png").resolve()),
            "training_dashboard": str((root / "training_dashboard.png").resolve()),
            "policy_comparison": str((root / "policy_comparison.png").resolve()),
            "eval_summary": str((root / "eval_summary.json").resolve()),
            "run_manifest": str((root / "run_manifest.json").resolve()),
        },
        "profile_defaults": profile_config,
    }
    return plan


def decide_curriculum_adjustment(
    *,
    tiny_summary: Optional[dict[str, Any]],
    difficulty: str,
    total_periods: int,
) -> dict[str, Any]:
    """Choose whether to retain or simplify the main TRL phase."""
    adjustment = {
        "action": "retain",
        "difficulty": str(difficulty),
        "total_periods": int(total_periods),
        "reasons": [],
    }
    if not tiny_summary:
        adjustment["reasons"].append("No successful tiny summary was available; retaining default curriculum.")
        return adjustment

    trained_metrics = dict(tiny_summary.get("policies", {}).get("trained", {}) or {})
    mean_total_reward = float(trained_metrics.get("mean_total_reward", 0.0) or 0.0)
    success_rate = float(trained_metrics.get("success_rate", 0.0) or 0.0)
    default_rate = float(trained_metrics.get("default_rate", 0.0) or 0.0)
    timeout_rate = float(trained_metrics.get("timeout_or_stepcap_rate", 0.0) or 0.0)
    if success_rate <= 0.05 or mean_total_reward <= 0.05 or default_rate >= 0.5 or timeout_rate >= 0.5:
        adjustment["action"] = "simplify"
        if difficulty == "hard":
            adjustment["difficulty"] = "medium"
            adjustment["reasons"].append("Tiny trained run collapsed on hard mode, so the main phase drops to medium.")
        else:
            adjustment["reasons"].append("Tiny trained run collapsed, so the main phase keeps the easier difficulty.")
        adjustment["total_periods"] = max(1, int(total_periods) - 1)
        adjustment["reasons"].append("Main phase reduces horizon to improve early reward density.")
    else:
        adjustment["reasons"].append("Tiny trained run showed enough signal to retain the default curriculum.")
    return adjustment


def run_backend_phase(
    *,
    backend: str,
    output_dir: Path,
    model_name: str,
    task_name: str,
    difficulty: str,
    total_periods: int,
    eval_seeds: list[int],
    inspection_seeds: list[int],
    eval_max_steps: int,
    arg_overrides: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    args_overrides = {
        "model_name": model_name,
        "task_name": task_name,
        "difficulty": difficulty,
        "total_periods": int(total_periods),
        "output_dir": str(output_dir),
        **arg_overrides,
    }
    try:
        if backend == "trl":
            training_args = make_trl_training_args(**args_overrides)
            training_run = run_trl_training_session(training_args)
        elif backend == "unsloth":
            training_args = make_unsloth_training_args(**args_overrides)
            training_run = run_unsloth_training_session(training_args)
        else:
            raise ValueError(f"Unsupported backend: {backend!r}")

        checkpoint_path = str(Path(training_run["checkpoint_path"]).resolve())
        dashboard = save_training_dashboard(training_run["trainer"], output_dir=str(output_dir))
        before_after = evaluate_before_after_policies(
            output_dir=str(output_dir),
            seeds=eval_seeds,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            max_steps=eval_max_steps,
            include_heuristic=False,
        )
        inspection = inspect_policy_checkpoint(
            output_dir=str(output_dir),
            policy_label=f"{backend}_trained",
            seeds=inspection_seeds,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            max_steps=eval_max_steps,
        )
        return {
            "status": "ok",
            "backend": backend,
            "output_dir": str(output_dir.resolve()),
            "checkpoint_path": checkpoint_path,
            "dashboard": dashboard,
            "before_after": before_after,
            "inspection": inspection,
        }
    except Exception as exc:
        error_path = output_dir / "phase_error.txt"
        error_path.write_text(str(exc), encoding="utf-8")
        return {
            "status": "failed",
            "backend": backend,
            "output_dir": str(output_dir.resolve()),
            "error": str(exc),
            "error_path": str(error_path.resolve()),
        }


def run_advanced_pipeline(
    *,
    profile: str = "submission",
    output_dir: str = DEFAULT_ADVANCED_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    unsloth_model_name: str = DEFAULT_UNSLOTH_MODEL,
    task_name: str = DEFAULT_TASK_NAME,
    difficulty: str = DEFAULT_DIFFICULTY,
    total_periods: int = DEFAULT_TOTAL_PERIODS,
) -> dict[str, Any]:
    """Run the full advanced pipeline and save a consolidated manifest."""
    plan = build_advanced_run_plan(
        profile=profile,
        output_dir=output_dir,
        model_name=model_name,
        unsloth_model_name=unsloth_model_name,
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    profile_defaults = dict(plan["profile_defaults"])
    eval_seeds = list(profile_defaults["eval_seeds"])
    inspection_seeds = list(profile_defaults["inspection_seeds"])
    eval_max_steps = int(profile_defaults["eval_max_steps"])

    tiny_trl = run_backend_phase(
        backend="trl",
        output_dir=root / "tiny_trl",
        model_name=model_name,
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        eval_seeds=eval_seeds,
        inspection_seeds=inspection_seeds,
        eval_max_steps=eval_max_steps,
        arg_overrides=dict(profile_defaults["tiny_trl"]),
    )
    tiny_unsloth = run_backend_phase(
        backend="unsloth",
        output_dir=root / "tiny_unsloth",
        model_name=unsloth_model_name,
        task_name=task_name,
        difficulty=difficulty,
        total_periods=total_periods,
        eval_seeds=eval_seeds,
        inspection_seeds=inspection_seeds,
        eval_max_steps=eval_max_steps,
        arg_overrides=dict(profile_defaults["tiny_unsloth"]),
    )

    checkpoint_paths = {
        "tiny_trl": tiny_trl["checkpoint_path"]
        for tiny_trl in [tiny_trl]
        if tiny_trl.get("status") == "ok" and tiny_trl.get("checkpoint_path")
    }
    if tiny_unsloth.get("status") == "ok" and tiny_unsloth.get("checkpoint_path"):
        checkpoint_paths["tiny_unsloth"] = str(tiny_unsloth["checkpoint_path"])

    tiny_comparison = None
    if checkpoint_paths:
        tiny_comparison = evaluate_policy_checkpoints(
            output_dir=str(root),
            seeds=eval_seeds,
            total_periods=total_periods,
            task_name=task_name,
            difficulty=difficulty,
            checkpoint_paths=checkpoint_paths,
            model_name=model_name,
            max_steps=eval_max_steps,
            include_base=True,
            include_heuristic=True,
            artifact_prefix="tiny_policy",
        )

    tiny_summary = None
    if tiny_trl.get("status") == "ok":
        tiny_summary = dict(tiny_trl["before_after"]["summary"])
    elif tiny_unsloth.get("status") == "ok":
        tiny_summary = dict(tiny_unsloth["before_after"]["summary"])
    curriculum_decision = decide_curriculum_adjustment(
        tiny_summary=tiny_summary,
        difficulty=difficulty,
        total_periods=total_periods,
    )

    main_overrides = dict(profile_defaults["main_trl"])
    main_output_dir = str(root)
    main_args = make_trl_training_args(
        model_name=model_name,
        task_name=task_name,
        difficulty=curriculum_decision["difficulty"],
        total_periods=curriculum_decision["total_periods"],
        output_dir=main_output_dir,
        **main_overrides,
    )
    main_training = run_trl_training_session(main_args)
    main_dashboard = save_training_dashboard(main_training["trainer"], output_dir=main_output_dir)
    final_checkpoint_path = str(Path(main_training["checkpoint_path"]).resolve())
    final_inspection = inspect_policy_checkpoint(
        output_dir=main_output_dir,
        policy_label="main_trl_trained",
        seeds=inspection_seeds,
        total_periods=curriculum_decision["total_periods"],
        task_name=task_name,
        difficulty=curriculum_decision["difficulty"],
        checkpoint_path=final_checkpoint_path,
        model_name=model_name,
        max_steps=eval_max_steps,
    )
    final_evaluation = evaluate_before_after_policies(
        output_dir=main_output_dir,
        seeds=eval_seeds,
        total_periods=curriculum_decision["total_periods"],
        task_name=task_name,
        difficulty=curriculum_decision["difficulty"],
        checkpoint_path=final_checkpoint_path,
        model_name=model_name,
        max_steps=eval_max_steps,
        include_heuristic=True,
    )

    manifest = {
        "profile": profile,
        "plan": plan,
        "curriculum_decision": curriculum_decision,
        "tiny_runs": {
            "trl": tiny_trl,
            "unsloth": tiny_unsloth,
        },
        "tiny_comparison": tiny_comparison,
        "main_training": {
            "checkpoint_path": final_checkpoint_path,
            "reward_log_path": main_dashboard["reward_log_path"],
            "training_dashboard_path": main_dashboard["training_dashboard_path"],
            "reward_curve_path": main_dashboard["reward_curve_path"],
            "inspection_path": final_inspection["inspection_path"],
        },
        "final_evaluation": final_evaluation,
    }
    manifest_path = save_run_manifest(manifest, output_dir=str(root))
    manifest["manifest_path"] = manifest_path
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the advanced orchestration entrypoint."""
    parser = argparse.ArgumentParser(description="Run the advanced GRPO notebook-first pipeline for SME liquidity.")
    parser.add_argument("--profile", choices=tuple(_PROFILE_DEFAULTS), default="submission")
    parser.add_argument("--output-dir", default=DEFAULT_ADVANCED_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--unsloth-model-name", default=DEFAULT_UNSLOTH_MODEL)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY)
    parser.add_argument("--total-periods", type=int, default=DEFAULT_TOTAL_PERIODS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run or preview the advanced training pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        plan = build_advanced_run_plan(
            profile=args.profile,
            output_dir=args.output_dir,
            model_name=args.model_name,
            unsloth_model_name=args.unsloth_model_name,
            task_name=args.task_name,
            difficulty=args.difficulty,
            total_periods=args.total_periods,
        )
        print(json.dumps(plan, indent=2))
        return 0

    manifest = run_advanced_pipeline(
        profile=args.profile,
        output_dir=args.output_dir,
        model_name=args.model_name,
        unsloth_model_name=args.unsloth_model_name,
        task_name=args.task_name,
        difficulty=args.difficulty,
        total_periods=args.total_periods,
    )
    print(json.dumps({"manifest_path": manifest["manifest_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
