#!/usr/bin/env python3
"""Notebook-facing GRPO helpers for the SME liquidity environment.

This module mirrors the role that ``kube_sre_gym.train`` plays in the
reference notebook: it is the reusable, environment-specific source of truth
for notebook orchestration and the simple CLI training entrypoint.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rl.demo import plot_rewards, save_run_manifest, save_training_dashboard
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
        "gradient_accumulation_steps": 2,
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
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        rubric_weight=args.rubric_weight,
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
            "num_samples": int(canonical_args.num_samples),
            "max_steps": int(canonical_args.max_steps) if canonical_args.max_steps is not None else None,
            "max_episode_steps": int(canonical_args.max_episode_steps),
            "num_generations": int(canonical_args.num_generations),
            "gradient_accumulation_steps": int(canonical_args.gradient_accumulation_steps),
            "learning_rate": float(canonical_args.learning_rate),
            "max_completion_length": int(canonical_args.max_completion_length),
            "use_vllm": bool(canonical_args.use_vllm),
            "vllm_mode": str(canonical_args.vllm_mode),
            "rubric_weight": float(canonical_args.rubric_weight),
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


def _normalize_bridge_diagnostics(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = dict(payload or {})
    for key in (
        "bridge_miss_count",
        "fifo_fallback_count",
        "signature_match_count",
        "prompt_env_fallback_count",
    ):
        diagnostics[key] = int(diagnostics.get(key, 0) or 0)
    diagnostics["strict_accuracy_valid"] = bool(
        diagnostics["bridge_miss_count"] <= 0 and diagnostics["prompt_env_fallback_count"] <= 0
    )
    return diagnostics


def _raise_for_invalid_bridge(diagnostics: dict[str, Any], *, output_dir: Path) -> None:
    if bool(diagnostics.get("strict_accuracy_valid", False)):
        return

    details: list[str] = []
    if int(diagnostics.get("bridge_miss_count", 0) or 0) > 0:
        details.append(f"bridge_miss_count={int(diagnostics['bridge_miss_count'])}")
    if int(diagnostics.get("prompt_env_fallback_count", 0) or 0) > 0:
        details.append(f"prompt_env_fallback_count={int(diagnostics['prompt_env_fallback_count'])}")
    joined_details = ", ".join(details) if details else "unknown bridge mismatch"
    raise RuntimeError(
        "Strict bridge validation failed for notebook/simple liquidity training. "
        "The reward curve would be inaccurate because TRL fell back to prompt-derived or unmatched rewards "
        f"instead of matched rollout completions ({joined_details}). "
        f"Inspect artifacts in {output_dir.resolve()} and fix the rollout bridge before trusting this curve."
    )


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
    return build_canonical_training_session(build_canonical_training_args(args))


def build_trainer(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    """Return a freshly built canonical liquidity training session and trainer."""
    session = build_training_session(args)
    return session, create_canonical_trainer(session)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run canonical liquidity GRPO training and save standard artifacts."""
    canonical_args = build_canonical_training_args(args)
    output_dir = Path(canonical_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke = None if args.skip_smoke_test else smoke_test_environment(canonical_args)
    result = run_canonical_training_session(canonical_args)
    episode_reward_log_path = _write_episode_reward_log(
        list(result.get("episode_reward_history", [])),
        output_path=output_dir / "episode_reward_log.json",
    )
    bridge_diagnostics = _normalize_bridge_diagnostics(result.get("bridge_diagnostics"))
    _raise_for_invalid_bridge(bridge_diagnostics, output_dir=output_dir)

    dashboard = save_training_dashboard(result["trainer"], output_dir=str(output_dir))
    reward_curve_path = Path(dashboard["reward_curve_path"])
    trainer_reward_log_path = Path(dashboard["reward_log_path"])
    plot_rewards(episode_reward_log_path, reward_curve_path)

    manifest = {
        "run_type": "simple_grpo_training",
        "plan": build_run_plan(args, canonical_args),
        "smoke_test": smoke,
        "training": {
            "checkpoint_path": str(Path(result["checkpoint_path"]).resolve()),
            "reward_log_path": str(episode_reward_log_path.resolve()),
            "episode_reward_log_path": str(episode_reward_log_path.resolve()),
            "trainer_reward_log_path": str(trainer_reward_log_path.resolve()),
            "reward_curve_path": str(reward_curve_path.resolve()),
            "training_dashboard_path": str(Path(dashboard["training_dashboard_path"]).resolve()),
            "history_points": int(dashboard["history_points"]),
            "zero_variance_warning": bool(dashboard["zero_variance_warning"]),
            "bridge_validation": bridge_diagnostics,
            "strict_accuracy_bridge_valid": bool(bridge_diagnostics["strict_accuracy_valid"]),
        },
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
