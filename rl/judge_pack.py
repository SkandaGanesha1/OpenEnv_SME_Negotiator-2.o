#!/usr/bin/env python3
"""Generate a compact judge-facing artifact bundle from inference results."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from rl.demo import run_heuristic_episode, run_policy_episode

_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=="
)


def _load_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_episodes(results: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for task in (results.get("tasks") or {}).values():
        for episode in task.get("episodes", []):
            if isinstance(episode, dict):
                yield episode


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_task_summary(task_name: str, task_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(task_payload.get("summary") or {})
    return {
        "task": task_name,
        "episode_count": int(len(task_payload.get("episodes", []))),
        "mean_score": float(summary.get("mean_final_score", 0.0) or 0.0),
        "mean_reward": float(summary.get("mean_total_reward", 0.0) or 0.0),
        "success_rate": float(summary.get("success_rate", 0.0) or 0.0),
        "avg_verifiable_reward": float(summary.get("avg_verifiable_reward", 0.0) or 0.0),
        "avg_final_payment_days": float(summary.get("avg_final_payment_days", 0.0) or 0.0),
        "default_rate": float(summary.get("default_rate", 0.0) or 0.0),
        "timeout_or_stepcap_rate": float(summary.get("timeout_or_stepcap_rate", 0.0) or 0.0),
        "avg_tool_call_count": float(summary.get("avg_tool_call_count", 0.0) or 0.0),
        "avg_tool_effective_count": float(summary.get("avg_tool_effective_count", 0.0) or 0.0),
    }


def _build_overall_summary(results: dict[str, Any]) -> dict[str, Any]:
    top_summary = dict(results.get("summary") or {})
    episodes = list(_iter_episodes(results))
    episode_summaries = [dict(item.get("episode_summary") or {}) for item in episodes]
    return {
        "episode_count": int(len(episodes)),
        "overall_mean_score": float(top_summary.get("overall_mean_score", 0.0) or 0.0),
        "overall_mean_reward": float(top_summary.get("overall_mean_reward", 0.0) or 0.0),
        "success_rate": float(top_summary.get("overall_success_rate", 0.0) or 0.0),
        "avg_verifiable_reward": float(top_summary.get("avg_verifiable_reward", 0.0) or 0.0),
        "avg_final_payment_days": float(top_summary.get("avg_final_payment_days", 0.0) or 0.0),
        "default_rate": float(top_summary.get("default_rate", 0.0) or 0.0),
        "timeout_or_stepcap_rate": float(top_summary.get("timeout_or_stepcap_rate", 0.0) or 0.0),
        "avg_tool_call_count": float(top_summary.get("avg_tool_call_count", 0.0) or 0.0),
        "avg_tool_effective_count": float(top_summary.get("avg_tool_effective_count", 0.0) or 0.0),
        "avg_tool_bonus": float(top_summary.get("avg_tool_bonus", 0.0) or 0.0),
        "avg_resolved_deal_count": float(top_summary.get("avg_resolved_deal_count", 0.0) or 0.0),
        "mean_final_payment_days_from_episodes": _safe_mean(
            [float(item.get("average_final_payment_days", 0.0) or 0.0) for item in episode_summaries]
        ),
    }


def _write_reward_curve(results: dict[str, Any], output_path: Path, source_path: Optional[Path] = None) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path is not None and source_path.exists():
        shutil.copyfile(source_path, output_path)
        return "copied"

    episodes = list(_iter_episodes(results))
    scores = [float(item.get("final_score", 0.0) or 0.0) for item in episodes]
    rewards = [float(item.get("total_reward", 0.0) or 0.0) for item in episodes]
    if not scores:
        output_path.write_bytes(_PLACEHOLDER_PNG)
        return "placeholder"

    try:
        import matplotlib.pyplot as plt

        xs = list(range(1, len(scores) + 1))
        fig, (ax_score, ax_reward) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax_score.plot(xs, scores, marker="o", label="final_score")
        ax_score.set_ylabel("Final Score")
        ax_score.legend()

        ax_reward.plot(xs, rewards, marker="o", color="green", label="total_reward")
        ax_reward.set_ylabel("Total Reward")
        ax_reward.set_xlabel("Episode")
        ax_reward.legend()

        fig.suptitle("SME Negotiator Judge Pack Curves")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return "generated"
    except Exception:
        output_path.write_bytes(_PLACEHOLDER_PNG)
        return "placeholder"


def _best_episode(results: dict[str, Any]) -> Optional[dict[str, Any]]:
    episodes = list(_iter_episodes(results))
    if not episodes:
        return None
    return max(episodes, key=lambda item: float(item.get("final_score", 0.0) or 0.0))


def _build_baseline_transcript() -> tuple[str, dict[str, Any]]:
    baseline = run_heuristic_episode(
        seed=1000,
        total_periods=3,
        task_name="liquidity-correlation-hard",
        difficulty="hard",
    )
    transcript = str(baseline.get("transcript", ""))
    summary = dict(baseline.get("summary") or {})
    return transcript, summary


def _build_trained_transcript(
    results: dict[str, Any],
    *,
    checkpoint_path: Optional[str],
) -> tuple[str, dict[str, Any], str]:
    if checkpoint_path:
        try:
            transcript = run_policy_episode(
                policy="trained",
                checkpoint_path=checkpoint_path,
                seed=1000,
                total_periods=3,
                task_name="liquidity-correlation-hard",
                difficulty="hard",
            )
            return transcript, {}, "checkpoint"
        except Exception as exc:
            fallback = _best_episode(results) or {}
            fallback_log = str(fallback.get("episode_log", "") or "No trained transcript available.")
            return fallback_log, dict(fallback.get("episode_summary") or {}), f"results_fallback:{type(exc).__name__}"

    fallback = _best_episode(results) or {}
    fallback_log = str(fallback.get("episode_log", "") or "No trained transcript available.")
    return fallback_log, dict(fallback.get("episode_summary") or {}), "results_best_episode"


def _excerpt(text: str, *, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines]) if lines else "(empty transcript)"


def build_judge_pack(
    *,
    results_file: str = "inference_results.json",
    output_dir: str = "outputs/judge_pack",
    checkpoint_path: Optional[str] = None,
    reward_curve_source: Optional[str] = None,
) -> dict[str, Any]:
    results_path = Path(results_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = _load_results(results_path)

    task_summaries = {
        task_name: _build_task_summary(task_name, dict(task_payload))
        for task_name, task_payload in (results.get("tasks") or {}).items()
    }
    overall_summary = _build_overall_summary(results)

    baseline_transcript, baseline_summary = _build_baseline_transcript()
    trained_transcript, trained_summary, trained_source = _build_trained_transcript(
        results,
        checkpoint_path=checkpoint_path,
    )

    baseline_path = output_path / "baseline_transcript.txt"
    baseline_path.write_text(baseline_transcript, encoding="utf-8")
    trained_path = output_path / "trained_transcript.txt"
    trained_path.write_text(trained_transcript, encoding="utf-8")

    curve_status = _write_reward_curve(
        results,
        output_path / "reward_curve.png",
        Path(reward_curve_source) if reward_curve_source else None,
    )

    judge_summary = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results_file": str(results_path),
            "checkpoint_path": checkpoint_path,
            "reward_curve_status": curve_status,
            "trained_transcript_source": trained_source,
            "inference_metadata": dict(results.get("metadata") or {}),
        },
        "tasks": task_summaries,
        "overall": overall_summary,
        "baseline_summary": baseline_summary,
        "trained_summary": trained_summary,
    }
    judge_summary_path = output_path / "judge_summary.json"
    judge_summary_path.write_text(json.dumps(judge_summary, indent=2), encoding="utf-8")

    comparison = (
        "## Before / After Snippet\n\n"
        "### Baseline heuristic excerpt\n\n"
        "```text\n"
        f"{_excerpt(baseline_transcript)}\n"
        "```\n\n"
        "### Trained / evaluated excerpt\n\n"
        "```text\n"
        f"{_excerpt(trained_transcript)}\n"
        "```\n"
    )

    task_rows = [
        f"| {name} | {summary['episode_count']} | {summary['mean_score']:.4f} | {summary['mean_reward']:.4f} | "
        f"{summary['success_rate']:.4f} | {summary['avg_verifiable_reward']:.4f} | "
        f"{summary['avg_final_payment_days']:.2f} | {summary['default_rate']:.4f} | "
        f"{summary['timeout_or_stepcap_rate']:.4f} | {summary['avg_tool_call_count']:.2f} | "
        f"{summary['avg_tool_effective_count']:.2f} |"
        for name, summary in task_summaries.items()
    ]
    judge_results = (
        "# SME Negotiator Judge Results\n\n"
        "## Overall\n\n"
        f"- Episodes: {overall_summary['episode_count']}\n"
        f"- Overall mean score: {overall_summary['overall_mean_score']:.4f}\n"
        f"- Overall mean reward: {overall_summary['overall_mean_reward']:.4f}\n"
        f"- Success rate: {overall_summary['success_rate']:.4f}\n"
        f"- Avg verifiable reward: {overall_summary['avg_verifiable_reward']:.4f}\n"
        f"- Avg final payment days: {overall_summary['avg_final_payment_days']:.2f}\n"
        f"- Default rate: {overall_summary['default_rate']:.4f}\n"
        f"- Timeout / step-cap rate: {overall_summary['timeout_or_stepcap_rate']:.4f}\n"
        f"- Avg tool call count: {overall_summary['avg_tool_call_count']:.2f}\n"
        f"- Avg tool effective count: {overall_summary['avg_tool_effective_count']:.2f}\n\n"
        "## Task Breakdown\n\n"
        "| Task | Episodes | Mean Score | Mean Reward | Success Rate | Avg Verifiable Reward | Avg Final Days | Default Rate | Timeout Rate | Avg Tool Calls | Avg Effective Tools |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{chr(10).join(task_rows) if task_rows else '| none | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 | 0.00 |'}\n\n"
        "## Artifact Notes\n\n"
        f"- Reward curve status: `{curve_status}`\n"
        f"- Trained transcript source: `{trained_source}`\n"
        f"- Results file: `{results_path}`\n"
        f"- Checkpoint path: `{checkpoint_path or 'none'}`\n\n"
        f"{comparison}"
    )
    judge_results_path = output_path / "judge_results.md"
    judge_results_path.write_text(judge_results, encoding="utf-8")

    return {
        "judge_summary_path": str(judge_summary_path),
        "judge_results_path": str(judge_results_path),
        "baseline_transcript_path": str(baseline_path),
        "trained_transcript_path": str(trained_path),
        "reward_curve_path": str(output_path / "reward_curve.png"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a judge-facing SME Negotiator artifact pack.")
    parser.add_argument("--results-file", default="inference_results.json")
    parser.add_argument("--output-dir", default="outputs/judge_pack")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--reward-curve-source", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    outputs = build_judge_pack(
        results_file=args.results_file,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        reward_curve_source=args.reward_curve_source,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
