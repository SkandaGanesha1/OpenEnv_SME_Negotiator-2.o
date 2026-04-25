"""Judge-pack artifact generation tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.judge_pack import build_judge_pack


def _workspace_tmp_dir(name: str) -> Path:
    path = PROJECT_ROOT / ".test_tmp" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_judge_pack_writes_required_artifacts(monkeypatch) -> None:
    tmp_path = _workspace_tmp_dir("judge_pack")
    results = {
        "metadata": {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "inference_env_mode": "liquidity",
            "inference_agent_mode": "router",
        },
        "tasks": {
            "EASY": {
                "episodes": [
                    {
                        "final_score": 0.6,
                        "total_reward": 0.62,
                        "success": True,
                        "episode_log": "easy-log",
                        "episode_summary": {
                            "verifiable_reward": 0.58,
                            "tool_bonus_total": 0.01,
                            "tool_call_count": 1,
                            "tool_effective_count": 1,
                            "average_final_payment_days": 42.0,
                            "resolved_deal_count": 2,
                            "defaulted_sme_count": 0,
                            "terminated_by_step_cap": False,
                        },
                    }
                ],
                "summary": {
                    "mean_final_score": 0.6,
                    "mean_total_reward": 0.62,
                    "success_rate": 1.0,
                    "avg_verifiable_reward": 0.58,
                    "avg_final_payment_days": 42.0,
                    "default_rate": 0.0,
                    "timeout_or_stepcap_rate": 0.0,
                    "avg_tool_call_count": 1.0,
                    "avg_tool_effective_count": 1.0,
                },
            },
            "HARD": {
                "episodes": [
                    {
                        "final_score": 0.2,
                        "total_reward": 0.25,
                        "success": False,
                        "episode_log": "hard-log",
                        "episode_summary": {
                            "verifiable_reward": 0.15,
                            "tool_bonus_total": 0.0,
                            "tool_call_count": 2,
                            "tool_effective_count": 1,
                            "average_final_payment_days": 78.0,
                            "resolved_deal_count": 1,
                            "defaulted_sme_count": 1,
                            "terminated_by_step_cap": True,
                        },
                    }
                ],
                "summary": {
                    "mean_final_score": 0.2,
                    "mean_total_reward": 0.25,
                    "success_rate": 0.0,
                    "avg_verifiable_reward": 0.15,
                    "avg_final_payment_days": 78.0,
                    "default_rate": 1.0,
                    "timeout_or_stepcap_rate": 1.0,
                    "avg_tool_call_count": 2.0,
                    "avg_tool_effective_count": 1.0,
                },
            },
        },
        "summary": {
            "overall_mean_score": 0.4,
            "overall_mean_reward": 0.435,
            "overall_success_rate": 0.5,
            "avg_verifiable_reward": 0.365,
            "avg_final_payment_days": 60.0,
            "default_rate": 0.5,
            "timeout_or_stepcap_rate": 0.5,
            "avg_tool_call_count": 1.5,
            "avg_tool_effective_count": 1.0,
            "avg_tool_bonus": 0.005,
            "avg_resolved_deal_count": 1.5,
        },
    }
    results_file = tmp_path / "inference_results.json"
    results_file.write_text(json.dumps(results), encoding="utf-8")

    monkeypatch.setattr(
        "rl.judge_pack.run_heuristic_episode",
        lambda **kwargs: {
            "transcript": "baseline transcript",
            "summary": {
                "verifiable_reward": 0.1,
                "tool_call_count": 1,
                "tool_effective_count": 1,
                "defaulted_sme_count": 0,
            },
        },
    )
    monkeypatch.setattr("rl.judge_pack.run_policy_episode", lambda **kwargs: "trained transcript")

    def _fake_curve(results_payload, output_path, source_path=None):
        output_path.write_bytes(b"png")
        return "generated"

    monkeypatch.setattr("rl.judge_pack._write_reward_curve", _fake_curve)

    output_dir = tmp_path / "judge_pack"
    outputs = build_judge_pack(
        results_file=str(results_file),
        output_dir=str(output_dir),
        checkpoint_path="outputs/fake-checkpoint",
    )

    assert Path(outputs["judge_summary_path"]).exists()
    assert Path(outputs["judge_results_path"]).exists()
    assert Path(outputs["baseline_transcript_path"]).exists()
    assert Path(outputs["trained_transcript_path"]).exists()
    assert Path(outputs["reward_curve_path"]).exists()
    assert Path(outputs["before_after_excerpt_path"]).exists()

    summary = json.loads(Path(outputs["judge_summary_path"]).read_text(encoding="utf-8"))
    assert summary["overall"]["default_rate"] == 0.5
    assert summary["tasks"]["EASY"]["mean_score"] == 0.6
    assert summary["metadata"]["trained_transcript_source"] == "checkpoint"

    judge_results = Path(outputs["judge_results_path"]).read_text(encoding="utf-8")
    assert "Before / After Snippet" in judge_results
    assert "Default rate: 0.5000" in judge_results


def test_build_judge_pack_marks_baseline_only_when_no_checkpoint(monkeypatch) -> None:
    tmp_path = _workspace_tmp_dir("judge_pack_baseline_only")
    results = {
        "metadata": {
            "timestamp": "2026-04-25T00:00:00+00:00",
            "inference_agent_mode": "router",
        },
        "tasks": {
            "MEDIUM": {
                "episodes": [
                    {
                        "final_score": 0.4,
                        "total_reward": 0.45,
                        "episode_log": "router-log",
                        "episode_summary": {"verifiable_reward": 0.4},
                    }
                ],
                "summary": {
                    "mean_final_score": 0.4,
                    "mean_total_reward": 0.45,
                    "success_rate": 0.0,
                    "avg_verifiable_reward": 0.4,
                    "avg_final_payment_days": 55.0,
                    "default_rate": 0.0,
                    "timeout_or_stepcap_rate": 0.0,
                    "avg_tool_call_count": 0.0,
                    "avg_tool_effective_count": 0.0,
                },
            }
        },
        "summary": {
            "overall_mean_score": 0.4,
            "overall_mean_reward": 0.45,
            "overall_success_rate": 0.0,
        },
    }
    results_file = tmp_path / "inference_results.json"
    results_file.write_text(json.dumps(results), encoding="utf-8")

    monkeypatch.setattr(
        "rl.judge_pack.run_heuristic_episode",
        lambda **kwargs: {"transcript": "baseline transcript", "summary": {}},
    )
    monkeypatch.setattr("rl.judge_pack._write_reward_curve", lambda *args, **kwargs: "generated")

    output_dir = tmp_path / "judge_pack"
    outputs = build_judge_pack(results_file=str(results_file), output_dir=str(output_dir))

    summary = json.loads(Path(outputs["judge_summary_path"]).read_text(encoding="utf-8"))
    assert summary["metadata"]["comparison_mode"] == "baseline_only"
    assert summary["metadata"]["generated_at"] == "2026-04-25T00:00:00+00:00"
