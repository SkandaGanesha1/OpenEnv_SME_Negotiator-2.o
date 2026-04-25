"""Generate a judge-facing OpenEnv hackathon readiness report.

The report is deterministic and does not require an LLM API key. It validates
the repo surface, runs legacy OpenEnv tasks, runs the advanced liquidity tasks,
and writes both JSON and Markdown artifacts under outputs/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rl.demo import run_heuristic_episode
from server.environment import SMENegotiatorEnvironment
from sme_negotiator_env.client import choose_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class TaskScore:
    task_id: str
    surface: str
    mean_reward: float
    success_rate: float
    episodes: int
    notes: str


def _file_exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def _run_legacy_task(difficulty: str, seeds: list[int]) -> TaskScore:
    rewards: list[float] = []
    successes: list[bool] = []
    for seed in seeds:
        env = SMENegotiatorEnvironment()
        obs = env.reset(seed=seed, difficulty=difficulty.lower())
        round_number = 0
        while not bool(obs.done) and round_number < 20:
            action = choose_action(obs, round_number)
            obs = env.step(action)
            round_number += 1
        reward = float(obs.reward or 0.0)
        rewards.append(reward)
        successes.append(bool(obs.done and reward >= 0.30))
    return TaskScore(
        task_id=f"payment-terms-{difficulty.lower()}",
        surface="OpenEnv HTTP-compatible legacy task",
        mean_reward=round(mean(rewards), 6),
        success_rate=round(mean(1.0 if item else 0.0 for item in successes), 6),
        episodes=len(seeds),
        notes="Runs through SMENegotiatorEnvironment with reset/step and typed actions.",
    )


def _run_liquidity_task(task_id: str, total_periods: int, seeds: list[int]) -> TaskScore:
    rewards: list[float] = []
    successes: list[bool] = []
    notes: list[str] = []
    max_steps = 40 if total_periods <= 1 else 90
    for seed in seeds:
        result = run_heuristic_episode(
            seed=seed,
            total_periods=total_periods,
            task_name=task_id,
            max_steps=max_steps,
        )
        summary = result["summary"]
        rewards.append(float(summary["total_reward"]))
        successes.append(
            bool(
                int(summary["defaulted_sme_count"]) == 0
                and int(summary["resolved_deal_count"]) > 0
                and float(summary["total_reward"]) > 0.0
            )
        )
        notes.append(
            "resolved={resolved_deal_count}, defaulted={defaulted_sme_count}, "
            "avg_days={average_final_payment_days}, tools={tool_effective_count}/{tool_call_count}".format(**summary)
        )
    return TaskScore(
        task_id=task_id,
        surface="Advanced in-process liquidity task used by GRPO training",
        mean_reward=round(mean(rewards), 6),
        success_rate=round(mean(1.0 if item else 0.0 for item in successes), 6),
        episodes=len(seeds),
        notes="; ".join(notes),
    )


def build_report() -> dict[str, Any]:
    seeds = [1000, 1001, 1002]
    compliance = {
        "openenv_manifest": _file_exists("openenv.yaml"),
        "fastapi_app": _file_exists("server/app.py"),
        "typed_models": _file_exists("sme_negotiator_env/models.py"),
        "client_package": _file_exists("sme_negotiator_env/client.py"),
        "dockerfile": _file_exists("docker/Dockerfile") or _file_exists("Dockerfile"),
        "trl_training_script": _file_exists("rl/train_grpo_trl.py"),
        "unsloth_training_script": _file_exists("rl/train_grpo_unsloth.py"),
        "colab_notebook": _file_exists("notebooks/colab_grpo_sme_liquidity.ipynb"),
        "hf_space_documented": "huggingface.co/spaces/Omkarchaithanya/sme-negotiator"
        in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
        "hf_blog_draft": _file_exists("huggingface/blog_post.md"),
        "hf_model_card": _file_exists("huggingface/model_card.md"),
        "hf_router_inference": "https://router.huggingface.co/v1"
        in (PROJECT_ROOT / "inference.py").read_text(encoding="utf-8"),
    }
    scores = [
        _run_legacy_task("easy", seeds),
        _run_legacy_task("medium", seeds),
        _run_legacy_task("hard", seeds),
        _run_liquidity_task("liquidity-stress-medium", 1, seeds),
        _run_liquidity_task("liquidity-correlation-hard", 2, seeds),
    ]
    return {
        "compliance": compliance,
        "task_scores": [asdict(score) for score in scores],
        "judge_mapping": {
            "environment_innovation_40": "SME payment-term negotiation with multi-agent buyer/financier/regulator state, cashflow simulation, and TReDS-style tools.",
            "storytelling_30": "README, HF blog draft, model card, crisis statistics, and Space link explain the real-world working-capital gap.",
            "reward_improvement_20": "Use Colab GRPO reward_curve.png plus this deterministic report; train medium before hard to avoid all-zero reward.",
            "reward_pipeline_10": "TRL and Unsloth scripts use deterministic rewards, shaping, tool bonuses, curriculum, self-play hooks, and LoRA training.",
        },
        "critical_recommendations": [
            "Expose the advanced liquidity task through the hosted OpenEnv surface or explicitly tell judges the Space is legacy while Colab/TRL uses the advanced in-process environment.",
            "Do not submit the 10-step hard-mode run as final evidence; use at least a 100-step medium run and a 300-step hard/curriculum run.",
            "Commit reward_curve.png and this report artifact so judges do not need to rerun Colab to see progress.",
            "Keep HF router credentials in environment variables only; never commit tokens.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Judge Readiness Report",
        "",
        "## Compliance Checklist",
        "",
        "| Item | Status |",
        "|---|---|",
    ]
    for key, value in report["compliance"].items():
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")
    lines.extend(["", "## Task Scores", "", "| Task | Surface | Mean reward | Success rate | Episodes | Notes |", "|---|---|---:|---:|---:|---|"])
    for score in report["task_scores"]:
        lines.append(
            f"| `{score['task_id']}` | {score['surface']} | {score['mean_reward']:.6f} | "
            f"{score['success_rate']:.2f} | {score['episodes']} | {score['notes']} |"
        )
    lines.extend(["", "## Judge Mapping", ""])
    for key, value in report["judge_mapping"].items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Critical Recommendations", ""])
    for item in report["critical_recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = OUTPUT_DIR / "judge_report.json"
    md_path = OUTPUT_DIR / "judge_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
