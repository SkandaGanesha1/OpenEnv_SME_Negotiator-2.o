"""Minimal self-rewarding DPO dataset builder for Stage 6."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Iterable, Optional

from rl.rubrics import PERSONAS, Persona, pairwise_preference_label, sample_persona


@dataclass(frozen=True)
class PreferenceExample:
    """One synthetic preference pair for later DPO training."""

    persona_name: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float


def build_preference_examples(
    episode_logs: Iterable[str],
    scorer: Callable[[str], dict[str, float]],
    *,
    seed: int = 1000,
    personas: Optional[list[Persona]] = None,
) -> list[PreferenceExample]:
    """Create synthetic preference pairs from recent episode logs."""
    logs = list(episode_logs)
    rng = Random(int(seed))
    chosen_personas = list(personas or PERSONAS)
    examples: list[PreferenceExample] = []

    for index in range(0, max(len(logs) - 1, 0), 2):
        episode_a = logs[index]
        episode_b = logs[index + 1]
        persona = sample_persona(rng, chosen_personas)
        score_a = scorer(episode_a)
        score_b = scorer(episode_b)
        reward_a = sum(persona.rubric_weights[key] * float(score_a.get(key, 0.0)) for key in persona.rubric_weights)
        reward_b = sum(persona.rubric_weights[key] * float(score_b.get(key, 0.0)) for key in persona.rubric_weights)
        label = pairwise_preference_label(persona, episode_a, episode_b, scorer)
        if label == 1:
            chosen, rejected = episode_a, episode_b
            chosen_reward, rejected_reward = reward_a, reward_b
        else:
            chosen, rejected = episode_b, episode_a
            chosen_reward, rejected_reward = reward_b, reward_a
        examples.append(
            PreferenceExample(
                persona_name=persona.name,
                chosen=chosen,
                rejected=rejected,
                chosen_reward=round(float(chosen_reward), 6),
                rejected_reward=round(float(rejected_reward), 6),
            )
        )

    return examples


def build_rule_based_rubric_scorer() -> Callable[[str], dict[str, float]]:
    """Return a deterministic rubric scorer derived from episode log lines.

    Parses [STEP] and [END] log lines to compute four rubric dimensions without
    any LLM call. Use this as a zero-cost fallback judge for demos and unit tests.
    """

    def scorer(episode_log: str) -> dict[str, float]:
        lines = episode_log.split("\n")
        final_score = 0.0
        for line in lines:
            if line.startswith("[END]") and "score=" in line:
                try:
                    score_part = next(p for p in line.split() if p.startswith("score="))
                    final_score = float(score_part.split("=", 1)[1])
                except (StopIteration, ValueError):
                    pass
        treds_count = sum(1 for ln in lines if "use_treds=true" in ln.lower())
        step_count = sum(1 for ln in lines if ln.startswith("[STEP]"))
        step_count = max(1, step_count)

        solvency = round(min(1.0, final_score * 1.05), 4)
        compliance = round(min(1.0, final_score), 4)
        relationship = round(min(1.0, 0.3 + treds_count * 0.2), 4)
        efficiency_bonus = max(0.0, 1.0 - step_count / 16.0) * 0.1
        growth = round(min(1.0, final_score * 0.9 + efficiency_bonus), 4)
        return {
            "solvency": solvency,
            "compliance": compliance,
            "relationship": relationship,
            "growth": growth,
        }

    return scorer


def load_episode_logs(path: str | Path) -> list[str]:
    """Load episode-log strings from a simple JSONL file."""
    logs: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if isinstance(payload, dict) and "episode_log" in payload:
                logs.append(str(payload["episode_log"]))
    return logs


def write_preference_dataset(path: str | Path, examples: Iterable[PreferenceExample]) -> int:
    """Write a JSONL DPO-style preference dataset and return the row count."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for preference-dataset generation."""
    parser = argparse.ArgumentParser(description="Build a synthetic DPO preference dataset from episode logs.")
    parser.add_argument("--episode-log-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="outputs/self_rewarding/preferences.jsonl")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Build a preference dataset using a deterministic local fake scorer."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    def fake_scorer(episode_log: str) -> dict[str, float]:
        baseline = min(1.0, max(0.0, len(episode_log) / 5000.0))
        return {
            "solvency": baseline,
            "compliance": min(1.0, baseline + 0.1),
            "relationship": max(0.0, baseline - 0.05),
            "growth": baseline,
        }

    logs = load_episode_logs(args.episode_log_jsonl)
    examples = build_preference_examples(logs, fake_scorer, seed=args.seed)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "episode_count": len(logs),
                    "preference_count": len(examples),
                    "output_jsonl": args.output_jsonl,
                },
                indent=2,
            )
        )
        return 0

    count = write_preference_dataset(args.output_jsonl, examples)
    print(json.dumps({"written_examples": count, "output_jsonl": args.output_jsonl}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
