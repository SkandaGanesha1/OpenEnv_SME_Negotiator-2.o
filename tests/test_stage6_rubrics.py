"""Stage 6 rubric and self-rewarding tests."""

from __future__ import annotations

import sys
from pathlib import Path
from random import Random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.rubrics import PERSONAS, pairwise_preference_label, persona_reward, sample_persona
from rl.self_rewarding_dpo import build_preference_examples


def test_persona_sampling_is_deterministic_for_fixed_rng_seed() -> None:
    first = sample_persona(Random(1400))
    second = sample_persona(Random(1400))
    assert first == second


def test_persona_weighted_reward_aggregation_is_correct() -> None:
    persona = PERSONAS[0]
    score = persona_reward(
        persona,
        {
            "solvency": 1.0,
            "compliance": 0.5,
            "relationship": 0.25,
            "growth": 0.0,
        },
    )
    expected = 0.4 * 1.0 + 0.3 * 0.5 + 0.1 * 0.25 + 0.2 * 0.0
    assert score == round(expected, 6)


def test_pairwise_preference_label_picks_higher_persona_reward() -> None:
    persona = PERSONAS[1]

    def scorer(episode_log: str) -> dict[str, float]:
        return {"growth": 1.0, "solvency": 0.8} if "A" in episode_log else {"growth": 0.1, "solvency": 0.9}

    assert pairwise_preference_label(persona, "episode A", "episode B", scorer) == 1


def test_preference_pair_generation_labels_higher_persona_reward_correctly() -> None:
    persona = PERSONAS[2]

    def scorer(episode_log: str) -> dict[str, float]:
        return (
            {"compliance": 1.0, "relationship": 0.5, "solvency": 0.4, "growth": 0.0}
            if "good" in episode_log
            else {"compliance": 0.2, "relationship": 0.2, "solvency": 0.9, "growth": 0.5}
        )

    examples = build_preference_examples(
        ["good episode", "bad episode"],
        scorer,
        seed=1401,
        personas=[persona],
    )

    assert len(examples) == 1
    assert examples[0].persona_name == persona.name
    assert examples[0].chosen == "good episode"
    assert examples[0].chosen_reward >= examples[0].rejected_reward
