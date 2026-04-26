"""Persona-weighted rubric helpers for Stage 6 training."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional


@dataclass(frozen=True)
class Persona:
    """One simulated expert persona used for training-side rubric overlays."""

    name: str
    description: str
    rubric_weights: dict[str, float]


PERSONAS: list[Persona] = [
    Persona(
        name="Conservative CFO",
        description="Prioritizes solvency, low risk, strong compliance, and conservative financing.",
        rubric_weights={"solvency": 0.4, "compliance": 0.3, "relationship": 0.1, "growth": 0.2},
    ),
    Persona(
        name="Aggressive Founder",
        description="Prioritizes revenue growth and contract wins, while tolerating higher financing costs.",
        rubric_weights={"solvency": 0.2, "compliance": 0.1, "relationship": 0.2, "growth": 0.5},
    ),
    Persona(
        name="Regulator",
        description="Focuses primarily on legal compliance, fairness, and prudent commercial conduct.",
        rubric_weights={"solvency": 0.2, "compliance": 0.6, "relationship": 0.2, "growth": 0.0},
    ),
]


def sample_persona(rng: Random, personas: Optional[list[Persona]] = None) -> Persona:
    """Sample a persona deterministically from the supplied RNG."""
    choices = list(personas or PERSONAS)
    return choices[rng.randrange(len(choices))]


def persona_reward(persona: Persona, rubric_scores: dict[str, float]) -> float:
    """Aggregate rubric dimensions into one scalar for the given persona."""
    return round(
        sum(float(persona.rubric_weights[key]) * float(rubric_scores.get(key, 0.0)) for key in persona.rubric_weights),
        6,
    )


def pairwise_preference_label(
    persona: Persona,
    episode_a: str,
    episode_b: str,
    scorer,
) -> int:
    """Return 1 when episode A is preferred, else 0 for episode B."""
    reward_a = persona_reward(persona, scorer(episode_a))
    reward_b = persona_reward(persona, scorer(episode_b))
    return 1 if reward_a >= reward_b else 0
