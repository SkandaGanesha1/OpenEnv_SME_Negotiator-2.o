"""Unit tests for the rewritten SME negotiation OpenEnv package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.sme_environment import SMENegotiatorEnvironment
from sme_negotiator_env.client import choose_action
from sme_negotiator_env.models import NegotiationAction


def test_reset_is_seed_deterministic() -> None:
    env1 = SMENegotiatorEnvironment()
    env2 = SMENegotiatorEnvironment()

    observation1 = env1.reset(seed=42, difficulty="easy")
    observation2 = env2.reset(seed=42, difficulty="easy")

    assert observation1.buyer_price == observation2.buyer_price
    assert observation1.buyer_days == observation2.buyer_days
    assert observation1.metadata.get("base_concede") == observation2.metadata.get("base_concede")


def test_reset_sets_difficulty_profile() -> None:
    env = SMENegotiatorEnvironment()

    easy = env.reset(seed=42, difficulty="easy")
    medium = env.reset(seed=42, difficulty="medium")
    hard = env.reset(seed=42, difficulty="hard")

    assert easy.max_rounds == 10
    assert medium.max_rounds == 12
    assert hard.max_rounds == 16
    assert hard.volume == 5000


def test_accept_completes_episode() -> None:
    env = SMENegotiatorEnvironment()
    observation = env.reset(seed=42, difficulty="easy")

    action = NegotiationAction(
        action_type="accept",
        price=observation.buyer_price,
        payment_days=observation.buyer_days,
        use_treds=False,
    )

    result = env.step(action)

    assert result.done is True
    assert result.buyer_accepted is True
    # Task grader scores the financial outcome (90d accept is rarely optimal credit)
    assert result.reward is not None


def test_reject_ends_episode_without_reward() -> None:
    env = SMENegotiatorEnvironment()
    env.reset(seed=42, difficulty="easy")

    result = env.step(
        NegotiationAction(
            action_type="reject",
            price=95.0,
            payment_days=30,
            use_treds=False,
        )
    )

    assert result.done is True
    assert result.buyer_accepted is False
    assert result.reward == 0.0


def test_choose_action_uses_current_observation() -> None:
    env = SMENegotiatorEnvironment()
    observation = env.reset(seed=42, difficulty="hard")

    action = choose_action(observation, round_number=0)

    assert action.action_type == "propose"
    assert action.price >= observation.cost_threshold
    assert action.payment_days <= observation.liquidity_threshold


def test_state_exposed_as_attribute() -> None:
    env = SMENegotiatorEnvironment()

    assert env.state is None
    env.reset(seed=42, difficulty="easy")

    assert env.state is not None
    assert hasattr(env.state, "episode_id")


def test_max_rounds_success_flag_matches_terminal_reward() -> None:
    env = SMENegotiatorEnvironment()
    observation = env.reset(seed=7, difficulty="medium")

    while not observation.done:
        observation = env.step(
            NegotiationAction(
                action_type="propose",
                price=max(observation.cost_threshold + 1.0, observation.buyer_price - 0.5),
                payment_days=max(observation.liquidity_threshold + 8, observation.buyer_days - 1),
                use_treds=False,
            )
        )

    assert observation.metadata.get("termination_reason") == "max_rounds_no_deal"
    assert observation.metadata.get("success") == (observation.reward > 0.0)
