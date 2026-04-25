"""Deterministic tests for additive reward-reporting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.environment import SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction
from sme_negotiator_env.reward_reporting import build_legacy_step_diagnostics, build_shadow_reward_report


def _run_legacy_trace() -> tuple[object, list[NegotiationAction], list[object], object]:
    env = SMENegotiatorEnvironment()
    reset_observation = env.reset(seed=1000, difficulty="EASY", task_name="payment-terms-easy")
    actions = [
        NegotiationAction(action_type="propose", price=90.0, payment_days=60, use_treds=False),
        NegotiationAction(action_type="propose", price=85.0, payment_days=60, use_treds=False),
        NegotiationAction(action_type="propose", price=82.0, payment_days=60, use_treds=False),
        NegotiationAction(action_type="accept", price=82.0, payment_days=60, use_treds=False),
    ]
    observations: list[object] = []
    for action in actions:
        observations.append(env.step(action))
    assert env.state is not None
    return reset_observation, actions, observations, env.state.model_copy(deep=True)


def test_shadow_rlvr_report_is_deterministic_for_fixed_trace() -> None:
    reset_observation, actions, observations, final_state = _run_legacy_trace()

    report_a = build_shadow_reward_report(
        reset_observation=reset_observation,
        actions=actions,
        step_observations=observations,
        seed=1000,
        final_state=final_state,
    )
    report_b = build_shadow_reward_report(
        reset_observation=reset_observation,
        actions=actions,
        step_observations=observations,
        seed=1000,
        final_state=final_state,
    )

    assert report_a.to_dict() == report_b.to_dict()


def test_reward_summary_keeps_live_rewards_unchanged() -> None:
    reset_observation, actions, observations, final_state = _run_legacy_trace()
    original_rewards = [float(observation.reward) for observation in observations]

    report = build_shadow_reward_report(
        reset_observation=reset_observation,
        actions=actions,
        step_observations=observations,
        seed=1000,
        final_state=final_state,
    )

    assert [float(observation.reward) for observation in observations] == original_rewards
    assert float(observations[-1].reward) == 0.99
    assert report.shadow_total_sme_reward >= report.shadow_verifiable_reward


def test_worsening_proposals_can_descend_before_high_terminal_score() -> None:
    _, actions, observations, _ = _run_legacy_trace()
    non_terminal_rewards = [float(observation.reward) for observation in observations[:-1]]
    terminal_reward = float(observations[-1].reward)

    assert len(actions) == 4
    assert non_terminal_rewards[0] > non_terminal_rewards[1] > non_terminal_rewards[2]
    assert terminal_reward > 0.9


def test_improving_proposals_have_non_descending_shaping_rewards() -> None:
    env = SMENegotiatorEnvironment()
    env.reset(seed=1000, difficulty="easy", task_name="payment-terms-easy")
    observations = [
        env.step(NegotiationAction(action_type="propose", price=85.0, payment_days=80, use_treds=False)),
        env.step(NegotiationAction(action_type="propose", price=86.0, payment_days=70, use_treds=False)),
        env.step(NegotiationAction(action_type="propose", price=87.0, payment_days=60, use_treds=False)),
    ]
    rewards = [float(observation.reward) for observation in observations]

    assert rewards[0] <= rewards[1] <= rewards[2]


def test_step_diagnostics_distinguish_shaping_from_terminal() -> None:
    _, _, observations, _ = _run_legacy_trace()

    shaping_diagnostics = build_legacy_step_diagnostics(
        observations[0],
        reward=float(observations[0].reward),
        last_valid_proposal={"price": 90.0, "payment_days": 60},
    )
    terminal_diagnostics = build_legacy_step_diagnostics(
        observations[-1],
        reward=float(observations[-1].reward),
        last_valid_proposal={"price": 82.0, "payment_days": 60},
    )

    assert shaping_diagnostics.legacy_terminal_score is None
    assert "partial_progress" in shaping_diagnostics.legacy_reward_branch
    assert terminal_diagnostics.legacy_terminal_score == float(observations[-1].reward)
    assert terminal_diagnostics.close_zone_flag is False
