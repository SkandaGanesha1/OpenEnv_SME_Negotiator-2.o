"""Split TRL reward functions for GRPOTrainer.

Each function extracts one sub-component from the environment's RewardBreakdown.
Use make_all_reward_funcs() to get the canonical (reward_funcs, reward_weights)
bundle ready for GRPOTrainer.

Reward weights:
    outcome_reward    weight=1.0  — terminal RLVR (solvency+liquidity+NPV+compliance)
    format_reward     weight=0.3  — valid action structure + non-empty reason
    process_reward    weight=0.2  — reasoning quality + tool strategic use
    anti_hack_penalty weight=1.0  — proposal loops + invalid accepts + tool dedup

The anti_hack_penalty has weight=1.0 so exploits receive the same magnitude
penalty as the primary reward signal.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .bridge import get_episode_log, get_episode_summary, get_final_reward

# ======================================================================= #
# Internal helpers                                                         #
# ======================================================================= #

def _get_breakdown(env: Any) -> Any:
    """Safely retrieve the RewardBreakdown from a NegotiatorEnvFactory."""
    try:
        from sme_negotiator_env.reward_breakdown import RewardBreakdown
        bd = getattr(env, "reward_breakdown", None)
        if bd is None:
            return RewardBreakdown(total=float(getattr(env, "reward", 0.0)))
        return bd
    except Exception:
        return _null_breakdown(env)


def _null_breakdown(env: Any) -> Any:
    try:
        from sme_negotiator_env.reward_breakdown import RewardBreakdown
        return RewardBreakdown(total=float(getattr(env, "reward", 0.0)))
    except Exception:
        class _Null:
            def terminal_component(self): return 0.0
            def process_component(self): return 0.0
            def penalty_total(self): return 0.0
            format_compliance = 0.0
        return _Null()


# ======================================================================= #
# Individual reward function factories                                      #
# ======================================================================= #

def make_outcome_reward(
    rubric_scorer: Optional[Callable] = None,
    rubric_weight: float = 0.0,
    summary_buffer: Optional[Any] = None,
) -> Callable:
    """Primary outcome reward: terminal_component() from RewardBreakdown.

    Combines solvency (35%), liquidity (20%), NPV (35%), compliance (10%).
    Identical logic to the existing make_reward_function() — drop-in replacement.

    Parameters
    ----------
    rubric_scorer:
        Optional callable that takes an episode_log string and returns a dict
        of {solvency, compliance, relationship, growth} scores.
    rubric_weight:
        Weight for rubric overlay (0.0 disables it).
    summary_buffer:
        Optional buffer that accumulates episode summaries for logging.
    """
    def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for env in environments:
            bd = _get_breakdown(env)
            base = float(bd.terminal_component())

            # Fall back to compute_final_reward() if no terminal components set
            if base == 0.0 and not getattr(bd, "is_terminal", False):
                try:
                    base = get_final_reward(env)
                except Exception:
                    base = float(getattr(env, "reward", 0.0))

            final = base
            if rubric_scorer is not None and rubric_weight > 0.0:
                try:
                    episode_log = get_episode_log(env)
                    rubric_scores = rubric_scorer(episode_log)
                    persona = getattr(env, "current_persona", None)
                    if persona is not None and hasattr(persona, "rubric_weights"):
                        overlay = sum(
                            float(rubric_scores.get(k, 0.0)) * float(v)
                            for k, v in persona.rubric_weights.items()
                            if isinstance(v, (int, float))
                        )
                        final += rubric_weight * overlay
                except Exception:
                    pass

            if summary_buffer is not None:
                try:
                    summary_buffer.append(get_episode_summary(env))
                except Exception:
                    pass

            rewards.append(round(final, 6))
        return rewards

    reward_func.__name__ = "outcome_reward"
    return reward_func


def make_format_reward() -> Callable:
    """Format compliance reward.

    Returns bd.format_compliance: 1.0 if the action had a non-trivial reason
    field, 0.0 otherwise. Encourages the model to always justify proposals.

    Weight: 0.3 (secondary signal, not task-defining)
    """
    def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for env in environments:
            bd = _get_breakdown(env)
            rewards.append(round(float(bd.format_compliance), 6))
        return rewards

    reward_func.__name__ = "format_reward"
    return reward_func


def make_process_reward() -> Callable:
    """Process supervision reward.

    Returns bd.process_component(): weighted sum of reasoning_quality (50%),
    tool_strategic_use (30%), and format_compliance (20%).

    Encourages the model to cite numeric context, use tools in the right order,
    and demonstrate step-level progress.

    Weight: 0.2 (lighter signal — approximated from heuristics, not verifiable)
    """
    def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for env in environments:
            bd = _get_breakdown(env)
            rewards.append(round(float(bd.process_component()), 6))
        return rewards

    reward_func.__name__ = "process_reward"
    return reward_func


def make_anti_hack_penalty() -> Callable:
    """Anti-cheat penalty (always <= 0).

    Returns bd.penalty_total(): sum of proposal_loop_penalty (-0.05 each),
    invalid_accept_penalty (-0.10), and tool_dedup_penalty (-0.01 each).

    Weight: 1.0 — same magnitude as outcome_reward so hacks receive full
    counterpressure rather than being negligible noise.
    """
    def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for env in environments:
            bd = _get_breakdown(env)
            rewards.append(round(float(bd.penalty_total()), 6))
        return rewards

    reward_func.__name__ = "anti_hack_penalty"
    return reward_func


# ======================================================================= #
# Convenience bundle                                                        #
# ======================================================================= #

def make_all_reward_funcs(
    rubric_scorer: Optional[Callable] = None,
    rubric_weight: float = 0.0,
    summary_buffer: Optional[Any] = None,
) -> tuple[list[Callable], list[float]]:
    """Return (reward_funcs, reward_weights) ready for GRPOTrainer.

    Usage::

        reward_funcs, reward_weights = make_all_reward_funcs(
            rubric_scorer=my_rubric,
            rubric_weight=0.1,
            summary_buffer=episode_log_buffer,
        )
        trainer = GRPOTrainer(
            reward_funcs=reward_funcs,
            args=GRPOConfig(reward_weights=reward_weights, ...),
            ...
        )

    The four functions share the same signature:
        def reward_func(environments: list[NegotiatorEnvFactory], **kwargs) -> list[float]

    Returns
    -------
    reward_funcs:
        [outcome_reward, format_reward, process_reward, anti_hack_penalty]
    reward_weights:
        [1.0, 0.3, 0.2, 1.0]
    """
    reward_funcs = [
        make_outcome_reward(rubric_scorer, rubric_weight, summary_buffer),
        make_format_reward(),
        make_process_reward(),
        make_anti_hack_penalty(),
    ]
    reward_weights = [1.0, 0.3, 0.2, 1.0]
    return reward_funcs, reward_weights
