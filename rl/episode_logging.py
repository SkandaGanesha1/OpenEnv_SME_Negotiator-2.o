"""Shared logging, metric, and rubric helpers for RL training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Optional, Protocol


class RubricScorerProtocol(Protocol):
    """Protocol for optional training-side rubric scoring."""

    def __call__(self, episode_log: str) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class EpisodeSummary:
    """Compact deterministic summary of one liquidity episode."""

    episode_completed: bool
    base_rl_reward: float
    tool_bonus_total: float
    env_reward_total: float
    success_no_default_positive_npv: bool
    average_final_payment_days: float
    tool_usage_count: int
    resolved_deal_count: int
    defaulted_sme_count: int
    curriculum_level: int = 0
    persona_name: Optional[str] = None
    buyer_policy_id: Optional[str] = None
    financier_policy_id: Optional[str] = None
    verifiable_reward: float = 0.0
    total_reward: float = 0.0
    tool_call_count: int = 0
    tool_effective_count: int = 0
    duplicate_tool_count: int = 0
    invalid_action_count: int = 0
    stall_step_count: int = 0
    terminated_by_step_cap: bool = False
    tool_backend_mode: Optional[str] = None


def build_episode_log(wrapper: object) -> str:
    """Build a compact deterministic text log from a bridge wrapper."""
    lines: list[str] = []

    prompt_text = str(getattr(wrapper, "prompt_text", "") or "").strip()
    if prompt_text:
        lines.append(f"Prompt={prompt_text}")

    task_name = getattr(wrapper, "task_name", None)
    difficulty = getattr(wrapper, "difficulty", None)
    seed = getattr(wrapper, "seed", None)
    total_periods = getattr(wrapper, "total_periods", None)
    curriculum_level = getattr(wrapper, "curriculum_level", None)
    persona_name = getattr(getattr(wrapper, "current_persona", None), "name", None)
    buyer_policy_id = getattr(wrapper, "buyer_policy_id", None)
    financier_policy_id = getattr(wrapper, "financier_policy_id", None)
    lines.append(
        "Config: "
        f"task_name={task_name} difficulty={difficulty} seed={seed} total_periods={total_periods} "
        f"curriculum_level={curriculum_level} persona={persona_name} "
        f"buyer_policy={buyer_policy_id} financier_policy={financier_policy_id}"
    )

    for item in list(getattr(wrapper, "episode_log_parts", [])):
        lines.append(str(item))

    env = getattr(wrapper, "env", None)
    state = getattr(env, "state", None)
    if state is not None:
        world_state = state.world_state
        lines.append(
            "World: "
            f"current_period={world_state.current_period}/{world_state.total_periods} "
            f"episode_step={world_state.episode_step} "
            f"resolved_deal_ids={state.resolved_deal_ids}"
        )
        for deal in world_state.deals:
            lines.append(
                "Deal: "
                f"id={deal.deal_id} status={deal.status} invoice_amount={deal.invoice_amount} "
                f"agreed_payment_days={deal.agreed_payment_days} financed={deal.financed} failed={deal.failed}"
            )
        for sme in world_state.smes:
            lines.append(
                "SME: "
                f"id={sme.sme_id} cash_balance={sme.cash_balance} current_utilization={sme.current_utilization} "
                f"defaulted={sme.defaulted} missed_supplier_payment={sme.missed_supplier_payment}"
            )

    summary = getattr(wrapper, "summarize_episode", None)
    if callable(summary):
        episode_summary = summary()
        lines.append(
                "Summary: "
                f"episode_completed={episode_summary.episode_completed} "
                f"base_rl_reward={episode_summary.base_rl_reward:.6f} "
                f"verifiable_reward={episode_summary.verifiable_reward:.6f} "
                f"total_reward={episode_summary.total_reward:.6f} "
                f"tool_bonus_total={episode_summary.tool_bonus_total:.6f} "
                f"env_reward_total={episode_summary.env_reward_total:.6f} "
                f"success_no_default_positive_npv={episode_summary.success_no_default_positive_npv} "
                f"average_final_payment_days={episode_summary.average_final_payment_days:.3f} "
                f"tool_usage_count={episode_summary.tool_usage_count} "
                f"tool_call_count={episode_summary.tool_call_count} "
                f"tool_effective_count={episode_summary.tool_effective_count} "
                f"duplicate_tool_count={episode_summary.duplicate_tool_count} "
                f"invalid_action_count={episode_summary.invalid_action_count} "
                f"stall_step_count={episode_summary.stall_step_count} "
                f"resolved_deal_count={episode_summary.resolved_deal_count} "
                f"defaulted_sme_count={episode_summary.defaulted_sme_count} "
                f"terminated_by_step_cap={episode_summary.terminated_by_step_cap} "
                f"tool_backend_mode={episode_summary.tool_backend_mode} "
                f"curriculum_level={episode_summary.curriculum_level} "
                f"persona_name={episode_summary.persona_name} "
                f"buyer_policy_id={episode_summary.buyer_policy_id} "
                f"financier_policy_id={episode_summary.financier_policy_id}"
        )

        # Include [STEP] and [END] markers so build_rule_based_rubric_scorer can parse them
        env_reward = float(episode_summary.env_reward_total or 0.0)
        score_val = float(episode_summary.total_reward or 0.0)
        lines.append(f"[STEP] step=1 reward={score_val:.2f}")
        use_treds_marker = "use_treds=true" if float(episode_summary.tool_usage_count or 0) > 0 else ""
        if use_treds_marker:
            lines.append(f"[STEP_DETAIL] {use_treds_marker}")
        success_flag = "true" if episode_summary.success_no_default_positive_npv else "false"
        lines.append(f"[END] success={success_flag} steps=1 score={score_val:.2f}")

    return "\n".join(lines)


def combine_rewards(
    base_reward: float,
    rubric_scores: Optional[dict[str, float]],
    rubric_weight: float,
) -> float:
    """Combine deterministic RL reward with optional rubric overlay."""
    if not rubric_scores or float(rubric_weight) <= 0.0:
        return float(base_reward)
    rubric_mean = mean(float(value) for value in rubric_scores.values())
    return float(base_reward) + float(rubric_weight) * float(rubric_mean)
