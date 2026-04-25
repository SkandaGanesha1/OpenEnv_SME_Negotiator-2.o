"""Canonical immutable reward breakdown contract for the SME negotiation environment.

Every component that produces or consumes reward data should import from here.
The RewardBreakdown dataclass is the single source of truth for all reward
sub-components exposed to the TRL training stack and W&B monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class RewardBreakdown:
    """Immutable record of all reward components for one environment step.

    frozen=True prevents post-construction mutation anywhere in the call stack.
    The ``total`` field is the scalar the environment surfaces as its step reward;
    the remaining fields are sub-components visible to the trainer and monitor.
    """

    total: float

    # ------------------------------------------------------------------ #
    # Terminal 4-component RLVR (meaningful only when is_terminal=True)   #
    # ------------------------------------------------------------------ #
    solvency: float = 0.0
    liquidity: float = 0.0
    npv: float = 0.0
    compliance: float = 0.0

    # ------------------------------------------------------------------ #
    # Dense shaping components (per-step, range [-1, 1])                  #
    # ------------------------------------------------------------------ #
    gap_term: float = 0.0
    days_term: float = 0.0
    alignment_term: float = 0.0

    # ------------------------------------------------------------------ #
    # Process supervision components (per-step, range [-1, 1])            #
    # ------------------------------------------------------------------ #
    reasoning_quality: float = 0.0
    tool_strategic_use: float = 0.0
    format_compliance: float = 0.0

    # ------------------------------------------------------------------ #
    # Anti-cheat penalties (always <= 0)                                  #
    # ------------------------------------------------------------------ #
    proposal_loop_penalty: float = 0.0
    invalid_accept_penalty: float = 0.0
    tool_dedup_penalty: float = 0.0

    # ------------------------------------------------------------------ #
    # Step metadata                                                        #
    # ------------------------------------------------------------------ #
    is_terminal: bool = False
    termination_reason: str = ""
    episode_step: int = 0

    # ------------------------------------------------------------------ #
    # Aggregate helpers                                                    #
    # ------------------------------------------------------------------ #

    def terminal_component(self) -> float:
        """Weighted sum of the four terminal RLVR components.

        Weights match compute_verifiable_reward: solvency=0.35, liquidity=0.20,
        npv=0.35, compliance=0.10. Valid only when is_terminal=True.
        """
        return (
            0.35 * self.solvency
            + 0.20 * self.liquidity
            + 0.35 * self.npv
            + 0.10 * self.compliance
        )

    def shaping_component(self) -> float:
        """Weighted sum of the three dense shaping components."""
        return (
            0.5 * self.gap_term
            + 0.3 * self.days_term
            + 0.2 * self.alignment_term
        )

    def process_component(self) -> float:
        """Weighted sum of process supervision signals."""
        return (
            0.5 * self.reasoning_quality
            + 0.3 * self.tool_strategic_use
            + 0.2 * self.format_compliance
        )

    def penalty_total(self) -> float:
        """Sum of all anti-cheat penalties (non-positive)."""
        return (
            self.proposal_loop_penalty
            + self.invalid_accept_penalty
            + self.tool_dedup_penalty
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a flat dict (suitable for metadata injection)."""
        d: dict[str, Any] = {}
        for f in fields(self):
            d[f.name] = getattr(self, f.name)
        # Add computed aggregates for convenience
        d["terminal_component"] = round(self.terminal_component(), 6)
        d["shaping_component"] = round(self.shaping_component(), 6)
        d["process_component"] = round(self.process_component(), 6)
        d["penalty_total"] = round(self.penalty_total(), 6)
        return d


def merge_breakdown_list(breakdowns: list[RewardBreakdown]) -> RewardBreakdown:
    """Aggregate a list of per-step RewardBreakdown objects into one episode summary.

    Terminal RLVR components are taken from the last entry with is_terminal=True.
    Shaping and process components are summed across all steps.
    Penalties are summed across all steps.
    """
    if not breakdowns:
        return RewardBreakdown(total=0.0)

    total = sum(b.total for b in breakdowns)

    # Terminal from last terminal step
    terminal_bd = next(
        (b for b in reversed(breakdowns) if b.is_terminal),
        breakdowns[-1],
    )

    # Sum shaping
    gap_term = sum(b.gap_term for b in breakdowns)
    days_term = sum(b.days_term for b in breakdowns)
    alignment_term = sum(b.alignment_term for b in breakdowns)

    # Sum process supervision
    reasoning_quality = sum(b.reasoning_quality for b in breakdowns)
    tool_strategic_use = sum(b.tool_strategic_use for b in breakdowns)
    format_compliance = sum(b.format_compliance for b in breakdowns)

    # Sum penalties
    proposal_loop_penalty = sum(b.proposal_loop_penalty for b in breakdowns)
    invalid_accept_penalty = sum(b.invalid_accept_penalty for b in breakdowns)
    tool_dedup_penalty = sum(b.tool_dedup_penalty for b in breakdowns)

    return RewardBreakdown(
        total=round(total, 6),
        solvency=terminal_bd.solvency,
        liquidity=terminal_bd.liquidity,
        npv=terminal_bd.npv,
        compliance=terminal_bd.compliance,
        gap_term=round(gap_term, 6),
        days_term=round(days_term, 6),
        alignment_term=round(alignment_term, 6),
        reasoning_quality=round(reasoning_quality, 6),
        tool_strategic_use=round(tool_strategic_use, 6),
        format_compliance=round(format_compliance, 6),
        proposal_loop_penalty=round(proposal_loop_penalty, 6),
        invalid_accept_penalty=round(invalid_accept_penalty, 6),
        tool_dedup_penalty=round(tool_dedup_penalty, 6),
        is_terminal=terminal_bd.is_terminal,
        termination_reason=terminal_bd.termination_reason,
        episode_step=breakdowns[-1].episode_step,
    )


def breakdown_to_trl_reward_dict(bd: RewardBreakdown) -> dict[str, float]:
    """Map a RewardBreakdown to the named reward columns GRPOTrainer expects.

    This is the canonical mapping used by rl/reward_functions.py to route
    each sub-component to its corresponding reward_func in the split list.

    reward_weights=[1.0, 0.3, 0.2, 1.0] maps to:
        outcome_reward, format_reward, process_reward, anti_hack_penalty
    """
    return {
        "outcome_reward": round(bd.terminal_component(), 6),
        "format_reward": round(bd.format_compliance, 6),
        "process_reward": round(bd.process_component(), 6),
        "anti_hack_penalty": round(bd.penalty_total(), 6),
    }
