"""Process supervision layer for the SME negotiation environment.

Evaluates HOW the agent is reasoning and acting, not just WHAT it decides.
All sub-graders are fully deterministic (regex/heuristic, no LLM calls).

Components:
- ReasoningQualityGrader:  scores the agent's ``reason`` field per step
- PlanExecutionGrader:     checks whether simulate_plan preceded hard-task accepts
- ToolSequenceGrader:      rewards optimal tool ordering (QUERY_TREDS → RUN_CASHFLOW_SIM)
- StepProgressTracker:     per-step monotone improvement bonuses / regression penalties
- ProcessSupervisor:       facade that owns all four sub-graders
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ======================================================================= #
# ReasoningQualityGrader                                                   #
# ======================================================================= #

class ReasoningQualityGrader:
    """Score the quality of the agent's ``reason`` field per step.

    Returns float in [0, 1]. Fully deterministic — no LLM calls.

    Scoring:
    - 0.0 if reason is missing, empty, or < MIN_REASON_LENGTH chars
    - 0.0 if reason matches a NEGATIVE_PATTERN (single-word, template copy)
    - 0.1 base for a non-empty reason above the length threshold
    - +0.13 per POSITIVE_PATTERN match (numeric references to domain concepts)
    - Capped at 1.0 (≈ 7 pattern matches needed for max score)
    """

    MIN_REASON_LENGTH: int = 15

    # Each pattern match adds 0.13 to the score
    POSITIVE_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b\d+\s*(?:days?|d)\b", re.I),               # "45 days"
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b", re.I),   # "22%"
        re.compile(r"\bwcg\b|working[\s\-]capital[\s\-]gap", re.I),
        re.compile(r"interest[\s\-]rate", re.I),
        re.compile(r"buyer[\s\-]days?|payment[\s\-]terms?", re.I),
        re.compile(r"\bnpv\b|net[\s\-]present[\s\-]value", re.I),
        re.compile(r"\btreds\b|financ(?:ing|e)|discount", re.I),
        re.compile(r"liquidity[\s\-]threshold|cash[\s\-](?:flow|balance)", re.I),
    ]

    NEGATIVE_PATTERNS: list[re.Pattern] = [
        re.compile(r"^(propose|accept|reject|ok|done)\s*$", re.I),
        re.compile(r"default action after failed", re.I),
        re.compile(r"^no reason\s*$", re.I),
        re.compile(r"^n/?a\s*$", re.I),
    ]

    def grade(self, reason: Optional[str], *, action_type: str = "propose") -> float:
        """Return a quality score for the given reason string."""
        if not reason:
            return 0.0
        text = reason.strip()
        if len(text) < self.MIN_REASON_LENGTH:
            return 0.0
        for pat in self.NEGATIVE_PATTERNS:
            if pat.search(text):
                return 0.0

        score = 0.1  # base for non-empty, length-passing reason
        for pat in self.POSITIVE_PATTERNS:
            if pat.search(text):
                score += 0.13
        return round(min(1.0, score), 4)


# ======================================================================= #
# PlanExecutionGrader                                                      #
# ======================================================================= #

class PlanExecutionGrader:
    """Check whether simulate_plan was called before accepting a hard-task deal.

    Tracks tool/simulate_plan history across the episode and returns a
    binary bonus at accept time.
    """

    def __init__(self) -> None:
        self._simulation_called_for_deal: set[str] = set()
        self._tool_called_for_deal: set[str] = set()

    def record_simulate_plan(self, deal_id: str) -> None:
        self._simulation_called_for_deal.add(deal_id)

    def record_tool_call(self, deal_id: str, tool_name: str) -> None:
        self._tool_called_for_deal.add(f"{deal_id}:{tool_name}")

    def grade_accept(
        self,
        deal_id: str,
        *,
        difficulty: str,
        dynamic_discounting: bool,
    ) -> float:
        """Return +0.1 if hard task and simulation was called before accept, else 0.0."""
        if difficulty != "hard" and not dynamic_discounting:
            return 0.0
        return 0.1 if deal_id in self._simulation_called_for_deal else 0.0

    def reset(self) -> None:
        self._simulation_called_for_deal.clear()
        self._tool_called_for_deal.clear()


# ======================================================================= #
# ToolSequenceGrader                                                       #
# ======================================================================= #

class ToolSequenceGrader:
    """Score the strategic ordering of tool calls relative to proposals.

    Optimal sequence: QUERY_TREDS → RUN_CASHFLOW_SIM → propose
    Penalized: proposal before any tool on medium/hard tasks
    Easy: neutral (0.0) — no tool use expected
    """

    def __init__(self) -> None:
        self._tool_sequence: dict[str, list[str]] = {}       # deal_id → tool names
        self._first_proposal_step: dict[str, int] = {}       # deal_id → step index
        self._current_step: int = 0

    def record_tool(self, deal_id: str, tool_name: str) -> None:
        self._tool_sequence.setdefault(deal_id, []).append(tool_name)

    def record_proposal(self, deal_id: str) -> None:
        if deal_id not in self._first_proposal_step:
            self._first_proposal_step[deal_id] = self._current_step

    def advance_step(self) -> None:
        self._current_step += 1

    def grade_episode_sequence(self, deal_id: str, *, difficulty: str) -> float:
        """Return per-deal sequence score in [-0.05, +0.05].

        Called at episode end or when a deal is resolved.
        """
        if difficulty == "easy":
            return 0.0

        sequence = self._tool_sequence.get(deal_id, [])
        first_proposal = self._first_proposal_step.get(deal_id)

        # Penalty: proposal before any tool call on medium/hard
        if not sequence and first_proposal is not None:
            return -0.02

        # Bonus: QUERY_TREDS appears before RUN_CASHFLOW_SIM
        has_treds = "QUERY_TREDS" in sequence
        has_sim = "RUN_CASHFLOW_SIM" in sequence
        if has_treds and has_sim:
            treds_idx = sequence.index("QUERY_TREDS")
            sim_idx = sequence.index("RUN_CASHFLOW_SIM")
            if treds_idx < sim_idx:
                return 0.05
        return 0.0

    def reset(self) -> None:
        self._tool_sequence.clear()
        self._first_proposal_step.clear()
        self._current_step = 0


# ======================================================================= #
# StepProgressTracker                                                      #
# ======================================================================= #

class StepProgressTracker:
    """Track whether each step moved the negotiation closer to the target.

    Per-step bonuses / penalties:
    - +0.02 if proposed/agreed days strictly improve (decrease) vs best so far
    - -0.01 if proposed/agreed days worsen (increase) vs best so far
    - 0.0  if same as best so far
    """

    def __init__(self, *, target_days: int) -> None:
        self._target_days = int(target_days)
        self._best_days_so_far: Optional[int] = None
        self._per_step_bonuses: list[float] = []

    def record_step_outcome(self, current_days: int) -> float:
        """Record one step's days and return the per-step progress bonus."""
        days = int(current_days)
        if self._best_days_so_far is None:
            self._best_days_so_far = days
            bonus = 0.0
        elif days < self._best_days_so_far:
            self._best_days_so_far = days
            bonus = 0.02
        elif days > self._best_days_so_far:
            bonus = -0.01
        else:
            bonus = 0.0
        self._per_step_bonuses.append(bonus)
        return bonus

    def total_progress_bonus(self) -> float:
        return round(sum(self._per_step_bonuses), 6)

    def reset(self) -> None:
        self._best_days_so_far = None
        self._per_step_bonuses.clear()


# ======================================================================= #
# ProcessSupervisor — facade                                               #
# ======================================================================= #

class ProcessSupervisor:
    """Top-level facade over all process supervision components.

    Instantiated once per episode by the environment (at reset time).
    The environment calls the ``on_*`` hooks at the appropriate points in
    its step logic; the facade accumulates signals that are later assembled
    into the RewardBreakdown.

    Usage::

        supervisor = ProcessSupervisor(target_days=45, difficulty="medium")
        # in step(), after routing:
        rq = supervisor.on_propose(deal_id, action)
        # at accept:
        rq, plan_bonus = supervisor.on_accept(deal_id, action, agreed_days)
        # per-step progress:
        progress = supervisor.on_step_outcome(current_buyer_days)
        # at episode end per deal:
        seq_score = supervisor.episode_tool_sequence_score(deal_id)
    """

    def __init__(self, *, target_days: int, difficulty: str) -> None:
        self._difficulty = str(difficulty).lower()
        self.reasoning_grader = ReasoningQualityGrader()
        self.plan_grader = PlanExecutionGrader()
        self.tool_sequence_grader = ToolSequenceGrader()
        self.progress_tracker = StepProgressTracker(target_days=target_days)

    # ------------------------------------------------------------------ #
    # Event hooks                                                          #
    # ------------------------------------------------------------------ #

    def on_tool_call(self, deal_id: str, tool_name: str) -> None:
        """Called after a tool action is dispatched."""
        self.plan_grader.record_tool_call(deal_id, tool_name)
        self.tool_sequence_grader.record_tool(deal_id, tool_name)
        self.tool_sequence_grader.advance_step()

    def on_simulate_plan(self, deal_id: str) -> None:
        """Called after a simulate_plan action is dispatched."""
        self.plan_grader.record_simulate_plan(deal_id)

    def on_propose(self, deal_id: str, action: object) -> float:
        """Called after a propose action passes validation.

        Returns the reasoning_quality score for this step.
        """
        self.tool_sequence_grader.record_proposal(deal_id)
        self.tool_sequence_grader.advance_step()
        reason = getattr(action, "reason", None)
        return self.reasoning_grader.grade(reason, action_type="propose")

    def on_accept(
        self,
        deal_id: str,
        action: object,
        agreed_days: int,
    ) -> tuple[float, float]:
        """Called after an accept action passes validation.

        Returns (reasoning_quality, plan_execution_bonus).
        """
        reason = getattr(action, "reason", None)
        rq = self.reasoning_grader.grade(reason, action_type="accept")
        dynamic_discounting = bool(getattr(action, "propose_dynamic_discounting", False))
        pe = self.plan_grader.grade_accept(
            deal_id,
            difficulty=self._difficulty,
            dynamic_discounting=dynamic_discounting,
        )
        self.progress_tracker.record_step_outcome(agreed_days)
        return rq, pe

    def on_step_outcome(self, current_days: int) -> float:
        """Called at the end of any step (propose/reject/tool) with current buyer days.

        Returns the per-step progress bonus/penalty.
        """
        return self.progress_tracker.record_step_outcome(current_days)

    def episode_tool_sequence_score(self, deal_id: str) -> float:
        """Return the tool-sequence score for a deal (call at deal resolution)."""
        return self.tool_sequence_grader.grade_episode_sequence(
            deal_id, difficulty=self._difficulty
        )

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset all sub-graders. Must be called at environment reset."""
        self.plan_grader.reset()
        self.tool_sequence_grader.reset()
        self.progress_tracker.reset()
        # ReasoningQualityGrader is stateless — no reset needed
