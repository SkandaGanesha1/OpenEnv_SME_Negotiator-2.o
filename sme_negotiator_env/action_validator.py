"""Per-episode action validation layer for the SME negotiation environment.

This module is the sole source of truth for whether an action is legally valid.
The environment calls validate() as a gate before processing any action.

Design principles:
- Proposal loops are soft violations (penalty, episode continues)
- Invalid accept is a hard violation (penalty + episode termination)
- Tool deduplication is soft (tool still executes, penalty applied)
- Temporal ordering violation is soft (advance_period with open deal)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass
class ProposalLoopState:
    """Mutable per-episode state for the proposal-loop guard.

    Fires when the same (price_rounded, days) tuple is proposed IDENTICAL_LIMIT
    or more times consecutively.
    """

    IDENTICAL_LIMIT: ClassVar[int] = 3

    _consecutive: dict[tuple[float, int], int] = field(default_factory=dict)
    _last_proposal: Optional[tuple[float, int]] = field(default=None)
    _total_proposals: int = field(default=0)

    def record_proposal(self, price: float, days: int) -> bool:
        """Record a proposal. Returns True if the loop guard fires."""
        key = (round(float(price), 2), int(days))
        if key == self._last_proposal:
            self._consecutive[key] = self._consecutive.get(key, 1) + 1
        else:
            self._consecutive = {key: 1}
            self._last_proposal = key
        self._total_proposals += 1
        return self._consecutive[key] >= self.IDENTICAL_LIMIT

    def reset(self) -> None:
        self._consecutive.clear()
        self._last_proposal = None
        self._total_proposals = 0


@dataclass(frozen=True)
class ValidationResult:
    """Immutable result of one action validation check."""

    is_valid: bool
    violation_type: Optional[str]   # "proposal_loop"|"invalid_accept"|"tool_dedup"|"temporal_order"
    penalty: float                  # 0.0 if valid; negative value if violation
    message: str
    should_terminate: bool          # True only for invalid_accept (hard violation)

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True, violation_type=None, penalty=0.0,
                   message="ok", should_terminate=False)


_OK = ValidationResult.ok()


class ActionValidator:
    """Stateful per-episode action validator.

    One instance is created per episode (at reset time) and reset whenever
    the environment resets. Thread-local to one episode — not safe for
    concurrent use across episodes.

    Usage::

        validator = ActionValidator()
        # inside environment.step():
        result = validator.validate(action, deal_id=deal_id)
        if result.should_terminate:
            return terminal_obs_with_penalty(result.penalty)
        if not result.is_valid:
            step_reward += result.penalty
    """

    def __init__(self, *, max_identical_proposals: int = 3) -> None:
        self._proposal_loop = ProposalLoopState()
        self._proposal_loop.IDENTICAL_LIMIT  # accessed via class var
        # Override if caller wants a different threshold
        self._max_identical = max_identical_proposals

        # Per-deal tool fingerprints seen this round (reset on advance_round)
        self._tool_calls_this_round: dict[str, list[str]] = {}

        # Last known buyer offer per deal_id: (price_rounded, days)
        self._last_buyer_offer: dict[str, tuple[float, int]] = {}

        # Last SME proposal per deal_id: (price_rounded, days)
        self._last_sme_proposal: dict[str, tuple[float, int]] = {}

        # Resolved deal ids — cannot be acted upon
        self._resolved_deal_ids: set[str] = set()

    # ------------------------------------------------------------------ #
    # State update hooks (called by environment after processing)          #
    # ------------------------------------------------------------------ #

    def record_buyer_offer(self, deal_id: str, price: float, days: int) -> None:
        """Called by the environment after a buyer counter-offer is emitted."""
        self._last_buyer_offer[deal_id] = (round(float(price), 2), int(days))

    def record_sme_proposal(self, deal_id: str, price: float, days: int) -> None:
        """Called by the environment after an SME proposal passes validation."""
        self._last_sme_proposal[deal_id] = (round(float(price), 2), int(days))

    def mark_deal_resolved(self, deal_id: str) -> None:
        """Called when a deal is accepted or rejected (no more actions on it)."""
        self._resolved_deal_ids.add(deal_id)

    def advance_round(self) -> None:
        """Reset round-scoped tool deduplication state when a round closes."""
        self._tool_calls_this_round.clear()

    # ------------------------------------------------------------------ #
    # Main validation entry point                                          #
    # ------------------------------------------------------------------ #

    def validate(self, action: object, *, deal_id: str) -> ValidationResult:
        """Validate one action for the given deal.

        Parameters
        ----------
        action:
            A NegotiationAction (or any object with action_type, price,
            payment_days, tool_name, tool_args attributes).
        deal_id:
            The deal this action targets. For single-deal environments, pass
            the episode_id.

        Returns
        -------
        ValidationResult
            Always returned (never raises). Callers check is_valid and
            should_terminate to decide what to do.
        """
        action_type = str(getattr(action, "action_type", "propose")).lower()

        if action_type == "propose":
            return self._validate_propose(action, deal_id=deal_id)
        if action_type == "accept":
            return self._validate_accept(action, deal_id=deal_id)
        if action_type == "advance_period":
            return self._validate_advance_period(deal_id=deal_id)
        if action_type == "tool":
            return self._validate_tool(action, deal_id=deal_id)
        # reject / simulate_plan are always structurally valid
        return _OK

    # ------------------------------------------------------------------ #
    # Per-action-type helpers                                              #
    # ------------------------------------------------------------------ #

    def _validate_propose(self, action: object, *, deal_id: str) -> ValidationResult:
        price = float(getattr(action, "price", 0.0))
        days = int(getattr(action, "payment_days", 0))
        triggered = self._proposal_loop.record_proposal(price, days)
        if triggered:
            return ValidationResult(
                is_valid=False,
                violation_type="proposal_loop",
                penalty=-0.05,
                message=(
                    f"Identical proposal ({price:.2f}, {days}d) repeated "
                    f"{self._max_identical}+ times. Explore different terms."
                ),
                should_terminate=False,
            )
        # Record the proposal so accept can verify it
        self.record_sme_proposal(deal_id, price, days)
        return _OK

    def _validate_accept(self, action: object, *, deal_id: str) -> ValidationResult:
        price = float(getattr(action, "price", 0.0))
        days = int(getattr(action, "payment_days", 0))

        buyer_offer = self._last_buyer_offer.get(deal_id)
        sme_proposal = self._last_sme_proposal.get(deal_id)

        echoes_buyer = (
            buyer_offer is not None
            and abs(price - buyer_offer[0]) < 1e-4
            and days == buyer_offer[1]
        )
        echoes_sme = (
            sme_proposal is not None
            and abs(price - sme_proposal[0]) < 1e-4
            and days == sme_proposal[1]
        )

        if not echoes_buyer and not echoes_sme:
            return ValidationResult(
                is_valid=False,
                violation_type="invalid_accept",
                penalty=-0.10,
                message=(
                    f"Accept terms ({price:.2f}, {days}d) do not match the last "
                    "buyer counter-offer or your last proposal. "
                    "You can only accept terms currently on the table."
                ),
                should_terminate=True,
            )
        return _OK

    def _validate_advance_period(self, *, deal_id: str) -> ValidationResult:
        # Warn if a known deal for this deal_id is still unresolved
        if (
            deal_id
            and deal_id not in self._resolved_deal_ids
            and deal_id in self._last_buyer_offer
        ):
            return ValidationResult(
                is_valid=False,
                violation_type="temporal_order",
                penalty=-0.02,
                message=(
                    f"advance_period called while deal '{deal_id}' is still open. "
                    "Consider resolving all open deals before advancing."
                ),
                should_terminate=False,
            )
        return _OK

    def _validate_tool(self, action: object, *, deal_id: str) -> ValidationResult:
        tool_name = str(getattr(action, "tool_name", "") or "")
        tool_args = getattr(action, "tool_args", None) or {}
        fingerprint = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

        seen = self._tool_calls_this_round.setdefault(deal_id, [])
        if fingerprint in seen:
            return ValidationResult(
                is_valid=True,          # tool still executes
                violation_type="tool_dedup",
                penalty=-0.01,
                message=(
                    f"Duplicate tool call {tool_name} with identical arguments "
                    "within the same round. Prefer varied queries."
                ),
                should_terminate=False,
            )
        seen.append(fingerprint)
        return _OK

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset all per-episode state. Must be called at environment reset."""
        self._proposal_loop.reset()
        self._tool_calls_this_round.clear()
        self._last_buyer_offer.clear()
        self._last_sme_proposal.clear()
        self._resolved_deal_ids.clear()
