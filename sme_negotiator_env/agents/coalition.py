"""Buyer coalition formation, maintenance, and defection mechanics.

A BuyerCoalition models the emergent coordination between two adversarially-typed
buyers in the hard-difficulty setting. Coalition formation is rational when joint
holdout maximises both buyers' payment-day expectations; defection is rational
when one buyer receives a split-deal offer that exceeds the coalition payoff.

This implements a deterministic Shapley-inspired payoff comparison — no LLM,
no stochastic draws beyond the seeded RNG in the buyer agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sme_negotiator_env.models import NegotiationAction, NegotiationObservation

if TYPE_CHECKING:
    from sme_negotiator_env.agents.buyer_agent import StrategicBuyerAgent


# ======================================================================= #
# CoalitionFormationPolicy                                                  #
# ======================================================================= #

@dataclass(frozen=True)
class CoalitionFormationPolicy:
    """Tunable parameters controlling coalition lifecycle.

    Attributes
    ----------
    form_threshold_rounds:
        Coalition forms after this many consecutive rounds where both buyers
        concede < 2 days AND the SME's last concession is also < 2 days.
    defection_payoff_ratio:
        Buyer defects if split_offer_days > coalition_expected_days × ratio.
        > 1.0 means buyers demand a meaningfully better deal to defect.
    dissolution_on_accept:
        If True, the coalition dissolves as soon as either buyer accepts.
    """

    form_threshold_rounds: int = 3
    defection_payoff_ratio: float = 1.1
    dissolution_on_accept: bool = True


# ======================================================================= #
# BuyerCoalition                                                            #
# ======================================================================= #

class BuyerCoalition:
    """Manages the lifecycle of a two-buyer coalition.

    The coalition tracks:
    - Whether it is currently active
    - Each buyer's last action (for coordination)
    - The joint demand (coordinated holdout position)
    - Stall rounds leading to formation
    - Defection events

    Parameters
    ----------
    buyer_a, buyer_b:
        The two StrategicBuyerAgent instances that can form a coalition.
    policy:
        Formation and defection policy parameters.
    """

    def __init__(
        self,
        buyer_a: "StrategicBuyerAgent",
        buyer_b: "StrategicBuyerAgent",
        policy: Optional[CoalitionFormationPolicy] = None,
    ) -> None:
        self._buyer_a = buyer_a
        self._buyer_b = buyer_b
        self._policy = policy or CoalitionFormationPolicy()
        self._is_active: bool = False
        self._formed_at_round: Optional[int] = None
        self._joint_demand_days: Optional[int] = None
        self._last_action_a: Optional[NegotiationAction] = None
        self._last_action_b: Optional[NegotiationAction] = None
        self._joint_stall_rounds: int = 0
        self._dissolved: bool = False

        # Wire back-reference into each buyer
        buyer_a._coalition = self
        buyer_b._coalition = self

    # ------------------------------------------------------------------ #
    # State accessors                                                       #
    # ------------------------------------------------------------------ #

    @property
    def is_active(self) -> bool:
        return self._is_active and not self._dissolved

    @property
    def formed_at_round(self) -> Optional[int]:
        return self._formed_at_round

    @property
    def joint_demand_days(self) -> Optional[int]:
        return self._joint_demand_days

    @property
    def buyer_ids(self) -> list[str]:
        return [self._buyer_a.buyer_id, self._buyer_b.buyer_id]

    def partner_last_action(self, buyer_id: str) -> Optional[NegotiationAction]:
        """Return the partner's last action for the requesting buyer."""
        if buyer_id == self._buyer_a.buyer_id:
            return self._last_action_b
        if buyer_id == self._buyer_b.buyer_id:
            return self._last_action_a
        return None

    def defection_risk(self) -> float:
        """Estimate current defection probability from coalition willingness.

        Lower coalition_willingness → easier to break the coalition with a
        split offer. Ranges from 0 (stable) to 1 (very unstable).
        """
        if not self.is_active:
            return 0.0
        # Average inverse coalition willingness of both members
        a_risk = 1.0 - self._buyer_a.latent_type.coalition_willingness
        b_risk = 1.0 - self._buyer_b.latent_type.coalition_willingness
        return round((a_risk + b_risk) / 2.0, 4)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                             #
    # ------------------------------------------------------------------ #

    def record_buyer_actions(
        self,
        action_a: NegotiationAction,
        action_b: NegotiationAction,
    ) -> None:
        """Store both buyers' latest actions for coordination purposes."""
        self._last_action_a = action_a
        self._last_action_b = action_b

    def attempt_formation(
        self,
        round_number: int,
        sme_last_concession_days: int,
    ) -> bool:
        """Attempt coalition formation; return True if coalition forms.

        Formation conditions:
        1. Both buyers are conceding < 2 days this round.
        2. SME also conceding < 2 days (joint stall).
        3. Stall has lasted >= form_threshold_rounds.
        4. Both buyers' coalition_willingness > 0.4 (willing to coordinate).
        """
        if self._is_active or self._dissolved:
            return False

        both_stalling = (
            self._last_action_a is not None
            and self._last_action_b is not None
        )
        sme_stalling = int(sme_last_concession_days) < 2

        if both_stalling and sme_stalling:
            self._joint_stall_rounds += 1
        else:
            self._joint_stall_rounds = 0

        willing_a = self._buyer_a.latent_type.coalition_willingness > 0.4
        willing_b = self._buyer_b.latent_type.coalition_willingness > 0.4

        if (
            self._joint_stall_rounds >= self._policy.form_threshold_rounds
            and willing_a
            and willing_b
        ):
            self._is_active = True
            self._formed_at_round = round_number
            # Set joint demand to the higher of the two current demands
            days_a = int(self._last_action_a.payment_days) if self._last_action_a else 90
            days_b = int(self._last_action_b.payment_days) if self._last_action_b else 90
            self._joint_demand_days = max(days_a, days_b)
            return True
        return False

    def joint_counter_offer(
        self,
        obs_a: NegotiationObservation,
        obs_b: NegotiationObservation,
    ) -> tuple[NegotiationAction, NegotiationAction]:
        """Return synchronized counter-offers: both buyers demand joint_demand_days."""
        demand = self._joint_demand_days or max(int(obs_a.buyer_days), int(obs_b.buyer_days))
        action_a = NegotiationAction(
            action_type="propose",
            price=float(obs_a.buyer_price),
            payment_days=demand,
            use_treds=False,
            reason=f"Coalition holdout: joint demand {demand} days",
        )
        action_b = NegotiationAction(
            action_type="propose",
            price=float(obs_b.buyer_price),
            payment_days=demand,
            use_treds=False,
            reason=f"Coalition holdout: joint demand {demand} days",
        )
        self._last_action_a = action_a
        self._last_action_b = action_b
        return action_a, action_b

    def process_split_offer(
        self,
        split_action: NegotiationAction,
    ) -> tuple[bool, Optional[str]]:
        """Evaluate whether a split-deal offer causes coalition defection.

        Returns (defection_occurred, defecting_buyer_id).

        The SME proposes different terms to buyer_a vs buyer_b via
        split_deal_buyer_a_days / split_deal_buyer_b_days. A buyer defects
        if their offered days exceed coalition_expected * defection_payoff_ratio.
        """
        if not self.is_active:
            return False, None

        coalition_days = self._joint_demand_days or 90
        ratio = self._policy.defection_payoff_ratio

        offer_a = split_action.split_deal_buyer_a_days
        offer_b = split_action.split_deal_buyer_b_days

        defector_id: Optional[str] = None

        if offer_a is not None and self._buyer_a.should_defect_coalition(
            int(offer_a), coalition_days, defection_payoff_ratio=ratio
        ):
            defector_id = self._buyer_a.buyer_id

        if offer_b is not None and self._buyer_b.should_defect_coalition(
            int(offer_b), coalition_days, defection_payoff_ratio=ratio
        ):
            # Take the defector with the better (higher days) offer
            if defector_id is None:
                defector_id = self._buyer_b.buyer_id
            else:
                a_days = int(offer_a) if offer_a else 0
                b_days = int(offer_b) if offer_b else 0
                if b_days > a_days:
                    defector_id = self._buyer_b.buyer_id

        if defector_id is not None:
            self.dissolve()
            # Non-defecting buyer becomes adversarial (betrayal effect)
            non_defector = (
                self._buyer_b if defector_id == self._buyer_a.buyer_id else self._buyer_a
            )
            from sme_negotiator_env.belief_state import ADVERSARIAL
            object.__setattr__(non_defector, "_latent_type", ADVERSARIAL)
            return True, defector_id

        return False, None

    def on_buyer_accept(self, accepting_buyer_id: str) -> None:
        """Dissolve the coalition if a buyer accepts and policy requires it."""
        if self._policy.dissolution_on_accept:
            self.dissolve()

    def dissolve(self) -> None:
        """Dissolve the coalition."""
        self._is_active = False
        self._dissolved = True

    def reset(self) -> None:
        """Reset coalition state for a new episode."""
        self._is_active = False
        self._formed_at_round = None
        self._joint_demand_days = None
        self._last_action_a = None
        self._last_action_b = None
        self._joint_stall_rounds = 0
        self._dissolved = False
