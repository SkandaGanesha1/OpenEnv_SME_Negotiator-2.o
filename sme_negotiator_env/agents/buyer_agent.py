"""Strategic buyer agent with theory-of-mind belief adaptation.

StrategicBuyerAgent wraps the existing HeuristicBuyerPolicy and overrides its
concession logic based on the buyer's latent type (cooperative/neutral/adversarial)
and its running belief about the SME's financial state.

Latent type is private information — the SME agent must infer it from observed
concession rates and then calibrate its counter-offers accordingly.
"""

from __future__ import annotations

import math
from random import Random
from typing import TYPE_CHECKING, Optional

from sme_negotiator_env.belief_state import (
    ADVERSARIAL,
    COOPERATIVE,
    NEUTRAL,
    BeliefStateManager,
    BuyerBeliefState,
    BuyerLatentType,
    assign_latent_type,
)
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation

if TYPE_CHECKING:
    from sme_negotiator_env.agents.coalition import BuyerCoalition


class StrategicBuyerAgent:
    """Buyer agent with latent strategic type and theory-of-mind belief tracking.

    Parameters
    ----------
    buyer_id:
        Stable identifier (e.g. "buyer_0").
    latent_type:
        Private strategic type — drives concession speed and coalition behaviour.
    rng:
        Seeded RNG; all randomness routed through this for reproducibility.
    coalition:
        Optional reference to the BuyerCoalition this buyer participates in.
    fallback_policy:
        Fallback for neutral / cooperative modes. If None, the agent uses its
        own heuristic concession logic.
    """

    def __init__(
        self,
        buyer_id: str,
        latent_type: BuyerLatentType,
        rng: Random,
        *,
        coalition: Optional["BuyerCoalition"] = None,
        fallback_policy=None,
    ) -> None:
        self.buyer_id = str(buyer_id)
        self._latent_type = latent_type
        self._rng = rng
        self._coalition = coalition
        self._fallback_policy = fallback_policy

        self._belief_manager = BeliefStateManager()
        self._belief: BuyerBeliefState = self._belief_manager.initial_buyer_belief(buyer_id)

        # Track last action for coalition coordination
        self._last_action: Optional[NegotiationAction] = None
        # Regulatory warning active flag (set by RegulatorAgent)
        self._regulatory_warning_active: bool = False
        self._regulatory_fear_boost: float = 0.0

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    @property
    def policy_id(self) -> str:
        return f"strategic_buyer:{self.buyer_id}:{self._latent_type.name}"

    @property
    def latent_type(self) -> BuyerLatentType:
        return self._latent_type

    @property
    def belief(self) -> BuyerBeliefState:
        return self._belief

    def act(self, observation: NegotiationObservation) -> NegotiationAction:
        """Return a strategic counter-offer based on latent type and belief state."""
        # If coalition is active, defer to coordinated strategy
        if self._coalition is not None and self._coalition.is_active:
            partner_last = self._coalition.partner_last_action(self.buyer_id)
            if partner_last is not None:
                return self._coalition_coordinated_action(observation, partner_last)

        # Effective type may be modified by regulatory warning
        effective_type = self._effective_latent_type()

        if effective_type.name == "cooperative":
            action = self._cooperative_action(observation)
        elif effective_type.name == "adversarial":
            action = self._adversarial_action(observation)
        else:
            action = self._neutral_action(observation)

        self._last_action = action
        return action

    def update_belief(
        self,
        sme_action: NegotiationAction,
        *,
        prev_sme_price: Optional[float] = None,
        current_sme_price: Optional[float] = None,
    ) -> None:
        """Update this buyer's belief based on an observed SME action."""
        self._belief = self._belief_manager.update_buyer_belief_on_sme_action(
            self._belief,
            sme_action,
            prev_sme_price=prev_sme_price,
            current_sme_price=current_sme_price,
        )

    def apply_regulatory_warning(self, fear_boost: float = 0.3) -> None:
        """Called by RegulatorAgent when a warning is issued against this buyer."""
        self._regulatory_warning_active = True
        self._regulatory_fear_boost = min(1.0, float(fear_boost))

    def clear_regulatory_warning(self) -> None:
        """Clear regulatory warning state (e.g., after buyer concedes sufficiently)."""
        self._regulatory_warning_active = False
        self._regulatory_fear_boost = 0.0

    def should_defect_coalition(
        self,
        split_offer_days: int,
        coalition_expected_days: int,
        *,
        defection_payoff_ratio: float = 1.1,
    ) -> bool:
        """Return True if accepting the SME's split offer is individually rational.

        Defection is rational when the split offer days are meaningfully better
        (fewer days = better for SME, worse for buyer), BUT buyers want MORE days.
        So defection is rational when split_offer_days > coalition_expected_days * ratio
        — i.e., buyer gets MORE days by defecting.
        """
        return float(split_offer_days) > float(coalition_expected_days) * defection_payoff_ratio

    # ------------------------------------------------------------------ #
    # Private strategy implementations                                      #
    # ------------------------------------------------------------------ #

    def _effective_latent_type(self) -> BuyerLatentType:
        """Return the effective type, possibly moderated by regulatory warning."""
        if not self._regulatory_warning_active:
            return self._latent_type

        combined_fear = min(
            1.0, self._latent_type.regulatory_fear + self._regulatory_fear_boost
        )
        # Fear > 0.5 → adversarial reverts to neutral, neutral stays neutral
        if combined_fear > 0.5 and self._latent_type.name == "adversarial":
            return NEUTRAL
        return self._latent_type

    def _cooperative_action(self, obs: NegotiationObservation) -> NegotiationAction:
        """Cooperative: rapid concession, accepts TReDS proposals immediately."""
        current_days = int(obs.buyer_days)
        # Concede 3–5 days per round
        concession = self._rng.randint(3, 5)
        proposed_days = max(int(obs.liquidity_threshold), current_days - concession)
        return NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=proposed_days,
            use_treds=bool(obs.treds_bonus > 0),
            reason="Cooperative buyer: accelerating concession toward agreement",
        )

    def _neutral_action(self, obs: NegotiationObservation) -> NegotiationAction:
        """Neutral: standard heuristic concession curve."""
        if self._fallback_policy is not None:
            return self._fallback_policy.act(obs)
        # Built-in neutral heuristic
        current_days = int(obs.buyer_days)
        concession = max(1, int(current_days * 0.04))  # 4% per round
        proposed_days = max(int(obs.liquidity_threshold) + 5, current_days - concession)
        return NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=proposed_days,
            use_treds=False,
            reason="Neutral buyer: standard concession",
        )

    def _adversarial_action(self, obs: NegotiationObservation) -> NegotiationAction:
        """Adversarial: holds position until late rounds, then soft-concedes.

        Adversarial buyers hold firm for 60% of max_rounds, then concede
        at a slow rate. If SME distress is estimated high, they hold even longer.
        """
        round_num = int(obs.round_number)
        max_rounds = max(1, int(obs.max_rounds))
        current_days = int(obs.buyer_days)
        distress = self._belief.sme_distress_estimate

        # Extend holdout if SME appears distressed
        holdout_fraction = 0.6 + 0.2 * distress
        if round_num < math.ceil(max_rounds * holdout_fraction):
            # Hold position — minimal concession
            concession = 0 if distress > 0.5 else 1
        else:
            # Late-game slow concession: 1–2 days
            concession = self._rng.randint(1, 2)

        proposed_days = max(int(obs.liquidity_threshold) + 10, current_days - concession)
        return NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=proposed_days,
            use_treds=False,
            reason="Adversarial buyer: holding position",
        )

    def _coalition_coordinated_action(
        self,
        obs: NegotiationObservation,
        partner_last: NegotiationAction,
    ) -> NegotiationAction:
        """Mirror partner's last payment_days demand for coordinated holdout."""
        partner_days = int(partner_last.payment_days or obs.buyer_days)
        # Coordinate on the HIGHER of our current demand and partner's demand
        joint_days = max(int(obs.buyer_days), partner_days)
        return NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=joint_days,
            use_treds=False,
            reason=f"Coalition holdout: coordinating at {joint_days} days",
        )


# ======================================================================= #
# Factory                                                                   #
# ======================================================================= #

def make_strategic_buyer(
    buyer_id: str,
    *,
    buyer_index: int,
    difficulty: str,
    seed: int,
    rng: Optional[Random] = None,
    fallback_policy=None,
) -> StrategicBuyerAgent:
    """Construct a StrategicBuyerAgent with deterministic latent type assignment."""
    latent_type = assign_latent_type(
        buyer_index=buyer_index,
        difficulty=difficulty,
        seed=seed,
    )
    agent_rng = rng or Random(seed + buyer_index * 13)
    return StrategicBuyerAgent(
        buyer_id=buyer_id,
        latent_type=latent_type,
        rng=agent_rng,
        fallback_policy=fallback_policy,
    )
