"""Theory-of-Mind belief tracking for the multi-agent SME negotiation environment.

All belief updates are deterministic (no LLM calls, no probability sampling) so
that episodes are fully reproducible from a seed. Each agent maintains a frozen
belief state; updates return a new frozen state rather than mutating in place.

Agent belief model:
- SME believes about: buyer latent types (cooperative/neutral/adversarial),
  buyer budget pressure, best financing rate discovered so far, and how
  credible the regulatory threat appears to buyers.
- Each buyer believes about: SME cash distress, SME's intent to use TReDS,
  SME's intent to invoke the regulator, and recent stall count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional

from sme_negotiator_env.models import NegotiationAction


# ======================================================================= #
# BuyerLatentType                                                           #
# ======================================================================= #

BUYER_LATENT_TYPES: dict[str, "BuyerLatentType"] = {}


@dataclass(frozen=True)
class BuyerLatentType:
    """Latent strategic type for a buyer agent.

    This is private information — the SME must infer it from observed behavior.

    Attributes
    ----------
    name:
        One of cooperative / neutral / adversarial.
    concession_multiplier:
        Scales daily-concession speed vs the heuristic baseline (>1 = faster).
    coalition_willingness:
        Probability that the buyer joins a coalition when conditions are met.
    regulatory_fear:
        Sensitivity to a RegulatorAgent warning (0 = ignore, 1 = immediate concession).
    """

    name: Literal["cooperative", "neutral", "adversarial"]
    concession_multiplier: float
    coalition_willingness: float
    regulatory_fear: float

    def __post_init__(self) -> None:
        BUYER_LATENT_TYPES[self.name] = self


COOPERATIVE = BuyerLatentType(
    name="cooperative",
    concession_multiplier=1.5,
    coalition_willingness=0.1,
    regulatory_fear=0.8,
)
NEUTRAL = BuyerLatentType(
    name="neutral",
    concession_multiplier=1.0,
    coalition_willingness=0.5,
    regulatory_fear=0.5,
)
ADVERSARIAL = BuyerLatentType(
    name="adversarial",
    concession_multiplier=0.4,
    coalition_willingness=0.9,
    regulatory_fear=0.2,
)


def get_latent_type(name: str) -> BuyerLatentType:
    """Return the canonical BuyerLatentType for the given name."""
    return BUYER_LATENT_TYPES[name]


def assign_latent_type(
    *,
    buyer_index: int,
    difficulty: str,
    seed: int,
) -> BuyerLatentType:
    """Deterministically assign a latent type from task difficulty and seed.

    Distribution:
    - easy:  cooperative (p=0.8) or neutral (p=0.2)
    - medium: neutral (p=0.7) or adversarial (p=0.3)
    - hard two-buyer: buyer_0 → adversarial, buyer_1 → neutral (always)
    - hard single buyer: adversarial (p=0.5) or neutral (p=0.5)
    """
    diff = str(difficulty).lower()
    # Deterministic draw from seed + buyer_index to avoid correlation between buyers
    draw = (seed * 31 + buyer_index * 97) % 100 / 100.0

    if diff == "easy":
        return COOPERATIVE if draw < 0.8 else NEUTRAL
    if diff == "medium":
        return NEUTRAL if draw < 0.7 else ADVERSARIAL
    if diff == "hard":
        if buyer_index == 0:
            return ADVERSARIAL
        return NEUTRAL
    # Default
    return NEUTRAL


# ======================================================================= #
# SMEBeliefState                                                            #
# ======================================================================= #

@dataclass(frozen=True)
class SMEBeliefState:
    """What the SME agent believes about the world's other agents.

    All fields in [0, 1] unless documented otherwise.
    """

    buyer_type_estimates: Dict[str, str]
    buyer_distress_estimates: Dict[str, float]
    financier_rate_floor: float
    regulatory_threat_credibility: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "buyer_type_estimates": dict(self.buyer_type_estimates),
            "buyer_distress_estimates": dict(self.buyer_distress_estimates),
            "financier_rate_floor": self.financier_rate_floor,
            "regulatory_threat_credibility": self.regulatory_threat_credibility,
        }


# ======================================================================= #
# BuyerBeliefState                                                          #
# ======================================================================= #

@dataclass(frozen=True)
class BuyerBeliefState:
    """What one buyer agent believes about the SME.

    All probability fields in [0, 1].
    """

    buyer_id: str
    sme_distress_estimate: float
    sme_treds_intent: float
    sme_regulator_intent: float
    rounds_without_concession: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "buyer_id": self.buyer_id,
            "sme_distress_estimate": self.sme_distress_estimate,
            "sme_treds_intent": self.sme_treds_intent,
            "sme_regulator_intent": self.sme_regulator_intent,
            "rounds_without_concession": self.rounds_without_concession,
        }


# ======================================================================= #
# BeliefStateManager                                                        #
# ======================================================================= #

_RAPID_PRICE_CONCESSION_THRESHOLD = 0.03  # > 3% price drop in one step → distress signal


class BeliefStateManager:
    """Core deterministic belief-update engine for all agents.

    All methods return new frozen state objects — no mutation.
    """

    # ------------------------------------------------------------------
    # Initial belief constructors
    # ------------------------------------------------------------------

    def initial_sme_belief(
        self,
        buyer_ids: list[str],
        *,
        initial_financier_rate: float = 0.18,
    ) -> SMEBeliefState:
        """Return a fresh SME belief at episode start — all types unknown."""
        return SMEBeliefState(
            buyer_type_estimates={bid: "neutral" for bid in buyer_ids},
            buyer_distress_estimates={bid: 0.0 for bid in buyer_ids},
            financier_rate_floor=float(initial_financier_rate),
            regulatory_threat_credibility=0.5,
        )

    def initial_buyer_belief(self, buyer_id: str) -> BuyerBeliefState:
        """Return a fresh buyer belief at episode start — SME type unknown."""
        return BuyerBeliefState(
            buyer_id=buyer_id,
            sme_distress_estimate=0.0,
            sme_treds_intent=0.0,
            sme_regulator_intent=0.0,
            rounds_without_concession=0,
        )

    # ------------------------------------------------------------------
    # SME belief updates (from observing buyer behaviour)
    # ------------------------------------------------------------------

    def update_sme_belief_on_buyer_action(
        self,
        sme_belief: SMEBeliefState,
        buyer_id: str,
        buyer_action: NegotiationAction,
        round_number: int,
        *,
        prev_buyer_days: int,
        legal_max_payment_days: int = 45,
    ) -> SMEBeliefState:
        """Update SME's belief about a buyer based on the buyer's latest action."""
        current_estimates = dict(sme_belief.buyer_type_estimates)
        current_pressures = dict(sme_belief.buyer_distress_estimates)
        credibility = sme_belief.regulatory_threat_credibility

        current_type = current_estimates.get(buyer_id, "neutral")
        proposed_days = int(buyer_action.payment_days or prev_buyer_days)
        concession_days = max(0, prev_buyer_days - proposed_days)

        # Slow concession → lean adversarial
        if round_number >= 3 and concession_days < 2:
            if current_type == "neutral":
                current_estimates[buyer_id] = "adversarial"
            elif current_type == "cooperative":
                current_estimates[buyer_id] = "neutral"

        # Fast concession → lean cooperative
        if concession_days >= 5:
            current_estimates[buyer_id] = "cooperative"

        # Buyer accepted penalty clause proposal → confirmed cooperative
        if buyer_action.propose_late_payment_penalty_clause:
            current_estimates[buyer_id] = "cooperative"

        # Buyer still demanding > legal_max → regulatory_fear update
        if proposed_days > legal_max_payment_days:
            credibility = min(1.0, credibility + 0.05)

        # Update buyer budget pressure from concession depth
        pressure = current_pressures.get(buyer_id, 0.0)
        if concession_days >= 3:
            pressure = max(0.0, pressure - 0.1)  # concession → buyer not under pressure
        elif concession_days == 0:
            pressure = min(1.0, pressure + 0.05)  # stall → buyer may be under pressure
        current_pressures[buyer_id] = round(pressure, 4)

        return replace(
            sme_belief,
            buyer_type_estimates=current_estimates,
            buyer_distress_estimates=current_pressures,
            regulatory_threat_credibility=round(credibility, 4),
        )

    def update_sme_belief_on_auction_result(
        self,
        sme_belief: SMEBeliefState,
        best_rate: float,
    ) -> SMEBeliefState:
        """Update SME's knowledge of the financier rate floor after an auction."""
        new_floor = min(sme_belief.financier_rate_floor, float(best_rate))
        return replace(sme_belief, financier_rate_floor=round(new_floor, 6))

    # ------------------------------------------------------------------
    # Buyer belief updates (from observing SME behaviour)
    # ------------------------------------------------------------------

    def update_buyer_belief_on_sme_action(
        self,
        buyer_belief: BuyerBeliefState,
        sme_action: NegotiationAction,
        *,
        prev_sme_price: Optional[float] = None,
        current_sme_price: Optional[float] = None,
    ) -> BuyerBeliefState:
        """Update one buyer's belief about the SME based on the SME's latest action."""
        distress = buyer_belief.sme_distress_estimate
        treds_intent = buyer_belief.sme_treds_intent
        reg_intent = buyer_belief.sme_regulator_intent
        stall = buyer_belief.rounds_without_concession

        action_type = str(sme_action.action_type).lower()

        # Rapid price concession → distress signal
        if (
            prev_sme_price is not None
            and current_sme_price is not None
            and prev_sme_price > 0
        ):
            drop_pct = (prev_sme_price - current_sme_price) / prev_sme_price
            if drop_pct > _RAPID_PRICE_CONCESSION_THRESHOLD:
                distress = min(1.0, distress + 0.15)

        # Tool calls affecting intent
        if action_type == "tool" and sme_action.tool_name == "QUERY_TREDS":
            treds_intent = min(1.0, treds_intent + 0.30)

        # Explicit regulator invocation
        if action_type == "invoke_regulator":
            reg_intent = 0.9

        # Distress disclosure
        if action_type == "signal_distress":
            level = str(sme_action.distress_disclosure_level or "low").lower()
            increment = {"low": 0.15, "medium": 0.30, "high": 0.50}.get(level, 0.15)
            distress = min(1.0, distress + increment)

        # Stall counter: SME proposes without changing days vs buyer
        if action_type == "propose":
            # If SME proposal days didn't change from last round, count as stall
            stall = stall + 1 if (sme_action.payment_days or 0) == 0 else 0

        return BuyerBeliefState(
            buyer_id=buyer_belief.buyer_id,
            sme_distress_estimate=round(distress, 4),
            sme_treds_intent=round(treds_intent, 4),
            sme_regulator_intent=round(reg_intent, 4),
            rounds_without_concession=int(stall),
        )
