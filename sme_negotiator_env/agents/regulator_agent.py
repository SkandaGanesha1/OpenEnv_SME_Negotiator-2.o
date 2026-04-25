"""Regulator agent modeling the MSME Samadhaan / ODR portal and Section 43B(h).

The RegulatorAgent monitors all active deals for MSMED Act violations and can
be explicitly invoked by the SME as a strategic threat. When a warning is
issued, the affected buyer's equilibrium strategy shifts toward concession
(adversarial → neutral) proportional to its regulatory_fear score.

Legal basis:
- MSMED Act 2006, Sections 15–24: max 45-day payment window; compound
  interest at 3× RBI bank rate on overdue amounts.
- Income Tax Act, Section 43B(h): buyers lose tax deduction for expenses
  paid to MSMEs beyond 45 days — direct P&L impact.

This file is intentionally self-contained (no imports from multi_agent_environment)
to avoid circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from sme_negotiator_env.models import NegotiationAction, NegotiationState, RegulatoryWarning, WorldState

if TYPE_CHECKING:
    from sme_negotiator_env.agents.buyer_agent import StrategicBuyerAgent


# ======================================================================= #
# RegulatorAgent                                                            #
# ======================================================================= #

class RegulatorAgent:
    """Models MSME Samadhaan / ODR portal + Section 43B(h) enforcement.

    Parameters
    ----------
    legal_max_payment_days:
        Statutory maximum (default 45, per MSMED Act 2006 Section 15).
    compound_interest_multiplier:
        Multiplier on RBI bank rate for late-payment compound interest penalty.
    rbi_bank_rate:
        Current RBI bank rate (used for penalty computation).
    auto_monitor:
        If True, monitor_deals() auto-scans for violations each step.
    """

    def __init__(
        self,
        legal_max_payment_days: int = 45,
        *,
        compound_interest_multiplier: float = 3.0,
        rbi_bank_rate: float = 0.065,
        auto_monitor: bool = True,
    ) -> None:
        self._legal_max = int(legal_max_payment_days)
        self._ci_multiplier = float(compound_interest_multiplier)
        self._rbi_rate = float(rbi_bank_rate)
        self._auto_monitor = bool(auto_monitor)
        self._active_warnings: list[RegulatoryWarning] = []

    # ------------------------------------------------------------------ #
    # Monitoring                                                            #
    # ------------------------------------------------------------------ #

    def monitor_deals(
        self,
        world_state: WorldState,
        current_negotiations: dict[str, NegotiationState],
        *,
        current_period: int = 0,
    ) -> list[RegulatoryWarning]:
        """Auto-scan all active deals for MSMED Act violations.

        Returns any NEW warnings generated this round (not previously known).
        """
        if not self._auto_monitor:
            return []

        new_warnings: list[RegulatoryWarning] = []

        for deal in world_state.deals:
            if deal.status not in ("open", "agreed"):
                continue

            buyer_id = deal.buyer_id
            deal_id = deal.deal_id

            # Determine effective payment days
            neg_state = current_negotiations.get(deal_id)
            if neg_state is not None and neg_state.agreed_terms is not None:
                effective_days = int(neg_state.agreed_terms)
            elif neg_state is not None:
                effective_days = int(neg_state.buyer_days)
            elif deal.agreed_payment_days is not None:
                effective_days = int(deal.agreed_payment_days)
            else:
                continue  # no days to check yet

            invoice_amount = float(deal.invoice_amount or 0.0)

            # Check: exceeds_45_days
            if effective_days > self._legal_max:
                has_penalty_clause = (
                    deal.late_payment_penalty_agreed
                    or (neg_state.late_payment_penalty_agreed if neg_state else False)
                )
                grace = effective_days <= self._legal_max + 7

                if not (grace and has_penalty_clause):
                    warning = self._make_warning(
                        deal_id=deal_id,
                        buyer_id=buyer_id,
                        violation_type="exceeds_45_days",
                        payment_days=effective_days,
                        invoice_amount=invoice_amount,
                        issued_period=current_period,
                        issued_by_sme=False,
                    )
                    if not self._is_duplicate(warning):
                        new_warnings.append(warning)
                        self._active_warnings.append(warning)

            # Check: tax deduction at risk (Section 43B(h))
            if effective_days > self._legal_max:
                tax_exposure = self.compute_buyer_tax_exposure(
                    effective_days, invoice_amount
                )
                if tax_exposure > 0:
                    w = RegulatoryWarning(
                        deal_id=deal_id,
                        buyer_id=buyer_id,
                        violation_type="tax_deduction_at_risk",
                        penalty_exposure_inr=round(tax_exposure, 2),
                        section_reference="Section 43B(h) Income Tax Act",
                        issued_period=current_period,
                        issued_by_sme=False,
                    )
                    if not self._is_duplicate(w):
                        new_warnings.append(w)
                        self._active_warnings.append(w)

        return new_warnings

    # ------------------------------------------------------------------ #
    # SME-invoked regulator threat                                          #
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        deal_id: str,
        buyer_id: str,
        current_proposal: NegotiationAction,
        world_state: WorldState,
        *,
        current_period: int = 0,
    ) -> RegulatoryWarning:
        """SME explicitly invokes the regulator — creates a high-credibility warning.

        The issued_by_sme=True flag increases the regulatory_fear boost applied
        to the buyer by the environment (because an explicit invocation is more
        credible than a passive monitoring alert).
        """
        proposed_days = int(current_proposal.payment_days or self._legal_max + 1)
        invoice_amount = 0.0
        for deal in world_state.deals:
            if deal.deal_id == deal_id:
                invoice_amount = float(deal.invoice_amount or 0.0)
                break

        excess = max(0, proposed_days - self._legal_max)
        penalty = self.compute_buyer_tax_exposure(proposed_days, invoice_amount)

        violation_type: Literal[
            "exceeds_45_days", "no_penalty_clause", "discount_rate_cap", "tax_deduction_at_risk"
        ]
        if excess > 0:
            violation_type = "exceeds_45_days"
            section_ref = "MSMED Act 2006 Sections 15-24 + Section 43B(h) ITA"
        else:
            violation_type = "no_penalty_clause"
            section_ref = "MSMED Act 2006 Section 16 (late-payment interest)"

        warning = RegulatoryWarning(
            deal_id=deal_id,
            buyer_id=buyer_id,
            violation_type=violation_type,
            penalty_exposure_inr=round(penalty, 2),
            section_reference=section_ref,
            is_active=True,
            issued_period=current_period,
            issued_by_sme=True,
        )
        self._active_warnings.append(warning)
        return warning

    # ------------------------------------------------------------------ #
    # Buyer equilibrium modification                                        #
    # ------------------------------------------------------------------ #

    def modify_buyer_equilibrium(
        self,
        buyer: "StrategicBuyerAgent",
        warning: RegulatoryWarning,
    ) -> "StrategicBuyerAgent":
        """Apply regulatory fear to the buyer agent based on the warning severity.

        Fear boost is larger when:
        - The warning is explicitly issued by the SME (more credible)
        - The buyer's existing regulatory_fear is high
        - The penalty exposure is material

        Returns the same buyer object (mutation applied in-place via public API).
        """
        base_fear_boost = 0.3
        if warning.issued_by_sme:
            base_fear_boost = 0.5

        # Scale by penalty materiality (cap at invoice > 100k INR = full boost)
        materiality = min(1.0, warning.penalty_exposure_inr / 100_000.0) if warning.penalty_exposure_inr > 0 else 0.5
        fear_boost = round(base_fear_boost * (0.5 + 0.5 * materiality), 4)

        buyer.apply_regulatory_warning(fear_boost=fear_boost)
        return buyer

    # ------------------------------------------------------------------ #
    # Penalty computations                                                  #
    # ------------------------------------------------------------------ #

    def compute_buyer_tax_exposure(
        self,
        payment_days: int,
        invoice_amount: float,
        *,
        days_outstanding: int = 0,
    ) -> float:
        """Deterministic Section 43B(h) deduction at risk.

        Returns the INR amount the buyer risks losing as a tax deduction if
        payment exceeds 45 days. This equals the invoice amount itself (the
        entire expense is disallowed) when beyond the legal limit.

        Also includes MSMED Act compound interest: invoice × 3 × RBI_rate × excess/365.
        """
        excess_days = max(0, int(payment_days) - self._legal_max)
        if excess_days <= 0 and days_outstanding <= 0:
            return 0.0

        # Section 43B(h): full invoice amount is at risk as a deduction
        tax_deduction_risk = float(invoice_amount)

        # MSMED Act compound interest component
        effective_excess = max(excess_days, int(days_outstanding))
        ci_penalty = (
            float(invoice_amount)
            * self._ci_multiplier
            * self._rbi_rate
            * effective_excess
            / 365.0
        )

        return round(max(tax_deduction_risk, ci_penalty), 2)

    def compute_msmed_penalty(
        self,
        invoice_amount: float,
        excess_days: int,
    ) -> float:
        """Compound interest penalty under MSMED Act Sections 15–24."""
        if excess_days <= 0 or invoice_amount <= 0:
            return 0.0
        return round(
            float(invoice_amount) * self._ci_multiplier * self._rbi_rate * excess_days / 365.0,
            2,
        )

    # ------------------------------------------------------------------ #
    # State management                                                      #
    # ------------------------------------------------------------------ #

    @property
    def active_warnings(self) -> list[RegulatoryWarning]:
        return [w for w in self._active_warnings if w.is_active]

    def reset(self) -> None:
        """Clear all warnings for a new episode."""
        self._active_warnings.clear()

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _make_warning(
        self,
        deal_id: str,
        buyer_id: str,
        violation_type: str,
        payment_days: int,
        invoice_amount: float,
        issued_period: int,
        issued_by_sme: bool,
    ) -> RegulatoryWarning:
        excess = max(0, payment_days - self._legal_max)
        penalty = self.compute_msmed_penalty(invoice_amount, excess)
        vtype: Literal[
            "exceeds_45_days", "no_penalty_clause", "discount_rate_cap", "tax_deduction_at_risk"
        ] = violation_type  # type: ignore[assignment]
        return RegulatoryWarning(
            deal_id=deal_id,
            buyer_id=buyer_id,
            violation_type=vtype,
            penalty_exposure_inr=round(penalty, 2),
            section_reference="MSMED Act 2006 Sections 15-24",
            issued_period=issued_period,
            issued_by_sme=issued_by_sme,
        )

    def _is_duplicate(self, warning: RegulatoryWarning) -> bool:
        for w in self._active_warnings:
            if w.deal_id == warning.deal_id and w.violation_type == warning.violation_type:
                return True
        return False
