"""Competitive multi-financier rate auction for the multi-agent SME environment.

FinancierCompetitionArena models four distinct financier types competing for
invoice discounting business. The SME triggers the auction via the
request_financier_auction action and receives up to 3 sorted bids in the
next observation — giving it genuine price discovery rather than a single
static quote.

Financier types and their characteristics:
- treds_platform: lowest rate, shortest tenor, requires invoice upload
- bank:           moderate rate, up to 90-day tenor, requires strong credit score
- nbfc:           higher rate, flexible conditions, highest advance amount
- microfinance:   highest rate, accessible when SME distress is high

All bid generation is deterministic from (deal_id, financier_type, seed).
"""

from __future__ import annotations

import hashlib
from random import Random
from typing import Any, Literal, Optional

from sme_negotiator_env.models import FinancierBid, FinancierState, WorldState
from sme_negotiator_env.tools import query_treds


# ======================================================================= #
# Rate spread parameters per financier type                                  #
# ======================================================================= #

_DEFAULT_SPREADS: dict[str, float] = {
    "treds_platform": 0.95,
    "bank": 1.05,
    "nbfc": 1.15,
    "microfinance": 1.35,
}

_MAX_TENORS: dict[str, int] = {
    "treds_platform": 45,
    "bank": 90,
    "nbfc": 60,
    "microfinance": 30,
}

_CONDITIONS: dict[str, list[str]] = {
    "treds_platform": ["requires_invoice_upload", "buyer_must_be_registered"],
    "bank": ["minimum_credit_score_0.4", "collateral_required_above_1M"],
    "nbfc": ["penalty_clause_preferred", "no_collateral_required"],
    "microfinance": ["high_distress_tier", "short_tenor_only"],
}


def _bid_seed(deal_id: str, financier_type: str, episode_seed: int) -> int:
    """Deterministic integer seed from deal_id + financier_type + episode_seed."""
    raw = f"{deal_id}:{financier_type}:{episode_seed}"
    return int(hashlib.sha1(raw.encode()).hexdigest()[:8], 16)


# ======================================================================= #
# FinancierCompetitionArena                                                  #
# ======================================================================= #

class FinancierCompetitionArena:
    """Runs a competitive rate auction across multiple financier types.

    Parameters
    ----------
    financiers:
        List of FinancierState objects from the WorldState. Used to derive
        the base interest rate and available capital.
    rng:
        Seeded RNG; used only for approval_probability jitter (deterministic).
    base_rate_spreads:
        Override the default spread multipliers per financier type.
    """

    def __init__(
        self,
        financiers: list[FinancierState],
        rng: Random,
        *,
        base_rate_spreads: Optional[dict[str, float]] = None,
    ) -> None:
        self._financiers = list(financiers)
        self._rng = rng
        self._spreads = dict(_DEFAULT_SPREADS)
        if base_rate_spreads:
            self._spreads.update(base_rate_spreads)
        self._last_bids: list[FinancierBid] = []

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def run_auction(
        self,
        world_state: WorldState,
        deal_id: str,
        *,
        sme_distress_signal: float = 0.0,
        invoice_amount: float = 0.0,
        episode_seed: int = 0,
    ) -> list[FinancierBid]:
        """Generate competitive bids from all financier types, sorted by rate.

        Only financiers that pass eligibility checks are included. The SME's
        distress signal unlocks the microfinance tier (otherwise excluded).

        Returns bids sorted ascending by annual_rate (best rate first).
        """
        # Derive base quote from existing deterministic TReDS tool
        try:
            treds_result = query_treds(world_state, deal_id)
            options = treds_result.get("quote_options", [])
            if options:
                base_rate = float(
                    min(options, key=lambda o: o["annual_discount_rate"])["annual_discount_rate"]
                )
            else:
                base_rate = float(world_state.baseline_discount_rate or 0.15)
        except Exception:
            base_rate = float(world_state.baseline_discount_rate or 0.15)

        available_capital = sum(
            float(f.available_capital) for f in self._financiers
        )

        bids: list[FinancierBid] = []
        sme = world_state.smes[0] if world_state.smes else None
        credit_score = float(1.0 - (sme.risk_score if sme is not None else 0.5))

        for ftype, spread in self._spreads.items():
            bid = self._build_bid(
                financier_type=ftype,
                base_rate=base_rate,
                spread=spread,
                available_capital=available_capital,
                invoice_amount=float(invoice_amount),
                credit_score=credit_score,
                sme_distress=float(sme_distress_signal),
                deal_id=deal_id,
                episode_seed=int(episode_seed),
            )
            if bid is not None:
                bids.append(bid)

        bids.sort(key=lambda b: b.annual_rate)
        self._last_bids = bids
        return bids

    def best_bid(self, bids: Optional[list[FinancierBid]] = None) -> Optional[FinancierBid]:
        """Return the bid with the lowest annual rate (best for SME)."""
        pool = bids if bids is not None else self._last_bids
        return pool[0] if pool else None

    def apply_winning_bid(
        self,
        world_state: WorldState,
        deal_id: str,
        bid: FinancierBid,
    ) -> WorldState:
        """Apply the selected financing bid to the world state (returns a copy).

        Deducts the advance_amount from the first financier's available_capital
        and marks the deal as financed in the world state.
        """
        copied = world_state.model_copy(deep=True)
        if copied.financier is not None:
            new_capital = max(
                0.0,
                float(copied.financier.available_capital) - float(bid.advance_amount),
            )
            copied.financier.available_capital = round(new_capital, 2)
            copied.financier.base_interest_rate = round(float(bid.annual_rate), 6)

        for deal in copied.deals:
            if deal.deal_id == deal_id:
                deal.financed = True
                deal.finance_rate = round(float(bid.annual_rate), 6)
                break

        return copied

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _build_bid(
        self,
        financier_type: str,
        base_rate: float,
        spread: float,
        available_capital: float,
        invoice_amount: float,
        credit_score: float,
        sme_distress: float,
        deal_id: str,
        episode_seed: int,
    ) -> Optional[FinancierBid]:
        """Build one bid, returning None if the financier is ineligible."""

        # Microfinance requires distress signal
        if financier_type == "microfinance" and sme_distress < 0.3:
            return None

        # Bank requires minimum credit score
        if financier_type == "bank" and credit_score < 0.4:
            return None

        raw_rate = base_rate * spread
        # Add small deterministic jitter from seed
        seed_val = _bid_seed(deal_id, financier_type, episode_seed)
        jitter = ((seed_val % 100) - 50) * 0.0001  # ±0.5%
        annual_rate = round(max(0.05, min(0.60, raw_rate + jitter)), 6)

        max_tenor = _MAX_TENORS.get(financier_type, 30)
        advance = round(min(available_capital, invoice_amount * 0.95), 2) if invoice_amount > 0 else 0.0

        # Approval probability from seeded RNG (deterministic)
        seed_rng = Random(seed_val)
        base_approval = 0.9 if financier_type == "treds_platform" else 0.75
        approval = round(min(1.0, base_approval + seed_rng.uniform(-0.1, 0.1)), 4)

        return FinancierBid(
            financier_id=f"{financier_type}_0",
            financier_type=financier_type,  # type: ignore[arg-type]
            annual_rate=annual_rate,
            max_tenor_days=max_tenor,
            advance_amount=advance,
            approval_probability=approval,
            conditions=list(_CONDITIONS.get(financier_type, [])),
        )
