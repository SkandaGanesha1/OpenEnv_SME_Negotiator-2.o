"""Tests for Theme #1 — Multi-Agent Interactions.

Covers: ToM belief updates, coalition formation/defection, regulator invocation,
financier auction ordering, social welfare computation, and backward compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path
from random import Random
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sme_negotiator_env.agents.buyer_agent import StrategicBuyerAgent
from sme_negotiator_env.agents.coalition import BuyerCoalition, CoalitionFormationPolicy
from sme_negotiator_env.agents.financier_agent import FinancierCompetitionArena
from sme_negotiator_env.agents.regulator_agent import RegulatorAgent
from sme_negotiator_env.belief_state import (
    ADVERSARIAL,
    COOPERATIVE,
    NEUTRAL,
    BeliefStateManager,
    assign_latent_type,
)
from sme_negotiator_env.models import (
    BuyerState,
    CoalitionStatus,
    DealState,
    FinancierBid,
    FinancierState,
    MultiAgentObservation,
    NegotiationAction,
    NegotiationObservation,
    NegotiationState,
    RegulatoryWarning,
    SMEAccountState,
    WorldState,
)


# ======================================================================= #
# Helpers                                                                   #
# ======================================================================= #

def _make_observation(
    *,
    buyer_days: int = 90,
    buyer_price: float = 85.0,
    round_number: int = 1,
    max_rounds: int = 10,
    liquidity_threshold: int = 30,
    treds_bonus: float = 0.0,
) -> NegotiationObservation:
    return NegotiationObservation(
        round_number=round_number,
        max_rounds=max_rounds,
        buyer_price=buyer_price,
        buyer_days=buyer_days,
        buyer_accepted=False,
        negotiation_done=False,
        cost_threshold=70.0,
        liquidity_threshold=liquidity_threshold,
        volume=100,
        difficulty="medium",
        price_score=0.5,
        days_score=0.5,
        treds_bonus=treds_bonus,
        step_reward=0.0,
        message="",
        interest_rate_annual=0.15,
    )


def _make_world_state(
    *,
    invoice_amount: float = 500_000.0,
    available_capital: float = 1_000_000.0,
) -> WorldState:
    sme = SMEAccountState(
        sme_id="sme_0",
        cash_balance=200_000.0,
        supplier_payment_days=30,
        credit_limit=1_000_000.0,
        current_utilization=0.2,
        risk_score=0.3,
    )
    buyer = BuyerState(
        buyer_id="buyer_0",
        demand_level=0.7,
        budget_per_period=500_000.0,
        default_tendency=0.1,
        baseline_payment_days=90,
    )
    fin = FinancierState(
        financier_id="fin_0",
        available_capital=available_capital,
        risk_appetite=0.7,
        base_interest_rate=0.15,
    )
    deal = DealState(
        deal_id="deal_0",
        sme_id="sme_0",
        buyer_id="buyer_0",
        invoice_amount=invoice_amount,
        created_period=0,
    )
    return WorldState(
        smes=[sme],
        buyers=[buyer],
        financier=fin,
        baseline_discount_rate=0.15,
        deals=[deal],
    )


def _make_adversarial_buyer(seed: int = 1) -> StrategicBuyerAgent:
    return StrategicBuyerAgent(
        buyer_id="buyer_0",
        latent_type=ADVERSARIAL,
        rng=Random(seed),
    )


def _make_neutral_buyer(seed: int = 2) -> StrategicBuyerAgent:
    return StrategicBuyerAgent(
        buyer_id="buyer_1",
        latent_type=NEUTRAL,
        rng=Random(seed),
    )


# ======================================================================= #
# 1. Belief updates                                                         #
# ======================================================================= #

def test_buyer_belief_updates_on_rapid_price_concession() -> None:
    """SME dropping price > 3% in one step raises buyer's distress estimate."""
    manager = BeliefStateManager()
    belief = manager.initial_buyer_belief("buyer_0")
    initial_distress = belief.sme_distress_estimate

    action = NegotiationAction(
        action_type="propose",
        price=85.0,
        payment_days=45,
        reason="SME propose",
    )
    updated = manager.update_buyer_belief_on_sme_action(
        belief,
        action,
        prev_sme_price=90.0,
        current_sme_price=85.0,
    )

    assert updated.sme_distress_estimate > initial_distress, (
        "Rapid SME price concession should raise buyer's distress estimate"
    )


def test_sme_belief_updates_on_repeated_buyer_stall() -> None:
    """SME classifies buyer as adversarial after 3+ rounds of minimal concession."""
    manager = BeliefStateManager()
    sme_belief = manager.initial_sme_belief(
        ["buyer_0"],
        initial_financier_rate=0.15,
    )

    assert sme_belief.buyer_type_estimates["buyer_0"] == "neutral"

    tiny_concession = NegotiationAction(
        action_type="propose",
        price=85.0,
        payment_days=89,  # only 1-day concession from 90 (< 2 threshold)
        reason="stalling",
    )
    # round_number >= 3 triggers the adversarial classification rule
    sme_belief = manager.update_sme_belief_on_buyer_action(
        sme_belief, "buyer_0", tiny_concession, round_number=3,
        prev_buyer_days=90,
    )

    assert sme_belief.buyer_type_estimates["buyer_0"] == "adversarial", (
        "3+ rounds of minimal concession should classify buyer as adversarial"
    )


def test_invoke_regulator_action_sets_regulator_intent_high() -> None:
    """Buyer sees SME invoke_regulator → sme_regulator_intent jumps to 0.9."""
    manager = BeliefStateManager()
    belief = manager.initial_buyer_belief("buyer_0")

    action = NegotiationAction(
        action_type="invoke_regulator",
        price=90.0,
        payment_days=45,
        reason="regulatory threat",
    )
    updated = manager.update_buyer_belief_on_sme_action(belief, action)
    assert updated.sme_regulator_intent >= 0.9


# ======================================================================= #
# 2. Coalition formation                                                    #
# ======================================================================= #

def test_coalition_forms_after_joint_stall() -> None:
    """Coalition forms after form_threshold_rounds consecutive joint stall rounds."""
    buyer_a = _make_adversarial_buyer(seed=10)
    buyer_b = StrategicBuyerAgent(
        buyer_id="buyer_1",
        latent_type=ADVERSARIAL,  # coalition_willingness=0.9 > 0.4
        rng=Random(11),
    )
    coalition = BuyerCoalition(
        buyer_a,
        buyer_b,
        CoalitionFormationPolicy(form_threshold_rounds=3),
    )

    action_a = NegotiationAction(action_type="propose", price=85.0, payment_days=88, reason="stall")
    action_b = NegotiationAction(action_type="propose", price=85.0, payment_days=89, reason="stall")

    formed = False
    for round_num in range(1, 6):
        coalition.record_buyer_actions(action_a, action_b)
        formed = coalition.attempt_formation(round_number=round_num, sme_last_concession_days=1)
        if formed:
            break

    assert formed, "Coalition should form after 3 joint stall rounds"
    assert coalition.is_active
    assert coalition.formed_at_round is not None
    assert coalition.joint_demand_days == max(88, 89)


def test_coalition_does_not_form_if_sme_conceding() -> None:
    """If SME concedes sufficiently (≥ 2 days), stall counter resets — no coalition."""
    buyer_a = _make_adversarial_buyer(seed=10)
    buyer_b = StrategicBuyerAgent(buyer_id="buyer_1", latent_type=ADVERSARIAL, rng=Random(11))
    coalition = BuyerCoalition(buyer_a, buyer_b)

    action_a = NegotiationAction(action_type="propose", price=85.0, payment_days=88, reason="stall")
    action_b = NegotiationAction(action_type="propose", price=85.0, payment_days=89, reason="stall")

    for round_num in range(1, 6):
        coalition.record_buyer_actions(action_a, action_b)
        # SME concedes ≥ 2 days — stall counter resets each round
        coalition.attempt_formation(round_number=round_num, sme_last_concession_days=3)

    assert not coalition.is_active, "Coalition should not form when SME is actively conceding"


# ======================================================================= #
# 3. Split deal causes defection                                            #
# ======================================================================= #

def test_split_deal_causes_coalition_defection() -> None:
    """propose_split_deal with preferential terms triggers rational defection."""
    buyer_a = _make_adversarial_buyer(seed=20)
    buyer_b = StrategicBuyerAgent(buyer_id="buyer_1", latent_type=ADVERSARIAL, rng=Random(21))
    coalition = BuyerCoalition(buyer_a, buyer_b, CoalitionFormationPolicy(form_threshold_rounds=3))

    stall_a = NegotiationAction(action_type="propose", price=85.0, payment_days=88, reason="stall")
    stall_b = NegotiationAction(action_type="propose", price=85.0, payment_days=88, reason="stall")
    for rnd in range(1, 6):
        coalition.record_buyer_actions(stall_a, stall_b)
        if coalition.attempt_formation(round_number=rnd, sme_last_concession_days=0):
            break

    assert coalition.is_active, "Precondition: coalition must be active"

    # Split offer: buyer_a gets 100 days > coalition_expected(88) * 1.1 = 96.8 → defect
    split_action = NegotiationAction(
        action_type="propose_split_deal",
        price=85.0,
        payment_days=45,
        split_deal_buyer_a_days=100,
        split_deal_buyer_b_days=60,
        split_deal_buyer_a_price=85.0,
        split_deal_buyer_b_price=85.0,
        reason="split deal",
    )
    defected, defector_id = coalition.process_split_offer(split_action)

    assert defected, "Buyer_a should defect when split offer >> coalition expected payoff"
    assert defector_id == "buyer_0"
    assert not coalition.is_active, "Coalition dissolves after defection"


def test_split_deal_no_defection_when_terms_not_better() -> None:
    """If split offer is not meaningfully better than coalition payoff, defection does NOT occur."""
    buyer_a = _make_adversarial_buyer(seed=30)
    buyer_b = StrategicBuyerAgent(buyer_id="buyer_1", latent_type=ADVERSARIAL, rng=Random(31))
    coalition = BuyerCoalition(buyer_a, buyer_b, CoalitionFormationPolicy(form_threshold_rounds=3))

    stall = NegotiationAction(action_type="propose", price=85.0, payment_days=88, reason="stall")
    for rnd in range(1, 6):
        coalition.record_buyer_actions(stall, stall)
        if coalition.attempt_formation(round_number=rnd, sme_last_concession_days=0):
            break

    assert coalition.is_active

    # Split offer: 90 days < 88 * 1.1 = 96.8 → no defection
    split_action = NegotiationAction(
        action_type="propose_split_deal",
        price=85.0,
        payment_days=45,
        split_deal_buyer_a_days=90,
        split_deal_buyer_b_days=90,
        split_deal_buyer_a_price=85.0,
        split_deal_buyer_b_price=85.0,
        reason="split deal",
    )
    defected, _ = coalition.process_split_offer(split_action)

    assert not defected, "No defection when split offer is not > coalition_expected * 1.1"
    assert coalition.is_active


# ======================================================================= #
# 4. Regulator invocation shifts adversarial buyer to neutral              #
# ======================================================================= #

def test_regulator_invocation_shifts_adversarial_buyer_to_neutral() -> None:
    """invoke_regulator with material penalty shifts adversarial buyer → neutral effective type."""
    buyer = _make_adversarial_buyer(seed=42)
    regulator = RegulatorAgent(legal_max_payment_days=45)
    world_state = _make_world_state(invoice_amount=200_000.0)

    assert buyer._effective_latent_type().name == "adversarial"

    current_proposal = NegotiationAction(
        action_type="propose", price=85.0, payment_days=60, reason="sme proposal"
    )
    warning = regulator.invoke(
        deal_id="deal_0",
        buyer_id="buyer_0",
        current_proposal=current_proposal,
        world_state=world_state,
        current_period=1,
    )
    assert warning.issued_by_sme is True
    assert warning.penalty_exposure_inr > 0.0

    regulator.modify_buyer_equilibrium(buyer, warning)

    effective = buyer._effective_latent_type()
    assert effective.name == "neutral", (
        f"Adversarial buyer should revert to neutral after regulatory warning; got {effective.name}"
    )


def test_regulator_computes_penalty_correctly() -> None:
    """Section 43B(h): full invoice amount at risk when payment exceeds 45 days."""
    regulator = RegulatorAgent(
        legal_max_payment_days=45,
        compound_interest_multiplier=3.0,
        rbi_bank_rate=0.065,
    )
    # 60 days → 15 excess; invoice = 100_000
    # Section 43B(h): full invoice (100_000) is at risk as a deduction
    penalty = regulator.compute_buyer_tax_exposure(60, 100_000.0)
    assert penalty == 100_000.0


def test_regulator_no_penalty_within_legal_days() -> None:
    """Penalty is zero when payment_days ≤ legal_max (45 days)."""
    regulator = RegulatorAgent(legal_max_payment_days=45)
    penalty = regulator.compute_buyer_tax_exposure(45, 100_000.0)
    assert penalty == 0.0


# ======================================================================= #
# 5. Financier auction returns sorted bids                                  #
# ======================================================================= #

def test_financier_auction_returns_sorted_bids() -> None:
    """Financier auction returns bids sorted by annual_rate ascending (best first)."""
    world_state = _make_world_state(invoice_amount=500_000.0, available_capital=2_000_000.0)
    fin = world_state.financier
    arena = FinancierCompetitionArena(financiers=[fin], rng=Random(42))

    bids = arena.run_auction(
        world_state,
        deal_id="deal_0",
        sme_distress_signal=0.5,
        invoice_amount=500_000.0,
        episode_seed=42,
    )

    assert len(bids) >= 2, "Should return at least 2 bids"
    rates = [b.annual_rate for b in bids]
    assert rates == sorted(rates), "Bids must be sorted ascending by annual_rate"
    assert all(0.05 <= b.annual_rate <= 0.60 for b in bids), "All rates in clamped range"


def test_financier_auction_microfinance_requires_distress() -> None:
    """Microfinance financier only appears when sme_distress_signal >= 0.3."""
    world_state = _make_world_state()
    fin = world_state.financier
    arena = FinancierCompetitionArena(financiers=[fin], rng=Random(7))

    bids_low = arena.run_auction(
        world_state, "deal_0", sme_distress_signal=0.0, invoice_amount=500_000.0, episode_seed=7
    )
    bids_high = arena.run_auction(
        world_state, "deal_0", sme_distress_signal=0.5, invoice_amount=500_000.0, episode_seed=7
    )

    low_types = {b.financier_type for b in bids_low}
    high_types = {b.financier_type for b in bids_high}
    assert "microfinance" not in low_types, "Microfinance excluded at low distress"
    assert "microfinance" in high_types, "Microfinance appears at high distress"


def test_financier_auction_is_deterministic() -> None:
    """Same seed + same world state produces identical bids."""
    world_state = _make_world_state()
    fin = world_state.financier

    arena1 = FinancierCompetitionArena(financiers=[fin], rng=Random(99))
    arena2 = FinancierCompetitionArena(financiers=[fin], rng=Random(99))

    bids1 = arena1.run_auction(world_state, "deal_0", sme_distress_signal=0.4, episode_seed=99)
    bids2 = arena2.run_auction(world_state, "deal_0", sme_distress_signal=0.4, episode_seed=99)

    assert [b.annual_rate for b in bids1] == [b.annual_rate for b in bids2]


# ======================================================================= #
# 6. Social welfare                                                         #
# ======================================================================= #

def test_social_welfare_improves_with_cooperative_resolution() -> None:
    """Social welfare is higher when regulator has no active warnings vs one active warning."""
    from server.multi_agent_environment import MultiAgentNegotiationEnvironment

    env = MultiAgentNegotiationEnvironment()
    env.reset(task_name="payment-terms-medium", difficulty="medium", seed=42)

    # Clean state: no active warnings
    _, _, sw_clean = env._compute_social_welfare(
        sme_reward=0.8,
        agreed_days=45,
        baseline_buyer_days=90,
        total_invoice=200_000.0,
    )

    # Add a warning so active_warnings becomes non-empty
    warning = RegulatoryWarning(
        deal_id="deal_0",
        buyer_id="buyer_0",
        violation_type="exceeds_45_days",
        penalty_exposure_inr=100_000.0,
        section_reference="MSMED Act",
        is_active=True,
        issued_period=1,
        issued_by_sme=False,
    )
    env._regulator._active_warnings.append(warning)

    _, _, sw_with_warning = env._compute_social_welfare(
        sme_reward=0.8,
        agreed_days=45,
        baseline_buyer_days=90,
        total_invoice=200_000.0,
    )

    assert sw_clean > sw_with_warning, (
        "Social welfare should be higher when no active regulatory warnings exist"
    )


# ======================================================================= #
# 7. Backward compatibility — existing action types unchanged              #
# ======================================================================= #

def test_backward_compat_existing_action_types_unchanged() -> None:
    """propose / accept action types on SMELiquidityEnvironment work identically as before."""
    from server.environment import SMELiquidityEnvironment

    env = SMELiquidityEnvironment()
    env.reset(task_name="payment-terms-easy", difficulty="easy", seed=100)

    propose = NegotiationAction(
        action_type="propose", price=95.0, payment_days=45, reason="propose"
    )
    obs = env.step(propose)
    assert obs is not None
    assert hasattr(obs, "round_number")


def test_new_action_fields_have_none_defaults() -> None:
    """All new NegotiationAction fields default to None — no breakage for existing callers."""
    action = NegotiationAction(
        action_type="propose", price=90.0, payment_days=45, reason="basic"
    )
    assert action.split_deal_buyer_a_days is None
    assert action.split_deal_buyer_b_days is None
    assert action.split_deal_buyer_a_price is None
    assert action.split_deal_buyer_b_price is None
    assert action.distress_disclosure_level is None


# ======================================================================= #
# 8. MultiAgentObservation extends LiquidityObservation contract           #
# ======================================================================= #

def test_multi_agent_obs_extends_liquidity_obs_contract() -> None:
    """MultiAgentObservation is a superset — all existing LiquidityObservation fields present."""
    from sme_negotiator_env.models import LiquidityObservation

    obs = MultiAgentObservation(
        round_number=1,
        max_rounds=10,
        buyer_price=85.0,
        buyer_days=90,
        buyer_accepted=False,
        negotiation_done=False,
        cost_threshold=70.0,
        liquidity_threshold=30,
        volume=100,
        difficulty="medium",
        price_score=0.5,
        days_score=0.5,
        treds_bonus=0.0,
        step_reward=0.0,
        message="",
        interest_rate_annual=0.15,
    )

    assert isinstance(obs, LiquidityObservation), (
        "MultiAgentObservation must be a subclass of LiquidityObservation"
    )

    # New multi-agent fields have safe defaults
    assert obs.opponent_signals == []
    assert obs.coalition_status is None
    assert obs.regulatory_warnings == []
    assert obs.financier_bids == []
    assert obs.sme_belief_estimate == {}
    assert obs.social_welfare_score == 0.0
    assert obs.buyer_surplus_estimate == 0.0

    # Base fields still accessible
    assert obs.buyer_days == 90
    assert obs.round_number == 1


def test_multi_agent_obs_coalition_status_roundtrip() -> None:
    """CoalitionStatus can be embedded in MultiAgentObservation and serialised."""
    status = CoalitionStatus(
        is_active=True,
        buyer_ids=["buyer_0", "buyer_1"],
        formed_at_round=3,
        joint_demand_days=88,
        defection_risk=0.1,
    )
    obs = MultiAgentObservation(
        round_number=3,
        max_rounds=10,
        buyer_price=85.0,
        buyer_days=88,
        buyer_accepted=False,
        negotiation_done=False,
        cost_threshold=70.0,
        liquidity_threshold=30,
        volume=100,
        difficulty="hard",
        price_score=0.5,
        days_score=0.5,
        treds_bonus=0.0,
        step_reward=0.0,
        message="",
        coalition_status=status,
    )
    serialised = obs.model_dump()
    assert serialised["coalition_status"]["is_active"] is True
    assert serialised["coalition_status"]["buyer_ids"] == ["buyer_0", "buyer_1"]
    assert serialised["coalition_status"]["joint_demand_days"] == 88


# ======================================================================= #
# 9. Latent type assignment determinism                                     #
# ======================================================================= #

def test_assign_latent_type_hard_difficulty_always_adversarial_neutral() -> None:
    """Hard difficulty: buyer_0 always adversarial, buyer_1 always neutral."""
    for seed in [1, 42, 999, 12345]:
        assert assign_latent_type(buyer_index=0, difficulty="hard", seed=seed).name == "adversarial"
        assert assign_latent_type(buyer_index=1, difficulty="hard", seed=seed).name == "neutral"


def test_assign_latent_type_easy_mostly_cooperative() -> None:
    """Easy difficulty: majority of seeds return cooperative (p=0.8 expected)."""
    results = [
        assign_latent_type(buyer_index=0, difficulty="easy", seed=s).name
        for s in range(100)
    ]
    cooperative_count = results.count("cooperative")
    assert cooperative_count >= 60, (
        f"Only {cooperative_count}/100 were cooperative on easy difficulty"
    )
