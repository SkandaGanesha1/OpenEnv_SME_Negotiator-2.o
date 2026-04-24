"""Stage 6 self-play and role-aware formatting tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.opponents import FinancierPolicy, FinancierQuote, TextPolicy
from server.environment import SMELiquidityEnvironment, SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation
from sme_negotiator_env.prompting import format_observation_text


class _FixedBuyerPolicy(TextPolicy):
    def act(self, observation: NegotiationObservation) -> NegotiationAction:
        return NegotiationAction(
            action_type="propose",
            price=92.0,
            payment_days=45,
            use_treds=False,
            reason="Fixed buyer policy",
        )


class _FixedFinancierPolicy(FinancierPolicy):
    def act(self, observation: NegotiationObservation) -> FinancierQuote:
        return FinancierQuote(
            approved=True,
            annual_rate=0.456,
            approved_amount=10_000_000.0,
            reason="Fixed financier approval",
        )


def test_default_buyer_hook_preserves_seeded_legacy_behavior() -> None:
    action = NegotiationAction(
        action_type="propose",
        price=95.0,
        payment_days=50,
        use_treds=False,
        reason="Counter offer",
    )
    env_default = SMENegotiatorEnvironment()
    env_explicit_none = SMENegotiatorEnvironment(buyer_policy=None)
    env_default.reset(seed=1200, task_name="payment-terms-medium")
    env_explicit_none.reset(seed=1200, task_name="payment-terms-medium")

    obs_default = env_default.step(action)
    obs_none = env_explicit_none.step(action)

    assert obs_default.model_dump() == obs_none.model_dump()


def test_custom_buyer_policy_changes_counter_offer_deterministically() -> None:
    env = SMENegotiatorEnvironment(buyer_policy=_FixedBuyerPolicy())
    env.reset(seed=1201, task_name="payment-terms-medium")

    observation = env.step(
        NegotiationAction(
            action_type="propose",
            price=96.0,
            payment_days=55,
            use_treds=False,
            reason="Try custom buyer",
        )
    )

    assert observation.buyer_price == pytest.approx(92.0)
    assert observation.buyer_days == 45
    assert observation.done is False


def test_custom_financier_policy_changes_quote_deterministically() -> None:
    env = SMELiquidityEnvironment(total_periods=1, financier_policy=_FixedFinancierPolicy())
    observation = env.reset(seed=1202, difficulty="hard", task_name="liquidity-correlation-hard")
    deal_id = observation.open_deal_ids[0]
    assert env.state is not None
    negotiation = env.state.current_negotiations[deal_id]

    env.step(
        NegotiationAction(
            action_type="accept",
            deal_id=deal_id,
            price=float(negotiation.buyer_price),
            payment_days=int(negotiation.buyer_days),
            use_treds=True,
            reason="Accept with financier policy",
        )
    )

    assert env.state is not None
    deal = next(item for item in env.state.world_state.deals if item.deal_id == deal_id)
    assert deal.status == "agreed"
    assert deal.financed is True
    assert deal.finance_rate == pytest.approx(0.456)
    assert any(event.event_type == "financier_quote" and event.deal_id == deal_id for event in env.state.history_tail)


def test_liquidity_env_passes_configured_buyer_policy_into_spawned_deal_envs() -> None:
    buyer_policy = _FixedBuyerPolicy()
    env = SMELiquidityEnvironment(total_periods=2, buyer_policy=buyer_policy)
    env.reset(seed=1203, difficulty="hard", task_name="liquidity-correlation-hard")

    assert env._deal_envs
    assert all(inner_env._buyer_policy is buyer_policy for inner_env in env._deal_envs.values())


def test_role_aware_prompting_preserves_default_sme_rendering() -> None:
    env = SMENegotiatorEnvironment()
    observation = env.reset(seed=1204, task_name="payment-terms-medium")

    default_text = format_observation_text(observation)
    sme_text = format_observation_text(observation, role="sme")
    buyer_text = format_observation_text(observation, role="buyer")
    financier_text = format_observation_text(observation, role="financier")

    assert default_text == sme_text
    assert "Role=SME" in default_text
    assert "Role=BUYER" in buyer_text
    assert "Role=FINANCIER" in financier_text
    assert buyer_text != financier_text
