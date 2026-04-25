"""Tests for inference policy guardrails and hard-task shortcut defaults."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference


def _hard_history() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                '{"action_type":"propose","price":89.0,"payment_days":30,'
                '"use_treds":true,"propose_dynamic_discounting":true,'
                '"dynamic_discount_annual_rate":0.02}'
            ),
        }
    ]


def test_hard_two_step_default_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_HARD_TWO_STEP", raising=False)
    assert inference._hard_two_step_policy_enabled() is False


def test_hard_two_step_can_be_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_HARD_TWO_STEP", "1")
    assert inference._hard_two_step_policy_enabled() is True


def test_no_forced_accept_when_default_disabled(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_HARD_TWO_STEP", raising=False)
    action = {"action_type": "propose", "price": 88.0, "payment_days": 30}

    out = inference._coerce_hard_accept_after_propose(
        action=action,
        history=_hard_history(),
        task_name="hard",
        round_number=1,
    )

    assert out["action_type"] == "propose"


def test_forced_accept_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_HARD_TWO_STEP", "1")
    action = {"action_type": "propose", "price": 88.0, "payment_days": 30}

    out = inference._coerce_hard_accept_after_propose(
        action=action,
        history=_hard_history(),
        task_name="hard",
        round_number=1,
    )

    assert out["action_type"] == "accept"
    assert out["propose_dynamic_discounting"] is True


def test_close_trigger_in_agreement_zone() -> None:
    observation = {
        "buyer_days": 44,
        "buyer_price": 83.0,
        "liquidity_threshold": 45,
        "cost_threshold": 80.0,
        "max_rounds": 12,
    }
    last_proposal = {
        "action_type": "propose",
        "price": 82.0,
        "payment_days": 45,
        "use_treds": False,
    }

    assert inference._should_close_deal(observation, "medium", round_number=6, last_valid_proposal=last_proposal)


def test_hard_stage1_enforces_dynamic_discounting_contract() -> None:
    observation = {
        "buyer_price": 95.0,
        "buyer_days": 95,
        "liquidity_threshold": 55,
        "cost_threshold": 78.0,
    }
    out = inference._normalize_stage1_proposal(
        {
            "action_type": "propose",
            "price": 88.0,
            "payment_days": 40,
            "use_treds": False,
            "propose_dynamic_discounting": False,
            "dynamic_discount_annual_rate": 0.5,
        },
        observation,
        "hard",
        round_number=0,
        last_valid_proposal=None,
    )

    assert out["action_type"] == "propose"
    assert out["propose_dynamic_discounting"] is True
    assert out["dynamic_discount_annual_rate"] == 0.02
    assert out["use_treds"] is True


def test_end_line_includes_score_field() -> None:
    line = inference._format_end_line(True, 4, 0.75, [0.1, 0.2, 0.15, 0.75])
    assert line.startswith("[END] success=true steps=4 score=0.75 rewards=")
    assert "0.10,0.20,0.15,0.75" in line


def test_liquidity_normalizer_suppresses_rejects_into_counter_proposals() -> None:
    observation = {
        "active_deal_id": "deal-1",
        "open_deal_ids": ["deal-1"],
        "buyer_price": 96.0,
        "buyer_days": 75,
        "liquidity_threshold": 45,
        "cost_threshold": 82.0,
        "metadata": {},
    }

    out = inference._normalize_liquidity_action_payload(
        {"action_type": "reject", "reason": "Counter offer does not meet thresholds."},
        observation,
        history=[],
        task_name="liquidity-stress-medium",
        round_number=1,
        last_valid_proposal=None,
    )

    assert out["action_type"] == "propose"
    assert out["deal_id"] == "deal-1"
    assert "Reject suppressed" in out["reason"]


def test_liquidity_normalizer_downgrades_invalid_accepts() -> None:
    observation = {
        "active_deal_id": "deal-1",
        "open_deal_ids": ["deal-1"],
        "buyer_price": 96.0,
        "buyer_days": 78,
        "liquidity_threshold": 45,
        "cost_threshold": 82.0,
        "metadata": {},
    }

    out = inference._normalize_liquidity_action_payload(
        {
            "action_type": "accept",
            "deal_id": "deal-1",
            "price": 96.0,
            "payment_days": 78,
            "use_treds": False,
            "reason": "Looks acceptable.",
        },
        observation,
        history=[],
        task_name="liquidity-stress-medium",
        round_number=2,
        last_valid_proposal=None,
    )

    assert out["action_type"] == "propose"
    assert out["deal_id"] == "deal-1"
    assert "Invalid accept downgraded" in out["reason"]


def test_medium_contract_strips_dynamic_discount_fields() -> None:
    observation = {
        "active_deal_id": "deal-1",
        "open_deal_ids": ["deal-1"],
        "buyer_price": 94.0,
        "buyer_days": 72,
        "liquidity_threshold": 45,
        "cost_threshold": 82.0,
        "metadata": {},
    }

    out = inference._normalize_liquidity_action_payload(
        {
            "action_type": "propose",
            "deal_id": "deal-1",
            "price": 90.0,
            "payment_days": 50,
            "use_treds": False,
            "propose_dynamic_discounting": True,
            "dynamic_discount_annual_rate": 0.27,
        },
        observation,
        history=[],
        task_name="liquidity-stress-medium",
        round_number=2,
        last_valid_proposal=None,
    )

    assert out["propose_dynamic_discounting"] is False
    assert out["dynamic_discount_annual_rate"] == 0.0
    assert out["propose_late_payment_penalty_clause"] is True


def test_compact_step_serialization_omits_irrelevant_tool_fields() -> None:
    action = inference.NegotiationAction(
        action_type="tool",
        deal_id="deal-1",
        tool_name="QUERY_TREDS",
        tool_args={"invoice_id": "deal-1", "deal_id": "deal-1"},
        price=83.5,
        payment_days=60,
        use_treds=True,
        propose_dynamic_discounting=True,
        dynamic_discount_annual_rate=0.02,
        reason="Inspect financing first.",
    )

    action_json = inference._serialize_step_action(action)

    assert '"tool_name":"QUERY_TREDS"' in action_json
    assert '"price"' not in action_json
    assert '"payment_days"' not in action_json
    assert '"dynamic_discount_annual_rate"' not in action_json
