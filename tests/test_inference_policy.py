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
