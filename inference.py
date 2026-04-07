#!/usr/bin/env python3
"""Inference runner for the SME negotiation environment."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from openai import OpenAI
from openenv.core.client_types import StepResult

from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.llm_action_parser import parse_llm_text_to_negotiation_action
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation

from server.environment import SMENegotiatorEnvironment

logger = logging.getLogger(__name__)

# Allow .env to override pre-set shell vars (e.g. stale Groq API_BASE_URL in the same terminal).
load_dotenv(override=True)

# LLM: Hugging Face OpenAI-compatible router by default (override with API_BASE_URL in .env)
API_BASE_URL = (os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1").strip()
# Hackathon / dashboard may set either HF_TOKEN or API_KEY for the OpenAI client.
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip() or None
# HF Router chat/completions only accepts models exposed as *chat* models — not every Hub id works.
MODEL_NAME = (os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct").strip()
# Docker / HF Space image tag (validator builds). Inference uses HTTP or in-process env — not from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "openenv-sme-negotiator:latest")

# Printed once per process when HF Inference returns HTTP 402 (quota / billing — not fixable in Python).
_PRINTED_HF_402_HINT = False
_MAX_REASON_CHARS = 160
_MAX_ERROR_CHARS = 240


def _is_hf_inference_402(exc: BaseException) -> bool:
    return "402" in str(exc)


def _maybe_print_hf_402_hint(exc: BaseException) -> None:
    """Explain 402 once: Inference Providers billing, not a bug in this repo."""
    global _PRINTED_HF_402_HINT
    if _PRINTED_HF_402_HINT or not _is_hf_inference_402(exc):
        return
    _PRINTED_HF_402_HINT = True
    print(
        "[WARN] Hugging Face returned HTTP 402: Inference Providers quota/credits exhausted for this token.\n"
        "  This is decided on HF servers — application code cannot turn it into a successful LLM call.\n"
        "  Fix: https://huggingface.co/settings/billing (add credits, prepaid, or PRO) or use a hackathon token.\n"
        "  Dev without HF: set API_BASE_URL=http://127.0.0.1:11434/v1 MODEL_NAME=llama3.2 (Ollama) and empty HF_TOKEN.\n"
        "  Optional: INFERENCE_SKIP_LLM_AFTER_402=1 skips further router calls after the first 402 (fallback only).\n",
        file=sys.stderr,
        flush=True,
    )


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _llm_url_looks_local(url: str) -> bool:
    u = url.lower()
    return "127.0.0.1" in u or "localhost" in u

# OpenEnv simulation server URL only (set OPENENV_BASE_URL when using HTTP/WebSocket client)
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
# If true, run SME negotiation in-process (no uv run server). Default false so inference matches deployed HTTP API.
OPENENV_IN_PROCESS = os.getenv("OPENENV_IN_PROCESS", "0").strip().lower() in ("1", "true", "yes", "on")

NEGOTIATION_SYSTEM_PROMPT = """
You represent an SME supplier in B2B negotiation (motivation: Razorpay Fix My Itch —
long buyer payment cycles vs faster supplier pay, working-capital stress, itch score 82.8 in B2B Services).
Respond ONLY with valid JSON containing keys: action_type, price, payment_days, use_treds, reason, and optionally
propose_late_payment_penalty_clause, propose_dynamic_discounting, dynamic_discount_annual_rate.
action_type must be one of: propose, accept, reject.

CRITICAL — accept actions:
When you send action_type="accept", payment_days MUST exactly match the payment_days from your
IMMEDIATELY PREVIOUS propose action. Never copy the buyer's current payment_days into an accept
(unless that number is also what you last proposed). Mismatches invalidate the deal on strict tasks.
Example: if you last proposed payment_days=60, your accept must also use payment_days=60.
Therefore: ALWAYS propose at your target payment_days FIRST, then accept on the NEXT step.

CRITICAL — reject:
NEVER use action_type="reject" unless you intentionally end with no deal. Rejection terminates the
episode immediately with zero reward. Prefer action_type="propose" to counter-offer, or
action_type="accept" to agree.

STRATEGY by task difficulty:
- easy: Propose payment_days=60 immediately (= LiquidityThreshold). Keep price ABOVE CostThreshold.
  Jump directly to target — do NOT reduce by 1-2 days per step.
- medium: Target payment_days<=45. Set propose_late_payment_penalty_clause=true in EVERY proposal.
  This unlocks partial credit even if days aren't perfect.
- hard: ALWAYS set propose_dynamic_discounting=true and dynamic_discount_annual_rate=0.02 in EVERY action.
  Target payment_days=30. The grader scores ONLY on dynamic discounting NPV — NOT on payment_days
  or use_treds alone. High discount rates (>0.05) produce negative NPV and zero reward.
  Without propose_dynamic_discounting=true you will score 0.
    HARD — negotiation style: buyer power is high, so expect multi-round bargaining.
    Keep dynamic discounting enabled while improving terms over multiple rounds; accept only when terms are favorable.

CRITICAL — TReDS usage policy:
- If buyer payment_days is materially above your liquidity threshold (usually by 10+ days), explicitly consider
    use_treds=true to unlock invoice financing and reduce liquidity stress.
- In medium/hard tasks, when buyer days remain stubbornly high, include at least one proposal with use_treds=true.
- If you decide not to use TReDS, explain briefly in reason (e.g., days already near liquidity target).

FEW-SHOT TReDS example (medium):
Observation: buyer_days=60, liquidity_threshold=45, cost_threshold=80.
Valid action JSON:
{"action_type":"propose","price":95.0,"payment_days":50,"use_treds":true,"reason":"Use TReDS to bridge working-capital gap while converging on terms","propose_late_payment_penalty_clause":true}

FEW-SHOT TReDS example (hard):
Observation: buyer_days=100, liquidity_threshold=55, cost_threshold=78.
Valid action JSON:
{"action_type":"propose","price":89.0,"payment_days":30,"use_treds":true,"reason":"Pair TReDS with dynamic discounting for cash-flow resilience","propose_dynamic_discounting":true,"dynamic_discount_annual_rate":0.02}

CRITICAL — price floor:
Your price must ALWAYS stay above cost_threshold from the observation. Never propose or accept
a price below cost_threshold — this causes reward penalty.

CRITICAL — negotiation speed:
You have only 10-16 rounds total. Make AGGRESSIVE proposals.
Do NOT reduce payment_days by 1-2 per step — jump directly to your target.
""".strip()


def _clip_ascii_text(value: Any, max_len: int) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "~"


def _serialize_step_action(action: NegotiationAction) -> str:
    """Serialize the action JSON on one line with compact separators for log safety."""
    return json.dumps(action.model_dump(), ensure_ascii=True, separators=(",", ":"))


def _format_step_error(llm_error: str | None) -> str:
    """Return parser-safe [STEP] error field: null token or compact JSON string."""
    if llm_error is None:
        return "null"
    return json.dumps(_clip_ascii_text(llm_error, _MAX_ERROR_CHARS), ensure_ascii=True)


def _format_end_line(success: bool, steps: int, score: float, rewards: List[float]) -> str:
    return (
        f'[END] success={"true" if success else "false"} steps={steps} '
        f'score={score:.2f} rewards={",".join(f"{r:.2f}" for r in rewards)}'
    )


def _task_margin(task_name: str) -> float:
    t = task_name.lower()
    if "easy" in t:
        return 2.0
    if "hard" in t:
        return 1.0
    return 1.5


def _task_target_days(task_name: str, liquidity_threshold: int) -> int:
    return 30 if "hard" in task_name.lower() else liquidity_threshold


def _hard_fields_valid(action_payload: Dict[str, Any]) -> bool:
    if not bool(action_payload.get("propose_dynamic_discounting", False)):
        return False
    rate = float(action_payload.get("dynamic_discount_annual_rate", 0.0))
    return 0.0 < rate <= 0.05


def _proposal_viable_for_close(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
) -> bool:
    cost = float(observation.get("cost_threshold", 0.0))
    liq = int(observation.get("liquidity_threshold", 0))
    margin = _task_margin(task_name)
    price_ok = float(action_payload.get("price", 0.0)) >= (cost + margin)
    days_ok = int(action_payload.get("payment_days", 10**9)) <= liq
    if "hard" in task_name.lower() and not _hard_fields_valid(action_payload):
        return False
    return price_ok and days_ok


def _enforce_task_contract_fields(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    out = dict(action_payload)
    t = task_name.lower()
    buyer_days = int(observation.get("buyer_days", 0))
    liq = int(observation.get("liquidity_threshold", 0))

    if "medium" in t:
        out["propose_late_payment_penalty_clause"] = True

    if "hard" in t:
        out["propose_dynamic_discounting"] = True
        rate = float(out.get("dynamic_discount_annual_rate", 0.02))
        if rate <= 0.0 or rate > 0.05:
            out["dynamic_discount_annual_rate"] = 0.02
        else:
            out["dynamic_discount_annual_rate"] = round(rate, 4)
        # Keep financing mechanic active when day-gap pressure is material.
        if buyer_days > liq + 10:
            out["use_treds"] = True
    return out


def _normalize_stage1_proposal(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
    round_number: int,
    last_valid_proposal: Dict[str, Any] | None,
) -> Dict[str, Any]:
    out = dict(action_payload)
    out["action_type"] = "propose"

    buyer_price = float(observation.get("buyer_price", 100.0))
    buyer_days = int(observation.get("buyer_days", 90))
    liq = int(observation.get("liquidity_threshold", 60))
    cost = float(observation.get("cost_threshold", 80.0))

    margin = _task_margin(task_name)
    target_days = _task_target_days(task_name, liq)
    day_step = 8 if "easy" in task_name.lower() else (5 if "hard" in task_name.lower() else 6)

    price_floor = cost + margin
    llm_price = float(out.get("price", buyer_price))
    llm_days = int(out.get("payment_days", buyer_days))

    deterministic_days = max(target_days, buyer_days - day_step)
    if last_valid_proposal is not None:
        deterministic_days = min(deterministic_days, int(last_valid_proposal.get("payment_days", deterministic_days)))
    proposed_days = max(target_days, min(llm_days, deterministic_days))

    deterministic_price = max(price_floor, buyer_price - (0.4 + 0.2 * round_number))
    proposed_price = max(price_floor, min(llm_price, deterministic_price, buyer_price))
    if last_valid_proposal is not None:
        proposed_price = min(proposed_price, float(last_valid_proposal.get("price", proposed_price)))

    out["payment_days"] = int(proposed_days)
    out["price"] = round(proposed_price, 2)
    out["use_treds"] = bool(out.get("use_treds", False))
    out = _enforce_task_contract_fields(out, observation, task_name)
    return out


def _should_close_deal(
    observation: Dict[str, Any],
    task_name: str,
    round_number: int,
    last_valid_proposal: Dict[str, Any] | None,
) -> bool:
    if last_valid_proposal is None:
        return False

    buyer_days = int(observation.get("buyer_days", 0))
    buyer_price = float(observation.get("buyer_price", 0.0))
    liq = int(observation.get("liquidity_threshold", 0))
    cost = float(observation.get("cost_threshold", 0.0))
    max_rounds = int(observation.get("max_rounds", 16))
    remaining = max_rounds - (round_number + 1)
    margin = _task_margin(task_name)

    in_zone = buyer_days <= liq and buyer_price >= (cost + margin)
    late_window = 4 if "hard" in task_name.lower() else 3

    if in_zone and _proposal_viable_for_close(last_valid_proposal, observation, task_name):
        return True
    if remaining <= late_window and _proposal_viable_for_close(last_valid_proposal, observation, task_name):
        return True
    return False


def _build_accept_from_last_proposal(
    last_valid_proposal: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    out = {
        "action_type": "accept",
        "price": float(last_valid_proposal.get("price", observation.get("buyer_price", 0.0))),
        "payment_days": int(last_valid_proposal.get("payment_days", observation.get("buyer_days", 0))),
        "use_treds": bool(last_valid_proposal.get("use_treds", False)),
        "reason": "Close deal in agreement zone before max rounds.",
        "propose_late_payment_penalty_clause": bool(
            last_valid_proposal.get("propose_late_payment_penalty_clause", False)
        ),
        "propose_dynamic_discounting": bool(last_valid_proposal.get("propose_dynamic_discounting", False)),
        "dynamic_discount_annual_rate": float(last_valid_proposal.get("dynamic_discount_annual_rate", 0.0)),
    }
    return _enforce_task_contract_fields(out, observation, task_name)

# Local OpenAI-compatible servers (Ollama, LM Studio) do not need a real key; HF router does.
_OPENAI_API_KEY = HF_TOKEN or ("not-needed" if _llm_url_looks_local(API_BASE_URL) else "")
client = OpenAI(base_url=API_BASE_URL, api_key=_OPENAI_API_KEY)


class InProcessSMENegotiatorBridge:
    """Same async shape as SMENegotiatorEnv but drives :class:`SMENegotiatorEnvironment` in-process.

    Use when you do not have ``uv run server`` running — no WebSocket/HTTP env process required.
    """

    def __init__(self) -> None:
        self._env = SMENegotiatorEnvironment()

    async def __aenter__(self) -> "InProcessSMENegotiatorBridge":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def reset(self, **kwargs: Any) -> StepResult[NegotiationObservation]:
        obs = self._env.reset(**kwargs)
        return StepResult(observation=obs, reward=obs.reward, done=bool(obs.done))

    async def step(self, action: NegotiationAction, **kwargs: Any) -> StepResult[NegotiationObservation]:
        obs = self._env.step(action, **kwargs)
        return StepResult(observation=obs, reward=obs.reward, done=bool(obs.done))


def _observation_to_dict(observation: Any) -> Dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return dict(observation)


def format_observation(obs: Dict[str, Any]) -> str:
    msg = (obs.get("message") or "").strip()
    scenario = f"EnvMessage={msg[:900]}{'…' if len(msg) > 900 else ''}\n" if msg else ""
    return (
        f"{scenario}"
        f"Round={obs.get('round_number')} | Task={obs.get('task_name')} | "
        f"BuyerPrice={obs.get('buyer_price')} | BuyerDays={obs.get('buyer_days')} | "
        f"LiquidityThreshold={obs.get('liquidity_threshold')} | CostThreshold={obs.get('cost_threshold')} | "
        f"MonthlyRevenueINR={obs.get('sme_monthly_revenue')} | WCGap={obs.get('working_capital_gap')} | "
        f"SupplierPayDays={obs.get('sme_supplier_payment_days')} | InterestAnnual={obs.get('interest_rate_annual')} | "
        f"BuyerPower={obs.get('buyer_power_score')}"
    )


def _safe_fallback_action(observation: Any, task_name: str = "", round_number: int = 0) -> Dict[str, Any]:
    is_hard = "hard" in task_name.lower()
    is_medium = "medium" in task_name.lower()

    obs_dict = _observation_to_dict(observation) if not isinstance(observation, dict) else observation
    buyer_days = int(obs_dict.get("buyer_days", 0))
    liquidity = int(obs_dict.get("liquidity_threshold", 0))
    buyer_price = float(obs_dict.get("buyer_price", 0.0))
    cost = float(obs_dict.get("cost_threshold", 0.0))
    max_rounds = int(obs_dict.get("max_rounds", 16))

    terms_acceptable = buyer_days <= liquidity and buyer_price >= cost
    near_end = round_number >= max_rounds - 3  # accept in last 3 rounds

    should_accept = terms_acceptable or near_end

    target_days = 30 if is_hard else liquidity

    return {
        "action_type": "accept" if should_accept else "propose",
        "price": round(max(cost + 1.0, buyer_price * 0.99), 2),
        "payment_days": target_days,
        "use_treds": False,
        "reason": "Fallback action",
        "propose_dynamic_discounting": is_hard,
        "dynamic_discount_annual_rate": 0.02 if is_hard else 0.0,
        "propose_late_payment_penalty_clause": is_medium,
    }


def _task_hint(task_name: str, observation: Dict[str, Any]) -> str:
    liq = observation.get("liquidity_threshold", 60)
    cost = observation.get("cost_threshold", 80)
    buyer_days = observation.get("buyer_days", 90)
    t = task_name.lower()
    if "easy" in t:
        return (
            f"TARGET: payment_days<={liq}. Propose payment_days={liq} with price above {cost}. "
            f"Reduce aggressively (5-10 days/step). Once you propose payment_days={liq}, ACCEPT next step."
        )
    if "medium" in t:
        return (
            f"TARGET: payment_days<={liq}. Set propose_late_payment_penalty_clause=true always. "
            f"Reduce aggressively (5-8 days/step). Propose payment_days={liq} then ACCEPT next step."
        )
    if "hard" in t:
        return (
            f"TARGET: payment_days=30. MUST set propose_dynamic_discounting=true, "
            f"dynamic_discount_annual_rate=0.02. Price must stay above {cost}. "
            f"Jump aggressively from {buyer_days} to 30. Do NOT inch down by 1 day. "
            f"Maintain dynamic discounting and negotiate across multiple rounds; avoid forced immediate accept."
        )
    return ""


def _maybe_enable_treds_guardrail(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
    round_number: int,
) -> Dict[str, Any]:
    """Guarantee at least one TReDS attempt in medium/hard when day-gap pressure is high."""
    t = task_name.lower()
    if "medium" not in t and "hard" not in t:
        return action_payload

    # Force early so the environment mechanic can influence later rounds.
    if round_number > 1:
        return action_payload

    if str(action_payload.get("action_type", "propose")).lower() != "propose":
        return action_payload

    buyer_days = int(observation.get("buyer_days", 0))
    liquidity = int(observation.get("liquidity_threshold", 0))
    if buyer_days <= liquidity + 10:
        return action_payload

    if bool(action_payload.get("use_treds", False)):
        return action_payload

    out = dict(action_payload)
    out["use_treds"] = True
    prior_reason = _clip_ascii_text(out.get("reason", ""), _MAX_REASON_CHARS)
    if prior_reason:
        out["reason"] = _clip_ascii_text(
            f"{prior_reason} | Guardrail: activate TReDS due to large day-gap.",
            _MAX_REASON_CHARS,
        )
    else:
        out["reason"] = "Guardrail: activate TReDS due to large day-gap."
    return out


def get_agent_action(observation: Dict[str, Any], history: List[dict], task_name: str) -> Dict[str, Any]:
    hint = _task_hint(task_name, observation)
    user_message = (
        f"Task={task_name}\n"
        f"{hint}\n"
        f"Current observation:\n{format_observation(observation)}\n"
        "Return only JSON action."
    )

    messages = [{"role": "system", "content": NEGOTIATION_SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=180,
    )

    content = completion.choices[0].message.content
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        action = json.loads(raw)
        if not isinstance(action, dict):
            raise ValueError("LLM JSON root must be an object")
        out: Dict[str, Any] = {
            "action_type": str(action.get("action_type", "propose")).lower(),
            "price": float(action.get("price", observation.get("buyer_price", 0.0))),
            "payment_days": int(action.get("payment_days", observation.get("buyer_days", 0))),
            "use_treds": bool(action.get("use_treds", False)),
            "reason": _clip_ascii_text(action.get("reason", ""), _MAX_REASON_CHARS),
        }
        if "propose_late_payment_penalty_clause" in action:
            out["propose_late_payment_penalty_clause"] = bool(action.get("propose_late_payment_penalty_clause"))
        if "propose_dynamic_discounting" in action:
            out["propose_dynamic_discounting"] = bool(action.get("propose_dynamic_discounting"))
        if "dynamic_discount_annual_rate" in action:
            out["dynamic_discount_annual_rate"] = float(action.get("dynamic_discount_annual_rate", 0.0))
        return out
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("LLM output not valid JSON (%s); using prose/regex parser. Raw snippet: %r", exc, raw[:300])
        parsed = parse_llm_text_to_negotiation_action(raw, observation, allow_json=False)
        return parsed.model_dump()


def _parse_last_assistant_action(history: List[dict]) -> Dict[str, Any] | None:
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        raw = (m.get("content") or "").strip()
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _hard_two_step_policy_enabled() -> bool:
    # Default disabled to preserve genuine hard-task dynamics and avoid shortcut behavior.
    return os.getenv("INFERENCE_HARD_TWO_STEP", "0").strip().lower() not in ("0", "false", "no", "off")


def _coerce_hard_accept_after_propose(
    action: Dict[str, Any],
    history: List[dict],
    task_name: str,
    round_number: int,
) -> Dict[str, Any]:
    """If last action was a qualifying propose, force accept with same terms (env accepts_own_proposal)."""
    if not _hard_two_step_policy_enabled():
        return action
    if "hard" not in task_name.lower() or round_number < 1:
        return action
    prev = _parse_last_assistant_action(history)
    if not prev:
        return action
    if str(prev.get("action_type", "")).lower() != "propose":
        return action
    if not prev.get("propose_dynamic_discounting"):
        return action
    if str(action.get("action_type", "propose")).lower() == "accept":
        return action
    p = float(prev.get("price", 0.0))
    d = int(prev.get("payment_days", 0))
    return {
        "action_type": "accept",
        "price": p,
        "payment_days": d,
        "use_treds": bool(prev.get("use_treds", False)),
        "reason": "Accept prior propose (hard two-step policy; matches env contract).",
        "propose_late_payment_penalty_clause": bool(prev.get("propose_late_payment_penalty_clause", False)),
        "propose_dynamic_discounting": True,
        "dynamic_discount_annual_rate": float(prev.get("dynamic_discount_annual_rate", 0.02)),
    }


def _to_model_action(action_payload: Dict[str, Any], observation: Any) -> NegotiationAction:
    action_type = str(action_payload.get("action_type", "propose")).lower()
    if action_type not in {"propose", "accept", "reject"}:
        action_type = "propose"

    price = float(action_payload.get("price", observation.buyer_price))
    payment_days = int(action_payload.get("payment_days", observation.buyer_days))
    use_treds = bool(action_payload.get("use_treds", False))
    reason = _clip_ascii_text(action_payload.get("reason", "Model-selected action"), _MAX_REASON_CHARS)

    return NegotiationAction(
        action_type=action_type,
        price=round(price, 2),
        payment_days=payment_days,
        use_treds=use_treds,
        reason=reason,
        propose_late_payment_penalty_clause=bool(action_payload.get("propose_late_payment_penalty_clause", False)),
        propose_dynamic_discounting=bool(action_payload.get("propose_dynamic_discounting", False)),
        dynamic_discount_annual_rate=float(action_payload.get("dynamic_discount_annual_rate", 0.0)),
    )


EnvClient = Union[SMENegotiatorEnv, InProcessSMENegotiatorBridge]


async def run_episode(env: EnvClient, difficulty: str, seed: int) -> Dict[str, Any]:
    """Run one episode using model-guided actions with strict stdout formatting."""

    task_name = difficulty.lower()
    episode_id = f"{task_name}-{seed}"
    history: List[dict] = []

    all_rewards: List[float] = []
    round_number = 0
    success = False
    forced_hard_accepts = 0
    result: Any = None
    observation: Any = None
    final_score = 0.0
    last_valid_proposal: Dict[str, Any] | None = None

    try:
        result = await env.reset(seed=seed, difficulty=difficulty, episode_id=episode_id, task_name=task_name)
        observation = result.observation

        print(
            f"[START] task={task_name} env=openenv-sme-negotiator model={MODEL_NAME}",
            flush=True,
        )

        # After first HF 402, optionally stop calling the router (still not a "fix" — avoids spam & wasted calls).
        skip_llm_after_402 = _env_truthy("INFERENCE_SKIP_LLM_AFTER_402", False)
        llm_blocked_402 = False

        # Termination is driven by the environment (``done``), including max rounds — do not stop early here.
        while not result.done:
            obs_dict = _observation_to_dict(observation)

            llm_error: str | None = None
            if llm_blocked_402:
                action_payload = _safe_fallback_action(observation, task_name, round_number)
                llm_error = (
                    "HF Inference 402 — further LLM calls skipped (INFERENCE_SKIP_LLM_AFTER_402=1). "
                    "Resolve quota at https://huggingface.co/settings/billing or use local Ollama."
                )
            else:
                try:
                    action_payload = get_agent_action(obs_dict, history, task_name)
                except Exception as e:
                    _maybe_print_hf_402_hint(e)
                    print(
                        f"[ERROR] LLM call failed: {type(e).__name__}: {e}",
                        file=sys.stderr, flush=True,
                    )
                    logger.warning(
                        "LLM call failed; using fallback action: %s: %s",
                        type(e).__name__,
                        e,
                    )
                    if os.getenv("INFERENCE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes"):
                        logger.exception("LLM traceback (INFERENCE_DEBUG_LLM=1)")
                    llm_error = str(e)
                    if skip_llm_after_402 and _is_hf_inference_402(e):
                        llm_blocked_402 = True
                    action_payload = _safe_fallback_action(observation, task_name, round_number)

            action_payload = _coerce_hard_accept_after_propose(
                action_payload, history, task_name, round_number
            )
            if str(action_payload.get("reason", "")).startswith("Accept prior propose (hard two-step policy"):
                forced_hard_accepts += 1

            if _should_close_deal(obs_dict, task_name, round_number, last_valid_proposal):
                action_payload = _build_accept_from_last_proposal(last_valid_proposal, obs_dict, task_name)
            else:
                action_payload = _normalize_stage1_proposal(
                    action_payload,
                    obs_dict,
                    task_name,
                    round_number,
                    last_valid_proposal,
                )
                action_payload = _maybe_enable_treds_guardrail(
                    action_payload,
                    obs_dict,
                    task_name,
                    round_number,
                )

            action = _to_model_action(action_payload, observation)
            action_json = _serialize_step_action(action)

            if action.action_type == "propose":
                last_valid_proposal = action.model_dump()

            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            all_rewards.append(reward)

            err_out = _format_step_error(llm_error)
            print(
                f'[STEP] step={round_number + 1} action={action_json} reward={reward:.2f} '
                f'done={"true" if done else "false"} error={err_out}',
                flush=True,
            )

            history.append({"role": "user", "content": format_observation(obs_dict)})
            history.append({"role": "assistant", "content": json.dumps(action_payload, ensure_ascii=True)})

            round_number += 1

        final_score = float(result.reward or 0.0)
        meta = getattr(result.observation, "metadata", None) or {}
        if isinstance(meta, dict) and "success" in meta:
            success = bool(meta["success"])
        else:
            success = bool(result.done and final_score > 0.0)
    finally:
        total_reward = sum(all_rewards)
        print(_format_end_line(success, round_number, final_score, all_rewards), flush=True)

    return {
        "difficulty": difficulty,
        "seed": seed,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps": round_number,
        "success": success,
        "forced_hard_accepts": forced_hard_accepts,
        "step_rewards": all_rewards,
        "final_observation": _observation_to_dict(observation) if observation is not None else {},
    }


async def main() -> None:
    """Run three episodes per difficulty and write a compact results file."""

    print(f"[CONFIG] LLM API_BASE_URL={API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)
    if "router.huggingface.co" in API_BASE_URL:
        print(
            "[CONFIG] Hugging Face router uses /v1/chat/completions — pick a chat/instruct model id "
            "(e.g. Qwen/Qwen2.5-7B-Instruct). If you see 'not a chat model', change MODEL_NAME in .env.",
            file=sys.stderr, flush=True,
        )
    if _llm_url_looks_local(API_BASE_URL):
        print(
            "[CONFIG] API_BASE_URL points to this machine (localhost/127.0.0.1). "
            "WinError 10061 / 'connection refused' means nothing is listening there — "
            "start your local OpenAI-compatible server (Ollama, LM Studio, vLLM, …), OR "
            "set API_BASE_URL=https://router.huggingface.co/v1 and HF_TOKEN for Hugging Face Inference.",
            file=sys.stderr, flush=True,
        )
    if not HF_TOKEN and not _llm_url_looks_local(API_BASE_URL):
        print(
            "[WARN] HF_TOKEN is empty. Hugging Face router usually requires HF_TOKEN in .env.",
            file=sys.stderr, flush=True,
        )
    elif not HF_TOKEN and _llm_url_looks_local(API_BASE_URL):
        print(
            "[WARN] HF_TOKEN is empty; OK only if your local server does not require a key.",
            file=sys.stderr, flush=True,
        )

    results: Dict[str, Any] = {
        "metadata": {
            "llm_api_base_url": API_BASE_URL,
            "openenv_base_url": OPENENV_BASE_URL,
            "openenv_in_process": OPENENV_IN_PROCESS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
        },
        "tasks": {},
    }

    if OPENENV_IN_PROCESS:
        env_manager: Any = InProcessSMENegotiatorBridge()
    else:
        env_manager = SMENegotiatorEnv(base_url=OPENENV_BASE_URL)

    try:
        async with env_manager as env:
            await _run_all_episodes(env, results)
    except ConnectionError as exc:
        print(
            "\n[openenv] Could not connect to the simulation server at "
            f"{OPENENV_BASE_URL} (WebSocket ws://…/ws).\n"
            "  Start it in another terminal:\n"
            "    uv run server\n"
            "  Or run inference without a server:\n"
            "    set OPENENV_IN_PROCESS=1\n"
            "    uv run python inference.py\n",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(1) from exc


async def _run_all_episodes(env: EnvClient, results: Dict[str, Any]) -> None:
    all_difficulties = ["EASY", "MEDIUM", "HARD"]
    task_filter = os.getenv("TASK_FILTER", "").strip().upper()
    if task_filter:
        allowed = {p.strip() for p in task_filter.split(",") if p.strip()}
        if allowed:
            all_difficulties = [d for d in all_difficulties if d in allowed]
        if not all_difficulties:
            print(f"[WARN] TASK_FILTER={task_filter!r} matched no tasks, running all.", file=sys.stderr, flush=True)
            all_difficulties = ["EASY", "MEDIUM", "HARD"]

    all_seeds = [1000, 1001, 1002]
    num_episodes = os.getenv("NUM_EPISODES", "").strip()
    if num_episodes:
        all_seeds = all_seeds[:int(num_episodes)]

    for difficulty in all_difficulties:
        episode_results: List[Dict[str, Any]] = []
        for seed in all_seeds:
            episode_results.append(await run_episode(env, difficulty, seed))

        scores = [episode["final_score"] for episode in episode_results]
        rewards = [episode["total_reward"] for episode in episode_results]
        successes = [episode["success"] for episode in episode_results]

        results["tasks"][difficulty] = {
            "episodes": episode_results,
            "summary": {
                "mean_final_score": sum(scores) / len(scores),
                "mean_total_reward": sum(rewards) / len(rewards),
                "success_rate": sum(1 for success in successes if success) / len(successes),
            },
        }

    _finalize_results_summary(results)


def _finalize_results_summary(results: Dict[str, Any]) -> None:
    overall_scores = [
        episode["final_score"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    overall_rewards = [
        episode["total_reward"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    overall_successes = [
        episode["success"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]

    results["summary"] = {
        "overall_mean_score": sum(overall_scores) / len(overall_scores),
        "overall_mean_reward": sum(overall_rewards) / len(overall_rewards),
        "overall_success_rate": sum(1 for success in overall_successes if success) / len(overall_successes),
    }

    with open("inference_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
