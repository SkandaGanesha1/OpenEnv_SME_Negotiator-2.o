#!/usr/bin/env python3
"""Inference runner for the SME negotiation environment."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from openenv.core.client_types import StepResult

from rl.bridge import format_observation as format_liquidity_observation
from rl.bridge import make_environment_factory, parse_action as parse_liquidity_action
from sme_negotiator_env.client import SMENegotiatorEnv, choose_action as choose_legacy_action
from sme_negotiator_env.llm_action_parser import parse_llm_text_to_negotiation_action
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation
from sme_negotiator_env.prompting import (
    action_payload_to_model_action,
    clip_ascii_text,
    format_observation_text,
    observation_to_dict,
)
from sme_negotiator_env.reward_reporting import (
    build_legacy_step_diagnostics,
    build_shadow_reward_report,
)

from server.environment import SMENegotiatorEnvironment

logger = logging.getLogger(__name__)

# Allow .env to override pre-set shell vars (e.g. stale Groq API_BASE_URL in the same terminal).
load_dotenv(override=True)

# LLM: Hugging Face OpenAI-compatible router by default (override with API_BASE_URL in .env)
API_BASE_URL = (os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1").strip()
# Hackathon / dashboard may set either HF_TOKEN or API_KEY for the OpenAI client.
# HF Router chat/completions only accepts models exposed as *chat* models — not every Hub id works.
MODEL_NAME = (os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct").strip()
# Docker / HF Space image tag (validator builds). Inference uses HTTP or in-process env — not from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "openenv-sme-negotiator:latest")

# Printed once per process when HF Inference returns HTTP 402 (quota / billing — not fixable in Python).
_PRINTED_HF_402_HINT = False
_MAX_REASON_CHARS = 160
_MAX_ERROR_CHARS = 240
_STRICT_EPS = 1e-6


def _strict_unit_interval(score: float) -> float:
    """Clamp scores to the strict open interval (0, 1)."""
    value = float(score)
    if not math.isfinite(value):
        return _STRICT_EPS
    return float(min(1.0 - _STRICT_EPS, max(_STRICT_EPS, value)))


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


def _resolve_router_token(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Resolve inference token precedence without relying on module reloads."""
    source = os.environ if env is None else env
    return (
        (source.get("HF_TOKEN") or source.get("API_KEY") or source.get("OPENAI_API_KEY") or "").strip() or None
    )


def _resolve_openai_client_key(api_base_url: str, env: Optional[Mapping[str, str]] = None) -> str:
    token = _resolve_router_token(env)
    if token:
        return token
    return "not-needed" if _llm_url_looks_local(api_base_url) else ""


def _normalize_inference_env_mode(value: str) -> str:
    mode = (value or DEFAULT_INFERENCE_ENV_MODE).strip().lower()
    if mode not in {"legacy", "liquidity"}:
        return DEFAULT_INFERENCE_ENV_MODE
    return mode


def _inference_env_mode() -> str:
    return _normalize_inference_env_mode(os.getenv("INFERENCE_ENV_MODE", DEFAULT_INFERENCE_ENV_MODE))


def _inference_reward_mode() -> str:
    mode = (os.getenv("INFERENCE_REWARD_MODE", DEFAULT_INFERENCE_REWARD_MODE) or DEFAULT_INFERENCE_REWARD_MODE).strip()
    if mode not in {"legacy", "legacy+shadow_rlvr", "legacy+full_debug"}:
        return DEFAULT_INFERENCE_REWARD_MODE
    return mode


def _normalize_inference_agent_mode(value: str) -> str:
    mode = (value or DEFAULT_INFERENCE_AGENT_MODE).strip().lower()
    if mode not in {"router", "heuristic"}:
        return DEFAULT_INFERENCE_AGENT_MODE
    return mode


def _inference_agent_mode() -> str:
    return _normalize_inference_agent_mode(os.getenv("INFERENCE_AGENT_MODE", DEFAULT_INFERENCE_AGENT_MODE))


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = (os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _inference_history_max_turns() -> int:
    return _int_env("INFERENCE_HISTORY_MAX_TURNS", 6, minimum=1)


def _inference_history_max_chars() -> int:
    return _int_env("INFERENCE_HISTORY_MAX_CHARS", 6000, minimum=500)


def _inference_history_summary_max_chars() -> int:
    return _int_env("INFERENCE_HISTORY_SUMMARY_MAX_CHARS", 1200, minimum=50)


def _inference_llm_max_tokens() -> int:
    return _int_env("INFERENCE_LLM_MAX_TOKENS", 160, minimum=32)


def _openenv_in_process_enabled() -> bool:
    return _env_truthy("OPENENV_IN_PROCESS", False)


def _runtime_banner(env_mode: str) -> str:
    if env_mode == "liquidity":
        return "LIQUIDITY_IN_PROCESS_ADVANCED"
    return "LEGACY_LIVE_BASELINE"


def _liquidity_task_for_difficulty(difficulty: str) -> str:
    explicit = os.getenv("INFERENCE_LIQUIDITY_TASK", "").strip()
    if explicit:
        return explicit
    return "liquidity-correlation-hard" if str(difficulty).lower() == "hard" else "liquidity-stress-medium"

# OpenEnv simulation server URL only (set OPENENV_BASE_URL when using HTTP/WebSocket client)
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")

DEFAULT_INFERENCE_ENV_MODE = "liquidity"
DEFAULT_INFERENCE_REWARD_MODE = "legacy+shadow_rlvr"
DEFAULT_INFERENCE_AGENT_MODE = "router"
HF_TOKEN = _resolve_router_token()

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

LIQUIDITY_SYSTEM_PROMPT = """
You represent an SME treasury agent operating in a long-horizon liquidity environment.
Respond ONLY with valid JSON. Supported action types are:
- propose
- accept
- reject
- tool
- simulate_plan
- advance_period

When action_type is:
- propose / accept: include price, payment_days, use_treds, optionally deal_id and negotiation clause fields.
- reject: include deal_id when available and a short reason.
- tool: include tool_name in {QUERY_TREDS, CHECK_COMPLIANCE, RUN_CASHFLOW_SIM} and tool_args.
- simulate_plan: include simulation_plan and optionally simulation_horizon.
- advance_period: no extra fields required.

Treasury priorities:
- Avoid default and preserve positive NPV.
- Use tools when tenor risk or compliance risk is unclear.
- Advance macro periods only when open negotiation work is exhausted for now.
- Include deal_id for deal-specific actions whenever possible.

MANDATORY RULE - period advancement:
- If open_deal_ids is empty, done=false, and current_period < total_periods, you MUST return
  {"action_type":"advance_period","reason":"No open deals remain in this macro period."}
- Do NOT propose, accept, reject, or call a deal-specific tool when there are no open deals.
""".strip()


def _clip_ascii_text(value: Any, max_len: int) -> str:
    return clip_ascii_text(value, max_len)


def _strict_clip_text(value: Any, max_len: int) -> str:
    if max_len <= 0:
        return ""
    text = _clip_ascii_text(value, max_len)
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return text[:1]
    return text[: max_len - 1] + "~"


def _compact_action_payload_for_log(action_payload: Mapping[str, Any]) -> Dict[str, Any]:
    action_type = str(action_payload.get("action_type", "propose") or "propose").lower()
    compact: Dict[str, Any] = {"action_type": action_type}
    deal_id = action_payload.get("deal_id")
    reason = _strict_clip_text(action_payload.get("reason", ""), _MAX_REASON_CHARS)

    if deal_id and action_type != "advance_period":
        compact["deal_id"] = str(deal_id)

    if action_type in {"propose", "accept"}:
        compact["price"] = round(float(action_payload.get("price", 0.0) or 0.0), 2)
        compact["payment_days"] = int(action_payload.get("payment_days", 0) or 0)
        compact["use_treds"] = bool(action_payload.get("use_treds", False))
        if bool(action_payload.get("propose_late_payment_penalty_clause", False)):
            compact["propose_late_payment_penalty_clause"] = True
        if bool(action_payload.get("propose_dynamic_discounting", False)):
            compact["propose_dynamic_discounting"] = True
            rate = float(action_payload.get("dynamic_discount_annual_rate", 0.0) or 0.0)
            if rate > 0.0:
                compact["dynamic_discount_annual_rate"] = round(rate, 4)
    elif action_type == "tool":
        tool_name = str(action_payload.get("tool_name", "") or "").upper()
        if tool_name:
            compact["tool_name"] = tool_name
        tool_args = action_payload.get("tool_args")
        if isinstance(tool_args, Mapping) and tool_args:
            compact["tool_args"] = dict(tool_args)
    elif action_type == "simulate_plan":
        simulation_plan = action_payload.get("simulation_plan")
        if isinstance(simulation_plan, Mapping) and simulation_plan:
            compact["simulation_plan"] = dict(simulation_plan)
        if action_payload.get("simulation_horizon") is not None:
            compact["simulation_horizon"] = int(action_payload.get("simulation_horizon") or 0)

    if reason:
        compact["reason"] = reason
    return compact


def _serialize_step_action(action: NegotiationAction) -> str:
    """Serialize the action JSON on one line with compact separators for log safety."""
    return json.dumps(_compact_action_payload_for_log(action.model_dump()), ensure_ascii=True, separators=(",", ":"))


def _format_step_error(llm_error: str | None) -> str:
    """Return parser-safe [STEP] error field: null token or compact JSON string."""
    if llm_error is None:
        return "null"
    return json.dumps(_clip_ascii_text(llm_error, _MAX_ERROR_CHARS), ensure_ascii=True)


def _format_score_for_log(score: float) -> str:
    """Format score to 2 decimal places per mandatory rules."""
    return f"{_strict_unit_interval(score):.2f}"


def _format_end_line(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
    judge_score: Optional[float] = None,
    termination_reason: Optional[str] = None,
    defaulted_sme_count: Optional[int] = None,
) -> str:
    base = (
        f'[END] success={"true" if success else "false"} steps={steps} score={_format_score_for_log(score)} '
        f'rewards={",".join(_format_score_for_log(r) for r in rewards)}'
    )
    if judge_score is not None:
        base += f" judge_score={judge_score:.3f}"
    if termination_reason:
        base += f" termination_reason={termination_reason}"
    if defaulted_sme_count is not None:
        base += f" defaulted_sme_count={int(defaulted_sme_count)}"
    return base


def _format_terminal_reward_line(
    *,
    verifiable_reward: float,
    final_score: float,
    success: bool,
    source: str,
) -> str:
    return (
        "[TERMINAL_REWARD] "
        f"source={source} "
        f"verifiable={float(verifiable_reward or 0.0):.4f} "
        f"final_score={_format_score_for_log(final_score)} "
        f"success={'true' if success else 'false'}"
    )


def _format_verifiable_reward_breakdown_line(
    reward_breakdown: Optional[Mapping[str, Any]],
    *,
    canonical_total: Optional[float] = None,
) -> str:
    if not reward_breakdown:
        if canonical_total is None:
            return "[VERIFIABLE_REWARD] breakdown=unavailable"
        return f"[VERIFIABLE_REWARD] total={float(canonical_total or 0.0):.4f} breakdown=unavailable"
    keys = ("solvency", "liquidity", "npv", "compliance", "total")
    if not all(key in reward_breakdown for key in keys):
        if canonical_total is None:
            return "[VERIFIABLE_REWARD] breakdown=unavailable"
        return f"[VERIFIABLE_REWARD] total={float(canonical_total or 0.0):.4f} breakdown=unavailable"
    total_value = float(canonical_total if canonical_total is not None else reward_breakdown.get("total", 0.0) or 0.0)
    return (
        "[VERIFIABLE_REWARD] "
        f"solvency={float(reward_breakdown.get('solvency', 0.0) or 0.0):.4f} "
        f"liquidity={float(reward_breakdown.get('liquidity', 0.0) or 0.0):.4f} "
        f"npv={float(reward_breakdown.get('npv', 0.0) or 0.0):.4f} "
        f"compliance={float(reward_breakdown.get('compliance', 0.0) or 0.0):.4f} "
        f"total={total_value:.4f}"
    )


def _format_period_summary_line(
    *,
    closed_period: int,
    current_period: int,
    total_periods: int,
    resolved_deal_count: int,
    defaulted_sme_count: int,
    cumulative_reward: float,
) -> str:
    return (
        "[PERIOD_SUMMARY] "
        f"closed_period={closed_period} "
        f"next_period={current_period}/{total_periods} "
        f"resolved_deal_count={resolved_deal_count} "
        f"defaulted_sme_count={defaulted_sme_count} "
        f"cumulative_reward={float(cumulative_reward):.4f}"
    )


def _ascii_sparkline(rewards: List[float]) -> str:
    """Return a single-line ASCII bar chart summarising per-step rewards."""
    if not rewards:
        return ""
    blocks = "._-:=+*#"
    mx = max(rewards) if max(rewards) > 0 else 1.0
    bars = [blocks[min(7, int(r / mx * 8))] for r in rewards]
    mn = min(rewards)
    avg = sum(rewards) / len(rewards)
    return f"{''.join(bars)} min={mn:.2f} max={mx:.2f} avg={avg:.2f}"


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
    return abs(rate - 0.02) <= 1e-9


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

    if "easy" in t or "medium" in t:
        out["propose_dynamic_discounting"] = False
        out["dynamic_discount_annual_rate"] = 0.0

    if "medium" in t:
        out["propose_late_payment_penalty_clause"] = True

    if "hard" in t:
        out["propose_dynamic_discounting"] = True
        out["dynamic_discount_annual_rate"] = 0.02
        if buyer_days > liq + 10 or str(observation.get("last_tool_name", "")).upper() == "QUERY_TREDS":
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
    # Anchor to buyer's current offer (always decreasing) instead of locking to previous proposal.
    # This lets the LLM re-anchor each round without getting frozen at a single value.
    proposed_days = max(target_days, min(llm_days, buyer_days))

    deterministic_price = max(price_floor, buyer_price - (0.4 + 0.2 * round_number))
    proposed_price = max(price_floor, min(llm_price, deterministic_price, buyer_price))
    # No monotonicity lock on price — the floor (cost + margin) already prevents below-cost proposals.

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
_OPENAI_API_KEY = _resolve_openai_client_key(API_BASE_URL)
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


class InProcessLiquidityBridge:
    """Async-compatible bridge over the existing Stage 5/6 liquidity wrapper."""

    def __init__(self) -> None:
        wrapper_cls = make_environment_factory()
        self._wrapper = wrapper_cls()

    async def __aenter__(self) -> "InProcessLiquidityBridge":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    @property
    def env(self) -> Any:
        return self._wrapper.env

    @property
    def reward_breakdown(self) -> Any:
        return getattr(self._wrapper, "reward_breakdown", None)

    def summarize_episode(self) -> Any:
        return self._wrapper.summarize_episode()

    def build_episode_log(self) -> str:
        return self._wrapper.build_episode_log()

    async def reset(self, **kwargs: Any) -> StepResult[Any]:
        self._wrapper.reset(**kwargs)
        observation = self._wrapper.last_observation
        assert observation is not None
        return StepResult(observation=observation, reward=observation.reward, done=bool(observation.done))

    async def step(self, action: NegotiationAction, **kwargs: Any) -> StepResult[Any]:
        action_type = str(action.action_type).lower()
        if action_type == "propose":
            self._wrapper.propose(
                price=float(action.price),
                payment_days=int(action.payment_days),
                use_treds=bool(action.use_treds),
                deal_id=action.deal_id,
                reason=action.reason,
                propose_late_payment_penalty_clause=bool(action.propose_late_payment_penalty_clause),
                propose_dynamic_discounting=bool(action.propose_dynamic_discounting),
                dynamic_discount_annual_rate=float(action.dynamic_discount_annual_rate),
            )
        elif action_type == "accept":
            self._wrapper.accept(
                price=float(action.price),
                payment_days=int(action.payment_days),
                use_treds=bool(action.use_treds),
                deal_id=action.deal_id,
                reason=action.reason,
                propose_late_payment_penalty_clause=bool(action.propose_late_payment_penalty_clause),
                propose_dynamic_discounting=bool(action.propose_dynamic_discounting),
                dynamic_discount_annual_rate=float(action.dynamic_discount_annual_rate),
            )
        elif action_type == "reject":
            self._wrapper.reject(deal_id=action.deal_id, reason=action.reason)
        elif action_type == "tool":
            tool_name = str(action.tool_name or "").upper()
            if tool_name == "QUERY_TREDS":
                invoice_id = str((action.tool_args or {}).get("invoice_id") or action.deal_id or "")
                self._wrapper.query_treds(invoice_id=invoice_id, deal_id=action.deal_id)
            elif tool_name == "CHECK_COMPLIANCE":
                contract_id = str((action.tool_args or {}).get("contract_id") or action.deal_id or "")
                self._wrapper.check_compliance(contract_id=contract_id, deal_id=action.deal_id)
            elif tool_name == "RUN_CASHFLOW_SIM":
                tool_args = action.tool_args or {}
                self._wrapper.run_cashflow_sim(
                    plan=tool_args.get("plan") if isinstance(tool_args.get("plan"), dict) else {},
                    horizon=int(tool_args.get("horizon")) if tool_args.get("horizon") is not None else None,
                    deal_id=action.deal_id,
                )
            else:
                raise ValueError(f"Unsupported tool_name for liquidity inference: {tool_name!r}")
        elif action_type == "simulate_plan":
            self._wrapper.simulate_plan(
                plan=action.simulation_plan or {},
                horizon=action.simulation_horizon,
                deal_id=action.deal_id,
            )
        elif action_type == "advance_period":
            self._wrapper.advance_period()
        else:
            raise ValueError(f"Unsupported liquidity action type: {action_type!r}")

        observation = self._wrapper.last_observation
        assert observation is not None
        return StepResult(observation=observation, reward=observation.reward, done=bool(observation.done))


def _observation_to_dict(observation: Any) -> Dict[str, Any]:
    return observation_to_dict(observation)


format_observation = format_observation_text


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


format_observation = format_observation_text


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


def _safe_liquidity_fallback_action(observation: Any) -> NegotiationAction:
    obs_dict = _observation_to_dict(observation) if not isinstance(observation, dict) else observation
    open_deal_ids = list(obs_dict.get("open_deal_ids") or [])
    active_deal_id = obs_dict.get("active_deal_id") or (open_deal_ids[0] if open_deal_ids else None)

    if not open_deal_ids:
        return NegotiationAction(action_type="advance_period", reason="No open deals remain in this macro period.")

    buyer_days = int(obs_dict.get("buyer_days", 0) or 0)
    liquidity_threshold = int(obs_dict.get("liquidity_threshold", 0) or 0)
    buyer_price = float(obs_dict.get("buyer_price", 0.0) or 0.0)

    if obs_dict.get("last_tool_name") != "QUERY_TREDS" and buyer_days > liquidity_threshold + 10:
        return NegotiationAction(
            action_type="tool",
            deal_id=str(active_deal_id),
            tool_name="QUERY_TREDS",
            tool_args={"invoice_id": str(active_deal_id), "deal_id": str(active_deal_id)},
            reason="Fallback: inspect TReDS quote before accepting long tenor.",
        )

    return NegotiationAction(
        action_type="accept",
        deal_id=str(active_deal_id),
        price=round(buyer_price, 2),
        payment_days=buyer_days,
        use_treds=bool(buyer_days > liquidity_threshold),
        reason="Fallback: accept deterministic current terms to progress the episode.",
    )


def _annotate_reason(reason: Any, note: str) -> str:
    base = _clip_ascii_text(reason or "", _MAX_REASON_CHARS)
    if not base:
        return _clip_ascii_text(note, _MAX_REASON_CHARS)
    return _clip_ascii_text(f"{base} | {note}", _MAX_REASON_CHARS)


def _explicit_surrender_requested(reason: Any) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    surrender_phrases = (
        "walk away",
        "no deal",
        "terminate",
        "end negotiation",
        "surrender",
        "abandon",
    )
    return any(phrase in text for phrase in surrender_phrases)


def _action_matches_observation_terms(action_payload: Dict[str, Any], observation: Dict[str, Any]) -> bool:
    return (
        round(float(action_payload.get("price", 0.0) or 0.0), 2) == round(float(observation.get("buyer_price", 0.0) or 0.0), 2)
        and int(action_payload.get("payment_days", -1) or -1) == int(observation.get("buyer_days", -2) or -2)
    )


def _action_matches_last_proposal(
    action_payload: Dict[str, Any],
    last_valid_proposal: Optional[Dict[str, Any]],
) -> bool:
    if last_valid_proposal is None:
        return False
    return (
        round(float(action_payload.get("price", 0.0) or 0.0), 2) == round(float(last_valid_proposal.get("price", 0.0) or 0.0), 2)
        and int(action_payload.get("payment_days", -1) or -1) == int(last_valid_proposal.get("payment_days", -2) or -2)
    )


def _same_action_shape(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    if not left or not right:
        return False
    return (
        str(left.get("action_type", "")).lower() == str(right.get("action_type", "")).lower()
        and round(float(left.get("price", 0.0) or 0.0), 2) == round(float(right.get("price", 0.0) or 0.0), 2)
        and int(left.get("payment_days", -1) or -1) == int(right.get("payment_days", -2) or -2)
        and str(left.get("tool_name", "") or "").upper() == str(right.get("tool_name", "") or "").upper()
    )


def _remaining_rounds(observation: Dict[str, Any], round_number: int) -> int:
    max_rounds = int(observation.get("max_rounds", 16) or 16)
    return max(max_rounds - (round_number + 1), 0)


def _late_close_window(task_name: str) -> int:
    return 4 if "hard" in task_name.lower() else 3


def _build_liquidity_escape_action(
    observation: Dict[str, Any],
    task_name: str,
    round_number: int,
    last_valid_proposal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    open_deal_ids = list(observation.get("open_deal_ids") or [])
    active_deal_id = observation.get("active_deal_id") or (open_deal_ids[0] if open_deal_ids else None)
    buyer_days = int(observation.get("buyer_days", 0) or 0)
    liquidity = int(observation.get("liquidity_threshold", 0) or 0)
    buyer_price = float(observation.get("buyer_price", 0.0) or 0.0)
    metadata = observation.get("metadata") or {}

    if not open_deal_ids:
        return {
            "action_type": "advance_period",
            "reason": "No open deals remain in this macro period.",
        }

    if observation.get("last_tool_name") != "QUERY_TREDS" and buyer_days > liquidity + 10:
        return {
            "action_type": "tool",
            "deal_id": str(active_deal_id),
            "tool_name": "QUERY_TREDS",
            "tool_args": {"invoice_id": str(active_deal_id), "deal_id": str(active_deal_id)},
            "reason": "Escape hatch: inspect TReDS terms before repeating the same negotiation move.",
        }

    if "hard" in task_name.lower() and not bool(metadata.get("simulation_projection_present", False)):
        return {
            "action_type": "simulate_plan",
            "deal_id": str(active_deal_id),
            "simulation_plan": {"advance_periods": 1},
            "simulation_horizon": 1,
            "reason": "Escape hatch: simulate the next macro step before continuing the hard negotiation.",
        }

    if last_valid_proposal is not None and _should_close_deal(observation, task_name, round_number, last_valid_proposal):
        out = _build_accept_from_last_proposal(last_valid_proposal, observation, task_name)
        out["reason"] = _annotate_reason(out.get("reason"), "Escape hatch: close the best valid proposal before stalling out.")
        return out

    target_days = _task_target_days(task_name, liquidity)
    previous_days = int(last_valid_proposal.get("payment_days", buyer_days) if last_valid_proposal else buyer_days)
    escape_days = max(target_days, previous_days - (8 if "hard" in task_name.lower() else 5))
    proposal = _normalize_stage1_proposal(
        {
            "action_type": "propose",
            "deal_id": str(active_deal_id),
            "price": buyer_price,
            "payment_days": escape_days,
            "use_treds": bool(buyer_days > liquidity + 10),
            "reason": "Escape hatch: materially change terms instead of repeating the same proposal.",
        },
        observation,
        task_name,
        round_number,
        last_valid_proposal,
    )
    proposal["deal_id"] = str(active_deal_id)
    return proposal


def _normalize_liquidity_raw_action_payload(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
    round_number: int,
    last_valid_proposal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    open_deal_ids = list(observation.get("open_deal_ids") or [])
    active_deal_id = observation.get("active_deal_id") or (open_deal_ids[0] if open_deal_ids else None)
    buyer_days = int(observation.get("buyer_days", 0) or 0)
    liquidity = int(observation.get("liquidity_threshold", 0) or 0)
    buyer_price = float(observation.get("buyer_price", 0.0) or 0.0)
    cost = float(observation.get("cost_threshold", 0.0) or 0.0)

    out = dict(action_payload)
    out["action_type"] = str(out.get("action_type", "propose") or "propose").lower()
    if active_deal_id and out["action_type"] != "advance_period":
        out["deal_id"] = str(active_deal_id)

    if out["action_type"] == "reject" and not _explicit_surrender_requested(out.get("reason")):
        proposal = _normalize_stage1_proposal(
            {
                "action_type": "propose",
                "deal_id": str(active_deal_id),
                "price": float(out.get("price", buyer_price) or buyer_price),
                "payment_days": int(out.get("payment_days", max(liquidity, buyer_days - 6)) or max(liquidity, buyer_days - 6)),
                "use_treds": bool(out.get("use_treds", buyer_days > liquidity + 10)),
                "reason": _annotate_reason(
                    out.get("reason"),
                    "Reject suppressed: convert to a counter-offer instead of ending the episode.",
                ),
            },
            observation,
            task_name,
            round_number,
            last_valid_proposal,
        )
        proposal["deal_id"] = str(active_deal_id)
        return proposal

    if out["action_type"] == "accept":
        if last_valid_proposal is not None and _action_matches_last_proposal(out, last_valid_proposal):
            accepted = _build_accept_from_last_proposal(last_valid_proposal, observation, task_name)
            accepted["deal_id"] = str(active_deal_id)
            return accepted

        acceptable_buyer_terms = (
            _action_matches_observation_terms(out, observation)
            and buyer_days <= liquidity
            and buyer_price >= (cost + _task_margin(task_name))
        )
        if acceptable_buyer_terms:
            out["deal_id"] = str(active_deal_id)
            return out

        proposal = _normalize_stage1_proposal(
            {
                "action_type": "propose",
                "deal_id": str(active_deal_id),
                "price": float(out.get("price", buyer_price) or buyer_price),
                "payment_days": int(last_valid_proposal.get("payment_days", buyer_days) if last_valid_proposal else buyer_days),
                "use_treds": bool(out.get("use_treds", buyer_days > liquidity + 10)),
                "reason": _annotate_reason(out.get("reason"), "Invalid accept downgraded to a counter-offer."),
            },
            observation,
            task_name,
            round_number,
            last_valid_proposal,
        )
        proposal["deal_id"] = str(active_deal_id)
        return proposal

    if out["action_type"] == "tool":
        tool_name = str(out.get("tool_name", "") or "").upper()
        if tool_name not in {"QUERY_TREDS", "CHECK_COMPLIANCE", "RUN_CASHFLOW_SIM"}:
            return _build_liquidity_escape_action(observation, task_name, round_number, last_valid_proposal)
        out["tool_name"] = tool_name
        if tool_name == "QUERY_TREDS":
            out["tool_args"] = {"invoice_id": str(active_deal_id), "deal_id": str(active_deal_id)}
        elif tool_name == "CHECK_COMPLIANCE":
            out["tool_args"] = {"contract_id": str(active_deal_id), "deal_id": str(active_deal_id)}
        else:
            out["tool_args"] = {"plan": {"advance_periods": 1}, "horizon": 1, "deal_id": str(active_deal_id)}
        out["deal_id"] = str(active_deal_id)
        out["reason"] = _annotate_reason(out.get("reason"), "Tool call normalized for the liquidity workflow.")
        return out

    if out["action_type"] == "simulate_plan":
        out["deal_id"] = str(active_deal_id)
        if not isinstance(out.get("simulation_plan"), dict):
            out["simulation_plan"] = {"advance_periods": 1}
        if out.get("simulation_horizon") is None:
            out["simulation_horizon"] = 1
        out["reason"] = _annotate_reason(out.get("reason"), "Read-only plan simulation before further negotiation.")
        return out

    if out["action_type"] == "advance_period":
        return {"action_type": "advance_period", "reason": "No open deals remain in this macro period."}

    out = _normalize_stage1_proposal(out, observation, task_name, round_number, last_valid_proposal)
    out["deal_id"] = str(active_deal_id)
    return out


def _apply_liquidity_task_contract_pass(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    out = dict(action_payload)
    action_type = str(out.get("action_type", "propose") or "propose").lower()
    if action_type in {"propose", "accept"}:
        return _enforce_task_contract_fields(out, observation, task_name)
    if action_type in {"tool", "simulate_plan", "advance_period"}:
        out.pop("price", None)
        out.pop("payment_days", None)
        out.pop("use_treds", None)
        out.pop("propose_late_payment_penalty_clause", None)
        out.pop("propose_dynamic_discounting", None)
        out.pop("dynamic_discount_annual_rate", None)
    return out


def _apply_liquidity_close_or_escape_rewrite(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    history: List[dict],
    task_name: str,
    round_number: int,
    last_valid_proposal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if str(action_payload.get("action_type", "")).lower() == "advance_period":
        return {"action_type": "advance_period", "reason": "No open deals remain in this macro period."}

    previous_action = _parse_last_assistant_action(history)
    remaining = _remaining_rounds(observation, round_number)
    closeable = last_valid_proposal is not None and _should_close_deal(observation, task_name, round_number, last_valid_proposal)

    if closeable and remaining <= _late_close_window(task_name):
        accepted = _build_accept_from_last_proposal(last_valid_proposal, observation, task_name)
        accepted["deal_id"] = str(observation.get("active_deal_id") or accepted.get("deal_id") or "")
        return accepted

    if str(action_payload.get("action_type", "")).lower() == "propose":
        if _detect_repetition(history, window=3) or _same_action_shape(action_payload, previous_action):
            return _build_liquidity_escape_action(observation, task_name, round_number, last_valid_proposal)
        if remaining <= _late_close_window(task_name) and not closeable:
            return _build_liquidity_escape_action(observation, task_name, round_number, last_valid_proposal)

    return action_payload


def _build_liquidity_heuristic_action(
    observation: Dict[str, Any],
    task_name: str,
    round_number: int,
    last_valid_proposal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if _should_auto_advance_liquidity_period(observation, done=bool(observation.get("done", False))):
        return {"action_type": "advance_period", "reason": "No open deals remain in this macro period."}

    open_deal_ids = list(observation.get("open_deal_ids") or [])
    active_deal_id = observation.get("active_deal_id") or (open_deal_ids[0] if open_deal_ids else None)
    buyer_days = int(observation.get("buyer_days", 0) or 0)
    liquidity = int(observation.get("liquidity_threshold", 0) or 0)

    if observation.get("last_tool_name") != "QUERY_TREDS" and buyer_days > liquidity + 10 and round_number <= 1:
        return {
            "action_type": "tool",
            "deal_id": str(active_deal_id),
            "tool_name": "QUERY_TREDS",
            "tool_args": {"invoice_id": str(active_deal_id), "deal_id": str(active_deal_id)},
            "reason": "Deterministic heuristic: inspect TReDS before negotiating long tenor deals.",
        }

    if "hard" in task_name.lower() and round_number <= 2 and not bool((observation.get("metadata") or {}).get("simulation_projection_present", False)):
        return {
            "action_type": "simulate_plan",
            "deal_id": str(active_deal_id),
            "simulation_plan": {"advance_periods": 1},
            "simulation_horizon": 1,
            "reason": "Deterministic heuristic: simulate one macro period before continuing the hard task.",
        }

    if last_valid_proposal is not None and _should_close_deal(observation, task_name, round_number, last_valid_proposal):
        out = _build_accept_from_last_proposal(last_valid_proposal, observation, task_name)
        out["reason"] = _annotate_reason(out.get("reason"), "Deterministic heuristic: close the best valid proposal.")
        out["deal_id"] = str(active_deal_id)
        return out

    proposal = _normalize_stage1_proposal(
        {
            "action_type": "propose",
            "deal_id": str(active_deal_id),
            "price": float(observation.get("buyer_price", 0.0) or 0.0),
            "payment_days": max(_task_target_days(task_name, liquidity), buyer_days - (10 if "hard" in task_name.lower() else 6)),
            "use_treds": bool(buyer_days > liquidity + 10),
            "reason": "Deterministic heuristic liquidity action.",
        },
        observation,
        task_name,
        round_number,
        last_valid_proposal,
    )
    proposal["deal_id"] = str(active_deal_id)
    return proposal


def _normalize_liquidity_action_payload(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    history: List[dict],
    task_name: str,
    round_number: int,
    last_valid_proposal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if _should_auto_advance_liquidity_period(observation, done=bool(observation.get("done", False))):
        return {"action_type": "advance_period", "reason": "No open deals remain in this macro period."}
    raw_action = _normalize_liquidity_raw_action_payload(
        action_payload,
        observation,
        task_name,
        round_number,
        last_valid_proposal,
    )
    task_contract_action = _apply_liquidity_task_contract_pass(raw_action, observation, task_name)
    return _apply_liquidity_close_or_escape_rewrite(
        task_contract_action,
        observation,
        history,
        task_name,
        round_number,
        last_valid_proposal,
    )


def _should_auto_advance_liquidity_period(observation: Any, *, done: bool = False) -> bool:
    obs_dict = _observation_to_dict(observation) if not isinstance(observation, dict) else observation
    if done or list(obs_dict.get("open_deal_ids") or []):
        return False
    current_period = int(obs_dict.get("current_period", 0) or 0)
    total_periods = int(obs_dict.get("total_periods", 0) or 0)
    return current_period < total_periods


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


def _detect_repetition(history: List[Dict[str, Any]], window: int = 3) -> bool:
    """Return True if the last `window` assistant turns produced identical action JSON."""
    assistant_msgs = [m["content"] for m in history if m.get("role") == "assistant"]
    if len(assistant_msgs) < window:
        return False
    return len(set(assistant_msgs[-window:])) == 1


def _diversity_hint(observation: Dict[str, Any], task_name: str) -> str:
    """Generate an escape instruction when the LLM is stuck repeating the same action."""
    buyer_days = int(observation.get("buyer_days", 0))
    liq = int(observation.get("liquidity_threshold", 0))
    cost = float(observation.get("cost_threshold", 0.0))
    gap = buyer_days - liq
    alt_days = max(liq, buyer_days - 8)
    return (
        f"[STUCK] You have proposed the same terms 3 times. The buyer has NOT accepted. "
        f"Change strategy NOW: (1) try use_treds=true to reduce buyer day floor, "
        f"(2) propose payment_days={alt_days} (a different value), "
        f"(3) add propose_late_payment_penalty_clause=true. "
        f"Day gap remaining: {gap}. Cost floor: {cost:.2f}. Do NOT repeat the same JSON."
    )


def _urgency_hint(observation: Dict[str, Any], history: List[Dict[str, Any]], task_name: str) -> str:
    """Return urgency/diversity prefix to inject into the next user turn."""
    max_rounds = int(observation.get("max_rounds", 16))
    round_number = int(observation.get("round_number", 0))
    remaining = max_rounds - round_number
    parts: List[str] = []
    if remaining <= 3:
        parts.append(f"[URGENT] Only {remaining} rounds left. You MUST accept now or score=0.")
    elif remaining <= 6:
        parts.append(f"[WARNING] {remaining} rounds left. Accelerate concessions.")
    if _detect_repetition(history, window=3):
        parts.append(_diversity_hint(observation, task_name))
    return "\n".join(parts)


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
    urgency = _urgency_hint(observation, history, task_name)
    prefix = (urgency + "\n") if urgency else ""
    user_message = (
        f"{prefix}Task={task_name}\n"
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
        if "tool_name" in action:
            out["tool_name"] = action.get("tool_name")
        if "tool_args" in action:
            out["tool_args"] = action.get("tool_args")
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


def get_liquidity_agent_action(
    observation: Dict[str, Any],
    history: List[dict],
    task_name: str,
    history_summary: str = "",
) -> NegotiationAction:
    messages = _build_liquidity_router_messages(
        history=history,
        history_summary=history_summary,
        task_name=task_name,
        observation=observation,
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=_inference_llm_max_tokens(),
    )
    content = completion.choices[0].message.content
    return parse_liquidity_action(str(content or ""), observation)


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


def _trim_history_tail(history: List[dict]) -> List[dict]:
    max_turn_pairs = _inference_history_max_turns()
    max_chars = _inference_history_max_chars()
    candidate = list(history[-(max_turn_pairs * 2) :])
    trimmed: List[dict] = []
    remaining_chars = max_chars
    for message in reversed(candidate):
        role = str(message.get("role", "user") or "user")
        content = _strict_clip_text(message.get("content", ""), min(max_chars, remaining_chars))
        if not content:
            continue
        trimmed.append({"role": role, "content": content})
        remaining_chars -= len(content)
        if remaining_chars <= 0:
            break
    trimmed.reverse()
    return trimmed


def _roll_history_summary(existing_summary: str, line: str) -> str:
    max_chars = _inference_history_summary_max_chars()
    existing_lines = [item.strip() for item in existing_summary.splitlines() if item.strip()]
    candidate_lines = existing_lines + [_strict_clip_text(line, 240)]
    kept: List[str] = []
    total_chars = 0
    for item in reversed(candidate_lines):
        item_len = len(item) + (1 if kept else 0)
        if total_chars + item_len > max_chars:
            continue
        kept.append(item)
        total_chars += item_len
    kept.reverse()
    return "\n".join(kept)


def _update_liquidity_history_summary(
    existing_summary: str,
    *,
    before_observation: Dict[str, Any],
    after_observation: Dict[str, Any],
    action_payload: Mapping[str, Any],
) -> str:
    before_active = before_observation.get("active_deal_id")
    after_active = after_observation.get("active_deal_id")
    before_period = int(before_observation.get("current_period", 0) or 0)
    after_period = int(after_observation.get("current_period", before_period) or before_period)
    before_resolved = set(str(item) for item in (before_observation.get("resolved_deal_ids") or []))
    after_resolved = set(str(item) for item in (after_observation.get("resolved_deal_ids") or []))
    resolved_now = sorted(after_resolved - before_resolved)
    summary = existing_summary

    for deal_id in resolved_now:
        summary = _roll_history_summary(
            summary,
            (
                f"Resolved {deal_id} via {str(action_payload.get('action_type', 'propose')).lower()} "
                f"price={round(float(action_payload.get('price', 0.0) or 0.0), 2)} "
                f"days={int(action_payload.get('payment_days', 0) or 0)}"
            ),
        )
    if after_period > before_period:
        summary = _roll_history_summary(
            summary,
            (
                f"Advanced period {before_period}->{after_period}; resolved={len(after_resolved)} "
                f"defaulted={int((after_observation.get('metadata') or {}).get('defaulted_sme_count', 0) or 0)}"
            ),
        )
    elif before_active and after_active and before_active != after_active:
        summary = _roll_history_summary(summary, f"Switched active deal {before_active}->{after_active}.")
    return summary


def _build_liquidity_router_messages(
    *,
    history: List[dict],
    history_summary: str,
    task_name: str,
    observation: Dict[str, Any],
) -> List[dict]:
    user_message = (
        f"Task={task_name}\n"
        f"Current liquidity observation:\n{format_liquidity_observation(observation)}\n"
        "Return only JSON."
    )
    messages: List[dict] = [{"role": "system", "content": LIQUIDITY_SYSTEM_PROMPT}]
    if history_summary.strip():
        messages.append(
            {
                "role": "system",
                "content": "Rolling prior context summary:\n"
                + _strict_clip_text(history_summary, _inference_history_summary_max_chars()),
            }
        )
    messages.extend(_trim_history_tail(history))
    messages.append({"role": "user", "content": user_message})
    return messages


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
    if action_type not in {"propose", "accept", "reject", "simulate_plan", "advance_period", "tool"}:
        action_type = "propose"

    obs_dict = _observation_to_dict(observation) if isinstance(observation, dict) else None
    default_price = observation.buyer_price if obs_dict is None else obs_dict.get("buyer_price", 0.0)
    default_days = observation.buyer_days if obs_dict is None else obs_dict.get("buyer_days", 0)
    price = float(action_payload.get("price", default_price))
    payment_days = int(action_payload.get("payment_days", default_days))
    use_treds = bool(action_payload.get("use_treds", False))
    normalized_payload = dict(action_payload)
    normalized_payload["action_type"] = action_type
    normalized_payload["price"] = round(price, 2)
    normalized_payload["payment_days"] = payment_days
    normalized_payload["use_treds"] = use_treds
    normalized_payload["reason"] = _clip_ascii_text(
        action_payload.get("reason", "Model-selected action"),
        _MAX_REASON_CHARS,
    )
    return action_payload_to_model_action(normalized_payload, observation)


def _legacy_final_state_from_env(env: Any) -> Optional[Any]:
    inner_env = getattr(env, "_env", None)
    return getattr(inner_env, "state", None)


def _print_legacy_step_debug(
    *,
    observation: Any,
    reward: float,
    last_valid_proposal: Optional[dict[str, Any]],
) -> None:
    diagnostics = build_legacy_step_diagnostics(
        observation,
        reward=reward,
        last_valid_proposal=last_valid_proposal,
    )
    print(
        "[REWARD_DEBUG] "
        f"legacy_step_reward={diagnostics.legacy_step_reward:.4f} "
        f'legacy_reward_branch="{diagnostics.legacy_reward_branch}" '
        f"close_zone_flag={'true' if diagnostics.close_zone_flag else 'false'}",
        file=sys.stderr,
        flush=True,
    )


def _print_shadow_reward_summary(report_mode: str, report: Any, final_score: float) -> None:
    print(
        "[REWARD_SUMMARY] "
        f"legacy_terminal_score={final_score:.4f} "
        f"shadow_verifiable_reward={report.shadow_verifiable_reward:.4f} "
        f"shadow_total_sme_reward={report.shadow_total_sme_reward:.4f} "
        f"npv_delta_vs_baseline={report.npv_delta_vs_baseline:.4f} "
        f"effective_receivable_days={report.effective_receivable_days} "
        f"legal_max_payment_days={report.legal_max_payment_days} "
        f"compliance_within_legal_cap={'true' if report.compliance_within_legal_cap else 'false'} "
        f"compliance_with_penalty_exception={'true' if report.compliance_with_penalty_exception else 'false'}",
        file=sys.stderr,
        flush=True,
    )
    if report_mode == "legacy+full_debug":
        shaping = ",".join(f"{value:.4f}" for value in report.shadow_shaping_rewards) or "none"
        print(
            "[REWARD_DEBUG] "
            f"shadow_shaping_rewards={shaping} "
            f"shadow_shaping_total={report.shadow_shaping_total:.4f} "
            f"default_flag={'true' if report.default_flag else 'false'} "
            f"missed_supplier_payment={'true' if report.missed_supplier_payment else 'false'} "
            f"cash_balance={report.cash_balance:.2f} "
            f"required_minimum_cash={report.required_minimum_cash:.2f} "
            f"current_utilization={report.current_utilization:.2f} "
            f"credit_limit={report.credit_limit:.2f}",
            file=sys.stderr,
            flush=True,
        )


def _print_liquidity_step_debug(observation: Any) -> None:
    obs_dict = _observation_to_dict(observation)
    metadata = obs_dict.get("metadata") or {}
    print(
        "[LIQUIDITY_REWARD] "
        f"env_reward={float(obs_dict.get('reward', 0.0) or 0.0):.4f} "
        f"latest_shaping_reward={float(metadata.get('latest_shaping_reward', 0.0) or 0.0):.4f} "
        f"tool_bonus_applied={float(metadata.get('tool_bonus_applied', 0.0) or 0.0):.4f} "
        f"latest_verifiable_reward={float(metadata.get('latest_verifiable_reward', 0.0) or 0.0):.4f} "
        f"reward_branch={json.dumps(str(metadata.get('legacy_reward_branch') or ''))} "
        f"active_deal_id={json.dumps(obs_dict.get('active_deal_id'))} "
        f"current_period={obs_dict.get('current_period')}/{obs_dict.get('total_periods')}",
        file=sys.stderr,
        flush=True,
    )


def _coerce_reward_breakdown_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            coerced = to_dict()
        except Exception:
            return None
        return dict(coerced) if isinstance(coerced, Mapping) else None
    return None


def _reward_breakdown_terminal_total(reward_breakdown: Mapping[str, Any]) -> Optional[float]:
    keys = ("solvency", "liquidity", "npv", "compliance")
    if all(key in reward_breakdown for key in keys):
        return round(
            0.35 * float(reward_breakdown.get("solvency", 0.0) or 0.0)
            + 0.20 * float(reward_breakdown.get("liquidity", 0.0) or 0.0)
            + 0.35 * float(reward_breakdown.get("npv", 0.0) or 0.0)
            + 0.10 * float(reward_breakdown.get("compliance", 0.0) or 0.0),
            6,
        )
    if "terminal_component" in reward_breakdown:
        return float(reward_breakdown.get("terminal_component", 0.0) or 0.0)
    return None


def _extract_terminal_reward_breakdown(
    env: Any,
    final_metadata: Mapping[str, Any],
    *,
    canonical_total: float,
) -> Optional[Dict[str, Any]]:
    direct_value = _coerce_reward_breakdown_dict(getattr(env, "reward_breakdown", None))
    fallback_value = _coerce_reward_breakdown_dict(final_metadata.get("reward_breakdown"))
    for candidate in (direct_value, fallback_value):
        if not candidate:
            continue
        terminal_total = _reward_breakdown_terminal_total(candidate)
        if terminal_total is None:
            continue
        if abs(float(terminal_total) - float(canonical_total)) <= 1e-6:
            candidate = dict(candidate)
            candidate["total"] = float(canonical_total)
            return candidate
    return None


def _build_canonical_reward_component_report(
    episode_summary: Mapping[str, Any],
    final_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    base_rl_reward = float(episode_summary.get("base_rl_reward", 0.0) or 0.0)
    verifiable_reward = float(episode_summary.get("verifiable_reward", 0.0) or 0.0)
    tool_bonus_total = float(episode_summary.get("tool_bonus_total", 0.0) or 0.0)
    total_reward = float(episode_summary.get("total_reward", 0.0) or 0.0)
    metadata_report = _coerce_reward_breakdown_dict(final_metadata.get("reward_component_report")) or {}
    lambda_shaping = float(metadata_report.get("lambda_shaping", 0.1) or 0.1)
    shaping_total = (
        round((base_rl_reward - verifiable_reward) / lambda_shaping, 6)
        if lambda_shaping > 0.0
        else 0.0
    )
    report: Dict[str, Any] = {
        "verifiable_reward": round(verifiable_reward, 6),
        "shaping_total": shaping_total,
        "tool_bonus": round(tool_bonus_total, 6),
        "total_reward": round(total_reward, 6),
        "success_no_default_positive_npv": bool(episode_summary.get("success_no_default_positive_npv", False)),
        "lambda_shaping": lambda_shaping,
    }
    if "npv_delta_vs_baseline" in metadata_report:
        report["npv_delta_vs_baseline"] = float(metadata_report.get("npv_delta_vs_baseline", 0.0) or 0.0)
    return report


def _print_liquidity_episode_summary(env: Any) -> dict[str, Any]:
    summary = {}
    summarize = getattr(env, "summarize_episode", None)
    if callable(summarize):
        episode_summary = summarize()
        summary = {
            "base_rl_reward": float(episode_summary.base_rl_reward),
            "verifiable_reward": float(episode_summary.verifiable_reward),
            "total_reward": float(episode_summary.total_reward),
            "tool_bonus_total": float(episode_summary.tool_bonus_total),
            "env_reward_total": float(episode_summary.env_reward_total),
            "success_no_default_positive_npv": bool(episode_summary.success_no_default_positive_npv),
            "average_final_payment_days": float(episode_summary.average_final_payment_days),
            "tool_usage_count": int(episode_summary.tool_usage_count),
            "tool_call_count": int(episode_summary.tool_call_count),
            "tool_effective_count": int(episode_summary.tool_effective_count),
            "duplicate_tool_count": int(episode_summary.duplicate_tool_count),
            "invalid_action_count": int(episode_summary.invalid_action_count),
            "stall_step_count": int(episode_summary.stall_step_count),
            "resolved_deal_count": int(episode_summary.resolved_deal_count),
            "defaulted_sme_count": int(episode_summary.defaulted_sme_count),
            "terminated_by_step_cap": bool(episode_summary.terminated_by_step_cap),
        }
        print(
            "[LIQUIDITY_SUMMARY] "
            f"base_rl_reward={summary['base_rl_reward']:.4f} "
            f"verifiable_reward={summary['verifiable_reward']:.4f} "
            f"total_reward={summary['total_reward']:.4f} "
            f"tool_bonus_total={summary['tool_bonus_total']:.4f} "
            f"env_reward_total={summary['env_reward_total']:.4f} "
            f"success_no_default_positive_npv={'true' if summary['success_no_default_positive_npv'] else 'false'} "
            f"average_final_payment_days={summary['average_final_payment_days']:.2f} "
            f"tool_usage_count={summary['tool_usage_count']} "
            f"tool_call_count={summary['tool_call_count']} "
            f"tool_effective_count={summary['tool_effective_count']} "
            f"resolved_deal_count={summary['resolved_deal_count']} "
            f"defaulted_sme_count={summary['defaulted_sme_count']} "
            f"terminated_by_step_cap={'true' if summary['terminated_by_step_cap'] else 'false'}",
            file=sys.stderr,
            flush=True,
        )
    return summary


def _aggregate_liquidity_episode_summaries(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    summaries = [
        episode.get("episode_summary")
        for episode in episodes
        if isinstance(episode.get("episode_summary"), dict)
    ]
    if not summaries:
        return {}

    count = float(len(summaries))
    return {
        "avg_verifiable_reward": sum(float(item.get("verifiable_reward", 0.0) or 0.0) for item in summaries) / count,
        "avg_tool_bonus": sum(float(item.get("tool_bonus_total", 0.0) or 0.0) for item in summaries) / count,
        "avg_tool_call_count": sum(float(item.get("tool_call_count", 0.0) or 0.0) for item in summaries) / count,
        "avg_tool_effective_count": sum(float(item.get("tool_effective_count", 0.0) or 0.0) for item in summaries) / count,
        "avg_final_payment_days": sum(float(item.get("average_final_payment_days", 0.0) or 0.0) for item in summaries) / count,
        "avg_resolved_deal_count": sum(float(item.get("resolved_deal_count", 0.0) or 0.0) for item in summaries) / count,
        "default_rate": sum(1.0 if int(item.get("defaulted_sme_count", 0) or 0) > 0 else 0.0 for item in summaries) / count,
        "timeout_or_stepcap_rate": sum(
            1.0 if bool(item.get("terminated_by_step_cap", False)) else 0.0 for item in summaries
        ) / count,
    }


EnvClient = Union[SMENegotiatorEnv, InProcessSMENegotiatorBridge, InProcessLiquidityBridge]


async def run_episode(env: EnvClient, difficulty: str, seed: int) -> Dict[str, Any]:
    """Run one legacy episode using model-guided actions with strict stdout formatting."""

    task_name = difficulty.lower()
    episode_id = f"{task_name}-{seed}"
    history: List[dict] = []

    all_rewards: List[float] = []
    round_number = 0
    success = False
    forced_hard_accepts = 0
    result: Any = None
    observation: Any = None
    final_score = _strict_unit_interval(0.0)
    last_valid_proposal: Dict[str, Any] | None = None
    reward_mode = _inference_reward_mode()
    reset_observation: Any = None
    step_actions: List[NegotiationAction] = []
    step_observations: List[Any] = []
    shadow_summary: Optional[dict[str, Any]] = None

    try:
        result = await env.reset(seed=seed, difficulty=difficulty, episode_id=episode_id, task_name=task_name)
        observation = result.observation
        reset_observation = observation

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
            if _inference_agent_mode() == "heuristic":
                action_payload = choose_legacy_action(observation, round_number).model_dump()
            elif llm_blocked_402:
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
            step_actions.append(action)
            step_observations.append(observation)

            err_out = _format_step_error(llm_error)
            print(
                f'[STEP] step={round_number + 1} action={action_json} reward={_format_score_for_log(reward)} '
                f'step_reward_raw={reward:.4f} '
                f'done={"true" if done else "false"} error={err_out}',
                flush=True,
            )
            if reward_mode != "legacy":
                _print_legacy_step_debug(
                    observation=observation,
                    reward=reward,
                    last_valid_proposal=last_valid_proposal,
                )

            # Feed RLVR signal back so the LLM knows if it's in the close zone.
            new_obs_dict = _observation_to_dict(observation)
            buyer_days_new = int(new_obs_dict.get("buyer_days", 99))
            liq_new = int(new_obs_dict.get("liquidity_threshold", 60))
            obs_text = format_observation(obs_dict)
            if reward_mode != "legacy":
                close_zone = "yes" if buyer_days_new <= liq_new else "no"
                obs_text += (
                    f"\n[RLVR_SIGNAL] step_reward={reward:.3f} "
                    f"close_zone={close_zone} "
                    f"buyer_days_now={buyer_days_new} liq_target={liq_new}"
                )
            history.append({"role": "user", "content": obs_text})
            history.append({"role": "assistant", "content": json.dumps(action_payload, ensure_ascii=True)})

            round_number += 1

        final_score = _strict_unit_interval(float(result.reward or 0.0))
        meta = getattr(result.observation, "metadata", None) or {}
        if isinstance(meta, dict) and "success" in meta:
            success = bool(meta["success"])
        else:
            success = bool(result.done and final_score > 0.0)
    finally:
        total_reward = sum(all_rewards)
        shadow_report = None
        if reward_mode != "legacy" and reset_observation is not None:
            shadow_report = build_shadow_reward_report(
                reset_observation=reset_observation,
                actions=step_actions,
                step_observations=step_observations,
                seed=seed,
                final_state=_legacy_final_state_from_env(env),
            )
            shadow_summary = shadow_report.to_dict()
            _print_shadow_reward_summary(reward_mode, shadow_report, final_score)
        # Compute rule-based persona judge score from accumulated step log lines.
        judge_score: Optional[float] = None
        try:
            from rl.self_rewarding_dpo import build_rule_based_rubric_scorer  # type: ignore[import]
            from rl.rubrics import PERSONAS, persona_reward  # type: ignore[import]
            step_log_text = "\n".join(
                f"[STEP] step={i + 1} reward={_format_score_for_log(r)}"
                for i, r in enumerate(all_rewards)
            ) + f"\n[END] score={_format_score_for_log(final_score)}"
            _scorer = build_rule_based_rubric_scorer()
            _rubric_scores = _scorer(step_log_text)
            judge_score = max(persona_reward(p, _rubric_scores) for p in PERSONAS)
        except Exception:
            pass
        if shadow_report is not None:
            print(
                _format_terminal_reward_line(
                    verifiable_reward=shadow_report.shadow_verifiable_reward,
                    final_score=final_score,
                    success=success,
                    source="shadow_rlvr",
                ),
                flush=True,
            )
        print(_format_end_line(success, round_number, final_score, all_rewards, judge_score), flush=True)
        if all_rewards:
            print(f"[REWARD_CURVE] {_ascii_sparkline(all_rewards)}", flush=True)

    return {
        "difficulty": difficulty,
        "seed": seed,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps": round_number,
        "success": success,
        "forced_hard_accepts": forced_hard_accepts,
        "step_rewards": all_rewards,
        "reward_mode": reward_mode,
        "final_observation": _observation_to_dict(observation) if observation is not None else {},
        "shadow_reward_report": shadow_summary,
    }


async def run_liquidity_episode(env: InProcessLiquidityBridge, difficulty: str, seed: int) -> Dict[str, Any]:
    """Run one advanced in-process liquidity episode."""

    task_name = _liquidity_task_for_difficulty(difficulty)
    history: List[dict] = []
    history_summary = ""
    all_rewards: List[float] = []
    last_valid_proposal_by_deal: Dict[str, Dict[str, Any]] = {}
    result: Any = await env.reset(
        seed=seed,
        difficulty=difficulty.lower(),
        task_name=task_name,
        total_periods=int(os.getenv("INFERENCE_TOTAL_PERIODS", "3") or "3"),
    )
    observation: Any = result.observation
    round_number = 0
    skip_llm_after_402 = _env_truthy("INFERENCE_SKIP_LLM_AFTER_402", False)
    llm_blocked_402 = False
    agent_mode = _inference_agent_mode()

    print(
        f"[START] task={task_name} env=sme-liquidity-inprocess model={MODEL_NAME}",
        flush=True,
    )

    while not result.done:
        obs_dict = _observation_to_dict(observation)
        llm_error: Optional[str] = None
        previous_period = int(obs_dict.get("current_period", 0) or 0)
        active_deal_id = str(obs_dict.get("active_deal_id") or "") or None
        last_valid_proposal = last_valid_proposal_by_deal.get(active_deal_id) if active_deal_id else None

        if _should_auto_advance_liquidity_period(obs_dict, done=bool(result.done)):
            action_payload = _safe_liquidity_fallback_action(observation).model_dump()
        elif agent_mode == "heuristic":
            action_payload = _build_liquidity_heuristic_action(
                obs_dict,
                task_name,
                round_number,
                last_valid_proposal,
            )
        elif llm_blocked_402:
            action_payload = _safe_liquidity_fallback_action(observation).model_dump()
            llm_error = (
                "HF Inference 402 - further LLM calls skipped (INFERENCE_SKIP_LLM_AFTER_402=1). "
                "Resolve quota at https://huggingface.co/settings/billing or use local Ollama."
            )
        else:
            try:
                action_payload = get_liquidity_agent_action(
                    obs_dict,
                    history,
                    task_name,
                    history_summary,
                ).model_dump()
            except Exception as e:
                _maybe_print_hf_402_hint(e)
                print(
                    f"[ERROR] LLM call failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.warning(
                    "Liquidity LLM call failed; using fallback action: %s: %s",
                    type(e).__name__,
                    e,
                )
                if os.getenv("INFERENCE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes"):
                    logger.exception("LLM traceback (INFERENCE_DEBUG_LLM=1)")
                llm_error = str(e)
                if skip_llm_after_402 and _is_hf_inference_402(e):
                    llm_blocked_402 = True
                action_payload = _safe_liquidity_fallback_action(observation).model_dump()

        action_payload = _normalize_liquidity_action_payload(
            action_payload,
            obs_dict,
            history,
            task_name,
            round_number,
            last_valid_proposal,
        )
        action = _to_model_action(action_payload, observation)
        action_json = _serialize_step_action(action)
        if action.action_type == "propose":
            proposal_deal_id = str(action.deal_id or active_deal_id or "")
            if proposal_deal_id:
                last_valid_proposal_by_deal[proposal_deal_id] = action.model_dump()
        result = await env.step(action)
        observation = result.observation
        reward = float(result.reward or 0.0)
        all_rewards.append(reward)

        print(
            f'[STEP] step={round_number + 1} action={action_json} reward={_format_score_for_log(reward)} '
            f'step_reward_raw={reward:.4f} '
            f'done={"true" if bool(result.done) else "false"} error={_format_step_error(llm_error)}',
            flush=True,
        )
        _print_liquidity_step_debug(observation)
        new_obs_dict = _observation_to_dict(observation)
        current_period = int(new_obs_dict.get("current_period", previous_period) or previous_period)
        history_summary = _update_liquidity_history_summary(
            history_summary,
            before_observation=obs_dict,
            after_observation=new_obs_dict,
            action_payload=action.model_dump(),
        )
        resolved_after = set(str(item) for item in (new_obs_dict.get("resolved_deal_ids") or []))
        for deal_id in list(last_valid_proposal_by_deal):
            if deal_id in resolved_after:
                last_valid_proposal_by_deal.pop(deal_id, None)
        next_active_deal_id = str(new_obs_dict.get("active_deal_id") or "") or None
        if current_period > previous_period:
            last_valid_proposal_by_deal.clear()
            history = []
        elif active_deal_id and next_active_deal_id and next_active_deal_id != active_deal_id:
            last_valid_proposal_by_deal.clear()
            history = []
        if action.action_type == "advance_period" and current_period > previous_period:
            print(
                _format_period_summary_line(
                    closed_period=previous_period,
                    current_period=current_period,
                    total_periods=int(new_obs_dict.get("total_periods", current_period) or current_period),
                    resolved_deal_count=int(
                        (new_obs_dict.get("metadata") or {}).get(
                            "resolved_deal_count",
                            len(new_obs_dict.get("resolved_deal_ids") or []),
                        )
                    ),
                    defaulted_sme_count=int((new_obs_dict.get("metadata") or {}).get("defaulted_sme_count", 0) or 0),
                    cumulative_reward=sum(all_rewards),
                ),
                flush=True,
            )

        history.append({"role": "user", "content": format_liquidity_observation(obs_dict)})
        history.append(
            {
                "role": "assistant",
                "content": json.dumps(action.model_dump(), ensure_ascii=True, separators=(",", ":")),
            }
        )
        history = _trim_history_tail(history)
        round_number += 1

    final_score = _strict_unit_interval(float(result.reward or 0.0))
    episode_summary = _print_liquidity_episode_summary(env)
    episode_log = env.build_episode_log()
    success = bool(episode_summary.get("success_no_default_positive_npv", False))
    total_reward = sum(all_rewards)
    final_obs_dict = _observation_to_dict(observation) if observation is not None else {}
    final_metadata = final_obs_dict.get("metadata") or {}
    canonical_verifiable_reward = float(episode_summary.get("verifiable_reward", 0.0) or 0.0)
    reward_breakdown = _extract_terminal_reward_breakdown(
        env,
        final_metadata,
        canonical_total=canonical_verifiable_reward,
    )
    reward_breakdown_line = _format_verifiable_reward_breakdown_line(
        reward_breakdown,
        canonical_total=canonical_verifiable_reward,
    )
    reward_component_report = _build_canonical_reward_component_report(episode_summary, final_metadata)
    episode_summary["reward_component_report"] = reward_component_report
    episode_summary["shaping_total"] = float(reward_component_report.get("shaping_total", 0.0) or 0.0)
    episode_summary["termination_reason"] = str(final_metadata.get("termination_reason", "") or "")
    # Compute rule-based persona judge score.
    judge_score_liq: Optional[float] = None
    try:
        from rl.self_rewarding_dpo import build_rule_based_rubric_scorer  # type: ignore[import]
        from rl.rubrics import PERSONAS, persona_reward  # type: ignore[import]
        _scorer_liq = build_rule_based_rubric_scorer()
        _rubric_liq = _scorer_liq(episode_log or "")
        judge_score_liq = max(persona_reward(p, _rubric_liq) for p in PERSONAS)
    except Exception:
        pass
    print(reward_breakdown_line, flush=True)
    print(
        _format_terminal_reward_line(
            verifiable_reward=canonical_verifiable_reward,
            final_score=final_score,
            success=success,
            source="liquidity_verifiable",
        ),
        flush=True,
    )
    print(
        _format_end_line(
            success,
            round_number,
            final_score,
            all_rewards,
            judge_score_liq,
            termination_reason=str(final_metadata.get("termination_reason", "") or ""),
            defaulted_sme_count=int(final_metadata.get("defaulted_sme_count", 0) or 0),
        ),
        flush=True,
    )
    if all_rewards:
        print(f"[REWARD_CURVE] {_ascii_sparkline(all_rewards)}", flush=True)

    return {
        "difficulty": difficulty,
        "task_name": task_name,
        "seed": seed,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps": round_number,
        "success": success,
        "step_rewards": all_rewards,
        "episode_summary": episode_summary,
        "episode_log": episode_log,
        "final_observation": final_obs_dict,
        "termination_reason": str(final_metadata.get("termination_reason", "") or ""),
        "defaulted_sme_count": int(final_metadata.get("defaulted_sme_count", 0) or 0),
        "reward_component_report": reward_component_report,
    }


async def main() -> None:
    """Run three episodes per difficulty and write a compact results file."""

    env_mode = _inference_env_mode()
    requested_in_process = _openenv_in_process_enabled()
    effective_in_process = requested_in_process or env_mode == "liquidity"
    print(f"[CONFIG] LLM API_BASE_URL={API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[MODE] {_runtime_banner(env_mode)}", file=sys.stderr, flush=True)
    print(f"[CONFIG] INFERENCE_ENV_MODE={env_mode}", file=sys.stderr, flush=True)
    print(f"[CONFIG] INFERENCE_REWARD_MODE={_inference_reward_mode()}", file=sys.stderr, flush=True)
    print(f"[CONFIG] INFERENCE_AGENT_MODE={_inference_agent_mode()}", file=sys.stderr, flush=True)
    print(f"[CONFIG] INFERENCE_HISTORY_MAX_TURNS={_inference_history_max_turns()}", file=sys.stderr, flush=True)
    print(f"[CONFIG] INFERENCE_HISTORY_MAX_CHARS={_inference_history_max_chars()}", file=sys.stderr, flush=True)
    print(
        f"[CONFIG] INFERENCE_HISTORY_SUMMARY_MAX_CHARS={_inference_history_summary_max_chars()}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[CONFIG] INFERENCE_LLM_MAX_TOKENS={_inference_llm_max_tokens()}", file=sys.stderr, flush=True)
    print(f"[CONFIG] OPENENV_IN_PROCESS_REQUESTED={'1' if requested_in_process else '0'}", file=sys.stderr, flush=True)
    print(f"[CONFIG] OPENENV_IN_PROCESS_EFFECTIVE={'1' if effective_in_process else '0'}", file=sys.stderr, flush=True)
    if env_mode == "liquidity":
        print(
            f"[CONFIG] INFERENCE_LIQUIDITY_TASK_HINT={_liquidity_task_for_difficulty('MEDIUM')}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[CONFIG] INFERENCE_TOTAL_PERIODS={int(os.getenv('INFERENCE_TOTAL_PERIODS', '3') or '3')}",
            file=sys.stderr,
            flush=True,
        )
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
            "openenv_in_process": effective_in_process,
            "openenv_in_process_requested": requested_in_process,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "inference_env_mode": env_mode,
            "inference_reward_mode": _inference_reward_mode(),
            "inference_agent_mode": _inference_agent_mode(),
            "inference_history_max_turns": _inference_history_max_turns(),
            "inference_history_max_chars": _inference_history_max_chars(),
            "inference_history_summary_max_chars": _inference_history_summary_max_chars(),
            "inference_llm_max_tokens": _inference_llm_max_tokens(),
        },
        "tasks": {},
    }

    if env_mode == "liquidity":
        if not requested_in_process:
            print(
                "[WARN] Liquidity inference runs in-process only because the live OpenEnv server still exposes the "
                "legacy single-deal environment.",
                file=sys.stderr,
                flush=True,
            )
        env_manager = InProcessLiquidityBridge()
    elif requested_in_process:
        env_manager = InProcessSMENegotiatorBridge()
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
            if _inference_env_mode() == "liquidity":
                assert isinstance(env, InProcessLiquidityBridge)
                episode_results.append(await run_liquidity_episode(env, difficulty, seed))
            else:
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
        liquidity_metrics = _aggregate_liquidity_episode_summaries(episode_results)
        if liquidity_metrics:
            results["tasks"][difficulty]["summary"].update(liquidity_metrics)

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
    all_episodes = [
        episode
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    liquidity_metrics = _aggregate_liquidity_episode_summaries(all_episodes)
    if liquidity_metrics:
        results["summary"].update(liquidity_metrics)

    with open("inference_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
