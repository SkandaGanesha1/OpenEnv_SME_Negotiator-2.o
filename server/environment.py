"""OpenEnv environment implementation for SME payment-term negotiation.

The canonical simulator lives in ``SMENegotiatorEnvironment``. It owns the
episode state machine, derives observations from internal state, applies
per-step shaping rewards for ongoing negotiations, and delegates terminal task
scoring to the pure functions in ``sme_negotiator_env.graders``.
"""

from __future__ import annotations

import json
import os
import sys
import math
from dataclasses import replace
from datetime import datetime, timezone
from random import Random
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

from openenv.core import Environment

from sme_negotiator_env.action_validator import ActionValidator
from sme_negotiator_env.process_supervisor import ProcessSupervisor
from sme_negotiator_env.graders import (
    TASK_GRADERS,
    compute_legacy_terminal_breakdown,
    compute_reward_component_report,
    compute_shaping_rewards,
    compute_tool_use_bonus,
    compute_verifiable_reward,
    grade_task_payment_terms_medium,
)
from sme_negotiator_env.problem_context import RAZORPAY_ITCH_BLURB
from sme_negotiator_env.models import (
    BuyerState,
    CashflowProjection,
    DealState,
    FinancierState,
    HistoryEvent,
    LiquidityEnvironmentState,
    LiquidityObservation,
    NegotiationAction,
    NegotiationObservation,
    NegotiationState,
    RewardComponentReport,
    SMEAccountState,
    ToolCallRecord,
    ToolResultEnvelope,
    WorldState,
    WorldSnapshot,
    default_negotiation_state,
)
from sme_negotiator_env.simulation import advance_world_state, apply_plan_to_world_state, simulate_cashflow
from sme_negotiator_env.task_config import TaskConfig, resolve_task_id, TASK_REGISTRY
from sme_negotiator_env.tools import run_cashflow_sim
from sme_negotiator_env.tool_backends import BaseToolBackend, LiveToolAdapter, build_tool_backend_registry

if TYPE_CHECKING:
    from rl.opponents import FinancierPolicy, FinancierQuote, TextPolicy


_STRICT_EPS = 1e-6


def _strict_unit_interval(score: float) -> float:
    """Map terminal scores into the strict open interval (0, 1)."""
    value = float(score)
    if not math.isfinite(value):
        return _STRICT_EPS
    return float(min(1.0 - _STRICT_EPS, max(_STRICT_EPS, value)))


class SMENegotiatorEnvironment(Environment):
    """Single-agent negotiation environment for SME vs buyer payment terms.

    One episode models a single negotiation over a bounded number of rounds.
    The environment keeps the full internal ``NegotiationState`` and exposes a
    structured ``NegotiationObservation`` after each transition. Rewards come
    from two sources:
    - ongoing shaping in ``_compute_reward`` while the negotiation is active
    - task-specific terminal grading via ``sme_negotiator_env.graders`` when the
      episode ends by agreement, rejection, or max rounds
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        buyer_policy: Optional["TextPolicy"] = None,
        buyer_policy_role: str = "buyer",
    ) -> None:
        self._rng = Random()
        self._task_config: TaskConfig = TASK_REGISTRY["payment-terms-medium"]
        self._seed = 1000
        self._base_concede = 0.02
        self._buyer_min_days_floor = 30
        self._buyer_price = 100.0
        self._buyer_days = 90
        self._initial_buyer_price = 100.0
        self._initial_buyer_days = 90
        self._deal_reached = False
        self._final_price: Optional[float] = None
        self._final_days: Optional[int] = None
        self._treds_used = False
        self._cumulative_reward = 0.0
        self._state: Optional[NegotiationState] = None
        # Last SME counter-offer (used to score accept actions even if the model echoes buyer days)
        self._last_sme_proposed_days: Optional[int] = None
        self._last_sme_proposed_price: Optional[float] = None
        self._buyer_policy = buyer_policy
        self._buyer_policy_role = str(buyer_policy_role or "buyer")
        # Phase 7: per-episode validator, process supervisor, mutation lock
        self._validator: ActionValidator = ActionValidator()
        self._supervisor: Optional[ProcessSupervisor] = None
        self._step_in_progress: bool = False
        self._step_penalties: list[float] = []

    @property
    def state(self) -> Optional[NegotiationState]:
        """Current full internal state for OpenEnv serialization and debugging."""

        return self._state

    def get_state(self) -> Optional[NegotiationState]:
        """Backward-compatible accessor for direct Python callers."""

        return self._state

    def state_info(self) -> dict[str, Any]:
        """Return a structured dict snapshot of current episode state.

        Satisfies the OpenEnv first-class state() contract. Callers that only
        need the raw NegotiationState should use get_state() / state instead.
        """
        if self._state is None:
            return {"episode_id": None, "step_count": 0, "deal_reached": False,
                    "history": [], "cumulative_reward": 0.0}
        history = []
        for evt in getattr(self._state, "history_events", []):
            if hasattr(evt, "model_dump"):
                history.append(evt.model_dump())
            elif hasattr(evt, "dict"):
                history.append(evt.dict())
            else:
                history.append(str(evt))
        return {
            "episode_id": self._state.episode_id,
            "step_count": int(self._state.step_count),
            "deal_reached": bool(self._state.deal_reached),
            "history": history,
            "cumulative_reward": round(float(self._state.cumulative_reward), 6),
            "task_name": self._task_config.name,
            "difficulty": self._task_config.difficulty,
        }

    def _now_utc_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_buyer_observation(
        self,
        *,
        action: NegotiationAction,
        proposed_price: float,
        proposed_days: int,
        price_drop: float,
        day_drop: int,
    ) -> NegotiationObservation:
        """Build a buyer-view observation for optional self-play buyer policies."""
        assert self._state is not None
        buyer_context = {
            "current_buyer_price": round(float(self._buyer_price), 2),
            "current_buyer_days": int(self._buyer_days),
            "proposed_price": round(float(proposed_price), 2),
            "proposed_days": int(proposed_days),
            "buyer_min_days_floor": int(self._buyer_min_days_floor),
            "price_drop": round(float(price_drop), 6),
            "day_drop": int(day_drop),
            "cost_threshold": round(float(self._task_config.cost_threshold), 2),
            "treds_used": bool(action.use_treds),
        }
        return self._obs_from_state(
            buyer_accepted=False,
            negotiation_done=False,
            step_reward=0.0,
            message="Buyer viewpoint. Respond to the supplier proposal.\nBUYER_CONTEXT=" + json.dumps(
                buyer_context,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            reward=0.0,
            done=False,
            metadata={"view_role": self._buyer_policy_role},
        )

    def _heuristic_buyer_action(
        self,
        *,
        action: NegotiationAction,
        proposed_price: float,
        proposed_days: int,
        price_drop: float,
        day_drop: int,
    ) -> NegotiationAction:
        """Return the current buyer heuristic as a structured counter-action."""
        next_price = round(
            max(self._task_config.cost_threshold, min(self._buyer_price, proposed_price) - price_drop),
            2,
        )
        next_days = max(self._buyer_min_days_floor, self._buyer_days - day_drop)
        return NegotiationAction(
            action_type="propose",
            price=next_price,
            payment_days=int(next_days),
            use_treds=bool(action.use_treds),
            reason="Heuristic buyer counter-offer",
        )

    def _apply_buyer_action(
        self,
        buyer_action: NegotiationAction,
        *,
        originating_action: NegotiationAction,
        proposed_price: float,
        proposed_days: int,
        prior_buyer_price: float,
        prior_buyer_days: int,
    ) -> NegotiationObservation:
        """Apply a buyer counter-action while preserving the legacy reward path."""
        assert self._state is not None

        buyer_action_type = str(buyer_action.action_type).lower()
        if buyer_action_type == "accept":
            agreed_price = proposed_price
            agreed_days = proposed_days
            self._deal_reached = True
            self._final_price = agreed_price
            self._final_days = agreed_days
            self._treds_used = bool(originating_action.use_treds) or self._treds_used
            self._apply_terminal_outcome_to_state(agreed_price, agreed_days, originating_action)
            terminal_reward = min(self._terminal_reward(), 0.99)
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Buyer accepted the supplier proposal"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            self._reward_debug_print("buyer_policy_accept", terminal_reward)
            return self._obs_from_state(
                buyer_accepted=True,
                negotiation_done=True,
                step_reward=terminal_reward,
                message="Buyer accepted the supplier proposal",
                reward=terminal_reward,
                done=True,
                metadata=self._episode_meta(
                    "buyer_policy_accept",
                    success=True,
                    termination_reason="buyer_policy_accept",
                ),
            )

        if buyer_action_type == "reject":
            terminal_reward = _strict_unit_interval(0.0)
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Buyer rejected the supplier proposal"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            self._reward_debug_print("buyer_policy_reject", terminal_reward)
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=terminal_reward,
                message="Buyer rejected the supplier proposal",
                reward=terminal_reward,
                done=True,
                metadata=self._episode_meta(
                    "buyer_policy_reject",
                    success=False,
                    termination_reason="buyer_policy_reject",
                ),
            )

        self._buyer_price = round(
            max(self._task_config.cost_threshold, min(float(self._buyer_price), float(buyer_action.price))),
            2,
        )
        self._buyer_days = max(self._buyer_min_days_floor, min(int(self._buyer_days), int(buyer_action.payment_days)))
        self._state.step_count += 1
        self._state.negotiation_round = self._state.step_count

        _use_v2_reward = os.getenv("SME_REWARD_V2", "0").strip() not in ("0", "false", "False")
        if _use_v2_reward:
            step_reward, reward_branch = self._compute_reward_v2(
                proposed_price,
                proposed_days,
                current_buyer_price=prior_buyer_price,
                current_buyer_days=prior_buyer_days,
                used_tool_this_step=bool(originating_action.use_treds),
            )
        else:
            step_reward, reward_branch = self._compute_reward(
                proposed_price,
                proposed_days,
                current_buyer_price=prior_buyer_price,
                current_buyer_days=prior_buyer_days,
            )

        self._cumulative_reward += step_reward
        self._state.cumulative_reward = self._cumulative_reward
        self._state.buyer_price = self._buyer_price
        self._state.buyer_days = self._buyer_days
        # Build an informative counter-offer message the LLM can read next turn.
        buyer_days_delta = prior_buyer_days - self._buyer_days
        tc = self._task_config
        buyer_msg = (
            f"Buyer countered: days={self._buyer_days} "
            f"(was {prior_buyer_days}, conceded {buyer_days_delta} days), "
            f"price={self._buyer_price:.2f} (was {prior_buyer_price:.2f}). "
            f"To close: payment_days<={tc.liquidity_threshold}, price>={tc.cost_threshold:.2f}."
        )
        self._state.message = buyer_msg
        self._reward_debug_print(reward_branch, step_reward)

        done = self._check_done()
        termination_reason = "ongoing"
        success = False
        if done:
            self._state.message = "Maximum rounds reached — no agreement"
            terminal_reward = self._terminal_reward()
            self._cumulative_reward = self._cumulative_reward - step_reward + terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            step_reward = terminal_reward
            reward_branch = f"max_rounds_terminal_grader:{terminal_reward:.4f}"
            self._reward_debug_print(reward_branch, step_reward)
            termination_reason = "max_rounds_no_deal"

        return self._obs_from_state(
            buyer_accepted=False,
            negotiation_done=done,
            step_reward=step_reward,
            message=self._state.message,
            reward=step_reward,
            done=done,
            metadata=self._episode_meta(
                reward_branch,
                success=success,
                termination_reason=termination_reason,
            ),
        )

    def _grader_fn(self):
        """Return the deterministic terminal grader for the active task."""
        return TASK_GRADERS.get(self._task_config.grader_id, grade_task_payment_terms_medium)

    def _apply_terminal_outcome_to_state(
        self,
        agreed_price: float,
        agreed_days: int,
        action: NegotiationAction,
    ) -> None:
        """Persist the final agreed terms onto the internal state before grading."""
        assert self._state is not None
        self._state.deal_reached = True
        self._state.final_price = agreed_price
        self._state.final_days = agreed_days
        self._state.agreed_terms = int(agreed_days)
        self._state.treds_used = bool(action.use_treds) or self._state.treds_used
        self._state.late_payment_penalty_agreed = bool(action.propose_late_payment_penalty_clause)
        self._state.dynamic_discounting_agreed = bool(action.propose_dynamic_discounting)
        self._state.agreed_dynamic_discount_annual = float(action.dynamic_discount_annual_rate)

    def _terminal_reward(self) -> float:
        """Compute the task's terminal score from the current internal state."""
        assert self._state is not None
        return _strict_unit_interval(float(self._grader_fn()(self._state)))

    def _compute_reward(
        self,
        proposed_price: float,
        proposed_days: int,
        *,
        current_buyer_price: float,
        current_buyer_days: int,
    ) -> Tuple[float, str]:
        """Compute the non-terminal shaping reward for an in-progress negotiation.

        This function scores only the current proposal relative to the buyer's
        current terms and the task's liquidity/cost constraints. It does not
        replace the task-level terminal grader. When an episode ends due to max
        rounds, the terminal score replaces the last partial reward.
        """

        pct = self._task_config
        cost = float(pct.cost_threshold)
        liq = int(pct.liquidity_threshold)
        init_p = float(self._initial_buyer_price)
        cur_d = int(current_buyer_days)

        if proposed_price < cost:
            return 0.12, "invalid_below_cost"

        if proposed_days > cur_d:
            return 0.1, "no_progress_days_worse_than_buyer"

        # Baseline-anchored progress avoids denominator-shrink artifacts across rounds.
        days_span = max(1, int(self._initial_buyer_days) - liq)
        days_improve = max(0.0, min(1.0, float(int(self._initial_buyer_days) - proposed_days) / float(days_span)))

        price_span = max(1e-9, init_p - cost)
        price_improve = max(0.0, min(1.0, (float(proposed_price) - cost) / price_span))

        improvement = max(0.0, min(1.0, 0.65 * days_improve + 0.35 * price_improve))

        if improvement < 1e-6:
            return 0.08, "no_progress_negligible"

        raw = 0.2 + 0.6 * improvement
        partial = min(0.3, raw * 0.3)
        detail = (
            f"improvement={improvement:.3f}|days_baseline={days_improve:.3f}|price_baseline={price_improve:.3f}"
        )
        return round(partial, 4), f"partial_progress:{detail}"

    def _compute_reward_v2(
        self,
        proposed_price: float,
        proposed_days: int,
        *,
        current_buyer_price: float,
        current_buyer_days: int,
        used_tool_this_step: bool = False,
    ) -> Tuple[float, str]:
        """V2 shaping: convergence-toward-agreement reward. Activated by SME_REWARD_V2=1.

        Unlike V1 which measures SME profitability (penalising price concessions),
        V2 measures how close the current proposal is to closing the deal. This
        produces a naturally increasing reward signal as negotiation converges —
        a cleaner gradient for RL training.
        """
        pct = self._task_config
        cost = float(pct.cost_threshold)
        liq = int(pct.liquidity_threshold)

        if proposed_price < cost:
            return 0.05, "v2_below_cost"
        if proposed_days > current_buyer_days:
            return 0.05, "v2_days_worse_than_buyer"

        max_days = max(1, int(self._initial_buyer_days))
        days_span = max(1, max_days - liq)
        # days_proximity: 0.0 when days=max_days (start), 1.0 when days=liq (goal)
        days_proximity = max(0.0, min(1.0, (max_days - proposed_days) / days_span))

        # Gap bonus: reward closing the gap between SME proposal and buyer counter-offer
        day_gap = max(0, proposed_days - current_buyer_days)
        gap_fraction = 1.0 - min(1.0, day_gap / max(1, days_span))
        gap_bonus = 0.15 * gap_fraction

        combined = 0.70 * days_proximity + 0.15 * gap_bonus
        shaped = 0.10 + 0.60 * combined  # range: [0.10, 0.70]

        if used_tool_this_step:
            shaped = min(0.75, shaped + 0.05)

        # Alignment bonus: proposals at the goal zone with no gap earn extra
        if day_gap == 0 and proposed_days <= liq:
            shaped = min(0.75, shaped + 0.10)

        detail = (
            f"days_prox={days_proximity:.3f}|gap_bonus={gap_bonus:.3f}"
            f"|combined={combined:.3f}|tool={used_tool_this_step}"
        )
        return round(min(0.75, shaped), 4), f"v2_convergence:{detail}"

    def _reward_debug_print(self, branch: str, step_reward: float) -> None:
        """Emit reward branch diagnostics when ``REWARD_DEBUG`` is enabled."""
        if os.getenv("REWARD_DEBUG", "0").strip() not in ("0", "false", "False", "no", "No"):
            print(
                f"[REWARD_DEBUG] branch={branch} step_reward={step_reward:.4f}",
                file=sys.stderr,
                flush=True,
            )

    def _check_done(self) -> bool:
        """Return ``True`` when the current episode has reached a terminal state."""
        assert self._state is not None
        if self._deal_reached:
            return True
        if self._state.step_count >= self._task_config.max_rounds:
            return True
        return False

    def _episode_meta(
        self,
        reward_branch: str,
        *,
        success: bool,
        termination_reason: str,
    ) -> dict[str, object]:
        """Build the metadata payload attached to observations for this episode."""
        assert self._state is not None
        return {
            "episode_id": self._state.episode_id,
            "task_name": self._task_config.name,
            "reward_branch": reward_branch,
            "success": success,
            "termination_reason": termination_reason,
        }

    def _obs_from_state(
        self,
        *,
        buyer_accepted: bool,
        negotiation_done: bool,
        step_reward: float,
        message: str,
        reward: float,
        done: bool,
        metadata: dict[str, object],
    ) -> NegotiationObservation:
        """Project the current internal state into the public observation schema."""
        assert self._state is not None
        tc = self._task_config
        return NegotiationObservation(
            round_number=self._state.negotiation_round,
            max_rounds=tc.max_rounds,
            buyer_price=self._buyer_price,
            buyer_days=self._buyer_days,
            buyer_accepted=buyer_accepted,
            negotiation_done=negotiation_done,
            cost_threshold=tc.cost_threshold,
            liquidity_threshold=tc.liquidity_threshold,
            volume=tc.volume,
            difficulty=tc.difficulty,
            price_score=0.0,
            days_score=0.0,
            treds_bonus=0.15 if self._treds_used else 0.0,
            step_reward=step_reward,
            message=message,
            reward=reward,
            done=done,
            metadata=metadata,
            task_name=tc.name,
            sme_monthly_revenue=tc.sme_monthly_revenue,
            working_capital_gap=self._state.working_capital_gap,
            interest_rate_annual=tc.interest_rate_annual,
            buyer_power_score=tc.buyer_power_score,
            secondary_buyer_power=tc.secondary_buyer_power,
            current_payment_terms_days=tc.current_payment_terms_days,
            sme_supplier_payment_days=tc.sme_supplier_payment_days,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "MEDIUM",
        **kwargs,
    ) -> NegotiationObservation:
        """Reset the environment for a new negotiation episode.

        The negotiation dynamics are deterministic for a given seed and task
        selection. As part of the baseline behavior, the human-readable message
        includes a live UTC timestamp from ``_now_utc_iso()``; Stage 0
        documents that behavior but intentionally does not change it.
        """

        task_config_override = kwargs.get("task_config_override")
        if isinstance(task_config_override, TaskConfig):
            self._task_config = task_config_override
        else:
            requested_task = kwargs.get("task_name") or kwargs.get("task")
            task_id = resolve_task_id(
                str(requested_task) if requested_task else None,
                difficulty=difficulty,
            )
            self._task_config = TASK_REGISTRY[task_id]
        tc = self._task_config

        self._seed = int(seed if seed is not None else 1000)
        self._rng = Random(self._seed)
        self._base_concede = self._rng.uniform(tc.concede_low, tc.concede_high)
        self._buyer_min_days_floor = tc.day_floor
        self._buyer_price = tc.initial_buyer_price
        self._buyer_days = tc.initial_buyer_days
        self._initial_buyer_price = tc.initial_buyer_price
        self._initial_buyer_days = tc.initial_buyer_days
        self._deal_reached = False
        self._final_price = None
        self._final_days = None
        self._treds_used = False
        self._cumulative_reward = 0.0
        self._last_sme_proposed_days = None
        self._last_sme_proposed_price = None

        episode_id = str(kwargs.get("episode_id") or f"{tc.difficulty}_{self._seed}")

        # Reset per-episode validation and supervision state
        self._validator.reset()
        self._step_in_progress = False
        self._step_penalties = []
        self._supervisor = ProcessSupervisor(
            target_days=int(tc.liquidity_threshold),
            difficulty=tc.difficulty,
        )

        self._state = default_negotiation_state(
            episode_id=episode_id,
            seed=self._seed,
            difficulty=tc.difficulty,
            task_name=tc.name,
            max_steps=tc.max_rounds,
            max_rounds=tc.max_rounds,
            buyer_price=self._buyer_price,
            buyer_days=self._buyer_days,
            initial_buyer_days=self._initial_buyer_days,
            cost_threshold=tc.cost_threshold,
            liquidity_threshold=tc.liquidity_threshold,
            volume=tc.volume,
            sme_monthly_revenue=tc.sme_monthly_revenue,
            current_payment_terms_days=tc.current_payment_terms_days,
            sme_supplier_payment_days=tc.sme_supplier_payment_days,
            interest_rate_annual=tc.interest_rate_annual,
            buyer_power_score=tc.buyer_power_score,
            secondary_buyer_power=tc.secondary_buyer_power,
            # Preserve the current reset message shape, including the live UTC
            # timestamp, because downstream tooling may inspect it informally.
            message=(
                f"{RAZORPAY_ITCH_BLURB} "
                f"Task: {tc.description} "
                f"{tc.context_note} "
                f"| Episode reset @ {self._now_utc_iso()} (task_id={tc.name}, base_concede={self._base_concede:.4f})"
            ),
        )

        # Seed validator with initial buyer offer so accept can verify against it at round 0
        self._validator.record_buyer_offer(episode_id, self._buyer_price, self._buyer_days)

        msg = self._state.message
        return self._obs_from_state(
            buyer_accepted=False,
            negotiation_done=False,
            step_reward=0.0,
            message=msg,
            reward=0.0,
            done=False,
            metadata={
                "episode_id": episode_id,
                "seed": self._seed,
                "base_concede": self._base_concede,
                "buyer_day_floor": self._buyer_min_days_floor,
                "task_name": tc.name,
                "task_description": tc.description,
                "context_note": tc.context_note,
            },
        )

    # Higher buyer power reduces the buyer's counter-offer concession size.
    def _buyer_counter_power_multiplier(self) -> float:
        """Higher buyer power → smaller concessions."""
        p = float(self._task_config.buyer_power_score)
        return max(0.2, 1.0 - 0.55 * p)

    def step(self, action: NegotiationAction, **kwargs) -> NegotiationObservation:
        """Apply one SME action and advance the negotiation state machine.

        ``propose`` updates the buyer counter-offer and yields shaping reward.
        ``accept`` and ``reject`` terminate the episode immediately and yield a
        terminal reward path. If max rounds are reached without agreement, the
        final observation is marked done and the terminal grader replaces the
        last shaping reward.
        """

        if self._state is None:
            self.reset(seed=self._seed, difficulty=self._task_config.difficulty)

        assert self._state is not None
        tc = self._task_config
        power_mult = self._buyer_counter_power_multiplier()

        if self._deal_reached or self._state.step_count >= tc.max_rounds:
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=0.0,
                message="Episode already completed",
                reward=0.0,
                done=True,
                metadata=self._episode_meta(
                    "already_completed",
                    success=False,
                    termination_reason="already_completed",
                ),
            )

        action_type = str(action.action_type).lower()

        # --- Mutation lock: detect re-entrant step() calls ---
        if self._step_in_progress:
            raise RuntimeError(
                "Re-entrant step() call detected — SMENegotiatorEnvironment is not thread-safe. "
                "Use separate environment instances for concurrent episodes."
            )
        self._step_in_progress = True

        # --- Anti-cheat / anti-hack validation (supplements existing checks) ---
        _validator_penalty = 0.0
        deal_id = self._state.episode_id
        if action_type in ("propose", "accept", "tool", "advance_period"):
            _vresult = self._validator.validate(action, deal_id=deal_id)
            if not _vresult.is_valid or _vresult.penalty != 0.0:
                _validator_penalty = float(_vresult.penalty)
                self._step_penalties.append(_validator_penalty)
                # Hard violation on invalid_accept: terminate now with penalty
                # (The existing accept validation below will also catch mismatches;
                # this path handles cases where the validator fires first.)
                if _vresult.should_terminate and _vresult.violation_type == "invalid_accept":
                    terminal_reward = max(
                        _strict_unit_interval(0.0),
                        _strict_unit_interval(0.0) + _validator_penalty,
                    )
                    terminal_reward = _strict_unit_interval(max(0.0, terminal_reward))
                    self._state.step_count += 1
                    self._state.negotiation_round = self._state.step_count
                    self._cumulative_reward += terminal_reward
                    self._state.cumulative_reward = self._cumulative_reward
                    self._state.message = _vresult.message
                    self._step_in_progress = False
                    return self._obs_from_state(
                        buyer_accepted=False,
                        negotiation_done=True,
                        step_reward=terminal_reward,
                        message=_vresult.message,
                        reward=terminal_reward,
                        done=True,
                        metadata=self._episode_meta(
                            "validator_invalid_accept",
                            success=False,
                            termination_reason="validator_invalid_accept",
                        ),
                    )

        if action_type in {"simulate_plan", "advance_period", "tool"}:
            self._step_in_progress = False
            terminal_reward = _strict_unit_interval(0.0)
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = f"Action type '{action_type}' is not supported by the legacy environment"
            self._reward_debug_print("unsupported_action_type", terminal_reward)
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=terminal_reward,
                message=self._state.message,
                reward=terminal_reward,
                done=True,
                metadata=self._episode_meta(
                    "unsupported_action_type",
                    success=False,
                    termination_reason="unsupported_action_type",
                ),
            )
        proposed_price = float(action.price)
        proposed_days = int(action.payment_days)
        use_treds = bool(action.use_treds)
        step_reward = 0.0
        message = "Buyer countered"

        if use_treds:
            reduction = self._rng.randint(5, 15)
            self._buyer_min_days_floor = max(0, self._buyer_min_days_floor - reduction)
            self._treds_used = True
            wc_gap = float(getattr(self._state, "working_capital_gap", 0.0) or 0.0)
            fin_rate = round(float(tc.interest_rate_annual) * 1.05, 4)
            fin_approved = round(wc_gap * 0.8, 2)
            coverage_pct = min(100, int(fin_approved / max(wc_gap, 1.0) * 100))
            message = (
                f"TReDS activated: buyer day floor reduced {reduction} days. "
                f"Financier: rate={fin_rate:.1%}/yr, approved=INR {fin_approved:,.0f} "
                f"({coverage_pct}% of WC gap)."
            )

        auto_accept = (
            self._buyer_price <= proposed_price
            and self._buyer_days <= tc.liquidity_threshold
        )
        accepts_current_buyer_offer = (
            action_type == "accept"
            and abs(proposed_price - self._buyer_price) < 1e-4
            and int(proposed_days) == int(self._buyer_days)
        )
        # SME accepts while echoing its last proposed days/price (not necessarily buyer's current counter)
        accepts_own_proposal = (
            action_type == "accept"
            and self._last_sme_proposed_days is not None
            and int(proposed_days) == int(self._last_sme_proposed_days)
            and (
                self._last_sme_proposed_price is None
                or abs(proposed_price - float(self._last_sme_proposed_price)) < 1e-4
            )
        )

        if action_type == "reject":
            terminal_reward = _strict_unit_interval(0.0)
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Agent rejected the negotiation"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            self._reward_debug_print("reject_episode", terminal_reward)
            self._validator.mark_deal_resolved(deal_id)
            self._step_in_progress = False
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=terminal_reward,
                message="Agent rejected the negotiation",
                reward=terminal_reward,
                done=True,
                metadata=self._episode_meta(
                    "reject_episode",
                    success=False,
                    termination_reason="agent_reject",
                ),
            )

        if action_type == "accept" and not auto_accept and not accepts_current_buyer_offer and not accepts_own_proposal:
            terminal_reward = _strict_unit_interval(0.0)
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "ACCEPT failed validation"
            self._reward_debug_print("invalid_accept_mismatch", terminal_reward)
            self._step_in_progress = False
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=terminal_reward,
                message="ACCEPT failed validation",
                reward=terminal_reward,
                done=True,
                metadata=self._episode_meta(
                    "invalid_accept",
                    success=False,
                    termination_reason="invalid_accept",
                ),
            )

        if auto_accept or action_type == "accept" or accepts_current_buyer_offer:
            agreed_price = proposed_price if proposed_price >= self._buyer_price else self._buyer_price
            # Use the better (lower) of proposed vs last-proposed so the LLM isn't penalised for echoing buyer_days.
            if action_type == "accept" and self._last_sme_proposed_days is not None:
                agreed_days = min(int(proposed_days), int(self._last_sme_proposed_days))
            else:
                agreed_days = proposed_days
            self._deal_reached = True
            self._final_price = agreed_price
            self._final_days = agreed_days
            self._treds_used = use_treds or self._treds_used
            self._apply_terminal_outcome_to_state(agreed_price, agreed_days, action)
            # Keep success strictly below 1.0 so no pipeline stage emits an exact endpoint score.
            terminal_reward = min(self._terminal_reward(), 0.99)

            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Deal reached"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            self._reward_debug_print("terminal_agreement", terminal_reward)
            success = bool(self._deal_reached)
            self._validator.mark_deal_resolved(deal_id)
            # Supervisor: record accept + build breakdown for metadata
            _accept_bd_meta: dict[str, Any] = {}
            if self._supervisor is not None:
                try:
                    _rq_accept, _plan_bonus = self._supervisor.on_accept(deal_id, action, agreed_days)
                except Exception:
                    pass
            try:
                _bd_accept = compute_legacy_terminal_breakdown(
                    self._state, task_name=self._task_config.name
                )
                _accept_bd_meta = _bd_accept.to_dict()
            except Exception:
                pass
            _accept_meta = self._episode_meta(
                "terminal_agreement", success=success, termination_reason="buyer_accepted_deal"
            )
            if _accept_bd_meta:
                _accept_meta["reward_breakdown"] = _accept_bd_meta
            if self._step_penalties:
                _accept_meta["validator_penalties"] = list(self._step_penalties)
            self._step_in_progress = False
            return self._obs_from_state(
                buyer_accepted=True,
                negotiation_done=True,
                step_reward=terminal_reward,
                message="Deal reached",
                reward=terminal_reward,
                done=True,
                metadata=_accept_meta,
            )

        prior_buyer_price = float(self._buyer_price)
        prior_buyer_days = int(self._buyer_days)

        if action_type == "propose":
            self._last_sme_proposed_days = proposed_days
            self._last_sme_proposed_price = proposed_price
            # Process supervision: track reasoning quality
            if self._supervisor is not None:
                _rq = self._supervisor.on_propose(deal_id, action)
                # format_compliance: 1.0 if reason provided, 0.0 otherwise
                _fmt = 1.0 if (action.reason and len(str(action.reason).strip()) >= 5) else 0.0

        price_jitter = self._rng.uniform(0.85, 1.15)
        price_drop = (
            self._buyer_price
            * self._base_concede
            * price_jitter
            * power_mult
        )
        next_price = round(
            max(tc.cost_threshold, min(self._buyer_price, proposed_price) - price_drop),
            2,
        )
        day_drop = max(
            1,
            int(round(self._rng.randint(tc.day_step_low, tc.day_step_high) * power_mult)),
        )
        next_days = max(self._buyer_min_days_floor, self._buyer_days - day_drop)

        self._buyer_price = next_price
        self._buyer_days = next_days
        self._state.step_count += 1
        self._state.negotiation_round = self._state.step_count

        _use_v2 = os.getenv("SME_REWARD_V2", "0").strip() not in ("0", "false", "False")
        if _use_v2:
            step_reward, reward_branch = self._compute_reward_v2(
                proposed_price,
                proposed_days,
                current_buyer_price=prior_buyer_price,
                current_buyer_days=prior_buyer_days,
                used_tool_this_step=use_treds,
            )
        else:
            step_reward, reward_branch = self._compute_reward(
                proposed_price,
                proposed_days,
                current_buyer_price=prior_buyer_price,
                current_buyer_days=prior_buyer_days,
            )

        # Apply soft validator penalty to step reward (proposal_loop, tool_dedup)
        step_reward = round(step_reward + _validator_penalty, 6)

        self._cumulative_reward += step_reward
        self._state.cumulative_reward = self._cumulative_reward
        self._state.buyer_price = self._buyer_price
        self._state.buyer_days = self._buyer_days

        # Update validator with new buyer offer so future accepts can verify
        self._validator.record_buyer_offer(deal_id, self._buyer_price, self._buyer_days)
        # Process supervision: per-step progress tracking
        if self._supervisor is not None:
            self._supervisor.on_step_outcome(self._buyer_days)

        # Enrich counter-offer message so LLM knows what the buyer conceded and what's needed to close.
        buyer_days_delta = prior_buyer_days - self._buyer_days
        if not message.startswith("TReDS"):
            message = (
                f"Buyer countered: days={self._buyer_days} "
                f"(was {prior_buyer_days}, conceded {buyer_days_delta} days), "
                f"price={self._buyer_price:.2f} (was {prior_buyer_price:.2f}). "
                f"To close: payment_days<={tc.liquidity_threshold}, price>={tc.cost_threshold:.2f}."
            )
        self._state.message = message

        self._reward_debug_print(reward_branch, step_reward)

        done = self._check_done()
        termination_reason = "ongoing"
        success = False
        if done:
            self._state.message = "Maximum rounds reached — no agreement"
            # Preserve the baseline rule: terminal grading replaces the last
            # partial shaping reward once max rounds end the episode.
            terminal_reward = self._terminal_reward()
            self._cumulative_reward = self._cumulative_reward - step_reward + terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            step_reward = terminal_reward
            reward_branch = f"max_rounds_terminal_grader:{terminal_reward:.4f}"
            self._reward_debug_print(reward_branch, step_reward)
            termination_reason = "max_rounds_no_deal"
            success = False

        # Build reward breakdown for metadata (legacy path: compliance-based approximation)
        _bd_meta: dict[str, Any] = {}
        if done and self._state is not None:
            try:
                _bd = compute_legacy_terminal_breakdown(self._state, task_name=self._task_config.name)
                _bd_meta = _bd.to_dict()
            except Exception:
                pass

        _step_meta = self._episode_meta(reward_branch, success=success, termination_reason=termination_reason)
        if _bd_meta:
            _step_meta["reward_breakdown"] = _bd_meta
        if self._step_penalties:
            _step_meta["validator_penalties"] = list(self._step_penalties)

        self._step_in_progress = False
        return self._obs_from_state(
            buyer_accepted=False,
            negotiation_done=done,
            step_reward=step_reward,
            message=self._state.message,
            reward=step_reward,
            done=done,
            metadata=_step_meta,
        )

_LEGACY_SME_NEGOTIATOR_STEP = SMENegotiatorEnvironment.step


def _stage6_sme_negotiator_step(
    self: SMENegotiatorEnvironment,
    action: NegotiationAction,
    **kwargs: Any,
) -> NegotiationObservation:
    """Optional Stage 6 buyer-policy hook layered over the legacy step path."""
    if getattr(self, "_buyer_policy", None) is None:
        return _LEGACY_SME_NEGOTIATOR_STEP(self, action, **kwargs)

    if self._state is None:
        self.reset(seed=self._seed, difficulty=self._task_config.difficulty)

    assert self._state is not None
    tc = self._task_config
    power_mult = self._buyer_counter_power_multiplier()

    if self._deal_reached or self._state.step_count >= tc.max_rounds:
        return self._obs_from_state(
            buyer_accepted=False,
            negotiation_done=True,
            step_reward=0.0,
            message="Episode already completed",
            reward=0.0,
            done=True,
            metadata=self._episode_meta(
                "already_completed",
                success=False,
                termination_reason="already_completed",
            ),
        )

    action_type = str(action.action_type).lower()
    if action_type in {"simulate_plan", "advance_period", "tool"}:
        return _LEGACY_SME_NEGOTIATOR_STEP(self, action, **kwargs)

    proposed_price = float(action.price)
    proposed_days = int(action.payment_days)
    use_treds = bool(action.use_treds)

    if use_treds:
        reduction = self._rng.randint(5, 15)
        self._buyer_min_days_floor = max(0, self._buyer_min_days_floor - reduction)
        self._treds_used = True

    auto_accept = self._buyer_price <= proposed_price and self._buyer_days <= tc.liquidity_threshold
    accepts_current_buyer_offer = (
        action_type == "accept"
        and abs(proposed_price - self._buyer_price) < 1e-4
        and int(proposed_days) == int(self._buyer_days)
    )
    accepts_own_proposal = (
        action_type == "accept"
        and self._last_sme_proposed_days is not None
        and int(proposed_days) == int(self._last_sme_proposed_days)
        and (
            self._last_sme_proposed_price is None
            or abs(proposed_price - float(self._last_sme_proposed_price)) < 1e-4
        )
    )

    if action_type == "reject":
        return _LEGACY_SME_NEGOTIATOR_STEP(self, action, **kwargs)

    if action_type == "accept" and not auto_accept and not accepts_current_buyer_offer and not accepts_own_proposal:
        return _LEGACY_SME_NEGOTIATOR_STEP(self, action, **kwargs)

    if auto_accept or action_type == "accept" or accepts_current_buyer_offer:
        return _LEGACY_SME_NEGOTIATOR_STEP(self, action, **kwargs)

    prior_buyer_price = float(self._buyer_price)
    prior_buyer_days = int(self._buyer_days)
    if action_type == "propose":
        self._last_sme_proposed_days = proposed_days
        self._last_sme_proposed_price = proposed_price

    price_jitter = self._rng.uniform(0.85, 1.15)
    price_drop = self._buyer_price * self._base_concede * price_jitter * power_mult
    day_drop = max(1, int(round(self._rng.randint(tc.day_step_low, tc.day_step_high) * power_mult)))
    buyer_observation = self._build_buyer_observation(
        action=action,
        proposed_price=proposed_price,
        proposed_days=proposed_days,
        price_drop=price_drop,
        day_drop=day_drop,
    )
    try:
        buyer_action = self._buyer_policy.act(buyer_observation)
    except Exception:
        buyer_action = self._heuristic_buyer_action(
            action=action,
            proposed_price=proposed_price,
            proposed_days=proposed_days,
            price_drop=price_drop,
            day_drop=day_drop,
        )

    return self._apply_buyer_action(
        buyer_action,
        originating_action=action,
        proposed_price=proposed_price,
        proposed_days=proposed_days,
        prior_buyer_price=prior_buyer_price,
        prior_buyer_days=prior_buyer_days,
    )


SMENegotiatorEnvironment.step = _stage6_sme_negotiator_step


class SMELiquidityEnvironment(Environment):
    """Long-horizon liquidity environment with explicit macro periods and planning.

    ``SMENegotiatorEnvironment`` remains the authoritative single-deal engine.
    This wrapper composes many per-deal engines into a shared world with:
    - multiple concurrent deal negotiations
    - explicit macro-period advancement
    - deterministic cashflow simulation for planning
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        total_periods: int = 6,
        *,
        buyer_policy: Optional["TextPolicy"] = None,
        financier_policy: Optional["FinancierPolicy"] = None,
        buyer_variance: float = 0.0,
        financier_variance: float = 0.0,
        tool_backend_mode: Literal["deterministic", "live"] = "deterministic",
        live_tool_adapters: Optional[dict[str, LiveToolAdapter]] = None,
    ) -> None:
        self._rng = Random()
        self._seed = 1000
        self._task_config: TaskConfig = TASK_REGISTRY["payment-terms-medium"]
        self._total_periods = max(1, int(total_periods))
        self._buyer_policy = buyer_policy
        self._financier_policy = financier_policy
        self._buyer_variance = max(0.0, float(buyer_variance))
        self._financier_variance = max(0.0, float(financier_variance))
        self._world_state: Optional[WorldState] = None
        self._state: Optional[LiquidityEnvironmentState] = None
        self._current_negotiation: Optional[NegotiationState] = None
        self._current_actor_type = "SME"
        self._current_actor_id = "sme_0"
        self._active_sme_id = "sme_0"
        self._active_buyer_id = "buyer_0"
        self._active_deal_id: Optional[str] = None
        self._financier_id = "financier_0"
        self._deal_envs: dict[str, SMENegotiatorEnvironment] = {}
        self._deal_observations: dict[str, NegotiationObservation] = {}
        self._current_negotiations: dict[str, NegotiationState] = {}
        self._deal_trajectories: dict[str, list[NegotiationState]] = {}
        self._deal_shaping_rewards: dict[str, list[float]] = {}
        self._rl_reward_trace: list[float] = []
        self._latest_verifiable_reward: Optional[float] = None
        self._legacy_last_reward: Optional[float] = None
        self._last_financier_quote_rate: Optional[float] = None
        self._last_simulation_result: Optional[CashflowProjection] = None
        self._resolved_deal_ids: list[str] = []
        self._tool_history: list[ToolCallRecord] = []
        self._history: list[HistoryEvent] = []
        self._last_tool_call: Optional[ToolCallRecord] = None
        self._last_tool_name: Optional[str] = None
        self._last_tool_args: Optional[dict[str, object]] = None
        self._last_tool_result: Optional[ToolResultEnvelope] = None
        self._pending_tool_bonus_by_deal: dict[str, float] = {}
        self._pending_tool_call_by_deal: dict[str, ToolCallRecord] = {}
        self._tool_spam_flag_by_deal: dict[str, bool] = {}
        self._tool_backend_mode: Literal["deterministic", "live"] = (
            "live" if str(tool_backend_mode).lower() == "live" else "deterministic"
        )
        self._tool_backends: dict[str, BaseToolBackend] = build_tool_backend_registry(
            mode=self._tool_backend_mode,
            live_adapters=live_tool_adapters,
        )
        self._tool_result_cache: dict[str, ToolResultEnvelope] = {}
        self._active_reward_component_report: Optional[RewardComponentReport] = None
        self._tool_call_count: int = 0
        self._tool_effective_count: int = 0
        self._duplicate_tool_count: int = 0
        self._invalid_action_count: int = 0
        self._stall_step_count: int = 0
        self._terminated_by_step_cap: bool = False
        self._episode_done: bool = False
        self._episode_step_cap: int = 0

    @property
    def state(self) -> Optional[LiquidityEnvironmentState]:
        """Current serialized state for the liquidity environment."""

        return self._state

    def get_state(self) -> Optional[LiquidityEnvironmentState]:
        """Backward-compatible accessor for direct Python callers."""

        return self._state

    def _period_offset(self, days: int) -> int:
        return max(1, int(math.ceil(max(int(days), 0) / 30.0)))

    def _stable_noise(self, label: str) -> float:
        """Deterministic zero-mean noise derived from the environment seed."""
        seeded = Random(f"{self._seed}:{label}")
        return seeded.uniform(-1.0, 1.0)

    def _buyer_variation(self, buyer_id: str) -> tuple[float, float, float]:
        """Return deterministic buyer perturbations for default tendency and power."""
        if self._buyer_variance <= 0.0:
            return 0.0, 0.0, 0.0
        tendency_delta = 0.10 * self._buyer_variance * self._stable_noise(f"buyer_default:{buyer_id}")
        power_delta = 0.08 * self._buyer_variance * self._stable_noise(f"buyer_power:{buyer_id}")
        concede_delta = 0.20 * self._buyer_variance * self._stable_noise(f"buyer_concede:{buyer_id}")
        return tendency_delta, power_delta, concede_delta

    def _financier_variation(self) -> tuple[float, float, float]:
        """Return deterministic financier perturbations for appetite, capital, and spread."""
        if self._financier_variance <= 0.0:
            return 0.0, 0.0, 0.0
        appetite_delta = 0.12 * self._financier_variance * self._stable_noise("financier_appetite")
        capital_delta = 0.35 * self._financier_variance * self._stable_noise("financier_capital")
        spread_delta = 0.04 * self._financier_variance * self._stable_noise("financier_spread")
        return appetite_delta, capital_delta, spread_delta

    def _copy_negotiation_state(
        self,
        state: NegotiationState,
        *,
        deal_id: str,
        buyer_id: str,
    ) -> NegotiationState:
        """Copy a wrapped deal state and attach stable world/deal ids."""

        return state.model_copy(
            deep=True,
            update={
                "sme_id": self._active_sme_id,
                "buyer_id": buyer_id,
                "financier_id": self._financier_id,
                "deal_id": deal_id,
            },
        )

    def _current_sme(self) -> SMEAccountState:
        assert self._world_state is not None
        return next(sme for sme in self._world_state.smes if sme.sme_id == self._active_sme_id)

    def _buyer_by_id(self, buyer_id: str) -> BuyerState:
        assert self._world_state is not None
        return next(buyer for buyer in self._world_state.buyers if buyer.buyer_id == buyer_id)

    def _get_deal_state(self, deal_id: str) -> DealState:
        assert self._world_state is not None
        return next(deal for deal in self._world_state.deals if deal.deal_id == deal_id)

    def _open_deal_ids(self) -> list[str]:
        assert self._world_state is not None
        return [deal.deal_id for deal in self._world_state.deals if deal.status == "open"]

    def _resolved_ids_from_world(self) -> list[str]:
        assert self._world_state is not None
        return [deal.deal_id for deal in self._world_state.deals if deal.status != "open"]

    def _set_active_deal_id(self, preferred: Optional[str] = None) -> None:
        open_ids = self._open_deal_ids() if self._world_state is not None else []
        if preferred and preferred in open_ids:
            self._active_deal_id = preferred
        elif self._active_deal_id in open_ids:
            pass
        elif open_ids:
            self._active_deal_id = open_ids[0]
        else:
            self._active_deal_id = None

        if self._active_deal_id and self._active_deal_id in self._current_negotiations:
            self._current_negotiation = self._current_negotiations[self._active_deal_id]
            self._active_buyer_id = self._current_negotiation.buyer_id

    def _current_basis_observation(self) -> NegotiationObservation:
        if self._active_deal_id and self._active_deal_id in self._deal_observations:
            return self._deal_observations[self._active_deal_id]
        if self._current_negotiation is not None and self._current_negotiation.deal_id in self._deal_observations:
            return self._deal_observations[self._current_negotiation.deal_id]
        if self._deal_observations:
            return next(iter(self._deal_observations.values()))
        raise RuntimeError("No deal observation is available to construct a liquidity observation.")

    def _normalize_tool_args(self, tool_args: Optional[dict[str, object]]) -> dict[str, object]:
        if not tool_args:
            return {}
        return json.loads(json.dumps(tool_args, sort_keys=True, default=str))

    def _history_tail(self) -> list[HistoryEvent]:
        return [event.model_copy(deep=True) for event in self._history[-8:]]

    def _current_pending_tool_bonus(self) -> float:
        if self._active_deal_id is None:
            return 0.0
        return float(self._pending_tool_bonus_by_deal.get(self._active_deal_id, 0.0))

    def _current_tool_spam_flag(self) -> bool:
        if self._active_deal_id is None:
            return False
        return bool(self._tool_spam_flag_by_deal.get(self._active_deal_id, False))

    def _episode_step_cap_for_world(self) -> int:
        assert self._world_state is not None
        deal_slots_per_period = max(1, len(self._world_state.buyers))
        per_deal_budget = max(1, int(self._task_config.max_rounds)) + 4
        macro_budget = max(1, int(self._world_state.total_periods))
        return max(8, macro_budget * deal_slots_per_period * per_deal_budget + macro_budget)

    def _tool_cache_key(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, object],
        context_fingerprint: str,
    ) -> str:
        return json.dumps(
            {
                "tool_name": str(tool_name).upper(),
                "tool_args": tool_args,
                "context_fingerprint": context_fingerprint,
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    def _current_reward_component_report(self, *, tool_bonus: float = 0.0) -> Optional[RewardComponentReport]:
        assert self._world_state is not None
        deal_id = self._active_deal_id
        if deal_id is None:
            return None
        trajectory = self._deal_trajectories.get(deal_id, [])
        if not trajectory:
            return None
        return compute_reward_component_report(
            self._world_state,
            trajectory,
            lambda_shaping=float(self._world_state.reward_lambda_shaping),
            tool_bonus=float(tool_bonus),
        )

    def _failure_termination_reason(self) -> Optional[str]:
        assert self._world_state is not None
        for sme in self._world_state.smes:
            if bool(sme.defaulted):
                return "sme_defaulted"
            if float(sme.cash_balance) < 0.0:
                return "negative_cash_balance"
            if float(sme.current_utilization) > float(sme.credit_limit):
                return "credit_limit_breached"
        return None

    def _tool_error_envelope(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, object],
        context_fingerprint: str,
        detail: str,
        error_code: str,
    ) -> ToolResultEnvelope:
        return ToolResultEnvelope(
            source="deterministic",
            backend_name="ToolErrorEnvelope",
            request_id=self._tool_cache_key(
                tool_name=tool_name,
                tool_args=tool_args,
                context_fingerprint=context_fingerprint,
            ),
            latency_ms=0,
            cache_hit=False,
            stale=False,
            normalized_payload={
                "error": error_code,
                "tool_name": str(tool_name).upper(),
                "detail": detail,
            },
            requested_source=self._tool_backend_mode,
        )

    def _deal_context_fingerprint(self, deal_id: Optional[str]) -> str:
        assert self._world_state is not None
        deal_payload: dict[str, object] = {"deal_id": deal_id}
        if deal_id and any(deal.deal_id == deal_id for deal in self._world_state.deals):
            deal = self._get_deal_state(deal_id)
            deal_payload = {
                "deal_id": deal.deal_id,
                "status": deal.status,
                "agreement_period": deal.agreement_period,
                "invoice_amount": round(float(deal.invoice_amount), 2),
                "agreed_payment_days": deal.agreed_payment_days,
                "financed": bool(deal.financed),
                "settled": bool(deal.settled),
                "failed": bool(deal.failed),
            }
        world_payload = {
            "current_period": int(self._world_state.current_period),
            "cash_balance": round(float(self._current_sme().cash_balance), 2),
            "open_deals": len(self._open_deal_ids()),
        }
        return json.dumps(
            {"deal": deal_payload, "world": world_payload},
            sort_keys=True,
            separators=(",", ":"),
        )

    def _append_history_event(
        self,
        *,
        actor: str,
        event_type: str,
        summary: str,
        deal_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict[str, object]] = None,
        tool_result: Optional[ToolResultEnvelope] = None,
    ) -> None:
        assert self._world_state is not None
        self._history.append(
            HistoryEvent(
                actor=actor,
                event_type=event_type,
                period_index=int(self._world_state.current_period),
                deal_id=deal_id,
                summary=summary,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
            )
        )

    def _resolve_tool_target_deal_id(self, tool_name: str, tool_args: dict[str, object]) -> Optional[str]:
        candidate = tool_args.get("deal_id")
        if tool_name == "QUERY_TREDS":
            candidate = tool_args.get("invoice_id", candidate)
        elif tool_name == "CHECK_COMPLIANCE":
            candidate = tool_args.get("contract_id", candidate)
        if candidate is None:
            return self._active_deal_id
        deal_id = str(candidate)
        if deal_id in self._current_negotiations:
            return deal_id
        return self._active_deal_id

    def _metadata_bundle(
        self,
        *,
        latest_shaping_reward: float = 0.0,
        latest_reward_branch: Optional[str] = None,
        total_sme_reward: Optional[float] = None,
        tool_bonus_applied: float = 0.0,
        termination_reason: Optional[str] = None,
    ) -> dict[str, object]:
        assert self._world_state is not None
        reward_component_report = self._active_reward_component_report
        metadata: dict[str, object] = {
            "reward_mode": "stage3_long_horizon",
            "active_deal_id": self._active_deal_id,
            "current_period": int(self._world_state.current_period),
            "episode_step": int(self._world_state.episode_step),
            "episode_step_cap": int(self._episode_step_cap),
            "legacy_inner_reward": self._legacy_last_reward,
            "legacy_reward_branch": latest_reward_branch,
            "latest_shaping_reward": latest_shaping_reward,
            "latest_verifiable_reward": self._latest_verifiable_reward,
            "simulation_projection_present": self._last_simulation_result is not None,
            "resolved_deal_count": len(self._resolved_deal_ids),
            "defaulted_sme_count": sum(1 for sme in self._world_state.smes if sme.defaulted),
            "last_tool_name": self._last_tool_name,
            "tool_backend_mode": self._tool_backend_mode,
            "tool_bonus_applied": round(float(tool_bonus_applied), 6),
            "pending_tool_bonus": round(self._current_pending_tool_bonus(), 6),
            "tool_spam_flag": self._current_tool_spam_flag(),
            "tool_history_length": len(self._tool_history),
            "tool_call_count": int(self._tool_call_count),
            "tool_effective_count": int(self._tool_effective_count),
            "duplicate_tool_count": int(self._duplicate_tool_count),
            "invalid_action_count": int(self._invalid_action_count),
            "stall_step_count": int(self._stall_step_count),
            "terminated_by_step_cap": bool(self._terminated_by_step_cap),
        }
        if self._active_deal_id is not None:
            metadata["trajectory_length"] = len(self._deal_trajectories.get(self._active_deal_id, []))
            metadata["shaping_rewards"] = list(self._deal_shaping_rewards.get(self._active_deal_id, []))
            metadata["shaping_lambda"] = float(self._world_state.reward_lambda_shaping)
        if total_sme_reward is not None:
            metadata["total_sme_reward"] = total_sme_reward
        if reward_component_report is not None:
            metadata["reward_component_report"] = reward_component_report.model_dump()
        if self._last_tool_result is not None:
            metadata["last_tool_source"] = self._last_tool_result.source
            metadata["last_tool_backend_name"] = self._last_tool_result.backend_name
            metadata["last_tool_cache_hit"] = bool(self._last_tool_result.cache_hit)
        if termination_reason is not None:
            metadata["termination_reason"] = termination_reason
        return metadata

    def _build_observation(
        self,
        observation: NegotiationObservation,
        *,
        reward_override: Optional[float] = None,
        done_override: Optional[bool] = None,
        extra_metadata: Optional[dict[str, object]] = None,
        simulation_projection: Optional[CashflowProjection] = None,
    ) -> LiquidityObservation:
        """Lift a wrapped deal observation into the long-horizon liquidity observation."""

        assert self._world_state is not None

        payload = observation.model_dump()
        if reward_override is not None:
            payload["reward"] = reward_override
            payload["step_reward"] = reward_override
        if done_override is not None:
            payload["done"] = done_override

        metadata = dict(payload.get("metadata") or {})
        if extra_metadata:
            metadata.update(extra_metadata)
        payload["metadata"] = metadata
        _proj = simulation_projection
        payload.update(
            {
                "agent_type": self._current_actor_type,
                "agent_id": self._current_actor_id,
                "current_actor": self._current_actor_type,
                "active_deal_id": self._active_deal_id,
                "open_deal_ids": self._open_deal_ids(),
                "resolved_deal_ids": list(self._resolved_deal_ids),
                "current_period": int(self._world_state.current_period),
                "total_periods": int(self._world_state.total_periods),
                "episode_step": int(self._world_state.episode_step),
                "simulation_projection": simulation_projection,
                "projected_balances": _proj.period_balances if _proj is not None else None,
                "projected_defaults": _proj.period_defaults if _proj is not None else None,
                "projected_penalties": _proj.period_penalties if _proj is not None else None,
                "last_tool_name": self._last_tool_name,
                "last_tool_args": dict(self._last_tool_args) if self._last_tool_args is not None else None,
                "last_tool_result": self._last_tool_result.model_copy(deep=True)
                if self._last_tool_result is not None
                else None,
                "reward_component_report": self._active_reward_component_report.model_copy(deep=True)
                if self._active_reward_component_report is not None
                else None,
                "history": self._history_tail(),
            }
        )
        return LiquidityObservation(**payload)

    def _task_config_for_buyer(self, buyer: BuyerState) -> TaskConfig:
        _, power_delta, concede_delta = self._buyer_variation(buyer.buyer_id)
        buyer_power = (
            float(self._task_config.buyer_power_score)
            if buyer.buyer_id == "buyer_0"
            else float(
                self._task_config.secondary_buyer_power
                if self._task_config.secondary_buyer_power is not None
                else min(1.0, self._task_config.buyer_power_score + 0.05)
            )
        )
        buyer_power = min(1.0, max(0.0, buyer_power + power_delta))
        initial_price = round(
            min(
                float(self._task_config.initial_buyer_price),
                float(buyer.budget_per_period) / max(float(self._task_config.volume), 1.0),
            ),
            2,
        )
        return replace(
            self._task_config,
            initial_buyer_price=max(float(self._task_config.cost_threshold), initial_price),
            initial_buyer_days=int(buyer.baseline_payment_days),
            buyer_power_score=buyer_power,
            concede_low=max(0.001, float(self._task_config.concede_low) * (1.0 + concede_delta)),
            concede_high=max(0.001, float(self._task_config.concede_high) * (1.0 + concede_delta)),
        )

    def _build_world_state(self) -> WorldState:
        tc = self._task_config
        required_minimum_cash = float(tc.minimum_cash_buffer_ratio) * float(tc.sme_monthly_revenue)
        initial_cash = float(tc.initial_cash_balance_ratio) * float(tc.sme_monthly_revenue)
        primary_default_delta, _, _ = self._buyer_variation("buyer_0")
        secondary_default_delta, _, _ = self._buyer_variation("buyer_1")
        primary_default = min(1.0, max(0.0, float(tc.primary_buyer_default_tendency) + primary_default_delta))
        secondary_default = float(
            tc.secondary_buyer_default_tendency
            if tc.secondary_buyer_default_tendency is not None
            else min(1.0, primary_default + 0.1)
        )
        secondary_default = min(1.0, max(0.0, secondary_default + secondary_default_delta))
        financier_appetite_delta, financier_capital_delta, _ = self._financier_variation()
        credit_limit = max(
            1.0,
            float(tc.credit_limit_multiplier)
            * float(tc.sme_monthly_revenue)
            * max(int(tc.current_payment_terms_days) - int(tc.sme_supplier_payment_days), 0)
            / 365.0,
        )
        return WorldState(
            smes=[
                SMEAccountState(
                    sme_id=self._active_sme_id,
                    cash_balance=round(initial_cash, 2),
                    supplier_payment_days=int(tc.sme_supplier_payment_days),
                    credit_limit=round(credit_limit, 2),
                    current_utilization=0.0,
                    risk_score=min(1.0, float(tc.buyer_power_score)),
                    required_minimum_cash=round(required_minimum_cash, 2),
                    defaulted=bool(initial_cash < 0.0),
                    missed_supplier_payment=bool(initial_cash < required_minimum_cash),
                )
            ],
            buyers=[
                BuyerState(
                    buyer_id="buyer_0",
                    demand_level=1.0,
                    budget_per_period=float(tc.initial_buyer_price) * float(tc.volume) * 1.2,
                    default_tendency=primary_default,
                    baseline_payment_days=int(tc.initial_buyer_days),
                ),
                BuyerState(
                    buyer_id="buyer_1",
                    demand_level=0.9,
                    budget_per_period=float(tc.initial_buyer_price) * float(tc.volume) * 1.1,
                    default_tendency=secondary_default,
                    baseline_payment_days=int(tc.initial_buyer_days),
                ),
            ],
            financier=FinancierState(
                financier_id=self._financier_id,
                available_capital=round(
                    float(tc.financier_capital_multiplier)
                    * (1.0 + financier_capital_delta)
                    * float(tc.initial_buyer_price)
                    * float(tc.volume),
                    2,
                ),
                risk_appetite=min(1.0, max(0.0, float(tc.financier_risk_appetite) + financier_appetite_delta)),
                base_interest_rate=float(tc.interest_rate_annual),
            ),
            legal_max_payment_days=int(tc.legal_max_payment_days),
            baseline_discount_rate=0.0,
            reward_lambda_shaping=float(tc.reward_lambda_shaping),
            current_period=0,
            total_periods=int(self._total_periods),
            episode_step=0,
            history=[],
            deals=[],
        )

    def _spawn_period_deals(self, period_index: int) -> None:
        assert self._world_state is not None
        if any(sme.defaulted for sme in self._world_state.smes):
            return

        for buyer in self._world_state.buyers:
            deal_index = sum(
                1
                for deal in self._world_state.deals
                if deal.created_period == period_index and deal.buyer_id == buyer.buyer_id
            )
            deal_id = f"deal_p{period_index}_{buyer.buyer_id}_{deal_index}"
            override = self._task_config_for_buyer(buyer)
            env = SMENegotiatorEnvironment(buyer_policy=self._buyer_policy)
            observation = env.reset(
                seed=self._seed + period_index * 100 + deal_index,
                difficulty=override.difficulty,
                episode_id=f"{deal_id}-{self._seed}",
                task_name=override.name,
                task_config_override=override,
            )
            assert env.state is not None

            copied_state = self._copy_negotiation_state(
                env.state,
                deal_id=deal_id,
                buyer_id=buyer.buyer_id,
            )
            self._deal_envs[deal_id] = env
            self._deal_observations[deal_id] = observation
            self._current_negotiations[deal_id] = copied_state
            self._deal_trajectories[deal_id] = [copied_state.model_copy(deep=True)]
            self._deal_shaping_rewards[deal_id] = []
            self._world_state.deals.append(
                DealState(
                    deal_id=deal_id,
                    sme_id=self._active_sme_id,
                    buyer_id=buyer.buyer_id,
                    status="open",
                    created_period=period_index,
                    invoice_amount=round(float(observation.buyer_price) * float(observation.volume), 2),
                    supplier_payment_amount=round(float(observation.cost_threshold) * float(observation.volume), 2),
                    volume=int(observation.volume),
                )
            )

        self._world_state = apply_plan_to_world_state(self._world_state, {})
        self._set_active_deal_id(self._active_deal_id or (self._open_deal_ids()[0] if self._open_deal_ids() else None))

    def _quote_financier_rate(self, buyer_id: str) -> Optional[float]:
        if self._world_state is None or self._world_state.financier is None:
            return None

        financier = self._world_state.financier
        buyer = self._buyer_by_id(buyer_id)
        sme = self._current_sme()
        _, _, spread_delta = self._financier_variation()
        jitter = self._rng.uniform(-0.002, 0.002)
        quote = float(financier.base_interest_rate)
        quote += 0.04 * float(sme.risk_score)
        quote += 0.03 * float(buyer.default_tendency)
        quote += spread_delta
        quote += jitter
        return round(min(0.95, max(0.0, quote)), 6)

    def _build_financier_observation(
        self,
        *,
        deal: DealState,
        negotiation_state: NegotiationState,
    ) -> NegotiationObservation:
        """Build a financier-view observation for Stage 6 financing policies."""
        basis = self._deal_observations.get(deal.deal_id) or self._current_basis_observation()
        requested_amount = round(float(deal.invoice_amount), 2)
        available_capital = (
            float(self._world_state.financier.available_capital)
            if self._world_state is not None and self._world_state.financier is not None
            else 0.0
        )
        heuristic_rate = self._quote_financier_rate(deal.buyer_id) or float(negotiation_state.interest_rate_annual)
        context = {
            "deal_id": deal.deal_id,
            "requested_amount": requested_amount,
            "available_capital": round(available_capital, 2),
            "heuristic_rate": round(float(heuristic_rate), 6),
            "buyer_id": deal.buyer_id,
            "treds_requested": bool(negotiation_state.treds_used),
        }
        return basis.model_copy(
            update={
                "message": "Financier viewpoint. Evaluate whether to finance the accepted deal.\nFINANCIER_CONTEXT="
                + json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            }
        )

    def _heuristic_financier_quote(
        self,
        *,
        deal: DealState,
        negotiation_state: NegotiationState,
    ) -> Any:
        """Return the current deterministic financing quote in a duck-typed object."""
        approved_amount = 0.0
        if self._world_state is not None and self._world_state.financier is not None:
            approved_amount = min(float(deal.invoice_amount), float(self._world_state.financier.available_capital))
        if TYPE_CHECKING:
            from rl.opponents import FinancierQuote

        from rl.opponents import FinancierQuote

        return FinancierQuote(
            approved=bool(negotiation_state.treds_used and approved_amount > 0.0),
            annual_rate=float(self._quote_financier_rate(deal.buyer_id) or negotiation_state.interest_rate_annual),
            approved_amount=round(float(approved_amount), 2),
            reason="Heuristic financier quote",
        )

    def _apply_financier_quote(
        self,
        *,
        deal: DealState,
        negotiation_state: NegotiationState,
        quote: Any,
    ) -> bool:
        """Apply an optional Stage 6 financier quote to the agreed deal."""
        approved = bool(getattr(quote, "approved", False))
        annual_rate = float(getattr(quote, "annual_rate", negotiation_state.interest_rate_annual))
        approved_amount = float(getattr(quote, "approved_amount", float(deal.invoice_amount)))
        self._last_financier_quote_rate = round(annual_rate, 6)
        if approved:
            deal.finance_rate = round(annual_rate, 6)
            deal.financed = bool(negotiation_state.treds_used)
            deal.financing_principal = round(min(float(deal.invoice_amount), approved_amount), 2)
            self._append_history_event(
                actor="FINANCIER",
                event_type="financier_quote",
                deal_id=deal.deal_id,
                summary=(
                    f"approved financing on {deal.deal_id} rate={annual_rate:.6f} "
                    f"amount={deal.financing_principal:.2f}"
                ),
            )
            return True
        self._append_history_event(
            actor="FINANCIER",
            event_type="financier_quote",
            deal_id=deal.deal_id,
            summary=f"declined financing on {deal.deal_id} requested_treds={bool(negotiation_state.treds_used)}",
        )
        return False

    def _close_open_deals_for_episode_end(self) -> None:
        assert self._world_state is not None
        for deal in self._world_state.deals:
            if deal.status == "open":
                deal.status = "failed"
                deal.failed = True
                if deal.deal_id not in self._resolved_deal_ids:
                    self._resolved_deal_ids.append(deal.deal_id)
        self._world_state = apply_plan_to_world_state(self._world_state, {})

    def _macro_terminal_reward(self) -> float:
        assert self._world_state is not None

        weighted_score = 0.0
        total_weight = 0.0
        for deal in self._world_state.deals:
            if deal.status == "open":
                continue
            trajectory = self._deal_trajectories.get(deal.deal_id, [])
            if not trajectory:
                continue
            score = compute_verifiable_reward(self._world_state, trajectory)
            weight = float(deal.invoice_amount) if float(deal.invoice_amount) > 0.0 else 1.0
            weighted_score += score * weight
            total_weight += weight

        if total_weight <= 0.0:
            return 0.0
        return round(weighted_score / total_weight, 6)

    def _rebuild_state_snapshot(self) -> None:
        assert self._world_state is not None

        self._resolved_deal_ids = self._resolved_ids_from_world()
        self._set_active_deal_id(self._active_deal_id)

        if self._active_deal_id and self._active_deal_id in self._current_negotiations:
            current_negotiation = self._current_negotiations[self._active_deal_id]
        elif self._current_negotiations:
            current_negotiation = next(iter(self._current_negotiations.values()))
        else:
            raise RuntimeError("Liquidity environment has no current negotiations to serialize.")

        self._current_negotiation = current_negotiation
        self._active_buyer_id = current_negotiation.buyer_id

        active_trajectory = self._deal_trajectories.get(current_negotiation.deal_id or "", [])
        active_shaping = self._deal_shaping_rewards.get(current_negotiation.deal_id or "", [])
        self._active_reward_component_report = self._current_reward_component_report(
            tool_bonus=self._current_pending_tool_bonus(),
        )
        self._state = LiquidityEnvironmentState(
            episode_id=current_negotiation.episode_id,
            seed=self._seed,
            task_name=current_negotiation.task_name,
            difficulty=current_negotiation.difficulty,
            step_count=int(current_negotiation.step_count),
            max_steps=int(current_negotiation.max_steps),
            current_actor_type=self._current_actor_type,
            current_actor_id=self._current_actor_id,
            active_sme_id=self._active_sme_id,
            active_buyer_id=self._active_buyer_id,
            world_state=self._world_state,
            current_negotiation=current_negotiation,
            active_deal_id=self._active_deal_id,
            current_negotiations={
                deal_id: state.model_copy(deep=True)
                for deal_id, state in self._current_negotiations.items()
            },
            deal_trajectories={
                deal_id: [state.model_copy(deep=True) for state in trajectory]
                for deal_id, trajectory in self._deal_trajectories.items()
            },
            trajectory=[state.model_copy(deep=True) for state in active_trajectory],
            shaping_rewards=list(active_shaping),
            cumulative_rl_reward=round(sum(self._rl_reward_trace), 6),
            latest_verifiable_reward=self._latest_verifiable_reward,
            legacy_last_reward=self._legacy_last_reward,
            last_financier_quote_rate=self._last_financier_quote_rate,
            resolved_deal_ids=list(self._resolved_deal_ids),
            last_simulation_result=self._last_simulation_result.model_copy(deep=True)
            if self._last_simulation_result is not None
            else None,
            tool_history=[record.model_copy(deep=True) for record in self._tool_history],
            last_tool_call=self._last_tool_call.model_copy(deep=True)
            if self._last_tool_call is not None
            else None,
            history_tail=self._history_tail(),
            pending_tool_bonus=round(self._current_pending_tool_bonus(), 6),
            active_reward_component_report=self._active_reward_component_report.model_copy(deep=True)
            if self._active_reward_component_report is not None
            else None,
            tool_call_count=int(self._tool_call_count),
            tool_effective_count=int(self._tool_effective_count),
            duplicate_tool_count=int(self._duplicate_tool_count),
            invalid_action_count=int(self._invalid_action_count),
            stall_step_count=int(self._stall_step_count),
            terminated_by_step_cap=bool(self._terminated_by_step_cap),
            episode_step_cap=int(self._episode_step_cap),
            tool_backend_mode=self._tool_backend_mode,
        )

    def _error_observation(
        self,
        message: str,
        *,
        done: bool = False,
        invalid_action: bool = False,
        termination_reason: Optional[str] = None,
    ) -> LiquidityObservation:
        if invalid_action:
            self._invalid_action_count += 1
            if self._world_state is not None:
                self._rebuild_state_snapshot()
        basis = self._current_basis_observation()
        observation = basis.model_copy(update={"message": message})
        return self._build_observation(
            observation,
            reward_override=0.0,
            done_override=done,
            extra_metadata=self._metadata_bundle(termination_reason=termination_reason),
            simulation_projection=self._last_simulation_result,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "MEDIUM",
        **kwargs,
    ) -> LiquidityObservation:
        """Reset the macro world, then spawn deterministic deal negotiations for period 0."""

        self._seed = int(seed if seed is not None else 1000)
        self._rng = Random(self._seed)
        self._current_actor_type = "SME"
        self._current_actor_id = self._active_sme_id
        self._last_financier_quote_rate = None
        self._latest_verifiable_reward = None
        self._legacy_last_reward = None
        self._last_simulation_result = None
        self._resolved_deal_ids = []
        self._rl_reward_trace = []
        self._deal_envs = {}
        self._deal_observations = {}
        self._current_negotiations = {}
        self._deal_trajectories = {}
        self._deal_shaping_rewards = {}
        self._active_deal_id = None
        self._current_negotiation = None
        self._tool_history = []
        self._history = []
        self._last_tool_call = None
        self._last_tool_name = None
        self._last_tool_args = None
        self._last_tool_result = None
        self._pending_tool_bonus_by_deal = {}
        self._pending_tool_call_by_deal = {}
        self._tool_spam_flag_by_deal = {}
        self._tool_result_cache = {}
        self._active_reward_component_report = None
        self._tool_call_count = 0
        self._tool_effective_count = 0
        self._duplicate_tool_count = 0
        self._invalid_action_count = 0
        self._stall_step_count = 0
        self._terminated_by_step_cap = False
        self._episode_done = False
        self._episode_step_cap = 0

        requested_task = kwargs.get("task_name") or kwargs.get("task")
        task_id = resolve_task_id(
            str(requested_task) if requested_task else None,
            difficulty=difficulty,
        )
        self._task_config = TASK_REGISTRY[task_id]
        self._world_state = self._build_world_state()
        self._episode_step_cap = self._episode_step_cap_for_world()
        self._spawn_period_deals(0)
        self._rebuild_state_snapshot()
        return self._build_observation(
            self._current_basis_observation(),
            reward_override=0.0,
            done_override=False,
            extra_metadata=self._metadata_bundle(),
        )

    def _run_cashflow_projection(
        self,
        *,
        plan: Optional[dict[str, object]],
        horizon: Optional[int],
    ) -> tuple[CashflowProjection, dict[str, object]]:
        assert self._world_state is not None

        remaining_periods = max(int(self._world_state.total_periods) - int(self._world_state.current_period), 0)
        resolved_horizon = int(horizon or remaining_periods)
        tool_result = run_cashflow_sim(
            self._world_state,
            plan or {},
            horizon=resolved_horizon,
        )
        projection = CashflowProjection(
            period_balances=list(tool_result["period_balances"]),
            period_defaults=list(tool_result["period_defaults"]),
            period_penalties=list(tool_result["period_penalties"]),
        )
        return projection, tool_result

    def _terminate_episode(
        self,
        *,
        termination_reason: str,
        terminated_by_step_cap: bool = False,
    ) -> LiquidityObservation:
        assert self._world_state is not None
        self._close_open_deals_for_episode_end()
        self._latest_verifiable_reward = self._macro_terminal_reward()
        self._terminated_by_step_cap = bool(terminated_by_step_cap)
        self._episode_done = True
        reward_value = float(self._latest_verifiable_reward or 0.0)
        self._append_history_event(
            actor="SYSTEM",
            event_type="episode_termination",
            deal_id=None,
            summary=(
                f"terminate reason={termination_reason} "
                f"step_cap={bool(terminated_by_step_cap)} reward={reward_value:.6f}"
            ),
        )
        self._rl_reward_trace.append(reward_value)
        self._rebuild_state_snapshot()
        return self._build_observation(
            self._current_basis_observation(),
            reward_override=reward_value,
            done_override=True,
            extra_metadata=self._metadata_bundle(
                total_sme_reward=reward_value,
                termination_reason=termination_reason,
            ),
            simulation_projection=self._last_simulation_result,
        )

    def _enforce_terminal_conditions(self, observation: LiquidityObservation) -> LiquidityObservation:
        assert self._world_state is not None
        if bool(observation.done):
            self._episode_done = True
            return observation
        failure_reason = self._failure_termination_reason()
        if failure_reason is not None:
            return self._terminate_episode(termination_reason=failure_reason)
        if int(self._episode_step_cap) > 0 and int(self._world_state.episode_step) >= int(self._episode_step_cap):
            return self._terminate_episode(
                termination_reason="episode_step_cap",
                terminated_by_step_cap=True,
            )
        return observation

    def _step_negotiation(self, action: NegotiationAction) -> LiquidityObservation:
        assert self._world_state is not None

        deal_id = action.deal_id or self._active_deal_id
        if not deal_id or deal_id not in self._deal_envs:
            return self._error_observation(
                "No active deal is available for a negotiation action.",
                invalid_action=True,
                termination_reason="missing_active_deal",
            )

        deal = self._get_deal_state(deal_id)
        if deal.status != "open":
            return self._error_observation(
                f"Deal '{deal_id}' is already resolved.",
                invalid_action=True,
                termination_reason="deal_already_resolved",
            )

        previous_state = self._current_negotiations.get(deal_id)
        self._world_state.episode_step += 1
        inner_action = action.model_copy(
            update={
                "deal_id": None,
                "simulation_plan": None,
                "simulation_horizon": None,
                "tool_name": None,
                "tool_args": None,
            }
        )
        observation = self._deal_envs[deal_id].step(inner_action)
        assert self._deal_envs[deal_id].state is not None

        buyer_id = deal.buyer_id
        self._legacy_last_reward = float(observation.reward) if observation.reward is not None else None
        self._deal_observations[deal_id] = observation
        copied_state = self._copy_negotiation_state(
            self._deal_envs[deal_id].state,
            deal_id=deal_id,
            buyer_id=buyer_id,
        )
        self._current_negotiations[deal_id] = copied_state
        self._current_negotiation = copied_state
        self._deal_trajectories.setdefault(deal_id, []).append(copied_state.model_copy(deep=True))
        self._active_deal_id = deal_id
        self._active_buyer_id = buyer_id
        self._latest_verifiable_reward = None
        if self._buyer_policy is not None:
            self._append_history_event(
                actor="BUYER",
                event_type="buyer_response",
                deal_id=deal_id,
                summary=str(observation.message),
            )

        if observation.done:
            if observation.buyer_accepted and copied_state.deal_reached:
                financing_enabled = bool(copied_state.treds_used)
                if self._financier_policy is None:
                    self._last_financier_quote_rate = self._quote_financier_rate(buyer_id)
                else:
                    financier_observation = self._build_financier_observation(
                        deal=deal,
                        negotiation_state=copied_state,
                    )
                    try:
                        quote = self._financier_policy.act(financier_observation)
                    except Exception:
                        quote = self._heuristic_financier_quote(
                            deal=deal,
                            negotiation_state=copied_state,
                        )
                    financing_enabled = self._apply_financier_quote(
                        deal=deal,
                        negotiation_state=copied_state,
                        quote=quote,
                    )
                    if not financing_enabled:
                        copied_state.treds_used = False
                self._world_state = apply_plan_to_world_state(
                    self._world_state,
                    {
                        "deal_decisions": {
                            deal_id: {
                                "decision": "accept",
                                "price": float(copied_state.final_price or copied_state.buyer_price),
                                "payment_days": int(
                                    copied_state.agreed_terms
                                    if copied_state.agreed_terms is not None
                                    else copied_state.buyer_days
                                ),
                                "use_treds": bool(copied_state.treds_used),
                                "propose_dynamic_discounting": bool(copied_state.dynamic_discounting_agreed),
                                "dynamic_discount_annual_rate": float(copied_state.agreed_dynamic_discount_annual),
                                "late_payment_penalty_agreed": bool(copied_state.late_payment_penalty_agreed),
                            }
                        },
                        "financing": {
                            deal_id: bool(financing_enabled)
                        },
                    },
                )
                updated_deal = self._get_deal_state(deal_id)
                if self._financier_policy is not None and financing_enabled and self._last_financier_quote_rate is not None:
                    updated_deal.finance_rate = round(float(self._last_financier_quote_rate), 6)
                updated_deal.supplier_payment_amount = round(
                    float(copied_state.cost_threshold) * float(copied_state.volume),
                    2,
                )
            else:
                termination_reason = (observation.metadata or {}).get("termination_reason")
                deal.status = "rejected" if str(action.action_type).lower() == "reject" or termination_reason == "buyer_policy_reject" else "failed"
                deal.failed = True
                self._world_state = apply_plan_to_world_state(self._world_state, {})

            if deal_id not in self._resolved_deal_ids:
                self._resolved_deal_ids.append(deal_id)
            self._set_active_deal_id()

        latest_shaping = 0.0
        if len(self._deal_trajectories[deal_id]) > 1:
            shaping_rewards = compute_shaping_rewards(self._deal_trajectories[deal_id])
            self._deal_shaping_rewards[deal_id] = shaping_rewards
            latest_shaping = shaping_rewards[-1]

        tool_bonus = compute_tool_use_bonus(
            latest_tool_call=self._pending_tool_call_by_deal.get(deal_id),
            current_deal_id=deal_id,
            current_step_index=int(self._world_state.episode_step),
            base_shaping_reward=latest_shaping,
            previous_state=previous_state.model_copy(deep=True) if previous_state is not None else None,
            next_state=copied_state.model_copy(deep=True),
            legal_max_payment_days=int(self._world_state.legal_max_payment_days),
            pending_tool_bonus=float(self._pending_tool_bonus_by_deal.get(deal_id, 0.0)),
        )
        reward_value = round(float(self._world_state.reward_lambda_shaping) * latest_shaping + tool_bonus, 6)
        pending_tool_call = self._pending_tool_call_by_deal.get(deal_id)
        if pending_tool_call is not None and float(tool_bonus) > 0.0:
            self._tool_effective_count += 1
        if (
            str(action.action_type).lower() == "propose"
            and not bool(observation.done)
            and float(reward_value) <= 0.0
        ):
            self._stall_step_count += 1
        self._append_history_event(
            actor="SME",
            event_type="negotiation_step",
            deal_id=deal_id,
            summary=(
                f"{str(action.action_type).lower()} on {deal_id}: "
                f"buyer_accepted={bool(observation.buyer_accepted)} "
                f"done={bool(observation.negotiation_done)} "
                f"reward={reward_value:.6f}"
            ),
        )
        self._pending_tool_call_by_deal.pop(deal_id, None)
        self._pending_tool_bonus_by_deal.pop(deal_id, None)
        self._tool_spam_flag_by_deal.pop(deal_id, None)
        self._rl_reward_trace.append(reward_value)
        self._rebuild_state_snapshot()
        self._active_reward_component_report = self._current_reward_component_report(tool_bonus=tool_bonus)
        if self._state is not None:
            self._state.active_reward_component_report = (
                self._active_reward_component_report.model_copy(deep=True)
                if self._active_reward_component_report is not None
                else None
            )
        return self._build_observation(
            observation,
            reward_override=reward_value,
            done_override=False,
            extra_metadata=self._metadata_bundle(
                latest_shaping_reward=latest_shaping,
                latest_reward_branch=(observation.metadata or {}).get("reward_branch"),
                tool_bonus_applied=tool_bonus,
            ),
            simulation_projection=self._last_simulation_result,
        )

    def _step_simulation(self, action: NegotiationAction) -> LiquidityObservation:
        assert self._world_state is not None

        self._world_state.episode_step += 1
        self._last_simulation_result, tool_result = self._run_cashflow_projection(
            plan=action.simulation_plan or {},
            horizon=action.simulation_horizon,
        )
        self._legacy_last_reward = None
        self._latest_verifiable_reward = None
        self._append_history_event(
            actor="SYSTEM",
            event_type="planning_projection",
            deal_id=action.deal_id or self._active_deal_id,
            summary=(
                f"simulate_plan projected ending_balance={tool_result['ending_balance']:.2f} "
                f"any_default={bool(tool_result['any_default'])}"
            ),
        )
        self._rl_reward_trace.append(0.0)
        self._rebuild_state_snapshot()
        return self._build_observation(
            self._current_basis_observation(),
            reward_override=0.0,
            done_override=False,
            extra_metadata=self._metadata_bundle(),
            simulation_projection=self._last_simulation_result,
        )

    def _step_tool(self, action: NegotiationAction) -> LiquidityObservation:
        assert self._world_state is not None

        tool_name = str(action.tool_name or "").upper()
        tool_args = self._normalize_tool_args(action.tool_args)
        deal_id = self._resolve_tool_target_deal_id(tool_name, tool_args)
        if deal_id and deal_id in self._current_negotiations:
            self._active_deal_id = deal_id
            self._current_negotiation = self._current_negotiations[deal_id]
            self._active_buyer_id = self._current_negotiation.buyer_id

        self._world_state.episode_step += 1
        self._legacy_last_reward = None
        self._latest_verifiable_reward = None

        context_fingerprint = self._deal_context_fingerprint(deal_id)
        cache_key = self._tool_cache_key(
            tool_name=tool_name,
            tool_args=tool_args,
            context_fingerprint=context_fingerprint,
        )
        duplicate_tool = False
        previous_tool = self._pending_tool_call_by_deal.get(deal_id or "")
        if previous_tool is not None:
            duplicate_tool = bool(
                previous_tool.tool_name == tool_name
                and previous_tool.tool_args == tool_args
                and previous_tool.context_fingerprint == context_fingerprint
                and int(previous_tool.period_index) == int(self._world_state.current_period)
            )
        self._tool_call_count += 1
        if duplicate_tool:
            self._duplicate_tool_count += 1

        if deal_id is not None:
            self._pending_tool_bonus_by_deal[deal_id] = -0.005 if duplicate_tool else 0.0
            self._tool_spam_flag_by_deal[deal_id] = duplicate_tool

        backend = self._tool_backends.get(tool_name)
        valid_tool = backend is not None
        if valid_tool:
            cached_result = self._tool_result_cache.get(cache_key)
            if cached_result is not None:
                tool_result = cached_result.model_copy(update={"cache_hit": True, "latency_ms": 0})
            else:
                try:
                    tool_result = backend.invoke(
                        world_state=self._world_state,
                        tool_args=tool_args,
                        context_fingerprint=context_fingerprint,
                        negotiation_state=self._current_negotiations.get(deal_id or ""),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self._invalid_action_count += 1
                    tool_result = self._tool_error_envelope(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        context_fingerprint=context_fingerprint,
                        detail=str(exc),
                        error_code="tool_execution_failed",
                    )
                else:
                    self._tool_result_cache[cache_key] = tool_result.model_copy(deep=True)

            payload = dict(tool_result.normalized_payload)
            if tool_name == "RUN_CASHFLOW_SIM" and "error" not in payload:
                projection = CashflowProjection(
                    period_balances=[float(item) for item in payload.get("period_balances", [])],
                    period_defaults=[bool(item) for item in payload.get("period_defaults", [])],
                    period_penalties=[float(item) for item in payload.get("period_penalties", [])],
                )
                self._last_simulation_result = projection
                simulation_projection = projection
            else:
                simulation_projection = self._last_simulation_result

            record = ToolCallRecord(
                step_index=int(self._world_state.episode_step),
                period_index=int(self._world_state.current_period),
                deal_id=deal_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                context_fingerprint=context_fingerprint,
            )
            self._tool_history.append(record)
            self._last_tool_call = record
            self._last_tool_name = tool_name
            self._last_tool_args = dict(tool_args)
            self._last_tool_result = tool_result.model_copy(deep=True)
            if deal_id is not None and "error" not in payload:
                self._pending_tool_call_by_deal[deal_id] = record
            elif deal_id is not None:
                self._pending_tool_call_by_deal.pop(deal_id, None)
        else:
            self._invalid_action_count += 1
            tool_result = self._tool_error_envelope(
                tool_name=tool_name,
                tool_args=tool_args,
                context_fingerprint=context_fingerprint,
                detail=f"Unknown tool '{tool_name}'",
                error_code="unknown_tool",
            )
            simulation_projection = self._last_simulation_result
            self._last_tool_name = tool_name
            self._last_tool_args = dict(tool_args)
            self._last_tool_result = tool_result.model_copy(deep=True)

        self._append_history_event(
            actor="TOOL",
            event_type="tool_call",
            deal_id=deal_id,
            summary=f"{tool_name} on {deal_id or 'world'}",
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        )
        self._rl_reward_trace.append(0.0)
        self._rebuild_state_snapshot()
        return self._build_observation(
            self._current_basis_observation(),
            reward_override=0.0,
            done_override=False,
            extra_metadata=self._metadata_bundle(),
            simulation_projection=simulation_projection,
        )

    def _step_advance_period(self) -> LiquidityObservation:
        assert self._world_state is not None

        if int(self._world_state.current_period) >= int(self._world_state.total_periods):
            return self._error_observation(
                "Macro episode already completed.",
                done=True,
                invalid_action=True,
                termination_reason="post_terminal_action",
            )

        self._world_state.episode_step += 1
        self._world_state = advance_world_state(self._world_state)
        self._legacy_last_reward = None

        if int(self._world_state.current_period) < int(self._world_state.total_periods):
            self._spawn_period_deals(int(self._world_state.current_period))
            reward_value = 0.0
            done = False
            self._latest_verifiable_reward = None
            total_sme_reward = None
        else:
            self._close_open_deals_for_episode_end()
            self._latest_verifiable_reward = self._macro_terminal_reward()
            reward_value = self._latest_verifiable_reward
            done = True
            total_sme_reward = self._latest_verifiable_reward
            self._episode_done = True

        self._append_history_event(
            actor="SYSTEM",
            event_type="advance_period",
            deal_id=None,
            summary=(
                f"advance_period -> current_period={int(self._world_state.current_period)} "
                f"done={done} reward={reward_value:.6f}"
            ),
        )
        self._rl_reward_trace.append(reward_value)
        self._rebuild_state_snapshot()
        return self._build_observation(
            self._current_basis_observation(),
            reward_override=reward_value,
            done_override=done,
            extra_metadata=self._metadata_bundle(
                total_sme_reward=total_sme_reward,
                termination_reason="macro_horizon_end" if done else None,
            ),
            simulation_projection=self._last_simulation_result,
        )

    def step(self, action: NegotiationAction, **kwargs) -> LiquidityObservation:
        """Advance a micro negotiation, run a simulation, or close a macro period."""

        if self._world_state is None:
            self.reset(seed=self._seed, difficulty=self._task_config.difficulty, **kwargs)
            assert self._world_state is not None
        if self._episode_done:
            return self._error_observation(
                "Episode already completed.",
                done=True,
                invalid_action=True,
                termination_reason="post_terminal_action",
            )

        action_type = str(action.action_type).lower()
        if action_type in {"propose", "accept", "reject"}:
            observation = self._step_negotiation(action)
            return self._enforce_terminal_conditions(observation)
        if action_type == "simulate_plan":
            observation = self._step_simulation(action)
            return self._enforce_terminal_conditions(observation)
        if action_type == "advance_period":
            observation = self._step_advance_period()
            return self._enforce_terminal_conditions(observation)
        if action_type == "tool":
            observation = self._step_tool(action)
            return self._enforce_terminal_conditions(observation)
        return self._error_observation(
            f"Unsupported liquidity action type '{action_type}'.",
            invalid_action=True,
            termination_reason="unsupported_action_type",
        )
