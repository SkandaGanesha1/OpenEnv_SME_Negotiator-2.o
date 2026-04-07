"""OpenEnv environment implementation for SME negotiation."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from random import Random
from typing import Optional, Tuple

from openenv.core import Environment

from sme_negotiator_env.graders import TASK_GRADERS, grade_task_payment_terms_medium
from sme_negotiator_env.problem_context import RAZORPAY_ITCH_BLURB
from sme_negotiator_env.models import (
    NegotiationAction,
    NegotiationObservation,
    NegotiationState,
    default_negotiation_state,
)
from sme_negotiator_env.task_config import TaskConfig, resolve_task_id, TASK_REGISTRY


class SMENegotiatorEnvironment(Environment):
    """OpenEnv environment for SME payment term negotiation."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
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

    @property
    def state(self) -> Optional[NegotiationState]:
        """Current episode state exposed as attribute for OpenEnv app serialization."""

        return self._state

    def get_state(self) -> Optional[NegotiationState]:
        """Backward-compatible state accessor for direct Python usage."""

        return self._state

    def _now_utc_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _grader_fn(self):
        return TASK_GRADERS.get(self._task_config.grader_id, grade_task_payment_terms_medium)

    def _apply_terminal_outcome_to_state(
        self,
        agreed_price: float,
        agreed_days: int,
        action: NegotiationAction,
    ) -> None:
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
        assert self._state is not None
        return float(self._grader_fn()(self._state))

    def _compute_reward(
        self,
        proposed_price: float,
        proposed_days: int,
        *,
        current_buyer_price: float,
        current_buyer_days: int,
    ) -> Tuple[float, str]:
        """Partial step reward for ongoing negotiation (non-terminal)."""

        pct = self._task_config
        cost = float(pct.cost_threshold)
        liq = int(pct.liquidity_threshold)
        init_p = float(self._initial_buyer_price)
        cur_d = int(current_buyer_days)
        cur_p = float(current_buyer_price)

        if proposed_price < cost:
            return 0.12, "invalid_below_cost"

        if proposed_days > cur_d:
            return 0.1, "no_progress_days_worse_than_buyer"

        days_delta_frac = max(0.0, min(1.0, float(cur_d - proposed_days) / max(float(cur_d), 1.0)))
        days_gap = max(1, cur_d - liq)
        days_toward_threshold = max(0.0, min(1.0, float(cur_d - proposed_days) / float(days_gap)))
        days_improve = max(0.0, min(1.0, 0.45 * days_delta_frac + 0.55 * days_toward_threshold))

        price_span = max(1e-9, init_p - cost)
        price_improve = max(0.0, min(1.0, (float(proposed_price) - cost) / price_span))

        improvement = max(0.0, min(1.0, 0.65 * days_improve + 0.35 * price_improve))

        if improvement < 1e-6:
            return 0.08, "no_progress_negligible"

        raw = 0.2 + 0.6 * improvement
        partial = min(0.3, raw * 0.3)
        detail = f"improvement={improvement:.3f}|days={days_improve:.3f}|price={price_improve:.3f}"
        return round(partial, 4), f"partial_progress:{detail}"

    def _reward_debug_print(self, branch: str, step_reward: float) -> None:
        if os.getenv("REWARD_DEBUG", "0").strip() not in ("0", "false", "False", "no", "No"):
            print(
                f"[REWARD_DEBUG] branch={branch} step_reward={step_reward:.4f}",
                file=sys.stderr,
                flush=True,
            )

    def _check_done(self) -> bool:
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
        """Reset the environment for a new episode."""

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
            message=(
                f"{RAZORPAY_ITCH_BLURB} "
                f"Task: {tc.description} "
                f"{tc.context_note} "
                f"| Episode reset @ {self._now_utc_iso()} (task_id={tc.name}, base_concede={self._base_concede:.4f})"
            ),
        )

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

    def _buyer_counter_power_multiplier(self) -> float:
        """Higher buyer power → smaller concessions."""
        p = float(self._task_config.buyer_power_score)
        return max(0.2, 1.0 - 0.55 * p)

    def step(self, action: NegotiationAction, **kwargs) -> NegotiationObservation:
        """Apply one action and advance the negotiation."""

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
        proposed_price = float(action.price)
        proposed_days = int(action.payment_days)
        use_treds = bool(action.use_treds)
        step_reward = 0.0
        message = "Buyer countered"

        if use_treds:
            reduction = self._rng.randint(5, 15)
            self._buyer_min_days_floor = max(0, self._buyer_min_days_floor - reduction)
            self._treds_used = True
            message = f"TReDS lowered buyer day floor by {reduction} days"

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
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += step_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Agent rejected the negotiation"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            self._reward_debug_print("reject_episode", step_reward)
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=step_reward,
                message="Agent rejected the negotiation",
                reward=step_reward,
                done=True,
                metadata=self._episode_meta(
                    "reject_episode",
                    success=False,
                    termination_reason="agent_reject",
                ),
            )

        if action_type == "accept" and not auto_accept and not accepts_current_buyer_offer and not accepts_own_proposal:
            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += step_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "ACCEPT failed validation"
            self._reward_debug_print("invalid_accept_mismatch", step_reward)
            return self._obs_from_state(
                buyer_accepted=False,
                negotiation_done=True,
                step_reward=step_reward,
                message="ACCEPT failed validation",
                reward=step_reward,
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
            terminal_reward = self._terminal_reward()

            self._state.step_count += 1
            self._state.negotiation_round = self._state.step_count
            self._cumulative_reward += terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Deal reached"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            self._reward_debug_print("terminal_agreement", terminal_reward)
            success = terminal_reward > 0.0
            return self._obs_from_state(
                buyer_accepted=True,
                negotiation_done=True,
                step_reward=terminal_reward,
                message="Deal reached",
                reward=terminal_reward,
                done=True,
                metadata=self._episode_meta(
                    "terminal_agreement",
                    success=success,
                    termination_reason="buyer_accepted_deal",
                ),
            )

        prior_buyer_price = float(self._buyer_price)
        prior_buyer_days = int(self._buyer_days)

        if action_type == "propose":
            self._last_sme_proposed_days = proposed_days
            self._last_sme_proposed_price = proposed_price

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
        self._state.message = message

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
            success = terminal_reward > 0.0

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
