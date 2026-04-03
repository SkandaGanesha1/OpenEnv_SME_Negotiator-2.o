"""OpenEnv environment implementation for SME negotiation."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

from openenv.core import Environment

from sme_negotiator_env.models import (
    NegotiationAction,
    NegotiationObservation,
    NegotiationState,
)


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    initial_buyer_price: float
    initial_buyer_days: int
    max_rounds: int
    volume: int
    cost_threshold: float
    liquidity_threshold: int
    concede_low: float
    concede_high: float
    day_floor: int
    day_step_low: int
    day_step_high: int


PRESETS: Dict[str, DifficultyPreset] = {
    "EASY": DifficultyPreset(
        name="EASY",
        initial_buyer_price=100.0,
        initial_buyer_days=30,
        max_rounds=5,
        volume=1000,
        cost_threshold=80.0,
        liquidity_threshold=45,
        concede_low=0.01,
        concede_high=0.03,
        day_floor=30,
        day_step_low=1,
        day_step_high=3,
    ),
    "MEDIUM": DifficultyPreset(
        name="MEDIUM",
        initial_buyer_price=100.0,
        initial_buyer_days=90,
        max_rounds=8,
        volume=1000,
        cost_threshold=80.0,
        liquidity_threshold=60,
        concede_low=0.005,
        concede_high=0.015,
        day_floor=60,
        day_step_low=2,
        day_step_high=6,
    ),
    "HARD": DifficultyPreset(
        name="HARD",
        initial_buyer_price=95.0,
        initial_buyer_days=120,
        max_rounds=12,
        volume=5000,
        cost_threshold=70.0,
        liquidity_threshold=30,
        concede_low=0.002,
        concede_high=0.008,
        day_floor=90,
        day_step_low=1,
        day_step_high=4,
    ),
}


class SMENegotiatorEnvironment(Environment):
    """OpenEnv environment for SME payment term negotiation."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._rng = Random()
        self._preset = PRESETS["EASY"]
        self._difficulty = "EASY"
        self._seed = 1000
        self._base_concede = 0.02
        self._buyer_min_days_floor = self._preset.day_floor
        self._buyer_price = self._preset.initial_buyer_price
        self._buyer_days = self._preset.initial_buyer_days
        self._initial_buyer_price = self._preset.initial_buyer_price
        self._deal_reached = False
        self._final_price: Optional[float] = None
        self._final_days: Optional[int] = None
        self._treds_used = False
        self._cumulative_reward = 0.0
        self._state: Optional[NegotiationState] = None

    def state(self) -> Optional[NegotiationState]:
        """Return the current episode state."""

        return self._state

    def _calculate_terminal_reward(self, use_treds: bool) -> float:
        """Calculate terminal reward from the current buyer state."""

        price_score = max(
            0.0,
            (self._buyer_price - self._preset.cost_threshold)
            / max(1e-9, self._initial_buyer_price - self._preset.cost_threshold),
        )
        if self._buyer_days <= self._preset.liquidity_threshold:
            days_score = 1.0
        else:
            days_score = max(
                0.0,
                1.0 - (self._buyer_days - self._preset.liquidity_threshold) / 60.0,
            )

        treds_bonus = 0.15 if (use_treds or self._treds_used) else 0.0
        return min(1.0, 0.6 * price_score + 0.3 * days_score + 0.1 + treds_bonus)

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "EASY",
        **kwargs,
    ) -> NegotiationObservation:
        """Reset the environment for a new episode."""

        self._difficulty = (difficulty or "EASY").upper()
        if self._difficulty not in PRESETS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        self._preset = PRESETS[self._difficulty]
        self._seed = int(seed if seed is not None else 1000)
        self._rng = Random(self._seed)
        self._base_concede = self._rng.uniform(self._preset.concede_low, self._preset.concede_high)
        self._buyer_min_days_floor = self._preset.day_floor
        self._buyer_price = self._preset.initial_buyer_price
        self._buyer_days = self._preset.initial_buyer_days
        self._initial_buyer_price = self._preset.initial_buyer_price
        self._deal_reached = False
        self._final_price = None
        self._final_days = None
        self._treds_used = False
        self._cumulative_reward = 0.0

        episode_id = kwargs.get("episode_id") or f"{self._difficulty.lower()}_{self._seed}"
        self._state = NegotiationState(
            episode_id=episode_id,
            seed=self._seed,
            difficulty=self._difficulty,
            step_count=0,
            max_steps=self._preset.max_rounds,
            deal_reached=False,
            final_price=None,
            final_days=None,
            treds_used=False,
            cumulative_reward=0.0,
            buyer_price=self._buyer_price,
            buyer_days=self._buyer_days,
            cost_threshold=self._preset.cost_threshold,
            liquidity_threshold=self._preset.liquidity_threshold,
            volume=self._preset.volume,
            message=f"Episode reset (base_concede={self._base_concede:.4f})",
        )

        return NegotiationObservation(
            round_number=0,
            max_rounds=self._preset.max_rounds,
            buyer_price=self._buyer_price,
            buyer_days=self._buyer_days,
            buyer_accepted=False,
            negotiation_done=False,
            cost_threshold=self._preset.cost_threshold,
            liquidity_threshold=self._preset.liquidity_threshold,
            volume=self._preset.volume,
            difficulty=self._difficulty,
            price_score=0.0,
            days_score=0.0,
            treds_bonus=0.0,
            step_reward=0.0,
            message=f"Episode reset (base_concede={self._base_concede:.4f})",
            reward=0.0,
            done=False,
            metadata={
                "episode_id": episode_id,
                "seed": self._seed,
                "base_concede": self._base_concede,
                "buyer_day_floor": self._buyer_min_days_floor,
            },
        )

    def step(self, action: NegotiationAction, **kwargs) -> NegotiationObservation:
        """Apply one action and advance the negotiation."""

        if self._state is None:
            self.reset(seed=self._seed, difficulty=self._difficulty)

        assert self._state is not None

        if self._deal_reached or self._state.step_count >= self._preset.max_rounds:
            return NegotiationObservation(
                round_number=self._state.step_count,
                max_rounds=self._preset.max_rounds,
                buyer_price=self._buyer_price,
                buyer_days=self._buyer_days,
                buyer_accepted=False,
                negotiation_done=True,
                cost_threshold=self._preset.cost_threshold,
                liquidity_threshold=self._preset.liquidity_threshold,
                volume=self._preset.volume,
                difficulty=self._difficulty,
                price_score=0.0,
                days_score=0.0,
                treds_bonus=0.0,
                step_reward=0.0,
                message="Episode already completed",
                reward=0.0,
                done=True,
                metadata={"episode_id": self._state.episode_id},
            )

        action_type = str(action.action_type).lower()
        proposed_price = float(action.price)
        proposed_days = int(action.payment_days)
        use_treds = bool(action.use_treds)
        step_reward = 0.0
        message = "Buyer countered"

        if proposed_price < self._preset.cost_threshold:
            step_reward -= 0.1

        if use_treds:
            reduction = self._rng.randint(5, 15)
            self._buyer_min_days_floor = max(0, self._buyer_min_days_floor - reduction)
            message = f"TReDS lowered buyer day floor by {reduction} days"

        auto_accept = (
            self._buyer_price <= proposed_price
            and self._buyer_days <= self._preset.liquidity_threshold
        )

        if action_type == "reject":
            self._state.step_count += 1
            self._cumulative_reward += step_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Agent rejected the negotiation"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            return NegotiationObservation(
                round_number=self._state.step_count,
                max_rounds=self._preset.max_rounds,
                buyer_price=self._buyer_price,
                buyer_days=self._buyer_days,
                buyer_accepted=False,
                negotiation_done=True,
                cost_threshold=self._preset.cost_threshold,
                liquidity_threshold=self._preset.liquidity_threshold,
                volume=self._preset.volume,
                difficulty=self._difficulty,
                price_score=0.0,
                days_score=0.0,
                treds_bonus=0.15 if self._treds_used else 0.0,
                step_reward=step_reward,
                message="Agent rejected the negotiation",
                reward=step_reward,
                done=True,
                metadata={"episode_id": self._state.episode_id},
            )

        if action_type == "accept" and not auto_accept:
            self._state.step_count += 1
            self._cumulative_reward += step_reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "ACCEPT failed validation"
            return NegotiationObservation(
                round_number=self._state.step_count,
                max_rounds=self._preset.max_rounds,
                buyer_price=self._buyer_price,
                buyer_days=self._buyer_days,
                buyer_accepted=False,
                negotiation_done=True,
                cost_threshold=self._preset.cost_threshold,
                liquidity_threshold=self._preset.liquidity_threshold,
                volume=self._preset.volume,
                difficulty=self._difficulty,
                price_score=0.0,
                days_score=0.0,
                treds_bonus=0.0,
                step_reward=step_reward,
                message="ACCEPT failed validation",
                reward=step_reward,
                done=True,
                metadata={"episode_id": self._state.episode_id},
            )

        if auto_accept or action_type == "accept":
            agreed_price = proposed_price if proposed_price >= self._buyer_price else self._buyer_price
            agreed_days = proposed_days
            terminal_reward = self._calculate_terminal_reward(use_treds)
            self._deal_reached = True
            self._final_price = agreed_price
            self._final_days = agreed_days
            self._treds_used = use_treds or self._treds_used
            self._state.step_count += 1
            self._cumulative_reward += terminal_reward
            self._state.deal_reached = True
            self._state.final_price = agreed_price
            self._state.final_days = agreed_days
            self._state.treds_used = self._treds_used
            self._state.cumulative_reward = self._cumulative_reward
            self._state.message = "Deal reached"
            self._state.buyer_price = self._buyer_price
            self._state.buyer_days = self._buyer_days
            return NegotiationObservation(
                round_number=self._state.step_count,
                max_rounds=self._preset.max_rounds,
                buyer_price=self._buyer_price,
                buyer_days=self._buyer_days,
                buyer_accepted=True,
                negotiation_done=True,
                cost_threshold=self._preset.cost_threshold,
                liquidity_threshold=self._preset.liquidity_threshold,
                volume=self._preset.volume,
                difficulty=self._difficulty,
                price_score=0.0,
                days_score=0.0,
                treds_bonus=0.15 if (use_treds or self._treds_used) else 0.0,
                step_reward=terminal_reward,
                message="Deal reached",
                reward=terminal_reward,
                done=True,
                metadata={"episode_id": self._state.episode_id},
            )

        price_jitter = self._rng.uniform(0.85, 1.15)
        price_drop = self._buyer_price * self._base_concede * price_jitter
        next_price = round(max(self._preset.cost_threshold, min(self._buyer_price, proposed_price) - price_drop), 2)
        day_drop = self._rng.randint(self._preset.day_step_low, self._preset.day_step_high)
        next_days = max(self._buyer_min_days_floor, self._buyer_days - day_drop)

        self._buyer_price = next_price
        self._buyer_days = next_days
        self._state.step_count += 1

        if self._buyer_price > self._preset.cost_threshold:
            step_reward += 0.02

        self._cumulative_reward += step_reward
        self._state.cumulative_reward = self._cumulative_reward
        self._state.buyer_price = self._buyer_price
        self._state.buyer_days = self._buyer_days
        self._state.message = message

        done = self._state.step_count >= self._preset.max_rounds
        if done:
            self._state.message = "Maximum rounds reached"
            terminal_reward = self._calculate_terminal_reward(use_treds)
            self._cumulative_reward = self._cumulative_reward - step_reward + terminal_reward
            self._state.cumulative_reward = self._cumulative_reward
            step_reward = terminal_reward

        return NegotiationObservation(
            round_number=self._state.step_count,
            max_rounds=self._preset.max_rounds,
            buyer_price=self._buyer_price,
            buyer_days=self._buyer_days,
            buyer_accepted=False,
            negotiation_done=done,
            cost_threshold=self._preset.cost_threshold,
            liquidity_threshold=self._preset.liquidity_threshold,
            volume=self._preset.volume,
            difficulty=self._difficulty,
            price_score=0.0,
            days_score=0.0,
            treds_bonus=0.15 if use_treds else 0.0,
            step_reward=step_reward,
            message=self._state.message,
            reward=step_reward,
            done=done,
            metadata={"episode_id": self._state.episode_id},
        )
