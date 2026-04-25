"""Multi-agent negotiation environment — Theme #1 implementation.

MultiAgentNegotiationEnvironment extends SMELiquidityEnvironment with:

1. StrategicBuyerAgent(s) — latent type (cooperative/neutral/adversarial) +
   theory-of-mind belief tracking; coalition-aware concession logic.
2. BuyerCoalition — two-buyer coalition lifecycle: formation after joint stall,
   coordinated holdout, rational defection on split-deal offers.
3. FinancierCompetitionArena — competitive rate auction across 4 financier types;
   SME sees up to 3 sorted bids in the observation.
4. RegulatorAgent — MSMED Act / Section 43B(h) enforcement; SME can invoke as
   a strategic threat to shift adversarial buyers toward concession.
5. BeliefStateManager — deterministic ToM belief updates for all agents.
6. 4 new SME action types:
   invoke_regulator, request_financier_auction, propose_split_deal, signal_distress
7. MultiAgentObservation — superset of LiquidityObservation with opponent_signals,
   coalition_status, regulatory_warnings, financier_bids, sme_belief_estimate,
   and social_welfare_score.

All existing action types (propose/accept/reject/tool/simulate_plan/advance_period)
continue to work identically — backward compatible with all 138 existing tests.
"""

from __future__ import annotations

import os
import uuid
from random import Random
from typing import Any, Optional

from server.environment import SMELiquidityEnvironment
from sme_negotiator_env.agents.buyer_agent import StrategicBuyerAgent, make_strategic_buyer
from sme_negotiator_env.agents.coalition import BuyerCoalition, CoalitionFormationPolicy
from sme_negotiator_env.agents.financier_agent import FinancierCompetitionArena
from sme_negotiator_env.agents.regulator_agent import RegulatorAgent
from sme_negotiator_env.belief_state import BeliefStateManager, SMEBeliefState
from sme_negotiator_env.models import (
    CoalitionStatus,
    FinancierBid,
    MultiAgentObservation,
    NegotiationAction,
    NegotiationObservation,
    OpponentSignal,
    RegulatoryWarning,
)
from sme_negotiator_env.task_config import resolve_task_id


_RELATIONSHIP_PENALTY = -0.05   # one-time cost for invoking the regulator


class MultiAgentNegotiationEnvironment(SMELiquidityEnvironment):
    """Multi-agent negotiation environment (Theme #1).

    Inherits all single-agent behaviour from SMELiquidityEnvironment and adds
    the multi-agent layer on top. Existing callers that pass only legacy action
    types continue to work without any changes.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ma_rng: Random = Random()
        # StrategicBuyerAgent(s) — populated on reset()
        self._buyer_agents: list[StrategicBuyerAgent] = []
        # BuyerCoalition — None until second buyer is present
        self._coalition: Optional[BuyerCoalition] = None
        # Financier arena
        self._financier_arena: Optional[FinancierCompetitionArena] = None
        # Regulator agent
        self._regulator: RegulatorAgent = RegulatorAgent()
        # Theory-of-mind belief manager
        self._belief_manager: BeliefStateManager = BeliefStateManager()
        self._sme_belief: Optional[SMEBeliefState] = None
        # Per-episode signal accumulator (reset each step)
        self._pending_signals: list[OpponentSignal] = []
        self._pending_warnings: list[RegulatoryWarning] = []
        self._pending_bids: list[FinancierBid] = []
        self._last_step_reward: float = 0.0
        # Track previous SME price for belief updates
        self._prev_sme_price: Optional[float] = None
        # Regulator invocation already used this episode (one-time relationship cost)
        self._regulator_invoked: bool = False
        self._ma_episode_seed: int = 0

    # ------------------------------------------------------------------ #
    # reset()                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, **kwargs: Any) -> MultiAgentObservation:
        base_obs = super().reset(**kwargs)

        seed = int(kwargs.get("seed", 42))
        task_name = str(kwargs.get("task_name", "payment-terms-medium"))
        difficulty = resolve_task_id(task_name)
        try:
            from sme_negotiator_env.task_config import TASK_REGISTRY
            task_cfg = TASK_REGISTRY.get(task_name)
            difficulty = task_cfg.difficulty if task_cfg else "medium"
        except Exception:
            difficulty = "medium"

        self._ma_episode_seed = seed
        self._ma_rng = Random(seed + 7)
        self._pending_signals = []
        self._pending_warnings = []
        self._pending_bids = []
        self._last_step_reward = 0.0
        self._prev_sme_price = None
        self._regulator_invoked = False

        # Determine buyer IDs from world state
        buyer_ids: list[str] = []
        try:
            ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
            buyer_ids = [b.buyer_id for b in ws.buyers]
        except Exception:
            buyer_ids = ["buyer_0"]

        # Build StrategicBuyerAgent(s)
        self._buyer_agents = [
            make_strategic_buyer(
                bid,
                buyer_index=i,
                difficulty=difficulty,
                seed=seed,
                rng=Random(seed + i * 13),
            )
            for i, bid in enumerate(buyer_ids)
        ]

        # BuyerCoalition (only when ≥ 2 buyers)
        self._coalition = None
        if len(self._buyer_agents) >= 2:
            self._coalition = BuyerCoalition(
                self._buyer_agents[0],
                self._buyer_agents[1],
                CoalitionFormationPolicy(),
            )

        # Regulator
        self._regulator.reset()
        try:
            ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
            self._regulator = RegulatorAgent(
                legal_max_payment_days=int(ws.legal_max_payment_days),
            )
        except Exception:
            self._regulator = RegulatorAgent()

        # Financier arena
        try:
            ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
            financiers = ([ws.financier] if ws.financier else [])
            self._financier_arena = FinancierCompetitionArena(
                financiers,
                self._ma_rng,
            )
        except Exception:
            self._financier_arena = None

        # Initial SME belief
        self._sme_belief = self._belief_manager.initial_sme_belief(buyer_ids)

        return self._wrap_obs(base_obs)

    # ------------------------------------------------------------------ #
    # step()                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action: NegotiationAction) -> MultiAgentObservation:
        self._pending_signals = []
        self._pending_warnings = []
        self._pending_bids = []
        prev_price = self._prev_sme_price

        action_type = str(action.action_type).lower()

        # ---- Route new multi-agent action types ---- #
        if action_type == "invoke_regulator":
            return self._handle_invoke_regulator(action, prev_price)
        if action_type == "request_financier_auction":
            return self._handle_financier_auction(action)
        if action_type == "propose_split_deal":
            return self._handle_split_deal(action, prev_price)
        if action_type == "signal_distress":
            return self._handle_signal_distress(action, prev_price)

        # ---- Legacy action types: delegate to parent ---- #
        self._prev_sme_price = float(action.price) if action.price else None
        base_obs = super().step(action)

        # Post-step multi-agent hooks
        self._update_buyer_beliefs(action, prev_price)
        self._check_coalition_formation(action)
        self._run_auto_regulatory_monitoring()

        return self._wrap_obs(base_obs)

    # ------------------------------------------------------------------ #
    # state()                                                               #
    # ------------------------------------------------------------------ #

    def state(self) -> dict[str, Any]:  # type: ignore[override]
        base = super().state() if callable(getattr(super(), "state", None)) else {}
        if not isinstance(base, dict):
            base = {}
        coalition_dict = None
        if self._coalition is not None:
            coalition_dict = {
                "is_active": self._coalition.is_active,
                "buyer_ids": self._coalition.buyer_ids,
                "formed_at_round": self._coalition.formed_at_round,
                "joint_demand_days": self._coalition.joint_demand_days,
                "defection_risk": self._coalition.defection_risk(),
            }
        base["multi_agent"] = {
            "buyer_agents": [
                {"buyer_id": a.buyer_id, "latent_type": a.latent_type.name}
                for a in self._buyer_agents
            ],
            "coalition": coalition_dict,
            "sme_belief": self._sme_belief.to_dict() if self._sme_belief else {},
            "active_regulatory_warnings": len(self._regulator.active_warnings),
        }
        return base

    # ------------------------------------------------------------------ #
    # New action handlers                                                   #
    # ------------------------------------------------------------------ #

    def _handle_invoke_regulator(
        self,
        action: NegotiationAction,
        prev_price: Optional[float],
    ) -> MultiAgentObservation:
        """SME explicitly invokes MSME Samadhaan / regulatory threat."""
        deal_id = self._active_deal_id()
        buyer_id = self._active_buyer_id()

        try:
            ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
            warning = self._regulator.invoke(
                deal_id, buyer_id, action, ws,
                current_period=self._current_period(),
            )
        except Exception:
            warning = RegulatoryWarning(
                deal_id=deal_id,
                buyer_id=buyer_id,
                violation_type="exceeds_45_days",
                penalty_exposure_inr=0.0,
                section_reference="MSMED Act 2006 Sections 15-24",
                issued_by_sme=True,
            )

        self._pending_warnings.append(warning)

        # Modify the active buyer's equilibrium
        active_buyer = self._buyer_agent_for(buyer_id)
        if active_buyer is not None:
            self._regulator.modify_buyer_equilibrium(active_buyer, warning)

        # Update buyer beliefs — they now know regulator was invoked
        self._update_buyer_beliefs(action, prev_price)

        self._pending_signals.append(OpponentSignal(
            sender_id="regulator_0",
            sender_type="regulator",
            signal_type="regulatory_warning",
            payload=warning.model_dump(),
            round_number=self._current_round(),
        ))

        # Relationship penalty (one-time)
        relationship_penalty = 0.0
        if not self._regulator_invoked:
            relationship_penalty = _RELATIONSHIP_PENALTY
            self._regulator_invoked = True

        dummy_obs = self._make_dummy_obs(step_reward=relationship_penalty)
        return self._wrap_obs(dummy_obs)

    def _handle_financier_auction(self, action: NegotiationAction) -> MultiAgentObservation:
        """Run competitive financing auction; return top 3 bids."""
        deal_id = self._active_deal_id()
        distress = 0.0
        if self._sme_belief and self._buyer_agents:
            buyer_beliefs = [a.belief for a in self._buyer_agents]
            distress = max(
                (b.sme_distress_estimate for b in buyer_beliefs), default=0.0
            )

        bids: list[FinancierBid] = []
        if self._financier_arena is not None:
            try:
                ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
                deal_amount = 0.0
                for d in ws.deals:
                    if d.deal_id == deal_id:
                        deal_amount = float(d.invoice_amount or 0.0)
                        break
                bids = self._financier_arena.run_auction(
                    ws,
                    deal_id,
                    sme_distress_signal=distress,
                    invoice_amount=deal_amount,
                    episode_seed=self._ma_episode_seed,
                )[:3]
            except Exception:
                bids = []

        self._pending_bids = bids

        # Update SME belief with best rate found
        if bids and self._sme_belief is not None:
            self._sme_belief = self._belief_manager.update_sme_belief_on_auction_result(
                self._sme_belief, bids[0].annual_rate
            )

        for bid in bids:
            self._pending_signals.append(OpponentSignal(
                sender_id=bid.financier_id,
                sender_type="financier",
                signal_type="financier_bid",
                payload=bid.model_dump(),
                round_number=self._current_round(),
            ))

        dummy_obs = self._make_dummy_obs(step_reward=0.0)
        return self._wrap_obs(dummy_obs)

    def _handle_split_deal(
        self,
        action: NegotiationAction,
        prev_price: Optional[float],
    ) -> MultiAgentObservation:
        """SME proposes different terms to buyer_a vs buyer_b to break coalition."""
        defected = False
        defector_id: Optional[str] = None

        if self._coalition is not None and self._coalition.is_active:
            defected, defector_id = self._coalition.process_split_offer(action)
            if defected:
                self._pending_signals.append(OpponentSignal(
                    sender_id=defector_id or "coalition",
                    sender_type="coalition",
                    signal_type="coalition_dissolved",
                    payload={"defecting_buyer_id": defector_id},
                    round_number=self._current_round(),
                ))
            else:
                # Coalition held — emit defection_intent signal to warn SME
                self._pending_signals.append(OpponentSignal(
                    sender_id="coalition",
                    sender_type="coalition",
                    signal_type="defection_intent",
                    payload={"coalition_held": True},
                    round_number=self._current_round(),
                ))
        else:
            # No coalition active — treat as a regular multi-target propose
            pass

        self._update_buyer_beliefs(action, prev_price)
        dummy_obs = self._make_dummy_obs(step_reward=0.0)
        return self._wrap_obs(dummy_obs)

    def _handle_signal_distress(
        self,
        action: NegotiationAction,
        prev_price: Optional[float],
    ) -> MultiAgentObservation:
        """SME reveals partial cash position to financier; unlocks higher credit tier."""
        # Update all buyer beliefs — they observe the distress signal
        for buyer_agent in self._buyer_agents:
            buyer_agent.update_belief(
                action,
                prev_sme_price=prev_price,
                current_sme_price=float(action.price) if action.price else None,
            )

        self._pending_signals.append(OpponentSignal(
            sender_id="sme_0",
            sender_type="buyer",
            signal_type="distress_ack",
            payload={"level": str(action.distress_disclosure_level or "low")},
            round_number=self._current_round(),
        ))

        dummy_obs = self._make_dummy_obs(step_reward=0.0)
        return self._wrap_obs(dummy_obs)

    # ------------------------------------------------------------------ #
    # Post-step multi-agent hooks                                           #
    # ------------------------------------------------------------------ #

    def _update_buyer_beliefs(
        self,
        sme_action: NegotiationAction,
        prev_sme_price: Optional[float],
    ) -> None:
        """Update each buyer's belief based on the latest SME action."""
        current_price = float(sme_action.price) if sme_action.price else None
        for buyer_agent in self._buyer_agents:
            buyer_agent.update_belief(
                sme_action,
                prev_sme_price=prev_sme_price,
                current_sme_price=current_price,
            )
        self._prev_sme_price = current_price

    def _check_coalition_formation(self, sme_action: NegotiationAction) -> None:
        """Attempt coalition formation after joint stall."""
        if self._coalition is None or self._coalition.is_active:
            return
        if len(self._buyer_agents) < 2:
            return

        # Estimate SME concession this round (payment_days change)
        sme_concession = max(0, int(sme_action.payment_days or 0))
        formed = self._coalition.attempt_formation(
            round_number=self._current_round(),
            sme_last_concession_days=sme_concession,
        )
        if formed:
            self._pending_signals.append(OpponentSignal(
                sender_id="coalition",
                sender_type="coalition",
                signal_type="coalition_formed",
                payload={
                    "buyer_ids": self._coalition.buyer_ids,
                    "joint_demand_days": self._coalition.joint_demand_days,
                },
                round_number=self._current_round(),
            ))

    def _run_auto_regulatory_monitoring(self) -> None:
        """Auto-scan for MSMED Act violations after each step."""
        try:
            ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
            current_negs = getattr(self._liquidity_state, "current_negotiations", {})
            new_warnings = self._regulator.monitor_deals(
                ws,
                current_negs,
                current_period=self._current_period(),
            )
            self._pending_warnings.extend(new_warnings)
            for w in new_warnings:
                self._pending_signals.append(OpponentSignal(
                    sender_id="regulator_0",
                    sender_type="regulator",
                    signal_type="regulatory_warning",
                    payload=w.model_dump(),
                    round_number=self._current_round(),
                ))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Social welfare computation                                            #
    # ------------------------------------------------------------------ #

    def _compute_social_welfare(
        self,
        sme_reward: float,
        agreed_days: Optional[int],
        *,
        baseline_buyer_days: int = 90,
        total_invoice: float = 0.0,
    ) -> tuple[float, float, float]:
        """Returns (sme_surplus, buyer_surplus, social_welfare).

        Social welfare is logged only — never added to the RL reward.
        """
        sme_surplus = max(0.0, min(1.0, float(sme_reward)))

        buyer_surplus = 0.0
        if agreed_days is not None and baseline_buyer_days > 0:
            buyer_surplus = max(
                0.0,
                min(1.0, float(agreed_days) / float(baseline_buyer_days)),
            )

        penalty_total = sum(
            w.penalty_exposure_inr for w in self._regulator.active_warnings
        )
        systemic_health = 1.0
        if total_invoice > 0:
            systemic_health = max(
                0.0, 1.0 - min(1.0, penalty_total / float(total_invoice))
            )

        social_welfare = round(
            0.5 * sme_surplus + 0.3 * buyer_surplus + 0.2 * systemic_health, 4
        )
        return sme_surplus, buyer_surplus, social_welfare

    # ------------------------------------------------------------------ #
    # Observation wrapping                                                  #
    # ------------------------------------------------------------------ #

    def _wrap_obs(self, base_obs: NegotiationObservation) -> MultiAgentObservation:
        """Promote any NegotiationObservation to a MultiAgentObservation."""
        base_dict = (
            base_obs.model_dump()
            if hasattr(base_obs, "model_dump")
            else dict(vars(base_obs))
        )

        coalition_status: Optional[CoalitionStatus] = None
        if self._coalition is not None:
            coalition_status = CoalitionStatus(
                is_active=self._coalition.is_active,
                buyer_ids=self._coalition.buyer_ids,
                formed_at_round=self._coalition.formed_at_round,
                joint_demand_days=self._coalition.joint_demand_days,
                defection_risk=self._coalition.defection_risk(),
            )

        sme_belief_dict = self._sme_belief.to_dict() if self._sme_belief else {}

        agreed_days = base_dict.get("buyer_days")
        total_invoice = 0.0
        try:
            ws = self._liquidity_state.world_state  # type: ignore[attr-defined]
            total_invoice = sum(float(d.invoice_amount or 0) for d in ws.deals)
        except Exception:
            pass

        sme_surplus, buyer_surplus, sw = self._compute_social_welfare(
            sme_reward=float(base_dict.get("reward", 0.0)),
            agreed_days=agreed_days,
            total_invoice=total_invoice,
        )

        # Merge all fields; MultiAgentObservation is a superset of LiquidityObservation
        ma_fields = {
            "opponent_signals": list(self._pending_signals),
            "coalition_status": coalition_status,
            "regulatory_warnings": list(self._pending_warnings),
            "financier_bids": list(self._pending_bids),
            "sme_belief_estimate": sme_belief_dict,
            "social_welfare_score": sw,
            "buyer_surplus_estimate": round(buyer_surplus, 4),
        }
        base_dict.update(ma_fields)

        # Fill missing LiquidityObservation fields with safe defaults
        _LIQUIDITY_DEFAULTS = {
            "agent_type": "SME", "agent_id": "sme_0", "current_actor": "SME",
            "active_deal_id": None, "open_deal_ids": [], "resolved_deal_ids": [],
            "current_period": 0, "total_periods": 1, "episode_step": 0,
            "simulation_projection": None, "projected_balances": None,
            "projected_defaults": None, "projected_penalties": None,
            "last_tool_name": None, "last_tool_args": None, "last_tool_result": None,
            "reward_component_report": None, "history": [],
        }
        for k, v in _LIQUIDITY_DEFAULTS.items():
            base_dict.setdefault(k, v)

        try:
            return MultiAgentObservation(**base_dict)
        except Exception:
            # Fallback: strip unknown fields and retry
            known = MultiAgentObservation.model_fields
            filtered = {k: v for k, v in base_dict.items() if k in known}
            filtered.update(ma_fields)
            for k, v in _LIQUIDITY_DEFAULTS.items():
                filtered.setdefault(k, v)
            return MultiAgentObservation(**filtered)

    def _make_dummy_obs(self, *, step_reward: float = 0.0) -> NegotiationObservation:
        """Build a minimal pass-through observation for new action types."""
        from sme_negotiator_env.models import NegotiationObservation
        try:
            # Try to get the current base observation from parent state
            current = getattr(self, "_last_obs", None)
            if current is not None and isinstance(current, NegotiationObservation):
                if hasattr(current, "model_copy"):
                    return current.model_copy(
                        update={"step_reward": step_reward, "reward": step_reward}
                    )
        except Exception:
            pass
        # Minimal safe defaults
        return NegotiationObservation(
            round_number=self._current_round(),
            max_rounds=20,
            buyer_price=100.0,
            buyer_days=60,
            buyer_accepted=False,
            negotiation_done=False,
            cost_threshold=80.0,
            liquidity_threshold=45,
            volume=1000,
            difficulty="medium",
            price_score=0.0,
            days_score=0.0,
            treds_bonus=0.0,
            step_reward=step_reward,
            message="",
            reward=step_reward,
            done=False,
            metadata={},
        )

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                 #
    # ------------------------------------------------------------------ #

    def _active_deal_id(self) -> str:
        try:
            return str(self._liquidity_state.active_deal_id or "deal_0")  # type: ignore[attr-defined]
        except Exception:
            return "deal_0"

    def _active_buyer_id(self) -> str:
        try:
            return str(self._liquidity_state.active_buyer_id)  # type: ignore[attr-defined]
        except Exception:
            return "buyer_0"

    def _current_round(self) -> int:
        try:
            return int(self._liquidity_state.current_negotiation.negotiation_round)  # type: ignore[attr-defined]
        except Exception:
            return 0

    def _current_period(self) -> int:
        try:
            return int(self._liquidity_state.world_state.current_period)  # type: ignore[attr-defined]
        except Exception:
            return 0

    def _buyer_agent_for(self, buyer_id: str) -> Optional[StrategicBuyerAgent]:
        for agent in self._buyer_agents:
            if agent.buyer_id == buyer_id:
                return agent
        return self._buyer_agents[0] if self._buyer_agents else None
