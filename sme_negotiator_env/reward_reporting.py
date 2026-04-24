"""Pure reward-reporting helpers for inference and demos.

These helpers are intentionally additive. They do not change any environment
reward semantics; they only explain how the currently selected path is scoring
an episode and, for the legacy single-deal path, compute a synthetic Stage 2
style shadow report for visibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from sme_negotiator_env.graders import (
    compute_npv_delta_vs_baseline,
    compute_shaping_rewards,
    compute_total_sme_reward,
    compute_verifiable_reward,
)
from sme_negotiator_env.models import BuyerState, NegotiationAction, NegotiationObservation, NegotiationState, SMEAccountState, WorldState
from sme_negotiator_env.task_config import TASK_REGISTRY, resolve_task_id


@dataclass(frozen=True)
class LegacyStepDiagnostics:
    """Supplemental diagnostics for one legacy inference step."""

    legacy_step_reward: float
    legacy_reward_branch: str
    legacy_terminal_score: Optional[float]
    close_zone_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowRewardReport:
    """Synthetic Stage 2-style report for a legacy single-deal episode."""

    synthetic_world_state: bool
    shadow_verifiable_reward: float
    shadow_shaping_rewards: list[float]
    shadow_shaping_total: float
    shadow_total_sme_reward: float
    npv_delta_vs_baseline: float
    effective_receivable_days: int
    legal_max_payment_days: int
    compliance_within_legal_cap: bool
    compliance_with_penalty_exception: bool
    default_flag: bool
    missed_supplier_payment: bool
    cash_balance: float
    required_minimum_cash: float
    credit_limit: float
    current_utilization: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Convert observation-like objects into plain dictionaries."""
    if observation is None:
        return {}
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return dict(observation)
    return dict(observation)


def reward_branch(observation: Any) -> str:
    """Return the legacy reward branch annotation from observation metadata."""
    obs = observation_to_dict(observation)
    metadata = obs.get("metadata") or {}
    if isinstance(metadata, dict):
        return str(metadata.get("reward_branch", "reset"))
    return "reset"


def close_zone_flag(
    observation: Any,
    *,
    last_valid_proposal: Optional[dict[str, Any]] = None,
) -> bool:
    """Return whether the latest public terms are already closeable for the SME."""
    obs = observation_to_dict(observation)
    liq = int(obs.get("liquidity_threshold", 0) or 0)
    buyer_days = int(obs.get("buyer_days", 0) or 0)
    buyer_price = float(obs.get("buyer_price", 0.0) or 0.0)
    cost = float(obs.get("cost_threshold", 0.0) or 0.0)
    in_zone = buyer_days <= liq and buyer_price >= cost
    if last_valid_proposal is None:
        return in_zone
    proposal_days = int(last_valid_proposal.get("payment_days", buyer_days) or buyer_days)
    proposal_price = float(last_valid_proposal.get("price", buyer_price) or buyer_price)
    return in_zone and proposal_days <= liq and proposal_price >= cost


def build_legacy_step_diagnostics(
    observation: Any,
    *,
    reward: float,
    last_valid_proposal: Optional[dict[str, Any]] = None,
) -> LegacyStepDiagnostics:
    """Build a supplemental per-step diagnostics payload for legacy inference."""
    obs = observation_to_dict(observation)
    return LegacyStepDiagnostics(
        legacy_step_reward=float(reward),
        legacy_reward_branch=reward_branch(obs),
        legacy_terminal_score=float(obs.get("reward")) if obs.get("done") else None,
        close_zone_flag=close_zone_flag(obs, last_valid_proposal=last_valid_proposal),
    )


def build_shadow_reward_report(
    *,
    reset_observation: Any,
    actions: list[NegotiationAction],
    step_observations: list[Any],
    seed: int,
    final_state: Optional[NegotiationState] = None,
) -> ShadowRewardReport:
    """Compute a synthetic Stage 2-style reward report for a legacy episode.

    The live single-deal path does not own a ``WorldState``. This helper
    therefore reconstructs a minimal deterministic world using the current task
    config plus public observations, then reuses the shared Stage 2 reward
    helpers for reporting only.
    """
    reset_obs = _coerce_observation(reset_observation)
    task_id = resolve_task_id(str(reset_obs.task_name or "payment-terms-medium"), difficulty=reset_obs.difficulty)
    task_config = TASK_REGISTRY[task_id]
    trajectory = _build_shadow_trajectory(
        reset_observation=reset_obs,
        actions=actions,
        step_observations=step_observations,
        seed=seed,
        final_state=final_state,
    )
    synthetic_world = _build_synthetic_world_state(task_config=task_config, final_state=trajectory[-1])
    shaping = compute_shaping_rewards(trajectory)
    return ShadowRewardReport(
        synthetic_world_state=True,
        shadow_verifiable_reward=round(compute_verifiable_reward(synthetic_world, trajectory), 6),
        shadow_shaping_rewards=shaping,
        shadow_shaping_total=round(sum(shaping), 6),
        shadow_total_sme_reward=round(
            compute_total_sme_reward(
                synthetic_world,
                trajectory,
                lambda_shaping=float(synthetic_world.reward_lambda_shaping),
            ),
            6,
        ),
        npv_delta_vs_baseline=round(compute_npv_delta_vs_baseline(synthetic_world, trajectory), 6),
        effective_receivable_days=_effective_receivable_days(trajectory[-1]),
        legal_max_payment_days=int(synthetic_world.legal_max_payment_days),
        compliance_within_legal_cap=_effective_receivable_days(trajectory[-1]) <= int(synthetic_world.legal_max_payment_days),
        compliance_with_penalty_exception=(
            _effective_receivable_days(trajectory[-1]) <= int(synthetic_world.legal_max_payment_days) + 7
            and bool(trajectory[-1].late_payment_penalty_agreed)
        ),
        default_flag=bool(
            synthetic_world.smes[0].defaulted
            or float(synthetic_world.smes[0].cash_balance) < 0.0
            or float(synthetic_world.smes[0].current_utilization) > float(synthetic_world.smes[0].credit_limit)
        ),
        missed_supplier_payment=bool(synthetic_world.smes[0].missed_supplier_payment),
        cash_balance=round(float(synthetic_world.smes[0].cash_balance), 6),
        required_minimum_cash=round(float(synthetic_world.smes[0].required_minimum_cash), 6),
        credit_limit=round(float(synthetic_world.smes[0].credit_limit), 6),
        current_utilization=round(float(synthetic_world.smes[0].current_utilization), 6),
    )


def _coerce_observation(observation: Any) -> NegotiationObservation:
    if isinstance(observation, NegotiationObservation):
        return observation
    return NegotiationObservation(**observation_to_dict(observation))


def _effective_receivable_days(state: NegotiationState) -> int:
    if state.agreed_terms is not None:
        return int(state.agreed_terms)
    return int(state.buyer_days)


def _build_shadow_trajectory(
    *,
    reset_observation: NegotiationObservation,
    actions: list[NegotiationAction],
    step_observations: list[Any],
    seed: int,
    final_state: Optional[NegotiationState],
) -> list[NegotiationState]:
    step_obs = [_coerce_observation(observation) for observation in step_observations]
    episode_id = str((reset_observation.metadata or {}).get("episode_id", f"shadow-{seed}"))
    cumulative_reward = 0.0
    initial_price = float(reset_observation.buyer_price)
    initial_days = int(reset_observation.buyer_days)

    trajectory: list[NegotiationState] = [
        _state_from_observation(
            observation=reset_observation,
            seed=seed,
            episode_id=episode_id,
            cumulative_reward=0.0,
            initial_buyer_price=initial_price,
            initial_buyer_days=initial_days,
            action=None,
            terminal_override=None,
        )
    ]

    for idx, (action, observation) in enumerate(zip(actions, step_obs), start=1):
        cumulative_reward += float(observation.reward or 0.0)
        terminal_override = final_state if idx == len(step_obs) else None
        trajectory.append(
            _state_from_observation(
                observation=observation,
                seed=seed,
                episode_id=episode_id,
                cumulative_reward=cumulative_reward,
                initial_buyer_price=initial_price,
                initial_buyer_days=initial_days,
                action=action,
                terminal_override=terminal_override,
            )
        )

    if len(trajectory) == 1:
        trajectory.append(trajectory[0].model_copy(deep=True))
    return trajectory


def _state_from_observation(
    *,
    observation: NegotiationObservation,
    seed: int,
    episode_id: str,
    cumulative_reward: float,
    initial_buyer_price: float,
    initial_buyer_days: int,
    action: Optional[NegotiationAction],
    terminal_override: Optional[NegotiationState],
) -> NegotiationState:
    if terminal_override is not None:
        return terminal_override.model_copy(
            deep=True,
            update={
                "cumulative_reward": round(float(cumulative_reward), 6),
                "buyer_price": float(observation.buyer_price),
                "buyer_days": int(observation.buyer_days),
                "message": str(observation.message),
            },
        )

    metadata = observation.metadata or {}
    action_payload = action.model_dump(exclude_none=True) if action is not None else {}
    terminal_success = bool(observation.done and observation.buyer_accepted and observation.negotiation_done)
    final_price = float(action_payload.get("price", observation.buyer_price)) if terminal_success else None
    final_days = int(action_payload.get("payment_days", observation.buyer_days)) if terminal_success else None
    use_treds = bool(action_payload.get("use_treds", False))
    late_penalty = bool(action_payload.get("propose_late_payment_penalty_clause", False))
    dynamic_discounting = bool(action_payload.get("propose_dynamic_discounting", False))
    dynamic_rate = float(action_payload.get("dynamic_discount_annual_rate", 0.0))

    return NegotiationState(
        episode_id=episode_id,
        seed=int(seed),
        difficulty=str(observation.difficulty),
        task_name=str(observation.task_name),
        step_count=int(observation.round_number),
        max_steps=int(observation.max_rounds),
        negotiation_round=int(observation.round_number),
        max_rounds=int(observation.max_rounds),
        deal_reached=terminal_success,
        final_price=round(final_price, 2) if final_price is not None else None,
        final_days=final_days,
        treds_used=use_treds,
        cumulative_reward=round(float(cumulative_reward), 6),
        buyer_price=round(float(observation.buyer_price), 2),
        buyer_days=int(observation.buyer_days),
        initial_buyer_days=int(initial_buyer_days),
        cost_threshold=round(float(observation.cost_threshold), 2),
        liquidity_threshold=int(observation.liquidity_threshold),
        volume=int(observation.volume),
        message=str(observation.message),
        sme_monthly_revenue=float(observation.sme_monthly_revenue),
        current_payment_terms_days=int(observation.current_payment_terms_days),
        sme_supplier_payment_days=int(observation.sme_supplier_payment_days),
        interest_rate_annual=float(observation.interest_rate_annual),
        buyer_power_score=float(observation.buyer_power_score),
        secondary_buyer_power=float(observation.secondary_buyer_power)
        if observation.secondary_buyer_power is not None
        else None,
        agreed_terms=final_days if terminal_success else None,
        late_payment_penalty_agreed=late_penalty,
        dynamic_discounting_agreed=dynamic_discounting,
        agreed_dynamic_discount_annual=dynamic_rate,
        sme_id="sme_0",
        buyer_id="buyer_0",
        financier_id="financier_0",
        deal_id=str(metadata.get("deal_id", "legacy-deal-0")),
    ).model_copy(
        update={
            "buyer_price": round(float(observation.buyer_price), 2),
            "buyer_days": int(observation.buyer_days),
            "initial_buyer_days": int(initial_buyer_days),
            "message": str(observation.message),
            "cumulative_reward": round(float(cumulative_reward), 6),
            "final_price": round(final_price, 2) if final_price is not None else None,
            "deal_reached": terminal_success,
            "agreed_terms": final_days if terminal_success else None,
            "treds_used": use_treds,
        }
    )


def _build_synthetic_world_state(*, task_config: Any, final_state: NegotiationState) -> WorldState:
    revenue = float(final_state.sme_monthly_revenue)
    required_minimum_cash = round(float(task_config.minimum_cash_buffer_ratio) * revenue, 2)
    initial_cash = float(task_config.initial_cash_balance_ratio) * revenue
    credit_limit = max(
        1.0,
        float(task_config.credit_limit_multiplier)
        * revenue
        * max(int(final_state.current_payment_terms_days) - int(final_state.sme_supplier_payment_days), 0)
        / 365.0,
    )

    effective_days = _effective_receivable_days(final_state)
    baseline_days = int(final_state.initial_buyer_days)
    days_relief = revenue * max(0, baseline_days - effective_days) / 365.0
    price_relief = max(0.0, float((final_state.final_price or final_state.buyer_price) - final_state.cost_threshold))
    price_relief *= max(int(final_state.volume), 1) * 0.15
    treds_relief = revenue * 0.04 if final_state.treds_used else 0.0
    cash_balance = round(initial_cash + days_relief + price_relief + treds_relief, 2)

    working_capital_gap = float(final_state.working_capital_gap)
    current_utilization = round(max(0.0, working_capital_gap - cash_balance), 2)
    missed_supplier_payment = bool(cash_balance < required_minimum_cash)
    default_flag = bool(cash_balance < 0.0 or current_utilization > credit_limit)

    return WorldState(
        smes=[
            SMEAccountState(
                sme_id=final_state.sme_id,
                cash_balance=cash_balance,
                supplier_payment_days=int(final_state.sme_supplier_payment_days),
                credit_limit=round(float(credit_limit), 2),
                current_utilization=current_utilization,
                risk_score=min(1.0, max(0.0, float(final_state.buyer_power_score))),
                required_minimum_cash=required_minimum_cash,
                defaulted=default_flag,
                missed_supplier_payment=missed_supplier_payment,
            )
        ],
        buyers=[
            BuyerState(
                buyer_id=final_state.buyer_id,
                demand_level=1.0,
                budget_per_period=float(initial_cash),
                default_tendency=float(task_config.primary_buyer_default_tendency),
                baseline_payment_days=int(final_state.initial_buyer_days),
            )
        ],
        financier=None,
        legal_max_payment_days=int(task_config.legal_max_payment_days),
        baseline_discount_rate=float(task_config.interest_rate_annual),
        reward_lambda_shaping=float(task_config.reward_lambda_shaping),
        current_period=0,
        total_periods=1,
        episode_step=max(int(final_state.step_count), 0),
    )
