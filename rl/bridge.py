"""In-process OpenEnv-to-TRL bridge for the liquidity environment."""

from __future__ import annotations

import json
from collections import defaultdict
from random import Random
from typing import Any, Optional

from server.environment import SMELiquidityEnvironment
from rl.curriculum import DifficultyConfig
from rl.opponents import OpponentPolicyManager
from rl.rubrics import PERSONAS, Persona, sample_persona
from sme_negotiator_env.graders import compute_reward_component_report
from sme_negotiator_env.llm_action_parser import parse_llm_text_to_negotiation_action
from sme_negotiator_env.models import LiquidityObservation, NegotiationAction
from sme_negotiator_env.prompting import (
    action_payload_to_model_action,
    conservative_default_action,
    format_observation_text,
    normalize_action_type,
    observation_to_dict,
)

from .episode_logging import EpisodeSummary, build_episode_log as _build_episode_log

_DEFAULT_FACTORY_CONFIG: dict[str, Any] = {
    "task_name": "liquidity-correlation-hard",
    "difficulty": "hard",
    "seed": 1000,
    "total_periods": 3,
    "prompt": (
        "You are an SME treasury agent. Complete the full liquidity episode. "
        "Use explicit tools, negotiate responsibly, advance macro periods when needed, and avoid default."
    ),
    "buyer_variance": 0.0,
    "financier_variance": 0.0,
    "curriculum_level": 0,
    "persona_mode": "off",
    "persona_name": None,
    "opponent_manager": None,
    "buyer_policy": None,
    "financier_policy": None,
    "buyer_policy_id": "heuristic_buyer",
    "financier_policy_id": "heuristic_financier",
    "lock_curriculum_config": False,
}

SUPPORTED_ACTION_TYPES: tuple[str, ...] = (
    "propose",
    "accept",
    "reject",
    "advance_period",
    "tool",
    "simulate_plan",
)
SUPPORTED_TOOL_NAMES: tuple[str, ...] = (
    "QUERY_TREDS",
    "CHECK_COMPLIANCE",
    "RUN_CASHFLOW_SIM",
)


def build_action_contract_text() -> str:
    """Return strict action-format instructions for training prompts."""
    return (
        "Output exactly one JSON object and nothing else. "
        f"Valid action_type values: {', '.join(SUPPORTED_ACTION_TYPES)}. "
        f"When action_type='tool', tool_name must be one of: {', '.join(SUPPORTED_TOOL_NAMES)}. "
        "Always include action_type. "
        "For propose/accept include price and payment_days. "
        "For tool include tool_args. "
        "For simulate_plan include simulation_plan and optional simulation_horizon. "
        "Do not write markdown, prose, or multiple actions."
    )


def format_observation(observation: Any, *, role: str = "sme", persona: Optional[str] = None) -> str:
    """Return a prompt-friendly string for a liquidity observation."""
    return format_observation_text(observation, role=role, persona=persona)


def _strip_json_fence(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return text


def parse_action(text: str, observation: Optional[Any] = None) -> NegotiationAction:
    """Parse offline model text into the canonical repo action schema."""
    candidate = _strip_json_fence(text)
    if candidate:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                normalized = dict(payload)
                normalized["action_type"] = normalize_action_type(normalized.get("action_type", "propose"))
                return action_payload_to_model_action(normalized, observation)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if observation is not None:
        parsed = parse_llm_text_to_negotiation_action(candidate, observation, allow_json=False)
        payload = parsed.model_dump()
        payload["action_type"] = normalize_action_type(payload.get("action_type", "propose"))
        return action_payload_to_model_action(payload, observation)

    return conservative_default_action()


def action_to_payload(action: NegotiationAction) -> dict[str, Any]:
    """Return a JSON-serializable action payload."""
    return action.model_dump(exclude_none=True)


def execute_action(wrapper: Any, action: NegotiationAction) -> str:
    """Dispatch a parsed action through the in-process wrapper."""
    payload = action_to_payload(action)
    action_type = str(payload.get("action_type", "propose")).lower()

    if action_type == "propose":
        return wrapper.propose(
            price=float(payload.get("price", 0.0) or 0.0),
            payment_days=int(payload.get("payment_days", 0) or 0),
            use_treds=bool(payload.get("use_treds", False)),
            deal_id=payload.get("deal_id"),
            reason=payload.get("reason"),
            propose_late_payment_penalty_clause=bool(payload.get("propose_late_payment_penalty_clause", False)),
            propose_dynamic_discounting=bool(payload.get("propose_dynamic_discounting", False)),
            dynamic_discount_annual_rate=float(payload.get("dynamic_discount_annual_rate", 0.0) or 0.0),
        )

    if action_type == "accept":
        return wrapper.accept(
            price=float(payload.get("price", 0.0) or 0.0),
            payment_days=int(payload.get("payment_days", 0) or 0),
            use_treds=bool(payload.get("use_treds", False)),
            deal_id=payload.get("deal_id"),
            reason=payload.get("reason"),
            propose_late_payment_penalty_clause=bool(payload.get("propose_late_payment_penalty_clause", False)),
            propose_dynamic_discounting=bool(payload.get("propose_dynamic_discounting", False)),
            dynamic_discount_annual_rate=float(payload.get("dynamic_discount_annual_rate", 0.0) or 0.0),
        )

    if action_type == "reject":
        return wrapper.reject(deal_id=payload.get("deal_id"), reason=payload.get("reason"))

    if action_type == "advance_period":
        return wrapper.advance_period()

    if action_type == "simulate_plan":
        return wrapper.simulate_plan(
            plan=dict(payload.get("simulation_plan", {}) or {}),
            horizon=payload.get("simulation_horizon"),
            deal_id=payload.get("deal_id"),
        )

    if action_type != "tool":
        raise ValueError(f"Unsupported action_type: {action_type!r}")

    tool_name = str(payload.get("tool_name", "") or "")
    tool_args = dict(payload.get("tool_args", {}) or {})
    deal_id = payload.get("deal_id")
    if tool_name == "QUERY_TREDS":
        invoice_id = str(tool_args.get("invoice_id", deal_id or "default"))
        return wrapper.query_treds(invoice_id=invoice_id, deal_id=deal_id)
    if tool_name == "CHECK_COMPLIANCE":
        contract_id = str(tool_args.get("contract_id", deal_id or "default"))
        return wrapper.check_compliance(contract_id=contract_id, deal_id=deal_id)
    if tool_name == "RUN_CASHFLOW_SIM":
        return wrapper.run_cashflow_sim(
            plan=dict(tool_args.get("plan", {}) or {}),
            horizon=tool_args.get("horizon"),
            deal_id=deal_id,
        )
    raise ValueError(f"Unsupported tool_name: {tool_name!r}")


def _coerce_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(prompt or "")


class InProcessEnvWrapper:
    """TRL-friendly zero-arg in-process wrapper for SMELiquidityEnvironment."""

    _factory_defaults: dict[str, Any] = dict(_DEFAULT_FACTORY_CONFIG)

    def __init__(self) -> None:
        self.env: Optional[SMELiquidityEnvironment] = None
        self.last_observation: Optional[LiquidityObservation] = None
        self.done: bool = False
        self.prompt_text: str = ""
        self.task_name: str = str(self._factory_defaults["task_name"])
        self.difficulty: str = str(self._factory_defaults["difficulty"])
        self.seed: int = int(self._factory_defaults["seed"])
        self.total_periods: int = int(self._factory_defaults["total_periods"])
        self.reset_kwargs: dict[str, Any] = {}
        self.tool_counts: dict[str, int] = defaultdict(int)
        self.episode_log_parts: list[str] = []
        self.env_reward_total: float = 0.0
        self.tool_bonus_total: float = 0.0
        self.buyer_variance: float = float(self._factory_defaults.get("buyer_variance", 0.0))
        self.financier_variance: float = float(self._factory_defaults.get("financier_variance", 0.0))
        self.curriculum_level: int = int(self._factory_defaults.get("curriculum_level", 0))
        self.current_persona: Optional[Persona] = None
        self.persona_mode: str = str(self._factory_defaults.get("persona_mode", "off"))
        self.persona_name: Optional[str] = self._factory_defaults.get("persona_name")
        self.opponent_manager: Optional[OpponentPolicyManager] = self._factory_defaults.get("opponent_manager")
        self.buyer_policy = self._factory_defaults.get("buyer_policy")
        self.financier_policy = self._factory_defaults.get("financier_policy")
        self.buyer_policy_id: str = str(self._factory_defaults.get("buyer_policy_id", "heuristic_buyer"))
        self.financier_policy_id: str = str(self._factory_defaults.get("financier_policy_id", "heuristic_financier"))
        self.last_default_flag: bool = False
        self._reset_count: int = 0

    @property
    def reward(self) -> float:
        """Accumulated per-step env reward; read by generic TRL reward functions."""
        return self.env_reward_total

    def reset(self, **kwargs: Any) -> str:
        """Reset the liquidity environment from dataset row fields.

        Args:
            **kwargs: Dataset row fields such as `prompt`, `task_name`,
                `difficulty`, `seed`, and `total_periods`.

        Returns:
            The formatted initial observation string.
        """
        config = dict(self._factory_defaults)
        config.update(kwargs)
        self.prompt_text = _coerce_prompt_text(config.get("prompt", ""))
        self.task_name = str(config.get("task_name", self._factory_defaults["task_name"]))
        self.difficulty = str(config.get("difficulty", self._factory_defaults["difficulty"]))
        self.seed = int(config.get("seed", self._factory_defaults["seed"]))
        lock_curriculum_config = bool(config.get("lock_curriculum_config", False))
        if lock_curriculum_config:
            self.total_periods = int(self._factory_defaults["total_periods"])
            self.buyer_variance = float(self._factory_defaults.get("buyer_variance", 0.0))
            self.financier_variance = float(self._factory_defaults.get("financier_variance", 0.0))
            self.curriculum_level = int(self._factory_defaults.get("curriculum_level", 0))
        else:
            self.total_periods = int(config.get("total_periods", self._factory_defaults["total_periods"]))
            self.buyer_variance = float(config.get("buyer_variance", self._factory_defaults.get("buyer_variance", 0.0)))
            self.financier_variance = float(
                config.get("financier_variance", self._factory_defaults.get("financier_variance", 0.0))
            )
            self.curriculum_level = int(config.get("curriculum_level", self._factory_defaults.get("curriculum_level", 0)))
        self.persona_mode = str(config.get("persona_mode", self._factory_defaults.get("persona_mode", "off")))
        self.persona_name = config.get("persona_name", self._factory_defaults.get("persona_name"))
        self.opponent_manager = config.get("opponent_manager", self._factory_defaults.get("opponent_manager"))
        self.buyer_policy = config.get("buyer_policy", self._factory_defaults.get("buyer_policy"))
        self.financier_policy = config.get("financier_policy", self._factory_defaults.get("financier_policy"))
        self.reset_kwargs = dict(config)
        self._reset_count += 1

        if self.opponent_manager is not None and (self.buyer_policy is None or self.financier_policy is None):
            (
                sampled_buyer_policy,
                sampled_financier_policy,
                sampled_buyer_policy_id,
                sampled_financier_policy_id,
            ) = self.opponent_manager.sample_policies(seed=self.seed + self._reset_count)
            if self.buyer_policy is None:
                self.buyer_policy = sampled_buyer_policy
            if self.financier_policy is None:
                self.financier_policy = sampled_financier_policy
            self.buyer_policy_id = str(config.get("buyer_policy_id", sampled_buyer_policy_id))
            self.financier_policy_id = str(config.get("financier_policy_id", sampled_financier_policy_id))
        else:
            self.buyer_policy_id = str(
                config.get(
                    "buyer_policy_id",
                    getattr(self.buyer_policy, "policy_id", self._factory_defaults.get("buyer_policy_id", "heuristic_buyer")),
                )
            )
            self.financier_policy_id = str(
                config.get(
                    "financier_policy_id",
                    getattr(
                        self.financier_policy,
                        "policy_id",
                        self._factory_defaults.get("financier_policy_id", "heuristic_financier"),
                    ),
                )
            )

        self.current_persona = self._resolve_persona()

        self.env = SMELiquidityEnvironment(
            total_periods=self.total_periods,
            buyer_policy=self.buyer_policy,
            financier_policy=self.financier_policy,
            buyer_variance=self.buyer_variance,
            financier_variance=self.financier_variance,
        )
        self.last_observation = self.env.reset(
            seed=self.seed,
            difficulty=self.difficulty,
            task_name=self.task_name,
        )
        self.done = bool(self.last_observation.done)
        self.tool_counts = defaultdict(int)
        self.episode_log_parts = []
        self.env_reward_total = 0.0
        self.tool_bonus_total = 0.0
        self.last_default_flag = False

        if self.prompt_text:
            self.episode_log_parts.append(f"PROMPT {self.prompt_text}")
        self.episode_log_parts.append(
            "STAGE6 "
            f"curriculum_level={self.curriculum_level} "
            f"buyer_variance={self.buyer_variance:.3f} "
            f"financier_variance={self.financier_variance:.3f} "
            f"persona={getattr(self.current_persona, 'name', None)} "
            f"buyer_policy={self.buyer_policy_id} "
            f"financier_policy={self.financier_policy_id}"
        )
        self.episode_log_parts.append(f"RESET {format_observation(self.last_observation)}")
        return format_observation(self.last_observation)

    def _resolve_persona(self) -> Optional[Persona]:
        if self.persona_mode == "off":
            return None
        if self.persona_mode == "fixed":
            if self.persona_name:
                for persona in PERSONAS:
                    if persona.name == self.persona_name:
                        return persona
            return PERSONAS[0]
        rng = Random(self.seed + self._reset_count * 997)
        return sample_persona(rng)

    def _require_live_episode(self) -> None:
        if self.env is None or self.last_observation is None:
            raise RuntimeError("Environment has not been reset.")
        if self.done:
            raise ValueError("Episode already completed.")

    def _apply_action(self, action: NegotiationAction, *, count_tool_name: Optional[str] = None) -> str:
        self._require_live_episode()
        assert self.env is not None
        observation = self.env.step(action)
        self.last_observation = observation
        self.done = bool(observation.done)
        self.env_reward_total += float(observation.reward or 0.0)
        metadata = observation.metadata or {}
        self.tool_bonus_total += float(metadata.get("tool_bonus_applied", 0.0))
        self.last_default_flag = bool(metadata.get("defaulted_sme_count", 0) or False)
        if count_tool_name is not None:
            self.tool_counts[count_tool_name] += 1

        self.episode_log_parts.append(
            "ACTION "
            + json.dumps(action.model_dump(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
        self.episode_log_parts.append(f"OBS {format_observation(observation)}")
        return format_observation(observation)

    def propose(
        self,
        price: float,
        payment_days: int,
        use_treds: bool = False,
        deal_id: Optional[str] = None,
        reason: Optional[str] = None,
        propose_late_payment_penalty_clause: bool = False,
        propose_dynamic_discounting: bool = False,
        dynamic_discount_annual_rate: float = 0.0,
    ) -> str:
        """Submit a counter-offer to the active or specified deal.

        Args:
            price: Proposed unit price.
            payment_days: Proposed payment tenor in days.
            use_treds: Whether to request TReDS financing.
            deal_id: Optional target deal id.
            reason: Optional short rationale.
            propose_late_payment_penalty_clause: Request a penalty clause.
            propose_dynamic_discounting: Request dynamic discounting.
            dynamic_discount_annual_rate: Annualized dynamic discount rate.

        Returns:
            The next formatted observation string.
        """
        return self._apply_action(
            NegotiationAction(
                action_type="propose",
                price=price,
                payment_days=payment_days,
                use_treds=use_treds,
                deal_id=deal_id,
                reason=reason,
                propose_late_payment_penalty_clause=propose_late_payment_penalty_clause,
                propose_dynamic_discounting=propose_dynamic_discounting,
                dynamic_discount_annual_rate=dynamic_discount_annual_rate,
            )
        )

    def accept(
        self,
        price: float,
        payment_days: int,
        use_treds: bool = False,
        deal_id: Optional[str] = None,
        reason: Optional[str] = None,
        propose_late_payment_penalty_clause: bool = False,
        propose_dynamic_discounting: bool = False,
        dynamic_discount_annual_rate: float = 0.0,
    ) -> str:
        """Accept terms for the active or specified deal.

        Args:
            price: Accepted unit price.
            payment_days: Accepted payment tenor.
            use_treds: Whether financing is accepted.
            deal_id: Optional target deal id.
            reason: Optional short rationale.
            propose_late_payment_penalty_clause: Keep penalty clause if needed.
            propose_dynamic_discounting: Keep dynamic discounting if needed.
            dynamic_discount_annual_rate: Dynamic discount rate.

        Returns:
            The next formatted observation string.
        """
        return self._apply_action(
            NegotiationAction(
                action_type="accept",
                price=price,
                payment_days=payment_days,
                use_treds=use_treds,
                deal_id=deal_id,
                reason=reason,
                propose_late_payment_penalty_clause=propose_late_payment_penalty_clause,
                propose_dynamic_discounting=propose_dynamic_discounting,
                dynamic_discount_annual_rate=dynamic_discount_annual_rate,
            )
        )

    def reject(self, deal_id: Optional[str] = None, reason: Optional[str] = None) -> str:
        """Reject the active or specified deal.

        Args:
            deal_id: Optional target deal id.
            reason: Optional short rationale.

        Returns:
            The next formatted observation string.
        """
        observation = self.last_observation
        default_price = float(observation.buyer_price) if observation is not None else 0.0
        default_days = int(observation.buyer_days) if observation is not None else 0
        return self._apply_action(
            NegotiationAction(
                action_type="reject",
                price=default_price,
                payment_days=default_days,
                deal_id=deal_id,
                reason=reason,
            )
        )

    def query_treds(self, invoice_id: str, deal_id: Optional[str] = None) -> str:
        """Query deterministic TReDS quotes for a deal-backed invoice.

        Args:
            invoice_id: The invoice or deal identifier.
            deal_id: Optional explicit deal id.

        Returns:
            The next formatted observation string.
        """
        return self._apply_action(
            NegotiationAction(
                action_type="tool",
                deal_id=deal_id,
                tool_name="QUERY_TREDS",
                tool_args={"invoice_id": invoice_id, "deal_id": deal_id or invoice_id},
            ),
            count_tool_name="QUERY_TREDS",
        )

    def check_compliance(self, contract_id: str, deal_id: Optional[str] = None) -> str:
        """Run deterministic compliance checks on a deal or contract.

        Args:
            contract_id: The contract or deal identifier.
            deal_id: Optional explicit deal id.

        Returns:
            The next formatted observation string.
        """
        return self._apply_action(
            NegotiationAction(
                action_type="tool",
                deal_id=deal_id,
                tool_name="CHECK_COMPLIANCE",
                tool_args={"contract_id": contract_id, "deal_id": deal_id or contract_id},
            ),
            count_tool_name="CHECK_COMPLIANCE",
        )

    def run_cashflow_sim(
        self,
        plan: dict[str, Any],
        horizon: Optional[int] = None,
        deal_id: Optional[str] = None,
    ) -> str:
        """Run the deterministic macro cashflow simulator.

        Args:
            plan: Planning payload for the simulator. Expected JSON structure::

                {
                  "deal_decisions": {
                    "<deal_id>": {
                      "decision": "accept",
                      "price": 95.0,
                      "payment_days": 45,
                      "use_treds": true
                    }
                  },
                  "financing": {"<deal_id>": true},
                  "advance_periods": 1
                }

            horizon: Number of macro periods to simulate (default: total_periods).
            deal_id: Optional deal scoping hint for the active negotiation.

        Returns:
            The next formatted observation string including projected cash balances,
            default flags, and penalty exposures per period.
        """
        tool_args: dict[str, Any] = {"plan": plan}
        if horizon is not None:
            tool_args["horizon"] = horizon
        if deal_id is not None:
            tool_args["deal_id"] = deal_id
        return self._apply_action(
            NegotiationAction(
                action_type="tool",
                deal_id=deal_id,
                tool_name="RUN_CASHFLOW_SIM",
                tool_args=tool_args,
            ),
            count_tool_name="RUN_CASHFLOW_SIM",
        )

    def simulate_plan(
        self,
        plan: dict[str, Any],
        horizon: Optional[int] = None,
        deal_id: Optional[str] = None,
    ) -> str:
        """Run a read-only plan simulation without consuming a tool bonus slot.

        Args:
            plan: Deterministic planning payload for the macro simulator.
            horizon: Optional number of periods to simulate ahead.
            deal_id: Optional active deal hint.

        Returns:
            The next formatted observation string with projected balances,
            defaults, and penalties, while leaving the real world state unchanged.
        """
        return self._apply_action(
            NegotiationAction(
                action_type="simulate_plan",
                deal_id=deal_id,
                simulation_plan=plan,
                simulation_horizon=horizon,
            )
        )

    def advance_period(self) -> str:
        """Advance the macro period by one step.

        Returns:
            The next formatted observation string.
        """
        return self._apply_action(NegotiationAction(action_type="advance_period"))

    def _weighted_deal_metrics(self) -> tuple[float, float, list[float]]:
        if self.env is None or self.env.state is None:
            return 0.0, 0.0, []

        state = self.env.state
        world_state = state.world_state
        deal_map = {deal.deal_id: deal for deal in world_state.deals}
        weighted_reward = 0.0
        weighted_npv_delta = 0.0
        total_weight = 0.0
        final_payment_days: list[float] = []

        for deal_id, trajectory in state.deal_trajectories.items():
            if not trajectory:
                continue
            deal = deal_map.get(deal_id)
            weight = float(deal.invoice_amount) if deal is not None and float(deal.invoice_amount) > 0.0 else 1.0
            report = compute_reward_component_report(
                world_state,
                trajectory,
                lambda_shaping=float(world_state.reward_lambda_shaping),
            )
            weighted_reward += weight * float(report.total_reward)
            weighted_npv_delta += weight * float(report.npv_delta_vs_baseline)
            total_weight += weight
            payment_days = None
            if deal is not None and deal.agreed_payment_days is not None:
                payment_days = float(deal.agreed_payment_days)
            elif trajectory[-1].agreed_terms is not None:
                payment_days = float(trajectory[-1].agreed_terms)
            elif trajectory[-1].buyer_days is not None:
                payment_days = float(trajectory[-1].buyer_days)
            if payment_days is not None:
                final_payment_days.append(payment_days)

        if total_weight <= 0.0:
            return 0.0, 0.0, final_payment_days
        return weighted_reward / total_weight, weighted_npv_delta / total_weight, final_payment_days

    def _weighted_component_metrics(self) -> tuple[float, float, float, list[float]]:
        if self.env is None or self.env.state is None:
            return 0.0, 0.0, 0.0, []

        state = self.env.state
        world_state = state.world_state
        deal_map = {deal.deal_id: deal for deal in world_state.deals}
        weighted_base_reward = 0.0
        weighted_verifiable_reward = 0.0
        weighted_npv_delta = 0.0
        total_weight = 0.0
        final_payment_days: list[float] = []

        for deal_id, trajectory in state.deal_trajectories.items():
            if not trajectory:
                continue
            deal = deal_map.get(deal_id)
            weight = float(deal.invoice_amount) if deal is not None and float(deal.invoice_amount) > 0.0 else 1.0
            report = compute_reward_component_report(
                world_state,
                trajectory,
                lambda_shaping=float(world_state.reward_lambda_shaping),
            )
            weighted_base_reward += weight * float(report.total_reward)
            weighted_verifiable_reward += weight * float(report.verifiable_reward)
            weighted_npv_delta += weight * float(report.npv_delta_vs_baseline)
            total_weight += weight
            if deal is not None and deal.agreed_payment_days is not None:
                final_payment_days.append(float(deal.agreed_payment_days))
            elif trajectory[-1].agreed_terms is not None:
                final_payment_days.append(float(trajectory[-1].agreed_terms))
            elif trajectory[-1].buyer_days is not None:
                final_payment_days.append(float(trajectory[-1].buyer_days))

        if total_weight <= 0.0:
            return 0.0, 0.0, 0.0, final_payment_days
        return (
            weighted_base_reward / total_weight,
            weighted_verifiable_reward / total_weight,
            weighted_npv_delta / total_weight,
            final_payment_days,
        )

    def compute_final_reward(self) -> float:
        """Compute the final RL reward used by training-side reward functions."""
        base_reward, _, _ = self._weighted_deal_metrics()
        return round(float(base_reward) + float(self.tool_bonus_total), 6)

    def build_episode_log(self) -> str:
        """Return a deterministic text log of the current episode."""
        return _build_episode_log(self)

    def summarize_episode(self) -> EpisodeSummary:
        """Summarize deterministic episode metrics for logging."""
        base_reward, weighted_npv_delta, final_payment_days = self._weighted_deal_metrics()
        _, verifiable_reward, _, _ = self._weighted_component_metrics()
        if self.env is None or self.env.state is None:
            return EpisodeSummary(
                episode_completed=False,
                base_rl_reward=0.0,
                tool_bonus_total=0.0,
                env_reward_total=0.0,
                success_no_default_positive_npv=False,
                average_final_payment_days=0.0,
                tool_usage_count=0,
                resolved_deal_count=0,
                defaulted_sme_count=0,
                verifiable_reward=0.0,
                total_reward=0.0,
                tool_call_count=0,
                tool_effective_count=0,
                duplicate_tool_count=0,
                invalid_action_count=0,
                stall_step_count=0,
                terminated_by_step_cap=False,
                tool_backend_mode=None,
            )

        state = self.env.state
        world_state = state.world_state
        defaulted_sme_count = sum(1 for sme in world_state.smes if sme.defaulted)
        self.last_default_flag = bool(defaulted_sme_count > 0)
        average_payment_days = (
            sum(final_payment_days) / len(final_payment_days)
            if final_payment_days
            else 0.0
        )
        tool_call_count = int(getattr(state, "tool_call_count", sum(int(count) for count in self.tool_counts.values())))
        tool_effective_count = int(getattr(state, "tool_effective_count", 0))
        duplicate_tool_count = int(getattr(state, "duplicate_tool_count", 0))
        invalid_action_count = int(getattr(state, "invalid_action_count", 0))
        stall_step_count = int(getattr(state, "stall_step_count", 0))
        terminated_by_step_cap = bool(getattr(state, "terminated_by_step_cap", False))
        total_reward = round(float(base_reward) + float(self.tool_bonus_total), 6)
        return EpisodeSummary(
            episode_completed=bool(self.done),
            base_rl_reward=round(float(base_reward), 6),
            tool_bonus_total=round(float(self.tool_bonus_total), 6),
            env_reward_total=round(float(self.env_reward_total), 6),
            success_no_default_positive_npv=bool(defaulted_sme_count == 0 and weighted_npv_delta > 0.0),
            average_final_payment_days=round(float(average_payment_days), 6),
            tool_usage_count=tool_call_count,
            resolved_deal_count=len(state.resolved_deal_ids),
            defaulted_sme_count=defaulted_sme_count,
            curriculum_level=self.curriculum_level,
            persona_name=getattr(self.current_persona, "name", None),
            buyer_policy_id=self.buyer_policy_id,
            financier_policy_id=self.financier_policy_id,
            verifiable_reward=round(float(verifiable_reward), 6),
            total_reward=total_reward,
            tool_call_count=tool_call_count,
            tool_effective_count=tool_effective_count,
            duplicate_tool_count=duplicate_tool_count,
            invalid_action_count=invalid_action_count,
            stall_step_count=stall_step_count,
            terminated_by_step_cap=terminated_by_step_cap,
            tool_backend_mode=str(getattr(state, "tool_backend_mode", None) or ""),
        )


def make_environment_factory(**factory_defaults: Any):
    """Return a zero-arg factory/class compatible with TRL environment_factory."""
    merged_defaults = dict(_DEFAULT_FACTORY_CONFIG)
    merged_defaults.update(factory_defaults)

    class ConfiguredInProcessEnvWrapper(InProcessEnvWrapper):
        _factory_defaults = merged_defaults

    ConfiguredInProcessEnvWrapper.__name__ = "ConfiguredInProcessEnvWrapper"
    return ConfiguredInProcessEnvWrapper


# ======================================================================= #
# NegotiatorEnvFactory — GRPOTrainer environment_factory drop-in           #
# ======================================================================= #

class NegotiatorEnvFactory:
    """GRPOTrainer-compatible environment factory for SME negotiation.

    One instance is created per generation rollout by GRPOTrainer when used
    as ``environment_factory=NegotiatorEnvFactory``.

    Public methods become tools exposed to the model.
    ``reset(**kwargs)`` is called once per episode with dataset row fields.
    ``raise ValueError`` inside a method to signal episode termination.

    The ``reward_breakdown`` property exposes per-component signals for the
    split reward functions in rl/reward_functions.py.
    """

    _factory_defaults: dict[str, Any] = dict(_DEFAULT_FACTORY_CONFIG)

    def __init__(self) -> None:
        self._wrapper = InProcessEnvWrapper()
        self._last_bd: Optional["RewardBreakdown"] = None
        self._step_breakdowns: list[Any] = []

    @property
    def reward(self) -> float:
        """Accumulated per-step reward. Read by TRL reward functions."""
        return self._wrapper.compute_final_reward()

    @property
    def reward_breakdown(self) -> Any:
        """Latest RewardBreakdown for split reward functions.

        Returns the last breakdown built from observation metadata.
        Falls back to a zero breakdown if none is available.
        """
        from sme_negotiator_env.reward_breakdown import RewardBreakdown, merge_breakdown_list
        if self._step_breakdowns:
            return merge_breakdown_list(self._step_breakdowns)
        if self._last_bd is not None:
            return self._last_bd
        return RewardBreakdown(
            total=self.reward,
            is_terminal=True,
            termination_reason="no_breakdown_available",
        )

    def _extract_and_cache_breakdown(self, obs_text_or_obs: Any) -> None:
        """Extract RewardBreakdown from observation metadata if present."""
        try:
            from sme_negotiator_env.reward_breakdown import RewardBreakdown
            obs = self._wrapper.last_observation
            if obs is None:
                return
            bd_dict = (obs.metadata or {}).get("reward_breakdown")
            if isinstance(bd_dict, dict):
                # Build from dict — only known fields
                bd = RewardBreakdown(
                    total=float(bd_dict.get("total", self.reward)),
                    solvency=float(bd_dict.get("solvency", 0.0)),
                    liquidity=float(bd_dict.get("liquidity", 0.0)),
                    npv=float(bd_dict.get("npv", 0.0)),
                    compliance=float(bd_dict.get("compliance", 0.0)),
                    gap_term=float(bd_dict.get("gap_term", 0.0)),
                    days_term=float(bd_dict.get("days_term", 0.0)),
                    alignment_term=float(bd_dict.get("alignment_term", 0.0)),
                    reasoning_quality=float(bd_dict.get("reasoning_quality", 0.0)),
                    tool_strategic_use=float(bd_dict.get("tool_strategic_use", 0.0)),
                    format_compliance=float(bd_dict.get("format_compliance", 0.0)),
                    proposal_loop_penalty=float(bd_dict.get("proposal_loop_penalty", 0.0)),
                    invalid_accept_penalty=float(bd_dict.get("invalid_accept_penalty", 0.0)),
                    tool_dedup_penalty=float(bd_dict.get("tool_dedup_penalty", 0.0)),
                    is_terminal=bool(bd_dict.get("is_terminal", False)),
                    termination_reason=str(bd_dict.get("termination_reason", "")),
                    episode_step=int(bd_dict.get("episode_step", 0)),
                )
                self._last_bd = bd
                self._step_breakdowns.append(bd)
        except Exception:
            pass

    def reset(self, **kwargs: Any) -> str:
        """Initialize a new negotiation episode.

        Receives dataset row fields from GRPOTrainer (task_name, difficulty,
        seed, total_periods, prompt, etc.).

        Returns:
            The formatted initial observation string shown to the model.
        """
        self._last_bd = None
        self._step_breakdowns = []
        merged = dict(self._factory_defaults)
        merged.update(kwargs)
        return self._wrapper.reset(**merged)

    def propose_terms(
        self,
        price: float,
        payment_days: int,
        use_treds: bool = False,
        reason: str = "",
    ) -> str:
        """Propose price and payment terms to the buyer.

        Args:
            price: Proposed price per unit in INR. Must be >= cost threshold.
            payment_days: Proposed settlement period in days. Lower is better for SME.
            use_treds: Whether to activate TReDS invoice financing.
            reason: Brief justification citing financial context (e.g. WCG, interest rate).

        Returns:
            Buyer counter-offer message with updated days and price.
        """
        obs_text = self._wrapper.propose(
            price=price,
            payment_days=payment_days,
            use_treds=use_treds,
            reason=reason or None,
        )
        self._extract_and_cache_breakdown(obs_text)
        obs = self._wrapper.last_observation
        if obs is not None and bool(obs.done) and "validator_invalid" in str(
            (obs.metadata or {}).get("termination_reason", "")
        ):
            raise ValueError(str((obs.metadata or {}).get("message", "Invalid action")))
        return obs_text

    def accept_offer(self, price: float, payment_days: int) -> str:
        """Accept the current buyer offer to close the deal.

        Must match the last buyer counter-offer or your last proposal exactly.

        Args:
            price: Accepted price per unit. Must match last offer on the table.
            payment_days: Accepted settlement period. Must match last offer on the table.

        Returns:
            Deal confirmation message with final agreed terms and episode score.
        """
        obs_text = self._wrapper.accept(price=price, payment_days=payment_days)
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def reject_offer(self, reason: str = "") -> str:
        """Reject the current buyer offer and end the negotiation.

        Use only if no acceptable deal can be reached. Rejection yields zero reward.

        Args:
            reason: Brief explanation for the rejection.

        Returns:
            Episode termination message.
        """
        obs_text = self._wrapper.reject(reason=reason or None)
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def use_tool(self, tool_name: str, tool_args: Optional[dict] = None) -> str:
        """Call a deterministic analysis tool before proposing terms.

        Call QUERY_TREDS before RUN_CASHFLOW_SIM for optimal tool sequence reward.

        Args:
            tool_name: One of QUERY_TREDS, CHECK_COMPLIANCE, RUN_CASHFLOW_SIM.
            tool_args: JSON-serializable arguments for the tool.

        Returns:
            Tool result as a formatted observation string.
        """
        tool_args = tool_args or {}
        if tool_name == "QUERY_TREDS":
            invoice_id = str(tool_args.get("invoice_id", "default"))
            obs_text = self.query_treds(invoice_id=invoice_id)
        elif tool_name == "CHECK_COMPLIANCE":
            contract_id = str(tool_args.get("contract_id", "default"))
            obs_text = self.check_compliance(contract_id=contract_id)
        elif tool_name == "RUN_CASHFLOW_SIM":
            plan = tool_args.get("plan", {})
            horizon = tool_args.get("horizon")
            obs_text = self.run_cashflow_sim(plan=plan, horizon=horizon)
        else:
            raise ValueError(
                f"Unknown tool: {tool_name}. "
                "Valid tools: QUERY_TREDS, CHECK_COMPLIANCE, RUN_CASHFLOW_SIM."
            )
        return obs_text

    def query_treds(self, invoice_id: str, deal_id: Optional[str] = None) -> str:
        """Query deterministic TReDS quotes for the active or specified deal."""
        obs_text = self._wrapper.query_treds(invoice_id=invoice_id, deal_id=deal_id)
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def check_compliance(self, contract_id: str, deal_id: Optional[str] = None) -> str:
        """Check policy compliance for the active or specified deal."""
        obs_text = self._wrapper.check_compliance(contract_id=contract_id, deal_id=deal_id)
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def run_cashflow_sim(
        self,
        plan: dict[str, Any],
        horizon: Optional[int] = None,
        deal_id: Optional[str] = None,
    ) -> str:
        """Run the deterministic cashflow simulation tool."""
        obs_text = self._wrapper.run_cashflow_sim(plan=plan, horizon=horizon, deal_id=deal_id)
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def simulate_plan(
        self,
        plan: dict[str, Any],
        horizon: Optional[int] = None,
        deal_id: Optional[str] = None,
    ) -> str:
        """Run a read-only macro plan simulation."""
        obs_text = self._wrapper.simulate_plan(plan=plan, horizon=horizon, deal_id=deal_id)
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def advance_period(self) -> str:
        """Advance to the next macro period after resolving all open deals.

        Args: (none)

        Returns:
            New period observation or episode completion message.
        """
        obs_text = self._wrapper.advance_period()
        self._extract_and_cache_breakdown(obs_text)
        return obs_text

    def summarize_episode(self) -> Any:
        """Return a structured episode summary for logging and monitoring."""
        return self._wrapper.summarize_episode()

    def build_episode_log(self) -> str:
        """Return the deterministic episode text log."""
        return self._wrapper.build_episode_log()

    @property
    def current_persona(self) -> Any:
        return self._wrapper.current_persona

    @property
    def done(self) -> bool:
        return self._wrapper.done


try:
    from sme_negotiator_env.reward_breakdown import RewardBreakdown  # noqa: F401
except ImportError:
    pass
