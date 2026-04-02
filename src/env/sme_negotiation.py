"""OpenEnv SME Negotiation environment - Core MDP implementation."""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from openenv.core.env_server.interfaces import Environment

from src.utils.models import (
	NegotiationState, NegotiationAction, NegotiationTerms, EpisodeResult, OfferRecord
)
from src.utils.grader import (
	DeterministicGrader, GraderConfig, BuyerProfile,
	calculate_buyer_counter_offer, evaluate_offer_acceptance
)


@dataclass
class TaskConfig:
	"""Configuration for a specific task difficulty level."""
	task_id: str
	name: str
	description: str
	initial_price_opp: float
	initial_days_opp: int
	volume: int
	max_rounds: int
	buyer_cooperation: str  # "cooperative", "strict", "aggressive"
    
	# SME parameters
	sme_cost: float
	sme_liquidity_threshold: int
    
	# Buyer hidden constraints
	buyer_p_max_multiplier: float      # P_max relative to SME cost
	buyer_d_min: int
	buyer_impatience_factor: float
	buyer_semantic_susceptibility: float


class SMENegotiationEnv(Environment):
	"""
	OpenEnv-compliant SME B2B Contract Negotiation environment.
    
	Implements episodic MDP with:
	- Deterministic state transitions
	- Multi-dimensional continuous action space
	- Deterministic grader (0.0-1.0 range)
	- Task stratification (Easy, Medium, Hard)
	"""
    
	# Predefined task configurations
	TASKS = {
		"easy": TaskConfig(
			task_id="easy",
			name="Single-Issue Price Optimization",
			description="Negotiate unit price with fixed payment terms (30 days)",
			initial_price_opp=100.0,
			initial_days_opp=30,
			volume=1000,
			max_rounds=5,
			buyer_cooperation="cooperative",
			sme_cost=80.0,
			sme_liquidity_threshold=45,
			buyer_p_max_multiplier=1.3,
			buyer_d_min=30,
			buyer_impatience_factor=0.3,
			buyer_semantic_susceptibility=0.2,
		),
		"medium": TaskConfig(
			task_id="medium",
			name="Bi-Dimensional Trade-off with Regulatory Boundaries",
			description="Negotiate price vs payment days; must respect 45-day regulatory limit",
			initial_price_opp=100.0,
			initial_days_opp=90,
			volume=1000,
			max_rounds=8,
			buyer_cooperation="strict",
			sme_cost=80.0,
			sme_liquidity_threshold=60,
			buyer_p_max_multiplier=1.2,
			buyer_d_min=60,
			buyer_impatience_factor=0.2,
			buyer_semantic_susceptibility=0.3,
		),
		"hard": TaskConfig(
			task_id="hard",
			name="Multi-Issue Non-Linear Optimization with TReDS",
			description="Restructure deal via TReDS; overcome buyer's rigid day constraints",
			initial_price_opp=95.0,
			initial_days_opp=120,
			volume=5000,
			max_rounds=12,
			buyer_cooperation="aggressive",
			sme_cost=70.0,
			sme_liquidity_threshold=30,
			buyer_p_max_multiplier=1.15,
			buyer_d_min=90,
			buyer_impatience_factor=0.1,
			buyer_semantic_susceptibility=0.4,
		),
	}
    
	def __init__(self):
		self.current_state: Optional[NegotiationState] = None
		self.task_config: Optional[TaskConfig] = None
		self.buyer_profile: Optional[BuyerProfile] = None
		self.grader: Optional[DeterministicGrader] = None
		self.episode_seed: int = 0
		self.grader_u_max: float = 1.0
		self.grader_u_min: float = 0.0
		self.rng: Optional[np.random.Generator] = None
    
	def reset(
		self,
		seed: Optional[int] = None,
		episode_id: Optional[str] = None,
		**kwargs,
	) -> NegotiationState:
		"""
		Reset environment for new episode (Gymnasium-style).
        
		Args:
			task_id: One of "easy", "medium", "hard"
			seed: Deterministic seed for reproducibility
        
		Returns:
			Initial observation (NegotiationState)
		"""
        
		task_id = kwargs.get("task_id") or episode_id or "easy"

		if task_id not in self.TASKS:
			raise ValueError(f"Unknown task_id: {task_id}")
        
		self.task_config = self.TASKS[task_id]

		if seed is None and "seed" in kwargs and kwargs["seed"] is not None:
			seed = int(kwargs["seed"])
        
		# Initialize using numpy RNG (CRITICAL for reproducibility)
		if seed is not None:
			self.episode_seed = seed
		else:
			self.episode_seed = int(np.random.default_rng().integers(0, 2**31))
        
		self.rng = np.random.default_rng(self.episode_seed)
        
		# Initialize grader
		grader_config = GraderConfig(
			sme_cost=self.task_config.sme_cost,
			sme_liquidity_threshold=self.task_config.sme_liquidity_threshold,
			market_discount_rate=0.08,
			internal_cost_of_capital=0.12,
		)
		self.grader = DeterministicGrader(grader_config)
        
		# Create buyer profile with hidden constraints
		self.buyer_profile = BuyerProfile(
			p_max=self.task_config.sme_cost * self.task_config.buyer_p_max_multiplier,
			d_min=self.task_config.buyer_d_min,
			impatience_factor=self.task_config.buyer_impatience_factor,
			semantic_susceptibility=self.task_config.buyer_semantic_susceptibility,
		)
        
		# Calculate grader normalization bounds (known at episode start for deterministic seeding)
		self.grader_u_max = self.grader.calculate_theoretical_max_utility(
			volume=self.task_config.volume,
			buyer_max_price=self.buyer_profile.p_max,
			buyer_min_days=self.buyer_profile.d_min,
		)
		self.grader_u_min = self.grader.calculate_break_even_utility(self.task_config.volume)
        
		# Initialize state
		self.current_state = NegotiationState(
			p_opp=self.task_config.initial_price_opp,
			d_opp=self.task_config.initial_days_opp,
			v_opp=self.task_config.volume,
			treds_opp=False,
			history=[
				OfferRecord(
					round=0,
					proposed_price=self.task_config.initial_price_opp,
					proposed_days=self.task_config.initial_days_opp,
					request_treds=False,
					justification="Initial buyer offer",
					party="buyer"
				)
			],
			c_sme=self.task_config.sme_cost,
			l_sme=self.task_config.sme_liquidity_threshold,
			r_discount=0.08,
			t_elapsed=0,
			t_max=self.task_config.max_rounds,
			task_id=task_id,
			episode_seed=self.episode_seed,
		)
        
		return self.current_state
    
	def step(
		self,
		action: NegotiationAction,
		timeout_s: Optional[float] = None,
		**kwargs,
	) -> NegotiationState:
		"""
		Execute one step in the negotiation (Gymnasium-style).
        
		Args:
			action: Agent's action (NegotiationAction)
        
		Returns:
			(observation, reward, terminated, info)
		"""

		# Auto-initialize if reset was never called
		if self.current_state is None:
			self.reset(seed=42)
        
		if self.current_state is None or self.task_config is None:
			raise RuntimeError("Environment not initialized. Call reset() first.")
        
		# Validate action
		is_valid, error_msg = action.validate_action()
		if not is_valid:
			return self._to_observation(0.0, True, {"error": error_msg, "success": False})
        
		info = {"action_type": action.action_type}
        
		# Handle REJECT: immediate episode termination with zero score
		if action.action_type == "REJECT":
			episode_result = EpisodeResult(
				success=False,
				score=0.0,
				failure_reason="Agent rejected negotiation",
				round_completed=self.current_state.t_elapsed,
			)
			return self._to_observation(0.0, True, episode_result.model_dump())
        
		# Handle ACCEPT: check if it aligns with opponent's last offer
		if action.action_type == "ACCEPT":
			# Agent must accept the exact opponent's last offer
			if (action.proposed_price is None or
				action.proposed_days is None):
				return self._to_observation(0.0, True, {"error": "ACCEPT requires parameters", "success": False})
            
			# Check if acceptance matches opponent's offer
			if (abs(action.proposed_price - self.current_state.p_opp) < 0.01 and
				action.proposed_days == self.current_state.d_opp):
                
				# Deal accepted - calculate final score
				episode_result = self._finalize_deal(
					final_price=action.proposed_price,
					final_days=action.proposed_days,
					final_volume=self.current_state.v_opp,
					treds_utilized=action.request_treds or self.current_state.treds_opp,
				)
				return self._to_observation(episode_result.score, True, episode_result.model_dump())
			else:
				info["error"] = "ACCEPT parameters don't match opponent's offer"
				info["success"] = False
				return self._to_observation(0.0, True, info)
        
		# Handle PROPOSE: generate counter-offer
		if action.action_type == "PROPOSE":
			# Check round limit
			if self.current_state.t_elapsed >= self.current_state.t_max:
				episode_result = EpisodeResult(
					success=False,
					score=0.0,
					failure_reason="Maximum rounds exceeded",
					round_completed=self.current_state.t_elapsed,
				)
				return self._to_observation(0.0, True, episode_result.model_dump())
            
			# Check if agent's proposal is acceptable to buyer
			# For easy task, require at least 2 rounds before accepting to ensure negotiation occurs
			min_rounds_for_acceptance = 2 if self.task_config.task_id == "easy" else 1
            
			if (self.current_state.t_elapsed >= min_rounds_for_acceptance and 
				evaluate_offer_acceptance(
					action.proposed_price,
					action.proposed_days,
					self.buyer_profile
				)):
				# Deal accepted!
				episode_result = self._finalize_deal(
					final_price=action.proposed_price,
					final_days=action.proposed_days,
					final_volume=self.current_state.v_opp,
					treds_utilized=action.request_treds,
				)
				return self._to_observation(episode_result.score, True, episode_result.model_dump())
            
			# Generate buyer counter-offer
			justification_quality = self._evaluate_justification_quality(
				action.justification
			)
            
			counter_price, counter_days = calculate_buyer_counter_offer(
				previous_price=action.proposed_price,
				previous_days=action.proposed_days,
				buyer_profile=self.buyer_profile,
				round_number=self.current_state.t_elapsed + 1,
				max_rounds=self.current_state.t_max,
				justification_quality=justification_quality,
			)
            
			# Add agent's action to history
			self.current_state.history.append(OfferRecord(
				round=self.current_state.t_elapsed + 1,
				proposed_price=action.proposed_price,
				proposed_days=action.proposed_days,
				request_treds=action.request_treds,
				justification=action.justification,
				party="agent"
			))
            
			# Update state with buyer's counter-offer
			self.current_state.p_opp = counter_price
			self.current_state.d_opp = counter_days
			self.current_state.treds_opp = action.request_treds  # Buyer acknowledges TReDS request
			self.current_state.t_elapsed += 1
            
			# Add counter-offer to history
			self.current_state.history.append(OfferRecord(
				round=self.current_state.t_elapsed,
				proposed_price=counter_price,
				proposed_days=counter_days,
				request_treds=action.request_treds,
				justification="Buyer counter-offer",
				party="buyer"
			))
            
			info["counter_price"] = counter_price
			info["counter_days"] = counter_days
            
			# Intermediate reward is always 0.0 (long-term credit assignment)
			return self._to_observation(0.0, False, info)
        
		return self._to_observation(0.0, False, info)

	def _to_observation(self, reward: float, terminated: bool, info: Dict) -> NegotiationState:
		"""Populate OpenEnv observation fields and return current state."""

		self.current_state.reward = float(reward)
		self.current_state.done = bool(terminated)
		self.current_state.metadata = dict(info)
		return self.current_state

	@property
	def state(self) -> Optional[NegotiationState]:
		"""Return the current observation state for the active episode."""

		return self.current_state
    
	def _finalize_deal(
		self,
		final_price: float,
		final_days: int,
		final_volume: int,
		treds_utilized: bool,
	) -> EpisodeResult:
		"""Calculate final score and create episode result."""
        
		score, grader_details = self.grader.calculate_grader_score(
			final_price=final_price,
			final_days=final_days,
			final_volume=final_volume,
			treds_utilized=treds_utilized,
			success=True,
			u_max=self.grader_u_max,
			u_min=self.grader_u_min,
		)
        
		return EpisodeResult(
			success=True,
			terms=NegotiationTerms(
				final_price=final_price,
				final_days=final_days,
				final_volume=final_volume,
				treds_utilized=treds_utilized,
			),
			score=score,
			npv_base=grader_details.get("npv_base"),
			final_utility=grader_details.get("utility"),
			round_completed=self.current_state.t_elapsed,
		)
    
	def _evaluate_justification_quality(self, justification: str) -> float:
		"""
		Evaluate semantic quality of agent's justification (0.0-1.0).
        
		This is a lightweight, deterministic heuristic on the server side.
		Does NOT use an LLM for evaluation (prevents exploit).
		"""
        
		if not justification or len(justification.strip()) == 0:
			return 0.0
        
		# Keywords that indicate financial reasoning
		financial_keywords = {
			"npv": 2.0, "cash flow": 2.0, "liquidity": 2.0,
			"cost of capital": 2.0, "discount": 1.5, "msmed": 2.0,
			"treds": 2.5, "payment terms": 1.5, "interest": 1.5,
			"working capital": 2.0, "regulatory": 1.5,
		}
        
		text_lower = justification.lower()
		score = 0.0
        
		for keyword, weight in financial_keywords.items():
			if keyword in text_lower:
				score += weight
        
		# Normalize: max ~20 points, cap at 1.0
		quality = min(1.0, score / 20.0)
        
		return quality
