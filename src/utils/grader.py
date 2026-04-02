"""Deterministic grader for OpenEnv SME Negotiation environment."""
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GraderConfig:
    """Configuration for the deterministic grader."""
    sme_cost: float                    # C_SME: SME's production cost
    sme_liquidity_threshold: int       # L_SME: Days SME can survive
    market_discount_rate: float        # r_discount: TReDS discount rate
    internal_cost_of_capital: float    # r: SME's cost of capital
    regulatory_max_days: int = 45      # MSMED Act compliance limit


class DeterministicGrader:
    """
    Implements the closed-form, deterministic reward grader.
    
    Produces scores strictly between 0.0 and 1.0 based on:
    - Normalized Net Present Value (NPV)
    - Regulatory compliance penalties
    - Liquidity survival constraints
    """
    
    def __init__(self, config: GraderConfig):
        self.config = config
    
    def calculate_grader_score(
        self,
        final_price: float,
        final_days: int,
        final_volume: int,
        treds_utilized: bool,
        success: bool,
        u_max: float,
        u_min: float
    ) -> Tuple[float, dict]:
        """
        Calculate deterministic or grader score.
        
        Returns:
            score: Float between 0.0 and 1.0
            details: Dictionary with calculation intermediates
        """
        
        details = {
            "success": success,
            "final_price": final_price,
            "final_days": final_days,
            "final_volume": final_volume,
            "treds_utilized": treds_utilized,
        }
        
        # If negotiation failed, score is 0.0
        if not success:
            return 0.0, details
        
        # Calculate baseline profit
        profit = (final_price - self.config.sme_cost) * final_volume
        details["profit"] = profit
        
        # Apply time-value of money (convert days to years)
        days_in_years = final_days / 365.0
        discount_factor = 1.0 / ((1.0 + self.config.internal_cost_of_capital) ** days_in_years)
        npv_base = profit * discount_factor
        details["npv_base"] = npv_base
        
        # Check if SME survives liquidity threshold
        if final_days > self.config.sme_liquidity_threshold and not treds_utilized:
            # Immediate bankruptcy condition
            details["utility"] = 0.0
            details["reason"] = "Failed liquidity threshold without TReDS"
            return 0.0, details
        
        # Apply regulatory liquidity risk penalty
        liquidity_penalty = self._calculate_liquidity_penalty(final_days, treds_utilized, npv_base)
        details["liquidity_penalty"] = liquidity_penalty
        
        # Final utility
        utility = npv_base - liquidity_penalty
        utility = max(0.0, utility)  # Prevent negative utility
        details["utility"] = utility
        
        # Normalize to [0.0, 1.0] range
        if u_max <= u_min:
            # Degenerate case
            score = 0.0
        else:
            score = (utility - u_min) / (u_max - u_min)
        
        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        details["score"] = score
        
        return score, details
    
    def _calculate_liquidity_penalty(
        self,
        final_days: int,
        treds_utilized: bool,
        npv_base: float
    ) -> float:
        """
        Calculate regulatory liquidity risk penalty.
        
        Formula:
        Ω(D_final) = 0 if D_final <= 45 or TReDS_final = True
        Ω(D_final) = NPV_base × (Delay Factor) if D_final > 45 and TReDS_final = False
        """
        
        if final_days <= self.config.regulatory_max_days or treds_utilized:
            return 0.0
        
        # Calculate delay factor penalty (increases with days beyond 45)
        excess_days = final_days - self.config.regulatory_max_days
        delay_factor = math.exp(excess_days / 30.0) - 1.0  # Exponential penalty
        
        penalty = npv_base * delay_factor
        return penalty
    
    def calculate_theoretical_max_utility(
        self,
        volume: int,
        buyer_max_price: float,
        buyer_min_days: int,
    ) -> float:
        """
        Calculate U_max: maximum achievable utility for normalization.
        
        This occurs when agent extracts buyer's absolute maximum price
        and minimum payment days.
        """
        profit = (buyer_max_price - self.config.sme_cost) * volume
        days_in_years = buyer_min_days / 365.0
        discount_factor = 1.0 / ((1.0 + self.config.internal_cost_of_capital) ** days_in_years)
        u_max = profit * discount_factor
        
        # No penalty since buyer_min_days is typically within acceptable range
        return max(u_max, 1.0)  # Ensure U_max >= 1.0
    
    def calculate_break_even_utility(self, volume: int) -> float:
        """
        Calculate U_min: break-even utility (SME makes no loss, no gain).
        
        This is when final_price == C_SME (production cost exactly matched).
        """
        profit = 0.0  # Break-even
        u_min = 0.0
        return u_min


@dataclass
class BuyerProfile:
    """Hidden buyer constraint profile."""
    p_max: float                  # Absolute maximum price willing to pay
    d_min: int                    # Minimum required payment days
    impatience_factor: float      # α: increases concession rate over time
    semantic_susceptibility: float # β: vulnerability to LLM persuasion (0.0-1.0)


def calculate_buyer_counter_offer(
    previous_price: float,
    previous_days: int,
    buyer_profile: BuyerProfile,
    round_number: int,
    max_rounds: int,
    justification_quality: float = 0.5
) -> Tuple[float, int]:
    """
    Deterministic buyer counter-offer generation.
    
    Uses time-dependent concession strategy with semantic acceleration.
    
    Args:
        previous_price: Agent's previous proposal
        previous_days: Agent's previous days proposal
        buyer_profile: Hidden buyer constraints
        round_number: Current round (1-indexed)
        max_rounds: Total rounds available
        justification_quality: Semantic quality (0.0-1.0) of agent's justification
    
    Returns:
        counter_price, counter_days
    """
    
    # Time-dependent concession factor (linear)
    if max_rounds <= 1:
        time_factor = 1.0
    else:
        time_factor = (round_number - 1) / (max_rounds - 1)
    
    # Semantic acceleration of concession
    semantic_acceleration = buyer_profile.semantic_susceptibility * justification_quality
    
    # Total concession pressure
    concession_pressure = buyer_profile.impatience_factor * (time_factor + semantic_acceleration)
    
    # Calculate new counter-offer
    # Price: move towards agent's proposal, but not beyond P_max
    price_delta = buyer_profile.p_max - previous_price
    counter_price = previous_price + (price_delta * concession_pressure * 0.1)
    counter_price = min(counter_price, buyer_profile.p_max)
    counter_price = max(counter_price, previous_price)
    
    # Days: more rigid, follows minimum with slight flexibility
    days_delta = buyer_profile.d_min - previous_days
    days_concession = (1.0 - buyer_profile.semantic_susceptibility) * semantic_acceleration
    counter_days = previous_days + int(days_delta * days_concession * 0.05)
    counter_days = max(counter_days, buyer_profile.d_min)
    
    return counter_price, counter_days


def evaluate_offer_acceptance(
    proposed_price: float,
    proposed_days: int,
    buyer_profile: BuyerProfile,
) -> bool:
    """
    Deterministic check whether buyer accepts the offer.
    
    Returns True if agent meets buyer's strict constraints.
    """
    return proposed_price <= buyer_profile.p_max and proposed_days >= buyer_profile.d_min
