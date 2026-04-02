"""Baseline agent for SME Negotiation Environment - LLM-based negotiator."""
import json
import os
from typing import Optional

from openai import OpenAI
from src.utils.models import NegotiationState, NegotiationAction


class LLMNegotiationAgent:
    """
    Baseline LLM-based negotiation agent.
    
    Uses an LLM (e.g., Nemotron 3 Super) to:
    1. Reason about financial constraints and regulatory boundaries
    2. Generate persuasive natural language justifications
    3. Produce deterministic JSON actions
    """
    
    def __init__(self, model_name: str = "nemotron-3-super"):
        """
        Initialize agent with LLM backbone.
        
        Args:
            model_name: HF model identifier
        """
        self.model_name = model_name
        self.llm = None  # Will be loaded on-demand
        self._client = None
    
    def generate_system_prompt(self, state: NegotiationState) -> str:
        """
        Generate system prompt with SME context and regulatory constraints.
        
        This prompt establishes the agent's identity and decision-making framework.
        """
        
        return f"""You are an AI agent representing a Small and Medium Enterprise (SME) negotiating a B2B contract with a large corporate buyer.

## Your SME's Financial Constraints:
- Production cost per unit: ₹{state.c_sme}
- Critical liquidity threshold: {state.l_sme} days (you cannot survive without cash beyond this)
- Current negotiation round: {state.t_elapsed + 1} of {state.t_max}

## Current Buyer's Offer:
- Unit Price: ₹{state.p_opp}
- Payment Terms: {state.d_opp} days
- Volume: {state.v_opp} units
- TReDS Eligible: {state.treds_opp}

## Critical Regulatory Requirements:
- MSMED Act mandates payment within 45 days for registered SMEs
- Beyond 45 days WITHOUT TReDS: your company faces severe liquidity risk
- TReDS Platform: allows you to auction trade receivables for immediate cash. Corporate buyer pays bank in {state.d_opp} days, you get cash on Day 1 (at a discounted rate of {state.r_discount * 100}%)

## Your Decision Framework:
You must balance three competing objectives:
1. SURVIVAL: Ensure payment arrives before Day {state.l_sme} (or use TReDS as workaround)
2. PROFITABILITY: Maximize unit price above your cost of ₹{state.c_sme}
3. FEASIBILITY: Recognize you cannot negotiably reduce payment days below the buyer's treasury constraints

## Action Options:
1. PROPOSE: Counter-offer with new price and/or payment terms, plus natural language justification
2. ACCEPT: Agree to opponent's current offer (only if it meets your constraints)
3. REJECT: Walk away from the deal

## Your Response Format:
Think through the financial implications step-by-step (especially NPV calculations and TReDS mechanics), then respond with ONLY a valid JSON action.

For PROPOSE actions, ensure your justification:
- Explains the financial logic behind your counter-offer
- References regulatory constraints (MSMED, TReDS) if relevant
- Demonstrates understanding of working capital and cost of capital implications
- Is persuasive but factually grounded

Remember: If you accept terms that would bankrupt your company (payment delay > {state.l_sme} days without TReDS), you get a score of 0.0.
"""
    
    def generate_observation_prompt(self, state: NegotiationState) -> str:
        """
        Generate user message with current negotiation state.
        """
        
        history_text = ""
        if state.history and len(state.history) > 1:
            history_text = "\n## Negotiation History:\n"
            for record in state.history[1:]:  # Skip initial offer
                history_text += f"Round {record.round} ({record.party.upper()}): "
                history_text += f"₹{record.proposed_price}/unit, {record.proposed_days} days"
                if record.request_treds:
                    history_text += " [TReDS requested]"
                history_text += f"\n  Justification: {record.justification}\n"
        
        return f"""Current negotiation state:

Buyer's Latest Offer:
- Price: ₹{state.p_opp} per unit
- Payment: {state.d_opp} days
- Volume: {state.v_opp} units
- TReDS willing: {state.treds_opp}

{history_text}

What is your next move? Respond with a JSON action object containing:
- action_type: "PROPOSE", "ACCEPT", or "REJECT"
- proposed_price: (required for PROPOSE)
- proposed_days: (required for PROPOSE)  
- request_treds: boolean
- justification: (required for PROPOSE, your negotiation rationale)

Example valid JSON:
{{"action_type": "PROPOSE", "proposed_price": 95.5, "proposed_days": 50, "request_treds": false, "justification": "..."}}
"""
    
    def act(self, state: NegotiationState) -> NegotiationAction:
        """
        Generate action based on current state.
        
        This method demonstrates the action generation pipeline.
        In production, this would call the actual LLM backend.
        """
        
        system_prompt = self.generate_system_prompt(state)
        observation_prompt = self.generate_observation_prompt(state)

        llm_action = self._call_llm(system_prompt, observation_prompt)
        if llm_action is not None:
            return llm_action

        return self._generate_fallback_action(state)

    def _get_client(self) -> Optional[OpenAI]:
        """Create an OpenAI-compatible client if credentials are available."""

        if self._client is not None:
            return self._client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or None
        try:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return None

        return self._client

    def _call_llm(self, system_prompt: str, observation_prompt: str) -> Optional[NegotiationAction]:
        """Call an OpenAI-compatible model and parse a JSON action response."""

        client = self._get_client()
        if client is None:
            return None

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            action = self._extract_json_action(content)
            if action is not None:
                return action
        except Exception:
            return None

        return None
    
    def _generate_fallback_action(self, state: NegotiationState) -> NegotiationAction:
        """
        Generate a reasonable fallback action (used for demo/offline testing).
        
        Implements basic logic: concede on price to reduce payment days.
        """
        
        # Calculate break-even margin
        margin = state.p_opp - state.c_sme
        
        # Check liquidity constraint
        if state.d_opp > state.l_sme and not state.treds_opp:
            # On hard task: try requesting TReDS and offering price reduction
            if state.t_elapsed >= state.t_max - 2:
                # Late game: evaluate accepting with TReDS
                return NegotiationAction(
                    action_type="ACCEPT",
                    proposed_price=state.p_opp,
                    proposed_days=state.d_opp,
                    request_treds=True,
                    justification="Accepting with TReDS for immediate liquidity access"
                )
            else:
                # Mid-game: propose TReDS-enabled deal at slight discount
                new_price = state.p_opp * 0.95  # 5% discount to encourage TReDS adoption
                return NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=new_price,
                    proposed_days=state.d_opp,
                    request_treds=True,
                    justification=(
                        "I can accept your {d_opp}-day terms if you process via TReDS. "
                        "This gives you the same cash flow and me immediate liquidity without "
                        "violating MSMED regulatory constraints. The {discount}% price reduction "
                        "offsets TReDS administrative friction. This is a win-win restructuring."
                    ).format(
                        d_opp=state.d_opp,
                        discount=int(5)
                    )
                )
        
        # Within liquidity threshold: focus on price optimization
        elif state.d_opp <= state.l_sme or (state.treds_opp and state.d_opp <= 90):
            # Concede slightly on price
            new_price = state.p_opp * 0.98
            if new_price > state.c_sme:
                return NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=new_price,
                    proposed_days=state.d_opp,
                    request_treds=state.treds_opp,
                    justification=(
                        f"I can work with {state.d_opp}-day terms at ₹{new_price:.2f}/unit. "
                        f"This covers my production costs (₹{state.c_sme}/unit) with reasonable margin "
                        f"while respecting your working capital requirements."
                    )
                )
        
        # Default: accept current offer if it's viable
        if state.d_opp <= state.l_sme or state.treds_opp:
            return NegotiationAction(
                action_type="ACCEPT",
                proposed_price=state.p_opp,
                proposed_days=state.d_opp,
                request_treds=state.treds_opp,
                justification="Accepting current terms"
            )
        
        # Last resort: reject if no viable path
        return NegotiationAction(
            action_type="REJECT",
            justification="Cannot accept terms that violate liquidity constraints"
        )
    
    def _extract_json_action(self, response_text: str) -> Optional[NegotiationAction]:
        """
        Extract JSON action from LLM response.
        
        Searches for valid JSON in the response text.
        """
        
        try:
            # Try to find JSON substring
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response_text[start:end]
                action_dict = json.loads(json_str)
                return NegotiationAction(**action_dict)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
