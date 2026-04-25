"""Multi-agent components for Theme #1 — Multi-Agent Interactions.

Exposes all strategic agent classes at the sme_negotiator_env.agents namespace.
RegulatoryWarning and FinancierBid are Pydantic models defined in models.py
to avoid circular imports; they are re-exported here for convenience.
"""

from sme_negotiator_env.models import FinancierBid, RegulatoryWarning
from sme_negotiator_env.agents.buyer_agent import BuyerLatentType, StrategicBuyerAgent
from sme_negotiator_env.agents.coalition import BuyerCoalition, CoalitionFormationPolicy
from sme_negotiator_env.agents.financier_agent import FinancierCompetitionArena
from sme_negotiator_env.agents.regulator_agent import RegulatorAgent

__all__ = [
    "BuyerLatentType",
    "StrategicBuyerAgent",
    "BuyerCoalition",
    "CoalitionFormationPolicy",
    "FinancierBid",
    "FinancierCompetitionArena",
    "RegulatorAgent",
    "RegulatoryWarning",
]
