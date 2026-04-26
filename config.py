"""Central configuration — task registry, UI constants, and default values."""
from __future__ import annotations

# ── Task registry ────────────────────────────────────────────────────────────
TASKS: dict[str, str] = {
    "🟢 Easy  — compress days ≤ 60": "payment-terms-easy",
    "🟡 Medium — days ≤ 45 + clause": "payment-terms-medium",
    "🔴 Hard  — dynamic discounting": "payment-terms-hard",
}

# UI label → wired NegotiationAction.action_type
UI_ACTION_CHOICES: list[str] = ["propose", "counter_offer", "accept", "reject"]

# ── Episode defaults ─────────────────────────────────────────────────────────
DEFAULT_SEED: int = 42
MAX_ROUNDS: int = 10

# ── Grader weights (must sum to 1.0) ─────────────────────────────────────────
GRADER_WEIGHTS: dict[str, float] = {
    "solvency":   0.35,
    "liquidity":  0.20,
    "npv":        0.35,
    "compliance": 0.10,
}

# ── Quick-connect snippet shown in the Reference tab ─────────────────────────
QUICK_CONNECT_SNIPPET: str = """\
from server.environment import SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction

env = SMENegotiatorEnvironment()
obs = env.reset(seed=42, task_name="payment-terms-easy")
# obs is a NegotiationObservation (Pydantic); use obs.model_dump() for a dict.

action = NegotiationAction(
    action_type="propose",
    price=float(obs.buyer_price),
    payment_days=int(obs.buyer_days),
    use_treds=False,
    reason="Sync API example",
)
obs2 = env.step(action)
print(obs2.model_dump())
"""
