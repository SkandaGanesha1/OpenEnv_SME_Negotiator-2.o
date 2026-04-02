"""Pydantic schemas and OpenEnv metadata helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class NegotiationTermsSchema(BaseModel):
    """Serialized final negotiation terms."""

    model_config = ConfigDict(extra="ignore")

    final_price: float
    final_days: int
    final_volume: int
    treds_utilized: bool = False
    price: Optional[float] = None
    days: Optional[int] = None
    volume: Optional[int] = None


class OfferRecordSchema(BaseModel):
    """Serialized negotiation history entry."""

    model_config = ConfigDict(extra="ignore")

    round: int
    proposed_price: float
    proposed_days: int
    request_treds: bool
    justification: str
    party: str


class NegotiationStateSchema(BaseModel):
    """Serialized environment observation."""

    model_config = ConfigDict(extra="ignore")

    task_id: str = ""
    t_elapsed: int = 0
    t_max: int = 10
    episode_seed: int = 0
    p_opp: float = 100.0
    d_opp: int = 30
    v_opp: int = 100
    treds_opp: bool = False
    c_sme: float = 50.0
    l_sme: float = 100.0
    r_discount: float = 0.08
    history: List[OfferRecordSchema] = Field(default_factory=list)
    done: bool = False


class NegotiationActionSchema(BaseModel):
    """Serialized negotiation action accepted by the server."""

    model_config = ConfigDict(extra="ignore")

    action_type: str
    proposed_price: Optional[float] = None
    proposed_days: Optional[int] = None
    request_treds: bool = False
    justification: Optional[str] = None


class EpisodeResultSchema(BaseModel):
    """Serialized episode result returned by the grader."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    terms: Optional[NegotiationTermsSchema] = None
    score: float = 0.0
    npv_base: float = 0.0
    final_utility: float = 0.0
    round_completed: int = 0
    u_max: float = 0.0
    u_min: float = 0.0
    treds_utilized: bool = False
    deal_reached: bool = False
    final_price: Optional[float] = None
    final_days: Optional[int] = None
    total_rounds: int = 0
    total_reward: float = 0.0
    normalized_reward: float = 0.0
    reason: str = ""
    failure_reason: Optional[str] = None


class TaskSchema(BaseModel):
    """Task metadata for validator-facing schema responses."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    description: str
    difficulty: int
    max_rounds: int
    negotiation_variables: Dict[str, bool]


def serialize_state(state: Any) -> Optional[NegotiationStateSchema]:
    """Convert a domain state object into the Pydantic schema."""

    if state is None:
        return None
    payload = state.model_dump() if hasattr(state, "model_dump") else state
    return NegotiationStateSchema.model_validate(payload)


def serialize_action(action: Any) -> NegotiationActionSchema:
    """Convert a domain action object into the Pydantic schema."""

    payload = action.model_dump() if hasattr(action, "model_dump") else action
    return NegotiationActionSchema.model_validate(payload)


def serialize_episode_result(result: Any) -> EpisodeResultSchema:
    """Convert a domain episode result into the Pydantic schema."""

    payload = result.model_dump() if hasattr(result, "model_dump") else result
    return EpisodeResultSchema.model_validate(payload)


def build_environment_schema(env: Any) -> Dict[str, Any]:
    """Build a validator-friendly OpenEnv schema payload from the live environment."""

    tasks: List[Dict[str, Any]] = []
    for task_id, config in env.TASKS.items():
        tasks.append(
            TaskSchema(
                id=config.task_id,
                name=config.name,
                description=config.description,
                difficulty={"easy": 1, "medium": 2, "hard": 3}.get(task_id, 0),
                max_rounds=config.max_rounds,
                negotiation_variables={
                    "price": True,
                    "days": task_id in ["medium", "hard"],
                    "volume": task_id == "hard",
                    "treds": task_id == "hard",
                },
            ).model_dump()
        )

    return {
        "api_version": "1.0",
        "metadata": {
            "name": "SME B2B Contract Negotiation",
            "version": "0.1.0",
            "description": "Deterministic OpenEnv-style negotiation environment for SME contract pricing and payment terms.",
            "author": "Omkarchaithanya",
        },
        "endpoints": {
            "health": "/health",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "schema": "GET /schema",
            "tasks": "GET /tasks",
        },
        "tasks": tasks,
        "state_schema": NegotiationStateSchema.model_json_schema(),
        "action_schema": NegotiationActionSchema.model_json_schema(),
        "result_schema": EpisodeResultSchema.model_json_schema(),
    }