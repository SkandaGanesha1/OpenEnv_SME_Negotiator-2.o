"""Manual FastAPI app entrypoint for SME Negotiation."""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction


class ResetRequest(BaseModel):
    """Request model for resetting an episode."""

    task_id: Optional[str] = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    """Request model for stepping an episode."""

    episode_id: Optional[str] = None
    action: Dict[str, Any] = Field(default_factory=dict)


def _dump_state(state: Any) -> Dict[str, Any]:
    """Normalize state serialization for responses."""

    if hasattr(state, "model_dump"):
        return state.model_dump()
    if hasattr(state, "to_dict"):
        return state.to_dict()
    if isinstance(state, dict):
        return state
    return dict(state)


def create_app() -> FastAPI:
    """Create FastAPI app with manual stateful environment routes."""

    app = FastAPI(title="OpenEnv SME Negotiator", version="0.1.0")
    app.state.environment = SMENegotiationEnv()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "healthy"}

    @app.post("/reset")
    def reset(payload: Optional[ResetRequest] = None) -> Dict[str, Any]:
        body = payload or ResetRequest()
        env = app.state.environment

        try:
            state = env.reset(
                task_id=body.task_id,
                seed=body.seed,
                episode_id=body.episode_id,
            )
            observation = _dump_state(state)
            return {
                "observation": observation,
                "reward": float(getattr(state, "reward", 0.0)),
                "done": bool(getattr(state, "done", False)),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/step")
    def step(payload: StepRequest) -> Dict[str, Any]:
        env = app.state.environment

        try:
            action = NegotiationAction(**payload.action)
            state = env.step(action, episode_id=payload.episode_id)
            observation = _dump_state(state)
            return {
                "observation": observation,
                "reward": float(getattr(state, "reward", 0.0)),
                "done": bool(getattr(state, "done", False)),
                "info": dict(getattr(state, "metadata", {})),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/state")
    def state() -> Dict[str, Any]:
        env = app.state.environment
        current = env.state
        if current is None:
            return {"state": None}
        return {"state": _dump_state(current)}

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        return {
            "name": "sme-negotiator",
            "endpoints": {
                "health": "/health",
                "reset": "/reset",
                "step": "/step",
                "state": "/state",
                "schema": "/schema",
            },
        }

    return app


app = create_app()
