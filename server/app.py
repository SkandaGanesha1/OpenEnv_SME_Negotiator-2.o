"""Manual OpenEnv app entrypoint for the SME negotiation environment."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from openenv.core import create_app
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected

from sme_negotiator_env.models import NegotiationAction, NegotiationObservation

from .concurrency import OpenEnvConcurrencyLimiter, max_concurrent_envs_from_env
from .sme_environment import SMENegotiatorEnvironment


def _benign_websocket_teardown(exc: BaseException) -> bool:
    """True when the client closed the socket and follow-up sends fail (openenv-core quirk)."""
    if isinstance(exc, (WebSocketDisconnect, ClientDisconnected)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "close message" in msg and "send" in msg:
            return True
    mod = getattr(type(exc), "__module__", "") or ""
    if mod.startswith("websockets.") and "ConnectionClosed" in type(exc).__name__:
        return True
    return False


def _wrap_ws_for_graceful_client_close(app: FastAPI) -> None:
    """Patch /ws so disconnects do not surface as unhandled RuntimeError (upstream openenv)."""
    for route in app.router.routes:
        if isinstance(route, WebSocketRoute) and route.path == "/ws":
            orig: Any = route.endpoint

            async def _safe_ws(websocket: WebSocket, *, _orig: Any = orig) -> None:
                try:
                    await _orig(websocket)
                except Exception as e:
                    if _benign_websocket_teardown(e):
                        return
                    raise

            route.endpoint = _safe_ws
            return


_max = max_concurrent_envs_from_env()

app = create_app(
    SMENegotiatorEnvironment,
    NegotiationAction,
    NegotiationObservation,
    env_name="sme-negotiator",
    max_concurrent_envs=_max,
)

_wrap_ws_for_graceful_client_close(app)

app.add_middleware(OpenEnvConcurrencyLimiter, max_concurrent=_max)


def main() -> None:
    """Run the server with uvicorn."""

    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
