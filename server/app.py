"""Manual OpenEnv app entrypoint for the SME negotiation environment."""

from __future__ import annotations

import os

from openenv.core import create_app
from sme_negotiator_env.models import NegotiationAction, NegotiationObservation

from .sme_environment import SMENegotiatorEnvironment


app = create_app(
    SMENegotiatorEnvironment,
    NegotiationAction,
    NegotiationObservation,
    env_name="sme-negotiator",
)


def main() -> None:
    """Run the server with uvicorn."""

    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
