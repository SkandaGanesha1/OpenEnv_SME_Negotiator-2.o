"""OpenEnv SDK-backed app entrypoint for SME Negotiation."""

from openenv.core.env_server.http_server import create_app as openenv_create_app

from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction, NegotiationState


def create_app():
    """Create FastAPI app via OpenEnv SDK factory."""

    return openenv_create_app(
        SMENegotiationEnv,
        NegotiationAction,
        NegotiationState,
        env_name="sme-negotiator",
    )


app = create_app()
