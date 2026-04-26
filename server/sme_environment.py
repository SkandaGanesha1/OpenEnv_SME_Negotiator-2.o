"""Backward-compatible import path for the SME environment.

The canonical implementations live in ``server.environment``. This module
exists solely to preserve older imports and references that expect
``server.sme_environment.SMENegotiatorEnvironment`` while also surfacing the
new Stage 1 ``SMELiquidityEnvironment`` without changing the live server wiring.
"""

from .environment import SMELiquidityEnvironment, SMENegotiatorEnvironment
