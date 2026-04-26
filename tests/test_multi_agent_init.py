"""Stage 7 initialization coverage for the liquidity world."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.environment import SMELiquidityEnvironment
from sme_negotiator_env.models import WorldState


def test_liquidity_world_state_initialization() -> None:
    env = SMELiquidityEnvironment(total_periods=3)
    observation = env.reset(seed=123, difficulty="hard", task_name="liquidity-correlation-hard")

    assert env.state is not None
    assert isinstance(env.state.world_state, WorldState)

    world_state = env.state.world_state
    assert len(world_state.smes) >= 1
    assert len(world_state.buyers) >= 1
    assert world_state.financier is not None
    assert world_state.current_period == 0
    assert world_state.total_periods == 3
    assert world_state.episode_step == 0

    assert observation.agent_type == "SME"
    assert observation.agent_id == "sme_0"
    assert observation.active_deal_id is not None
    assert len(observation.open_deal_ids) >= 1
    assert observation.resolved_deal_ids == []
    assert observation.history == []
    assert observation.last_tool_name is None
    assert observation.last_tool_args is None
    assert observation.last_tool_result is None
    assert observation.simulation_projection is None
    assert observation.projected_balances is None
    assert observation.projected_defaults is None
    assert observation.projected_penalties is None
