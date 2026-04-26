#!/usr/bin/env python3
"""Compatibility wrapper for the notebook-facing liquidity GRPO module."""

from __future__ import annotations

from rl.train_grpo_liquidity import (
    build_arg_parser,
    build_canonical_training_args,
    build_run_plan,
    build_trainer,
    build_training_session,
    main,
    make_training_args,
    plot_rewards,
    resolve_profile_config,
    run_simple_training,
    run_training,
    smoke_test_environment,
)


if __name__ == "__main__":
    raise SystemExit(main())
