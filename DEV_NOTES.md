# Stage 0 Developer Notes

This document freezes the current Round 1 baseline and maps the repository's
existing implementation without changing runtime behavior.

## Baseline Freeze

- Working branch: `round1_baseline`
- Immutable baseline tag: `round1-submission-baseline`
- Stage 0 goal: documentation and code comments only

The baseline environment behavior is intentionally preserved. Stage 0 does not
change task ids, manifest wiring, serialized model fields, grader formulas, or
the external OpenEnv API.

## OpenEnv Wiring

- `openenv.yaml` is the OpenEnv manifest.
- The manifest points to `server.app:app`.
- `server/app.py` builds the FastAPI app with:
  - `SMENegotiatorEnvironment`
  - `NegotiationAction`
  - `NegotiationObservation`
- `server/environment.py` contains the canonical simulator.
- `server/sme_environment.py` is only a backward-compatible re-export of the
  environment class.

## Repository Layout

- `server/environment.py`
  - Main OpenEnv environment implementation.
  - Owns `reset()`, `step()`, episode bookkeeping, internal state, shaping
    reward flow, and terminal reward dispatch.
- `server/app.py`
  - FastAPI/OpenEnv app entrypoint used by `openenv.yaml`.
- `sme_negotiator_env/models.py`
  - Canonical typed models for action, observation, and internal state.
- `sme_negotiator_env/graders.py`
  - Deterministic terminal graders and the hard-task NPV helper.
- `sme_negotiator_env/task_config.py`
  - Canonical task registry and task-id resolution.
- `sme_negotiator_env/client.py`
  - Typed client and heuristic policy example.
- `inference.py`
  - Baseline LLM runner and policy harness. It can drive the environment over
    HTTP/WebSocket using `SMENegotiatorEnv` or in process using
    `InProcessSMENegotiatorBridge`.
- `models.py`, `graders.py`, `client.py`
  - Root-level compatibility shims that re-export the canonical package modules
    for validator and legacy import compatibility.

## Model Roles

- `NegotiationAction`
  - Public agent input contract passed to `step()`.
  - Serialized across the client/server OpenEnv interface.
- `NegotiationObservation`
  - Public observation returned by `reset()` and `step()`.
  - Exposes buyer terms, task context, reward, done flag, and metadata.
- `NegotiationState`
  - Full internal episode state stored on the environment.
  - Used by the simulator and terminal graders.
  - Contains fields not all agents need to see directly, such as agreement flags
    and task-specific financial context.

Observation is derived from internal state plus current buyer-offer bookkeeping.
Action is the only agent-controlled input. State is the authoritative internal
record used for grading.

## Task and Grader Mapping

The canonical task ids are:

- `payment-terms-easy`
- `payment-terms-medium`
- `payment-terms-hard`

Those same ids line up across:

- `openenv.yaml`
- `sme_negotiator_env.task_config.TASK_REGISTRY`
- `sme_negotiator_env.graders.TASK_GRADERS`

Current grader mapping:

- `payment-terms-easy` -> `grade_task_payment_terms_easy`
- `payment-terms-medium` -> `grade_task_payment_terms_medium`
- `payment-terms-hard` -> `grade_task_dynamic_discounting_hard`

## Reward Structure

- Per-step shaping reward lives in
  `server.environment.SMENegotiatorEnvironment._compute_reward`.
- Terminal task scoring lives in `sme_negotiator_env/graders.py`.
- `_terminal_reward()` selects the active task grader using the current task's
  `grader_id`.
- On agreement, the environment writes the agreed terms into
  `NegotiationState`, then computes the terminal score.
- On rejection or invalid accept, the environment terminates immediately using
  the current strict-open-interval terminal fallback path.
- On max rounds without a deal, the terminal grader replaces the last shaping
  reward instead of being added on top of it.

Important Stage 0 note: the reset message currently includes a live UTC
timestamp from `_now_utc_iso()`. That is existing behavior and is documented
here rather than changed.

## Stage 2 Reward Path

Stage 2 adds a separate RL-oriented reward decomposition on the
`SMELiquidityEnvironment` path only. The live OpenEnv server wiring and the
Round 1 task ids continue to use `SMENegotiatorEnvironment`.

- Legacy Round 1 path
  - `SMENegotiatorEnvironment` remains numerically unchanged for the current
    `payment-terms-*` tasks.
  - `TASK_GRADERS` still map those task ids to the same terminal grader
    functions.
- Stage 2 liquidity path
  - `SMELiquidityEnvironment` maintains a trajectory buffer of
    `NegotiationState` snapshots.
  - `sme_negotiator_env.graders.compute_shaping_rewards(...)` computes dense,
    deterministic per-transition progress signals.
  - `sme_negotiator_env.graders.compute_verifiable_reward(...)` computes the
    deterministic terminal reward from `WorldState` plus the episode
    trajectory.
  - `sme_negotiator_env.graders.compute_total_sme_reward(...)` combines those
    pieces as `terminal + lambda * sum(shaping)`.
  - `LiquidityObservation.reward` and `step_reward` are overwritten with the
    Stage 2 RL reward, while the wrapped single-deal reward is preserved in
    observation metadata under `legacy_inner_reward`.
- External-only rubric hook
  - `sme_negotiator_env.graders.compute_rubric_score(...)` is a stub that
    raises `NotImplementedError`.
  - It is intentionally not called by `step()`, the deterministic graders, or
    the live OpenEnv server path.

Stage 2 also adds config-only task ids in `TASK_REGISTRY`:

- `liquidity-stress-medium`
- `liquidity-correlation-hard`

These are not exposed in `openenv.yaml` yet. They exist for in-process Stage 2
liquidity-environment experiments without changing live manifest wiring.

## Stage 4 Tool-Using Liquidity Path

Stage 4 extends only `SMELiquidityEnvironment` and keeps the live server path
unchanged.

- Action contract
  - `NegotiationAction` keeps the existing lowercase actions and adds
    `action_type="tool"`.
  - Tool invocations use `tool_name` and `tool_args`.
- Deterministic enterprise tools
  - `QUERY_TREDS` returns deterministic financing quotes from current world and
    deal state.
  - `CHECK_COMPLIANCE` returns deterministic pass/fail plus violated clauses.
  - `RUN_CASHFLOW_SIM` wraps the Stage 3 pure cashflow simulator.
- Observation/state additions
  - Tool outputs and recent history live only on `LiquidityObservation` and
    `LiquidityEnvironmentState`.
  - `NegotiationObservation` remains frozen for legacy compatibility.
- Reward behavior
  - Core Stage 2 terminal and shaping formulas stay unchanged.
  - Stage 4 adds only a tiny deterministic liquidity-only tool bonus/penalty
    layer on top of the existing liquidity reward routing.
- Compatibility
  - `SMENegotiatorEnvironment` still rejects non-legacy actions, now including
    `tool`, deterministically.
  - `server.app` and `openenv.yaml` remain unchanged.

## Stage 5 RL Training Integration

Stage 5 adds training integration only; it does not change environment
semantics.

- New package: `rl/`
  - `rl/bridge.py`
    - zero-arg `environment_factory` bridge for TRL/OpenEnv-style GRPO
    - direct in-process use of `SMELiquidityEnvironment`
    - typed public tool methods (`propose`, `accept`, `reject`,
      `query_treds`, `check_compliance`, `run_cashflow_sim`,
      `advance_period`)
  - `rl/episode_logging.py`
    - deterministic episode summaries
    - optional rubric overlay helper
  - `rl/train_grpo_trl.py`
    - canonical TRL GRPO entrypoint
  - `rl/train_grpo_unsloth.py`
    - optional Unsloth-accelerated GRPO entrypoint
- Shared prompt/action normalization
  - pure helpers now live in `sme_negotiator_env/prompting.py`
  - `inference.py` and the RL bridge share those helpers without importing each
    other's runtime side effects
- Metrics-only grader addition
  - `compute_npv_delta_vs_baseline(world_state, trajectory)` is additive and is
    used for RL-side logging only
- Packaging
  - optional dependency groups: `rl` and `rl-unsloth`

Important compatibility note: the live OpenEnv server still serves
`SMENegotiatorEnvironment`; the Stage 5 bridge runs only in process against
`SMELiquidityEnvironment`.

## Stage 6 Self-Improvement Layer

Stage 6 stays training-side and liquidity-only.

- `rl/opponents.py`
  - heuristic and snapshot-backed buyer / financier policies
  - rolling opponent-zoo management for self-play
- `rl/curriculum.py`
  - deterministic curriculum levels over `total_periods`,
    `buyer_variance`, and `financier_variance`
- `rl/rubrics.py`
  - persona sampling and persona-weighted rubric aggregation
- `rl/self_rewarding_dpo.py`
  - preference-dataset construction from episode logs

Compatibility guardrails remain the same:

- no Stage 6 behavior is wired into `server.app`
- graders remain deterministic and environment-side
- rubric judging stays external and optional
- true multi-SME competition is still deferred to a future stage

## OpenEnv API Invariants

- `reset(seed=..., difficulty=..., task_name=...)`
  - Initializes task config and RNG from the selected task and seed.
  - Returns a `NegotiationObservation`.
- `step(action)`
  - Accepts a `NegotiationAction`.
  - Returns a `NegotiationObservation` carrying `reward`, `done`, and
    `metadata`.
- `state`
  - Exposes the current internal `NegotiationState` for serialization and
    debugging.

Stage 0 preserves all method signatures and serialized field names exactly as
they exist in the baseline.

## Tests and Current Guardrails

Run the baseline test suite with:

```powershell
uv run --extra dev pytest tests/ -q
```

Current test coverage is organized as follows:

- `tests/test_environment.py`
  - Seeded reset determinism for initial buyer terms and `base_concede`
  - Difficulty-specific profiles
  - Basic accept/reject terminal behavior
  - Heuristic action sanity
  - `env.state` exposure
  - Max-round terminal success/reward metadata
- `tests/test_score_range_guardrails.py`
  - Strict `(0, 1)` score guardrails for helper functions, graders, and terminal
    reward paths
- `tests/test_inference_policy.py`
  - Inference guardrails for hard-task shortcut behavior, contract enforcement,
    close-deal logic, and final log line formatting

OpenEnv manifest compatibility can be checked with:

```powershell
openenv validate
```

## Stage 0 Reference Trace

Before the Stage 0 edits, a small reference trace was captured for `easy`,
`medium`, and `hard` using fixed seeds and a fixed action sequence
(`propose` then `reject`). That trace is used only as a manual regression check
to confirm that rewards, `done`, buyer terms, and terminal metadata remain
unchanged after the documentation updates.
