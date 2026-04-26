# Stage 6 Self-Play and Simulated Experts

Stage 6 extends the liquidity training path only. The live OpenEnv server and
deterministic in-environment grading stay unchanged.

## Scope

- `server.app` still serves the legacy `SMENegotiatorEnvironment`
- `openenv.yaml` is unchanged
- Stage 6 lives in the in-process RL stack under `rl/`
- deterministic graders and reward formulas remain the primary objective

## Self-Play

Self-play is implemented through training-side opponent policies in
`rl/opponents.py`.

- buyer policies implement `TextPolicy.act(observation) -> NegotiationAction`
- financier policies implement `FinancierPolicy.act(observation) -> FinancierQuote`
- heuristic policies preserve the current default behavior
- snapshot-backed LLM policies are optional and are loaded from saved SME
  checkpoints

When `SMENegotiatorEnvironment` or `SMELiquidityEnvironment` is created without
an injected policy, behavior remains unchanged. When a policy is injected, the
environment uses the same state-transition path but swaps in the policy's buyer
response or financier quote.

The training scripts maintain a rolling local snapshot zoo:

- snapshots are saved under `output_dir/snapshots/`
- the latest snapshot is sampled most often
- older snapshots are still sampled
- heuristics remain as a fallback before warmup and for stability

## Auto-Curriculum

Curriculum logic lives in `rl/curriculum.py` and only adjusts supported real
knobs:

- `total_periods`
- `buyer_variance`
- `financier_variance`

Stage 6 does not introduce true multi-SME competition. Variance remains
deterministic because it is derived from the environment seed.

Promotion is based on moving averages of:

- base RL reward
- SME default rate

The trainer callbacks record `EpisodeSummary` values and only promote after the
window is full and thresholds are satisfied.

## Simulated Experts

Persona-weighted rubric overlays live entirely outside the environment in
`rl/rubrics.py` and `rl/self_rewarding_dpo.py`.

- personas include Conservative CFO, Aggressive Founder, and Regulator
- persona sampling is deterministic for a fixed seed
- rubric judging is provider-agnostic and optional
- the rubric overlay is added only in training code

The environment itself never calls an external judge and never mutates its
deterministic reward semantics because of rubric scoring.

## Self-Rewarding / DPO

`rl/self_rewarding_dpo.py` provides a minimal dataset-building path:

- collect recent episode logs
- score them with an external scorer
- generate persona-weighted pairwise preferences
- write JSONL preference data for later DPO-style fine-tuning

This companion flow is intentionally separate from the main GRPO loop.
