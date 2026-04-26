# Judge Readiness Report

## Compliance Checklist

| Item | Status |
|---|---|
| `openenv_manifest` | PASS |
| `fastapi_app` | PASS |
| `typed_models` | PASS |
| `client_package` | PASS |
| `dockerfile` | PASS |
| `trl_training_script` | PASS |
| `unsloth_training_script` | PASS |
| `colab_notebook` | PASS |
| `hf_space_documented` | PASS |
| `hf_blog_draft` | PASS |
| `hf_model_card` | PASS |
| `hf_router_inference` | PASS |

## Task Scores

| Task | Surface | Mean reward | Success rate | Episodes | Notes |
|---|---|---:|---:|---:|---|
| `payment-terms-easy` | OpenEnv HTTP-compatible legacy task | 0.990000 | 1.00 | 3 | Runs through SMENegotiatorEnvironment with reset/step and typed actions. |
| `payment-terms-medium` | OpenEnv HTTP-compatible legacy task | 0.990000 | 1.00 | 3 | Runs through SMENegotiatorEnvironment with reset/step and typed actions. |
| `payment-terms-hard` | OpenEnv HTTP-compatible legacy task | 0.713773 | 1.00 | 3 | Runs through SMENegotiatorEnvironment with reset/step and typed actions. |
| `liquidity-stress-medium` | Advanced in-process liquidity task used by GRPO training | 0.808521 | 1.00 | 3 | resolved=2, defaulted=0, avg_days=44.0, tools=1/1; resolved=2, defaulted=0, avg_days=46.0, tools=1/1; resolved=2, defaulted=0, avg_days=49.0, tools=1/1 |
| `liquidity-correlation-hard` | Advanced in-process liquidity task used by GRPO training | 0.935427 | 1.00 | 3 | resolved=4, defaulted=0, avg_days=71.5, tools=1/1; resolved=4, defaulted=0, avg_days=70.5, tools=1/1; resolved=4, defaulted=0, avg_days=67.5, tools=1/1 |

## Judge Mapping

- **environment_innovation_40**: SME payment-term negotiation with multi-agent buyer/financier/regulator state, cashflow simulation, and TReDS-style tools.
- **storytelling_30**: README, HF blog draft, model card, crisis statistics, and Space link explain the real-world working-capital gap.
- **reward_improvement_20**: Use Colab GRPO reward_curve.png plus this deterministic report; train medium before hard to avoid all-zero reward.
- **reward_pipeline_10**: TRL and Unsloth scripts use deterministic rewards, shaping, tool bonuses, curriculum, self-play hooks, and LoRA training.

## Critical Recommendations

- Expose the advanced liquidity task through the hosted OpenEnv surface or explicitly tell judges the Space is legacy while Colab/TRL uses the advanced in-process environment.
- Do not submit the 10-step hard-mode run as final evidence; use at least a 100-step medium run and a 300-step hard/curriculum run.
- Commit reward_curve.png and this report artifact so judges do not need to rerun Colab to see progress.
- Keep HF router credentials in environment variables only; never commit tokens.
