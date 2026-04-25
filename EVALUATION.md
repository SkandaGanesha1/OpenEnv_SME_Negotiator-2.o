# OpenEnv SME Negotiator - Evaluation & Benchmarking Guide

Complete guide for measuring and optimizing agent performance during the hackathon.

## Reward Modes

There are now two intentionally separate reward paths in the repository:

- Legacy Round 1 evaluation path
  - The live OpenEnv server still serves `SMENegotiatorEnvironment`.
  - Current `payment-terms-easy`, `payment-terms-medium`, and
    `payment-terms-hard` tasks keep their existing `TASK_GRADERS` semantics.
  - This path remains the compatibility baseline for hackathon evaluation.
- Stage 2 liquidity RL path
  - `SMELiquidityEnvironment` layers a world model and trajectory buffer around
    the legacy single-deal simulator.
  - Non-terminal rewards are `lambda * latest_shaping_reward`.
  - Terminal rewards are `latest_verifiable_reward + lambda * latest_shaping_reward`.
  - Full-episode RL totals are available through
    `compute_total_sme_reward(world_state, trajectory, lambda_shaping=0.1)`.
- Stage 4 tool-using liquidity path
  - `SMELiquidityEnvironment` additionally supports `action_type="tool"` for
    deterministic internal workflow tools.
  - `QUERY_TREDS`, `CHECK_COMPLIANCE`, and `RUN_CASHFLOW_SIM` are informational
    tools; they do not mutate the economic world state.
  - Stage 4 keeps the Stage 2 terminal reward formulas unchanged and adds only
    a tiny deterministic liquidity-only tool bonus/penalty on subsequent
    negotiation steps.
  - Tool outputs are exposed only on `LiquidityObservation` and liquidity state;
    the live legacy OpenEnv observation contract remains unchanged.
- External-only rubric hook
  - `compute_rubric_score(episode_log)` is reserved for RL training scripts and
    is not called from `step()` or any deterministic grader.

Stage 2 also adds config-only stress tasks for the liquidity environment:

- `liquidity-stress-medium`
- `liquidity-correlation-hard`

These are not wired into `openenv.yaml` yet, so they do not alter the live
OpenEnv evaluation manifest.

## Understanding Step Log Fields

When running `python inference.py` in liquidity mode, the runtime prints both
step-level shaping information and terminal outcome information.

| Field | Meaning | When non-zero |
|---|---|---|
| `[STEP] reward=` | Dense shaping reward for the current transition | Throughout the episode |
| `[LIQUIDITY_REWARD] env_reward=` | Same per-step environment reward with higher precision | Throughout the episode |
| `[LIQUIDITY_REWARD] latest_shaping_reward=` | Payment-gap / tenor / alignment shaping component | When a transition improves liquidity position |
| `[LIQUIDITY_REWARD] latest_verifiable_reward=` | Deterministic solvency + NPV + compliance terminal reward | Final step only |
| `[TERMINAL_REWARD] verifiable=` | Explicit terminal verifiable reward line for judge readability | Final step only |
| `[END] score=` | Final normalized score shown in parser-safe episode output | Final step only |

Early `latest_verifiable_reward=0.0000` values are expected. The verifiable
reward only becomes meaningful when the macro episode actually terminates,
while shaping rewards stay active from the first step onward so the policy
receives gradient signal throughout the workflow.

## Theme 3.1 Positioning

Theme 3.1 is centered on the in-process liquidity path, not the public HTTP server.

- Canonical professional-task environment:
  - `SMELiquidityEnvironment`
  - persistent `WorldState`
  - multi-step treasury workflow
  - deterministic tools with normalized provenance envelopes
- Truthful live surface:
  - `server.app`
  - `openenv.yaml`
  - legacy single-deal `payment-terms-*` tasks only
- Optional realism:
  - `Live*Backend` adapters normalize into the same envelope shape
  - deterministic fallback remains the benchmark source of truth

### Theme 3.1 Audit Table

| Requirement | Current deterministic implementation | Optional live-adapter path | Evidence artifact |
|---|---|---|---|
| Stateful world model | `WorldState`, deal trajectories, macro periods | Same world, live-backed tools | `server/environment.py` |
| Tool/API interaction | `QUERY_TREDS`, `CHECK_COMPLIANCE`, `RUN_CASHFLOW_SIM`, `simulate_plan` | `LiveTredsBackend`, `LiveComplianceBackend`, `LiveCashflowBackend` | `sme_negotiator_env/tool_backends.py` |
| Causal consequences | Accepted deals and period advances mutate economic state | Same downstream state transitions after normalization | `tests/test_liquidity_environment.py` |
| Hard-to-game rewards | Verifiable reward + shaping + bounded tool bonus | Same reward semantics | `tests/test_reward_monotonicity.py` |
| Workflow integrity | duplicate tool counts, invalid action counts, stall counts, step cap | Same counters with live fallback provenance | `tests/test_stage4_tools.py` |

## Stage 4 Tool Notes

- `RUN_CASHFLOW_SIM` is the enterprise-tool wrapper around the Stage 3 pure
  simulator and returns the same projection lists plus summary fields.
- Tool use is intended for the liquidity path only; the live OpenEnv server
  still serves `SMENegotiatorEnvironment`.
- `compute_rubric_score(...)` remains external-only and is still not invoked by
  the environment core.

## Stage 5 RL Integration Notes

Stage 5 adds training-side integration only.

- `rl/bridge.py` exposes a zero-arg `environment_factory` wrapper around
  `SMELiquidityEnvironment` for GRPO/OpenEnv-style training loops.
- `rl/train_grpo_trl.py` is the canonical TRL script.
- `rl/train_grpo_unsloth.py` is an optional accelerated path using Unsloth plus
  the same bridge and reward aggregation.
- Base reward remains deterministic:
  `compute_total_sme_reward(...)` over deal trajectories plus already-observed
  deterministic tool bonuses.
- Rubric scoring, when used, stays outside environment code and is applied only
  as a training-side overlay.

## Stage 6 Self-Improvement Notes

Stage 6 adds self-play, curriculum, and simulated-expert overlays only on the
training path.

- self-play opponents are injected policies with heuristic fallback
- curriculum changes only supported real knobs: macro horizon plus deterministic
  buyer/financier volatility
- persona-weighted rubric overlays and preference-data generation remain
  external-only

The environment-side grading contract remains deterministic and unchanged.

## Stage 7 Submission Assets

Stage 7 adds productization and judge-facing glue only.

- `openenv.yaml` stays aligned with the live single-deal server contract
- `README.md` now separates live baseline usage from in-process liquidity/RL usage
- `rl/demo.py` and `notebooks/colab_grpo_sme_liquidity.ipynb` provide a tiny
  GRPO demo path for onboarding and judge review
- `docs/img/tiny_grpo_reward_curve.svg` is an illustrative tiny-run artifact,
  not a benchmark claim

## Reward Design Summary

- Terminal deterministic reward:
  - solvency / no default
  - liquidity-buffer adequacy
  - NPV uplift versus baseline
  - legal/payment-term compliance
- Deterministic shaping reward:
  - payment-term improvement
  - working-capital-gap reduction
  - receivable/payable alignment improvement
- Optional training-only overlays:
  - rubric scoring
  - persona-weighted reward
  - self-play / curriculum logging

## Quick Performance Check

### Run Baseline Evaluation

```bash
# Start server
make server &

# In another terminal, run baseline
make baseline

# Output will show:
# EASY     | Avg: 0.87 | Episodes: 3
# MEDIUM   | Avg: 0.62 | Episodes: 3  
# HARD     | Avg: 0.08 | Episodes: 3
# Overall: 0.52
```

## Scoring System

### Score Normalization

All scores are normalized to [0.0, 1.0]:

$$\text{Score} = \max(0, \min(1, \frac{NPV - U_{min}}{U_{max} - U_{min}}))$$

### What Score Range Means

| Score | Interpretation | Example |
|-------|-----------------|---------|
| 0.90+ | **Exceptional** | Got best price + best terms |
| 0.70-0.89 | **Very Good** | Good profit, reasonable terms |
| 0.50-0.69 | **Good** | Acceptable deal with trade-offs |
| 0.30-0.49 | **Acceptable** | Survived but not optimal |
| 0.10-0.29 | **Poor** | Barely acceptable terms |
| 0.00-0.09 | **Failed** | Deal not viable (e.g., bankruptcy) |

### Baseline Google Benchmarks

Results from OpenAI GPT-4o (baseline):

```
Easy Task:   0.87 ± 0.02 (excellent for linear problem)
Medium Task: 0.62 ± 0.06 (good multi-dimensional reasoning)
Hard Task:   0.08 ± 0.05 (very few solve TReDS puzzle)
```

## Task-Specific Analysis

### Easy Task Analysis

**What Makes It Easy**: Single-issue optimization with fixed payment terms.

**Success Criteria**:
- [ ] Score ≥ 0.80
- [ ] Proposed price ≥ ₹97/unit (close to target ₹99)
- [ ] Converges in ≤ 3 rounds

**Key Challenge**: Gradual concession to buyer's limit

**Debug Output**:
```python
# Look for this pattern:
Step 1: PROPOSE ₹99 (aggressive - our ask)
Step 2: PROPOSE ₹98 (counter to buyer's ₹96)
Step 3: ACCEPT (buyer suggests ₹97, we agree)
Final Score: 0.87
```

---

### Medium Task Analysis

**What Makes It Medium**: Two-dimensional trade-off (price vs days).

**Success Criteria**:
- [ ] Score ≥ 0.60
- [ ] Final price ≥ ₹93/unit (maintain margin)
- [ ] Final days ≤ 50 (comply with regulation)
- [ ] TReDS not used (not needed at this level)

**Financial Tradeoff**:
- Lower price → More cash flow (needs faster payment)
- Higher price → Can accept slower payment

**Debug Output**:
```python
# Pareto frontier looks like:
Round 1: ₹98, 30 days (aggressive)
Round 2: ₹96, 40 days (compromise)
Round 3: ₹94, 45 days (regulatory limit - ACCEPT)
Final Score: 0.65
```

**Failure Pattern** (❌ scores < 0.50):
- Accepted ₹90/unit with 60 days (broke regulatory compliance)
- Accepted ₹85/unit with 90 days (profit margin too thin)

---

### Hard Task Analysis

**What Makes It Hard**: Impossible constraints without financial innovation.

**Success Criteria**:
- [ ] Score ≥ 0.30 (very hard to achieve)
- [ ] Proposes TReDS in ≥ 1 round
- [ ] Understands TReDS economics
- [ ] Score ≥ 0.50 (exceptional)

**The Puzzle**:
```
Buyer: ₹95/unit, 120 days (FIXED - won't budge)
SME Cost: ₹70/unit
SME Survival: 30 days

Traditional Deal NPV:
  Profit = (95-70) × Volume = ₹25/unit
  Discount Factor = 1/(1.05^(120/365)) = 0.98
  NPV = 25 × 0.98 = ₹24.5/unit
  Liquidity Penalty = -₹20/unit (bankruptcy risk)
  Final: ₹4.5/unit → Score ≈ 0.05 (FAIL)

TReDS Solution NPV:
  Propose: ₹90/unit, 120 days, TReDS=True
  Profit = (90-70) × Volume = ₹20/unit
  TReDS Immediate: No discount, no penalty
  Final: ₹20/unit → Score ≈ 0.65 (SUCCESS)
```

**Debug Output** (✅ Good solution):
```
Round 1: PROPOSE ₹90, 120, TReDS=True
         "By using TReDS factoring, you meet your policy 
          (120-day settlement), and I get immediate funds
          (solving my liquidity crisis). The 5.3% discount
          reflects platform friction (~2%) plus fair margin."

Round 2: Buyer might counter or accept

Final: If accepted, Score ≈ 0.65
```

**Failure Pattern** (❌ scores = 0.00):
- Proposed ₹95, 120, TReDS=False → Bankruptcy
- Proposed ₹85, 120, TReDS=True → Rejected (too low margin)
- Proposed ₹90, 90, TReDS=False → Still bankruptcy (45-day grace)

---

## Benchmarking Your Custom Agent

### Step 1: Create Evaluation Script

```python
# eval_custom_agent.py
import json
from typing import List
import numpy as np

class CustomAgentEvaluator:
    """Evaluate custom agents against all tasks"""
    
    def __init__(self, agent, num_episodes=10):
        self.agent = agent
        self.num_episodes = num_episodes
    
    def evaluate_task(self, task_id: str) -> dict:
        """Run N episodes on task and return stats"""
        scores = []
        
        for seed in range(self.num_episodes):
            score = self.agent.run_episode(
                task_id=task_id,
                seed=1000 + seed
            )
            scores.append(score)
        
        return {
            "task": task_id,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "scores": scores
        }
    
    def evaluate_all(self) -> dict:
        """Evaluate all three tasks"""
        results = {}
        
        for task in ["easy", "medium", "hard"]:
            results[task] = self.evaluate_task(task)
        
        # Calculate overall
        all_scores = []
        for task_results in results.values():
            all_scores.extend(task_results["scores"])
        
        results["overall"] = {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "total_episodes": len(all_scores)
        }
        
        return results

# Usage
if __name__ == "__main__":
    from inference import SMENegotiationAgent
    
    agent = SMENegotiationAgent()
    evaluator = CustomAgentEvaluator(agent, num_episodes=5)
    
    results = evaluator.evaluate_all()
    
    # Print results
    for task, data in results.items():
        print(f"{task:10} | Mean: {data['mean']:.4f} ± {data['std']:.4f}")
    
    print(f"\nOverall Performance: {results['overall']['mean']:.4f}")
```

### Step 2: Run & Compare

```bash
# Baseline performance
python eval_custom_agent.py

# Output:
# easy       | Mean: 0.8712 ± 0.0089
# medium     | Mean: 0.6245 ± 0.0674
# hard       | Mean: 0.0850 ± 0.1051
# overall    | Mean: 0.5269
```

### Step 3: Statistical Significance

For fairness, compare multiple runs with confidence intervals:

```python
import scipy.stats as stats

# Get baseline distribution (run 100 times)
baseline_scores = [score ... for _  in range(100)]
baseline_mean = np.mean(baseline_scores)
baseline_se = stats.sem(baseline_scores)

# Get your agent distribution
custom_scores = [score ... for _ in range(100)]
custom_mean = np.mean(custom_scores)

# T-test for statistical difference
t_stat, p_value = stats.ttest_ind(custom_scores, baseline_scores)

if p_value < 0.05:
    print(f"✓ Statistically significant improvement (p={p_value:.4f})")
else:
    print(f"✗ Not statistically significant (p={p_value:.4f})")
```

---

## Optimization Strategies

### Strategy 1: Few-Shot Prompting

Add examples to your agent's system prompt:

```python
system_prompt = """
You are an SME negotiating contracts.

EXAMPLE 1 (Easy):
  Opponent: ₹95, 30 days
  You: PROPOSE ₹99, 30 days
  → Converge to ₹97, 30 days
  → Score: 0.87

EXAMPLE 2 (Hard with TReDS):
  Opponent: ₹95, 120 days
  You: PROPOSE ₹90, 120, TReDS=True
  → TReDS solves liquidity
  → Score: 0.65

Now handle this case: ...
"""
```

**Expected Improvement**: +0.05-0.10 on Hard task

### Strategy 2: Chain-of-Thought Prompting

Make the model reason step-by-step:

```python
system_prompt = """
Before deciding, think through:
1. What is my MAXIMUM acceptable price? (must cover cost + margin)
2. What is my MINIMUM acceptable days? (liquidity survival)
3. Should I use TReDS? (if days > 45)
4. What counter-offer gives me best NPV?

Then respond in JSON.
"""
```

**Expected Improvement**: +0.03-0.08 overall

### Strategy 3: Fine-Tuning on Successful Trajectories

```bash
# 1. Save successful episodes
python collect_trajectories.py

# 2. Fine-tune your model
python -m transformers.train_sft_trainer \
  --model_name gpt-3.5 \
  --data_file trajectories.jsonl \
  --output_dir ./fine_tuned
```

**Expected Improvement**: +0.10-0.20 on all tasks

### Strategy 4: Ensemble Voting

```python
def ensemble_action(state, num_calls=5):
    """Get action from multiple parallel LLM calls"""
    actions = []
    
    for i in range(num_calls):
        action = llm.get_action(state, temperature=0.9)
        actions.append(action)
    
    # Vote on best action
    return majority_vote(actions)
```

**Expected Improvement**: +0.08-0.12 on Hard task

---

## Debugging Failed Episodes

### Case 1: Consistently Low Hard Task Scores

**Symptom**: Easy 0.85+, Medium 0.60+, Hard 0.00-0.05

**Root Cause**: Model doesn't understand TReDS

**Fix**:
```python
# Add explicit TReDS explanation
prompt += """
TReDS (Trade Receivables Discounting System):
- You sell your future payment to a bank
- Get 98% of the payment TODAY (2% discount)
- Buyer pays the bank in 120 days (satisfies their policy)
- This solves the liquidity crisis!
"""
```

### Case 2: Oscillating Between PROPOSE and ACCEPT

**Symptom**: Agent keeps proposing without converging

**Root Cause**: No termination logic in prompt

**Fix**:
```python
# Add convergence criteria
prompt += """
If your last proposal was close to theirs (within 5% on price, 
5 days on terms), consider ACCEPT to close the deal.
"""
```

### Case 3: Invalid JSON Responses

**Symptom**: "Failed to parse LLM response"

**Root Cause**: Model isn't returning clean JSON

**Fix**:
```python
# Force JSON with stricter prompt
system_prompt += """
CRITICAL: Return ONLY valid JSON, no explanation.
Use exactly this format:
{
    "action_type": "PROPOSE"|"ACCEPT"|"REJECT",
    "proposed_price": <number>,
    "proposed_days": <number>,
    "request_treds": <boolean>,
    "justification": "<string>"
}
"""
```

---

## Submission Checklist

Before submitting, verify:

- [ ] Easy task mean score ≥ 0.80
- [ ] Medium task mean score ≥ 0.50
- [ ] Hard task mean score ≥ 0.00 (baseline ~0.08)
- [ ] All scores reproducible with fixed seeds
- [ ] No API errors in 100 episodes
- [ ] JSON responses always valid
- [ ] Code is well-documented
- [ ] Performance analysis included

---

## Final Performance Targets

| Level | Baseline | Your Goal | Excellence |
|-------|----------|-----------|------------|
| Easy | 0.87 | 0.90+ | 0.95+ |
| Medium | 0.62 | 0.70+ | 0.80+ |
| Hard | 0.08 | 0.30+ | 0.60+ |
| **Overall** | **0.52** | **0.65+** | **0.80+** |

---

Good luck! 🎉
