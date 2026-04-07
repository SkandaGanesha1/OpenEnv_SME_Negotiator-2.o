---
title: Sme Negotiator
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# OpenEnv SME Negotiator: Reinforcement Learning for B2B Contract Negotiation

A rigorous, OpenEnv-compliant reinforcement learning environment for SME B2B contract negotiation. This environment addresses a critical untouched domain identified in the Razorpay "Fix My Itch" dataset: **the asymmetry of negotiation power for Small and Medium Enterprises (SMEs)**.

## Quick Start

### Hugging Face Hub (this Space)

The Python package is **`sme_negotiator_env`** (underscores), the client is **`SMENegotiatorEnv`**, and **`NegotiationAction` has no `message=` field** — use `reason` for text. The server listens on port **7860** (not 8000).

```python
import asyncio
from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.models import NegotiationAction

async def main():
    # Public Space (replace with your Space URL if different)
    base = "https://omkarchaithanya-sme-negotiator.hf.space"
    async with SMENegotiatorEnv(base_url=base) as env:
        result = await env.reset(seed=42, task_name="payment-terms-easy")
        obs = result.observation
        action = NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=max(0, int(obs.buyer_days) - 5),
            use_treds=False,
            reason="Counter-offer to improve payment terms",
        )
        result = await env.step(action)
        print(result.reward, result.done)

asyncio.run(main())
```

Install from the Space (or clone this repo) so `sme_negotiator_env` is importable, e.g. `pip install -e .` from the repository root.

**Contribute via OpenEnv CLI** (optional):

```bash
openenv fork omkarchaithanya/sme-negotiator --repo-id <your-username>/<your-env-repo>
cd <forked-repo>
openenv push omkarchaithanya/sme-negotiator --create-pr
```

This Space exposes an **OpenEnv** HTTP API on port **7860** (`/health`, `/reset`, `/step`, `/state`). Package name on PyPI-style installs: **`openenv-sme-negotiator`**; import path: **`sme_negotiator_env`**.

### Connect from Python (typed client — install the env package first)

```bash
cd openenv-sme-negotiator
uv sync   # or: pip install -e .
```

```python
import asyncio
from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.models import NegotiationAction

async def main():
    # Local server (default port in this repo)
    base = "http://127.0.0.1:7860"
    # Public Hugging Face Space (no trailing slash)
    # base = "https://omkarchaithanya-sme-negotiator.hf.space"

    async with SMENegotiatorEnv(base_url=base) as env:
        result = await env.reset(seed=42, task_name="payment-terms-easy")
        obs = result.observation
        action = NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=max(0, int(obs.buyer_days) - 5),
            use_treds=False,
            reason="Counter-offer",
        )
        result = await env.step(action)
        print(result.observation, result.reward, result.done)

asyncio.run(main())
```

Actions use **`action_type`** (`propose` | `accept` | `reject`), **`price`**, **`payment_days`**, and optional **`use_treds`**, **`reason`**, plus optional **`propose_late_payment_penalty_clause`**, **`propose_dynamic_discounting`**, **`dynamic_discount_annual_rate`** for medium/hard tasks. There is **no** `message=` field on `NegotiationAction`.

### Connect without installing this repo (generic dict client)

```python
import asyncio
from openenv.core import GenericEnvClient

async def main():
    async with GenericEnvClient(base_url="http://127.0.0.1:7860") as env:
        r = await env.reset()
        r = await env.step({"action_type": "propose", "price": 100.0, "payment_days": 80})

asyncio.run(main())
```

### Pull and run the Space locally via OpenEnv (Docker)

Requires Docker. **`from_env` is async** and opens a WebSocket session to the running server.

```python
import asyncio
from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.models import NegotiationAction

async def main():
    env = await SMENegotiatorEnv.from_env("Omkarchaithanya/sme-negotiator", use_docker=True)
    try:
        result = await env.reset(task_name="payment-terms-medium")
        result = await env.step(
            NegotiationAction(
                action_type="propose",
                price=100.0,
                payment_days=55,
                propose_late_payment_penalty_clause=True,
            )
        )
        print(result.reward, result.done)
    finally:
        await env.close()

asyncio.run(main())
```

### Contribute

- **GitHub:** push a branch and open a PR to the project’s GitHub repository.
- **Hub CLI:** if you use the OpenEnv CLI, you can fork/push with `openenv fork` / `openenv push` as in the OpenEnv docs; set **`--repo-id`** to your fork.

## Problem statement (Razorpay [Fix My Itch](https://www.razorpay.com/m/fix-my-itch/))

**Listed problem:** *Why can't SMEs negotiate favorable payment terms with large buyers?* — **B2B Services**, **itch score 82.8**.

**Summary:** Small suppliers often face **60–90+ day** buyer payment terms while paying their own suppliers in **~30 days**, creating **working-capital gaps** and reliance on **expensive short-term loans (~18–24% interest)**. Large buyers may refuse to shorten terms; **no single real-world “mechanism”** fixes this everywhere—this repo provides a **simulated, graded negotiation** for research and RL baselines (not legal advice or enforcement).

### Key regulatory / market context (India-oriented)
- **MSMED Act (India)**: Corporate buyers must settle payments within 45 days
- **TReDS Platform**: Allows SMEs to auction trade receivables for immediate cash
- **Real-world impact**: Millions of SMEs globally struggle with working capital gaps

### Hackathon Spec Alignment

This project is structured to make the evaluation contract easy to audit:

- **`[START]`**: emitted once at the beginning of each episode with task, environment, and model metadata.
- **`[STEP]`**: emitted after every action with the serialized action, step reward, done flag, and parser-safe error field.
- **`[END]`**: emitted once per episode with success flag, number of steps, and the reward trace.

Relevant configuration lives in `inference.py` and is documented here for reviewers:

- **`API_BASE_URL`**: OpenAI-compatible LLM endpoint.
- **`HF_TOKEN`**: token for Hugging Face router or hosted models.
- **`MODEL_NAME`**: chat-capable model id used for `chat/completions`.
- **`OPENENV_BASE_URL`**: negotiation server URL, default `http://127.0.0.1:7860`.
- **`OPENENV_IN_PROCESS`**: set to `1` to run the environment in-process instead of starting `uv run server` separately.
- **`INFERENCE_HARD_TWO_STEP`**: hard-task accept shortcut toggle; default `0` to preserve benchmark integrity.

The practical baseline command is:

```bash
python inference.py
```

If you prefer the managed environment form, use:

```bash
uv run python inference.py
```

### Economic Context Reading Pack (for judges and users)

Use this as a single entry point to external context behind the environment design and scoring assumptions.

- **All-in-one section**: This block consolidates news, policy commentary, financing trends, and practitioner perspective on delayed MSME receivables.

#### News and market reports
- Economic Times: https://m.economictimes.com/small-biz/sme-sector/over-rs-7-3-lakh-crore-in-msme-receivables-stuck-due-to-delayed-payments-basant-kaur-c2fo/articleshow/126496632.cms
- BusinessWorld: https://www.businessworld.in/article/despite-reforms-india-s-msme-delayed-receivables-still-exceed-4-6-of-gva-581337
- Storyboard18: https://www.storyboard18.com/how-it-works/delayed-payments-to-msmes-fall-to-rs-7-34-lakh-crore-report-84822.htm

#### Policy and ecosystem analysis
- Massachusetts Entrepreneurship Center (Delayed Payments): https://massentrepreneurship.org/delayed-payments/
- Corporate Counsel (India legal commentary): https://corporatecounsel.in/post/the-issues-of-delayed-payments-to-msmes-in-india

#### SME financing trends and practitioner perspective
- Recur Club (SME financing trends): https://www.recurclub.com/blog/sme-financing-trends
- LinkedIn practitioner post: https://www.linkedin.com/posts/himanshu-mehta-977949a2_msme-vendorpayments-businessethics-activity-7436185677647749120-h-Jr

## Environment Architecture

### MDP (summary)

- **State**: See `NegotiationObservation` / `NegotiationState` — buyer offer, rounds, cost and liquidity limits, optional working-capital context.
- **Actions**: `action_type` ∈ `propose` | `accept` | `reject` with `price`, `payment_days`, optional `use_treds`, `reason`, and for medium/hard optional `propose_late_payment_penalty_clause`, `propose_dynamic_discounting`, `dynamic_discount_annual_rate` (`sme_negotiator_env/models.py`).

### Rewards (actual behavior)

- **Step rewards**: Non-terminal steps receive a **bounded partial reward** from `SMENegotiatorEnvironment._compute_reward` (progress signal). Set `REWARD_DEBUG=0` to silence `[REWARD_DEBUG]` logs.
- **Output streams**: `[START]`, `[STEP]`, `[END]` lines are emitted on `stdout` for evaluators; `[REWARD_DEBUG]` is emitted on `stderr` and is not part of judge parsing.
- **Terminal reward**: Deterministic **task grader** on `NegotiationState` in `sme_negotiator_env/graders.py` (0.0–1.0).

### Deterministic graders (terminal score)

| Task | `task_name` | What gets full credit |
|------|-------------|------------------------|
| Easy | `payment-terms-easy` | Deal reached and agreed days ≤ **liquidity threshold** (60d); partial tier if ≤ cap+15d |
| Medium | `payment-terms-medium` | Agreed days ≤ **45d** (liquidity cap), or stricter band with **late payment penalty** flag |
| Hard | `payment-terms-hard` | **Dynamic discounting** agreed (`propose_dynamic_discounting`) — score from **NPV vs status quo** (`compute_financing_npv_vs_status_quo`). **Not** graded on `use_treds` alone |

`use_treds` still affects **simulation** (buyer day floor) and observations; for **hard task terminal score**, optimize **dynamic discounting + NPV**, not TReDS alone. For medium/hard trajectories with large payment-day gaps, include at least one `use_treds=true` proposal so the mechanism is exercised.

The default inference runner includes a conservative guardrail: in medium/hard tasks, if `buyer_days > liquidity_threshold + 10` during early rounds, one `propose` action is upgraded to `use_treds=true` so the TReDS mechanic is not left dormant.

### RL training stacks

This repo ships the **environment** and LLM **baseline** (`inference.py`). For PPO / GRPO / TRL, wire this env like any other OpenEnv — there is **no** bundled `src.rl` or `src.data_structures` in this repository.

## Task stratification (configured in `sme_negotiator_env/task_config.py`)

### Easy — `payment-terms-easy`

- Buyer opens at **100** / **90d**; liquidity threshold **60d**; **10** rounds.
- Grader: agreed days vs **60d** cap (see `grade_task_payment_terms_easy`).

### Medium — `payment-terms-medium`

- Buyer opens at **100** / **60d**; liquidity **45d**; **12** rounds; optional late-penalty clause for a partial tier.
- Grader: `grade_task_payment_terms_medium`.

### Hard — `payment-terms-hard`

- Consortium-style settings (volume, buyer power); **16** rounds.
- Grader: `grade_task_dynamic_discounting_hard` (NPV improvement when dynamic discounting is agreed).

## Reward Visibility & Logging

All step-by-step rewards are now **visible and logged** at three levels:

1. **Episode Level**: Each step prints instantaneous reward
2. **Cumulative Level**: Running total reward shown after each action
3. **Summary Level**: Complete reward breakdown in JSON results file

Example output:
```
📍 ROUND 1/12
   Action: PROPOSE       | Price: ₹95.00   | Days: 30  | TReDS: False
   ✓ Step Reward:     +0.102347
   ✓ Cumulative:      0.102347
   → Buyer Counter:   ₹98.00/unit @ 32 days
───────────────────────────────────────────────────────────────────────

📊 EPISODE SUMMARY
═══════════════════════════════════════════════════════════════════════
💰 REWARD BREAKDOWN:
  Final Round Score:     0.458923
  Cumulative Reward:     1.240567  ← TOTAL EARNED
  Average Reward/Step:   0.155071
  Reward History:        [0.102347, 0.145289, 0.124568, 0.145290, 0.156890, 0.165203]
```

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd openenv-sme-negotiator

uv sync
# or: pip install -e .

# Optional dev dependencies
uv pip install -e ".[dev]"
```

## Quick start: in-process environment

```python
from server.sme_environment import SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction

env = SMENegotiatorEnvironment()
obs = env.reset(seed=42, difficulty="easy")

print(f"Buyer: ₹{obs.buyer_price}/unit, {obs.buyer_days} days | cost ≤ ₹{obs.cost_threshold} | liquidity ≤ {obs.liquidity_threshold}d")

action = NegotiationAction(
    action_type="propose",
    price=95.0,
    payment_days=55,
    use_treds=False,
    reason="Counter-offer within liquidity target",
)
obs = env.step(action)
print(obs.reward, obs.done, obs.message)
```

## Running the server

### Local

```bash
uv run server
# listens on http://0.0.0.0:7860 — same as `openenv.yaml` and Docker
```

### Docker

```bash
docker build -f docker/Dockerfile -t openenv-sme-negotiator:latest .
docker run -p 7860:7860 openenv-sme-negotiator:latest
```

### Hugging Face Spaces Deployment
```bash
# Install OpenEnv CLI (when available)
pip install openenv-cli

# Deploy to HF Space
openenv deploy --space-id your-username/sme-negotiator
```

## Heuristic policy (built-in)

```python
from sme_negotiator_env.client import choose_action
from server.sme_environment import SMENegotiatorEnvironment

env = SMENegotiatorEnvironment()
obs = env.reset(seed=123, difficulty="medium")
total = 0.0
for round_number in range(32):
    if obs.done:
        break
    action = choose_action(obs, round_number)
    obs = env.step(action)
    total += float(obs.reward or 0.0)
print("final_reward", obs.reward, "cumulative_steps", total)
```

## Evaluation Methodology

### Phase 1: Automated Validation
- ✅ OpenEnv spec compliance
- ✅ HF Spaces deployment
- ✅ Deterministic grader (no LLM-as-judge)
- ✅ Isolated Docker containerization

### Phase 2: Baseline Performance
- Run Nemotron 3 Super on all tasks (100 episodes per task)
- Verify score variance (non-zero)
- Establish performance envelope

### Phase 3: Human Review
- Real-world economic utility
- Exploit prevention
- Regulatory authenticity

## Scoring Benchmark

Expected baseline (Nemotron 3 Super) performance:
| Task | Easy | Medium | Hard |
|------|------|--------|------|
| Mean Score | 0.88 | 0.62 | 0.08 |
| Pass Rate (score > 0.3) | 100% | 85% | 12% |

(These are illustrative; actual results validate environment quality)

## Key Design Principles

1. **Deterministic & Reproducible**: Fixed seeds guarantee identical trajectories
2. **Secure & Sandboxed**: Server-side grader immune to reward hacking
3. **Realistic Constraints**: Based on actual MSMED Act regulations and TReDS mechanics
4. **Multi-Modal Reasoning**: Requires both quantitative (financial) and qualitative (LLM) capabilities
5. **Scalable**: Supports async rollouts for distributed RL training

## Project structure

```
openenv-sme-negotiator/
├── docker/Dockerfile
├── server/
│   ├── app.py              # create_app + WebSocket teardown workaround
│   ├── environment.py      # SMENegotiatorEnvironment (MDP)
│   ├── sme_environment.py  # re-export
│   └── concurrency.py
├── sme_negotiator_env/
│   ├── models.py
│   ├── task_config.py
│   ├── graders.py
│   ├── client.py
│   └── llm_action_parser.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
└── tests/test_environment.py
```

## Hard task: dynamic discounting (matches `graders.py`)

Terminal score uses **NPV improvement vs status quo** when `propose_dynamic_discounting` is agreed. See `grade_task_dynamic_discounting_hard` and `compute_financing_npv_vs_status_quo`. Use the action flags `propose_dynamic_discounting` / `dynamic_discount_annual_rate` — not TReDS alone — for hard-task credit.

## References

1. **MSMED Act**: Section 43B(h), Income Tax Act 1961
2. **TReDS Platform**: RBI Trade Receivables Discounting System
3. **OpenEnv Spec**: https://huggingface.co/docs/openenv
4. **Razorpay "Fix My Itch"**: https://www.razorpay.com/reports/fix-my-itch
5. **Nemotron 3 Super**: NVIDIA's Hybrid Mamba-Transformer model

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{openenv_sme_negotiator,
  title={OpenEnv SME Negotiator: B2B Payment-Term Negotiation Environment},
  year={2026},
  url={https://github.com/YOUR_USERNAME/openenv-sme-negotiator}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Baseline LLM inference (`inference.py`)

Uses an **OpenAI-compatible** API (default: [Hugging Face Inference router](https://router.huggingface.co/v1)) and either:

- **`OPENENV_IN_PROCESS=1`** — runs `SMENegotiatorEnvironment` in-process (no separate server), or  
- **`OPENENV_BASE_URL=http://127.0.0.1:7860`** — talks to `uv run server` over HTTP/WebSocket.

### Setup

```bash
cp .env.example .env
# Set HF_TOKEN for Hugging Face models; or use another provider’s API key + API_BASE_URL
```

### Run

```bash
# Terminal 1 (optional if not using OPENENV_IN_PROCESS=1)
uv run server

# Terminal 2
uv run python inference.py
```

You can also run the script directly from an activated Python environment:

```bash
python inference.py
```

Outputs are written to `inference_results.json` (gitignored by default).

### Environment variables (see `inference.py`)

| Variable | Role |
|----------|------|
| `API_BASE_URL` | LLM OpenAI-compatible base URL |
| `HF_TOKEN` | Token for Hugging Face router / hosted models |
| `MODEL_NAME` | Chat-capable model id for `chat/completions` (e.g. `Qwen/Qwen2.5-7B-Instruct` on HF Router; some Mistral ids are not routed as chat) |
| `OPENENV_BASE_URL` | Negotiation server (default `http://127.0.0.1:7860`) |
| `OPENENV_IN_PROCESS` | `1` = no separate server |
| `INFERENCE_HARD_TWO_STEP` | Hard-task accept shortcut toggle. **Default disabled** (`0`) for benchmark integrity; set `1` only for debugging/smoke runs. |

## Contact & support

Use your repository’s **Issues** and **Discussions** tabs on GitHub.
