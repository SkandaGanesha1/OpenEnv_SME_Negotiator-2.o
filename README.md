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

## Problem Statement

SMEs face severe liquidity crises due to extended payment terms imposed by large corporate buyers. With an "Itch Score" of 82.8 (highest in B2B sectors), the negotiation friction is both massive and current unsolved by existing RL benchmarks.

### Key Regulatory Context
- **MSMED Act (India)**: Corporate buyers must settle payments within 45 days
- **TReDS Platform**: Allows SMEs to auction trade receivables for immediate cash
- **Real-world impact**: Millions of SMEs globally struggle with working capital gaps

## Environment Architecture

### Markov Decision Process (MDP) Formulation

**State Space** ($S$):
- Current opponent proposal (price, days, volume, TReDS willingness)
- Negotiation history (chronological offers and justifications)
- SME constraints (production cost, liquidity threshold, market rates)
- Temporal context (current round, deadline)

**Action Space** ($A$):
- **PROPOSE**: Counter-offer with price/days and natural language justification
- **ACCEPT**: Agree to opponent's current offer
- **REJECT**: Walk away from negotiation

**Reward Function**:
- Intermediate: Always 0.0 (forces long-term credit assignment)
- Terminal: Normalized NPV (0.0-1.0) factoring liquidity penalties and regulatory compliance

### Deterministic Grader

The environment implements a **cheating-proof, reproducible grader** based on closed-form mathematical formulas:

$$Score = \max(0.0, \min(1.0, \frac{U - U_{min}}{U_{max} - U_{min}}))$$

Where:
- $U = NPV_{base} - \Omega(D_{final})$  (final utility)
- $NPV_{base} = Profit \times \left(\frac{1}{(1+r)^{D/365}}\right)$
- $\Omega(D) = \begin{cases} 0 & \text{if } D \le 45 \text{ or } TReDS \\ NPV \times e^{(D-45)/30} - 1 & \text{otherwise} \end{cases}$

## Task Stratification

### Easy Task: Single-Issue Price Optimization
- **Goal**: Maximize unit price while payment terms are fixed at 30 days
- **Complexity**: Simple linear concession strategy
- **Expected Scores**: 0.85-0.95 (most LLMs should succeed)

**Use Case**: Baseline sanity check for environment integration

### Medium Task: Bi-Dimensional Trade-Off with Regulatory Boundaries
- **Goal**: Balance price margin vs payment term reduction
- **Constraint**: SME survival threshold at 60 days; buyer rigid below that
- **Regulatory Limit**: 45-day MSMED Act compliance
- **Expected Scores**: 0.50-0.75 (requires financial reasoning)

**Use Case**: Test Pareto frontier exploration and multi-dimensional optimization

### Hard Task: Non-Linear Financial Restructuring with TReDS
- **Goal**: Overcome rigid buyer constraints via TreDS financial engineering
- **Constraint**: 
  - Buyer absolutely cannot concede below 90 days (treasury lock)
  - SME only survives 30 days
  - Impossible deadline for traditional negotiation
- **Solution Path**: Restructure deal via TReDS, offering strategic price discount
- **Expected Scores**: Near 0.0 for standard models; >0.5 for frontier models

**Use Case**: Test complex multi-hop financial reasoning and legal/regulatory knowledge

## Advanced RL Mechanisms

### PPO (Proximal Policy Optimization) Analysis

The environment includes sophisticated PPO analysis tools for policy gradient optimization:

**Key Components:**
- **Generalized Advantage Estimation (GAE)**: Low-variance advantage estimates using temporal smoothing
- **Clipped Policy Loss**: Trust region enforcement via probability ratio clipping (epsilon=0.2)
- **Entropy Regularization**: Exploration encouragement through action entropy bonus
- **Value Function Decomposition**: Separate critic network for better credit assignment

**Usage:**
```python
from src.rl.advanced_mechanisms import PPOAnalyzer, PPOConfig

config = PPOConfig(
    learning_rate=1e-4,
    clip_ratio=0.2,
    gae_lambda=0.95,  # Advantage smoothing
    entropy_coeff=0.01,
)

analyzer = PPOAnalyzer(config)
advantages, returns = analyzer.compute_gae(rewards, values)
policy_loss, metrics = analyzer.compute_policy_loss(old_probs, new_probs, advantages)
```

### Advanced Data Structures

#### 1. Negotiation Graph (MCTS-Style)
Graph-based history of negotiation trajectory with value propagation:
```python
from src.data_structures import NegotiationGraph

graph = NegotiationGraph("episode_001")
graph.add_state("s0", round=0, p_opp=100.0, d_opp=30, v_opp=100, reward=0.0)
graph.add_state("s1", round=1, p_opp=95.0, d_opp=30, v_opp=100, reward=0.1, 
                parent_state="s0", action_taken="PROPOSE")

critical_path = graph.get_critical_path()  # Highest-reward trajectory
stats = graph.get_statistics()
```

#### 2. Offer Priority Queue
Intelligent ranking of historical offers by NPV score:
```python
from src.data_structures import OfferPriorityQueue

queue = OfferPriorityQueue(max_size=100)
queue.add_offer(round=1, price=95.0, days=30, 
                discount_factor=0.99, npv_score=0.45)

best_offer = queue.pop_best_offer()  # Retrieves highest NPV
top_k = queue.get_top_k(5)  # Top 5 offers
```

#### 3. Temporal Offer Buffer
Experience replay buffer with trajectory analytics:
```python
from src.data_structures import OfferBuffer

buffer = OfferBuffer(max_buffer_size=1000)
buffer.add_experience(offer, reward, state)

best_trajectories = buffer.get_best_trajectories(k=5)
statistics = buffer.compute_offer_statistics()
```

#### 4. State Similarity Matching
K-NN style state retrieval for few-shot learning:
```python
from src.data_structures import StateComparator

similar = StateComparator.find_similar_states(
    query_state, 
    all_states, 
    k=3,  # Return 3 most similar
    threshold=50.0  # Distance threshold
)
```

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
git clone https://github.com/SkandaGanesha1/ENV.git
cd openenv-sme-negotiator

# Install in development mode
pip install -e .

# Install with dev tools
pip install -e ".[dev]"
```

## Quick Start: Basic Usage

```python
from server.sme_environment import SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction

# Initialize environment
env = SMENegotiatorEnvironment()

# Reset for Easy task with deterministic seed
observation = env.reset(seed=42, difficulty="EASY")

print(f"Initial buyer offer: ₹{observation.buyer_price}/unit, {observation.buyer_days} days")
print(f"Your production cost: ₹{observation.cost_threshold}/unit")
print(f"Liquidity threshold: {observation.liquidity_threshold} days")

# Generate an action
action = NegotiationAction(
    action_type="PROPOSE",
    proposed_price=95.0,
    proposed_days=30,
    request_treds=False,
    justification="At ₹95/unit with 30-day terms, this contract covers my production costs (₹80/unit) with reasonable margin. Please consider this counter-offer."
)

# Step through environment
observation, reward, terminated, info = env.step(action)

if terminated:
    print(f"Episode ended. Score: {info.get('score', 0.0):.2f}")
else:
    print(f"Buyer counter-offer: ₹{observation.p_opp}/unit, {observation.d_opp} days")
```

## Running the Server

### Local Development
```bash
# Start FastAPI server on http://localhost:8000
python -m uvicorn server.app:app --reload --port 8000
```

### Docker Deployment
```bash
# Build container
docker build -f docker/Dockerfile -t openenv-sme-negotiator:latest .

# Run container
docker run -p 8000:8000 openenv-sme-negotiator:latest
```

### Hugging Face Spaces Deployment
```bash
# Install OpenEnv CLI (when available)
pip install openenv-cli

# Deploy to HF Space
openenv deploy --space-id your-username/sme-negotiator
```

## Advanced: Custom Agent

```python
from sme_negotiator_env.client import choose_action
from server.sme_environment import SMENegotiatorEnvironment

# Initialize environment
env = SMENegotiatorEnvironment()

# Run episode
observation = env.reset(seed=123, difficulty="MEDIUM")
total_reward = 0.0
round_number = 0
agent_days = max(observation.liquidity_threshold, observation.buyer_days // 2)

while not observation.done:
  action, agent_days = choose_action(observation, round_number, agent_days)

    # Environment steps
  observation = env.step(action)
  total_reward += observation.reward

  if observation.done:
    print(f"Final score: {observation.reward:.3f}")
        break

  round_number += 1
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

## Project Structure

```
openenv-sme-negotiator/
├── server/
│   ├── app.py                       # OpenEnv server entrypoint
│   ├── sme_environment.py           # Core MDP environment
│   └── Dockerfile                   # Container image
├── sme_negotiator_env/
│   ├── client.py                    # Typed client and heuristic policy
│   └── models.py                    # OpenEnv models
├── tests/
│   └── test_environment.py          # Unit tests
├── pyproject.toml                   # Project metadata & dependencies
└── README.md                        # This file
```

## Mathematical Formulation: HardTask TReDS Solution

### Problem Setup
- Buyer demands: ₹95/unit, 120 days
- SME cost: ₹70/unit
- SME survives only: 30 days
- Standard negotiation fails (SME bankruptcy)

### TReDS-Enabled Solution
1. **Agent proposes**: ₹90/unit, 120 days, TReDS=True
2. **Justification**: "By processing via TReDS, your treasury pays the bank in 120 days (satisfying your policy), but I receive discounted funds immediately, solving my liquidity constraint. The ₹5/unit discount (5.3%) covers your platform friction."
3. **Financial Result**:
   - SME Profit: (90-70) × Volume = ₹20/unit value
   - NPV with TReDS: Immediate cash → No delay penalty
   - Score: ~0.65 (significant success for Hard task)

### Why This Is Hard
- Requires understanding TReDS mechanics
- Needs multi-step lookahead (propose → counter → accept)
- Demands natural language persuasion
- Must balance three competing dimensions (price, days, TReDS mode)

## References

1. **MSMED Act**: Section 43B(h), Income Tax Act 1961
2. **TReDS Platform**: RBI Trade Receivables Discounting System
3. **OpenEnv Spec**: https://huggingface.co/docs/openenv
4. **Razorpay "Fix My Itch"**: https://www.razorpay.com/reports/fix-my-itch
5. **Nemotron 3 Super**: NVIDIA's Hybrid Mamba-Transformer model

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{openenv_sme_negotiator_2024,
  title={OpenEnv SME Negotiator: An RL Environment for B2B Contract Negotiation},
  author={Omkarchaithanya},
  year={2024},
  url={https://github.com/SkandaGanesha1/ENV}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Running Baseline Inference with OpenAI's GPT

The `inference.py` script provides a complete baseline agent using OpenAI's LLM (GPT-4, GPT-4o, etc.) to negotiate in the SME environment.

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export API_BASE_URL="http://localhost:8000"  # or your server URL
   export MODEL_NAME="gpt-4o"                    # gpt-4, gpt-4o, etc.
   ```

3. **Start the server** (in another terminal):
   ```bash
    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Running Inference

Execute the baseline:

```bash
python inference.py
```

This will:
- Connect to the SME negotiation environment server
- Run 3 episodes each on Easy, Medium, and Hard tasks
- Use GPT-4o (or your configured model) to generate negotiation actions
- Output step-by-step decisions and final scores

### Example Output

```
============================================================
  OPENENV SME NEGOTIATION - BASELINE INFERENCE
============================================================

✓ Connected to server at http://localhost:8000

============================================================
Episode: EASY, Seed: 1000
============================================================

Step 1:
  Action: PROPOSE
    Price: ₹98
    Days: 30
    TReDS: False
  Reward: 0.0000

Step 2:
  Action: ACCEPT
  Reward: 0.87

...

============================================================
  BASELINE INFERENCE SUMMARY
============================================================

EASY     | Avg: 0.8712 | Episodes: 3 | Scores: ['0.8712', '0.8650', '0.8875']
MEDIUM   | Avg: 0.6245 | Episodes: 3 | Scores: ['0.6100', '0.6345', '0.6390']
HARD     | Avg: 0.0850 | Episodes: 3 | Scores: ['0.0000', '0.1200', '0.1450']

Overall Average Score: 0.5269
Total Episodes: 9

============================================================
✓ Baseline inference complete
============================================================
```

### How It Works

1. **System Prompt**: Instructs GPT to act as an SME business manager
2. **State Representation**: Converts environment state to natural language
3. **LLM Action Generation**: Asks GPT to generate JSON negotiation actions
4. **Environment Interaction**: Sends actions to server, receives rewards
5. **Episode Loop**: Continues until termination or max steps

### Key Features

- **Task Aware**: Different prompts for Easy/Medium/Hard tasks
- **Conversation History**: Includes prior rounds for contextual reasoning
- **TReDS Awareness**: Explicitly teaches the model about TReDS financing
- **Fallback Handling**: Safe JSON parsing with fallback actions
- **Async Ready**: Can be extended with parallel episodes

### Expected Performance

Baseline performance with GPT-4o:
- **Easy**: 0.85-0.95 (straightforward price negotiation)
- **Medium**: 0.50-0.70 (requires multi-dimensional reasoning)
- **Hard**: 0.00-0.20 (very few models find TReDS solution)

### Customization

To run different models or configurations:

```bash
# Use GPT-4 Turbo
MODEL_NAME="gpt-4-turbo" python inference.py

# Use local LLM via OpenAI-compatible API
API_BASE_URL="http://localhost:8000/v1" python inference.py

# Run more episodes per task (modify agent.py)
```

### Performance Analysis

After running inference, you can analyze:

1. **Score distribution**: Are scores consistent or variable?
2. **Decision patterns**: What strategy did GPT discover?
3. **Failure modes**: Where did Hard task negotiations break down?
4. **Financial validity**: Do proposed prices respect cost constraints?

## Contact & Support

- **Issues**: GitHub Issues tracker
- **Discussions**: GitHub Discussions
- **Repository**: https://github.com/SkandaGanesha1/ENV
