# OpenEnv SME Negotiator - Hackathon Setup Guide

Quick start guide for Scaler hackathon participants.

## Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version
- **OpenAI API Key**: For baseline inference (get at https://platform.openai.com/account/api-keys)
- **Git**: For cloning repository

## 1. Clone & Navigate

```bash
git clone https://github.com/scaler/openenv-sme-negotiator.git
cd openenv-sme-negotiator
```

## 2. Install Dependencies

```bash
# Install project and all dependencies
pip install -e .

# Or just core requirements
pip install -r requirements.txt
```

## 3. Environment Setup

Create a `.env` file in the project root:

```bash
cat > .env << EOF
OPENAI_API_KEY=your-api-key-here
API_BASE_URL=http://localhost:8000
MODEL_NAME=gpt-4o
HF_TOKEN=your-hf-token-optional
EOF
```

Then load it:

```bash
source .env  # On Linux/Mac
# or
set -a; source .env; set +a  # Bash with PowerShell
```

## 4. Start the Server

In one terminal:

```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 5. Run Baseline Inference

In another terminal:

```bash
python inference.py
```

This will run 9 episodes (3 per task: Easy, Medium, Hard) and output scores.

## 6. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest --cov=src/ tests/
```

## 7. Verify Installation

```bash
python run_diagnostics.py
```

This will check:
- ✓ Python version
- ✓ Core dependencies installed
- ✓ Server connectivity
- ✓ Environment configuration
- ✓ LLM API access

---

## Project Structure

```
├── src/
│   ├── env/
│   │   └── sme_negotiation.py       # Core environment MDP
│   ├── agents/
│   │   └── llm_agent.py             # Baseline LLM agent
│   ├── utils/
│   │   ├── models.py                # Pydantic schemas
│   │   ├── grader.py                # Deterministic grader
│   │   └── constants.py             # Task parameters
│   └── server.py                    # FastAPI WebSocket server
├── eval/
│   └── evaluation.py                # Evaluation framework
├── training/
│   └── dpo_training.py              # Fine-tuning scripts
├── tests/
│   └── test_environment.py          # Unit tests
├── inference.py                     # Baseline inference script
└── README.md                        # Full documentation
```

---

## Key Concepts

### Task Stratification

| Task | Difficulty | Goal | Expected Score |
|------|-----------|------|-----------------|
| **Easy** | Baseline | Maximize price (fixed 30-day terms) | 0.85-0.95 |
| **Medium** | Intermediate | Trade-off price vs payment days | 0.50-0.75 |
| **Hard** | Expert | Use TReDS to overcome impossible constraints | 0.00-0.20 |

### Score Calculation

$$Score = \max(0, \min(1, \frac{NPV - U_{min}}{U_{max} - U_{min}}))$$

Where NPV factors in:
- **Profit margin**: price - cost
- **Time value**: discounting via (1/(1+r)^(days/365))
- **Liquidity penalty**: exponential penalty if days > 45 (without TReDS)

### Environment Workflow

1. **Reset**: Initialize environment with task and seed
2. **Step**: Agent proposes PROPOSE/ACCEPT/REJECT action
3. **State Update**: Environment updates based on negotiation dynamics
4. **Reward**: Get score (0 intermediate, normalized NPV at terminal)
5. **Done**: Episode terminates when deal accepted or max rounds reached

---

## Common Issues & Troubleshooting

### Issue: "OPENAI_API_KEY not set"

**Fix**:
```bash
export OPENAI_API_KEY="sk-..."  # Linux/Mac
set OPENAI_API_KEY=sk-...        # Windows CMD
$env:OPENAI_API_KEY="sk-..."     # PowerShell
```

### Issue: "Cannot connect to server"

**Fix**: Make sure server is running (step 4), then:
```bash
# Test connectivity
curl http://localhost:8000/health

# If fails, check port 8000 is not in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows
```

### Issue: "ModuleNotFoundError: No module named 'src'"

**Fix**: Install package in development mode:
```bash
pip install -e .
```

### Issue: "RateLimitError from OpenAI"

**Fix**: Your API quota is exhausted. Check:
- Usage at https://platform.openai.com/account/usage/overview
- Billing is active
- API key is correct

---

## Performance Optimization

### Running Distributed Inference

For hackathon, you can speed up evaluation:

```python
# inference.py (pseudocode)
import asyncio

async def run_all_tasks():
    agent = SMENegotiationAgent()
    tasks = []
    
    # Queue all episodes in parallel
    for task_id in ["easy", "medium", "hard"]:
        for seed in range(10):
            tasks.append(agent.run_episode(task_id, seed))
    
    # Run concurrently
    results = await asyncio.gather(*tasks)
    return results

# Run
asyncio.run(run_all_tasks())
```

### Using Caching

Responses can be cached to avoid redundant LLM calls:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_llm_action(state_hash, prompt_hash):
    return get_llm_action(state, prompt)
```

---

## Next Steps for the Hackathon

### Day 1: Familiarization
- [ ] Run baseline inference successfully
- [ ] Review task specifications in README
- [ ] Understand score calculation (see grader.py)

### Day 2: Analysis & Strategy
- [ ] Analyze baseline performance patterns
- [ ] Identify failure modes on Hard task
- [ ] Brainstorm improved agent strategies

### Day 3: Implementation
- [ ] Implement custom agent (e.g., fine-tuned LLM, RL algorithm)
- [ ] Test on all three tasks
- [ ] Optimize for highest overall score

### Day 4: Submission
- [ ] Create evaluation script
- [ ] Document your approach
- [ ] Submit final results

---

## Resources

- **OpenAI Models**: https://platform.openai.com/docs/models
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **TReDS Platform**: https://www.tredsindia.com/

## Support

- **Questions**: Check GitHub Issues
- **Documentation**: See README.md for full details
- **Community**: Join Scaler forums

---

**Good luck with your hackathon! 🚀**
