# OpenEnv SME Negotiator - Hackathon Setup Guide

Quick start guide for project participants.

## Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version
- **OpenAI API Key**: For baseline inference (get at https://platform.openai.com/account/api-keys)
- **Git**: For cloning repository

## 1. Clone & navigate

```bash
git clone <YOUR_REPOSITORY_URL>
cd openenv-sme-negotiator
```

## 2. Install dependencies

```bash
uv sync
# or: pip install -e .

# Dev tools (pytest)
uv sync --extra dev
```

## 3. Environment setup

Copy `.env.example` to `.env` and set:

- **`HF_TOKEN`** — for Hugging Face Inference router (default `API_BASE_URL` in `inference.py`)
- **`API_BASE_URL`** / **`MODEL_NAME`** — any OpenAI-compatible LLM
- **`OPENENV_BASE_URL=http://127.0.0.1:7860`** — negotiation server (or use **`OPENENV_IN_PROCESS=1`** to skip a separate server)

## 4. Start the server (optional)

If you are **not** using `OPENENV_IN_PROCESS=1`:

```bash
uv run server
# or: python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

You should see Uvicorn on **`http://0.0.0.0:7860`** (matches `openenv.yaml` and Docker).

## 5. Run baseline inference

```bash
uv run python inference.py
```

## 6. Run tests

```bash
uv run pytest tests/ -v
```

## 7. Verify installation

```bash
make diagnostic
# or: python -c "from server.app import app; print(app.title)" && uv run pytest tests/ -q
```

---

## Project Structure

```
├── server/
│   ├── app.py                       # OpenEnv server entrypoint
│   ├── environment.py               # Core environment MDP
│   └── sme_environment.py           # Re-export
├── sme_negotiator_env/
│   ├── client.py                    # Typed client and heuristic policy
│   └── models.py                    # OpenEnv models
├── tests/
│   └── test_environment.py          # Unit tests
├── inference.py                     # Baseline inference script
├── openenv.yaml                     # OpenEnv manifest
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

### Score calculation

- **Step**: Partial shaping reward + terminal grader output in `sme_negotiator_env/graders.py`.
- **Details**: See README “Deterministic graders” and `server/environment.py`.

### Environment workflow

1. **Reset**: Task + seed (`task_name` or `difficulty`)
2. **Step**: `action_type` propose / accept / reject with numeric fields (`models.py`)
3. **Reward**: Partial + terminal (see above)
4. **Done**: Deal, reject, or max rounds

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

**Fix**: Make sure the server is running (step 4), then:

```bash
curl http://127.0.0.1:7860/health
```

If it fails, check that port **7860** is free (`netstat -ano | findstr :7860` on Windows).

### Issue: "ModuleNotFoundError"

**Fix**: Install the package in editable mode: `uv sync` or `pip install -e .`

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
- **Community**: Join the project forums

---

**Good luck with your hackathon! 🚀**
