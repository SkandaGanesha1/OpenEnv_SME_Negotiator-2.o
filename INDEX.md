# OpenEnv SME Negotiator - Hackathon Resources Index

Complete reference guide to all documentation, code, and tools.

---

## 🚀 Quick Start (Choose One Path)

### Fastest path

```bash
bash quickstart.sh              # if present: quick setup
uv sync
make diagnostic                 # import check + tests
make baseline                   # requires LLM credentials in .env
```

### Manual path

```bash
uv sync
# copy .env.example -> .env and set HF_TOKEN / API_BASE_URL / MODEL_NAME
uv run server                   # terminal 1, port 7860
uv run python inference.py      # terminal 2 (or OPENENV_IN_PROCESS=1)
```

### Docker

```bash
make docker-build
make docker-run                 # http://127.0.0.1:7860
```

---

## 📚 Documentation Files

### Essential Reading (Start Here)

| File | Purpose | Read Time |
|------|---------|-----------|
| [README.md](README.md) | Full project overview + baseline section | 10 min |
| [SETUP.md](SETUP.md) | Step-by-step installation guide | 5 min |
| [EVALUATION.md](EVALUATION.md) | How to measure & optimize performance | 15 min |

### Reference Documents

| File | Purpose | Use When |
|------|---------|----------|
| [pyproject.toml](pyproject.toml) | Project metadata & dependencies | Building/publishing |
| openenv.yaml | Environment config | Customizing task parameters |
| Makefile | Development commands | Running tasks |

### Hackathon Spec Alignment

Reviewers can check the evaluation contract quickly in one place:

- [README.md](README.md) documents the `[START]` / `[STEP]` / `[END]` output format used by `inference.py`.
- [README.md](README.md) also lists the key environment variables: `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME`, `OPENENV_BASE_URL`, `OPENENV_IN_PROCESS`, and `INFERENCE_HARD_TWO_STEP`.
- [README.md](README.md) includes the direct `python inference.py` command and the `uv run python inference.py` variant.
- [README.md](README.md) spells out the TReDS simulation effect and the NPV-based hard grader.

---

## 💻 Core Code Files

### Your Main Entry Points

```
Part 1: Run Baseline
├── inference.py              ← LLM agent for negotiation
├── openenv.yaml              ← OpenEnv manifest
├── pyproject.toml            ← All dependencies
└── .env                      ← API keys (create this)

Part 2: Understanding the Environment
├── server/environment.py      ← Core MDP (`SMENegotiatorEnvironment`)
├── server/app.py               ← OpenEnv `create_app` entrypoint
├── server/sme_environment.py  ← Re-export of environment
└── sme_negotiator_env/models.py

Part 3: Testing
└── tests/test_environment.py   ← Unit tests
```

### Running Different Components

```bash
# Run baseline inference (uses OpenAI API)
python inference.py

# Start the game server (port 7860)
make server
# or: uv run server

# Run unit tests
make test

# Quick health: imports + pytest
make diagnostic
```

---

## 🎯 Task Specifications

### All Three Tasks Explained

#### Easy Task: Price Optimization
- **Focus**: Payment days vs liquidity cap (see `task_config.py` / `graders.py`)
- **Opening buyer offer**: 100 @ 90d; liquidity threshold 60d
- **Goal**: Agree terms ≤ 60d for full credit
- **Python Example**:
  ```python
    env = SMENegotiatorEnvironment()
    obs = env.reset(seed=42, difficulty="easy")
  ```

#### Medium Task: Pareto Frontier
- **Focus**: Multi-dimensional trade-off
- **Timeline**: Negotiable, 45-day regulatory compliance
- **Baseline Score**: 0.62
- **Goal**: Optimal price+days combination
- **Financial Formula**: Higher days → can accept lower price

#### Hard Task: Dynamic discounting (NPV)
- **Focus**: Agree **dynamic discounting**; terminal score from NPV vs status quo (`graders.py`)
- **Note**: `use_treds` affects simulation but hard grading is **not** TReDS-only
- **Key Insight**: Set `propose_dynamic_discounting` and a viable `dynamic_discount_annual_rate`

---

## 📊 Performance Benchmarks

### Baseline Results (OpenAI GPT-4o)

```
Easy   Task: 0.87 ± 0.02  (excellent)
Medium Task: 0.62 ± 0.07  (good)
Hard   Task: 0.08 ± 0.11  (very hard)
─────────────────────────────
Overall:  0.52             (median across all)
```

### Your Target Improvements

| Task | Beat Baseline By | Suggested Targets |
|------|------------------|-------------------|
| Easy | +0.05 | 0.92+ (margin margin) |
| Medium | +0.15 | 0.77+ (strong reasoning) |
| Hard | +0.30 | 0.38+ (TReDS insight) |

---

## 🛠️ Useful Commands

### Development

```bash
make install          # Install package
make install-dev      # Install with test tools
make test             # Run tests with output
make test-cov         # Run with coverage report
make lint             # Check code quality
make format           # Auto-format code
make typecheck        # Type checking
make clean            # Remove build artifacts
```

### Running

```bash
make server           # Start FastAPI server
make baseline         # Run baseline inference
make diagnostic       # Verify setup
make examples         # Run example scripts
make docs             # View documentation
```

### Docker

```bash
make docker-build     # Build container image
make docker-run       # Run in container
make docker-push      # Push to registry
```

---

## 🤔 Common Questions

### "How do I run the baseline?"
→ See [SETUP.md](SETUP.md) Step 5, or just run `make baseline`

### "What does each score mean?"
→ See [EVALUATION.md](EVALUATION.md) - Score Range section

### "How can I improve scores?"
→ See [EVALUATION.md](EVALUATION.md) - Optimization Strategies section

### "Why is Hard task so hard?"
→ See [EVALUATION.md](EVALUATION.md) - Hard Task Analysis section

### "Can I use a different LLM?"
→ Yes! Modify `inference.py` to use your LLM (Llama, Claude, etc.)

### "How is reproducibility guaranteed?"
→ Fixed seeds in environment + deterministic grader = same score every time

---

## 🔧 Customization Guide

### Using a Different LLM

```python
# In inference.py, replace OpenAI client with:

# Option 1: Local Llama via Ollama
from ollama import Ollama
client = Ollama(model="llama2")

# Option 2: Anthropic Claude
from anthropic import Anthropic
client = Anthropic()

# Option 3: HuggingFace Transformers
from transformers import pipeline
client = pipeline("text-generation", model="meta-llama/Llama-2-7b")
```

### Modifying Agent Strategy

```python
# In SMENegotiationAgent.build_system_prompt():
if task_id == "hard":
    return base + """
    ADDED INSTRUCTION:
    - Always check if TReDS is applicable (days > 45)
    - TReDS costs ~2% but solves liquidity instantly
    - Compare NPV with vs without TReDS
    """
```

### Running Custom Episodes

```python
# Create your own run_custom_inference.py
from server.sme_environment import SMENegotiatorEnvironment

env = SMENegotiatorEnvironment()
obs = env.reset(seed=42, difficulty="EASY")

for step in range(12):
    # Your custom logic here
    action = {
        "action_type": "PROPOSE",
        "proposed_price": 98,
        "proposed_days": 30,
        "request_treds": False,
        "justification": "Testing"
    }
    
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"Final score: {reward:.4f}")
        break
```

---

## 📈 Improvement Ideas

### Level 1: Prompt Engineering
- [ ] Add few-shot examples to system prompt
- [ ] Use chain-of-thought prompting
- [ ] Add financial constraints checklist
- **Expected boost**: +0.05

### Level 2: Algorithm Enhancement
- [ ] Multi-turn conversation planning
- [ ] State-value function for decision making
- [ ] Ensemble voting across multiple LLM calls
- **Expected boost**: +0.10

### Level 3: Model Optimization
- [ ] Fine-tune on successful trajectories
- [ ] RL training on top of environment
- [ ] Mixture of experts with task routing
- **Expected boost**: +0.15+

---

## 🏆 Submission Requirements

### Minimum Viable Submission
- Custom agent code (`.py` file)
- Evaluation results (scores on all tasks)
- Brief explanation (how your agent works)
- Reproducible with fixed seeds

### Excellence Submission
- All MVs + performance improvement analysis
- Optimization techniques used (e.g., prompting strategy)
- Failure case analysis (why Hard task is hard?)
- Comparison against baseline
- Code quality (tests, documentation, type hints)

---

## 📞 Support & Resources

### Internal Resources
- GitHub Issues: Ask questions
- GitHub Discussions: Share ideas
- Pull Requests: Contribute improvements

### External Resources  
- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Financial NPV Calculator](https://www.investopedia.com/terms/n/npv.asp)
- [TReDS Platform Info](https://www.tredsindia.com/)

---

## ✅ Final Pre-Submission Checklist

- [ ] Can run `make baseline` successfully
- [ ] Can run `make test` successfully
- [ ] Easy task score ≥ 0.80
- [ ] Medium task score ≥ 0.50
- [ ] Hard task score ≥ 0.00
- [ ] Code is formatted (`make format`)
- [ ] Code passes linting (`make lint`)
- [ ] Documentation is complete
- [ ] Results are reproducible with seed=42
- [ ] No external dependencies missing

---

## 🎓 Learning Path

**Day 1**: Learn
- Read [README.md](README.md) completely
- Run `make baseline` successfully
- Understand task structure

**Day 2**: Understand  
- Read [EVALUATION.md](EVALUATION.md)
- Analyze baseline outputs
- Study failure cases

**Day 3**: Improve
- Implement Improvement (from "Improvement Ideas")
- Measure performance
- Compare vs baseline

**Day 4**: Polish
- Documentation
- Code quality
- Final testing

**Day 5**: Submit
- Verify checklist
- Generate final results
- Submit solution

---

**Happy hacking! 🚀**

*Last updated: 2024 | OpenEnv SME Negotiator Hackathon*
