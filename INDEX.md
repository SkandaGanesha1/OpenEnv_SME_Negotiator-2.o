# OpenEnv SME Negotiator - Hackathon Resources Index

Complete reference guide to all documentation, code, and tools.

---

## 🚀 Quick Start (Choose One Path)

### Fastest Path (3 minutes)
```bash
bash quickstart.sh              # Auto-setup everything
make baseline                   # Run baseline inference
```

### Manual Path (5 minutes)
```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
make server &
make baseline
```

### Docker Path (2 minutes)
```bash
make docker-build
make docker-run
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
├── server/sme_environment.py  ← Core MDP environment
├── server/app.py              ← OpenEnv server entrypoint
└── sme_negotiator_env/models.py ← Score and state models

Part 3: Testing
├── tests/test_environment.py   ← Unit tests
└── test_comprehensive.py       ← Full integration tests
```

### Running Different Components

```bash
# Run baseline inference (uses OpenAI API)
python inference.py

# Start the game server
make server
# or
python -m uvicorn server.app:app --reload

# Run unit tests
make test

# Run diagnostics
make diagnostic
# or
python run_diagnostics.py

# Format code
make format

# Type check
make typecheck
```

---

## 🎯 Task Specifications

### All Three Tasks Explained

#### Easy Task: Price Optimization
- **Focus**: Single-issue negotiation
- **Timeline**: Fixed 30 days
- **Baseline Score**: 0.87
- **Goal**: Maximize price
- **Python Example**:
  ```python
    env = SMENegotiatorEnvironment()
    obs = env.reset(seed=42, difficulty="EASY")
    # obs.buyer_price = ₹96, obs.buyer_days = 30, obs.cost_threshold = ₹70
  ```

#### Medium Task: Pareto Frontier
- **Focus**: Multi-dimensional trade-off
- **Timeline**: Negotiable, 45-day regulatory compliance
- **Baseline Score**: 0.62
- **Goal**: Optimal price+days combination
- **Financial Formula**: Higher days → can accept lower price

#### Hard Task: TReDS Innovation
- **Focus**: Complex financial engineering
- **Timeline**: 120-day buyer constraint, 30-day SME survival
- **Baseline Score**: 0.08
- **Goal**: Recognize TReDS as solution
- **Key Insight**: Propose lower price with TReDS factoring

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
