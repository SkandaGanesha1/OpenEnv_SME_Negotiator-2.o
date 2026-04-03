# OpenEnv SME Negotiator - Troubleshooting Guide

Solutions to common problems during hackathon.

---

## Installation Issues

### Problem: `pip install -e .` Fails

**Error**: `error: could not create '/usr/local/lib/python3.10/dist-packages/': Permission denied`

**Solution**:
```bash
# Use --user flag or install in virtual environment
pip install --user -e .

# Better: Use virtual environment
python3 -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
pip install -e .
```

---

### Problem: ModuleNotFoundError After Installation

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Reinstall in development mode
pip install -e .

# Verify installation
python -c "from server.sme_environment import SMENegotiatorEnvironment; print('✓ OK')"
```

---

### Problem: Requirements Installation Fails

**Error**: `ERROR: Could not find a version that satisfies the requirement torch==2.3.0`

**Solution**:
```bash
# Update pip first
pip install --upgrade pip

# Install torch separately (may need special flags for GPU)
pip install torch==2.3.0

# Then install rest of requirements
pip install -r requirements.txt
```

---

## Server Issues

### Problem: Port 8000 Already in Use

**Error**: `OSError: [Errno 48] Address already in use` or similar

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000                          # Linux/Mac
netstat -ano | findstr :8000           # Windows

# Kill the process
kill -9 <PID>                          # Linux/Mac
taskkill /PID <PID> /F                 # Windows

# Or use different port
make server -- --port 8001
# Then set: export API_BASE_URL="http://localhost:8001"
```

---

### Problem: Server Crashes on Startup

**Error**: `Traceback (most recent call last): ... FastAPI error`

**Solution**:
```bash
# Check if dependencies installed
python -c "from fastapi import FastAPI; print('✓ OK')"

# Reinstall FastAPI
pip install fastapi==0.104.1 uvicorn==0.24.0

# Run with verbose logging
python -m uvicorn server.app:app --log-level debug
```

---

### Problem: Cannot Connect to Server

**Error**: `Cannot connect to http://localhost:8000`

**Solution**:
1. **Verify server is running**:
   ```bash
   # In terminal where you ran 'make server'
   # You should see: "Uvicorn running on http://0.0.0.0:8000"
   ```

2. **Test connectivity**:
   ```bash
   curl http://localhost:8000/health
   
   # If fails, try localhost by hostname
   curl http://127.0.0.1:8000/health
   ```

3. **Check firewall** (if remote server):
   ```bash
   # Ensure port 8000 is open
   sudo ufw allow 8000  # Linux
   ```

---

## OpenAI API Issues

### Problem: "OPENAI_API_KEY not set"

**Error**: `ERROR: OPENAI_API_KEY environment variable not set`

**Solution**:

On **Linux/Mac**:
```bash
# Set for current session
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $OPENAI_API_KEY

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo "export OPENAI_API_KEY='sk-...'" >> ~/.bashrc
source ~/.bashrc
```

On **Windows (PowerShell)**:
```powershell
# Set for current session
$env:OPENAI_API_KEY="sk-..."

# Verify
Write-Host $env:OPENAI_API_KEY

# Make permanent
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")
```

---

### Problem: "Invalid API Key"

**Error**: `AuthenticationError: Incorrect API key provided`

**Solution**:
1. Verify API key at https://platform.openai.com/account/api-keys
2. Copy it exactly (no extra spaces)
3. If still fails, regenerate new key
4. Check key is not expired/revoked

---

### Problem: "Rate Limit Exceeded"

**Error**: `RateLimitError: Rate limit exceeded` or `429 Too Many Requests`

**Solution**:
```python
# Option 1: Wait and retry
import time
time.sleep(60)  # Wait 60 seconds

# Option 2: Use exponential backoff
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai():
    # Your OpenAI call
    pass

# Option 3: Batch requests with delays
for episode in range(num_episodes):
    run_episode()
    time.sleep(1)  # Add delay between episodes
```

---

### Problem: "Insufficient Quota"

**Error**: `InsufficientQuotaError: You exceeded your current quota`

**Solution**:
1. Check usage: https://platform.openai.com/account/usage/overview
2. Check billing: https://platform.openai.com/account/billing/overview
3. Add payment method if needed
4. Request quota increase if you're a heavy user

---

## Inference Runtime Issues

### Problem: inference.py Hangs or Freezes

**Error**: Script runs but doesn't output anything for 5+ minutes

**Solution**:
```bash
# Kill the hung process
Ctrl+C

# Run with timeout
timeout 30 python inference.py  # Cancel if > 30 sec

# Check for infinite loops - reduce num_episodes
# Edit inference.py:
# Change: for episode in range(3):
# To:     for episode in range(1):

# Run again
python inference.py
```

---

### Problem: JSON Parse Error in LLM Response

**Error**: `Failed to parse LLM response: ...`

**Solution**:
This is expected fallback behavior. The script will:
1. Print the unparsed response
2. Use a default safe action
3. Continue running

To reduce this:
```python
# In inference.py, update system prompt to be clearer:
"Output ONLY valid JSON with no markdown formatting"

# Use JSON mode if available:
response = client.chat.completions.create(
    ...,
    response_format={"type": "json_object"}  # GPT-4o feature
)
```

---

### Problem: Very Low Scores on All Tasks

**Scores**: All scores near 0.0 even on Easy

**Solution**:
1. **Check financial validity**:
   ```python
   # Add to inference.py after each step:
   if obs['p_proposed'] < obs['c_sme'] * 0.95:
       print(f"⚠️ Price ₹{obs['p_proposed']} is below cost!")
   ```

2. **Ensure agent understands constraints**:
   ```python
   system_prompt += """
   CRITICAL CONSTRAINTS:
   - Never propose price below ₹70/unit (your cost)
   - Never accept days > 45 without TReDS (regulatory)
   - TReDS only works if days > 45
   """
   ```

3. **Verify environment is working**:
   ```bash
   pytest tests/test_environment.py::test_scoring -v
   ```

---

### Problem: Server Returns 500 Error

**Error**: `HTTPError: 500 Internal Server Error`

**Solution**:
1. Check server logs (where you ran `make server`)
2. Look for stack trace with actual error
3. Common causes:
   - Invalid action format (wrong field names)
   - Cost > Price
   - Invalid task_id

**Fix example**:
```python
# ✗ Wrong format
action = {"action": "PROPOSE", ...}  # Should be "action_type"

# ✓ Correct format
action = {"action_type": "PROPOSE", ...}
```

---

## Performance Issues

### Problem: Inference Very Slow

**Timing**: Each episode takes 30+ seconds

**Solution**:
1. **Check LLM latency**:
   ```bash
   # Time a single API call
   time python -c "
   from openai import OpenAI
   client = OpenAI()
   response = client.chat.completions.create(
       model='gpt-4o',
       messages=[{'role': 'user', 'content': 'Hi'}]
   )
   "
   # If > 10 sec, LLM is slow
   ```

2. **Reduce verbosity**:
   ```python
   # In inference.py, comment out debug prints
   # print(f"Step {step_count + 1}:")
   # print(f"  Action: {action['action_type']}")
   ```

3. **Use faster model**:
   ```bash
   export MODEL_NAME="gpt-3.5-turbo"  # Faster, cheaper
   python inference.py
   ```

---

### Problem: High API Costs

**Issue**: Using $10+ in OpenAI credits per run

**Solution**:
```bash
# Use cheaper/faster model
export MODEL_NAME="gpt-3.5-turbo"
# Cost: ~$0.50 vs $5+ for GPT-4

# Reduce episodes
# In inference.py, change: for episode in range(3):
# To: for episode in range(1):

# Implement caching
# Cache LLM responses for identical states
```

---

## Testing Issues

### Problem: pytest Fails or Tests Don't Run

**Error**: `collecting error` or `ModuleNotFoundError`

**Solution**:
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Install package in development mode
pip install -e .

# Run tests with verbose output
pytest tests/ -v -s

# Run specific test file
pytest tests/test_environment.py -v
```

---

### Problem: Test Timeout (takes 1+ hour)

**Solution**:
```bash
# Run only fast tests
pytest tests/test_environment.py::test_reset -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Run with timeout
pytest tests/ --timeout=60  # Fail if test > 60 sec
```

---

## Code Issues

### Problem: Type Errors in IDE

**Error**: Red squiggles on imports like `from gymnasium import Env`

**Solution**:
```bash
# Install type stubs
pip install types-All

# Or disable type checking in IDE settings
# (see IDE documentation)
```

---

### Problem: ImportError for gymnasium vs gym

**Error**: `ModuleNotFoundError: No module named 'gym'` or `gymnasium`

**Solution**:
```bash
# Install latest gymnasium (replaces gym)
pip install gymnasium==0.29.1

# In code, use:
import gymnasium as gym  # Not just 'import gym'
```

---

## Data & Reproducibility

### Problem: Different Scores with Same Seed

**Issue**: Running same code with seed=42 gives different scores

**Possible causes**:
1. **Different Python version** → Different float precision
2. **Different LLM responses** → GPT is non-deterministic even with temp=0
3. **Environment version changed** → Update dependencies

**Solution**:
```bash
# Lock all dependencies
pip freeze > requirements_lock.txt

# In future, install from lock file
pip install -r requirements_lock.txt

# For LLM non-determinism, accept small variance
# Use seed for reproducible *trajectories*, not scores
```

---

### Problem: "Seed Not Respected"

**Issue**: Different scores between runs even with seed

**Solution**:
```python
# Ensure seed is set BEFORE any random operations
env.reset(task_id="easy", seed=42)  # Must pass seed

# Verify environment respects seed
for seed in [1, 1]:
    obs = env.reset(seed=seed)
    # Should get identical obs both times
```

---

## Debugging Help

### Enable Debug Output

```bash
# Set environment variables
export DEBUG=1
export LOG_LEVEL=DEBUG

# Run with verbose output
python -u inference.py  # -u forces unbuffered output

# Check what's happening step-by-step
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from inference import SMENegotiationAgent
agent = SMENegotiationAgent()
result = agent.run_episode('easy', seed=42)
print(f'Result: {result}')
"
```

---

### Get Help

If you're stuck:

1. **Check existing documentation**:
   - README.md
   - SETUP.md
   - EVALUATION.md
   - Index.md (this file)

2. **Review the code**:
   - inference.py (well-commented)
   - server/sme_environment.py
   - sme_negotiator_env/models.py

3. **Test incrementally**:
   ```python
   # Instead of running full inference
   # Test each piece separately
   agent = SMENegotiationAgent()
   obs = agent.reset_episode("easy")
   action = agent.get_llm_action(system_prompt, user_prompt)
   result = agent.step_episode(action)
   ```

4. **Ask for help**:
   - GitHub Issues
   - Slack/Discord channels
   - Forum discussions

---

## Common Wisdom

| Problem | Quick Fix |
|---------|-----------|
| Everything broken | `pip install -e .` then `make test` |
| API fails | `echo $OPENAI_API_KEY` (verify set) |
| Server won't start | Kill process on port 8000 |
| Scores low | Check `price > cost` and `days < 120` |
| JSON parse error | Safe fallback used, not a blocker |
| Tests fail | Run `pytest tests/test_env.py -v` for more detail |

---

**Still stuck? Enable debug logging and check the error messages carefully!** 🔍
