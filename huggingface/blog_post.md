---
title: "Train an SME Liquidity Negotiator with GRPO on Colab"
thumbnail: /blog/assets/openenv-sme-negotiator-grpo/thumbnail.png
authors:
- user: Omkarchaithanya
tags:
- grpo
- reinforcement-learning
- lora
- openenv
- agents
- finance
---

# Train an SME Liquidity Negotiator with GRPO on Colab

![OpenEnv SME Negotiator thumbnail](assets/openenv-sme-negotiator-grpo/thumbnail.png)

Train a tool-using language-model agent to negotiate payment terms for a small supplier under working-capital stress. The whole demo runs from a Colab notebook, trains a LoRA adapter with GRPO, and publishes the result to the Hugging Face Hub.

The setup is intentionally practical:

```text
SME liquidity environment -> Qwen2.5 policy -> GRPO -> LoRA adapter -> Hugging Face model repo
```

The agent plays the SME. It sees buyer price, buyer payment days, supplier payment deadlines, working-capital gap, financing cost, buyer power, and TReDS-style invoice discounting options. It then decides whether to propose terms, accept, reject, use a financing tool, or advance the macro cashflow period.

This is not a chatbot demo. It is a reinforcement-learning loop around a simulated business negotiation where the model has to protect liquidity, avoid default, and preserve NPV.

---

## Get The Code

The project is on GitHub:

```bash
git clone https://github.com/SkandaGanesha1/OpenEnv_SME_Negotiator-2.o.git
cd OpenEnv_SME_Negotiator-2.o
```

The Colab notebook lives here:

```text
notebooks/colab_grpo_sme_liquidity.ipynb
```

The public Space for the environment is here:

```text
https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator
```

The model card and blog assets are included in:

```text
huggingface/model_card.md
huggingface/blog_post.md
scripts/publish_hf_model.py
```

---

## What You Need

For the quickest path, use Google Colab with a T4 GPU.

Recommended:

- Colab GPU runtime
- Hugging Face account
- `HF_TOKEN` saved in Colab Secrets if you want to publish the trained adapter
- Python 3.11 or 3.12
- A clean runtime before training

The demo uses:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Trainer: TRL `GRPOTrainer`
- Adapter: LoRA with PEFT
- Environment: OpenEnv SME Negotiator
- Task: `liquidity-correlation-hard`

---

## Step 1: Install Dependencies

In Colab, start from a fresh runtime and run:

```python
%pip install -q -U pip
%pip install -q "trl>=0.29.0" "transformers>=4.45.0" "peft>=0.19.1" \
  "datasets" "accelerate" "huggingface_hub>=0.20.0" "matplotlib"
```

Then install the project:

```python
%pip install -q -e ".[rl]"
```

If you see a PEFT import error later, restart the runtime after upgrading PEFT:

```python
%pip install -q -U "peft>=0.19.1"
```

---

## Step 2: Configure The Run

The default notebook config is a tiny smoke test:

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TASK_NAME = "liquidity-correlation-hard"
TOTAL_PERIODS = 2
STEPS = 10
NUM_SAMPLES = 8
OUTPUT_DIR = Path("outputs/colab_grpo_sme_liquidity")
```

For a more useful training run, start easier:

```python
TASK_NAME = "liquidity-stress-medium"
TOTAL_PERIODS = 1
STEPS = 100
NUM_SAMPLES = 16
```

Then move to the hard task once rewards are nonzero:

```python
TASK_NAME = "liquidity-correlation-hard"
TOTAL_PERIODS = 2
STEPS = 300
NUM_SAMPLES = 32
```

---

## Step 3: Keep Colab Memory Stable

The notebook sets PyTorch allocator options before `torch` initializes:

```python
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

The GRPO helper also uses memory-conscious defaults:

- LoRA adapter training instead of full fine-tuning
- half precision on CUDA
- disabled KV cache during training
- gradient checkpointing
- shorter demo completions
- generation batch size kept small

This is what makes the 1.5B model usable on a Colab T4.

---

## Step 4: Run A Smoke Test

Before training, verify that the environment can reset and produce a valid observation:

```python
from server.environment import SMELiquidityEnvironment

env = SMELiquidityEnvironment(total_periods=TOTAL_PERIODS)
obs = env.reset(seed=42, difficulty="hard", task_name=TASK_NAME)
print(obs)
```

The initial observation includes the business state:

- buyer price
- buyer payment days
- SME liquidity threshold
- supplier payment days
- working-capital gap
- annual financing rate
- active deal id
- open deals

If this works, the environment is ready for GRPO.

---

## Step 5: Train With GRPO

Launch the training helper:

```python
from rl.demo import demo_train_grpo

history = demo_train_grpo(
    model_name=MODEL_NAME,
    steps=STEPS,
    total_periods=TOTAL_PERIODS,
    task_name=TASK_NAME,
    num_samples=NUM_SAMPLES,
    output_dir=str(OUTPUT_DIR),
)
```

You should see something like:

```text
trainable params: 9,232,384 || all params: 1,552,946,688 || trainable%: 0.5945
```

That means LoRA is active. Only a small adapter is being trained.

Do not worry if early training loss is `0.000000`. GRPO compares multiple completions for the same prompt. If both completions get the same reward, especially `[0.0, 0.0]`, the advantage is zero and the policy loss can be zero. The important signal is the reward history and episode transcripts, not the loss alone.

---

## Step 6: Plot Rewards

The notebook stores logged rewards in `history`:

```python
steps = history.get("steps", [])
avg_reward = history.get("avg_reward", [])

plt.figure(figsize=(7, 4))
plt.plot(steps, avg_reward, marker="o")
plt.xlabel("Training step")
plt.ylabel("Average episode reward")
plt.title("GRPO on SME Liquidity Negotiation")
plt.grid(alpha=0.3)
plt.show()
```

For a 10-step smoke run, the curve may be flat. For a longer run, look for:

- more nonzero rewards
- fewer invalid actions
- better final payment days
- fewer SME defaults
- better use of financing tools

---

## Step 7: Compare Before And After

Run the baseline heuristic:

```python
from rl.demo import run_policy_episode

print(run_policy_episode(
    policy="heuristic",
    seed=123,
    total_periods=TOTAL_PERIODS,
    task_name=TASK_NAME,
))
```

Then run the trained adapter:

```python
checkpoint_path = history.get("checkpoint_path")

print(run_policy_episode(
    policy="trained",
    seed=123,
    total_periods=TOTAL_PERIODS,
    task_name=TASK_NAME,
    checkpoint_path=checkpoint_path,
))
```

The trained checkpoint is a PEFT adapter. It should be loaded with the base model plus `PeftModel.from_pretrained(...)`, not as a standalone full model.

---

## Step 8: Publish To Hugging Face

After training succeeds, set your repo id:

```python
HF_REPO_ID = "YOUR_USERNAME/openenv-sme-negotiator-grpo"
```

Then publish the adapter and model card from Colab:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=HF_REPO_ID,
    repo_type="model",
    folder_path=str(history["checkpoint_path"]),
    ignore_patterns=["*.tmp", "*.log", "checkpoint-*"],
)
api.upload_file(
    repo_id=HF_REPO_ID,
    repo_type="model",
    path_or_fileobj="huggingface/model_card.md",
    path_in_repo="README.md",
)
```

Or publish locally:

```bash
python scripts/publish_hf_model.py \
  --repo-id YOUR_USERNAME/openenv-sme-negotiator-grpo
```

The model repo will contain the LoRA adapter files plus the model card.

---

## How It Works

The training loop has five main pieces:

```text
Dataset rows -> Environment factory -> Model generations -> Tool actions -> Reward function
```

Each dataset row seeds one negotiation episode. The environment exposes tool-like methods to the model:

```text
propose_terms
accept_offer
reject_offer
use_tool
advance_period
```

The reward function reads the final environment state and computes a deterministic SME outcome. It rewards policies that:

- keep the SME solvent
- improve payment terms
- preserve positive NPV
- use financing tools only when useful
- avoid invalid or repeated tool calls

The hard task is intentionally sparse. Most random early policies fail. That is why curriculum helps: train on `liquidity-stress-medium` first, then move to `liquidity-correlation-hard`.

---

## Troubleshooting

### CUDA out of memory

Restart the Colab runtime, then rerun from the top. If it still fails, reduce:

```python
STEPS = 50
NUM_SAMPLES = 8
```

And keep:

```python
"num_generations": 2
"generation_batch_size": 2
"max_completion_length": 256
```

### Training loss is zero

This is normal early in GRPO when both completions receive the same reward. Increase steps, start on the easier task, or add more generations if memory allows.

### PEFT import error

Upgrade PEFT and restart:

```python
%pip install -q -U "peft>=0.19.1"
```

### `run_policy_episode` is not defined

Import it again:

```python
from rl.demo import run_policy_episode
```

Colab forgets variables after a runtime restart.

### Checkpoint path missing

Run the training cell first and confirm:

```python
history["checkpoint_path"]
```

The expected default is:

```text
outputs/colab_grpo_sme_liquidity/final-demo-model
```

---

## What To Try Next

Once the smoke test works, the next improvements are straightforward:

- train longer on the medium task
- move to hard mode only after rewards become nonzero
- increase `num_generations` to 4 if the GPU allows it
- export successful trajectories as a dataset
- compare heuristic, base model, and GRPO adapter policies
- publish a Space that runs the trained adapter interactively

---

## Links

- Space: https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator
- Base model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Project code: https://github.com/SkandaGanesha1/OpenEnv_SME_Negotiator-2.o
- Model card template: `huggingface/model_card.md`
- Upload script: `scripts/publish_hf_model.py`

---

## Safety Note

This is a simulated negotiation benchmark. It is not financial advice. Do not use it to make real credit, lending, invoice-discounting, or supplier-payment decisions.
