---
license: mit
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
tags:
- reinforcement-learning
- grpo
- lora
- openenv
- sme-finance
- negotiation
- liquidity-risk
- tool-use
pipeline_tag: text-generation
---

# OpenEnv SME Negotiator GRPO

This repository contains a GRPO-trained policy adapter for the OpenEnv SME Liquidity Negotiator environment. The policy starts from `Qwen/Qwen2.5-1.5B-Instruct` and is trained to act as an SME negotiating buyer payment terms under working-capital pressure.

The environment simulates invoice-size, payment-days, buyer-power, financing, and default-risk tradeoffs. The hard task, `liquidity-correlation-hard`, stresses the policy with correlated buyer risk, thin cash buffers, tight supplier payment terms, and limited financing capacity.

## Intended Use

This model is intended for research and demonstration of reinforcement learning in simulated commercial negotiation workflows. It can be used to study:

- payment-term negotiation under liquidity constraints
- tool-using policies for invoice financing decisions
- reward design for solvency, NPV, and process discipline
- OpenEnv-compatible environment training loops

This model is not financial advice and should not be used to make real credit, lending, invoice-discounting, or supplier-payment decisions.

## Training Summary

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: GRPO
- Adapter method: LoRA
- Environment: OpenEnv SME Negotiator
- Task: `liquidity-correlation-hard`
- Demo notebook: `notebooks/colab_grpo_sme_liquidity.ipynb`
- Default demo steps: 10
- Default demo samples: 8
- Max completion length: 256

The Colab workflow uses memory-conscious defaults for T4-class GPUs: LoRA training, half precision, disabled KV cache, gradient checkpointing, and PyTorch expandable CUDA allocation segments.

## Reward Design

The training loop rewards policies that improve SME outcomes while avoiding brittle behavior:

- terminal solvency and positive NPV
- shorter buyer payment terms where possible
- sensible use of invoice financing tools
- avoidance of repeated invalid or spammy tool calls
- process alignment with the negotiation task

## Limitations

The environment is a deterministic simulation with simplified assumptions. Real-world SME finance depends on jurisdiction, lender underwriting, buyer credit quality, invoice validity, legal enforceability, and market conditions. The model may produce invalid JSON, overuse tools, or optimize for simulator-specific behavior.

## Reproducibility

Run the Colab notebook or use the local helper:

```python
from rl.demo import demo_train_grpo

history = demo_train_grpo(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    task_name="liquidity-correlation-hard",
    steps=10,
    num_samples=8,
    output_dir="outputs/colab_grpo_sme_liquidity",
)
```

After training, publish the generated checkpoint folder:

```bash
python scripts/publish_hf_model.py --repo-id YOUR_USERNAME/openenv-sme-negotiator-grpo
```

## Project Links

- Space: https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator
- Code: https://github.com/SkandaGanesha1/OpenEnv_SME_Negotiator-2.o
