# Judge one-pager — OpenEnv SME Negotiator

**Skim time:** < 60 seconds · **Simulated benchmark — not financial advice.**

---

## 1) Problem (why this exists)

| | |
|---|---|
| **Hook** | Supplier ships **now**; buyer wants **60–90+** day settlement; payroll and vendors run on a **shorter** clock. |
| **Scale** | **₹8.1 lakh crore** delayed MSME receivables (README crisis table + linked sources). **6.4+ crore** MSMEs; **30+ crore** jobs; **~31%** GDP; **~48%** exports. |
| **Mechanism** | **TReDS** (RBI invoice discounting) **₹0 → ₹2.4 lakh crore** volume — README RBI link. |

*Full table + screenshots:* [README.md](../README.md) (crisis section, `EconomicTimes.png`, `Razorpay.png`).

---

## 2) Environment (what we built)

| | |
|---|---|
| **Surface** | **Gradio** “**ClearPay**” UI (`python app.py`) — **Active Negotiation** chat, **Your Next Move** (Action / Price / Days), **Deal Economics** scorecard, **System Observation JSON**. |
| **Core MDP** | `SMENegotiatorEnvironment` — `reset` / `step`, typed actions, deterministic terminal score. |
| **Tasks (UI labels)** | `🟢 Easy — compress days ≤ 60` · `🟡 Medium — days ≤ 45 + clause` · `🔴 Hard — dynamic discounting` → `payment-terms-easy/medium/hard`. |
| **Advanced** | Liquidity + GRPO path in Colab (`liquidity-stress-medium`, `liquidity-correlation-hard`) — see [EVALUATION.md](../EVALUATION.md). |

*Observation field cheat sheet:* Gradio tab **📖 Reference & API** (ships with app).

---

## 3) What the policy learned (evidence)

| Task | Mean reward | Where |
|------|-------------:|-------|
| `payment-terms-easy` | **0.99** | [outputs/judge_report.md](../outputs/judge_report.md) |
| `payment-terms-medium` | **0.99** | same |
| `payment-terms-hard` | **0.71** | same |
| `liquidity-stress-medium` | **0.81** | same |
| `liquidity-correlation-hard` | **0.94** | same |

**Contrast:** Public LLM baseline on hard single-deal task ≈ **0.08** — [EVALUATION.md](../EVALUATION.md) / README baseline narrative.

**Plain English:** solvency preserved; risky days compressed; **TReDS** / early-payment tools used when they improve scored outcomes; invalid tool spam penalised (liquidity notes in judge report).

---

## 4) Where to verify (60-second audit trail)

| Artifact | Path / URL |
|----------|------------|
| **Live Space (legacy OpenEnv tasks)** | [HF Space](https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator) |
| **Local Gradio** | `python app.py` → `http://127.0.0.1:7860/` (or next free port) |
| **Manifest** | [openenv.yaml](../openenv.yaml) |
| **Judge scores** | [outputs/judge_report.md](../outputs/judge_report.md) |
| **Demo script** | [docs/storytelling/DEMO_SCRIPT.md](storytelling/DEMO_SCRIPT.md) |
| **Illustrative reward curve (always in repo)** | [docs/img/tiny_grpo_reward_curve.svg](img/tiny_grpo_reward_curve.svg) |
| **Colab training narrative** | [notebooks/colab_grpo_sme_liquidity.ipynb](../notebooks/colab_grpo_sme_liquidity.ipynb) · [huggingface/blog_post.md](../huggingface/blog_post.md) |

**Note:** README links to `outputs/grpo_sme_liquidity_colab/*.png` — regenerate via Colab before embedding in slides if missing.
