"""
Gradio UI for the OpenEnv SME Negotiator — interactive B2B payment-term negotiation demo.

This app lets users step through a single-deal negotiation episode manually or
watch a heuristic agent play it out, without needing to run the FastAPI server
separately.  It uses the in-process SMENegotiatorEnvironment directly.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Any, Optional

# ── make sure the repo root is importable even when launched from a sub-directory ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _maybe_reexec_into_project_venv() -> None:
    """
    On Windows, many users run `python app.py` from the global interpreter.
    If the project venv exists, re-exec into it before importing heavy deps.
    """
    venv_python = os.path.join(_ROOT, ".venv311", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        return

    current_python = os.path.abspath(sys.executable)
    target_python = os.path.abspath(venv_python)
    if current_python == target_python:
        return

    # Guard against accidental loops if exec fails repeatedly.
    if os.environ.get("SME_NEGOTIATOR_VENV_REEXEC") == "1":
        return

    child_env = dict(os.environ)
    child_env["SME_NEGOTIATOR_VENV_REEXEC"] = "1"
    result = subprocess.run([target_python, *sys.argv], cwd=_ROOT, env=child_env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    _maybe_reexec_into_project_venv()

import gradio as gr

from server.environment import SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction
from sme_negotiator_env.task_config import TASK_REGISTRY, TaskConfig

# ─────────────────────────────────────────────────────────────────────────────
# Global environment instance (one per Gradio session via gr.State)
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "🟢 Easy  — compress days ≤ 60":   "payment-terms-easy",
    "🟡 Medium — days ≤ 45 + clause":  "payment-terms-medium",
    "🔴 Hard  — dynamic discounting":   "payment-terms-hard",
}

ACTION_TYPES = ["propose", "accept", "reject"]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a new environment and run reset(), returning (env, obs_dict)
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_dict(obs) -> dict:
    """Normalise an observation object (Pydantic model or plain dict) to a dict."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs)


def _make_env_and_reset(task_label: str, seed: int) -> tuple[SMENegotiatorEnvironment, dict]:
    task_id  = TASKS[task_label]
    env = SMENegotiatorEnvironment()
    obs      = env.reset(seed=seed, task_name=task_id)
    return env, _obs_to_dict(obs)


def _obs_to_display(obs_dict: dict) -> tuple[str, str]:
    """Return (markdown_summary, json_string) for the current observation."""
    md = f"""
| Field | Value |
|---|---|
| **Round** | {obs_dict.get('round_number', '?')} / {obs_dict.get('max_rounds', '?')} |
| **Buyer price** | ₹ {obs_dict.get('buyer_price', 0):.2f} / unit |
| **Buyer days** | {obs_dict.get('buyer_days', 0)} days |
| **Liquidity threshold** | {obs_dict.get('liquidity_threshold', 0)} days |
| **Working capital gap** | ₹ {obs_dict.get('working_capital_gap', 0):,.0f} |
| **Interest rate (annual)** | {obs_dict.get('interest_rate_annual', 0)*100:.1f}% |
| **Buyer power score** | {obs_dict.get('buyer_power_score', 0):.2f} |
| **TReDS available** | {'✅ Yes' if obs_dict.get('treds_available') else 'ℹ️ via use_treds flag'} |
| **Step reward** | {obs_dict.get('step_reward', 0):.4f} |
| **Message** | {obs_dict.get('message', '')} |
"""
    return md.strip(), json.dumps(obs_dict, indent=2, default=str)


def _score_bar(score: float) -> str:
    """ASCII progress bar for a [0, 1] score."""
    filled = max(0, min(10, round(score * 10)))
    bar    = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {score:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio callback functions
# ─────────────────────────────────────────────────────────────────────────────

def reset_episode(task_label: str, seed: int, env_state: dict) -> tuple:
    """Initialise a fresh episode and return all UI updates."""
    try:
        env, obs = _make_env_and_reset(task_label, int(seed))
        task_id  = TASKS[task_label]
        cfg: TaskConfig = TASK_REGISTRY[task_id]

        env_state["env"]      = env
        env_state["obs"]      = obs
        env_state["done"]     = False
        env_state["log"]      = []
        env_state["cum_rew"]  = 0.0
        env_state["step"]     = 0

        obs_md, obs_json = _obs_to_display(obs)
        task_md = f"""
**Task:** `{task_id}`
**Difficulty:** {cfg.difficulty}
**Description:** {cfg.description}
**Context:** {cfg.context_note}
"""
        status = f"✅ Episode started — task **{task_id}** | seed {seed}"
        log_text = "--- Episode started ---\n"
        score_text = _score_bar(0.0)

        # Default proposed values from the opening observation
        p_price = float(obs.get("buyer_price", 100.0))
        p_days  = int(obs.get("buyer_days", 90))

        return (
            env_state,
            task_md.strip(),
            obs_md,
            obs_json,
            status,
            log_text,
            score_text,
            gr.update(interactive=True),   # step button
            gr.update(interactive=False),  # reset button during play — keep enabled
            p_price,
            p_days,
        )
    except Exception as exc:
        err = f"❌ Reset failed: {exc}"
        return (
            env_state, err, "", "{}", err, "", _score_bar(0.0),
            gr.update(interactive=False), gr.update(interactive=True),
            100.0, 90,
        )


def step_action(
    action_type: str,
    price: float,
    payment_days: int,
    use_treds: bool,
    propose_clause: bool,
    propose_dd: bool,
    dd_rate: float,
    reason: str,
    env_state: dict,
) -> tuple:
    """Apply one negotiation action and return UI updates."""
    env: Optional[SMENegotiatorEnvironment] = env_state.get("env")
    if env is None:
        err = "⚠️ No active episode — click **Reset / New Episode** first."
        return (env_state, "", "{}", err, env_state.get("log_text", ""), _score_bar(0.0))

    if env_state.get("done"):
        info = "ℹ️ Episode already finished — reset to start a new one."
        return (env_state, "", "{}", info, env_state.get("log_text", ""), _score_bar(env_state.get("cum_rew", 0.0)))

    try:
        action = NegotiationAction(
            action_type=action_type,
            price=float(price),
            payment_days=int(payment_days),
            use_treds=bool(use_treds),
            propose_late_payment_penalty_clause=bool(propose_clause),
            propose_dynamic_discounting=bool(propose_dd),
            dynamic_discount_annual_rate=float(dd_rate),
            reason=reason or "",
        )

        obs_after = env.step(action)
        obs_dict = _obs_to_dict(obs_after)

        env_state["obs"]    = obs_dict
        env_state["done"]   = bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))
        step_idx = env_state.get("step", 0) + 1
        env_state["step"]   = step_idx
        step_reward = float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))
        cum_rew = env_state.get("cum_rew", 0.0) + step_reward
        env_state["cum_rew"] = cum_rew

        obs_md, obs_json = _obs_to_display(obs_dict)
        done = env_state["done"]

        log_line = (
            f"Step {step_idx:02d} | {action_type:7s} | "
            f"price=₹{price:.1f} days={payment_days} | "
            f"reward={step_reward:.4f} | done={done}\n"
        )
        prev_log = env_state.get("log_text", "--- Episode started ---\n")
        env_state["log_text"] = prev_log + log_line

        if done:
            env_state["cum_rew"] = cum_rew
            status = (
                f"🏁 Episode finished — cumulative reward: **{cum_rew:.4f}** | "
                f"deal reached: {obs_dict.get('buyer_accepted', obs_dict.get('negotiation_done', '?'))}"
            )
        else:
            status = f"⏩ Step {step_idx} applied — reward: {step_reward:.4f} | cumulative: {cum_rew:.4f}"

        score_text = _score_bar(min(1.0, max(0.0, cum_rew)))
        return (env_state, obs_md, obs_json, status, env_state["log_text"], score_text)

    except Exception as exc:
        err = f"❌ Step failed: {exc}"
        return (env_state, "", "{}", err, env_state.get("log_text", ""), _score_bar(0.0))


def heuristic_play(task_label: str, seed: int) -> tuple:
    """Run a full heuristic episode (compress payment days greedily) and show results."""
    try:
        task_id = TASKS[task_label]
        env = SMENegotiatorEnvironment()
        obs = env.reset(seed=int(seed), task_name=task_id)
        obs_dict = _obs_to_dict(obs)

        lines = [f"=== Heuristic playthrough — {task_id} (seed {seed}) ===\n"]
        cum_rew = 0.0
        step = 0
        done = False

        while not done:
            step += 1
            b_days  = int(obs_dict.get("buyer_days", 60))
            b_price = float(obs_dict.get("buyer_price", 100.0))
            liq     = int(obs_dict.get("liquidity_threshold", 45))

            # Simple heuristic: propose target days; accept if within threshold
            if b_days <= liq:
                action_type = "accept"
                p_days = b_days
            else:
                action_type = "propose"
                p_days = max(liq, b_days - 5)

            action = NegotiationAction(
                action_type=action_type,
                price=b_price,
                payment_days=p_days,
                use_treds=(b_days > liq + 15),
                propose_late_payment_penalty_clause=(task_id == "payment-terms-medium"),
                propose_dynamic_discounting=(task_id == "payment-terms-hard"),
                dynamic_discount_annual_rate=0.08 if task_id == "payment-terms-hard" else 0.0,
            )

            obs_after = env.step(action)
            obs_dict = _obs_to_dict(obs_after)
            step_reward = float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))
            cum_rew += step_reward
            done = bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))

            lines.append(
                f"Step {step:02d} | {action_type:7s} | price=₹{b_price:.1f} days={p_days} | "
                f"reward={step_reward:.4f} | done={done}"
            )
            if step >= 25:  # safety cap
                break

        final_md, final_json = _obs_to_display(obs_dict)
        transcript = "\n".join(lines) + f"\n\nTotal reward: {cum_rew:.4f}"
        score_text = _score_bar(min(1.0, max(0.0, cum_rew)))
        status = f"🤖 Heuristic finished — {step} steps | cumulative reward: **{cum_rew:.4f}**"
        return (final_md, final_json, status, transcript, score_text)

    except Exception as exc:
        err = f"❌ Heuristic playthrough failed: {exc}"
        return ("", "{}", err, err, _score_bar(0.0))


def compute_grader_score(task_label: str, agreed_days: int, agreed_price: float,
                          deal_reached: bool, late_clause: bool,
                          dynamic_dd: bool, dd_rate: float) -> str:
    """Standalone grader calculator — shows what score a set of terms would receive."""
    from sme_negotiator_env.graders import TASK_GRADERS
    from sme_negotiator_env.models import NegotiationState

    task_id = TASKS[task_label]
    cfg = TASK_REGISTRY[task_id]

    dummy_state = NegotiationState(
        episode_id="demo",
        seed=0,
        difficulty=cfg.difficulty,
        task_name=task_id,
        step_count=1,
        max_steps=cfg.max_rounds,
        max_rounds=cfg.max_rounds,
        deal_reached=deal_reached,
        final_price=float(agreed_price) if deal_reached else None,
        final_days=int(agreed_days) if deal_reached else None,
        treds_used=False,
        cumulative_reward=0.0,
        buyer_price=float(cfg.initial_buyer_price),
        buyer_days=int(cfg.initial_buyer_days),
        initial_buyer_days=int(cfg.initial_buyer_days),
        cost_threshold=float(cfg.cost_threshold),
        liquidity_threshold=int(cfg.liquidity_threshold),
        volume=int(cfg.volume),
        message="",
        sme_monthly_revenue=float(cfg.sme_monthly_revenue),
        current_payment_terms_days=int(agreed_days) if deal_reached else cfg.current_payment_terms_days,
        sme_supplier_payment_days=int(cfg.sme_supplier_payment_days),
        interest_rate_annual=float(cfg.interest_rate_annual),
        buyer_power_score=float(cfg.buyer_power_score),
        agreed_terms=int(agreed_days) if deal_reached else None,
        late_payment_penalty_agreed=bool(late_clause),
        dynamic_discounting_agreed=bool(dynamic_dd),
        agreed_dynamic_discount_annual=float(dd_rate),
    )

    grader = TASK_GRADERS.get(task_id)
    if grader is None:
        return f"No grader found for task `{task_id}`"

    score = grader(dummy_state)
    bar   = _score_bar(score)
    return f"**Terminal score:** {score:.6f}\n\n{bar}"


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.header-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              border-radius: 12px; padding: 24px; margin-bottom: 16px; color: white; }
.header-box h1 { font-size: 2rem; margin: 0 0 6px 0; }
.header-box p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }
.score-box { font-family: monospace; font-size: 1.1rem; font-weight: bold; padding: 8px; }
.task-badge-easy   { background: #d4edda; border-left: 4px solid #28a745; padding: 8px 12px; border-radius: 6px; }
.task-badge-medium { background: #fff3cd; border-left: 4px solid #ffc107; padding: 8px 12px; border-radius: 6px; }
.task-badge-hard   { background: #f8d7da; border-left: 4px solid #dc3545; padding: 8px 12px; border-radius: 6px; }
footer { text-align: center; opacity: 0.6; font-size: 0.8rem; margin-top: 32px; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SME Negotiator") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box">
      <h1>🤝 OpenEnv SME Negotiator</h1>
      <p>
        Interactive B2B payment-term negotiation environment — train and evaluate agents
        that defend SME liquidity against large buyers in India's ₹8.1 lakh crore
        delayed-payment crisis.
      </p>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── TAB 1: Interactive negotiation ───────────────────────────────────
        with gr.Tab("🎮 Interactive Negotiation"):

            env_state = gr.State({"env": None, "obs": {}, "done": False,
                                  "log": [], "cum_rew": 0.0, "step": 0,
                                  "log_text": ""})

            with gr.Row():
                # ── Left column: setup + action ──────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Episode Setup")

                    task_sel = gr.Dropdown(
                        choices=list(TASKS.keys()),
                        value=list(TASKS.keys())[0],
                        label="Task difficulty",
                        info="Choose the negotiation challenge level",
                    )
                    seed_num = gr.Number(value=42, label="Random seed", precision=0, minimum=0, maximum=99999)
                    reset_btn = gr.Button("🔄 Reset / New Episode", variant="primary")

                    task_info = gr.Markdown("*Click Reset to start an episode.*")

                    gr.Markdown("---")
                    gr.Markdown("### 🕹️ Take Action")

                    action_type = gr.Radio(
                        choices=ACTION_TYPES,
                        value="propose",
                        label="Action type",
                        info="propose = counter-offer | accept = agree | reject = walk away",
                    )
                    with gr.Row():
                        price_in  = gr.Number(value=100.0, label="Price (₹/unit)", minimum=0, step=0.5)
                        days_in   = gr.Number(value=60, label="Payment days", minimum=0, maximum=365, precision=0)

                    with gr.Accordion("⚡ Advanced action options", open=False):
                        use_treds    = gr.Checkbox(label="use_treds — invoke TReDS financing", value=False)
                        late_clause  = gr.Checkbox(label="propose_late_payment_penalty_clause (medium task)", value=False)
                        propose_dd   = gr.Checkbox(label="propose_dynamic_discounting (hard task)", value=False)
                        dd_rate      = gr.Slider(minimum=0.0, maximum=0.30, step=0.01, value=0.08,
                                                  label="Dynamic discount annual rate (0–30%)")
                        reason_in    = gr.Textbox(label="Reason (optional)", placeholder="e.g. Need faster payment for payroll",
                                                   lines=2, max_lines=3)

                    step_btn = gr.Button("▶ Apply Action", variant="primary", interactive=False)

                # ── Right column: observation + feedback ─────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Current Observation")
                    obs_status  = gr.Markdown("*Start an episode to see the environment state.*")
                    obs_table   = gr.Markdown("")

                    with gr.Row():
                        score_disp = gr.Textbox(label="Cumulative Reward", value=_score_bar(0.0),
                                                 elem_classes="score-box", interactive=False)

                    with gr.Accordion("🔍 Full JSON observation", open=False):
                        obs_json = gr.Code(language="json", label="Observation JSON", value="{}")

                    gr.Markdown("### 📋 Episode Log")
                    ep_log = gr.Textbox(value="", lines=10, max_lines=16, interactive=False,
                                         label="Step-by-step log", show_label=True)

            # ── Examples ────────────────────────────────────────────────────
            gr.Markdown("### 💡 Quick Examples")
            gr.Examples(
                examples=[
                    [list(TASKS.keys())[0], 42],
                    [list(TASKS.keys())[1], 7],
                    [list(TASKS.keys())[2], 99],
                ],
                inputs=[task_sel, seed_num],
                label="Preset episode configurations",
            )

            # ── Wiring ───────────────────────────────────────────────────────
            reset_outputs = [env_state, task_info, obs_table, obs_json, obs_status,
                             ep_log, score_disp, step_btn, reset_btn, price_in, days_in]

            reset_btn.click(
                fn=reset_episode,
                inputs=[task_sel, seed_num, env_state],
                outputs=reset_outputs,
            )

            step_btn.click(
                fn=step_action,
                inputs=[action_type, price_in, days_in, use_treds,
                        late_clause, propose_dd, dd_rate, reason_in, env_state],
                outputs=[env_state, obs_table, obs_json, obs_status, ep_log, score_disp],
            )

        # ── TAB 2: Heuristic auto-play ────────────────────────────────────────
        with gr.Tab("🤖 Heuristic Auto-Play"):
            gr.Markdown("""
**Watch a greedy heuristic agent** compress payment days toward the liquidity threshold.
It proposes `buyer_days − 5` each round and accepts when terms are within the threshold.
For the hard task it also proposes dynamic discounting at 8% annual rate.
""")
            with gr.Row():
                h_task = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0],
                                      label="Task")
                h_seed = gr.Number(value=42, label="Seed", minimum=0, maximum=99999, precision=0)
            h_run_btn = gr.Button("▶ Run Heuristic Playthrough", variant="primary")

            h_status    = gr.Markdown("")
            h_score     = gr.Textbox(label="Final reward", interactive=False, elem_classes="score-box")

            with gr.Row():
                with gr.Column(scale=1):
                    h_obs_table = gr.Markdown("", label="Final observation")
                with gr.Column(scale=1):
                    h_obs_json  = gr.Code(language="json", label="Final observation JSON", value="{}")

            h_transcript = gr.Textbox(label="Playthrough transcript", lines=14,
                                       interactive=False, show_label=True)

            gr.Examples(
                examples=[
                    [list(TASKS.keys())[0], 1],
                    [list(TASKS.keys())[1], 22],
                    [list(TASKS.keys())[2], 77],
                ],
                inputs=[h_task, h_seed],
            )

            h_run_btn.click(
                fn=heuristic_play,
                inputs=[h_task, h_seed],
                outputs=[h_obs_table, h_obs_json, h_status, h_transcript, h_score],
            )

        # ── TAB 3: Grader calculator ──────────────────────────────────────────
        with gr.Tab("🧮 Grader Calculator"):
            gr.Markdown("""
**Compute the deterministic terminal score** for any set of agreed terms —
without running a full episode.  Useful for understanding the reward function.
""")
            with gr.Row():
                with gr.Column():
                    gc_task       = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0],
                                                 label="Task")
                    gc_deal       = gr.Checkbox(label="Deal reached", value=True)
                    gc_days       = gr.Slider(minimum=0, maximum=120, step=1, value=55,
                                              label="Agreed payment days")
                    gc_price      = gr.Slider(minimum=70.0, maximum=110.0, step=0.5, value=100.0,
                                              label="Agreed price (₹/unit)")
                    gc_clause     = gr.Checkbox(label="Late payment penalty clause agreed (medium)", value=False)
                    gc_dd         = gr.Checkbox(label="Dynamic discounting agreed (hard)", value=False)
                    gc_dd_rate    = gr.Slider(minimum=0.0, maximum=0.30, step=0.01, value=0.08,
                                              label="Dynamic discount annual rate")
                    gc_compute    = gr.Button("Calculate Score", variant="primary")

                with gr.Column():
                    gc_result = gr.Markdown("*Fill in the form and click Calculate.*")

            gr.Examples(
                examples=[
                    [list(TASKS.keys())[0], 55, 100.0, True, False, False, 0.0],
                    [list(TASKS.keys())[1], 44, 100.0, True, False, True, 0.0],
                    [list(TASKS.keys())[2], 50, 96.0, True, False, True, 0.08],
                    [list(TASKS.keys())[0], 80, 100.0, True, False, False, 0.0],
                ],
                inputs=[gc_task, gc_days, gc_price, gc_deal, gc_clause, gc_dd, gc_dd_rate],
            )

            gc_compute.click(
                fn=compute_grader_score,
                inputs=[gc_task, gc_days, gc_price, gc_deal, gc_clause, gc_dd, gc_dd_rate],
                outputs=[gc_result],
            )

        # ── TAB 4: Reference ──────────────────────────────────────────────────
        with gr.Tab("📖 Reference"):
            gr.Markdown("""
## Observation space

| Field | Meaning |
|---|---|
| `buyer_price` | Buyer's current offer (₹/unit) |
| `buyer_days`  | Buyer's proposed settlement period (days) |
| `liquidity_threshold` | Max days before SME cash crisis |
| `working_capital_gap` | INR shortfall to bridge during the wait |
| `sme_supplier_payment_days` | When SME must pay its own suppliers |
| `interest_rate_annual` | Cost of bridging finance (annual %) |
| `buyer_power_score` | Buyer negotiating leverage [0–1] |
| `round_number` | Current negotiation round |
| `max_rounds` | Episode length cap |
| `step_reward` | Shaping reward for this step |

## Action space

| Field | Meaning |
|---|---|
| `action_type` | `propose` / `accept` / `reject` |
| `price` | Your counter-price (₹/unit) |
| `payment_days` | Your proposed settlement period |
| `use_treds` | Invoke TReDS financing |
| `propose_late_payment_penalty_clause` | Add late-payment clause (medium) |
| `propose_dynamic_discounting` | Propose early-payment discount (hard) |
| `dynamic_discount_annual_rate` | Annualised discount rate e.g. 0.08 = 8% |
| `reason` | Free-text agent reasoning |

## Task ladder

| Task | Opens at | Goal | Full credit |
|---|---|---|---|
| payment-terms-easy | ₹100 / 90 days | Compress days ≤ 60 | agreed_days ≤ 60 |
| payment-terms-medium | ₹100 / 60 days | Days ≤ 45 + penalty clause | agreed_days ≤ 45 |
| payment-terms-hard | ₹96 / 100 days | Dynamic discounting NPV uplift | propose_dynamic_discounting + NPV > baseline |

## Links

- [GitHub](https://github.com/SkandaGanesha1/ENV)
- [HF Space](https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator)
- [OpenEnv docs](https://huggingface.co/docs/openenv)
- [EVALUATION.md](EVALUATION.md)
""")

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer" style="text-align:center;opacity:0.6;font-size:0.8rem;margin-top:32px;">
      OpenEnv SME Negotiator · MIT License ·
      <a href="https://github.com/SkandaGanesha1/ENV" target="_blank">GitHub</a> ·
      <a href="https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator" target="_blank">HF Space</a>
    </div>
    """)


if __name__ == "__main__":
    requested_port = int(os.getenv("GRADIO_PORT", "7861"))
    max_attempts = 20
    port = requested_port

    for _ in range(max_attempts):
        try:
            if port == requested_port:
                print(f"[startup] Launching on requested port {port}")
            else:
                print(f"[startup] Requested port {requested_port} busy; launching on fallback port {port}")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                theme=gr.themes.Soft(primary_hue="blue"),
                css=CUSTOM_CSS,
            )
            break
        except OSError:
            print(f"[startup] Port {port} unavailable, trying {port + 1}...")
            port += 1
    else:
        raise RuntimeError(
            f"Could not find a free port starting from {requested_port} "
            f"after {max_attempts} attempts. Set GRADIO_PORT to a known free port."
        )
