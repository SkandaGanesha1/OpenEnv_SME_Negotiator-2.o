"""
Gradio UI for the OpenEnv SME Negotiator — two-column interactive playground.

Uses ``SMENegotiatorEnvironment`` in-process with a left control column (status,
actions, simulation fields) and a right column (chat, last-step JSON, quick
connect snippet). Heuristic auto-play and grader tools live in secondary tabs.
"""

from __future__ import annotations

import json
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
TASKS = {
    "🟢 Easy  — compress days ≤ 60": "payment-terms-easy",
    "🟡 Medium — days ≤ 45 + clause": "payment-terms-medium",
    "🔴 Hard  — dynamic discounting": "payment-terms-hard",
}

# UI-only label: env ``NegotiationAction`` has no ``counter_offer``; map to ``propose``.
UI_ACTION_CHOICES = ["propose", "counter_offer", "accept", "reject"]

QUICK_CONNECT_SNIPPET = '''\
from server.environment import SMENegotiatorEnvironment
from sme_negotiator_env.models import NegotiationAction

env = SMENegotiatorEnvironment()
obs = env.reset(seed=42, task_name="payment-terms-easy")
# obs is a NegotiationObservation (Pydantic); use obs.model_dump() for a dict.

action = NegotiationAction(
    action_type="propose",
    price=float(obs.buyer_price),
    payment_days=int(obs.buyer_days),
    use_treds=False,
    reason="Sync API example",
)
obs2 = env.step(action)
print(obs2.model_dump())
'''


def _obs_to_dict(obs: Any) -> dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs)


def _map_ui_action_to_literal(ui_action: str) -> str:
    """Map Gradio dropdown value to ``NegotiationAction.action_type`` literal."""
    a = (ui_action or "propose").strip().lower()
    # counter_offer is UI-only — judges see the label; wire uses propose.
    if a == "counter_offer":
        return "propose"
    return a


def _parse_simulation_plan(raw: str) -> Optional[dict[str, Any]]:
    text = (raw or "").strip()
    if not text or text.lower() == "default":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("simulation_plan must be valid JSON, 'default', or empty.") from None
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError("simulation_plan JSON must be an object (dict).")
    return parsed


def _score_bar(score: float) -> str:
    filled = max(0, min(10, round(float(score) * 10)))
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {score:.4f}"


def _status_html(*, round_number: int, last_price: float, done: bool) -> str:
    """HTML strip for round / last buyer price (post-step) / Active vs Done."""
    badge = "Done" if done else "Active"
    badge_cls = "done" if done else "active"
    return f"""
<div class="status-bar">
  <span><strong>Round</strong> {int(round_number)}</span>
  <span><strong>Last price</strong> ₹{float(last_price):.2f}/unit</span>
  <span class="badge {badge_cls}">{badge}</span>
</div>
"""


def _build_last_step_json(obs_dict: dict[str, Any]) -> dict[str, Any]:
    """Shape for ``gr.JSON``: reward, done, and a small derived ``info`` subset."""
    reward = float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))
    done = bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))
    buyer_price = float(obs_dict.get("buyer_price", 0.0))
    buyer_days = int(obs_dict.get("buyer_days", 0))
    rnd = int(obs_dict.get("round_number", 0))
    volume = int(obs_dict.get("volume", 0))
    deal_value = (buyer_price * volume) if volume > 0 else None
    meta = obs_dict.get("metadata")
    meta_out: Optional[dict[str, Any]] = None
    if isinstance(meta, dict):
        meta_out = {k: meta[k] for k in list(meta.keys())[:12]}
    info: dict[str, Any] = {
        "buyer_price": buyer_price,
        "buyer_days": buyer_days,
        "round_number": rnd,
        "message": obs_dict.get("message", ""),
        "buyer_accepted": bool(obs_dict.get("buyer_accepted", False)),
        "negotiation_done": bool(obs_dict.get("negotiation_done", False)),
        "deal_value": deal_value,
        "payment_terms": buyer_days,
        "episode_step": rnd,
    }
    if meta_out is not None:
        info["metadata"] = meta_out
    return {"reward": reward, "done": done, "info": info}


def _make_env_and_reset(task_label: str, seed: int) -> tuple[SMENegotiatorEnvironment, dict[str, Any]]:
    task_id = TASKS[task_label]
    env = SMENegotiatorEnvironment()
    obs = env.reset(seed=int(seed), task_name=task_id)
    return env, _obs_to_dict(obs)


def _format_user_turn(
    *,
    ui_action: str,
    wired_action: str,
    price: float,
    days: int,
    reason: str,
    deal_id: str,
    use_treds: bool,
    sim_plan_raw: str,
    horizon: Optional[int],
) -> str:
    lines = [
        f"**UI action:** `{ui_action}` → **wired:** `{wired_action}`",
        f"**Price (INR/unit):** {price:g}  **Payment days:** {int(days)}",
        f"**use_treds:** {use_treds}",
    ]
    if deal_id.strip():
        lines.append(f"**deal_id:** `{deal_id.strip()}`")
    if reason.strip():
        lines.append(f"**Reason:** {reason.strip()}")
    sp = (sim_plan_raw or "").strip()
    if sp and sp.lower() != "default":
        lines.append(f"**simulation_plan (raw):** `{sp[:200]}{'…' if len(sp) > 200 else ''}`")
    if horizon is not None and int(horizon) > 0:
        lines.append(f"**simulation_horizon:** {int(horizon)}")
    return "\n".join(lines)


def _format_assistant_turn(obs_dict: dict[str, Any]) -> str:
    reward = float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))
    done = bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))
    parts = [
        f"**reward:** {reward:.4f}  **done:** {done}",
        f"**negotiation_done:** {obs_dict.get('negotiation_done')}  "
        f"**buyer_accepted:** {obs_dict.get('buyer_accepted')}",
        f"**buyer_price:** ₹{float(obs_dict.get('buyer_price', 0)):.2f}/unit  "
        f"**buyer_days:** {int(obs_dict.get('buyer_days', 0))}",
    ]
    msg = obs_dict.get("message") or ""
    if msg:
        parts.append(f"**message:** {msg}")
    return "\n".join(parts)


def reset_episode(
    task_label: str,
    seed: int,
    state: dict[str, Any],
) -> tuple[Any, ...]:
    """New episode: fresh env, empty chat/JSON, status + price/days from first obs."""
    try:
        env, obs = _make_env_and_reset(task_label, int(seed))
        task_id = TASKS[task_label]
        cfg: TaskConfig = TASK_REGISTRY[task_id]

        state = {
            "env": env,
            "messages": [],
            "last_payload": {"reward": 0.0, "done": False, "info": {}},
            "task_label": task_label,
            "seed": int(seed),
            "cum_rew": 0.0,
        }

        p_price = float(obs.get("buyer_price", 100.0))
        p_days = int(obs.get("buyer_days", 90))
        last_price = float(obs.get("buyer_price", 0.0))
        rnd = int(obs.get("round_number", 0))
        done0 = bool(obs.get("done", obs.get("negotiation_done", False)))

        status_html = _status_html(round_number=rnd, last_price=last_price, done=done0)
        task_md = (
            f"**Task:** `{task_id}`  \n**Difficulty:** {cfg.difficulty}  \n"
            f"**Description:** {cfg.description}  \n**Context:** {cfg.context_note}"
        )
        json_payload = _build_last_step_json(obs)

        return (
            state,
            status_html,
            task_md,
            [],
            json_payload,
            p_price,
            p_days,
            gr.update(interactive=True),
        )
    except Exception as exc:
        err = str(exc)
        state = {"env": None, "messages": [], "last_payload": {"reward": 0.0, "done": False, "info": {}}, "cum_rew": 0.0}
        status_html = f'<div class="status-bar error">Reset failed: {err}</div>'
        return (
            state,
            status_html,
            f"❌ {err}",
            [],
            {"reward": 0.0, "done": False, "info": {"message": err}},
            100.0,
            90,
            gr.update(interactive=False),
        )


def submit_step(
    ui_action: str,
    price: float,
    payment_days: int,
    reason: str,
    deal_id: str,
    use_treds: bool,
    simulation_plan_raw: str,
    simulation_horizon: float,
    propose_clause: bool,
    propose_dd: bool,
    dd_rate: float,
    state: dict[str, Any],
) -> tuple[Any, ...]:
    env: Optional[SMENegotiatorEnvironment] = state.get("env")
    messages: list = list(state.get("messages") or [])

    if env is None:
        err = "⚠️ No active episode — use **Reset Episode** first."
        messages = messages + [[_format_user_turn(
            ui_action=ui_action or "",
            wired_action="—",
            price=float(price),
            days=int(payment_days),
            reason=reason or "",
            deal_id=deal_id or "",
            use_treds=bool(use_treds),
            sim_plan_raw=simulation_plan_raw or "",
            horizon=int(simulation_horizon) if simulation_horizon else None,
        ), err]]
        return (
            state,
            _status_html(round_number=0, last_price=0.0, done=False),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
            gr.update(),
        )

    wired = _map_ui_action_to_literal(ui_action or "propose")

    try:
        sim_plan = _parse_simulation_plan(simulation_plan_raw or "")
    except ValueError as ve:
        err = str(ve)
        user_t = _format_user_turn(
            ui_action=ui_action or "",
            wired_action=wired,
            price=float(price),
            days=int(payment_days),
            reason=reason or "",
            deal_id=deal_id or "",
            use_treds=bool(use_treds),
            sim_plan_raw=simulation_plan_raw or "",
            horizon=int(simulation_horizon) if simulation_horizon else None,
        )
        messages = messages + [[user_t, f"**Parse error:** {err}"]]
        return (
            state,
            gr.update(),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
            gr.update(),
        )

    horizon_int: Optional[int] = None
    if simulation_horizon is not None and float(simulation_horizon) > 0:
        horizon_int = int(simulation_horizon)

    action = NegotiationAction(
        action_type=wired,  # type: ignore[arg-type]
        price=float(price),
        payment_days=int(payment_days),
        use_treds=bool(use_treds),
        reason=(reason or None) or None,
        deal_id=(deal_id.strip() or None),
        simulation_plan=sim_plan,
        simulation_horizon=horizon_int,
        propose_late_payment_penalty_clause=bool(propose_clause),
        propose_dynamic_discounting=bool(propose_dd),
        dynamic_discount_annual_rate=float(dd_rate),
    )

    try:
        obs_after = env.step(action)
        obs_dict = _obs_to_dict(obs_after)
    except Exception as exc:
        err = f"❌ Step failed: {exc}"
        user_t = _format_user_turn(
            ui_action=ui_action or "",
            wired_action=wired,
            price=float(price),
            days=int(payment_days),
            reason=reason or "",
            deal_id=deal_id or "",
            use_treds=bool(use_treds),
            sim_plan_raw=simulation_plan_raw or "",
            horizon=horizon_int,
        )
        messages = messages + [[user_t, err]]
        return (state, gr.update(), messages, {"reward": 0.0, "done": False, "info": {"message": err}}, gr.update(), gr.update())

    done = bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))
    step_reward = float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))
    cum = float(state.get("cum_rew", 0.0)) + step_reward
    state = dict(state)
    state["cum_rew"] = cum

    last_price = float(obs_dict.get("buyer_price", 0.0))
    rnd = int(obs_dict.get("round_number", 0))
    status_html = _status_html(round_number=rnd, last_price=last_price, done=done)

    user_t = _format_user_turn(
        ui_action=ui_action or "",
        wired_action=wired,
        price=float(price),
        days=int(payment_days),
        reason=reason or "",
        deal_id=deal_id or "",
        use_treds=bool(use_treds),
        sim_plan_raw=simulation_plan_raw or "",
        horizon=horizon_int,
    )
    assistant_t = _format_assistant_turn(obs_dict)
    messages = messages + [[user_t, assistant_t]]

    json_payload = _build_last_step_json(obs_dict)

    return (
        state,
        status_html,
        messages,
        json_payload,
        float(obs_dict.get("buyer_price", price)),
        int(obs_dict.get("buyer_days", payment_days)),
        gr.update(interactive=not done),
    )


def heuristic_play(task_label: str, seed: int) -> tuple[Any, ...]:
    """Greedy heuristic episode (legacy tab)."""
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
            b_days = int(obs_dict.get("buyer_days", 60))
            b_price = float(obs_dict.get("buyer_price", 100.0))
            liq = int(obs_dict.get("liquidity_threshold", 45))

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
            if step >= 25:
                break

        final_json = json.dumps(obs_dict, indent=2, default=str)
        transcript = "\n".join(lines) + f"\n\nTotal reward: {cum_rew:.4f}"
        score_text = _score_bar(min(1.0, max(0.0, cum_rew)))
        status = f"🤖 Heuristic finished — {step} steps | cumulative reward: **{cum_rew:.4f}**"
        return (final_json, status, transcript, score_text)

    except Exception as exc:
        err = f"❌ Heuristic playthrough failed: {exc}"
        return ("{}", err, err, _score_bar(0.0))


def compute_grader_score(
    task_label: str,
    agreed_days: int,
    agreed_price: float,
    deal_reached: bool,
    late_clause: bool,
    dynamic_dd: bool,
    dd_rate: float,
) -> str:
    """Standalone grader calculator (legacy tab)."""
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
    bar = _score_bar(score)
    return f"**Terminal score:** {score:.6f}\n\n{bar}"


CUSTOM_CSS = """
.status-bar {
  display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
  padding: 10px 14px; border-radius: 8px;
  background: linear-gradient(90deg, #e8f0fe 0%, #f5f9ff 100%);
  border: 1px solid #c8d9f0; margin-bottom: 12px; font-size: 0.95rem;
}
.status-bar .badge { padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: 0.8rem; }
.status-bar .badge.active { background: #1a73e8; color: white; }
.status-bar .badge.done { background: #5f6368; color: white; }
.status-bar.error { background: #fce8e6; border-color: #f5c2c0; }
.header-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              border-radius: 12px; padding: 24px; margin-bottom: 16px; color: white; }
.header-box h1 { font-size: 2rem; margin: 0 0 6px 0; }
.header-box p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }
.score-box { font-family: monospace; font-size: 1.1rem; font-weight: bold; padding: 8px; }
footer { text-align: center; opacity: 0.6; font-size: 0.8rem; margin-top: 24px; }
"""


THEME = gr.themes.Soft(primary_hue="blue")

# Theme is passed to ``launch()`` for Gradio 6+ (constructor theme is deprecated).
with gr.Blocks(title="SME Negotiator") as demo:
    gr.HTML("""
    <div class="header-box">
      <h1>🤝 OpenEnv SME Negotiator</h1>
      <p>
        Two-column playground for B2B payment-term negotiation (INR / ₹ per unit).
        Use <strong>Reset Episode</strong> then <strong>Submit Step</strong>.
        <em>counter_offer</em> in the UI maps to <code>propose</code> on the wire.
      </p>
    </div>
    """)

    session_state = gr.State(
        {
            "env": None,
            "messages": [],
            "last_payload": {"reward": 0.0, "done": False, "info": {}},
            "cum_rew": 0.0,
        }
    )

    with gr.Row():
        # ── Left: controls ───────────────────────────────────────────────
        with gr.Column(scale=1):
            status_bar = gr.HTML(value=_status_html(round_number=0, last_price=0.0, done=False))

            with gr.Accordion("Episode", open=True):
                task_sel = gr.Dropdown(
                    choices=list(TASKS.keys()),
                    value=list(TASKS.keys())[0],
                    label="Task",
                )
                seed_num = gr.Number(value=42, label="Random seed", precision=0, minimum=0, maximum=99999)
                reset_btn = gr.Button("Reset Episode", variant="primary")
                task_info = gr.Markdown("*Reset to load task details.*")

            ui_action = gr.Dropdown(
                choices=UI_ACTION_CHOICES,
                value="propose",
                label="Action type",
                info="counter_offer is shown for UX; it is sent as propose.",
            )
            gr.Markdown(
                "<small>Prices are **INR per unit** (₹/unit), matching the simulator.</small>"
            )
            with gr.Row():
                price_in = gr.Number(value=100.0, label="Price (INR/unit)", minimum=0, step=0.5)
                days_in = gr.Number(value=90, label="Payment days", minimum=0, maximum=365, precision=0)

            reason_in = gr.Textbox(label="Reason (optional)", lines=2, placeholder="e.g. liquidity bridge")
            deal_id_in = gr.Textbox(label="Deal ID (optional)", placeholder="Passed to NegotiationAction.deal_id")
            use_treds = gr.Checkbox(label="Use TReDS (use_treds)", value=False)

            with gr.Accordion("Simulation settings", open=False):
                simulation_plan_tb = gr.Textbox(
                    value="default",
                    label="simulation_plan",
                    info="JSON object, the word default, or empty for None.",
                )
                simulation_horizon_n = gr.Number(
                    value=0,
                    label="simulation_horizon",
                    precision=0,
                    minimum=0,
                    info="Optional; used when horizon > 0.",
                )

            with gr.Accordion("Advanced (medium / hard tasks)", open=False):
                late_clause = gr.Checkbox(
                    label="propose_late_payment_penalty_clause",
                    value=False,
                )
                propose_dd = gr.Checkbox(label="propose_dynamic_discounting", value=False)
                dd_rate = gr.Slider(
                    minimum=0.0,
                    maximum=0.30,
                    step=0.01,
                    value=0.08,
                    label="dynamic_discount_annual_rate",
                )

            submit_btn = gr.Button("Submit Step", variant="primary", interactive=False)

        # ── Right: history + JSON + snippet ──────────────────────────────
        with gr.Column(scale=2):
            gr.HTML("<h3 style='margin:0 0 8px 0;'>Negotiation History</h3>")
            chat = gr.Chatbot(label="Chat", height=300)
            last_json = gr.JSON(label="Last step (normalized)")
            quick_code = gr.Code(
                value=QUICK_CONNECT_SNIPPET,
                language="python",
                label="Quick Connect (sync API)",
                interactive=False,
            )

    with gr.Tabs():
        with gr.Tab("🤖 Heuristic Auto-Play"):
            gr.Markdown(
                "Greedy playthrough: compress days toward the liquidity threshold; "
                "hard task enables dynamic discounting at 8%."
            )
            with gr.Row():
                h_task = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0], label="Task")
                h_seed = gr.Number(value=42, label="Seed", minimum=0, maximum=99999, precision=0)
            h_run_btn = gr.Button("Run Heuristic Playthrough", variant="primary")
            h_status = gr.Markdown("")
            h_score = gr.Textbox(label="Score bar", interactive=False, elem_classes="score-box")
            h_obs_json = gr.Code(language="json", label="Final observation JSON", value="{}")
            h_transcript = gr.Textbox(label="Transcript", lines=12, interactive=False)

            h_run_btn.click(
                fn=heuristic_play,
                inputs=[h_task, h_seed],
                outputs=[h_obs_json, h_status, h_transcript, h_score],
            )

        with gr.Tab("🧮 Grader Calculator"):
            gr.Markdown("Terminal score for a hypothetical outcome (no episode run).")
            with gr.Row():
                with gr.Column():
                    gc_task = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0], label="Task")
                    gc_deal = gr.Checkbox(label="Deal reached", value=True)
                    gc_days = gr.Slider(minimum=0, maximum=120, step=1, value=55, label="Agreed payment days")
                    gc_price = gr.Slider(minimum=70.0, maximum=110.0, step=0.5, value=100.0, label="Agreed price (₹/unit)")
                    gc_clause = gr.Checkbox(label="Late payment penalty clause agreed", value=False)
                    gc_dd = gr.Checkbox(label="Dynamic discounting agreed", value=False)
                    gc_dd_rate = gr.Slider(minimum=0.0, maximum=0.30, step=0.01, value=0.08, label="Dynamic discount annual rate")
                    gc_compute = gr.Button("Calculate Score", variant="primary")
                with gr.Column():
                    gc_result = gr.Markdown("*Fill in the form and click Calculate.*")

            gc_compute.click(
                fn=compute_grader_score,
                inputs=[gc_task, gc_days, gc_price, gc_deal, gc_clause, gc_dd, gc_dd_rate],
                outputs=[gc_result],
            )

        with gr.Tab("📖 Reference"):
            gr.Markdown("""
| Field | Meaning |
|---|---|
| `buyer_price` | Buyer's current offer (₹/unit) |
| `buyer_days` | Buyer's proposed settlement period (days) |
| `round_number` | Current negotiation round |
| `step_reward` / `reward` | Shaping or terminal reward signal |

**Actions:** `propose`, `accept`, `reject`, plus planning/tool modes in the Python API.

- [GitHub](https://github.com/SkandaGanesha1/ENV)
- [HF Space](https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator)
""")

    gr.HTML("""
    <div class="footer" style="text-align:center;opacity:0.6;font-size:0.8rem;margin-top:16px;">
      OpenEnv SME Negotiator · MIT License ·
      <a href="https://github.com/SkandaGanesha1/ENV" target="_blank">GitHub</a>
    </div>
    """)

    reset_outputs = [
        session_state,
        status_bar,
        task_info,
        chat,
        last_json,
        price_in,
        days_in,
        submit_btn,
    ]
    reset_btn.click(
        fn=reset_episode,
        inputs=[task_sel, seed_num, session_state],
        outputs=reset_outputs,
    )

    submit_outputs = [
        session_state,
        status_bar,
        chat,
        last_json,
        price_in,
        days_in,
        submit_btn,
    ]
    submit_btn.click(
        fn=submit_step,
        inputs=[
            ui_action,
            price_in,
            days_in,
            reason_in,
            deal_id_in,
            use_treds,
            simulation_plan_tb,
            simulation_horizon_n,
            late_clause,
            propose_dd,
            dd_rate,
            session_state,
        ],
        outputs=submit_outputs,
    )


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
                theme=THEME,
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
