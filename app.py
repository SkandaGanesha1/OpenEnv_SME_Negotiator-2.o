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

# ── Backend modules ────────────────────────────────────────────────────────────
from action_handler import ActionHandler
from reward_engine import RewardEngine
from session_store import SessionStore
from step_logger import StepLogger

_action_handler = ActionHandler()
_reward_engine = RewardEngine()
_logger = StepLogger()

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

TRAINING_AND_JUDGE_MARKDOWN = r"""
### Two modes, one environment

| Mode | What happens |
|---|---|
| **This Playground** | You drive the environment by hand. **Reset Episode** loads a scenario; **Submit Step** sends a `NegotiationAction`. Each step returns an observation with `reward` and `done`. |
| **RL Training** | The *same* `SMENegotiatorEnvironment` runs inside a GRPO training loop: policy proposes actions → env returns rewards → trainer updates weights. Entry points: `rl/train_grpo_trl.py`, `rl/train_grpo_unsloth.py`, notebook `notebooks/colab_grpo_sme_liquidity.ipynb`. |

### Reward (two layers)

**Per-step shaping** — emitted every `env.step()`. Encodes: liquidity pressure, days improvement, buyer reaction, solvency proximity. Bounded to `[−1, 1]`.

**Terminal benchmark score** — deterministic graders in `sme_negotiator_env/graders.py` map the final `NegotiationState` to a score in `[0, 1]`:

```
score = 0.35 × solvency + 0.20 × liquidity + 0.35 × NPV + 0.10 × compliance
```

The **Grader Calculator** tab lets you compute this for any hypothetical outcome.

### Task rubric

| Task | What the agent must do | Extra levers |
|---|---|---|
| **Easy** | Compress payment days to ≤ 60 | Price + Days |
| **Medium** | Tighten to ≤ 45 days + add late-payment penalty clause | + Clause |
| **Hard** | Negotiate dynamic discounting with a coherent annual rate | + TReDS + DD rate |

### For LLM judges / offline policy training

- `compute_verifiable_reward` in `sme_negotiator_env/graders.py` gives RLVR-style decomposed reward without touching legacy terminal scores.
- `rl/judge_pack.py` bundles evaluation artifacts; `EVALUATION.md` documents full benchmark methodology.

*All prices are INR per unit (₹/unit), matching the simulator.*
"""


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
    pct = max(0.0, min(1.0, float(score))) * 100
    cls = "positive" if score >= 0.7 else ("zero" if score >= 0.4 else "negative")
    color = {"positive": "#22C55E", "zero": "#F59E0B", "negative": "#EF4444"}[cls]
    return (
        f'<div class="reward-panel" style="padding:10px 14px">'
        f'<div class="rp-row">'
        f'<span class="rp-label">Score</span>'
        f'<div class="rp-bar-wrap"><div class="rp-bar {cls}" style="width:{pct:.1f}%"></div></div>'
        f'<span class="rp-val {cls}">{score:.4f}</span>'
        f'</div>'
        f'<div class="rp-footer" style="color:{color}">{"Excellent" if score >= 0.8 else "Good" if score >= 0.6 else "Needs improvement"}</div>'
        f'</div>'
    )


def _scenario_html(cfg: "TaskConfig", obs: dict) -> str:
    """Rich scenario context card shown after Reset Episode."""
    liq = int(obs.get("liquidity_threshold", cfg.liquidity_threshold))
    b_price = float(obs.get("buyer_price", cfg.initial_buyer_price))
    b_days = int(obs.get("buyer_days", cfg.initial_buyer_days))
    volume = int(obs.get("volume", cfg.volume))
    revenue = float(getattr(cfg, "sme_monthly_revenue", 500000))
    interest = float(getattr(cfg, "interest_rate_annual", 0.22))
    bp_score = float(getattr(cfg, "buyer_power_score", 0.6))
    supplier_days = int(getattr(cfg, "sme_supplier_payment_days", 30))

    difficulty_meta = {
        "easy":   ("🟢", "Easy", "#22C55E", "Compress days → ≤60. Single-lever negotiation."),
        "medium": ("🟡", "Medium", "#F5A623", "Days ≤ 45 + late-payment penalty clause."),
        "hard":   ("🔴", "Hard", "#EF4444", "Dynamic discounting + TReDS. Multi-lever economics."),
    }
    diff = str(cfg.difficulty).lower()
    icon, label, color, goal = difficulty_meta.get(diff, ("⚪", diff, "#7B8BAA", ""))

    gap_days = max(0, b_days - supplier_days)
    working_capital = (revenue / 30) * gap_days
    interest_cost = working_capital * (interest / 365) * gap_days

    return f"""
<div class="scenario-card">
  <div class="sc-header">
    <div class="sc-title">
      <span class="sc-badge" style="background:rgba({_hex_to_rgb(color)},0.12);color:{color};border-color:rgba({_hex_to_rgb(color)},0.3)">{icon} {label}</span>
      <span class="sc-name">{cfg.description}</span>
    </div>
    <div class="sc-goal">🎯 {goal}</div>
  </div>
  <div class="sc-grid">
    <div class="sc-metric">
      <div class="sc-m-label">Buyer Opening Offer</div>
      <div class="sc-m-value">₹{b_price:.0f}/unit · {b_days}d</div>
    </div>
    <div class="sc-metric">
      <div class="sc-m-label">Liquidity Threshold</div>
      <div class="sc-m-value danger">{liq} days</div>
      <div class="sc-m-sub">SME cash-flow limit</div>
    </div>
    <div class="sc-metric">
      <div class="sc-m-label">Working Capital Gap</div>
      <div class="sc-m-value">₹{working_capital/1000:.0f}k</div>
      <div class="sc-m-sub">{gap_days}d gap · {interest*100:.0f}%/yr cost</div>
    </div>
    <div class="sc-metric">
      <div class="sc-m-label">Order Volume</div>
      <div class="sc-m-value">{volume:,} units</div>
      <div class="sc-m-sub">₹{b_price*volume/1000:.0f}k deal value</div>
    </div>
    <div class="sc-metric">
      <div class="sc-m-label">Buyer Power</div>
      <div class="sc-m-value">{bp_score:.1f}/1.0</div>
      <div class="sc-m-sub">Negotiating leverage</div>
    </div>
    <div class="sc-metric">
      <div class="sc-m-label">SME Revenue</div>
      <div class="sc-m-value">₹{revenue/1000:.0f}k/mo</div>
      <div class="sc-m-sub">Interest cost: ₹{interest_cost:.0f}/episode</div>
    </div>
  </div>
  <div class="sc-law">
    ⚖️ <strong>MSMED Act § 43B(h):</strong> Buyers lose tax deduction if MSME payment exceeds <strong>45 days</strong>.
    &nbsp;·&nbsp; <strong>MSME Samadhaan:</strong> compound interest at 3× RBI rate on overdue payments.
  </div>
</div>
"""


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def _reward_html(step_reward: float, cum_reward: float, step_num: int) -> str:
    """Reward breakdown panel shown after each Submit Step."""
    if step_num == 0:
        return """
<div class="reward-panel rp-idle">
  <div class="rp-idle-body">
    <span class="rp-idle-icon">📈</span>
    <span class="rp-idle-msg">Reward signal will appear here after your first <strong>Submit Step</strong></span>
  </div>
  <div class="rp-footer">Grading: 0.35 × solvency + 0.20 × liquidity + 0.35 × NPV + 0.10 × compliance</div>
</div>
"""
    step_pct = min(100, max(0, int(abs(step_reward) * 100)))
    cum_pct  = min(100, max(0, int(abs(cum_reward) / max(1, step_num) * 100)))
    step_cls = "positive" if step_reward >= 0 else "negative"
    cum_cls  = "positive" if cum_reward  >= 0 else "negative"
    step_sign = "▲" if step_reward >= 0 else "▼"
    cum_sign  = "▲" if cum_reward  >= 0 else "▼"
    return f"""
<div class="reward-panel">
  <div class="rp-row">
    <span class="rp-label">Step reward</span>
    <div class="rp-bar-wrap">
      <div class="rp-bar {step_cls}" style="width:{step_pct}%"></div>
    </div>
    <span class="rp-val {step_cls}">{step_sign} {step_reward:+.4f}</span>
  </div>
  <div class="rp-row">
    <span class="rp-label">Cumulative</span>
    <div class="rp-bar-wrap">
      <div class="rp-bar {cum_cls}" style="width:{cum_pct}%"></div>
    </div>
    <span class="rp-val {cum_cls}">{cum_sign} {cum_reward:+.4f}</span>
  </div>
  <div class="rp-footer">Step {step_num} &nbsp;·&nbsp; Grading: 35% solvency + 20% liquidity + 35% NPV + 10% compliance</div>
</div>
"""


def _status_html(
    *,
    round_number: int,
    last_price: float,
    done: bool,
    reward: float = 0.0,
    buyer_accepted: bool = False,
    max_rounds: int = 10,
) -> str:
    """Status bar with reward pill, progress rail, and deal outcome card."""
    if done:
        badge, badge_cls = "Done", "done"
    elif round_number == 0:
        badge, badge_cls = "Ready", "ready"
    else:
        badge, badge_cls = "Active", "active"
    price_str = f"₹{float(last_price):.2f}" if last_price else "—"

    # Reward pill
    if reward > 0:
        pill = f'<span class="reward-pill positive">▲ {reward:+.4f}</span>'
    elif reward < 0:
        pill = f'<span class="reward-pill negative">▼ {reward:+.4f}</span>'
    elif round_number > 0:
        pill = f'<span class="reward-pill zero">± 0.0000</span>'
    else:
        pill = ""

    # Progress rail
    pct = min(100, int(round_number / max(max_rounds, 1) * 100))
    fill_cls = "progress-fill done" if done else "progress-fill"
    progress = f'<div class="progress-rail"><div class="{fill_cls}" style="width:{pct}%"></div></div>'

    # Deal outcome card
    deal_card = ""
    if done:
        if buyer_accepted:
            deal_card = (
                f'<div class="deal-card accepted">'
                f'<span class="deal-icon">✅</span>'
                f'<strong>Deal Accepted</strong>'
                f'<span class="deal-meta">{price_str}/unit · {int(round_number)} rounds</span>'
                f'</div>'
            )
        else:
            deal_card = (
                f'<div class="deal-card rejected">'
                f'<span class="deal-icon">❌</span>'
                f'<strong>No Deal</strong>'
                f'<span class="deal-meta">Episode ended · {int(round_number)} rounds</span>'
                f'</div>'
            )

    return f"""
<div class="status-bar">
  <span class="badge {badge_cls}">{badge}</span>
  <span class="sep">·</span>
  <span>Round <strong>{int(round_number)}</strong></span>
  <span class="sep">·</span>
  <span>Buyer offer <strong>{price_str}/unit</strong></span>
  {pill}
</div>
{progress}
{deal_card}
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
    state: SessionStore,
) -> tuple[Any, ...]:
    """New episode: fresh env, empty chat/JSON, status + scenario card + price/days from first obs."""
    try:
        env, obs = _make_env_and_reset(task_label, int(seed))
        task_id = TASKS[task_label]
        cfg: TaskConfig = TASK_REGISTRY[task_id]

        store = SessionStore()
        ep_id = store.start_episode(env=env, task_label=task_label, seed=int(seed))
        _logger.log_reset(episode_id=ep_id, task_label=task_label, seed=int(seed))

        p_price = float(obs.get("buyer_price", 100.0))
        p_days = int(obs.get("buyer_days", 90))
        last_price = float(obs.get("buyer_price", 0.0))
        rnd = int(obs.get("round_number", 0))
        done0 = bool(obs.get("done", obs.get("negotiation_done", False)))

        status_html = _status_html(round_number=rnd, last_price=last_price, done=done0, reward=0.0)
        scenario_html = _scenario_html(cfg, obs)
        reward_html = _reward_html(0.0, 0.0, 0)
        json_payload = _build_last_step_json(obs)

        return (
            store,
            status_html,
            scenario_html,
            reward_html,
            [],
            json_payload,
            p_price,
            p_days,
            gr.update(interactive=True),
        )
    except Exception as exc:
        err = str(exc)
        store = SessionStore()
        status_html = f'<div class="status-bar error">⚠ Reset failed: {err}</div>'
        empty_scenario = "<div class='scenario-card error-card'>⚠ Could not load scenario. Check task config.</div>"
        return (
            store,
            status_html,
            empty_scenario,
            _reward_html(0.0, 0.0, 0),
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
    state: SessionStore,
) -> tuple[Any, ...]:
    store: SessionStore = state if isinstance(state, SessionStore) else SessionStore()
    env: Optional[SMENegotiatorEnvironment] = store.env
    messages: list = list(store.messages or [])

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
            store,
            _status_html(round_number=0, last_price=0.0, done=False),
            _reward_html(0.0, 0.0, 0),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
            gr.update(),
            gr.update(),
        )

    wired = _action_handler.map_ui_action(ui_action or "propose")

    # ── Pre-flight validation ────────────────────────────────────────────────
    ok, val_err = _action_handler.validate(
        price=float(price),
        payment_days=int(payment_days),
        use_treds=bool(use_treds),
        propose_dynamic_discounting=bool(propose_dd),
        dynamic_discount_annual_rate=float(dd_rate),
        action_type=wired,
    )
    if not ok:
        user_t = _format_user_turn(
            ui_action=ui_action or "", wired_action=wired,
            price=float(price), days=int(payment_days),
            reason=reason or "", deal_id=deal_id or "",
            use_treds=bool(use_treds), sim_plan_raw=simulation_plan_raw or "",
            horizon=int(simulation_horizon) if simulation_horizon else None,
        )
        err_msg = _action_handler.format_error(val_err)
        messages = messages + [[user_t, err_msg]]
        _logger.log_validation_error(
            episode_id=store.episode_id, step=store.step_num,
            error=val_err, action_type=wired,
            price=float(price), payment_days=int(payment_days),
        )
        return (
            store, gr.update(), gr.update(), messages,
            {"reward": 0.0, "done": False, "info": {"message": val_err}},
            gr.update(), gr.update(), gr.update(),
        )

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
            store,
            gr.update(),
            gr.update(),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
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
        return (store, gr.update(), gr.update(), messages, {"reward": 0.0, "done": False, "info": {"message": err}}, gr.update(), gr.update(), gr.update())

    done = bool(obs_dict.get("done", obs_dict.get("negotiation_done", False)))
    step_reward = float(obs_dict.get("reward", obs_dict.get("step_reward", 0.0)))

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

    # ── Update session store ─────────────────────────────────────────────────
    store.record_step(
        user_message=user_t,
        assistant_message=assistant_t,
        reward=step_reward,
        done=done,
        obs_dict=obs_dict,
    )
    _logger.log_step(
        episode_id=store.episode_id,
        step=store.step_num,
        action_type=wired,
        price=float(price),
        payment_days=int(payment_days),
        use_treds=bool(use_treds),
        reward=step_reward,
        cum_reward=store.cum_rew,
        done=done,
        buyer_price=obs_dict.get("buyer_price"),
        buyer_days=obs_dict.get("buyer_days"),
        round_number=obs_dict.get("round_number"),
    )
    if done:
        _logger.log_episode_end(
            episode_id=store.episode_id,
            task_label=store.task_label,
            steps=store.step_num,
            total_reward=store.cum_rew,
        )

    last_price = float(obs_dict.get("buyer_price", 0.0))
    rnd = int(obs_dict.get("round_number", 0))
    accepted = bool(obs_dict.get("buyer_accepted", False))
    status_html = _status_html(
        round_number=rnd, last_price=last_price, done=done,
        reward=step_reward, buyer_accepted=accepted,
    )
    rwd_html = _reward_html(step_reward, store.cum_rew, store.step_num)
    json_payload = _build_last_step_json(obs_dict)

    return (
        store,
        status_html,
        rwd_html,
        store.messages,
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@400;600;700&display=swap');

/* ══ TOKENS ══════════════════════════════════════════════════════════════ */
:root {
  --bg:      #0B1120;
  --s1:      #111827;
  --s2:      #1E293B;
  --s3:      #0F172A;
  --border:  #334155;
  --txt:     #E2E8F0;
  --txt2:    #94A3B8;
  --txt3:    #64748B;
  --accent:  #3B82F6;
  --success: #22C55E;
  --danger:  #EF4444;
  --amber:   #F59E0B;
  --r:       8px;
  --rl:      12px;
}

/* ══ BASE ════════════════════════════════════════════════════════════════ */
body, .gradio-container { background: var(--bg) !important; font-family: 'Sora', ui-sans-serif, sans-serif !important; }
.gradio-container { max-width: 1440px !important; padding: 14px 18px !important; background-image: radial-gradient(circle, #1A2640 1px, transparent 1px) !important; background-size: 28px 28px !important; }
.gr-panel, .gr-box, .block { background: transparent !important; border: none !important; box-shadow: none !important; }
pre, code, .cm-editor, .cm-content, .cm-line { font-family: 'JetBrains Mono', monospace !important; }

/* ══ HEADER BOX ══════════════════════════════════════════════════════════ */
.header-box {
  display: flex; gap: 28px; align-items: flex-start;
  background: linear-gradient(135deg, #060D1F 0%, #0B1735 60%, #0F2050 100%);
  border: 1px solid #1A3060; border-radius: var(--rl);
  padding: 28px 32px; margin-bottom: 16px;
  position: relative; overflow: hidden; transition: background 0.8s ease;
}
.header-box::before {
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(ellipse at 70% 40%, rgba(59,130,246,.12) 0%, transparent 55%),
              radial-gradient(ellipse at 20% 80%, rgba(167,139,250,.07) 0%, transparent 45%);
}
.header-box.deal-accepted { background: linear-gradient(135deg, #071F12, #0D3B22) !important; border-color: rgba(34,197,94,.35) !important; }
.header-box.deal-rejected { background: linear-gradient(135deg, #1A0707, #3B0D0D) !important; border-color: rgba(239,68,68,.3)  !important; }
.hb-left { flex: 1; }
.hb-right { flex-shrink: 0; width: 310px; display: flex; flex-direction: column; gap: 14px; }
.hb-eyebrow { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin-bottom: 6px; }
.hb-title { font-size: 1.85rem; font-weight: 700; color: #fff; margin: 0 0 8px; letter-spacing: -0.02em; }
.hb-sub { font-size: 0.875rem; color: #8AABDA; line-height: 1.65; margin: 0 0 14px; max-width: 540px; }
.hb-sub strong { color: #B8D4F5; }
.hb-chips { display: flex; flex-wrap: wrap; gap: 7px; }
.chip { display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }
.chip-blue  { background: rgba(59,130,246,.14);  color: #7EB6FF; border: 1px solid rgba(59,130,246,.3); }
.chip-amber { background: rgba(245,158,11,.12);  color: #FBD38D; border: 1px solid rgba(245,158,11,.3); }
.chip-green { background: rgba(34,197,94,.10);   color: #86EFAC; border: 1px solid rgba(34,197,94,.3); }
.chip-gray  { background: rgba(148,163,184,.08); color: var(--txt2); border: 1px solid var(--border); }

/* step flow */
.hb-flow { display: flex; align-items: center; gap: 4px; }
.hf-step { display: flex; flex-direction: column; align-items: center; gap: 3px; background: var(--s1); border: 1px solid var(--border); border-radius: var(--r); padding: 8px 12px; min-width: 58px; transition: border-color .2s, background .2s; }
.hf-step.active { background: rgba(59,130,246,.15); border-color: var(--accent); }
.hf-step.done   { background: rgba(34,197,94,.08);  border-color: rgba(34,197,94,.35); }
.hf-num { font-size: 0.72rem; font-weight: 700; color: var(--txt3); font-family: 'JetBrains Mono', monospace; }
.hf-step.active .hf-num { color: var(--accent); }
.hf-step.done   .hf-num { color: var(--success); }
.hf-label { font-size: 0.6rem; font-weight: 600; color: var(--txt2); text-align: center; }
.hf-arrow { color: var(--border); font-size: 0.9rem; }

/* reward formula card */
.hb-reward-formula { background: rgba(59,130,246,.06); border: 1px solid rgba(59,130,246,.18); border-radius: var(--r); padding: 10px 14px; }
.rf-title { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--txt2); margin-bottom: 8px; }
.rf-body  { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.rf-term  { font-size: 0.68rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }
.rf-term.solvency   { background: rgba(34,197,94,.14);   color: #86EFAC; }
.rf-term.liquidity  { background: rgba(59,130,246,.14);  color: #93C5FD; }
.rf-term.npv        { background: rgba(167,139,250,.14); color: #C4B5FD; }
.rf-term.compliance { background: rgba(245,158,11,.12);  color: #FBD38D; }
.rf-plus { color: var(--txt3); font-size: 0.85rem; }

/* ══ PANEL LABELS & DIVIDERS ══════════════════════════════════════════════ */
.panel-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--txt3); padding: 0 0 10px; margin-bottom: 6px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 6px; }
.card-head   { font-size: 0.75rem; font-weight: 700; color: var(--txt2); margin: 12px 0 8px; padding: 0 0 6px; border-bottom: 1px solid var(--border); }
.divider     { height: 1px; background: var(--border); margin: 12px 0; }
.section-divider { display: flex; align-items: center; gap: 10px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--txt3); margin: 14px 0 8px; }
.section-divider::before, .section-divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.tabs-head   { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--txt3); padding: 16px 2px 8px; border-top: 1px solid var(--border); margin-top: 4px; }
.unit-hint   { font-size: 0.68rem; color: var(--txt3); font-family: 'JetBrains Mono', monospace; padding: 3px 0 8px; }

/* ══ STATUS BAR ══════════════════════════════════════════════════════════ */
.status-bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 10px 16px; background: var(--s2); border: 1px solid var(--border); border-radius: var(--r); font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: var(--txt) !important; margin-bottom: 4px; }
.status-bar span, .status-bar strong { color: var(--txt) !important; }
.status-bar .sep { color: var(--txt3); }
.status-bar.error { background: rgba(239,68,68,.07); border-color: rgba(239,68,68,.3); color: #FCA5A5 !important; }
.badge { padding: 3px 10px; border-radius: 999px; font-size: 0.68rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.badge.ready  { background: rgba(148,163,184,.10); color: var(--txt2); border: 1px solid var(--border); }
.badge.active { background: rgba(59,130,246,.15);  color: #93C5FD;    border: 1px solid rgba(59,130,246,.4); }
.badge.done   { background: rgba(34,197,94,.12);   color: #86EFAC;    border: 1px solid rgba(34,197,94,.35); }
.reward-pill { padding: 2px 9px; border-radius: 999px; font-weight: 700; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; }
.reward-pill.positive { background: rgba(34,197,94,.14);   color: #86EFAC; border: 1px solid rgba(34,197,94,.3); }
.reward-pill.negative { background: rgba(239,68,68,.12);   color: #FCA5A5; border: 1px solid rgba(239,68,68,.28); }
.reward-pill.zero     { background: rgba(148,163,184,.08); color: var(--txt2); border: 1px solid var(--border); }
.progress-rail { height: 4px; background: var(--s3); border-radius: 2px; margin: 4px 0 6px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 2px; transition: width .45s ease; background: linear-gradient(90deg, var(--accent), #14B8A6); min-width: 4px; }
.progress-fill.done { background: linear-gradient(90deg, var(--success), #14B8A6); }
@keyframes deal-in { from { opacity:0; transform:translateY(-5px); } to { opacity:1; transform:translateY(0); } }
.deal-card { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: var(--r); margin-top: 6px; font-size: 0.84rem; animation: deal-in .3s ease; }
.deal-card.accepted { background: rgba(34,197,94,.07); border: 1px solid rgba(34,197,94,.25); border-left: 3px solid var(--success); }
.deal-card.rejected { background: rgba(239,68,68,.07); border: 1px solid rgba(239,68,68,.22); border-left: 3px solid var(--danger); }
.deal-card .deal-icon { font-size: 1.1rem; }
.deal-card strong { color: var(--txt); font-weight: 700; }
.deal-card .deal-meta { color: var(--txt2); font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; margin-left: auto; }

/* ══ SCENARIO CARD ════════════════════════════════════════════════════════ */
.scenario-card { background: var(--s1); border: 1px solid var(--border); border-radius: var(--rl); padding: 16px 20px; margin-bottom: 10px; }
.scenario-card.empty-scenario { text-align: center; padding: 28px 20px; background: linear-gradient(135deg, var(--s1), var(--s2)); border-style: dashed; }
.scenario-card.error-card { border-color: rgba(239,68,68,.3); background: rgba(239,68,68,.04); }
.es-icon  { font-size: 2rem; margin-bottom: 8px; }
.es-title { font-size: 1rem; font-weight: 700; color: var(--txt); margin-bottom: 4px; }
.es-sub   { font-size: 0.8rem; color: var(--txt2); line-height: 1.65; }
.es-sub strong { color: var(--accent); }
.sc-header { margin-bottom: 12px; }
.sc-title  { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.sc-badge  { font-size: 0.68rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; border: 1px solid; }
.sc-name   { font-size: 0.88rem; font-weight: 600; color: var(--txt); }
.sc-goal   { font-size: 0.75rem; color: var(--txt2); }
.sc-grid   { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
.sc-metric { background: var(--s2); border: 1px solid var(--border); border-radius: var(--r); padding: 10px 12px; transition: border-color .18s, transform .15s; cursor: default; }
.sc-metric:hover { border-color: #475569 !important; transform: translateY(-1px); }
.sc-m-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--txt3); margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
.sc-m-value { font-size: 1rem; font-weight: 700; color: var(--txt); font-family: 'JetBrains Mono', monospace; }
.sc-m-value.danger { color: #FCA5A5; }
.sc-m-sub   { font-size: 0.62rem; color: var(--txt3); margin-top: 2px; }
.sc-law { background: rgba(245,158,11,.06); border: 1px solid rgba(245,158,11,.2); border-radius: var(--r); padding: 8px 12px; font-size: 0.73rem; color: #D4A744; line-height: 1.5; }
.sc-law strong { color: #FBD38D; }

/* ══ REWARD PANEL ════════════════════════════════════════════════════════ */
.reward-panel { background: var(--s1); border: 1px solid var(--border); border-radius: var(--r); padding: 12px 14px; }
.rp-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.rp-label { font-size: 0.65rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--txt2); width: 76px; flex-shrink: 0; font-family: 'JetBrains Mono', monospace; }
.rp-bar-wrap { flex: 1; background: var(--s3); border-radius: 4px; height: 8px; overflow: hidden; }
.rp-bar { height: 100%; border-radius: 4px; transition: width .45s ease; min-width: 3px; }
.rp-bar.positive { background: linear-gradient(90deg, #16A34A, var(--success)); }
.rp-bar.negative { background: linear-gradient(90deg, #DC2626, var(--danger)); }
.rp-bar.zero     { background: var(--border); }
.rp-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.72rem; width: 72px; text-align: right; }
.rp-val.positive { color: #86EFAC; }
.rp-val.negative { color: #FCA5A5; }
.rp-val.zero     { color: var(--txt2); }
.rp-footer { font-size: 0.63rem; color: var(--txt3); padding-top: 6px; border-top: 1px solid var(--border); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.rp-idle-body { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 14px 0 12px; }
.rp-idle-icon { font-size: 1.2rem; }
.rp-idle-msg  { font-size: 0.75rem; color: var(--txt2); line-height: 1.5; }
.rp-idle-msg strong { color: var(--txt); }

/* ══ COMPARE STRIP ════════════════════════════════════════════════════════ */
.compare-strip { display: flex; align-items: center; justify-content: space-between; padding: 7px 12px; margin: 6px 0; border-radius: var(--r); background: rgba(59,130,246,.04); border: 1px solid rgba(59,130,246,.12); font-size: 0.7rem; font-weight: 600; }
.cs-side { display: flex; align-items: center; gap: 5px; }
.cs-dot  { width: 6px; height: 6px; border-radius: 50%; }
.cs-side.you   .cs-dot { background: var(--accent); }
.cs-side.you            { color: #93C5FD; }
.cs-side.buyer .cs-dot { background: var(--amber); }
.cs-side.buyer           { color: #FBD38D; }
.cs-arrow { color: var(--txt3); }

/* ══ INPUTS ══════════════════════════════════════════════════════════════ */
.gradio-container input[type="number"], .gradio-container input[type="text"], .gradio-container textarea, .gradio-container select { background: var(--s2) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; color: var(--txt) !important; font-size: 0.875rem !important; }
.gradio-container input:focus-visible, .gradio-container textarea:focus-visible { outline: none !important; border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(59,130,246,.18) !important; }
.gradio-container label { color: var(--txt2) !important; font-size: 0.8rem !important; }
.gradio-container .wrap { background: var(--s2) !important; border-color: var(--border) !important; }

/* ══ ACCORDIONS ══════════════════════════════════════════════════════════ */
.gr-accordion, details { background: var(--s2) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; margin-bottom: 6px !important; }
.gr-accordion summary, details > summary { padding: 9px 14px !important; font-size: 0.82rem !important; font-weight: 600 !important; color: var(--txt) !important; }
.about-accordion.gr-accordion, .about-accordion details { background: rgba(59,130,246,.03) !important; border-color: rgba(59,130,246,.18) !important; }

/* ══ BUTTONS ══════════════════════════════════════════════════════════════ */
button.btn-reset, .btn-reset button { background: transparent !important; border: 1px solid var(--accent) !important; color: var(--accent) !important; font-weight: 600 !important; border-radius: var(--r) !important; transition: background .15s !important; }
button.btn-reset:hover, .btn-reset button:hover { background: rgba(59,130,246,.08) !important; }
button.btn-submit, .btn-submit button { background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important; border: none !important; color: #fff !important; font-weight: 700 !important; font-size: 0.95rem !important; border-radius: var(--r) !important; box-shadow: 0 2px 18px rgba(59,130,246,.35), inset 0 1px 0 rgba(255,255,255,.1) !important; transition: box-shadow .2s, transform .12s !important; }
button.btn-submit:hover:not(:disabled), .btn-submit button:hover:not(:disabled) { box-shadow: 0 4px 28px rgba(59,130,246,.55), inset 0 1px 0 rgba(255,255,255,.1) !important; transform: translateY(-1px) !important; }
button.btn-submit:disabled, .btn-submit button:disabled { background: var(--s2) !important; border: 1px solid var(--border) !important; color: var(--txt3) !important; box-shadow: none !important; }
button:focus-visible { outline: none !important; box-shadow: 0 0 0 3px rgba(59,130,246,.25) !important; }
.gr-button-primary { background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important; border: none !important; color: #fff !important; }

/* ══ TABS ════════════════════════════════════════════════════════════════ */
.bottom-tabs.gradio-tabs, .gradio-container .tabs { background: var(--s1) !important; border: 1px solid var(--border) !important; border-radius: var(--rl) !important; }
.gradio-container .tab-nav { border-bottom: 1px solid var(--border) !important; padding: 0 12px !important; }
.gradio-container .tab-nav button { font-size: 0.82rem !important; font-weight: 600 !important; color: var(--txt2) !important; padding: 10px 16px !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; transition: color .15s, border-color .15s !important; }
.gradio-container .tab-nav button:hover { color: var(--txt) !important; }
.gradio-container .tab-nav button.selected { color: var(--txt) !important; border-bottom-color: var(--accent) !important; }
.tab-intro { font-size: 0.8rem; color: var(--txt2); background: var(--s2); border: 1px solid var(--border); border-radius: var(--r); padding: 10px 14px; margin-bottom: 12px; line-height: 1.6; }
.tab-intro strong { color: var(--txt); }
@keyframes fade-in { from { opacity:0; } to { opacity:1; } }
.tabitem { animation: fade-in .18s ease; }

/* ══ CHATBOT / JSON / CODE ════════════════════════════════════════════════ */
.gr-chatbot, .chatbot { background: var(--s2) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }
.gr-chatbot .user, .chatbot .user { background: rgba(59,130,246,.08) !important; border: 1px solid rgba(59,130,246,.18) !important; }
.gr-chatbot .bot,  .chatbot .bot  { background: var(--s1) !important; border: 1px solid var(--border) !important; }
.gr-json, .json-holder { background: var(--s3) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; font-family: 'JetBrains Mono', monospace !important; }
.gr-json:not(:empty) { border-left: 3px solid var(--accent) !important; }
.gr-code, .code-wrap { background: var(--s3) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }

/* ══ MISC ════════════════════════════════════════════════════════════════ */
.footer { display: flex; align-items: center; justify-content: center; gap: 8px; padding: 14px; border-top: 1px solid var(--border); color: var(--txt3); font-size: 0.73rem; margin-top: 8px; }
.footer a { color: var(--txt2); text-decoration: none; }
.footer a:hover { color: var(--accent); }
.ft-sep { color: var(--border); }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
@keyframes fade-up { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes msg-in  { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

/* ══ MOBILE ══════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
  .header-box { flex-direction: column; padding: 18px; }
  .hb-right { width: 100%; }
  .hb-title { font-size: 1.45rem; }
  .sc-grid { grid-template-columns: repeat(2, 1fr); }
  .gradio-container { padding: 0 8px !important; }
  button.btn-submit, .btn-submit button { position: sticky !important; bottom: 10px !important; z-index: 100 !important; }
}
"""


THEME = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(title="OpenEnv SME Negotiator", css=CUSTOM_CSS) as demo:

    # ── Global header ────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box" id="sme-header">
      <div class="hb-left">
        <div class="hb-eyebrow">OpenEnv · RL Benchmark</div>
        <h1 class="hb-title">🤝 SME Negotiator</h1>
        <p class="hb-sub">
          An Indian SME negotiates payment terms against a scripted buyer.
          Shorter receivables = better cash-flow. The agent must balance
          <strong>price</strong>, <strong>payment days</strong>, <strong>legal compliance</strong>
          and optionally <strong>TReDS financing</strong> to keep the business solvent.
        </p>
        <div class="hb-chips">
          <span class="chip chip-blue">64M+ SMEs in India</span>
          <span class="chip chip-amber">₹8.1L cr stuck in late payments</span>
          <span class="chip chip-green">MSMED Act: 45-day legal limit</span>
          <span class="chip chip-gray">TRL GRPO · Unsloth · RLVR</span>
        </div>
      </div>
      <div class="hb-right">
        <div class="hb-flow">
          <div class="hf-step" id="sme-step1"><span class="hf-num">1</span><span class="hf-label">Pick Task</span></div>
          <div class="hf-arrow">→</div>
          <div class="hf-step" id="sme-step2"><span class="hf-num">2</span><span class="hf-label">Reset</span></div>
          <div class="hf-arrow">→</div>
          <div class="hf-step" id="sme-step3"><span class="hf-num">3</span><span class="hf-label">Propose</span></div>
          <div class="hf-arrow">→</div>
          <div class="hf-step" id="sme-step4"><span class="hf-num">4</span><span class="hf-label">Submit</span></div>
        </div>
        <div class="hb-reward-formula">
          <div class="rf-title">Benchmark Score Formula</div>
          <div class="rf-body">
            <span class="rf-term solvency">0.35 × Solvency</span>
            <span class="rf-plus">+</span>
            <span class="rf-term liquidity">0.20 × Liquidity</span>
            <span class="rf-plus">+</span>
            <span class="rf-term npv">0.35 × NPV</span>
            <span class="rf-plus">+</span>
            <span class="rf-term compliance">0.10 × Compliance</span>
          </div>
        </div>
      </div>
    </div>
    """)

    session_state = gr.State(SessionStore())

    with gr.Row(equal_height=False):
        # ── LEFT: Control Panel ──────────────────────────────────────────
        with gr.Column(scale=4, min_width=320):
            gr.HTML("<div class='panel-label'>⚙ Control Panel</div>")

            # Episode config card
            gr.HTML("<div class='card-head'>Episode Configuration</div>")
            task_sel = gr.Dropdown(
                choices=list(TASKS.keys()),
                value=list(TASKS.keys())[0],
                label="Task difficulty",
                info="Easy → Medium → Hard adds more negotiation levers.",
            )
            seed_num = gr.Number(value=42, label="Random seed", precision=0, minimum=0, maximum=99999,
                                 info="Controls buyer behaviour. Same seed = reproducible episode.")
            reset_btn = gr.Button("⟳  Reset Episode", variant="secondary", elem_classes=["btn-reset"])

            # Status bar
            status_bar = gr.HTML(value=_status_html(round_number=0, last_price=0.0, done=False))

            gr.HTML("<div class='divider'></div>")

            # Action builder card
            gr.HTML("<div class='card-head'>⚡ Action Builder</div>")
            ui_action = gr.Dropdown(
                choices=UI_ACTION_CHOICES, value="propose",
                label="Action type",
                info="counter_offer = propose on the wire. accept / reject end the episode.",
            )
            gr.HTML("""
            <div class='compare-strip'>
              <div class='cs-side you'><div class='cs-dot'></div>Your proposal</div>
              <div class='cs-arrow'>⟷</div>
              <div class='cs-side buyer'><div class='cs-dot'></div>Buyer's counter</div>
            </div>
            """)
            with gr.Row():
                price_in = gr.Number(value=100.0, label="Price (₹/unit)", minimum=0, step=0.5)
                days_in  = gr.Number(value=90,    label="Payment days",   minimum=0, maximum=365, precision=0)
            gr.HTML("<div class='unit-hint'>Fields auto-update to buyer's last counter-offer after each step</div>")

            with gr.Accordion("🔧 Advanced Options", open=False):
                reason_in  = gr.Textbox(label="Reason (optional)", lines=2, placeholder="e.g. liquidity bridge — passed as action.reason")
                deal_id_in = gr.Textbox(label="Deal ID (optional)", placeholder="Maps to NegotiationAction.deal_id")
                use_treds  = gr.Checkbox(label="Enable TReDS financing — sell receivables to a financier today", value=False)

            with gr.Accordion("📐 Medium / Hard task levers", open=False):
                late_clause = gr.Checkbox(label="Propose late-payment penalty clause (Medium+)", value=False)
                propose_dd  = gr.Checkbox(label="Propose dynamic discounting (Hard)", value=False)
                dd_rate     = gr.Slider(minimum=0.0, maximum=0.30, step=0.01, value=0.08,
                                        label="Dynamic discount annual rate",
                                        info="5–8% typical. Hard grader rewards coherent rates over blind acceptance.")

            with gr.Accordion("🔬 Simulation settings", open=False):
                simulation_plan_tb  = gr.Textbox(value="default", label="simulation_plan",
                                                 info="JSON object, 'default', or empty for None.")
                simulation_horizon_n = gr.Number(value=0, label="simulation_horizon", precision=0, minimum=0,
                                                 info="Optional planning horizon > 0.")

            submit_btn = gr.Button("▶  Submit Step", variant="primary", interactive=False, elem_classes=["btn-submit"])

        # ── RIGHT: Monitor Panel ─────────────────────────────────────────
        with gr.Column(scale=6, min_width=480):
            gr.HTML("<div class='panel-label'>📊 Monitor</div>")

            # Scenario context card (populated after Reset)
            scenario_box = gr.HTML(value="""
            <div class='scenario-card empty-scenario'>
              <div class='es-icon'>🏭</div>
              <div class='es-title'>No scenario loaded</div>
              <div class='es-sub'>Pick a task difficulty and seed, then click <strong>Reset Episode</strong> to load scenario details.</div>
            </div>
            """)

            # About section — 3 nested accordions
            with gr.Accordion("ℹ About this environment", open=False, elem_classes=["about-accordion"]):
                with gr.Accordion("How modes work", open=False):
                    gr.Markdown("""
| Mode | What happens |
|---|---|
| **This Playground** | You drive the environment by hand. **Reset Episode** loads a scenario; **Submit Step** sends a `NegotiationAction`. Each step returns an observation with `reward` and `done`. |
| **RL Training** | The *same* `SMENegotiatorEnvironment` runs inside a GRPO training loop. Entry points: `rl/train_grpo_trl.py`, `rl/train_grpo_unsloth.py`, notebook `notebooks/colab_grpo_sme_liquidity.ipynb`. |
""")
                with gr.Accordion("Reward explained", open=False):
                    gr.Markdown("""
**Per-step shaping** — emitted every `env.step()`. Encodes: liquidity pressure, days improvement, buyer reaction, solvency proximity. Bounded to `[−1, 1]`.

**Terminal benchmark score** — deterministic graders map the final `NegotiationState` to a score in `[0, 1]`:

```
score = 0.35 × solvency + 0.20 × liquidity + 0.35 × NPV + 0.10 × compliance
```

The **Grader Calculator** tab lets you compute this for any hypothetical outcome.
""")
                with gr.Accordion("Task rubric", open=False):
                    gr.Markdown("""
| Task | What the agent must do | Extra levers |
|---|---|---|
| **Easy** | Compress payment days to ≤ 60 | Price + Days |
| **Medium** | Tighten to ≤ 45 days + add late-payment penalty clause | + Clause |
| **Hard** | Negotiate dynamic discounting with a coherent annual rate | + TReDS + DD rate |

`compute_verifiable_reward` in `sme_negotiator_env/graders.py` gives RLVR-style decomposed reward. `rl/judge_pack.py` bundles evaluation artifacts; `EVALUATION.md` documents full benchmark methodology. *All prices are INR per unit (₹/unit).*
""")

            # Negotiation transcript
            gr.HTML("<div class='section-divider'><span>💬 Negotiation Transcript</span></div>")
            chat = gr.Chatbot(
                show_label=False, height=280,
                placeholder="<div style='text-align:center;padding:32px 16px;color:#4A5568;font-size:0.84rem;line-height:1.8'>"
                            "🤝<br><br>No turns yet.<br>Reset the episode and submit an action<br>to see the live negotiation transcript.</div>",
            )

            # Reward panel (populated after Submit)
            gr.HTML("<div class='section-divider'><span>📈 Reward Signal</span></div>")
            reward_box = gr.HTML(value=_reward_html(0.0, 0.0, 0))

            # Last step JSON
            gr.HTML("<div class='section-divider'><span>🔍 Last Step Observation</span></div>")
            last_json = gr.JSON(label="", show_label=False)

    # ── Bottom tabs ──────────────────────────────────────────────────────
    gr.HTML("<div class='tabs-head'>Tools</div>")
    with gr.Tabs(elem_classes=["bottom-tabs"]):

        with gr.Tab("🤖 Heuristic Auto-Play", elem_classes=["tab-autoplay"]):
            gr.HTML("""
            <div class='tab-intro'>
              <strong>Greedy baseline policy:</strong> proposes the buyer's current days − 5 each round until it hits the liquidity threshold, then accepts.
              Hard task adds dynamic discounting at 8%. Use this to see what a non-RL baseline scores.
            </div>
            """)
            with gr.Row():
                h_task = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0], label="Task")
                h_seed = gr.Number(value=42, label="Seed", minimum=0, maximum=99999, precision=0)
            h_run_btn   = gr.Button("▶ Run Heuristic Playthrough", variant="primary")
            h_status    = gr.Markdown("")
            h_score     = gr.HTML(value="")
            h_obs_json  = gr.Code(language="json", label="Final observation JSON", value="{}")
            h_transcript = gr.Textbox(label="Step-by-step transcript", lines=12, interactive=False)
            h_run_btn.click(fn=heuristic_play, inputs=[h_task, h_seed],
                            outputs=[h_obs_json, h_status, h_transcript, h_score])

        with gr.Tab("🧮 Grader Calculator", elem_classes=["tab-grader"]):
            gr.HTML("""
            <div class='tab-intro'>
              Compute the <strong>terminal benchmark score</strong> for any hypothetical deal outcome — no episode needed.
              Same deterministic graders used in official evaluation. Score = 0.35×Solvency + 0.20×Liquidity + 0.35×NPV + 0.10×Compliance.
            </div>
            """)
            with gr.Row():
                with gr.Column():
                    gc_task    = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0], label="Task")
                    gc_deal    = gr.Checkbox(label="Deal was reached", value=True)
                    gc_days    = gr.Slider(minimum=0, maximum=120, step=1,  value=55,  label="Agreed payment days")
                    gc_price   = gr.Slider(minimum=70.0, maximum=110.0, step=0.5, value=100.0, label="Agreed price (₹/unit)")
                    gc_clause  = gr.Checkbox(label="Late-payment penalty clause agreed", value=False)
                    gc_dd      = gr.Checkbox(label="Dynamic discounting agreed", value=False)
                    gc_dd_rate = gr.Slider(minimum=0.0, maximum=0.30, step=0.01, value=0.08, label="Dynamic discount annual rate")
                    gc_compute = gr.Button("Calculate Terminal Score", variant="primary")
                with gr.Column():
                    gc_result = gr.Markdown("*Configure deal terms on the left and click Calculate.*")
            gc_compute.click(fn=compute_grader_score,
                             inputs=[gc_task, gc_days, gc_price, gc_deal, gc_clause, gc_dd, gc_dd_rate],
                             outputs=[gc_result])

        with gr.Tab("📖 Reference & API", elem_classes=["tab-reference"]):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
### Observation fields (`NegotiationObservation`)
| Field | Type | Meaning |
|---|---|---|
| `buyer_price` | float | Buyer's current ₹/unit offer |
| `buyer_days` | int | Buyer's proposed payment period |
| `round_number` | int | Current negotiation round |
| `liquidity_threshold` | int | Max days SME can survive without cash |
| `step_reward` / `reward` | float | Per-step shaping signal ∈ [−1, 1] |
| `buyer_accepted` | bool | Did buyer accept last proposal? |
| `negotiation_done` | bool | Episode ended? |
| `buyer_power_score` | float | Buyer's leverage [0, 1] |
| `volume` | int | Units in this deal |
| `message` | str | Buyer's verbal response |

### Action type meanings
| Action | Effect |
|---|---|
| `propose` | SME makes a counter-offer (price + days) |
| `accept` | SME accepts buyer's current terms — ends episode |
| `reject` | SME walks away — ends episode with penalty |

### Reward grading (terminal)
```
score = 0.35 × solvency  +  0.20 × liquidity
      + 0.35 × NPV_improvement  +  0.10 × compliance
```
`compliance` = 1.0 if agreed days ≤ 45 (MSMED Act), 0.5 if ≤ 60, 0 otherwise.

🔗 [GitHub](https://github.com/SkandaGanesha1/ENV) · [HF Space](https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator)
""")
                with gr.Column():
                    gr.HTML("<div class='card-head'>Quick Connect (Python)</div>")
                    gr.Code(value=QUICK_CONNECT_SNIPPET, language="python",
                            label="", show_label=False, interactive=False)

    gr.HTML("""
    <div class="footer">
      <span>OpenEnv SME Negotiator</span>
      <span class="ft-sep">·</span>
      <span>MIT License</span>
      <span class="ft-sep">·</span>
      <a href="https://github.com/SkandaGanesha1/ENV" target="_blank">GitHub</a>
      <span class="ft-sep">·</span>
      <a href="https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator" target="_blank">HF Space</a>
    </div>
    <script>
    (function() {
      function setStep(n) {
        for (var i = 1; i <= 4; i++) {
          var el = document.getElementById('sme-step' + i);
          if (!el) continue;
          el.classList.remove('active','done');
          if (i < n)  el.classList.add('done');
          if (i === n) el.classList.add('active');
        }
      }
      function ready(fn) {
        if (document.readyState !== 'loading') fn(); else document.addEventListener('DOMContentLoaded', fn);
      }
      ready(function() {
        setStep(1);
        document.addEventListener('click', function(e) {
          var r = e.target.closest('.btn-reset');
          if (r) { setStep(2); var t = r.textContent; r.textContent = '⟳  Loading…'; setTimeout(function(){ r.textContent=t; setStep(3); }, 2200); }
          var s = e.target.closest('.btn-submit');
          if (s && !s.disabled) { s.classList.add('loading'); setStep(4); setTimeout(function(){ s.classList.remove('loading'); setStep(3); }, 3500); }
        });
        /* Animate header on deal outcome */
        new MutationObserver(function() {
          var h = document.getElementById('sme-header');
          if (!h) return;
          h.classList.toggle('deal-accepted', !!document.querySelector('.deal-card.accepted'));
          h.classList.toggle('deal-rejected', !!document.querySelector('.deal-card.rejected'));
        }).observe(document.body, { childList:true, subtree:true });
      });
    })();
    </script>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────
    reset_outputs = [session_state, status_bar, scenario_box, reward_box, chat, last_json, price_in, days_in, submit_btn]
    reset_btn.click(fn=reset_episode, inputs=[task_sel, seed_num, session_state], outputs=reset_outputs)

    submit_outputs = [session_state, status_bar, reward_box, chat, last_json, price_in, days_in, submit_btn]
    submit_btn.click(
        fn=submit_step,
        inputs=[ui_action, price_in, days_in, reason_in, deal_id_in, use_treds,
                simulation_plan_tb, simulation_horizon_n, late_clause, propose_dd, dd_rate, session_state],
        outputs=submit_outputs,
    )


if __name__ == "__main__":
    # HF Spaces / Docker often inject PORT=7860. Local override: GRADIO_PORT.
    _port_raw = (os.getenv("PORT") or os.getenv("GRADIO_PORT") or "7860").strip()
    requested_port = int(_port_raw)
    max_attempts = 20
    port = requested_port

    for _ in range(max_attempts):
        try:
            if port == requested_port:
                print(f"[startup] Launching on port {port} (from PORT / GRADIO_PORT / default 7860)")
            else:
                print(f"[startup] Port {requested_port} busy; trying fallback port {port}")
            # 0.0.0.0 is bind-only; browsers need localhost / 127.0.0.1 (ERR_ADDRESS_INVALID for http://0.0.0.0/...).
            print(f"[startup] Open this in your browser: http://127.0.0.1:{port}/")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                theme=THEME,
                show_error=True,
            )
            break
        except OSError as exc:
            print(f"[startup] Port {port} unavailable ({exc!r}); trying {port + 1}...")
            port += 1
    else:
        raise RuntimeError(
            f"Could not find a free port starting from {requested_port} "
            f"after {max_attempts} attempts. Set GRADIO_PORT to a known free port."
        )
