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
from collections.abc import Iterable
from typing import Any, Optional

import altair as alt
import pandas as pd

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
from config import QUICK_CONNECT_SNIPPET, TASKS, UI_ACTION_CHOICES
from reward_engine import RewardEngine
from session_store import SessionStore
from step_logger import StepLogger

_action_handler = ActionHandler()
_reward_engine = RewardEngine()
_logger = StepLogger()

# TASKS, UI_ACTION_CHOICES, QUICK_CONNECT_SNIPPET imported from config.py
# ─────────────────────────────────────────────────────────────────────────────


def _obs_to_dict(obs: Any) -> dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs)



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
    """
    Terminal benchmark score card (0–1) with verdict, component breakdown,
    and animated gradient bar so judges instantly understand the outcome.
    """
    pct = max(0.0, min(1.0, float(score))) * 100
    cls = "positive" if score >= 0.7 else ("zero" if score >= 0.4 else "negative")

    if score >= 0.85:
        verdict_label = "Excellent"
        verdict_icon = "🏆"
        verdict_desc = "Policy found SME-friendly payment terms with strong financials."
        subtitle = "High solvency, strong liquidity buffer, good NPV, and fully compliant days."
    elif score >= 0.7:
        verdict_label = "Good"
        verdict_icon = "✅"
        verdict_desc = "Solid deal — meets most objectives with minor trade-offs."
        subtitle = "Most sub-scores satisfied; small concessions in liquidity or NPV."
    elif score >= 0.5:
        verdict_label = "Fair"
        verdict_icon = "⚠️"
        verdict_desc = "Workable but sub-optimal — notable gaps remain."
        subtitle = "Some objectives met, but solvency or compliance could be improved."
    elif score >= 0.3:
        verdict_label = "Weak"
        verdict_icon = "⚡"
        verdict_desc = "Risky terms for the SME — needs significant improvement."
        subtitle = "Long receivables, weak liquidity, or non-compliant payment window."
    else:
        verdict_label = "Poor"
        verdict_icon = "❌"
        verdict_desc = "Deal is dangerous for SME cash-flow and solvency."
        subtitle = "Almost all sub-components scored low; policy failed to protect the SME."

    return f"""
<div class="reward-panel score-panel-v2">
    <div class="sp2-top">
        <div class="sp2-left">
            <div class="sp2-eyebrow">Terminal Benchmark Score</div>
            <div class="sp2-verdict-row">
                <span class="sp2-verdict-pill {cls}">{verdict_icon} {verdict_label}</span>
                <span class="sp2-score-chip {cls}">{score:.4f}</span>
            </div>
            <div class="sp2-desc">{verdict_desc}</div>
            <div class="sp2-sub">{subtitle}</div>
        </div>
    </div>

    <div class="sp2-bar-section">
        <div class="sp2-bar-track">
            <div class="sp2-bar-fill {cls}" style="width:{pct:.1f}%"></div>
            <div class="sp2-marker" style="left:40%"><span>0.4</span></div>
            <div class="sp2-marker" style="left:70%"><span>0.7</span></div>
        </div>
        <div class="sp2-bar-labels">
            <span class="sp2-bl sp2-bl-low">Needs work</span>
            <span class="sp2-bl sp2-bl-mid">Fair</span>
            <span class="sp2-bl sp2-bl-high">Strong</span>
        </div>
    </div>

    <div class="sp2-components">
        <div class="sp2-comp-title">Score Composition</div>
        <div class="sp2-comp-grid">
            <div class="sp2-comp-chip solvency"><span class="sp2-w">35%</span> Solvency</div>
            <div class="sp2-comp-chip liquidity"><span class="sp2-w">20%</span> Liquidity</div>
            <div class="sp2-comp-chip npv"><span class="sp2-w">35%</span> NPV</div>
            <div class="sp2-comp-chip compliance"><span class="sp2-w">10%</span> Compliance</div>
        </div>
    </div>

    <div class="sp2-footer">
        Deterministic grading · same formula used in official RL benchmark evaluation
    </div>
</div>
"""


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
    """
    Premium reward signal panel — judges see the current round's reward,
    the running cumulative total, and a short legend explaining the signal.
    """
    if step_num == 0:
        return """
<div class="rp2 rp2-idle">
    <div class="rp2-idle-hero">
        <div class="rp2-pulse-ring">
            <span class="rp2-pulse-icon">📊</span>
        </div>
        <div class="rp2-idle-copy">
            <div class="rp2-idle-title">Reward signal starts after your first step</div>
            <div class="rp2-idle-sub">
                Click <strong>Submit Step</strong> to send a negotiation action.
                The environment returns a reward ∈ [−1, 1] reflecting how good
                the proposed terms are for the SME.
            </div>
        </div>
    </div>
    <div class="rp2-legend">
        <div class="rp2-legend-title">What drives the reward?</div>
        <div class="rp2-legend-grid">
            <div class="rp2-lg-item rp2-lg-pos">
                <span class="rp2-lg-dot pos"></span>
                <span>Shorter payment days, stronger liquidity, solvency improvement</span>
            </div>
            <div class="rp2-lg-item rp2-lg-neg">
                <span class="rp2-lg-dot neg"></span>
                <span>Longer days, weaker cash-flow, risky or non-compliant terms</span>
            </div>
        </div>
    </div>
    <div class="rp2-footer">
        The chart below will plot the reward <em>curvature</em> — the shape of the
        learning signal across the entire negotiation episode.
    </div>
</div>
"""
    # ── Active state (after at least one step) ───────────────────────────
    step_cls = "positive" if step_reward >= 0 else "negative"
    cum_cls = "positive" if cum_reward >= 0 else "negative"
    step_sign = "▲" if step_reward >= 0 else "▼"
    cum_sign = "▲" if cum_reward >= 0 else "▼"
    step_pct = min(100, max(4, int(abs(step_reward) * 100)))
    cum_pct = min(100, max(4, int(abs(cum_reward) / max(1, step_num) * 100)))
    avg_rew = cum_reward / max(1, step_num)
    avg_cls = "positive" if avg_rew >= 0 else "negative"

    return f"""
<div class="rp2 rp2-active rp2-border-{step_cls}">
    <div class="rp2-metrics">
        <div class="rp2-metric-card rp2-mc-step {step_cls}">
            <div class="rp2-mc-label">Step Reward</div>
            <div class="rp2-mc-val">{step_sign} {step_reward:+.4f}</div>
            <div class="rp2-mc-bar-track">
                <div class="rp2-mc-bar {step_cls}" style="width:{step_pct}%"></div>
            </div>
        </div>
        <div class="rp2-metric-card rp2-mc-cum {cum_cls}">
            <div class="rp2-mc-label">Cumulative</div>
            <div class="rp2-mc-val">{cum_sign} {cum_reward:+.4f}</div>
            <div class="rp2-mc-bar-track">
                <div class="rp2-mc-bar {cum_cls}" style="width:{cum_pct}%"></div>
            </div>
        </div>
        <div class="rp2-metric-card rp2-mc-avg {avg_cls}">
            <div class="rp2-mc-label">Avg / Round</div>
            <div class="rp2-mc-val">{avg_rew:+.4f}</div>
            <div class="rp2-mc-sub">over {step_num} round{'s' if step_num != 1 else ''}</div>
        </div>
    </div>

    <div class="rp2-legend rp2-legend-compact">
        <div class="rp2-legend-grid">
            <div class="rp2-lg-item rp2-lg-pos">
                <span class="rp2-lg-dot pos"></span>
                <span>↓ days · ↑ liquidity · solvency</span>
            </div>
            <div class="rp2-lg-item rp2-lg-neg">
                <span class="rp2-lg-dot neg"></span>
                <span>↑ days · risk · non-compliance</span>
            </div>
        </div>
    </div>

    <div class="rp2-footer">
        Round {step_num} · reward ∈ [−1, 1] · terminal score = 35% solvency + 20% liquidity + 35% NPV + 10% compliance
    </div>
</div>
"""


def _extract_reward_history(state: SessionStore) -> list[dict[str, float]]:
    """
    Build chart-friendly reward history from SessionStore.
    Prefers the reliable ``_step_rewards`` list; falls back to message parsing.
    """
    if not isinstance(state, SessionStore):
        return []

    # ── Fast path: use the internal step-reward list ──────────────────────
    rewards = getattr(state, "_step_rewards", None) or []
    if rewards:
        history: list[dict[str, float]] = []
        cumulative = 0.0
        for i, r in enumerate(rewards, 1):
            cumulative += r
            history.append({
                "round": float(i),
                "step_reward": float(r),
                "cumulative_reward": cumulative,
            })
        return history

    # ── Fallback: parse from message transcript ──────────────────────────
    raw_messages = getattr(state, "messages", None) or []
    if not isinstance(raw_messages, Iterable):
        return []

    history = []
    step = 0
    cumulative = 0.0

    for i in range(0, len(raw_messages), 2):
        if not isinstance(raw_messages[i], dict):
            continue
        user_msg = raw_messages[i].get("content", "")
        assistant_msg = raw_messages[i+1].get("content", "") if i+1 < len(raw_messages) else ""
        text = f"{user_msg}\n{assistant_msg}"

        marker = "**reward:**"
        if marker not in text:
            continue

        try:
            reward_part = text.split(marker, 1)[1].strip().split()[0]
            reward = float(reward_part)
        except Exception:
            continue

        step += 1
        cumulative += reward
        history.append({
            "round": float(step),
            "step_reward": reward,
            "cumulative_reward": cumulative,
        })

    return history


def _empty_reward_chart():
    """Placeholder chart shown before any steps are taken."""
    placeholder = pd.DataFrame({
        "round": [0, 1, 2, 3],
        "value": [0.0, 0.0, 0.0, 0.0],
        "series": ["Waiting for data…"] * 4,
    })
    return (
        alt.Chart(placeholder)
        .mark_line(strokeDash=[4, 4], opacity=0.25, strokeWidth=1.5)
        .encode(
            x=alt.X("round:Q", title="Negotiation Round",
                     axis=alt.Axis(labelColor="#64748B", titleColor="#94A3B8", tickMinStep=1)),
            y=alt.Y("value:Q", title="Reward",
                     axis=alt.Axis(labelColor="#64748B", titleColor="#94A3B8")),
            color=alt.Color("series:N", legend=alt.Legend(
                title=None, orient="top", labelColor="#64748B",
            )),
        )
        .properties(height=240, title=alt.TitleParams(
            "Reward curvature will appear here",
            color="#64748B", fontSize=12, anchor="middle",
        ))
        .configure_view(stroke=None, fill="#111827")
        .configure(background="#111827")
        .configure_axis(gridColor="#1E293B", domainColor="#334155")
    )


def _reward_curve_chart(state: SessionStore):
    """
    Altair chart for judges: step reward + cumulative reward across rounds.
    Shows lines with area fill for visual emphasis and a zero-line reference.
    """
    history = _extract_reward_history(state)
    if not history:
        return _empty_reward_chart()

    df = pd.DataFrame(history)

    # Rename for human-readable legend
    long_df = df.melt(
        id_vars=["round"],
        value_vars=["step_reward", "cumulative_reward"],
        var_name="series",
        value_name="value",
    )
    label_map = {"step_reward": "Step Reward", "cumulative_reward": "Cumulative Reward"}
    long_df["series"] = long_df["series"].map(label_map)

    color_scale = alt.Scale(
        domain=["Step Reward", "Cumulative Reward"],
        range=["#60A5FA", "#34D399"],
    )

    # ── Base encoding ────────────────────────────────────────────────────
    base = (
        alt.Chart(long_df)
        .encode(
            x=alt.X(
                "round:Q",
                axis=alt.Axis(labelColor="#94A3B8", titleColor="#CBD5E1",
                              tickMinStep=1, grid=True, gridOpacity=0.15),
                title="Negotiation Round",
            ),
            y=alt.Y(
                "value:Q",
                axis=alt.Axis(labelColor="#94A3B8", titleColor="#CBD5E1",
                              grid=True, gridOpacity=0.15),
                title="Reward",
            ),
            color=alt.Color(
                "series:N",
                scale=color_scale,
                legend=alt.Legend(
                    title=None, orient="top",
                    labelColor="#E2E8F0", symbolStrokeWidth=4,
                    labelFontSize=11, symbolSize=80,
                ),
            ),
            tooltip=[
                alt.Tooltip("round:Q", title="Round"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Reward", format=".4f"),
            ],
        )
        .properties(height=240)
    )

    # ── Layers ───────────────────────────────────────────────────────────
    area = base.mark_area(opacity=0.08, interpolate="monotone")
    lines = base.mark_line(strokeWidth=2.5, interpolate="monotone")
    points = base.mark_circle(size=55, opacity=0.9)
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[6, 4], color="#475569", strokeWidth=1)
        .encode(y="y:Q")
    )

    chart = (
        (zero_rule + area + lines + points)
        .configure_view(stroke=None, fill="#111827")
        .configure(background="#111827")
        .configure_axis(gridColor="#1E293B", domainColor="#334155")
    )

    return chart


def _status_html(
    *,
    round_number: int,
    last_price: float,
    buyer_days: int = 0,
    done: bool,
    reward: float = 0.0,
    buyer_accepted: bool = False,
    max_rounds: int = 10,
    business_impact: str = "",
) -> str:
    """Hardware Monitor-style status bar."""
    if done:
        badge, badge_color = "Done", "#86EFAC"
        icon_stat = "🏁"
    elif round_number == 0:
        badge, badge_color = "Ready", "#94A3B8"
        icon_stat = "💤"
    else:
        badge, badge_color = "Active", "#F59E0B"
        icon_stat = "⚡"
    
    price_str = f"₹{float(last_price):.2f}" if last_price else "—"
    
    pct = min(100, int((round_number / max(max_rounds, 1)) * 100))
    days_pct = min(100, int((buyer_days / 120) * 100)) if buyer_days else 0

    reward_str = f"{reward:+.4f}" if round_number > 0 else "—"
    outcome_str = "Deal Reached" if (done and buyer_accepted) else ("No Deal" if done else "Pending")

    deal_card = ""
    if done:
        if buyer_accepted:
            deal_card = f"<div class='deal-card accepted'><span class='deal-icon'>✅</span><strong>Deal Accepted</strong><span class='deal-meta'>{price_str}/unit · {int(round_number)} rounds</span>{business_impact}</div>"
        else:
            deal_card = f"<div class='deal-card rejected'><span class='deal-icon'>❌</span><strong>No Deal</strong><span class='deal-meta'>Episode ended · {int(round_number)} rounds</span>{business_impact}</div>"

    return f"""
<div class="gpu-card">
  <div class="gpu-head">
    <div class="gpu-title">Negotiation Engine (Live)</div>
    <div class="gpu-tag"># {int(round_number)}</div>
  </div>
  
  <div class="gpu-body">
    <!-- Row 1 -->
    <div class="gpu-metric">
      <div class="gpu-m-label-row">
        <span class="gpu-m-icon" style="color: {badge_color};">{icon_stat}</span>
        <span>Status</span>
      </div>
      <div class="gpu-m-val" style="color: {badge_color};">{badge}</div>
    </div>
    
    <div class="gpu-bar-container">
      <div class="gpu-bar-head">
        <div class="gpu-bar-label-row">
          <span class="gpu-bar-icon" style="color: #9CA3AF;">🎛️</span>
          <span>Negotiation Load</span>
        </div>
        <span class="gpu-bar-pct">{pct}%</span>
      </div>
      <div class="gpu-progress">
        <div class="gpu-progress-fill load" style="width: {pct}%"></div>
      </div>
      <div class="gpu-bar-foot">Round {int(round_number)} / {max_rounds} Max</div>
    </div>
    
    <!-- Row 2 -->
    <div class="gpu-metric">
      <div class="gpu-m-label-row">
        <span class="gpu-m-icon" style="color: #3B82F6;">🪙</span>
        <span>Buyer Offer</span>
      </div>
      <div class="gpu-m-val" style="color: #93C5FD;">{price_str}</div>
    </div>
    
    <div class="gpu-bar-container">
      <div class="gpu-bar-head">
        <div class="gpu-bar-label-row">
          <span class="gpu-bar-icon" style="color: #3B82F6;">💾</span>
          <span>Payment Terms</span>
        </div>
        <span class="gpu-bar-pct">{buyer_days} Days</span>
      </div>
      <div class="gpu-progress">
        <div class="gpu-progress-fill memory" style="width: {days_pct}%"></div>
      </div>
      <div class="gpu-bar-foot">SME Liquidity Target</div>
    </div>
  </div>
  
  <div class="gpu-footer">
    <div class="gpu-f-metric">
      <span class="gpu-f-icon" style="color: #A78BFA;">⏱️</span>
      <div class="gpu-f-col">
        <span class="gpu-f-label">Reward Signal</span>
        <span class="gpu-f-val" style="color: #E9D5FF;">{reward_str}</span>
      </div>
    </div>
    <div class="gpu-f-metric">
      <span class="gpu-f-icon" style="color: #FBBF24;">⚡</span>
      <div class="gpu-f-col">
        <span class="gpu-f-label">Outcome</span>
        <span class="gpu-f-val" style="color: #FDE68A;">{outcome_str}</span>
      </div>
    </div>
  </div>
</div>
{deal_card}
"""


def _build_last_step_json(obs_dict: dict[str, Any]) -> dict[str, Any]:
    """Shape for ``gr.JSON``: reward, done, and a small derived ``info`` subset."""
    reward = _reward_engine.extract_step_reward(obs_dict)
    done = _reward_engine.is_done(obs_dict)
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
    if deal_id and str(deal_id).strip():
        lines.append(f"**deal_id:** `{str(deal_id).strip()}`")
    if reason and str(reason).strip():
        lines.append(f"**Reason:** {str(reason).strip()}")
    sp = (sim_plan_raw or "").strip()
    if sp and sp.lower() != "default":
        lines.append(f"**simulation_plan (raw):** `{sp[:200]}{'…' if len(sp) > 200 else ''}`")
    if horizon is not None and int(horizon) > 0:
        lines.append(f"**simulation_horizon:** {int(horizon)}")
    return "\n".join(lines)


def _format_assistant_turn(obs_dict: dict[str, Any]) -> str:
    reward = _reward_engine.extract_step_reward(obs_dict)
    done = _reward_engine.is_done(obs_dict)
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

        status_html = _status_html(round_number=rnd, last_price=last_price, buyer_days=p_days, done=done0, reward=0.0)
        scenario_html = _scenario_html(cfg, obs)
        reward_html = _reward_html(0.0, 0.0, 0)
        json_payload = _build_last_step_json(obs)

        return (
            store,
            status_html,
            scenario_html,
            reward_html,
            _reward_curve_chart(store),
            [],
            json_payload,
            p_price,
            p_days,
            gr.update(interactive=True),
            gr.update(visible=False, value=None),
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
            _reward_curve_chart(store),
            [],
            {"reward": 0.0, "done": False, "info": {"message": err}},
            100.0,
            90,
            gr.update(interactive=False),
            gr.update(visible=False, value=None),
        )


def _generate_contract_html(obs_dict: dict, cfg: Any, task_label: str) -> str:
    import tempfile
    from datetime import datetime
    
    agreed_price = float(obs_dict.get("buyer_price", 0))
    agreed_days = int(obs_dict.get("buyer_days", 0))
    volume = int(getattr(cfg, "volume", 10000))
    total_value = agreed_price * volume
    
    diff = getattr(cfg, "difficulty", "")
    late_clause = "Yes (3x RBI Bank Rate)" if diff in ["medium", "hard"] else "Standard Terms"
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    html = f"""
    <html>
    <head>
        <title>Legal Contract - SME Negotiator</title>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 40px; color: #333; max-width: 800px; margin: 0 auto; line-height: 1.6; background: #fff; }}
            h1 {{ border-bottom: 2px solid #2563EB; padding-bottom: 10px; color: #1E3A8A; }}
            .header {{ display: flex; justify-content: space-between; margin-bottom: 30px; }}
            .box {{ border: 1px solid #E5E7EB; padding: 20px; border-radius: 8px; margin-bottom: 20px; background: #F9FAFB; }}
            .box h3 {{ margin-top: 0; color: #374151; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #E5E7EB; padding-bottom: 8px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
            .item {{ display: flex; justify-content: space-between; border-bottom: 1px dashed #D1D5DB; padding: 5px 0; }}
            .item-label {{ font-weight: 600; color: #4B5563; font-size: 14px; }}
            .item-val {{ font-family: monospace; font-size: 15px; color: #111827; font-weight: bold; }}
            .signatures {{ margin-top: 80px; display: grid; grid-template-columns: 1fr 1fr; gap: 60px; }}
            .sig-line {{ border-top: 1px solid #000; margin-top: 50px; padding-top: 10px; text-align: center; font-weight: bold; color: #111; }}
            .stamp {{ border: 3px solid #16A34A; color: #16A34A; padding: 12px 24px; font-weight: 900; font-family: Impact, sans-serif; transform: rotate(-5deg); display: inline-block; position: absolute; margin-top: -40px; margin-left: 250px; opacity: 0.9; font-size: 28px; letter-spacing: 2px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <div style="font-size: 24px; font-weight: 800; color: #2563EB; letter-spacing: -0.5px;">OpenEnv</div>
                <div style="color: #64748B; font-size: 14px;">Procurement Department</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 18px; font-weight: 800; color: #1E293B;">PURCHASE AGREEMENT</div>
                <div style="color: #475569; font-size: 14px;">Date: {date_str}</div>
                <div style="color: #475569; font-size: 14px;">Ref: DEAL-{abs(hash(str(total_value) + str(agreed_days))) % 1000000:06d}</div>
            </div>
        </div>
        
        <p style="font-size: 15px; color: #334155;">This Purchase Agreement is executed between the <strong>Buyer</strong> and the <strong>SME Supplier</strong>, confirming the successful negotiation of supply terms under the benchmark task <strong>{task_label}</strong>.</p>
        
        <div class="box">
            <h3>Commercial Terms</h3>
            <div class="grid">
                <div class="item"><span class="item-label">Agreed Unit Price:</span> <span class="item-val">₹{agreed_price:,.2f}</span></div>
                <div class="item"><span class="item-label">Order Volume:</span> <span class="item-val">{volume:,} units</span></div>
                <div class="item" style="grid-column: 1 / 3; font-size: 18px; border-bottom: 2px solid #94A3B8; margin-top: 10px; padding-bottom: 10px;">
                    <span class="item-label" style="font-size: 18px;">Total Deal Value:</span> 
                    <span class="item-val" style="font-size: 20px; color: #0F172A;">₹{total_value:,.2f}</span>
                </div>
            </div>
        </div>

        <div class="box" style="background: #EFF6FF; border-color: #BFDBFE;">
            <h3 style="color: #1E40AF; border-color: #BFDBFE;">Payment & Liquidity Terms</h3>
            <div class="grid">
                <div class="item"><span class="item-label" style="color: #1E3A8A;">Agreed Payment Days:</span> <span class="item-val" style="color: #1D4ED8; font-size: 16px;">{agreed_days} Days</span></div>
                <div class="item"><span class="item-label" style="color: #1E3A8A;">Late Payment Clause:</span> <span class="item-val" style="color: #1D4ED8;">{late_clause}</span></div>
            </div>
            <p style="font-size: 12px; color: #475569; margin-top: 15px; font-style: italic;">* Compliant with MSMED Act 2006. Any delay beyond {max(45, agreed_days)} days will attract compound interest at 3x the RBI bank rate.</p>
        </div>
        
        <div class="stamp">EXECUTED & BOUND</div>

        <div class="signatures">
            <div>
                <div class="sig-line">Authorized Signatory<br><span style="font-weight: normal; font-size: 13px; color: #475569;">Buyer Corporation</span></div>
            </div>
            <div>
                <div class="sig-line">Authorized Signatory<br><span style="font-weight: normal; font-size: 13px; color: #475569;">SME Enterprise</span></div>
            </div>
        </div>
    </body>
    </html>
    """
    
    fd, path = tempfile.mkstemp(suffix=".html", prefix="SME_Contract_")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


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
    messages: list = list(store.messages or [])

    if not store.is_active:
        err = "⚠️ No active episode — use **Reset Episode** first."
        user_t = _format_user_turn(
            ui_action=ui_action or "",
            wired_action="—",
            price=float(price),
            days=int(payment_days),
            reason=reason or "",
            deal_id=deal_id or "",
            use_treds=bool(use_treds),
            sim_plan_raw=simulation_plan_raw or "",
            horizon=int(simulation_horizon) if simulation_horizon else None,
        )
        messages = messages + [
            {"role": "user", "content": user_t},
            {"role": "assistant", "content": err}
        ]
        return (
            store,
            _status_html(round_number=0, last_price=0.0, buyer_days=0, done=False),
            _reward_html(0.0, 0.0, 0),
            _reward_curve_chart(store),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    env = store.env
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
        messages = messages + [
            {"role": "user", "content": user_t},
            {"role": "assistant", "content": err_msg}
        ]
        _logger.log_validation_error(
            episode_id=store.episode_id, step=store.step_num,
            error=val_err, action_type=wired,
            price=float(price), payment_days=int(payment_days),
        )
        return (
            store, gr.update(), gr.update(), gr.update(), messages,
            {"reward": 0.0, "done": False, "info": {"message": val_err}},
            gr.update(), gr.update(), gr.update(), gr.update(),
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
        messages = messages + [
            {"role": "user", "content": user_t},
            {"role": "assistant", "content": f"**Parse error:** {err}"}
        ]
        return (
            store,
            gr.update(),
            gr.update(),
            _reward_curve_chart(store),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
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
        deal_id=(str(deal_id).strip() if deal_id else None),
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
        messages = messages + [
            {"role": "user", "content": user_t},
            {"role": "assistant", "content": err}
        ]
        return (
            store,
            gr.update(),
            gr.update(),
            _reward_curve_chart(store),
            messages,
            {"reward": 0.0, "done": False, "info": {"message": err}},
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    done = _reward_engine.is_done(obs_dict)
    step_reward = _reward_engine.extract_step_reward(obs_dict)

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
    accepted = _reward_engine.buyer_accepted(obs_dict)
    
    business_impact = ""
    contract_path = None
    if done and accepted:
        task_id = TASKS.get(store.task_label)
        cfg = TASK_REGISTRY.get(task_id) if task_id else None
        if cfg:
            revenue = getattr(cfg, "sme_monthly_revenue", 500000.0)
            initial_days = getattr(cfg, "initial_buyer_days", 90)
            interest = getattr(cfg, "interest_rate_annual", 0.22)
            agreed_days = last_price_days = int(obs_dict.get("buyer_days", 0))
            
            saved_days = initial_days - agreed_days
            if saved_days > 0:
                capital_freed = (revenue / 30) * saved_days
                business_impact = f"<div style='margin-top:8px;font-size:0.85rem;color:#86EFAC;background:rgba(34,197,94,0.1);padding:8px;border-radius:4px;border:1px solid rgba(34,197,94,0.2)'>💸 <strong>Business Impact:</strong> Freed up <strong>₹{capital_freed:,.0f}</strong> in working capital by reducing terms by {saved_days} days.</div>"
            else:
                business_impact = f"<div style='margin-top:8px;font-size:0.85rem;color:#FCA5A5;background:rgba(239,68,68,0.1);padding:8px;border-radius:4px;border:1px solid rgba(239,68,68,0.2)'>⚠️ <strong>Business Impact:</strong> Working capital tied up for {agreed_days} days.</div>"
                
            contract_path = _generate_contract_html(obs_dict, cfg, store.task_label)

    status_html = _status_html(
        round_number=rnd, last_price=last_price, buyer_days=int(obs_dict.get("buyer_days", 0)), done=done,
        reward=step_reward, buyer_accepted=accepted,
        business_impact=business_impact,
    )
    rwd_html = _reward_html(step_reward, store.cum_rew, store.step_num)
    reward_curve = _reward_curve_chart(store)
    json_payload = _build_last_step_json(obs_dict)

    return (
        store,
        status_html,
        rwd_html,
        reward_curve,
        store.messages,
        json_payload,
        float(obs_dict.get("buyer_price", price)),
        int(obs_dict.get("buyer_days", payment_days)),
        gr.update(interactive=not done),
        gr.update(visible=bool(contract_path), value=contract_path)
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
            step_reward = _reward_engine.extract_step_reward(obs_dict)
            cum_rew += step_reward
            done = _reward_engine.is_done(obs_dict)

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
    task_id = TASKS[task_label]
    cfg = TASK_REGISTRY[task_id]

    state_kwargs = {
        "episode_id": "demo",
        "seed": 0,
        "difficulty": cfg.difficulty,
        "task_name": task_id,
        "step_count": 1,
        "max_steps": cfg.max_rounds,
        "max_rounds": cfg.max_rounds,
        "deal_reached": deal_reached,
        "final_price": float(agreed_price) if deal_reached else None,
        "final_days": int(agreed_days) if deal_reached else None,
        "treds_used": False,
        "cumulative_reward": 0.0,
        "buyer_price": float(cfg.initial_buyer_price),
        "buyer_days": int(cfg.initial_buyer_days),
        "initial_buyer_days": int(cfg.initial_buyer_days),
        "cost_threshold": float(cfg.cost_threshold),
        "liquidity_threshold": int(cfg.liquidity_threshold),
        "volume": int(cfg.volume),
        "message": "",
        "sme_monthly_revenue": float(cfg.sme_monthly_revenue),
        "current_payment_terms_days": int(agreed_days) if deal_reached else cfg.current_payment_terms_days,
        "sme_supplier_payment_days": int(cfg.sme_supplier_payment_days),
        "interest_rate_annual": float(cfg.interest_rate_annual),
        "buyer_power_score": float(cfg.buyer_power_score),
        "agreed_terms": int(agreed_days) if deal_reached else None,
        "late_payment_penalty_agreed": bool(late_clause),
        "dynamic_discounting_agreed": bool(dynamic_dd),
        "agreed_dynamic_discount_annual": float(dd_rate),
    }

    score = _reward_engine.compute_terminal_score(task_id, state_kwargs)
    if score is None:
        return f"No grader found for task `{task_id}`"
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

/* --- Terminal score panel tweaks --- */
.score-panel {
    margin-top: 4px;
}

.score-panel .rp-header-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 10px;
}

.score-panel .rp-header-main {
    flex: 1;
}

.score-panel .rp-verdict {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--txt);
    margin-top: 2px;
}

.score-panel .rp-sub {
    font-size: 0.74rem;
    color: var(--txt2);
    margin-top: 4px;
    line-height: 1.5;
}

.rp-score-chip {
    padding: 6px 10px;
    border-radius: 999px;
    font-family: JetBrains Mono, monospace;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid var(--border);
    min-width: 70px;
    text-align: center;
}

.rp-score-chip.positive {
    background: rgba(34,197,94,.10);
    border-color: rgba(34,197,94,.35);
    color: #86EFAC;
}

.rp-score-chip.zero {
    background: rgba(245,158,11,.10);
    border-color: rgba(245,158,11,.35);
    color: #FCD34D;
}

.rp-score-chip.negative {
    background: rgba(239,68,68,.10);
    border-color: rgba(239,68,68,.30);
    color: #FCA5A5;
}

.rp-bar-wrap-with-legend .rp-bar-track {
    position: relative;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--s3);
}

/* ══ GPU HARDWARE MONITOR CARD ═══════════════════════════════════════════ */
.gpu-card { background: #131416; border: 1px solid #2B2E35; border-radius: 12px; padding: 16px; font-family: 'JetBrains Mono', monospace; color: #fff; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); width: 100%; box-sizing: border-box; }
.gpu-head { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2B2E35; padding-bottom: 10px; margin-bottom: 16px; }
.gpu-title { font-size: 0.95rem; font-weight: 700; color: #F3F4F6; font-family: 'Sora', sans-serif; }
.gpu-tag { background: #2B2E35; color: #9CA3AF; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; font-weight: 600; }

.gpu-body { display: grid; grid-template-columns: minmax(130px, 1fr) 2fr; gap: 20px 32px; margin-bottom: 24px; }
.gpu-metric { display: flex; flex-direction: column; gap: 6px; }
.gpu-m-label-row { display: flex; align-items: center; gap: 8px; color: #9CA3AF; font-size: 0.75rem; }
.gpu-m-icon { font-size: 1.1rem; width: 16px; display: flex; justify-content: center; }
.gpu-m-val { font-size: 1.15rem; font-weight: 700; padding-left: 24px; }

.gpu-bar-container { display: flex; flex-direction: column; gap: 6px; }
.gpu-bar-head { display: flex; align-items: center; justify-content: space-between; font-size: 0.75rem; color: #9CA3AF; }
.gpu-bar-label-row { display: flex; align-items: center; gap: 8px; }
.gpu-bar-icon { font-size: 1.1rem; width: 16px; display: flex; justify-content: center; }
.gpu-bar-pct { font-weight: 700; color: #fff; }
.gpu-progress { height: 8px; background: #2B2E35; border-radius: 4px; overflow: hidden; margin-top: 2px; }
.gpu-progress-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
.gpu-progress-fill.load { background: #F59E0B; }
.gpu-progress-fill.memory { background: #3B82F6; }
.gpu-bar-foot { font-size: 0.65rem; color: #6B7280; text-align: left; padding-left: 24px; margin-top: 2px; }

.gpu-footer { display: grid; grid-template-columns: minmax(130px, 1fr) 2fr; gap: 32px; border-top: 1px solid #2B2E35; padding-top: 18px; }
.gpu-f-metric { display: flex; align-items: flex-start; gap: 8px; }
.gpu-f-icon { font-size: 1.1rem; width: 16px; display: flex; justify-content: center; margin-top: 2px; }
.gpu-f-col { display: flex; flex-direction: column; gap: 4px; }
.gpu-f-label { font-size: 0.7rem; color: #9CA3AF; }
.gpu-f-val { font-size: 0.95rem; font-weight: 700; }

.rp-bar-region {
    position: absolute;
    top: 0;
    bottom: 0;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--txt3);
    mix-blend-mode: screen;
}

.rp-bar-low {
    left: 0;
    width: 34%;
    background: linear-gradient(90deg, rgba(239,68,68,.28), transparent);
}

.rp-bar-mid {
    left: 33%;
    width: 34%;
    background: linear-gradient(90deg, rgba(245,158,11,.25), transparent);
}

.rp-bar-high {
    right: 0;
    width: 33%;
    background: linear-gradient(90deg, rgba(34,197,94,.25), transparent);
}

/* --- Reward panel legend & curve hint --- */
.reward-panel .rp-legend {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed var(--border);
}

.rp-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    color: var(--txt3);
}

.rp-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
}

.rp-dot-pos {
    background: #22C55E;
}

.rp-dot-neg {
    background: #EF4444;
}

.rp-curve-hint {
    margin-top: 10px;
    padding: 8px 10px;
    border-radius: var(--r);
    background: rgba(15,23,42,.6);
    border: 1px dashed var(--border);
}

.rp-curve-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--txt2);
    margin-bottom: 4px;
}

.rp-curve-sparkline {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 24px;
    margin-bottom: 4px;
}

.rp-curve-seg {
    flex: 1;
    border-radius: 999px;
    background: linear-gradient(180deg, var(--accent), transparent);
    opacity: 0.55;
}

/* Simple “curved” feel by varying segment heights */
.rp-curve-seg.seg-1 { height: 35%; }
.rp-curve-seg.seg-2 { height: 55%; }
.rp-curve-seg.seg-3 { height: 80%; }
.rp-curve-seg.seg-4 { height: 60%; }
.rp-curve-seg.seg-5 { height: 45%; }

.rp-curve-caption {
    font-size: 0.68rem;
    color: var(--txt3);
    line-height: 1.4;
}

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

/* ══ UPGRADED REWARD PANEL v2 ════════════════════════════════════════════ */
.rp2 {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: var(--rl);
    padding: 16px 18px;
    animation: fade-up .25s ease;
}

/* Idle state */
.rp2-idle { border-style: dashed; }
.rp2-idle-hero {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 0 14px;
}
.rp2-pulse-ring {
    position: relative;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.rp2-pulse-ring::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid rgba(59,130,246,.25);
    animation: rp2-pulse 2s ease-in-out infinite;
}
@keyframes rp2-pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.2); opacity: 0; }
}
.rp2-pulse-icon { font-size: 1.5rem; }
.rp2-idle-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--txt);
    margin-bottom: 4px;
}
.rp2-idle-sub {
    font-size: 0.75rem;
    color: var(--txt2);
    line-height: 1.6;
}
.rp2-idle-sub strong { color: var(--txt); }

/* Legend */
.rp2-legend {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
}
.rp2-legend-title {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--txt3);
    margin-bottom: 6px;
}
.rp2-legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.rp2-lg-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: var(--txt2);
}
.rp2-lg-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.rp2-lg-dot.pos { background: #22C55E; box-shadow: 0 0 6px rgba(34,197,94,.4); }
.rp2-lg-dot.neg { background: #EF4444; box-shadow: 0 0 6px rgba(239,68,68,.4); }

.rp2-legend-compact { margin-top: 8px; padding-top: 8px; }

/* Footer */
.rp2-footer {
    font-size: 0.63rem;
    color: var(--txt3);
    padding-top: 8px;
    border-top: 1px solid var(--border);
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}
.rp2-footer em { color: var(--txt2); font-style: italic; }

/* Active state with gradient accent border */
.rp2-active { border-style: solid; }
.rp2-border-positive {
    border-color: rgba(34,197,94,.25);
    border-left: 3px solid #22C55E;
    background: linear-gradient(135deg, rgba(34,197,94,.03), var(--s1) 40%);
}
.rp2-border-negative {
    border-color: rgba(239,68,68,.25);
    border-left: 3px solid #EF4444;
    background: linear-gradient(135deg, rgba(239,68,68,.03), var(--s1) 40%);
}

/* Metrics grid */
.rp2-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 8px;
}
.rp2-metric-card {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 10px 12px;
    transition: border-color .2s;
}
.rp2-metric-card:hover { border-color: #475569; }
.rp2-mc-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--txt3);
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.rp2-mc-val {
    font-size: 1.05rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}
.rp2-mc-val { color: var(--txt2); }
.rp2-metric-card.positive .rp2-mc-val { color: #86EFAC; }
.rp2-metric-card.negative .rp2-mc-val { color: #FCA5A5; }
.rp2-mc-bar-track {
    height: 4px;
    background: var(--s3);
    border-radius: 999px;
    overflow: hidden;
}
.rp2-mc-bar {
    height: 100%;
    border-radius: 999px;
    transition: width .45s ease;
    min-width: 3px;
}
.rp2-mc-bar.positive { background: linear-gradient(90deg, #16A34A, #22C55E); }
.rp2-mc-bar.negative { background: linear-gradient(90deg, #DC2626, #EF4444); }
.rp2-mc-sub {
    font-size: 0.62rem;
    color: var(--txt3);
    margin-top: 4px;
}

/* ══ UPGRADED SCORE PANEL v2 ═════════════════════════════════════════════ */
.score-panel-v2 {
    margin-top: 6px;
    border-radius: var(--rl);
    animation: fade-up .3s ease;
}
.sp2-top {
    margin-bottom: 14px;
}
.sp2-eyebrow {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--txt3);
    margin-bottom: 8px;
}
.sp2-verdict-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.sp2-verdict-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}
.sp2-verdict-pill.positive {
    background: rgba(34,197,94,.12);
    color: #86EFAC;
    border: 1px solid rgba(34,197,94,.3);
}
.sp2-verdict-pill.zero {
    background: rgba(245,158,11,.10);
    color: #FCD34D;
    border: 1px solid rgba(245,158,11,.3);
}
.sp2-verdict-pill.negative {
    background: rgba(239,68,68,.10);
    color: #FCA5A5;
    border: 1px solid rgba(239,68,68,.3);
}
.sp2-score-chip {
    padding: 4px 10px;
    border-radius: var(--r);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    border: 1px solid var(--border);
}
.sp2-score-chip.positive {
    background: rgba(34,197,94,.08);
    border-color: rgba(34,197,94,.3);
    color: #86EFAC;
}
.sp2-score-chip.zero {
    background: rgba(245,158,11,.08);
    border-color: rgba(245,158,11,.3);
    color: #FCD34D;
}
.sp2-score-chip.negative {
    background: rgba(239,68,68,.08);
    border-color: rgba(239,68,68,.3);
    color: #FCA5A5;
}
.sp2-desc {
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--txt);
    margin-bottom: 2px;
}
.sp2-sub {
    font-size: 0.74rem;
    color: var(--txt2);
    line-height: 1.5;
}

/* Score bar */
.sp2-bar-section { margin-bottom: 14px; }
.sp2-bar-track {
    position: relative;
    height: 12px;
    border-radius: 999px;
    background: var(--s3);
    overflow: visible;
}
.sp2-bar-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    border-radius: 999px;
    transition: width .6s ease;
    min-width: 4px;
}
.sp2-bar-fill.positive { background: linear-gradient(90deg, #16A34A, #22C55E, #34D399); }
.sp2-bar-fill.zero     { background: linear-gradient(90deg, #D97706, #F59E0B, #FBBF24); }
.sp2-bar-fill.negative { background: linear-gradient(90deg, #DC2626, #EF4444, #F87171); }
.sp2-marker {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: var(--txt3);
    opacity: 0.5;
}
.sp2-marker span {
    position: absolute;
    top: -16px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--txt3);
}
.sp2-bar-labels {
    display: flex;
    justify-content: space-between;
    padding: 4px 4px 0;
}
.sp2-bl {
    font-size: 0.58rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.sp2-bl-low  { color: #FCA5A5; }
.sp2-bl-mid  { color: #FCD34D; }
.sp2-bl-high { color: #86EFAC; }

/* Components grid */
.sp2-components {
    margin-top: 4px;
    padding: 10px 0 0;
    border-top: 1px dashed var(--border);
}
.sp2-comp-title {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--txt3);
    margin-bottom: 8px;
}
.sp2-comp-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.sp2-comp-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid transparent;
}
.sp2-w {
    font-weight: 800;
    font-size: 0.72rem;
}
.sp2-comp-chip.solvency   { background: rgba(34,197,94,.10);   color: #86EFAC; border-color: rgba(34,197,94,.25); }
.sp2-comp-chip.liquidity  { background: rgba(59,130,246,.10);  color: #93C5FD; border-color: rgba(59,130,246,.25); }
.sp2-comp-chip.npv        { background: rgba(167,139,250,.10); color: #C4B5FD; border-color: rgba(167,139,250,.25); }
.sp2-comp-chip.compliance { background: rgba(245,158,11,.08);  color: #FBD38D; border-color: rgba(245,158,11,.25); }

.sp2-footer {
    font-size: 0.62rem;
    color: var(--txt3);
    padding-top: 8px;
    border-top: 1px solid var(--border);
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* Reward curve plot container dark-mode override */
.gr-plot, .gradio-container .plot-container {
    background: var(--s1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 8px !important;
    margin-top: 8px !important;
}

/* ══ MOBILE v2 ═══════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
  .rp2-metrics { grid-template-columns: 1fr; }
  .sp2-comp-grid { flex-direction: column; }
  .rp2-idle-hero { flex-direction: column; text-align: center; }
  .sp2-verdict-row { flex-wrap: wrap; }
}
"""


THEME = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(title="OpenEnv SME Negotiator") as demo:

    # ── Global header ────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box" id="sme-header" style="background: var(--bg-surface); border-bottom: 1px solid var(--border); box-shadow: none;">
      <div class="hb-left">
        <div class="hb-eyebrow" style="color: var(--primary);">Shram Setu Copilot</div>
        <h1 class="hb-title" style="font-size: 1.5rem; letter-spacing: -0.5px;">ClearPay - B2B Negotiation Assistant</h1>
        <p class="hb-sub" style="font-size: 0.9rem; color: var(--text-muted);">
            Seamlessly negotiate payment terms with enterprise buyers. 
            Our AI helps you secure liquidity while maintaining strong relationships.
        </p>
      </div>
      <div class="hb-right" style="display: flex; align-items: center; justify-content: flex-end;">
        <div style="background: rgba(16, 185, 129, 0.1); color: var(--success); padding: 8px 16px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(16, 185, 129, 0.2);">
          ✓ System Online
        </div>
      </div>
    </div>
    """)

    session_state = gr.State(SessionStore())

    with gr.Row(equal_height=False):
        # ── LEFT: Active Negotiation (Chat & Actions) ────────────────────
        with gr.Column(scale=6, min_width=480):
            gr.HTML("<div class='panel-label'>💬 Active Negotiation</div>")
            
            chat = gr.Chatbot(
                show_label=False, height=450,
                placeholder="<div style='text-align:center;padding:32px 16px;color:#94A3B8;font-size:1rem;'>"
                            "👋 Welcome to ClearPay Copilot.<br>Click 'Start New Negotiation' to begin.</div>",
            )
            
            gr.HTML("<div class='card-head' style='margin-top: 24px;'>Your Next Move</div>")
            
            with gr.Row():
                ui_action = gr.Dropdown(
                    choices=UI_ACTION_CHOICES, value="propose",
                    label="Action",
                    info="Choose to propose, accept, or reject."
                )
                price_in = gr.Number(value=100.0, label="Price (₹/unit)", minimum=0, step=0.5)
                days_in  = gr.Number(value=90, label="Payment days", minimum=0, maximum=365, precision=0)

            submit_btn = gr.Button("▶  Submit Action", variant="primary", interactive=False, elem_classes=["btn-submit"])

            # Advanced RL Options
            with gr.Accordion("⚙️ Pro Options: Deal Structuring & Setup", open=False):
                with gr.Row():
                    task_sel = gr.Dropdown(choices=list(TASKS.keys()), value=list(TASKS.keys())[0], label="Market Condition (Task Difficulty)")
                    seed_num = gr.Number(value=42, label="Random Seed (Buyer Behavior)", precision=0)
                reason_in  = gr.Textbox(label="Reasoning / Strategy Note", lines=1)
                deal_id_in = gr.Textbox(label="Deal Reference ID")
                with gr.Row():
                    use_treds  = gr.Checkbox(label="Enable TReDS Financing", value=False)
                    late_clause = gr.Checkbox(label="Strict Late-Payment Clause", value=False)
                    propose_dd  = gr.Checkbox(label="Offer Dynamic Discounting", value=False)
                dd_rate     = gr.Slider(value=0.08, label="Discount Rate")
                simulation_plan_tb  = gr.Textbox(value="default", label="Simulation Plan")
                simulation_horizon_n = gr.Number(value=0, label="Simulation Horizon")

        # ── RIGHT: Live Deal Economics ───────────────────────────────────
        with gr.Column(scale=4, min_width=320):
            gr.HTML("<div class='panel-label'>📊 Deal Economics</div>")
            
            reset_btn = gr.Button("+ Start New Negotiation", variant="secondary", elem_classes=["btn-reset"])
            
            gr.HTML("<div style='height: 16px;'></div>")
            status_bar = gr.HTML(value=_status_html(round_number=0, last_price=0.0, done=False))
            contract_btn = gr.DownloadButton("📄 Download Legal Contract", visible=False, variant="secondary", elem_classes=["btn-contract"])

            # Benchmark Artifacts
            with gr.Accordion("🧠 AI Under the Hood: RL Insights", open=False):
                gr.Markdown("<div style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;'>Transparency panel showing the mathematical reinforcement learning signals powering this negotiation.</div>")
                scenario_box = gr.HTML()
                reward_box = gr.HTML()
                reward_plot = gr.Plot()
                last_json = gr.JSON(label="System Observation JSON")

    # ── Bottom tabs ──────────────────────────────────────────────────────
    gr.HTML("<div class='tabs-head' style='margin-top: 32px;'>Developer Tools & API</div>")
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
    reset_outputs = (
        session_state,
        status_bar,
        scenario_box,
        reward_box,
        reward_plot,
        chat,
        last_json,
        price_in,
        days_in,
        submit_btn,
        contract_btn,
    )
    reset_btn.click(fn=reset_episode, inputs=[task_sel, seed_num, session_state], outputs=reset_outputs)

    submit_outputs = (
        session_state,
        status_bar,
        reward_box,
        reward_plot,
        chat,
        last_json,
        price_in,
        days_in,
        submit_btn,
        contract_btn,
    )
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
                css=CUSTOM_CSS,
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
