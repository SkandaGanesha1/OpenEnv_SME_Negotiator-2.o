#!/usr/bin/env python3
"""Inference runner for the SME negotiation environment."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.models import NegotiationAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

NEGOTIATION_SYSTEM_PROMPT = (
    "You are a B2B negotiation assistant. Respond ONLY with valid JSON containing "
    "keys: action_type, price, payment_days, use_treds, reason. "
    "action_type must be one of: propose, accept, reject."
)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


def _observation_to_dict(observation: Any) -> Dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return dict(observation)


def format_observation(obs: Dict[str, Any]) -> str:
    return (
        f"Round={obs.get('round_number')} | BuyerPrice={obs.get('buyer_price')} | "
        f"BuyerDays={obs.get('buyer_days')} | LiquidityThreshold={obs.get('liquidity_threshold')} | "
        f"CostThreshold={obs.get('cost_threshold')}"
    )


def _safe_fallback_action(observation: Any) -> Dict[str, Any]:
    return {
        "action_type": "propose",
        "price": round(max(float(observation.cost_threshold) + 1.0, float(observation.buyer_price) * 0.99), 2),
        "payment_days": int(max(int(observation.liquidity_threshold), int(observation.buyer_days) - 5)),
        "use_treds": bool(int(observation.buyer_days) > int(observation.liquidity_threshold) + 20),
        "reason": "Fallback action due to model output issue",
    }


def get_agent_action(observation: Dict[str, Any], history: List[dict], task_name: str) -> Dict[str, Any]:
    user_message = (
        f"Task={task_name}\n"
        f"Current observation:\n{format_observation(observation)}\n"
        "Return only JSON action."
    )

    messages = [{"role": "system", "content": NEGOTIATION_SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=180,
    )

    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    action = json.loads(raw)
    return {
        "action_type": str(action.get("action_type", "propose")).lower(),
        "price": float(action.get("price", observation.get("buyer_price", 0.0))),
        "payment_days": int(action.get("payment_days", observation.get("buyer_days", 0))),
        "use_treds": bool(action.get("use_treds", False)),
        "reason": str(action.get("reason", "")),
    }


def _to_model_action(action_payload: Dict[str, Any], observation: Any) -> NegotiationAction:
    action_type = str(action_payload.get("action_type", "propose")).lower()
    if action_type not in {"propose", "accept", "reject"}:
        action_type = "propose"

    price = float(action_payload.get("price", observation.buyer_price))
    payment_days = int(action_payload.get("payment_days", observation.buyer_days))
    use_treds = bool(action_payload.get("use_treds", False))
    reason = str(action_payload.get("reason", "Model-selected action"))

    return NegotiationAction(
        action_type=action_type,
        price=round(price, 2),
        payment_days=payment_days,
        use_treds=use_treds,
        reason=reason,
    )


async def run_episode(env: SMENegotiatorEnv, difficulty: str, seed: int) -> Dict[str, Any]:
    """Run one episode using model-guided actions with strict stdout formatting."""

    task_name = difficulty.lower()
    episode_id = f"{task_name}-{seed}"
    history: List[dict] = []

    result = await env.reset(seed=seed, difficulty=difficulty, episode_id=episode_id, task_name=task_name)
    observation = result.observation

    print(
        f"[START] EPISODE_ID={episode_id} TASK={task_name} "
        f"SEED={seed} OBS={json.dumps(_observation_to_dict(observation), ensure_ascii=True)}",
        flush=True,
    )

    round_number = 0
    step_rewards: List[float] = []
    total_reward = 0.0

    while not result.done and round_number < observation.max_rounds:
        obs_dict = _observation_to_dict(observation)

        try:
            action_payload = get_agent_action(obs_dict, history, task_name)
        except Exception:
            action_payload = _safe_fallback_action(observation)

        action = _to_model_action(action_payload, observation)

        print(
            f"[STEP] EPISODE_ID={episode_id} ROUND={round_number + 1} "
            f"ACTION={json.dumps(action.model_dump(), ensure_ascii=True)} "
            f"OBS={json.dumps(obs_dict, ensure_ascii=True)}",
            flush=True,
        )

        history.append({"role": "user", "content": format_observation(obs_dict)})
        history.append({"role": "assistant", "content": json.dumps(action_payload, ensure_ascii=True)})

        result = await env.step(action)
        observation = result.observation
        reward = float(result.reward or 0.0)
        total_reward += reward
        step_rewards.append(reward)
        round_number += 1

    final_score = float(result.reward or 0.0)
    success = bool(result.done and final_score > 0.0)

    print(
        f"[END] EPISODE_ID={episode_id} FINAL_SCORE={final_score} "
        f"TOTAL_REWARD={total_reward} STEPS={round_number}",
        flush=True,
    )

    return {
        "difficulty": difficulty,
        "seed": seed,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps": round_number,
        "success": success,
        "step_rewards": step_rewards,
        "final_observation": _observation_to_dict(observation),
    }


async def main() -> None:
    """Run three episodes per difficulty and write a compact results file."""

    results: Dict[str, Any] = {
        "metadata": {
            "api_base_url": API_BASE_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
        },
        "tasks": {},
    }

    async with SMENegotiatorEnv(base_url=API_BASE_URL) as env:
        for difficulty in ["EASY", "MEDIUM", "HARD"]:
            episode_results: List[Dict[str, Any]] = []
            for seed in [1000, 1001, 1002]:
                episode_results.append(await run_episode(env, difficulty, seed))

            scores = [episode["final_score"] for episode in episode_results]
            rewards = [episode["total_reward"] for episode in episode_results]
            successes = [episode["success"] for episode in episode_results]

            results["tasks"][difficulty] = {
                "episodes": episode_results,
                "summary": {
                    "mean_final_score": sum(scores) / len(scores),
                    "mean_total_reward": sum(rewards) / len(rewards),
                    "success_rate": sum(1 for success in successes if success) / len(successes),
                },
            }

    overall_scores = [
        episode["final_score"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    overall_rewards = [
        episode["total_reward"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    overall_successes = [
        episode["success"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]

    results["summary"] = {
        "overall_mean_score": sum(overall_scores) / len(overall_scores),
        "overall_mean_reward": sum(overall_rewards) / len(overall_rewards),
        "overall_success_rate": sum(1 for success in overall_successes if success) / len(overall_successes),
    }

    with open("inference_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
