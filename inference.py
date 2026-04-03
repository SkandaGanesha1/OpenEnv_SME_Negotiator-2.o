#!/usr/bin/env python3
"""Baseline inference runner for the SME negotiation environment."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.models import NegotiationAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _observation_to_dict(observation: Any) -> Dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return dict(observation)


def choose_action(observation: Any, round_number: int, agent_days: int) -> tuple[NegotiationAction, int]:
    """Choose the next action using the current observation only."""

    difficulty = str(getattr(observation, "difficulty", "")).upper()
    print(
        f"  accept check: buyer_days={observation.buyer_days}, "
        f"liquidity_threshold={observation.liquidity_threshold}"
    )

    if difficulty == "EASY" and round_number == 0:
        next_agent_days = max(observation.liquidity_threshold, observation.buyer_days)
        return (
            NegotiationAction(
                action_type="propose",
                price=round(max(observation.cost_threshold + 2.0, observation.buyer_price * 0.995), 2),
                payment_days=next_agent_days,
                use_treds=False,
                reason="Easy-task opening offer",
            ),
            next_agent_days,
        )

    if difficulty == "EASY" and round_number == 1 and observation.buyer_price > observation.cost_threshold:
        return (
            NegotiationAction(
                action_type="accept",
                price=observation.buyer_price,
                payment_days=observation.buyer_days,
                use_treds=False,
                reason="Easy-task acceptance after one counter-offer",
            ),
            agent_days,
        )

    if observation.buyer_price > observation.cost_threshold and observation.buyer_days <= observation.liquidity_threshold:
        return (
            NegotiationAction(
                action_type="accept",
                price=observation.buyer_price,
                payment_days=observation.buyer_days,
                use_treds=False,
                reason="Current offer is viable",
            ),
            agent_days,
        )

    if round_number >= 4 and observation.buyer_days > observation.liquidity_threshold * 2:
        target_days = observation.liquidity_threshold + 15
        return (
            NegotiationAction(
                action_type="propose",
                price=round(max(observation.cost_threshold + 1.0, observation.buyer_price * 0.95), 2),
                payment_days=target_days,
                use_treds=True,
                reason="TReDS-enabled proposal to unlock shorter payment terms",
            ),
            target_days,
        )

    if round_number >= 6 and observation.buyer_days > observation.liquidity_threshold + 3:
        target_days = observation.liquidity_threshold + 15
        return (
            NegotiationAction(
                action_type="propose",
                price=round(max(observation.cost_threshold + 1.0, observation.buyer_price * 0.98), 2),
                payment_days=target_days,
                use_treds=True,
                reason="Late-round TReDS proposal to pull buyer days toward liquidity threshold",
            ),
            target_days,
        )

    if round_number == 0:
        agent_days = max(observation.liquidity_threshold, observation.buyer_days // 2)
    else:
        agent_days = max(observation.liquidity_threshold, agent_days - 8)

    return (
        NegotiationAction(
            action_type="propose",
            price=round(max(observation.cost_threshold + 2.0, observation.buyer_price * 0.995), 2),
            payment_days=agent_days,
            use_treds=False,
            reason="Counter-offer focused on payment-term reduction",
        ),
        agent_days,
    )


async def run_episode(env: SMENegotiatorEnv, difficulty: str, seed: int) -> Dict[str, Any]:
    """Run one episode with a state-aware heuristic policy."""

    result = await env.reset(seed=seed, difficulty=difficulty)
    observation = result.observation
    round_number = 0
    step_rewards: List[float] = []
    total_reward = 0.0

    print("\n" + "=" * 72)
    print(f"EPISODE START: {difficulty} | seed={seed}")
    print("=" * 72)
    print(
        f"Initial buyer: ₹{observation.buyer_price:.2f} / unit @ {observation.buyer_days} days "
        f"| volume={observation.volume}"
    )
    print(
        f"Thresholds: cost=₹{observation.cost_threshold:.2f}, "
        f"liquidity={observation.liquidity_threshold} days"
    )

    agent_days = max(observation.liquidity_threshold, observation.buyer_days // 2)

    while not result.done and round_number < observation.max_rounds:
        action, agent_days = choose_action(observation, round_number, agent_days)
        print(
            f"Round {round_number + 1}/{observation.max_rounds}: "
            f"{action.action_type.upper()} price=₹{action.price:.2f}, "
            f"days={action.payment_days}, treds={action.use_treds}"
        )

        result = await env.step(action)
        observation = result.observation
        reward = float(result.reward or 0.0)
        total_reward += reward
        step_rewards.append(reward)

        print(
            f"  buyer -> ₹{observation.buyer_price:.2f} / unit @ {observation.buyer_days} days"
        )
        print(
            f"  step reward={reward:.4f} | cumulative={total_reward:.4f} | done={result.done}"
        )

        round_number += 1

    final_score = float(result.reward or 0.0)
    success = bool(result.done and final_score > 0.0)

    print("-" * 72)
    print(
        f"Episode summary: final_score={final_score:.4f}, total_reward={total_reward:.4f}, "
        f"steps={round_number}, success={success}"
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

    print("=" * 72)
    print("SME NEGOTIATION BASELINE INFERENCE")
    print("=" * 72)
    print(f"API_BASE_URL={API_BASE_URL}")

    results: Dict[str, Any] = {
        "metadata": {
            "api_base_url": API_BASE_URL,
            "timestamp": datetime.utcnow().isoformat(),
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

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for difficulty, task_data in results["tasks"].items():
        summary = task_data["summary"]
        print(
            f"{difficulty}: mean_score={summary['mean_final_score']:.4f}, "
            f"mean_reward={summary['mean_total_reward']:.4f}, "
            f"success_rate={summary['success_rate']:.2%}"
        )
    print(
        f"Overall: mean_score={results['summary']['overall_mean_score']:.4f}, "
        f"mean_reward={results['summary']['overall_mean_reward']:.4f}, "
        f"success_rate={results['summary']['overall_success_rate']:.2%}"
    )
    print("Results saved to inference_results.json")


if __name__ == "__main__":
    asyncio.run(main())
