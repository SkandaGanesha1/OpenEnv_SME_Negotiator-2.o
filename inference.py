#!/usr/bin/env python3
"""
OpenEnv SME Negotiation - Baseline Inference Script
Demonstrates agent learning through the OpenEnv SME environment.

Uses OpenAI GPT-4 to negotiate B2B contracts optimally.
Required environment variables:
  - OPENAI_API_KEY: Your OpenAI API key
  - API_BASE_URL: Server URL (default: http://localhost:8000)
  - MODEL_NAME: Model to use (default: gpt-4o)
"""

import os
import json
import asyncio
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
from dataclasses import dataclass

from openai import AsyncOpenAI, OpenAI

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Validation
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class InferenceResult:
    """Results from inference on a task."""
    task_id: str
    score: float
    steps: int
    episodes: int
    avg_reward: float
    std_reward: float
    success_rate: float


class SMENegotiationAgent:
    """AI Agent for B2B contract negotiation using OpenAI LLM."""
    
    def __init__(self, server_url: str = API_BASE_URL):
        self.server_url = server_url
        self.client = httpx.Client(timeout=30.0)
        self.conversation_history = []
    
    def build_system_prompt(self, task_id: str) -> str:
        """Build task-specific system prompt."""
        base = """You are an experienced SME (small-medium enterprise) business manager 
negotiating a B2B contract with a corporate buyer. Your goal is to maximize profitability 
while ensuring business survival.

Key constraints you must respect:
1. Production cost limits profitability (don't go below cost)
2. Payment terms affect working capital (liquidity crisis if days > threshold)
3. MSMED Act Section 43B: payments must be within 45 days OR use TReDS platform
4. TReDS platform enables factoring for immediate liquidity (costs ~2% friction)

Financial model you use:
  NPV = (Price - Cost) × Volume × (1 / (1 + rate)^(Days/365))
  
Strategy notes:
- Early rounds: propose aggressive counters to test buyer limits
- Middle rounds: move toward mutually acceptable terms
- Late rounds: close deal if acceptable, walk away if not viable

Output format: Return ONLY valid JSON (no markdown, no explanation):
{
    "action_type": "PROPOSE" | "ACCEPT" | "REJECT",
    "proposed_price": <float> (if PROPOSE),
    "proposed_days": <int> (if PROPOSE),
    "request_treds": <bool>,
    "justification": "<your reasoning>"
}
"""
        
        if task_id == "easy":
            return base + "\nTASK: Easy - Price negotiation with fixed 30-day terms."
        elif task_id == "medium":
            return base + "\nTASK: Medium - Negotiate both price and payment days."
        elif task_id == "hard":
            return base + """
TASK: Hard - Expert negotiation with extreme constraints.
WARNING: Survival depends on TReDS or aggressive pricing.
Be prepared to use TReDS (request_treds=true) to unlock liquidity relief."""
        
        return base
    
    def build_user_prompt(self, state: Dict[str, Any], round_num: int) -> str:
        """Build round-specific user prompt."""
        history_text = "Negotiation history:\n"
        for record in state.get("history", []):
            party = "You" if record["party"] == "agent" else "Buyer"
            history_text += f"  R{record['round']} ({party}): "
            history_text += f"₹{record['proposed_price']}, {record['proposed_days']}d"
            if record.get("request_treds"):
                history_text += " [TReDS]"
            history_text += "\n"
        
        return f"""Round {round_num + 1}/{state['t_max']}

Current Buyer Offer:
  Price: ₹{state['p_opp']}/unit
  Days: {state['d_opp']}
  Volume: {state['v_opp']} units

Your Constraints:
  Cost: ₹{state['c_sme']}/unit (FLOOR)
  Liquidity Threshold: {state['l_sme']} days (CRITICAL)
  Days Until Crisis: {state['l_sme'] - state['t_elapsed']} days

{history_text}

Decide: PROPOSE counter, ACCEPT if agreeable, or REJECT if impossible.
JSON response only:"""
    
    def reset_episode(self, task_id: str = "easy", seed: Optional[int] = None) -> Dict:
        """Reset environment for new episode."""
        response = self.client.post(
            f"{self.server_url}/reset",
            json={"task_id": task_id, "seed": seed}
        )
        response.raise_for_status()
        return response.json()
    
    def step_episode(self, action: Dict) -> Dict:
        """Step environment with action."""
        response = self.client.post(
            f"{self.server_url}/step",
            json=action
        )
        response.raise_for_status()
        return response.json()
    
    def get_llm_action(self, system_prompt: str, user_prompt: str) -> Dict:
        """Get action from OpenAI LLM."""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON action
            action_dict = json.loads(content)
            return action_dict
        
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {content}")
            # Return safe fallback
            return {
                "action_type": "PROPOSE",
                "proposed_price": 95,
                "proposed_days": 30,
                "request_treds": False,
                "justification": "Fallback action"
            }
        except Exception as e:
            print(f"LLM error: {e}")
            return {
                "action_type": "PROPOSE",
                "proposed_price": 95,
                "proposed_days": 30,
                "request_treds": False,
                "justification": "Error recovery"
            }
    
    async def run_episode(self, task_id: str = "easy", seed: Optional[int] = None) -> Dict:
        """Run single episode and return score with visible step rewards."""
        system_prompt = self.build_system_prompt(task_id)
        
        # Reset
        obs = self.reset_episode(task_id=task_id, seed=seed)
        
        total_reward = 0.0
        step_count = 0
        max_steps = obs.get("t_max", 12)
        step_rewards = []  # Track rewards at each step
        
        print(f"\n{'='*70}")
        print(f"🎬 EPISODE START: {task_id.upper()} | Seed: {seed}")
        print(f"{'='*70}")
        print(f"Initial Buyer Offer: ₹{obs.get('p_opp', 0)}/unit @ {obs.get('d_opp', 0)} days | Vol: {obs.get('v_opp', 0)} units")
        print(f"Your Cost Threshold: ₹{obs.get('c_sme', 0)}/unit | Liquidity Threshold: {obs.get('l_sme', 0)} days")
        print(f"{'─'*70}\n")
        
        while step_count < max_steps:
            # Build prompt
            user_prompt = self.build_user_prompt(obs, step_count)
            
            # Get LLM action
            action = self.get_llm_action(system_prompt, user_prompt)
            
            print(f"📍 ROUND {step_count + 1}/{max_steps}")
            print(f"   Action: {action['action_type']:<12}", end="")
            if action['action_type'] == 'PROPOSE':
                print(f" | Price: ₹{action.get('proposed_price', 'N/A'):<7.2f} | Days: {action.get('proposed_days', 'N/A'):<3} | " +
                      f"TReDS: {action.get('request_treds', False)}")
            elif action['action_type'] == 'ACCEPT':
                print(f" | Accepting buyer offer ₹{obs.get('p_opp', 0):.2f} @ {obs.get('d_opp', 0)} days")
            else:
                print(f" | Rejecting offer")
            
            # Step environment
            result = self.step_episode(action)
            obs = result.get("observation", {})
            reward = result.get("reward", 0.0)
            terminated = result.get("terminated", False)
            
            step_rewards.append(reward)
            total_reward += reward
            step_count += 1
            
            # *** CRITICAL VISIBLE REWARD DISPLAY ***
            print(f"   ✓ Step Reward:     +{reward:8.6f}")
            print(f"   ✓ Cumulative:      {total_reward:8.6f}")
            print(f"   → Buyer Counter:   ₹{obs.get('p_opp', 0):.2f}/unit @ {obs.get('d_opp', 0)} days")
            
            print(f"{'─'*70}\n")
            
            if terminated:
                print(f"✓ Episode Completed in {step_count} steps")
                break
        
        avg_reward = total_reward / step_count if step_count > 0 else 0.0
        
        # Final Summary (HIGHLY VISIBLE)
        print(f"\n{'='*70}")
        print(f"📊 EPISODE SUMMARY")
        print(f"{'='*70}")
        print(f"Task:              {task_id.upper()}")
        print(f"Seed:              {seed}")
        print(f"Steps Completed:   {step_count}/{max_steps}")
        print(f"\n💰 REWARD BREAKDOWN:")
        print(f"  Final Round Score:     {reward:8.6f}")
        print(f"  Cumulative Reward:     {total_reward:8.6f}  ← TOTAL EARNED")
        print(f"  Average Reward/Step:   {avg_reward:8.6f}")
        print(f"  Reward History:        {[f'{r:.4f}' for r in step_rewards]}")
        success = reward > 0.0
        print(f"\n{'Status:            ✅ SUCCESS' if success else 'Status:            ❌ FAILED'}")
        print(f"{'='*70}\n")
        
        return {
            "task_id": task_id,
            "seed": seed,
            "final_score": reward,
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "step_rewards": step_rewards,
            "steps": step_count,
            "success": success
        }


async def main():
    """Run baseline inference on all tasks with visible reward logging."""
    
    print("\n" + "="*70)
    print("  OPENENV SME NEGOTIATION - BASELINE INFERENCE WITH REWARDS")
    print("="*70)
    print("NOTE: All step-by-step rewards are displayed below\n")
    
    agent = SMENegotiationAgent(server_url=API_BASE_URL)
    
    # Test connectivity
    try:
        response = agent.client.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"ERROR: Server not responding (status {response.status_code})")
            print(f"Make sure server is running at {API_BASE_URL}")
            sys.exit(1)
        print(f"✓ Connected to server at {API_BASE_URL}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {API_BASE_URL}")
        print(f"  {e}")
        sys.exit(1)
    
    # Run evaluation on all tasks
    results = {
        "metadata": {
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "timestamp": str(datetime.now()) if 'datetime' in dir() else "unknown"
        },
        "tasks": {}
    }
    
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'#'*70}")
        print(f"# Starting {task_id.upper()} task - 3 episodes")
        print(f"{'#'*70}\n")
        
        episode_results = []
        
        for episode in range(3):  # 3 episodes per task
            result = await agent.run_episode(task_id=task_id, seed=1000 + episode)
            episode_results.append(result)
        
        # Compute statistics
        scores = [r["final_score"] for r in episode_results]
        total_rewards = [r["total_reward"] for r in episode_results]
        
        task_summary = {
            "episodes": episode_results,
            "statistics": {
                "final_scores": {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "values": scores
                },
                "total_rewards": {
                    "mean": sum(total_rewards) / len(total_rewards),
                    "total": sum(total_rewards),
                    "values": total_rewards
                },
                "success_rate": sum(1 for r in episode_results if r["success"]) / len(episode_results)
            }
        }
        
        results["tasks"][task_id] = task_summary
    
    # Print final summary
    print("\n" + "="*70)
    print("  BASELINE INFERENCE FINAL SUMMARY")
    print("="*70 + "\n")
    
    all_scores = []
    all_rewards = []
    
    for task_id, data in results["tasks"].items():
        stats = data["statistics"]
        print(f"\n{task_id.upper()} Task:")
        print(f"  Episodes:        {len(data['episodes'])}")
        print(f"  Final Scores:    Mean {stats['final_scores']['mean']:.4f} "
              f"[{stats['final_scores']['min']:.4f}, {stats['final_scores']['max']:.4f}]")
        print(f"  Total Rewards:   Mean {stats['total_rewards']['mean']:.4f} "
              f"(Total: {stats['total_rewards']['total']:.4f})")
        print(f"  Success Rate:    {stats['success_rate']*100:.1f}%")
        print(f"  Reward Scores:   {[f'{s:.4f}' for s in stats['final_scores']['values']]}")
        print(f"  Cumul. Rewards:  {[f'{r:.4f}' for r in stats['total_rewards']['values']]}")
        
        all_scores.extend(stats['final_scores']['values'])
        all_rewards.extend(stats['total_rewards']['values'])
    
    print(f"\n{'─'*70}")
    print(f"\nOVERALL METRICS:")
    print(f"  Total Episodes:      {sum(len(data['episodes']) for data in results['tasks'].values())}")
    print(f"  Overall Avg Score:   {sum(all_scores) / len(all_scores):.4f}")
    print(f"  Overall Total Reward:{sum(all_rewards):.4f}")
    print(f"  Global Success Rate: {sum(1 for s in all_scores if s > 0) / len(all_scores) * 100:.1f}%")
    
    # Save results to JSON
    results_file = "inference_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {results_file}")
    except Exception as e:
        print(f"\n⚠ Could not save results: {e}")
    
    print("\n" + "="*70)
    print("✓ Baseline inference complete - all rewards visible above")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
