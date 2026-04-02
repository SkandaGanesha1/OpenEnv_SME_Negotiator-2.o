#!/usr/bin/env python3
"""
OpenEnv SME Negotiation - Baseline Inference Script
Demonstrates agent learning through the OpenEnv SME environment.

Uses OpenAI GPT-4 to negotiate B2B contracts optimally.
Required environment variables:
    - OPENAI_API_KEY: Your OpenAI API key (or leave empty and use HF_TOKEN)
  - API_BASE_URL: Server URL (default: http://localhost:8000)
  - MODEL_NAME: Model to use (default: gpt-4o)
    - LLM_BASE_URL: Optional OpenAI-compatible model endpoint URL
"""

import os
import json
import asyncio
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
from dataclasses import dataclass
from dotenv import load_dotenv

from openai import AsyncOpenAI, OpenAI

# Auto-load variables from project root .env file if present.
load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
HF_ROUTER_BASE_URL = os.getenv("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# Choose provider/token deterministically to avoid auth mismatches.
if LLM_BASE_URL:
    llm_base_lower = LLM_BASE_URL.lower()
    if "huggingface" in llm_base_lower:
        if not HF_TOKEN:
            print("ERROR: LLM_BASE_URL points to Hugging Face but HF_TOKEN is not set")
            sys.exit(1)
        API_TOKEN = HF_TOKEN
        ACTIVE_MODEL = HF_MODEL_NAME or MODEL_NAME
        client = OpenAI(api_key=API_TOKEN, base_url=LLM_BASE_URL)
        ACTIVE_PROVIDER = "huggingface"
    else:
        if not OPENAI_API_KEY:
            print("ERROR: Non-HF LLM_BASE_URL requires OPENAI_API_KEY")
            sys.exit(1)
        API_TOKEN = OPENAI_API_KEY
        ACTIVE_MODEL = MODEL_NAME
        client = OpenAI(api_key=API_TOKEN, base_url=LLM_BASE_URL)
        ACTIVE_PROVIDER = "custom-openai-compatible"
else:
    # No explicit LLM base URL: prefer OpenAI, else default to HF router.
    if OPENAI_API_KEY:
        API_TOKEN = OPENAI_API_KEY
        ACTIVE_MODEL = MODEL_NAME
        client = OpenAI(api_key=API_TOKEN)
        ACTIVE_PROVIDER = "openai"
    elif HF_TOKEN:
        API_TOKEN = HF_TOKEN
        ACTIVE_MODEL = HF_MODEL_NAME or MODEL_NAME
        client = OpenAI(api_key=API_TOKEN, base_url=HF_ROUTER_BASE_URL)
        ACTIVE_PROVIDER = "huggingface"
    else:
        print("ERROR: Set OPENAI_API_KEY or HF_TOKEN in your environment/.env file")
        sys.exit(1)

# Secondary failover client: if OpenAI key hits quota and HF token exists, retry via HF router.
hf_fallback_client = None
if HF_TOKEN:
    # Avoid creating duplicate client when primary is already HF router/custom HF base URL.
    if not LLM_BASE_URL or "huggingface" not in LLM_BASE_URL.lower():
        hf_fallback_client = OpenAI(api_key=HF_TOKEN, base_url=HF_ROUTER_BASE_URL)


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
        self.hf_fallback_disabled = False
        self.llm_disabled = False
        self.llm_disabled_reason = ""

    def heuristic_action(self, state: Dict[str, Any], task_id: str) -> Dict:
        """Deterministic heuristic policy used when remote LLM is unavailable."""
        price = float(state.get("p_opp", 95.0))
        days = int(state.get("d_opp", 30))
        cost = float(state.get("c_sme", 80.0))
        round_idx = int(state.get("t_elapsed", 0))
        max_rounds = max(1, int(state.get("t_max", 8)))
        liquidity_limit = int(state.get("l_sme", 60))

        # Accept late-round profitable offers when they are liquidity-safe,
        # or can be made liquidity-safe using TReDS on medium/hard tasks.
        if round_idx >= max_rounds - 3 and price >= cost * 1.02:
            can_accept_direct = days <= liquidity_limit
            can_accept_with_treds = task_id in {"medium", "hard"} and days <= 120
            if can_accept_direct or can_accept_with_treds:
                use_treds = bool(can_accept_with_treds and days > liquidity_limit)
                return {
                    "action_type": "ACCEPT",
                    "proposed_price": price,
                    "proposed_days": days,
                    "request_treds": use_treds,
                    "justification": "Heuristic accept: profitable and liquidity-feasible"
                }

        # For hard task, early explicit TReDS restructuring helps unlock utility.
        if task_id == "hard" and round_idx <= 2 and price >= cost * 1.01:
            return {
                "action_type": "PROPOSE",
                "proposed_price": round(max(cost * 1.03, price * 0.97), 2),
                "proposed_days": min(days, 60),
                "request_treds": True,
                "justification": "Heuristic TReDS restructuring to manage long payment cycle"
            }

        # Conservative counter-offer with slight concessions over time.
        concession = min(0.15, 0.02 * (round_idx + 1))
        target_price = max(cost * 1.02, price * (1.0 - concession))
        target_days = days
        use_treds = False

        if task_id in {"medium", "hard"}:
            target_days = min(days, 45 if task_id == "medium" else 60)
            if days > liquidity_limit:
                use_treds = True

        return {
            "action_type": "PROPOSE",
            "proposed_price": round(target_price, 2),
            "proposed_days": int(max(1, min(365, target_days))),
            "request_treds": use_treds,
            "justification": "Heuristic counter-offer for stable progression"
        }
    
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
        payload = response.json()
        return payload.get("observation", payload.get("state", payload))
    
    def step_episode(self, action: Dict) -> Dict:
        """Step environment with action."""
        response = self.client.post(
            f"{self.server_url}/step",
            json={"action": action}
        )
        response.raise_for_status()
        payload = response.json()
        # Keep backward compatibility with alternate field names.
        if "done" in payload and "terminated" not in payload:
            payload["terminated"] = payload.get("done", False)
        return payload
    
    def get_llm_action(self, system_prompt: str, user_prompt: str, state: Dict[str, Any], task_id: str) -> Dict:
        """Get action from OpenAI LLM."""
        if self.llm_disabled:
            return self.heuristic_action(state, task_id)

        content = ""
        try:
            response = client.chat.completions.create(
                model=ACTIVE_MODEL,
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
            return self.normalize_action(action_dict, state, task_id)
        
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {content}")
            return self.normalize_action(self.heuristic_action(state, task_id), state, task_id)
        except Exception as e:
            err_text = str(e)
            # Auto-failover to HF when OpenAI quota is exhausted.
            if (
                hf_fallback_client is not None
                and not self.hf_fallback_disabled
                and (
                "insufficient_quota" in err_text or "Error code: 429" in err_text
                )
            ):
                try:
                    print("LLM quota exhausted on primary provider. Retrying via Hugging Face Router...")
                    response = hf_fallback_client.chat.completions.create(
                        model=HF_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    content = response.choices[0].message.content
                    action_dict = json.loads(content)
                    return action_dict
                except Exception as hf_err:
                    print(f"HF fallback also failed: {hf_err}")
                    hf_err_text = str(hf_err)
                    if (
                        "model_not_found" in hf_err_text
                        or "does not exist" in hf_err_text
                        or "insufficient permissions" in hf_err_text
                        or "Error code: 401" in hf_err_text
                        or "Error code: 402" in hf_err_text
                        or "Error code: 403" in hf_err_text
                    ):
                        self.hf_fallback_disabled = True
                        print("HF fallback disabled for this run. Check HF_MODEL_NAME/token permissions.")

            if (
                "Error code: 401" in err_text
                or "Error code: 402" in err_text
                or "Error code: 403" in err_text
                or "insufficient_quota" in err_text
                or "Error code: 429" in err_text
            ):
                self.llm_disabled = True
                self.llm_disabled_reason = err_text
                print("Primary LLM disabled for this run due to provider auth/billing limits. Using heuristic policy.")
            print(f"LLM error: {e}")
            return self.normalize_action(self.heuristic_action(state, task_id), state, task_id)

    def normalize_action(self, action: Dict[str, Any], state: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Safety layer to keep actions realistic and prevent self-sabotage."""
        price = float(state.get("p_opp", 100.0))
        days = int(state.get("d_opp", 30))
        cost = float(state.get("c_sme", 80.0))
        liquidity = int(state.get("l_sme", 60))

        action_type = str(action.get("action_type", "PROPOSE")).upper()
        proposed_price = float(action.get("proposed_price", price))
        proposed_days = int(action.get("proposed_days", days))
        request_treds = bool(action.get("request_treds", False))

        # If current offer is already viable, don't reject it blindly.
        if action_type == "REJECT":
            if price >= cost * 1.02 and (days <= liquidity or task_id in {"medium", "hard"}):
                action_type = "ACCEPT"
                proposed_price = price
                proposed_days = days
                request_treds = days > liquidity and task_id in {"medium", "hard"}

        # Keep proposals in a realistic band around current negotiation zone.
        if action_type == "PROPOSE":
            floor_price = max(cost * 1.01, price * 0.90)
            ceil_price = price * 1.08
            proposed_price = max(floor_price, min(ceil_price, proposed_price))

            if task_id == "easy":
                proposed_days = 30
                request_treds = False
            elif task_id == "medium":
                proposed_days = max(30, min(60, proposed_days))
            else:
                proposed_days = max(30, min(90, proposed_days))
                if proposed_days > liquidity:
                    request_treds = True

        return {
            "action_type": action_type,
            "proposed_price": round(proposed_price, 2),
            "proposed_days": int(proposed_days),
            "request_treds": request_treds,
            "justification": action.get("justification", "Policy-normalized action")
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
            action = self.get_llm_action(system_prompt, user_prompt, obs, task_id)
            
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
            obs = result.get("observation", result.get("state", {}))
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
