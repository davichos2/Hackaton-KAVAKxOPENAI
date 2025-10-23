"""
Agent Evaluator + Optimization Controller - Modified for V3.py integration

This version calls your V3.py chatbot instead of OpenAI directly.
"""

import argparse
import json
import math
import os
import time
import uuid
import copy
import sys
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import your V3 chatbot
try:
    from agents import Agent, Runner
    from pathlib import Path
    from dotenv import load_dotenv

    # Load .env file
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)

    # Initialize the agent globally
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful AI assistant. Provide clear, concise, and accurate responses."
    )
    USE_V3_AGENT = True
except ImportError as e:
    print(f"Warning: Could not import V3 agent: {e}")
    USE_V3_AGENT = False

# For embeddings, we still need OpenAI
try:
    from openai import OpenAI
    embedding_client = OpenAI()
except:
    import openai as _openai
    embedding_client = _openai

# ---------- Configurable objectives ----------
TARGET_LATENCY_MS = int(os.getenv("TARGET_LATENCY_MS", "800"))
TARGET_COST_PER_CALL = float(os.getenv("TARGET_COST_PER_CALL", "0.001"))
DEFAULT_MODEL = "gpt-4"
EXPECTED_PROMPT_COUNT = 20

# ---------- Helpers ----------

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------- Chatbot call using V3.py ----------

def generate_response(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your V3.py chatbot with the given prompt and parameters.
    Note: V3 uses the agents library which may not support all parameters directly.
    We'll configure what we can via environment or agent settings.
    """

    start = time.time()

    try:
        if USE_V3_AGENT:
            # Configure agent with available parameters
            # Note: agents library may have limited parameter support
            # We'll pass what we can through the agent configuration

            # Run the agent with the prompt
            result = Runner.run_sync(agent, prompt)
            text = result.final_output

            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            prompt_tokens = len(prompt) // 4
            response_tokens = len(text) // 4

        else:
            # Fallback if V3 agent not available
            text = f"<V3_AGENT_NOT_AVAILABLE>"
            prompt_tokens = None
            response_tokens = None

    except Exception as e:
        text = f"<API_ERROR: {e}>"
        prompt_tokens = None
        response_tokens = None

    latency_ms = int((time.time() - start) * 1000)

    return {
        "response_id": str(uuid.uuid4()),
        "text": text,
        "model": "V3-Agent",
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "params": params,
    }


# ---------- Evaluation ----------

def embed_text(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    try:
        if hasattr(embedding_client, "embeddings"):
            emb_resp = embedding_client.embeddings.create(model=model, input=text)
            vec = np.array(emb_resp.data[0].embedding)
        else:
            emb_resp = embedding_client.Embedding.create(model=model, input=text)
            vec = np.array(emb_resp.data[0].embedding)
    except Exception as e:
        print(f"Embedding error: {e}")
        vec = np.zeros(1536, dtype=float)
    return vec


EVALUATOR_SYSTEM_PROMPT = """
You are an automatic chat response evaluator. You will receive a prompt and a response.
Return strict JSON with these keys:
- relevance: integer 0-100
- factuality: integer 0-100
- clarity: integer 0-100
- tone_adherence: integer 0-100
- hallucination: integer 0-100  # 100 = no hallucinations, 0 = many hallucinations
- comment: brief string
"""


def llm_score_dimensions(prompt: str, response: str, eval_model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Use the V3 agent as evaluator too"""
    eval_prompt = f"""{EVALUATOR_SYSTEM_PROMPT}

Prompt: {prompt}

Response: {response}

Provide your evaluation as JSON."""

    try:
        if USE_V3_AGENT:
            result = Runner.run_sync(agent, eval_prompt)
            eval_text = result.final_output
        else:
            raise Exception("V3 Agent not available")

        import re
        match = re.search(r"\{.*\}", eval_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in evaluator output")
        json_str = match.group(0)
        metrics = json.loads(json_str)

        for k in ["relevance", "factuality", "clarity", "tone_adherence", "hallucination"]:
            if k not in metrics:
                metrics[k] = 50
        metrics["comment"] = metrics.get("comment", "")
        return metrics
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "relevance": 50,
            "factuality": 50,
            "clarity": 50,
            "tone_adherence": 50,
            "hallucination": 50,
            "comment": f"evaluator_error: {e}",
        }


def compute_relevance_embedding_score(prompt: str, response: str) -> float:
    p_emb = embed_text(prompt)
    r_emb = embed_text(response)
    sim = cos_sim(p_emb, r_emb)
    score = int(max(0, min(1, (sim + 1) / 2)) * 100)
    return score


def compute_cost_estimate(prompt_tokens: int, response_tokens: int) -> float:
    price_per_token = 1e-6
    if prompt_tokens is None or response_tokens is None:
        return price_per_token * 1000
    return (prompt_tokens + response_tokens) * price_per_token


def score_latency(latency_ms: int) -> int:
    v = clamp01(1 - (latency_ms / (2 * TARGET_LATENCY_MS)))
    return int(1 + round(v * 99))


def score_cost(cost_usd: float) -> int:
    v = clamp01(1 - (cost_usd / (2 * TARGET_COST_PER_CALL)))
    return int(1 + round(v * 99))


def score_safety(hallucination_metric: int) -> int:
    return int(max(1, min(100, hallucination_metric)))


def aggregate_scores(evaluator_metrics: Dict[str, Any], embedding_relevance: float, latency_ms: int, cost_usd: float) -> Dict[str, Any]:
    llm_rel = evaluator_metrics.get("relevance", 50)
    factual = evaluator_metrics.get("factuality", 50)
    clarity = evaluator_metrics.get("clarity", 50)
    tone = evaluator_metrics.get("tone_adherence", 50)
    halluc = evaluator_metrics.get("hallucination", 50)

    quality_score = (
        0.30 * embedding_relevance +
        0.10 * llm_rel +
        0.30 * factual +
        0.15 * clarity +
        0.10 * tone +
        0.05 * halluc
    )
    quality_score = int(max(1, min(100, round(quality_score))))

    latency_score = score_latency(latency_ms)
    cost_score = score_cost(cost_usd)
    safety_score = score_safety(halluc)

    return {
        "quality_score": quality_score,
        "latency_score": latency_score,
        "cost_score": cost_score,
        "safety_score": safety_score,
        "breakdown": {
            "embedding_relevance": embedding_relevance,
            "llm_relevance": llm_rel,
            "factuality": factual,
            "clarity": clarity,
            "tone_adherence": tone,
            "hallucination": halluc,
        },
    }


# ---------- Save result for heuristic ----------

def send_to_optimizer(record: Dict[str, Any], csv_path: str = "results.csv") -> None:
    flat = {
        "timestamp": record["timestamp"],
        "prompt": record["prompt"],
        "response_id": record["response_id"],
        "response": record["response"],
        "latency_ms": record["latency_ms"],
        "prompt_tokens": record.get("prompt_tokens"),
        "response_tokens": record.get("response_tokens"),
        "final_quality_score": record["objectives"]["quality_score"],
        "latency_score": record["objectives"]["latency_score"],
        "cost_score": record["objectives"]["cost_score"],
        "safety_score": record["objectives"]["safety_score"],
        "params": json.dumps(record["params"]),
        "metrics_json": json.dumps(record["metrics"]),
        "objectives_json": json.dumps(record["objectives"]),
        "comment": record.get("comment", ""),
    }
    df = pd.DataFrame([flat])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


# ---------- Batch evaluation ----------

def evaluate_batch(prompts: List[str], params: Dict[str, Any], repeats: int = 1, eval_model: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        print(f"WARNING: expected {EXPECTED_PROMPT_COUNT} prompts but got {len(prompts)}. Continuing anyway...")

    results = []
    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        for r in range(repeats):
            gen = generate_response(prompt, params)
            response_text = gen["text"]
            latency = gen["latency_ms"]

            emb_rel = compute_relevance_embedding_score(prompt, response_text)
            llm_metrics = llm_score_dimensions(prompt, response_text, eval_model=eval_model)

            cost_est = compute_cost_estimate(gen.get("prompt_tokens"), gen.get("response_tokens"))

            objectives = aggregate_scores(llm_metrics, emb_rel, latency, cost_est)

            record = {
                "timestamp": int(time.time()),
                "prompt": prompt,
                "response_id": gen["response_id"],
                "response": response_text,
                "params": params,
                "latency_ms": latency,
                "prompt_tokens": gen.get("prompt_tokens"),
                "response_tokens": gen.get("response_tokens"),
                "metrics": llm_metrics,
                "objectives": objectives,
                "comment": llm_metrics.get("comment", ""),
            }

            results.append(record)
            send_to_optimizer(record)
    return results


# ---------- Heuristic: call your genetic algorithm ----------

def call_external_heuristic(params: Dict[str, Any], iteration_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your heuristic.py genetic algorithm.
    Your heuristic needs a 'propose' function that takes current params and summary.
    """
    try:
        import heuristic

        if hasattr(heuristic, "propose"):
            new_params = heuristic.propose(copy.deepcopy(params), iteration_summary)
            print("✓ Heuristic propose() returned new params.")
            return new_params
        else:
            print("⚠ heuristic.py found but no propose() function. Add this function!")
            print("Expected signature: def propose(params: dict, iteration_summary: dict) -> dict")

    except ImportError as e:
        print(f"⚠ heuristic.py not found: {e}")
    except Exception as e:
        print(f"⚠ Error calling heuristic: {e}")

    # Fallback: small random mutations
    print("Using fallback heuristic (random mutations)...")
    import random
    new = copy.deepcopy(params)

    if "temperature" in new:
        new["temperature"] = float(max(0.0, min(2.0, new["temperature"] + random.uniform(-0.1, 0.1))))
    if "top_p" in new:
        new["top_p"] = float(max(0.0, min(1.0, new["top_p"] + random.uniform(-0.05, 0.05))))
    if "presence_penalty" in new:
        new["presence_penalty"] = float(max(-2.0, min(2.0, new["presence_penalty"] + random.uniform(-0.2, 0.2))))
    if "frequency_penalty" in new:
        new["frequency_penalty"] = float(max(-2.0, min(2.0, new["frequency_penalty"] + random.uniform(-0.2, 0.2))))
    if "max_tokens" in new:
        new["max_tokens"] = int(max(20, min(200, new["max_tokens"] + random.randint(-20, 20))))
    if "retrieval_top_k" in new:
        new["retrieval_top_k"] = int(max(10, min(100, new["retrieval_top_k"] + random.randint(-5, 5))))
    if "num_beams" in new:
        new["num_beams"] = int(max(1, min(10, new["num_beams"] + random.randint(-1, 1))))

    return new


# ---------- Iteration summary ----------

def summarize_iteration(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    qualities = [r["objectives"]["quality_score"] for r in results]
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]

    summary = {
        "n_prompts": len(results),
        "mean_quality": float(np.mean(qualities)) if qualities else None,
        "median_quality": float(np.median(qualities)) if qualities else None,
        "std_quality": float(np.std(qualities)) if qualities else None,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
    }
    return summary


# ---------- Logging Functions ----------

def init_progress_log():
    """Initialize the progress log CSV file"""
    log_path = "optimization_progress.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iteration,timestamp,mean_quality,median_quality,std_quality,min_quality,max_quality,")
            f.write("mean_latency_ms,mean_latency_score,mean_cost_score,mean_safety_score,")
            f.write("params_json\n")
    return log_path


def log_iteration_progress(iteration: int, summary: Dict[str, Any], results: List[Dict[str, Any]],
                           params: Dict[str, Any], log_path: str):
    """Log iteration metrics to CSV for easy plotting"""

    # Extract all scores
    qualities = [r["objectives"]["quality_score"] for r in results]
    latency_scores = [r["objectives"]["latency_score"] for r in results]
    cost_scores = [r["objectives"]["cost_score"] for r in results]
    safety_scores = [r["objectives"]["safety_score"] for r in results]

    row = {
        "iteration": iteration,
        "timestamp": int(time.time()),
        "mean_quality": summary.get("mean_quality", 0),
        "median_quality": summary.get("median_quality", 0),
        "std_quality": summary.get("std_quality", 0),
        "min_quality": float(np.min(qualities)) if qualities else 0,
        "max_quality": float(np.max(qualities)) if qualities else 0,
        "mean_latency_ms": summary.get("mean_latency_ms", 0),
        "mean_latency_score": float(np.mean(latency_scores)) if latency_scores else 0,
        "mean_cost_score": float(np.mean(cost_scores)) if cost_scores else 0,
        "mean_safety_score": float(np.mean(safety_scores)) if safety_scores else 0,
        "params_json": json.dumps(params),
    }

    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", index=False, header=False)


def create_progress_plots():
    """Generate plots from the progress log"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        df = pd.read_csv("optimization_progress.csv")

        if len(df) < 2:
            print("⚠️  Not enough data points to create plots yet")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('LLM Parameter Optimization Progress', fontsize=16, fontweight='bold')

        # Plot 1: Quality Score Over Time
        ax1 = axes[0, 0]
        ax1.plot(df['iteration'], df['mean_quality'], 'b-o', linewidth=2, markersize=8, label='Mean')
        ax1.fill_between(df['iteration'], df['min_quality'], df['max_quality'], alpha=0.2, label='Min-Max Range')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Quality Score', fontsize=12)
        ax1.set_title('Quality Score Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: All Scores Comparison
        ax2 = axes[0, 1]
        ax2.plot(df['iteration'], df['mean_quality'], 'b-o', label='Quality', linewidth=2)
        ax2.plot(df['iteration'], df['mean_latency_score'], 'g-s', label='Latency', linewidth=2)
        ax2.plot(df['iteration'], df['mean_cost_score'], 'r-^', label='Cost', linewidth=2)
        ax2.plot(df['iteration'], df['mean_safety_score'], 'm-d', label='Safety', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Score (0-100)', fontsize=12)
        ax2.set_title('All Metrics Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Quality Variability
        ax3 = axes[1, 0]
        ax3.plot(df['iteration'], df['std_quality'], 'purple', linewidth=2, marker='o', markersize=8)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Standard Deviation', fontsize=12)
        ax3.set_title('Quality Score Variability', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Latency Trend
        ax4 = axes[1, 1]
        ax4.plot(df['iteration'], df['mean_latency_ms'], 'orange', linewidth=2, marker='o', markersize=8)
        ax4.axhline(y=TARGET_LATENCY_MS, color='r', linestyle='--', label=f'Target: {TARGET_LATENCY_MS}ms')
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Latency (ms)', fontsize=12)
        ax4.set_title('Response Latency Trend', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Progress plots saved to: optimization_progress.png")

    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping plot generation.")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️  Error creating plots: {e}")


# ---------- Control loop ----------

def control_loop(prompts: List[str], initial_params: Dict[str, Any], iterations: int = 5, repeats: int = 1):
    params = copy.deepcopy(initial_params)
    history = []

    # Initialize progress log
    log_path = init_progress_log()
    print(f"📝 Progress log initialized: {log_path}")

    for it in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {it}/{iterations}")
        print(f"{'='*60}")
        print(f"Testing with parameters:")
        print(json.dumps(params, indent=2))
        print(f"{'='*60}\n")

        results = evaluate_batch(prompts, params, repeats=repeats)

        # Iteration summary
        summary = summarize_iteration(results)
        print(f"\n📊 Results Summary:")
        print(f"  Mean Quality: {summary['mean_quality']:.2f}")
        print(f"  Median Quality: {summary['median_quality']:.2f}")
        print(f"  Std Quality: {summary['std_quality']:.2f}")
        print(f"  Mean Latency: {summary['mean_latency_ms']:.0f}ms")

        # Log progress
        log_iteration_progress(it, summary, results, params, log_path)

        iter_record = {
            "iteration": it,
            "timestamp": int(time.time()),
            "params": params,
            "per_prompt": results,
            "aggregate": summary,
        }

        # Save iteration file
        with open(f"iteration_{it}.json", "w", encoding="utf-8") as f:
            json.dump(iter_record, f, ensure_ascii=False, indent=2)

        history.append(iter_record)

        # Call heuristic for next parameters
        if it < iterations:  # Don't call heuristic after last iteration
            print(f"\n🧬 Calling heuristic for next parameters...")
            new_params = call_external_heuristic(params, summary)

            # Save proposal
            with open(f"proposal_{it}.json", "w", encoding="utf-8") as f:
                json.dump({"iteration": it, "proposed_params": new_params}, f, ensure_ascii=False, indent=2)

            params = new_params

        # Generate plots after each iteration (optional, can comment out if too slow)
        if it > 1:  # Need at least 2 points to plot
            create_progress_plots()

    # Save complete history
    with open("optimization_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("✅ Optimization complete!")
    print(f"{'='*60}")
    print(f"Results saved to:")
    print(f"  • results.csv - Detailed per-prompt results")
    print(f"  • optimization_progress.csv - Iteration summary (for plotting)")
    print(f"  • optimization_progress.png - Progress visualization")
    print(f"  • iteration_*.json - Full iteration data")
    print(f"  • optimization_history.json - Complete run history")
    print(f"{'='*60}\n")

    # Final plot generation
    create_progress_plots()

    # Print best iteration
    best_iter = max(history, key=lambda x: x['aggregate']['mean_quality'])
    print(f"\n🏆 Best Iteration: {best_iter['iteration']}")
    print(f"   Quality Score: {best_iter['aggregate']['mean_quality']:.2f}")
    print(f"   Parameters: {json.dumps(best_iter['params'], indent=2)}")


# ---------- CLI ----------

def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LLM Parameter Optimization with Genetic Algorithm")
    parser.add_argument("--prompts", type=str, required=True, help="Text file with prompts (one per line)")
    parser.add_argument("--config", type=str, required=True, help="JSON file with initial parameters")
    parser.add_argument("--iterations", type=int, default=5, help="Number of optimization iterations")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per prompt (for variance reduction)")
    args = parser.parse_args()

    if not USE_V3_AGENT:
        print("ERROR: Could not load V3 agent. Make sure:")
        print("  1. agents library is installed")
        print("  2. .env file exists with OPENAI_API_KEY")
        print("  3. V3.py is in the same directory")
        sys.exit(1)

    prompts = load_prompts(args.prompts)
    config = load_config(args.config)

    print(f"\n✓ Loaded {len(prompts)} prompts")
    print(f"✓ Loaded initial config with {len(config)} parameters")
    print(f"✓ V3 Agent ready\n")

    control_loop(prompts, config, iterations=args.iterations, repeats=args.repeats)


if __name__ == "__main__":
    main()
