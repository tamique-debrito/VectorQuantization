"""
Parameter sweep harness for the Evolutionary `BaseUnit` learner.

Runs `SimpleEvolutionaryAlg` across combinations of:
  - dataset_type: which kind of (input -> target) structure each toy dataset has
  - mutation_probability
  - top_q_percent
  - num_data_obj

For each combination it records per-generation (avg, max, validated max) performance
and writes the full history to a JSON file in `sweep_results/`.

This is intentionally cheap-by-default so you can iterate quickly. Bump the loop sizes
in `main()` once you've confirmed which axes matter.
"""

import argparse
import itertools
import json
import os
import time
from typing import Any, Dict, List

from Evolutionary.evolutionary_algorithms import SimpleEvolutionaryAlg


def run_single(
    *,
    pop_size: int,
    num_data_obj: int,
    trains_per_unit: int,
    examples_per_train: int,
    epochs_per_train: int,
    mutation_probability: float,
    top_q_percent: float,
    dataset_type: str,
    num_generations: int,
) -> Dict[str, Any]:
    """Run one EA configuration and return its full per-generation history."""
    alg = SimpleEvolutionaryAlg(
        pop_size=pop_size,
        num_data_obj=num_data_obj,
        num_op_obj=num_data_obj * num_data_obj,
        trains_per_unit=trains_per_unit,
        examples_per_train=examples_per_train,
        epochs_per_train=epochs_per_train,
        mutation_probability=mutation_probability,
        top_q_percent=top_q_percent,
        dataset_type=dataset_type,
    )
    history: List[Dict[str, float]] = []
    for gen in range(num_generations):
        stats = alg.run_evolutionary_step(verbose=False)
        stats["generation"] = gen
        history.append(stats)
    return {
        "config": {
            "pop_size": pop_size,
            "num_data_obj": num_data_obj,
            "trains_per_unit": trains_per_unit,
            "examples_per_train": examples_per_train,
            "epochs_per_train": epochs_per_train,
            "mutation_probability": mutation_probability,
            "top_q_percent": top_q_percent,
            "dataset_type": dataset_type,
            "num_generations": num_generations,
        },
        "history": history,
        "final_validated_max": history[-1]["validated_max_performance"],
        "final_avg": history[-1]["avg_performance"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="sweep_results")
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--trains_per_unit", type=int, default=200)
    parser.add_argument("--epochs_per_train", type=int, default=5)
    parser.add_argument("--examples_per_train", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Axes to sweep. Edit these to focus the search.
    dataset_types = ["random", "function", "identity", "permutation"]
    mutation_probabilities = [0.05, 0.2, 0.5]
    top_q_percents = [0.1, 0.3]
    nums_data_obj = [4, 6]

    configs = list(itertools.product(
        dataset_types, mutation_probabilities, top_q_percents, nums_data_obj
    ))
    print(f"Running {len(configs)} configurations...")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"sweep_{timestamp}.json")
    results = []

    for i, (dt, mp, tq, nd) in enumerate(configs):
        t0 = time.time()
        # examples_per_train can't exceed num_data_obj (assertion in TrainUnit)
        examples = min(args.examples_per_train, nd)
        result = run_single(
            pop_size=args.pop_size,
            num_data_obj=nd,
            trains_per_unit=args.trains_per_unit,
            examples_per_train=examples,
            epochs_per_train=args.epochs_per_train,
            mutation_probability=mp,
            top_q_percent=tq,
            dataset_type=dt,
            num_generations=args.num_generations,
        )
        elapsed = time.time() - t0
        result["elapsed_seconds"] = elapsed
        results.append(result)
        print(
            f"[{i+1}/{len(configs)}] "
            f"dt={dt:11s} mp={mp:.2f} tq={tq:.2f} nd={nd} "
            f"-> final_val_max={result['final_validated_max']:.4f} "
            f"final_avg={result['final_avg']:.4f} "
            f"({elapsed:.1f}s)"
        )
        # Write incrementally so partial results survive interruption.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results written to {out_path}")
    print_summary(results)


def print_summary(results: List[Dict[str, Any]]):
    """Print top-5 configurations by final validated max performance."""
    ranked = sorted(results, key=lambda r: r["final_validated_max"], reverse=True)
    print("\nTop 5 configurations (by final validated max performance):")
    for r in ranked[:5]:
        c = r["config"]
        print(
            f"  val_max={r['final_validated_max']:.4f} avg={r['final_avg']:.4f}  "
            f"dt={c['dataset_type']} mp={c['mutation_probability']} "
            f"tq={c['top_q_percent']} nd={c['num_data_obj']}"
        )


if __name__ == "__main__":
    main()
