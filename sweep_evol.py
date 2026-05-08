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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_single_kwargs(kwargs):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    t0 = time.time()
    result = run_single(**kwargs)
    result["elapsed_seconds"] = time.time() - t0
    return result


def _csv_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _csv_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _csv_strs(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="sweep_results")
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--trains_per_unit", type=int, default=200)
    parser.add_argument("--epochs_per_train", type=int, default=5)
    parser.add_argument("--examples_per_train", type=int, default=1)
    # Axis lists. Comma-separated. Each combination becomes one EA run.
    parser.add_argument("--pop_sizes", type=_csv_ints, default=[100])
    parser.add_argument("--dataset_types", type=_csv_strs,
                        default=["identity", "function", "permutation", "random"])
    parser.add_argument("--mutation_probabilities", type=_csv_floats, default=[0.2, 0.5])
    parser.add_argument("--top_q_percents", type=_csv_floats, default=[0.3])
    parser.add_argument("--nums_data_obj", type=_csv_ints, default=[4, 6, 10])
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Process pool workers. Default: cpu_count - 1.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configs = list(itertools.product(
        args.dataset_types,
        args.mutation_probabilities,
        args.top_q_percents,
        args.nums_data_obj,
        args.pop_sizes,
    ))
    print(f"Running {len(configs)} configurations across {args.workers} workers...")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"sweep_{timestamp}.json")
    results: List[Dict[str, Any]] = []

    # Build the kwargs payload for each config so workers can call run_single directly.
    jobs = []
    for dt, mp, tq, nd, pop in configs:
        jobs.append({
            "pop_size": pop,
            "num_data_obj": nd,
            "trains_per_unit": args.trains_per_unit,
            "examples_per_train": min(args.examples_per_train, nd),
            "epochs_per_train": args.epochs_per_train,
            "mutation_probability": mp,
            "top_q_percent": tq,
            "dataset_type": dt,
            "num_generations": args.num_generations,
        })

    t_start = time.time()
    if args.workers <= 1:
        # Serial path — useful for debugging since workers swallow stdout.
        for i, kwargs in enumerate(jobs):
            r = _run_single_kwargs(kwargs)
            results.append(r)
            _print_progress(i + 1, len(jobs), r)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_single_kwargs, k): k for k in jobs}
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                done += 1
                _print_progress(done, len(jobs), r)
                # Incremental save (order is non-deterministic, but each entry has its config).
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. Results written to {out_path}")
    print_summary(results)


def _print_progress(done: int, total: int, r: Dict[str, Any]):
    c = r["config"]
    print(
        f"[{done}/{total}] "
        f"dt={c['dataset_type']:11s} mp={c['mutation_probability']:.2f} "
        f"tq={c['top_q_percent']:.2f} nd={c['num_data_obj']} "
        f"pop={c['pop_size']:>5} "
        f"-> final_val_max={r['final_validated_max']:.4f} "
        f"final_avg={r['final_avg']:.4f} "
        f"({r['elapsed_seconds']:.1f}s)",
        flush=True,
    )


def print_summary(results: List[Dict[str, Any]]):
    """Print top-5 configurations by final validated max performance."""
    ranked = sorted(results, key=lambda r: r["final_validated_max"], reverse=True)
    print("\nTop 5 configurations (by final validated max performance):")
    for r in ranked[:5]:
        c = r["config"]
        print(
            f"  val_max={r['final_validated_max']:.4f} avg={r['final_avg']:.4f}  "
            f"dt={c['dataset_type']} mp={c['mutation_probability']} "
            f"tq={c['top_q_percent']} nd={c['num_data_obj']} pop={c['pop_size']}"
        )


if __name__ == "__main__":
    main()
