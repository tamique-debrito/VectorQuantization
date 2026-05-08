"""
Read a bench_seeding run and print per-variant trajectories + ranking.

Usage:
  python analyze_bench_seeding.py                                  # latest run
  python analyze_bench_seeding.py bench_seeding_results/<dir>      # specific
"""

import glob
import json
import os
import sys


SAMPLE_GENS = [0, 5, 10, 25, 50, 100, 200, 300, 500, 999]


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        runs = sorted(glob.glob("bench_seeding_results/*/"))
        if not runs:
            print("No bench_seeding_results found.")
            sys.exit(1)
        target = runs[-1]
    bench_path = target if target.endswith(".json") else os.path.join(target, "bench.json")
    print(f"Reading {bench_path}")
    data = json.load(open(bench_path))

    cfg = data["config"]
    print(f"Config: nd_small={cfg['nd_small']} -> nd_big={cfg['nd_big']} "
          f"dataset={cfg['dataset_type']} pop={cfg['pop_size']}  "
          f"gens(small/big)={cfg['num_generations_small']}/{cfg['num_generations_big']}\n")

    if data.get("small_history"):
        sh = data["small_history"]
        print("nd_small evolution trajectory (validated_max):")
        sample = [g for g in SAMPLE_GENS if g < len(sh)]
        for g in sample:
            print(f"  gen {g:>3}: {sh[g]['validated_max_performance']:.4f}")
        print(f"  final ({len(sh)-1}): {sh[-1]['validated_max_performance']:.4f}")
        print()

    results = data["results"]
    max_hist = max(len(r["history"]) for r in results)
    sample = [g for g in SAMPLE_GENS if g < max_hist]
    print("Per-variant trajectory at nd_big:")
    head = "  ".join(f"g{g:>3}" for g in sample)
    print(f"{'variant':<24} {head}    final")
    print("-" * (24 + len(head) + 12))
    for r in sorted(results, key=lambda r: r["final_validated_max"], reverse=True):
        cells = []
        for g in sample:
            if g < len(r["history"]):
                cells.append(f"{r['history'][g]['validated_max_performance']:.3f}")
            else:
                cells.append("  -- ")
        traj = "  ".join(cells)
        final = r["final_validated_max"]
        print(f"{r['variant']:<24} {traj}    {final:.4f}")

    # Compare to baseline_random.
    bench_baseline = next(
        (r for r in results if r["variant"] == "baseline_random"), None)
    if bench_baseline is not None:
        print(f"\nDelta vs baseline_random (final val_max):")
        for r in sorted(results, key=lambda r: r["final_validated_max"], reverse=True):
            if r["variant"] == "baseline_random":
                continue
            delta = r["final_validated_max"] - bench_baseline["final_validated_max"]
            print(f"  {r['variant']:<24} {delta:+.4f}")


if __name__ == "__main__":
    main()
