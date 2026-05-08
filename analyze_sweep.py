"""
Read a sweep result JSON and print per-config performance over time.

Usage:
  python analyze_sweep.py                          # latest sweep
  python analyze_sweep.py sweep_results/foo.json   # specific file

Reports each config's (avg, max, validated_max) at sampled generations,
plus best-final summary.
"""

import glob
import json
import sys
from collections import defaultdict


SAMPLE_GENS = [0, 5, 10, 25, 50, 100, 200, 300, 400, 499]


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob("sweep_results/sweep_*.json"))
        if not files:
            print("No sweep results found in sweep_results/")
            sys.exit(1)
        path = files[-1]

    print(f"Reading {path}")
    data = json.load(open(path))
    print(f"{len(data)} configs\n")

    # Sort configs by final validated max for stable reading.
    data_sorted = sorted(data, key=lambda r: r["final_validated_max"], reverse=True)

    print("Per-config trajectory (validated_max at sampled generations):")
    header_gens = [g for g in SAMPLE_GENS if g < max(len(r["history"]) for r in data)]
    header = "  ".join(f"g{g:>3}" for g in header_gens)
    print(f"{'config':<70} {header}    final")
    print("-" * (70 + len(header) + 12))

    for r in data_sorted:
        c = r["config"]
        pop = c.get("pop_size", "?")
        label = (
            f"dt={c['dataset_type']:<11} nd={c['num_data_obj']:>2} "
            f"mp={c['mutation_probability']:.2f} tq={c['top_q_percent']:.2f} "
            f"pop={pop}"
        )
        history = r["history"]
        cells = []
        for g in header_gens:
            if g < len(history):
                cells.append(f"{history[g]['validated_max_performance']:.3f}")
            else:
                cells.append("  -- ")
        traj = "  ".join(cells)
        final = r["final_validated_max"]
        print(f"{label:<70} {traj}    {final:.4f}")

    print("\nBest by (dataset_type, num_data_obj):")
    best = defaultdict(lambda: (-1.0, None))
    for r in data:
        c = r["config"]
        key = (c["dataset_type"], c["num_data_obj"])
        if r["final_validated_max"] > best[key][0]:
            best[key] = (r["final_validated_max"], c)
    for (dt, nd), (v, c) in sorted(best.items()):
        pop = c.get("pop_size", "?")
        print(
            f"  {dt:<12} nd={nd:>2}  -> {v:.4f}  "
            f"(chance={1/nd:.3f})  "
            f"[mp={c['mutation_probability']} tq={c['top_q_percent']} pop={pop}]"
        )


if __name__ == "__main__":
    main()
