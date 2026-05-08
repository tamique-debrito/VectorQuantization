"""
Benchmark seeding strategies for lifting an evolved nd_small population into
nd_big.

Three stages:

  Stage A — evolve a small (nd_small) population for `num_generations_small`
            generations and persist it to disk. Skipped if `--load_pool` is given.

  Stage B — for each lift variant (in parallel processes), build an nd_big
            seed population from the small pool, save the seed snapshot, then
            run `num_generations_big` generations of evolution at nd_big and
            save the resulting final population. Records per-generation history.

  Stage C — write a combined results JSON and print a per-variant summary.

Outputs (under bench_seeding_results/<timestamp>/):
  small_pop.json          — Stage A output
  seed_<variant>.json     — pre-evolution lifted population per variant
  final_<variant>.json    — post-evolution population per variant
  bench.json              — per-variant trajectories and final scores
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence

from Evolutionary.evolutionary_algorithms import SimpleEvolutionaryAlg, simple_eval
from Evolutionary.population_io import (
    load_population,
    save_population,
    serialize_population,
)
from Evolutionary.seeding import LIFT_VARIANTS, lift_population
from Evolutionary.train_unit import TrainUnit


def _csv_strs(s):
    return [x.strip() for x in s.split(",") if x.strip()]


# -------- Stage A: evolve a small population ----------------------------------------

def stage_a_evolve_small(args, out_dir: str) -> Dict[str, Any]:
    print(f"[Stage A] evolving nd_small={args.nd_small} for "
          f"{args.num_generations_small} generations, pop={args.pop_size}")
    alg = SimpleEvolutionaryAlg(
        pop_size=args.pop_size,
        num_data_obj=args.nd_small,
        num_op_obj=args.nd_small * args.nd_small,
        trains_per_unit=args.trains_per_unit,
        examples_per_train=min(args.examples_per_train, args.nd_small),
        epochs_per_train=args.epochs_per_train,
        mutation_probability=args.mutation_probability,
        top_q_percent=args.top_q_percent,
        dataset_type=args.dataset_type,
        workers=args.workers_stage_a,
    )
    history = []
    for g in range(args.num_generations_small):
        s = alg.run_evolutionary_step(verbose=False)
        s["generation"] = g
        history.append(s)
        if (g + 1) % 10 == 0 or g == 0:
            print(f"  small gen {g+1}: avg={s['avg_performance']:.3f} "
                  f"max={s['max_performance']:.3f} "
                  f"val_max={s['validated_max_performance']:.3f}")

    units = [u for u, _ in alg.population]
    # Tag with the eval performance so the persisted snapshot is informative.
    for unit, (_, perf) in zip(units, alg.population):
        setattr(unit, "performance", perf)

    payload = serialize_population(
        units,
        num_data_obj=args.nd_small,
        num_op_obj=args.nd_small * args.nd_small,
        num_update_obj=args.nd_small * args.nd_small,
        dataset_type=args.dataset_type,
        source="stage_a_random_init",
        extra={"history": history,
               "num_generations": args.num_generations_small},
    )
    path = os.path.join(out_dir, "small_pop.json")
    save_population(path, payload)
    print(f"[Stage A] saved {path}")
    return {"path": path, "config": payload["config"], "history": history}


# -------- Stage B: lift + evolve at nd_big -------------------------------------------

def _stage_b_one_variant(kwargs):
    """Worker entry point. Pickled and shipped to a subprocess.

    Loads the small population from disk inside the worker so the small units
    don't have to travel through the ProcessPoolExecutor pickle boundary.
    """
    small_pool_path = kwargs["small_pool_path"]
    variant = kwargs["variant"]
    nd_small = kwargs["nd_small"]
    nd_big = kwargs["nd_big"]
    pop_size = kwargs["pop_size"]
    out_dir = kwargs["out_dir"]
    args_dict = kwargs["args_dict"]

    t0 = time.time()
    pool_payload = load_population(small_pool_path)
    small_pool = pool_payload["units"]

    # Filter the small pool down to the top fraction by recorded performance,
    # so seeding draws structure from units that actually learned the task —
    # not from the whole population (which still contains low-fitness mutants).
    seed_top_q = args_dict.get("seed_top_q", 1.0)
    if seed_top_q < 1.0:
        scored = [(u, getattr(u, "performance", 0.0)) for u in small_pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = max(1, int(len(scored) * seed_top_q))
        small_pool = [u for u, _ in scored[:keep]]
        print(f"  [{variant}] filtered small pool to top {keep}/{len(scored)} "
              f"(min perf in kept: {scored[keep-1][1]:.3f})", flush=True)

    # Lift.
    lifted = lift_population(
        small_pool=small_pool,
        nd_small=nd_small,
        nd_big=nd_big,
        variant=variant,
        pop_size=pop_size,
    )
    # Persist the seed snapshot before evolution.
    seed_path = os.path.join(out_dir, f"seed_{variant}.json")
    save_population(seed_path, serialize_population(
        lifted,
        num_data_obj=nd_big,
        num_op_obj=nd_big * nd_big,
        num_update_obj=nd_big * nd_big,
        dataset_type=args_dict["dataset_type"],
        source=f"lifted_{variant}",
        extra={"nd_small": nd_small, "lift_variant": variant},
    ))

    # Evolve at nd_big from the lifted seed.
    alg = SimpleEvolutionaryAlg(
        pop_size=pop_size,
        num_data_obj=nd_big,
        num_op_obj=nd_big * nd_big,
        trains_per_unit=args_dict["trains_per_unit"],
        examples_per_train=min(args_dict["examples_per_train"], nd_big),
        epochs_per_train=args_dict["epochs_per_train"],
        mutation_probability=args_dict["mutation_probability"],
        top_q_percent=args_dict["top_q_percent"],
        dataset_type=args_dict["dataset_type"],
        initial_population=lifted,
    )
    history = []
    for g in range(args_dict["num_generations_big"]):
        s = alg.run_evolutionary_step(verbose=False)
        s["generation"] = g
        history.append(s)

    # Persist the final population.
    final_units = [u for u, _ in alg.population]
    for unit, (_, perf) in zip(final_units, alg.population):
        setattr(unit, "performance", perf)
    final_path = os.path.join(out_dir, f"final_{variant}.json")
    save_population(final_path, serialize_population(
        final_units,
        num_data_obj=nd_big,
        num_op_obj=nd_big * nd_big,
        num_update_obj=nd_big * nd_big,
        dataset_type=args_dict["dataset_type"],
        source=f"evolved_from_{variant}",
        extra={"nd_small": nd_small, "lift_variant": variant,
               "num_generations": args_dict["num_generations_big"]},
    ))

    return {
        "variant": variant,
        "history": history,
        "seed_path": seed_path,
        "final_path": final_path,
        "elapsed_seconds": time.time() - t0,
        "final_validated_max": history[-1]["validated_max_performance"],
        "final_avg": history[-1]["avg_performance"],
    }


# -------- driver ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="bench_seeding_results")
    p.add_argument("--pop_size", type=int, default=100,
                   help="Default pop size for both stages. Override per stage "
                        "with --pop_size_small / --pop_size_big.")
    p.add_argument("--pop_size_small", type=int, default=None)
    p.add_argument("--pop_size_big", type=int, default=None)
    p.add_argument("--nd_small", type=int, default=4)
    p.add_argument("--nd_big", type=int, default=8)
    p.add_argument("--num_generations_small", type=int, default=50)
    p.add_argument("--num_generations_big", type=int, default=100)
    p.add_argument("--trains_per_unit", type=int, default=80)
    p.add_argument("--epochs_per_train", type=int, default=5)
    p.add_argument("--examples_per_train", type=int, default=3)
    p.add_argument("--mutation_probability", type=float, default=0.2)
    p.add_argument("--top_q_percent", type=float, default=0.3)
    p.add_argument("--seed_top_q", type=float, default=0.3,
                   help="Fraction of the Stage A population (top by performance) "
                        "used as parents when lifting. 1.0 = use all units, "
                        "matching the pre-existing behaviour.")
    p.add_argument("--dataset_type", default="identity")
    p.add_argument("--variants", type=_csv_strs,
                   default=sorted(LIFT_VARIANTS.keys()))
    p.add_argument("--load_pool", default=None,
                   help="Path to a small_pop.json from a previous run. "
                        "If set, Stage A is skipped.")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Workers for Stage B (parallel across variants).",
    )
    p.add_argument(
        "--workers_stage_a",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Intra-run workers for Stage A's per-unit fitness evaluation. "
             "Worth raising for large pop_size_small.",
    )
    args = p.parse_args()
    if args.pop_size_small is None:
        args.pop_size_small = args.pop_size
    if args.pop_size_big is None:
        args.pop_size_big = args.pop_size
    # Stage A reads `args.pop_size` directly; route through `pop_size_small`.
    args.pop_size = args.pop_size_small

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Stage A.
    if args.load_pool:
        small_pool_path = args.load_pool
        print(f"[Stage A] skipped; loading small pool from {small_pool_path}")
        small_history = None
    else:
        a = stage_a_evolve_small(args, run_dir)
        small_pool_path = a["path"]
        small_history = a["history"]

    # Stage B (parallel over variants).
    print(f"[Stage B] lifting + evolving {len(args.variants)} variants "
          f"(workers={args.workers})")
    args_dict = {
        "trains_per_unit": args.trains_per_unit,
        "examples_per_train": args.examples_per_train,
        "epochs_per_train": args.epochs_per_train,
        "mutation_probability": args.mutation_probability,
        "top_q_percent": args.top_q_percent,
        "dataset_type": args.dataset_type,
        "num_generations_big": args.num_generations_big,
        "seed_top_q": args.seed_top_q,
    }
    jobs = [
        {
            "small_pool_path": small_pool_path,
            "variant": v,
            "nd_small": args.nd_small,
            "nd_big": args.nd_big,
            "pop_size": args.pop_size_big,
            "out_dir": run_dir,
            "args_dict": args_dict,
        }
        for v in args.variants
    ]

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    if args.workers <= 1:
        for j in jobs:
            r = _stage_b_one_variant(j)
            results.append(r)
            _print_variant_summary(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_stage_b_one_variant, j): j["variant"]
                       for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                _print_variant_summary(r)

    print(f"[Stage B] done in {time.time()-t0:.1f}s")

    # Stage C — combined output.
    bench_path = os.path.join(run_dir, "bench.json")
    with open(bench_path, "w") as f:
        json.dump({
            "config": vars(args),
            "small_history": small_history,
            "results": results,
        }, f, indent=2)
    print(f"[Stage C] wrote {bench_path}")

    # Ranked summary.
    ranked = sorted(results, key=lambda r: r["final_validated_max"], reverse=True)
    print("\nRanking (final validated max):")
    for r in ranked:
        print(f"  {r['variant']:22s}  val_max={r['final_validated_max']:.4f}  "
              f"avg={r['final_avg']:.4f}  ({r['elapsed_seconds']:.1f}s)")


def _print_variant_summary(r):
    print(f"  [{r['variant']:22s}] final_val_max={r['final_validated_max']:.4f}  "
          f"avg={r['final_avg']:.4f}  ({r['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
