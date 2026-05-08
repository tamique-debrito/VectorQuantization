"""
Seed an EA population with mutated copies of a known-correct unit and check
whether the EA converges back to the correct mapping.

Uses `dataset_type="identity"` so the target is a fixed permutation (identity)
across all training datasets; this is the cleanest test case. Once
`fixed_permutation` dataset support exists, we can repeat for arbitrary perms.

Compares three population seeding strategies:
  - control: fresh random initial population (matches default EA behavior)
  - perturbed: every unit starts as identity-correct then mutated with prob p
  - mixed: a few correct-perturbed units seeded among an otherwise-random pop
"""

import argparse
import random

from Evolutionary.abstract_unit import Mat
from Evolutionary.base_unit import base_unit_factory
from Evolutionary.construct_unit import construct_for_permutation
from Evolutionary.evolutionary_algorithms import SimpleEvolutionaryAlg
from Evolutionary.mutations import simple_mutate


def build_random_unit(nd: int):
    BU = base_unit_factory(nd, num_mat=nd * nd)
    return BU(Mat(random.randrange(BU.NUM_OPERATOR_OBJ)))


def build_perturbed_unit(nd: int, mutation_p: float):
    correct = construct_for_permutation(list(range(nd)))  # identity
    return simple_mutate(correct, mutation_p)


def run(name: str, alg: SimpleEvolutionaryAlg, num_generations: int):
    print(f"\n=== {name} ===")
    for g in range(num_generations):
        s = alg.run_evolutionary_step(verbose=False)
        if g in {0, 1, 2, 3, 5, 10, 15, 20, 29} or g == num_generations - 1:
            print(f"  gen {g:>2}: avg={s['avg_performance']:.3f}  "
                  f"max={s['max_performance']:.3f}  val={s['validated_max_performance']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nd", type=int, default=4)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=30)
    parser.add_argument("--mutation_p_perturbation", type=float, default=0.1,
                        help="Mutation prob applied to the correct unit when seeding.")
    parser.add_argument("--seed_count", type=int, default=10,
                        help="In the 'mixed' run, how many seeded units to inject.")
    parser.add_argument("--ea_mutation_p", type=float, default=0.05)
    parser.add_argument("--top_q", type=float, default=0.3)
    parser.add_argument("--trains_per_unit", type=int, default=100)
    parser.add_argument("--epochs_per_train", type=int, default=5)
    parser.add_argument("--examples_per_train", type=int, default=3)
    args = parser.parse_args()

    common_kwargs = dict(
        pop_size=args.pop_size,
        num_data_obj=args.nd,
        num_op_obj=args.nd * args.nd,
        trains_per_unit=args.trains_per_unit,
        examples_per_train=args.examples_per_train,
        epochs_per_train=args.epochs_per_train,
        mutation_probability=args.ea_mutation_p,
        top_q_percent=args.top_q,
        dataset_type="identity",
    )

    # Control: standard random init.
    print("Building control (random init) ...")
    control_alg = SimpleEvolutionaryAlg(**common_kwargs)
    run("control: random init", control_alg, args.num_generations)

    # All-perturbed: every unit is a perturbed correct.
    print("\nBuilding all-perturbed init ...")
    perturbed_pop = [build_perturbed_unit(args.nd, args.mutation_p_perturbation)
                     for _ in range(args.pop_size)]
    perturbed_alg = SimpleEvolutionaryAlg(**common_kwargs, initial_population=perturbed_pop)
    run(f"all-perturbed (p={args.mutation_p_perturbation}) seed", perturbed_alg, args.num_generations)

    # Mixed: a few perturbed correct, rest random.
    print("\nBuilding mixed init ...")
    mixed_pop = ([build_perturbed_unit(args.nd, args.mutation_p_perturbation)
                  for _ in range(args.seed_count)]
                 + [build_random_unit(args.nd) for _ in range(args.pop_size - args.seed_count)])
    mixed_alg = SimpleEvolutionaryAlg(**common_kwargs, initial_population=mixed_pop)
    run(f"mixed: {args.seed_count} perturbed + {args.pop_size - args.seed_count} random",
        mixed_alg, args.num_generations)


if __name__ == "__main__":
    main()
