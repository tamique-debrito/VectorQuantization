"""
Verify that a deterministically-constructed unit achieves 1.0 on its target
function for any permutation, and observe how mutation degrades performance.

This is the empirical companion to the claim that identity and any other
permutation should be mathematically equivalent: both are realizable by the
same construction strategy (lock FORWARD to f). If `permutation` performs worse
than `identity` in the EA sweep, the difference is in selection / dataset
sampling, not in expressive capacity of the table mechanic.
"""

import argparse
import random
from typing import Callable, List

from Evolutionary.abstract_unit import Vec
from Evolutionary.construct_unit import construct_for_permutation
from Evolutionary.mutations import simple_mutate


def evaluate_on_function(
    unit,
    f: Callable[[int], int],
    nd: int,
    num_trials: int = 100,
    num_examples: int = 3,
    num_epochs: int = 5,
    seed: int = 0,
) -> float:
    """Score `unit` on a target function f, mirroring the static_train loop.

    Each trial: sample `num_examples` random inputs, compute targets via f,
    then for `num_epochs` walks through the dataset run forward → check →
    find_back_info → update. Aggregate accuracy across all examples in all
    epochs of all trials.
    """
    rng = random.Random(seed)
    correct = 0
    total = 0
    for _ in range(num_trials):
        dataset = [(Vec(rng.randrange(nd)), None) for _ in range(num_examples)]
        dataset = [(input_vec, Vec(f(input_vec.v_idx))) for input_vec, _ in dataset]
        for _epoch in range(num_epochs):
            for input_vec, target_vec in dataset:
                out = unit.forward(input_vec)
                if out.v_idx == target_vec.v_idx:
                    correct += 1
                total += 1
                back_info = unit.find_back_info(target_vec)
                unit.update(back_info)
    return correct / total


def random_permutation(nd: int, rng: random.Random) -> List[int]:
    perm = list(range(nd))
    rng.shuffle(perm)
    return perm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nd", type=int, default=4)
    parser.add_argument("--num_trials", type=int, default=200)
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Pick a few permutations to test, including identity and a random one.
    nd = args.nd
    perms = [
        list(range(nd)),                           # identity
        list(reversed(range(nd))),                 # full reverse
        [(i + 1) % nd for i in range(nd)],         # cyclic shift
        random_permutation(nd, rng),
        random_permutation(nd, rng),
    ]

    print(f"== Construction correctness (nd={nd}) ==")
    print(f"{'permutation':<30} {'score':>8}")
    for perm in perms:
        unit = construct_for_permutation(perm)
        score = evaluate_on_function(
            unit, lambda v, p=perm: p[v], nd,
            num_trials=args.num_trials,
            num_examples=args.num_examples,
            num_epochs=args.num_epochs,
            seed=args.seed,
        )
        print(f"{str(perm):<30} {score:>8.4f}")

    # Pick one "interesting" non-identity permutation for the perturbation study.
    target_perm = perms[3]
    target_fn = lambda v, p=target_perm: p[v]

    print(f"\n== Perturbation study (target perm = {target_perm}) ==")
    print(f"{'mutation_p':>10} {'mean':>8} {'min':>8} {'max':>8}  (3 trials per p)")
    base_unit = construct_for_permutation(target_perm)
    base_score = evaluate_on_function(
        base_unit, target_fn, nd,
        num_trials=args.num_trials,
        num_examples=args.num_examples,
        num_epochs=args.num_epochs,
        seed=args.seed,
    )
    print(f"{'(none)':>10} {base_score:>8.4f}")
    for p in [0.01, 0.05, 0.1, 0.2, 0.5]:
        scores = []
        for trial_seed in range(3):
            rng = random.Random(trial_seed * 1000 + 1)
            random.seed(trial_seed * 1000 + 1)  # simple_mutate uses module-level random
            perturbed = simple_mutate(base_unit, p)
            scores.append(evaluate_on_function(
                perturbed, target_fn, nd,
                num_trials=args.num_trials,
                num_examples=args.num_examples,
                num_epochs=args.num_epochs,
                seed=args.seed,
            ))
        print(f"{p:>10.2f} {sum(scores)/len(scores):>8.4f} {min(scores):>8.4f} {max(scores):>8.4f}")


if __name__ == "__main__":
    main()
