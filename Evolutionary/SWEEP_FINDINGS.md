# Sweep findings

Notes accumulated while running `sweep_evol.py` on the scalar `BaseUnit` learner.
Replace / append as more runs happen.

## Run 1 — initial 48-config grid

Command:
```
python sweep_evol.py --pop_size 100 --num_generations 8 \
  --trains_per_unit 100 --epochs_per_train 5
```

Grid: `dataset_type ∈ {random, function, identity, permutation}`,
`mutation_probability ∈ {0.05, 0.2, 0.5}`, `top_q_percent ∈ {0.1, 0.3}`,
`num_data_obj ∈ {4, 6}`. `examples_per_train = 1`.

### Headline numbers (final validated max performance)

| dataset      | nd=4 best | nd=6 best | chance (1/nd) |
|--------------|-----------|-----------|---------------|
| identity     | **0.668** | 0.367     | 0.250 / 0.167 |
| random       | 0.384     | 0.229     | 0.250 / 0.167 |
| function     | 0.392     | 0.224     | 0.250 / 0.167 |
| permutation  | 0.374     | 0.224     | 0.250 / 0.167 |

### What this tells us

1. **Only `identity` is meaningfully learned.** It's not just above chance — it's
   ~2.7× chance for nd=4. The other three task types barely beat chance, suggesting
   the population isn't actually finding tables that *learn* a non-trivial mapping
   under this compute budget.

2. **`examples_per_train=1` is too small a signal.** With a single (input, target)
   pair per training run, "function", "random", and "permutation" all degenerate
   into the same task: memorize one pair. So the dataset_type variation isn't
   testing what we hoped at this setting. Worth re-running with
   `examples_per_train ≥ 2` for nd ≥ 4.

3. **`nd=6` is uniformly harder than `nd=4`** — expected, since the search space
   for the 5 lookup tables grows like `nd^k`. With only 8 generations × 100 pop
   we may simply be undersampling.

4. **Best identity config**: `mp=0.2, tq=0.3, nd=4` → 0.668. A bias toward higher
   `top_q_percent` (keep more of the population) shows up in the top 5 — but the
   signal is noisy at this budget.

### Next things to try
- Re-run with `examples_per_train=2` or `3` so non-identity tasks have real
  structure to learn within a single training session.
- Push `num_generations` higher (20-30) on a narrower grid to see whether
  improvement is still happening at gen 8.
- Add a held-out generalization check (train on K examples, test on different K)
  rather than scoring purely on training accuracy. Currently a unit can win
  by overfitting one example, which inflates `function` / `random` scores.

## Run 2 — 500 generations, pop=100, nd ∈ {4, 6, 10}

Command:
```
python sweep_evol.py --pop_size 100 --num_generations 500 \
  --trains_per_unit 80 --epochs_per_train 5 --examples_per_train 3
```

24 configs, completed in ~5 min on 13 workers.

### What this tells us

- **Trajectories flatten within ~5 generations.** Across all 24 configs the
  validated-max performance moves by < 0.05 between gen 5 and gen 499. **More
  generations alone do not help** — the EA converges, gets stuck, and
  continuing past ~25 generations buys essentially nothing.
- Best identity nd=4 final: 0.614 (vs 0.753 from a 30-gen run with pop=150 in
  an earlier ad-hoc test). So spending more compute on **wider populations**
  beat spending it on more generations.
- nd=6 and nd=10 essentially flat at chance for all dataset types.

## Run 3 — pop ∈ {2000, 5000}, 20 generations

Command:
```
python sweep_evol.py --pop_sizes 2000,5000 \
  --nums_data_obj 4,6,10 --dataset_types identity,function,permutation \
  --mutation_probabilities 0.2 --top_q_percents 0.3 \
  --num_generations 20 --trains_per_unit 60 --epochs_per_train 5 \
  --examples_per_train 3
```

18 configs, completed in 400s on 13 workers.

### Headline comparison across all sweeps (best validated_max)

| dataset×nd       | pop=100, 500g | pop=2000, 20g | pop=5000, 20g | chance |
|------------------|---------------|---------------|---------------|--------|
| identity nd=4    | 0.614         | 0.517         | **0.724**     | 0.250  |
| identity nd=6    | 0.265         | **0.317**     | 0.260         | 0.167  |
| identity nd=10   | 0.129         | **0.146**     | 0.145         | 0.100  |
| function nd=4    | 0.302         | 0.308         | **0.326**     | 0.250  |
| function nd=6    | 0.175         | 0.172         | **0.181**     | 0.167  |
| function nd=10   | 0.105         | 0.100         | 0.100         | 0.100  |
| permutation nd=4 | 0.303         | 0.302         | **0.333**     | 0.250  |
| permutation nd=6 | 0.179         | 0.181         | **0.195**     | 0.167  |
| permutation nd=10| 0.101         | 0.103         | 0.101         | 0.100  |

### What this tells us

1. **Pop scaling helps `identity` a lot.** identity nd=4 jumped from 0.614 (pop=100,
   500 gens) to **0.724** (pop=5000, 20 gens). Confirms the hypothesis that
   diversity > iteration depth.

2. **Pop scaling barely helps `function` / `permutation`.** Even at pop=5000:
   - function nd=4: 0.326 vs 0.302 at pop=100 → ~+0.02
   - permutation nd=4: 0.333 vs 0.303 at pop=100 → ~+0.03

   At nd ≥ 6, no measurable improvement at all. So **the wall on harder tasks
   isn't just about exploration budget** — even sampling 5000 random tables
   doesn't surface a meaningfully better solution for function or permutation.

3. **At pop=5000, identity nd=4 is essentially solved at gen 0** (val_max = 0.747
   with no evolution yet — i.e. the random initialization sampled good units).
   The EA basically just maintains that level. The identity task is more of a
   *random-search problem* than an *evolutionary* one at this scale.

4. **Trajectories are flat in this regime too.** Even with pop=5000, gen 0 ≈ gen 19
   for almost every config. The EA isn't doing useful work past the initial
   sampling — selection + recombination quality is the binding constraint,
   confirming the diagnosis from Run 2.

### Conclusion across runs

- **Identity** is solvable, but it's basically a random-search win. Nothing
  about the table-evolution setup is making it learn — sampling enough random
  tables eventually finds one with a near-identity `FORWARD_TABLE`.
- **Function / permutation** look unreachable with the current EA mechanism
  no matter how much compute we throw at it. The 5-table representation
  doesn't seem to support smooth fitness gradients on these tasks: random
  sampling doesn't help, mutation doesn't move things, splice doesn't compose
  partial solutions.
- The next move is **mechanism, not budget**: tournament selection, block /
  per-table crossover, or seeding-based scaling (`Evolutionary/SEEDING_NOTES.md`).

## Run 4 — POST-FIX recalibration

Commit `e264e3d` ("fix bug with parameter association") changed
`Evolutionary/base_unit.py` so `forward` / `find_back_info` / `backward` /
`update` read tables via `self.X` instead of `BaseUnit.X`. Pre-fix, every
per-instance table override produced by `simple_mutate`, `splice_units`, and
`Evolutionary/seeding.py` was silently ignored. The EA was effectively only
selecting on `w_idx` luck — table mutations had no effect on `forward`
behavior.

`verify_construction.py` confirms the fix: a deterministically constructed
unit (every column of FORWARD_TABLE = f) reaches **1.0** on its target
permutation, and mutation degrades smoothly (1.0 → 0.95 at p=0.05, 0.54 at
p=0.5). So there is now a real fitness gradient for the EA to climb.

### Recalibration sweep — pop=80, 30 generations, 16 configs

Command:
```
python sweep_evol.py --pop_sizes 80 --nums_data_obj 4,6 \
  --dataset_types identity,function,permutation,random \
  --mutation_probabilities 0.05,0.2 --top_q_percents 0.3 \
  --num_generations 30 --trains_per_unit 60 --epochs_per_train 5 \
  --examples_per_train 3
```

| dataset×nd       | pre-fix best (any pop/gen) | **post-fix (pop=80, 30g)** | chance |
|------------------|----------------------------|----------------------------|--------|
| identity nd=4    | 0.724 (pop=5000)           | **0.747**                  | 0.250  |
| identity nd=6    | 0.317 (pop=2000)           | **0.549**                  | 0.167  |
| function nd=4    | 0.326 (pop=5000)           | **0.355**                  | 0.250  |
| function nd=6    | 0.181 (pop=5000)           | 0.187                      | 0.167  |
| permutation nd=4 | 0.333 (pop=5000)           | **0.359**                  | 0.250  |
| permutation nd=6 | 0.195 (pop=5000)           | 0.170                      | 0.167  |

Trajectories no longer flat: identity nd=4 climbs `0.41 → 0.58 → 0.77 → 0.76`
across gens 0/5/10/25 (vs essentially flat after gen 5 pre-fix). identity
nd=6 also climbs steadily over all 30 generations — still improving when the
run ended.

### Recalibration of bench_seeding — identity, nd_small=4 → nd_big=8

Command (per seed):
```
python bench_seeding.py --pop_size 60 --num_generations_small 30 \
  --num_generations_big 40 --trains_per_unit 60 --epochs_per_train 5 \
  --examples_per_train 3 --dataset_type identity
```

Two seeds, identical ranking:

| variant              | seed 1 val_max | seed 2 val_max | delta vs baseline |
|----------------------|----------------|----------------|-------------------|
| **block_tile_shift_iso** | **0.822**     | **0.743**      | **+0.50**         |
| block_tile_collapse  | 0.736          | 0.636          | +0.42             |
| modular_tile         | 0.522          | 0.599          | +0.27             |
| random_extend        | 0.479          | 0.494          | +0.20             |
| baseline_random      | 0.285          | 0.290          | (baseline)        |
| block_tile_permute   | 0.275          | 0.251          | -0.02             |

`block_tile_shift_iso` (input-in-block-b → output-in-block-b — the
isomorphism case) **wins by ~+0.5 over baseline**, robustly. `block_tile_permute`
(deliberately routes block 0 inputs to block 1's output range) is the *worst*
of all variants, including baseline. Structural preservation across the lift
matters a lot.

For function and permutation tasks with the same params: every variant
hovers in [0.11, 0.14] (chance = 0.125). Stage A only achieves ~0.35 on
those tasks at nd_small=4, so the lifted "structure" is essentially noise
and seeding has nothing useful to transfer. Need a stronger Stage A (longer
runs, bigger pop, or different fitness signal) before seeding can help on
the harder tasks.

### Updated conclusions

- **The "more compute doesn't help" diagnosis from Runs 1-3 was an artifact
  of the bug.** With table mutations actually taking effect, the EA *does*
  improve over generations and benefits from larger populations.
- **Identity is now genuinely learnable** by the EA (not just by random
  initialization). nd=6 went from "stuck at 0.32" to "climbing past 0.55 in
  30 gens with only pop=80." Worth re-running the original sweeps to map
  the new improvement curves.
- **Seeding works for identity.** The shift-iso variant transfers small-unit
  structure cleanly; the permute variant actively destroys it. Confirms the
  user's intuition that block-local closure is the structurally meaningful
  property.
- **Seeding still doesn't help function / permutation** — but only because
  Stage A doesn't produce useful structure for those tasks. Likely fixable
  by giving Stage A more compute (or by using a different `dataset_type`
  signal, or by curriculum-style scheduling).
- Next: rerun the sweep on `function` / `permutation` with much bigger Stage
  A budgets to see if the new (working) EA can crack them given enough room.
