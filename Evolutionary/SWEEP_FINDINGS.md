# Sweep findings

Notes accumulated while running `sweep_evol.py` on the scalar `BaseUnit` learner.
Replace / append as more runs happen.

## Theoretical context: ceilings and chance levels per dataset_type

A `BaseUnit` of dimension `nd = N` has `num_op = N²` columns in its
`FORWARD_TABLE` — that's the most "modes of operation" any single unit can
hold. The dataset_types differ in how big the space of possible target
mappings is, which sets the ceiling on what one unit can achieve when each
training session sees a *fresh* draw from that space.

| dataset_type    | what's drawn per training session    | task domain size      | columns / domain (coverage) |
|-----------------|--------------------------------------|-----------------------|-----------------------------|
| `identity`      | nothing — target is always `input`   | 1                     | 1                           |
| `permutation`   | a fresh random permutation           | `N!`                  | `N² / N!`                   |
| `function`      | a fresh random function              | `N^N`                 | `N² / N^N`                  |
| `random`        | each (input, target) independent     | unbounded / inconsistent | 0 (no learnable structure) |

**Chance level** (random output matching target) is `1/N` for every task.

**Coverage** is the crude upper bound: even if every column of `FORWARD_TABLE`
encoded a different element of the task domain, only `min(1, N²/|domain|)` of
the draws can be "perfectly" handled by some column. Non-covered draws are
bounded above by the best-partial-match a column can offer (which is
substantially better than chance for small `N`, but approaches chance as `N`
grows). A reasonable rough ceiling is

```
ceiling(N) ≈ min(1, N²/|domain|) + (1 − min(1, N²/|domain|)) × best_partial_match(N)
best_partial_match(N) ≈ 1/N + small_bonus_from_choosing_best_of_N²_columns
```

Concrete numbers (chance + coverage, the two solid bounds):

| nd | chance (1/N) | perm coverage (N²/N!) | func coverage (N²/N^N) | identity ceiling |
|----|-------------:|----------------------:|-----------------------:|-----------------:|
| 4  | 0.250        | 16/24 ≈ **0.667**     | 16/256 ≈ **0.063**     | 1.000            |
| 6  | 0.167        | 36/720 ≈ **0.050**    | 36/46656 ≈ **8e-4**    | 1.000            |
| 8  | 0.125        | 64/40320 ≈ **0.0016** | 64/16M ≈ **4e-6**      | 1.000            |
| 10 | 0.100        | 100/3.6M ≈ **3e-5**   | 100/10¹⁰ = 1e-8        | 1.000            |

**Implications:**
- `identity` and any *fixed* target (single permutation, single function) are
  always reachable in principle: ceiling = 1.0 for any `N`, and
  `construct_for_function` proves this constructively.
- `permutation` (drawn fresh per call) has a *hard* coverage drop with `N`:
  for `N=4` the unit can in principle "tile" two-thirds of all permutations
  one per column; by `N=8` it can only tile ~0.16% of them. Above ~`N=5`,
  even with infinite EA budget, val_max is bounded well below 1.0.
- `function` collapses even faster — by `N=6` the coverage is already < 0.1%.
- `random` has no learnable structure; ceiling = chance.

The empirical numbers below should be read against these ceilings, not against
1.0. A `permutation nd=4` final of ~0.49 with pop=5000 is at ~70% of its
coverage-bound ceiling of 0.667; a `permutation nd=6` final of ~0.18 is
already *above* the coverage-bound for perfect handling (because of partial
matches in non-covered columns) — there is essentially no headroom there
without enlarging the unit.

### Important caveats on what these scores actually measure

Two subtleties about the `permutation` / `function` task definitions and the
fitness number that show up across all runs below. Read every reported score
through these:

**1. `permutation` does NOT test that a unit has learned the full
permutation.** Each training session samples `examples_per_train` inputs (e.g.
3) *with replacement* from `[0, N)` and scores the unit only on those sampled
positions of the permutation. The other `N − k` positions are never seen and
never tested. So a high `permutation` score really means "the unit can fit a
3-point slice of a random permutation," not "the unit has internalized the
permutation as a whole." Many more FORWARD columns satisfy the partial
constraint than satisfy the full perm — the practical scoring ceiling under
this task definition is substantially higher than the coverage-bound numbers
above, but doing well on it does *not* imply generalization to unseen
positions of the same permutation. Same caveat for `function`. To test
genuine permutation-learning we'd need either `examples_per_train = N` (drawn
without replacement, so every position is touched once) or an explicit
held-out test split.

**2. The reported score is a weighted average over training epochs, not
the unit's final accuracy.** Per-session score is computed by `simple_eval` in
`Evolutionary/train_unit.py`:

```python
weights = [1, 2, ..., num_epochs - 1]
score = sum(acc[i] * weights[i-1] for i in 1..num_epochs-1) / sum(weights)
```

— it drops epoch 0 entirely (giving the unit time to adapt) then averages
epochs 1..K-1 with linearly increasing weights. The final per-unit fitness is
the mean of these weighted averages over `trains_per_unit` sessions. So a
unit that reaches accuracy 1.0 by the last epoch of every session can still
report fitness well below 1.0, because earlier epochs in each session
(typically 0.3-0.7 while w_idx is being routed to the right column) drag
the average down. **A reported score of 0.7 is consistent with the unit
hitting 1.0 by epoch 4-5 most of the time but not converging instantly.**
For a "what does the unit converge to" view of the same data, look at
last-epoch accuracy directly rather than at the fitness number.

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

| dataset      | nd=4 best | nd=6 best | chance (1/nd) | ceiling nd=4 / nd=6 |
|--------------|-----------|-----------|---------------|---------------------|
| identity     | **0.668** | 0.367     | 0.250 / 0.167 | 1.000 / 1.000       |
| random       | 0.384     | 0.229     | 0.250 / 0.167 | chance              |
| function     | 0.392     | 0.224     | 0.250 / 0.167 | ≈0.30 / ≈chance     |
| permutation  | 0.374     | 0.224     | 0.250 / 0.167 | ≈0.75 / ≈chance     |

(NOTE: this whole run was pre-fix, so the EA was effectively random search — see
"Run 4" below.)

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

| dataset×nd       | pop=100, 500g | pop=2000, 20g | pop=5000, 20g | chance | ceiling (cov. + partial) |
|------------------|---------------|---------------|---------------|--------|--------------------------|
| identity nd=4    | 0.614         | 0.517         | **0.724**     | 0.250  | 1.000                    |
| identity nd=6    | 0.265         | **0.317**     | 0.260         | 0.167  | 1.000                    |
| identity nd=10   | 0.129         | **0.146**     | 0.145         | 0.100  | 1.000                    |
| function nd=4    | 0.302         | 0.308         | **0.326**     | 0.250  | ≈ 0.30 (cov 0.063)       |
| function nd=6    | 0.175         | 0.172         | **0.181**     | 0.167  | ≈ chance (cov 8e-4)      |
| function nd=10   | 0.105         | 0.100         | 0.100         | 0.100  | ≈ chance (cov ~1e-8)     |
| permutation nd=4 | 0.303         | 0.302         | **0.333**     | 0.250  | ≈ 0.75 (cov 0.667)       |
| permutation nd=6 | 0.179         | 0.181         | **0.195**     | 0.167  | ≈ chance (cov 0.05)      |
| permutation nd=10| 0.101         | 0.103         | 0.101         | 0.100  | ≈ chance (cov 3e-5)      |

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

| dataset×nd       | pre-fix best (any pop/gen) | **post-fix (pop=80, 30g)** | chance | ceiling          |
|------------------|----------------------------|----------------------------|--------|------------------|
| identity nd=4    | 0.724 (pop=5000)           | **0.747**                  | 0.250  | 1.000            |
| identity nd=6    | 0.317 (pop=2000)           | **0.549**                  | 0.167  | 1.000            |
| function nd=4    | 0.326 (pop=5000)           | **0.355**                  | 0.250  | ≈ 0.30           |
| function nd=6    | 0.181 (pop=5000)           | 0.187                      | 0.167  | ≈ chance         |
| permutation nd=4 | 0.333 (pop=5000)           | **0.359**                  | 0.250  | ≈ 0.75           |
| permutation nd=6 | 0.195 (pop=5000)           | 0.170                      | 0.167  | ≈ chance         |

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

## Run 5 — big Stage A + top-q seed filter, nd=6 → 12

Two changes to the bench harness for this run:
1. `--pop_size_small` / `--pop_size_big` separated so Stage A can be much
   larger without inflating Stage B.
2. `--seed_top_q` (default 0.3) filters the Stage A population to its top
   fraction by recorded performance before lifting. Previously
   `lift_population` sampled uniformly from the whole pool, including
   low-fitness mutants; now it samples only from units that actually learned.

Command:
```
python bench_seeding.py --pop_size_small 2000 --pop_size_big 80 \
  --nd_small 6 --nd_big 12 --num_generations_small 100 \
  --num_generations_big 50 --trains_per_unit 60 --epochs_per_train 5 \
  --examples_per_train 3 --seed_top_q 0.3 --dataset_type identity
```

### Stage A trajectory (nd=6, identity, pop=2000)

| gen   | avg   | max   | val_max |
|-------|-------|-------|---------|
|  1    | 0.166 | 0.314 | 0.278   |
| 10    | 0.286 | 0.433 | 0.379   |
| 30    | 0.504 | 0.669 | 0.630   |
| 50    | 0.716 | 0.861 | 0.812   |
| 70    | 0.885 | 0.993 | 0.979   |
| 80    | 0.928 | 1.000 | **1.000** |
| 100   | 0.942 | 1.000 | 1.000   |

So the EA *does* drive identity nd=6 to ceiling given enough budget. The
"plateau at 0.74" seen at smaller scale was a budget issue, not a
mechanism issue. Top-30% filter kept 600/2000 units; all kept units had
performance = 1.000.

### Stage B (nd_big=12, pop=80, 50 gens, identity)

| variant              | val_max | avg   | structural explanation |
|----------------------|---------|-------|------------------------|
| **block_tile_shift_iso** | **1.000** | 0.944 | block-local closure preserved → big unit *is* identity |
| **block_tile_permute**   | **1.000** | 0.936 | output block routing distorted, but the small unit's update path adapts `w_idx` to compensate within the first epoch |
| block_tile_collapse  | 0.502   | 0.470 | outputs always collapse to block 0 → block-1 inputs can never produce block-1 outputs (theoretical ceiling 0.5) |
| modular_tile         | 0.500   | 0.468 | outputs mod nd_small → identical 0.5 ceiling |
| random_extend        | 0.497   | 0.423 | block-1 inputs hit randomly-filled extended cells → ~0.5 ceiling |
| baseline_random      | 0.148   | 0.142 | random init at nd=12; 50 gens × pop=80 only buys ~+0.06 over chance (chance=0.083 at nd=12) |

The 0.5 ceiling for collapse / modular / random_extend is theoretically
predicted: each of those variants either drops the output shift or
randomizes half the input → output mapping, so block-1 inputs (half the
input space) cannot match block-1 targets.

### What this run says

- **The EA can reach ceiling at nd=6 with adequate budget.** Pop scaling
  *does* help once the bug is fixed; the previous "wider pop barely beats
  smaller pop" finding was a measurement artifact.
- **`block_tile_shift_iso` lifts a perfect small unit into a perfect big
  unit "for free."** No nd=12 evolution needed to reach 1.0 — the 50 gens
  of Stage B just maintain it. Compute saved: Stage B baseline_random reached
  only 0.15 with the same budget. To match shift_iso's 1.0 from scratch
  would presumably require ~Stage-A-scale budget at nd=12, which is much
  more expensive than at nd=6.
- **`block_tile_permute` also reaches 1.0**, even though its forward routing
  is structurally "wrong" for identity. Mechanism: the small unit's update
  loop is preserved within each block, so within an epoch the `w_idx`
  reroutes to something that produces correct outputs. This is a stronger
  result than expected — the lift's value isn't only in its forward
  structure, but in preserving the small unit's *learning procedure*.
- **The top-q seed filter probably matters but wasn't isolated in this run.**
  All kept units had perf=1.0 here because Stage A converged. At smaller
  Stage A scales the filter would matter more; worth a separate ablation.
- **Open**: replicate on `function` / `permutation` once Stage A reaches
  high val_max for those (likely needs even bigger budget given coverage
  ceilings — see "Theoretical context" at top of file).
