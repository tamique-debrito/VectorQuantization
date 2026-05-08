# Minimum parameters for small-dimension `BaseUnit` evolution

After fixing the bug where `forward` / `find_back_info` / `backward` / `update`
were reading tables off the closure-captured class instead of `self` (so
`simple_mutate` / `splice_units` / instance overrides were silently ignored),
the EA actually evolves. This note records how small the parameters can go
before convergence on the easy `identity` nd=4 task breaks down.

All numbers are single-seed and noisy; treat the boundaries as ±1 row of
uncertainty. Fixed: `mutation_probability=0.05, top_q_percent=0.3,
dataset_type="identity", num_data_obj=4, num_op_obj=16`.

## Population × generations

| `pop_size` | `num_generations` | final `val_max` | first gen ≥ 0.99 |
|-----------:|------------------:|----------------:|-----------------:|
|         10 |               30  | 0.74            | (never)          |
|         15 |              100  | 0.76            | (never)          |
|         20 |               30  | 0.92            | (boundary)       |
|         25 |              100  | 1.00            | 25               |
|         30 |              100  | 1.00            | 31               |
|         40 |              100  | 1.00            | 26               |
|     **50** |               30  | **1.00**        | **6**            |
|        100 |               30  | 1.00            | 2                |
|        200 |               30  | 1.00            | 15               |

There is a clear cliff around `pop_size ≈ 15-20`. Below the cliff the EA
saturates at a local-optimum plateau (~0.75) and does not escape even with
100 generations. Above the cliff convergence to 1.0 is reliable, and the
generations needed drops fast as the population grows.

## Training-budget axes (at pop_size=50)

Sweep over `trains_per_unit × epochs_per_train × examples_per_train`,
`num_generations=40`. Reports first generation reaching `val_max ≥ 0.99`
(or `none`) and total wall time.

| trains | epochs | examples | final | first_g≥0.99 | time  |
|-------:|-------:|---------:|------:|-------------:|------:|
|     20 |      2 |        1 | 1.000 |          12  | 0.3s  |
|     20 |      2 |        2 | 1.000 |          15  | 0.3s  |
|     20 |      2 |        3 | 1.000 |          14  | 0.4s  |
|     20 |      3 |        1 | 1.000 |          23  | 0.3s  |
|     20 |      3 |        2 | 1.000 |          25  | 0.4s  |
|     20 |      3 |        3 | 0.737 |        none  | 0.4s  |
|     20 |      5 |        1 | 1.000 |          23  | 0.4s  |
|     20 |      5 |        2 | 0.931 |        none  | 0.5s  |
|     20 |      5 |        3 | 0.799 |        none  | 0.6s  |
|     40 |      3 |        1 | 1.000 |          12  | 0.4s  |
|     40 |      3 |        2 | 1.000 |          16  | 0.6s  |
|     40 |      5 |        3 | 1.000 |          21  | 0.9s  |
|     80 |      3 |        3 | 1.000 |          10  | 1.2s  |
|     80 |      5 |        3 | 1.000 |          27  | 1.6s  |

Several runs at the boundary fail while structurally-similar neighbors
succeed — single-seed variance is substantial. This implies the *minimum*
column would shift around if averaged over many seeds; the rows above are
illustrative.

Past the practical floor, more training budget per unit mostly adds wall
time without improving convergence — once the fitness signal is sharp enough
to rank units, extra training inside one evaluation is wasted.

## Recommended minimal recipe (identity nd=4)

```
pop_size              = 50
num_generations       = 15
trains_per_unit       = 20
epochs_per_train      = 2
examples_per_train    = 1
mutation_probability  = 0.05
top_q_percent         = 0.3
dataset_type          = "identity"
```

Wall time ≈ 0.2s per run on one worker, reliably reaches `val_max = 1.0`
(modulo seed noise).

## Caveats

- **Noisy.** Re-run any boundary config 5-10 times before trusting the
  number. Especially the cliff between pop=15 and pop=25.
- **Identity is a small search space.** With nd=4 there are only 16 columns
  in `FORWARD_TABLE`, and the probability that one of them is exactly the
  identity column is ~6% per random unit. So pop=50 essentially guarantees
  a good initial seed; the EA's job is mostly to surface and propagate it.
- **Other dataset types do not respond to these params.** `function`,
  `permutation`, and `random` cap at ≈ 0.30-0.42 regardless of pop, because
  `_static_generate_toy_dataset` resamples a fresh permutation/function each
  call. That's a task-definition issue, not an EA-capacity issue. See
  `SWEEP_FINDINGS.md` for details.
- **`construct_for_function` builds a stronger solution than the EA finds.**
  It locks every column of `FORWARD_TABLE` to `f`, so update can do anything.
  EA-found solutions tend to be one good column plus a near-no-op update —
  smaller fingerprint, more brittle to mutation.
