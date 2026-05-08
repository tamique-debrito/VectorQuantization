# Seeding nd_big from nd_small populations — design notes

Companion to `COMPOSITION_NOTES.md`. Documents the lifting strategies in
`Evolutionary/seeding.py` and the benchmark in `bench_seeding.py`.

## Why

Sweeps on the scalar `BaseUnit` learner showed `identity` learns well at
small `num_data_obj` (nd) but `function` / `permutation` plateau near chance
even with many generations at nd ∈ {4, 6}. Lifting to a bigger nd via fresh
random init throws away whatever the small population learned.

If a non-trivial lift strategy beats `baseline_random` we have a tool for
scaling nd without paying full search cost. If none of them help, we've
learned that the 5-table representation isn't compositional in any obvious
way — useful information for the composition work in `COMPOSITION_NOTES.md`.

## Table axis kinds

Each of the 5 lookup tables in `BaseUnit` has three axes (in1, in2, out).
Each axis is one of three "kinds":

| Kind     | Axis size at nd |
|----------|-----------------|
| `data`   | nd              |
| `op`     | nd²             |
| `update` | nd²             |

The kinds for each table (declared in `seeding.py` as `TABLE_SPECS`):

| Table                  | in1     | in2  | out    |
|------------------------|---------|------|--------|
| FORWARD_TABLE          | data    | op   | data   |
| FIND_BACK_INFO_TABLE   | data    | data | data   |
| BACKWARD_TABLE         | data    | op   | data   |
| FIND_UPDATE_TABLE      | data    | data | update |
| APPLY_UPDATE_TABLE     | update  | op   | op     |

When lifting nd_small → nd_big, each axis of each table grows from
`size(kind, nd_small)` to `size(kind, nd_big)`.

## Lift variants

All variants take a small population `small_pool` and produce a big-population
of `pop_size` units. Each lifted unit is tagged with `lift_variant`,
`lift_parents` (the sampled small-pool indices), and — for `block_tile_permute`
— `lift_permutation`.

1. **`baseline_random`** — fresh random nd_big unit. Control.
2. **`random_extend`** — preserve small cells at their original `(i, j)`
   coordinates; new rows / cols filled with random valid values.
3. **`modular_tile`** — `T_big[i][j] = T_small[i mod s_in1_small][j mod s_in2_small]`.
   Outputs stay in the small-output-range.
4. **`block_tile_collapse`** — requires `nd_big = k · nd_small`. Sample k small
   units U_0..U_{k-1}. Block id `b` is taken from axis 0; the small unit
   consulted is `U_b mod k`. Outputs collapse to the small range (no shift).
5. **`block_tile_shift_iso`** — same block partition; outputs shifted by `b`
   so block-b's output lands in block-b's slice of the big output range. This
   realises k truly parallel small units. The "isomorphism" case: input in
   block b → output in block b.
6. **`block_tile_permute`** — like `shift_iso` but composes a non-identity
   permutation π over block indices on the output: block b's output lands in
   block π(b)'s slice.

For block-tile variants:
- The unit selector `b` is computed from axis 0 only. Tables whose axis 0 is
  data have k blocks; those whose axis 0 is op/update (only APPLY_UPDATE) have
  k² blocks, folded onto k via `b mod k`.
- `j` (axis 1) is reduced by `j mod s_in2_small`, giving a deterministic
  cross-block default rather than a separate fallback table.
- `w_idx` for the lifted unit defaults to U_0's `w_idx`.

## Persistence format

`Evolutionary/population_io.py` provides `save_population` / `load_population`.
Each saved file has:

```json
{
  "config": {"num_data_obj": 4, "num_op_obj": 16, "num_update_obj": 16,
             "dataset_type": "identity", "source": "stage_a_random_init", ...},
  "units": [
    {"w_idx": 7,
     "tables": {"FORWARD_TABLE": [[...]], ...},
     "lift_variant": "modular_tile",   // optional
     "lift_parents": [3],               // optional
     "performance": 0.42}              // optional
  ]
}
```

Load reconstructs a single fresh `BaseUnit` class via `base_unit_factory(nd, ...)`
and attaches each unit's tables as instance attributes. Per-unit metadata
(`lift_variant`, `lift_parents`, `performance`, `lift_permutation`) round-trips.

## Benchmark protocol (`bench_seeding.py`)

Per run:
- Stage A: evolve nd_small for `--num_generations_small` from random init.
  Persist the final population as `small_pop.json` along with the per-generation
  history.
- Stage B (parallel over variants): each worker loads the small pool, lifts it
  via one variant, persists `seed_<variant>.json`, runs `--num_generations_big`
  generations starting from the lifted seed (via `initial_population=` on
  `SimpleEvolutionaryAlg`), persists `final_<variant>.json`.
- Stage C: `bench.json` summarises all variant trajectories and the run config.

Analyse with `analyze_bench_seeding.py`: prints per-variant trajectories at
sampled generations and a delta-vs-baseline table.

## Verification protocol

- Unit tests: `python -m unittest Evolutionary.test_seeding -v`. Covers shape
  and range for all variants, plus the structural invariants for
  `random_extend` (small cells preserved), `modular_tile` (modular equality),
  `block_tile_shift_iso` (block-closure on axis 0), `block_tile_permute`
  (output lands in permuted block).
- Smoke run: `python bench_seeding.py --pop_size 20 --num_generations_small 3 --num_generations_big 3 --trains_per_unit 10 --epochs_per_train 3` finishes in seconds and writes the full directory of artifacts.
- Real comparison: bigger pop / longer runs across multiple `--dataset_type`
  values; rerun several times for noise. A variant "wins" only if its mean
  final val_max reliably beats `baseline_random` across seeds.

## Open future directions

- Per-unit lift_variant could be allowed to be heterogeneous within a single
  big population (e.g. half iso, half collapse). Currently each pop is
  homogeneous to keep the comparison clean.
- For `block_tile_permute`, current code chooses one random non-identity
  permutation per unit. Could be extended to enumerate all permutations and
  evaluate each.
- Cross-block fallback in block variants currently uses `j mod s_in2_small`.
  Other policies (route to block 0, route to a "default" small unit) could be
  added.
- Block-tile variants currently require `nd_big = k · nd_small`. A relaxed
  version with `nd_big > k · nd_small` would need to choose how to fill the
  remainder rows/cols (e.g. random, or fold onto block 0).
