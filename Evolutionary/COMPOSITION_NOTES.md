# Composing Units into Vector / Matrix Modules — Design Notes

Future work, not yet implemented. Captures the design direction so we don't lose it
while iterating on the scalar `BaseUnit` first.

## Motivation

Current `BaseUnit` is a single quantized scalar "neuron": one input index, one weight
index, one output index, lookup tables for forward / find-back-info / backward /
find-update / apply-update. To make this useful beyond toy single-element learning,
units need to compose into vector- and matrix-shaped modules — the equivalent of going
from a scalar multiply to a real linear layer.

## Required new operation types

A vector module of width `d` is conceptually `d` scalar units in parallel, plus *inter-position
operations* that mix information across positions. At minimum:

1. **Pointwise op (binary, same position)** — analog of elementwise add / hadamard.
   Two operand vector units of width `d` produce a width-`d` output; position `i` of the
   output depends only on position `i` of each operand.

2. **Cross op (binary, all-to-all)** — analog of dot product / matmul column.
   Width-`d` × width-`d` → scalar (or width-`d` if we want a full matmul-style reducer).
   Position `i` of the output depends on all positions of both operands.

Each new op type needs its own set of lookup tables. The five-table scheme used by `BaseUnit`
(forward, find-back-info, backward, find-update, apply-update) does not transfer directly,
because backward must now route gradient-analog information through *two* operands rather
than one. Sketch:

| Op       | Forward table shape                       | Backward routing                                      |
|----------|-------------------------------------------|-------------------------------------------------------|
| Pointwise| `NUM_DATA × NUM_DATA → NUM_DATA` (per pos)| `find_back_info(target, out) → (back_a, back_b)` per pos |
| Cross    | `NUM_DATA^d × NUM_DATA^d → NUM_DATA`      | scalar back-info splits into two width-`d` back-info  |

The cross op's forward table is exponential in `d`, so it cannot be a literal lookup table.
Two plausible escapes: (a) factor it as a sum/reduction of `d` pointwise pair lookups
(matches the structure of a real dot product), or (b) replace the table with a small
learned/evolved program. Option (a) is the natural starting point.

## Backpropagation redesign

For a binary op the backward pass needs:
- A way to compute, given the forward output and a target, what each operand "should have been."
- A way to map that operand-target back to a parameter update for the operand unit (which
  may itself be a composite).

This is essentially the chain rule reimplemented in lookup-table land. The `find_back_info`
table currently encodes "(target_idx, output_idx) → back_info_idx". Composite ops need
"(target_idx, output_idx) → (back_info_a_idx, back_info_b_idx)" — so a pair of tables, or
one table whose output ranges over `NUM_DATA × NUM_DATA`.

For the cross op factored as a reduction over pointwise pair ops, backward becomes a
reduction-tree backward: split the scalar back-info into per-position back-info, then
reuse the pointwise-op backward.

## Open questions to settle before implementing

- Does each op type evolve its tables independently, or are tables shared across op types?
- Are operands always leaf scalar units, or can a vector op take another vector op as input
  (true compositionality)? Probably yes, but the eval harness gets more complicated.
- What's the fitness signal? Probably "this composite module can learn a synthetic
  vector-input → vector-output dataset by running its own forward/backward/update loop."
- Do we evolve the composition graph too, or fix it (e.g. always one cross op feeding one
  pointwise op) and only evolve the tables inside?

## Recommended order of attack

1. Get scalar `BaseUnit` solidly learning across a variety of toy datasets and parameter
   regimes (current step — see iteration plan).
2. Implement `PointwiseUnit` (width-`d` parallel scalar units, no cross-position mixing).
   Confirm it can learn an elementwise function.
3. Implement `CrossUnit` as a reduction of pointwise pair lookups. Confirm it can learn
   a dot-product-like task.
4. Compose: `Cross → Pointwise → loss`. Evolve all tables jointly. This is the first
   "matrix multiply-like" learner.
