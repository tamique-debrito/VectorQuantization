"""
Lift evolved nd_small populations into the nd_big space using one of several
seeding strategies. Each lifted big-unit carries a `lift_variant` tag (and the
indices of the small units it was sampled from) so downstream analysis can
trace performance back to the strategy that produced the unit.

See `Evolutionary/SEEDING_NOTES.md` for the design rationale.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from Evolutionary.abstract_unit import AbstractBaseUnit, Mat
from Evolutionary.base_unit import base_unit_factory


# Each table's three axes are one of these "kinds". The kind determines the
# axis size: `data` is `nd`, `op`/`update` are `nd**2` (in this codebase).
# Tuple is (axis_in1, axis_in2, axis_out).
TABLE_SPECS: Dict[str, Tuple[str, str, str]] = {
    "FORWARD_TABLE":        ("data",   "op",   "data"),
    "FIND_BACK_INFO_TABLE": ("data",   "data", "data"),
    "BACKWARD_TABLE":       ("data",   "op",   "data"),
    "FIND_UPDATE_TABLE":    ("data",   "data", "update"),
    "APPLY_UPDATE_TABLE":   ("update", "op",   "op"),
}


def _kind_size(kind: str, nd: int) -> int:
    return nd if kind == "data" else nd * nd


def _make_big_class(nd_big: int):
    return base_unit_factory(nd_big, num_mat=nd_big * nd_big, num_update_obj=nd_big * nd_big)


def _attach(unit: AbstractBaseUnit, tables: Dict[str, List[List[int]]],
            variant: str, parents: Sequence[int]) -> None:
    for name, t in tables.items():
        setattr(unit, name, t)
    setattr(unit, "lift_variant", variant)
    setattr(unit, "lift_parents", list(parents))


# -------- variant: baseline_random ----------------------------------------------------

def _lift_baseline_random(small_pool: Sequence[AbstractBaseUnit], nd_small: int,
                          nd_big: int, big_cls) -> AbstractBaseUnit:
    # Build a fully fresh random nd_big unit. Each call to base_unit_factory
    # randomizes class-level tables, so we copy them onto the instance to keep
    # this unit independent of others sampling the same factory cache.
    fresh_cls = _make_big_class(nd_big)
    unit = big_cls(Mat(random.randrange(nd_big * nd_big)))
    tables = {name: getattr(fresh_cls, name) for name in TABLE_SPECS}
    _attach(unit, tables, variant="baseline_random", parents=[])
    return unit


# -------- variant: random_extend ------------------------------------------------------

def _lift_random_extend(small_pool: Sequence[AbstractBaseUnit], nd_small: int,
                        nd_big: int, big_cls) -> AbstractBaseUnit:
    parent_idx = random.randrange(len(small_pool))
    parent = small_pool[parent_idx]
    tables: Dict[str, List[List[int]]] = {}
    for name, (k_in1, k_in2, k_out) in TABLE_SPECS.items():
        s1_b = _kind_size(k_in1, nd_big)
        s2_b = _kind_size(k_in2, nd_big)
        s_out_b = _kind_size(k_out, nd_big)
        s1_s = _kind_size(k_in1, nd_small)
        s2_s = _kind_size(k_in2, nd_small)
        small = getattr(parent, name)
        big = [[0] * s2_b for _ in range(s1_b)]
        for i in range(s1_b):
            for j in range(s2_b):
                if i < s1_s and j < s2_s:
                    big[i][j] = small[i][j]
                else:
                    big[i][j] = random.randrange(s_out_b)
        tables[name] = big
    unit = big_cls(Mat(int(parent.w.w_idx)))
    _attach(unit, tables, variant="random_extend", parents=[parent_idx])
    return unit


# -------- variant: modular_tile -------------------------------------------------------

def _lift_modular_tile(small_pool: Sequence[AbstractBaseUnit], nd_small: int,
                       nd_big: int, big_cls) -> AbstractBaseUnit:
    parent_idx = random.randrange(len(small_pool))
    parent = small_pool[parent_idx]
    tables: Dict[str, List[List[int]]] = {}
    for name, (k_in1, k_in2, _k_out) in TABLE_SPECS.items():
        s1_b = _kind_size(k_in1, nd_big)
        s2_b = _kind_size(k_in2, nd_big)
        s1_s = _kind_size(k_in1, nd_small)
        s2_s = _kind_size(k_in2, nd_small)
        small = getattr(parent, name)
        big = [
            [small[i % s1_s][j % s2_s] for j in range(s2_b)]
            for i in range(s1_b)
        ]
        tables[name] = big
    unit = big_cls(Mat(int(parent.w.w_idx)))
    _attach(unit, tables, variant="modular_tile", parents=[parent_idx])
    return unit


# -------- block-tile family -----------------------------------------------------------

def _block_tile(
    small_pool: Sequence[AbstractBaseUnit],
    parent_idxs: Sequence[int],
    nd_small: int,
    nd_big: int,
    out_shift_fn: Callable[[int, str, int, int], int],
) -> Dict[str, List[List[int]]]:
    """Build the 5 lifted tables for a block-tile-style variant.

    `out_shift_fn(b, out_kind, out_small_val, k)` returns the shifted big-space
    output value, where `b` is the block id selected from axis 0.
    """
    k = nd_big // nd_small
    parents = [small_pool[i] for i in parent_idxs]
    tables: Dict[str, List[List[int]]] = {}
    for name, (k_in1, k_in2, k_out) in TABLE_SPECS.items():
        s1_b = _kind_size(k_in1, nd_big)
        s2_b = _kind_size(k_in2, nd_big)
        s1_s = _kind_size(k_in1, nd_small)
        s2_s = _kind_size(k_in2, nd_small)
        big = [[0] * s2_b for _ in range(s1_b)]
        for i in range(s1_b):
            # Block id is taken from axis 0; mod k so update-kind axes (with k**2
            # blocks) fold down onto the k available small units.
            b = (i // s1_s) % k
            i_small = i % s1_s
            small_table = getattr(parents[b], name)
            for j in range(s2_b):
                j_small = j % s2_s
                small_val = small_table[i_small][j_small]
                big[i][j] = out_shift_fn(b, k_out, small_val, k)
        tables[name] = big
    return tables


def _shift_collapse(_b: int, _kind: str, val: int, _k: int) -> int:
    return val


def _make_iso_shift(nd_small: int) -> Callable[[int, str, int, int], int]:
    def shift(b: int, kind: str, val: int, _k: int) -> int:
        if kind == "data":
            return val + b * nd_small
        return val + b * (nd_small * nd_small)
    return shift


def _make_permute_shift(nd_small: int, perm: Sequence[int]) -> Callable[[int, str, int, int], int]:
    def shift(b: int, kind: str, val: int, _k: int) -> int:
        pb = perm[b]
        if kind == "data":
            return val + pb * nd_small
        return val + pb * (nd_small * nd_small)
    return shift


def _lift_block_tile_collapse(small_pool: Sequence[AbstractBaseUnit], nd_small: int,
                              nd_big: int, big_cls) -> AbstractBaseUnit:
    k = nd_big // nd_small
    if nd_big % nd_small != 0:
        raise ValueError("block_tile variants require nd_big % nd_small == 0")
    parent_idxs = [random.randrange(len(small_pool)) for _ in range(k)]
    tables = _block_tile(small_pool, parent_idxs, nd_small, nd_big, _shift_collapse)
    unit = big_cls(Mat(int(small_pool[parent_idxs[0]].w.w_idx)))
    _attach(unit, tables, variant="block_tile_collapse", parents=parent_idxs)
    return unit


def _lift_block_tile_shift_iso(small_pool: Sequence[AbstractBaseUnit], nd_small: int,
                               nd_big: int, big_cls) -> AbstractBaseUnit:
    k = nd_big // nd_small
    if nd_big % nd_small != 0:
        raise ValueError("block_tile variants require nd_big % nd_small == 0")
    parent_idxs = [random.randrange(len(small_pool)) for _ in range(k)]
    tables = _block_tile(small_pool, parent_idxs, nd_small, nd_big, _make_iso_shift(nd_small))
    # w_idx for the iso lift: place parent[0]'s w in block 0's op slice (already
    # at the start of the op space, so no shift needed).
    unit = big_cls(Mat(int(small_pool[parent_idxs[0]].w.w_idx)))
    _attach(unit, tables, variant="block_tile_shift_iso", parents=parent_idxs)
    return unit


def _lift_block_tile_permute(small_pool: Sequence[AbstractBaseUnit], nd_small: int,
                             nd_big: int, big_cls) -> AbstractBaseUnit:
    k = nd_big // nd_small
    if nd_big % nd_small != 0:
        raise ValueError("block_tile variants require nd_big % nd_small == 0")
    parent_idxs = [random.randrange(len(small_pool)) for _ in range(k)]
    # Pick a non-identity permutation if k > 1; for k == 1 fall back to identity.
    # Bug history: previous version reject-sampled with a 10-attempt cap. For
    # small k (e.g. k=2 has only 2 perms with one identity), a few percent of
    # units would silently fall through to the identity perm — degenerating
    # `block_tile_permute` into `block_tile_shift_iso`, which is a *correct*
    # identity lift and inflated the variant's score in Run 5. Now we sample
    # uniformly from non-identity permutations directly.
    perm = list(range(k))
    if k > 1:
        identity = list(range(k))
        # Sample until we get a non-identity perm. Probability of identity is
        # 1/k!, so this terminates almost always after one shuffle for k >= 3
        # and at most a handful of tries for k == 2.
        while True:
            random.shuffle(perm)
            if perm != identity:
                break
    tables = _block_tile(small_pool, parent_idxs, nd_small, nd_big,
                         _make_permute_shift(nd_small, perm))
    unit = big_cls(Mat(int(small_pool[parent_idxs[0]].w.w_idx)))
    _attach(unit, tables, variant="block_tile_permute", parents=parent_idxs)
    setattr(unit, "lift_permutation", perm)
    return unit


# -------- registry + entry point ------------------------------------------------------

LIFT_VARIANTS: Dict[str, Callable] = {
    "baseline_random":      _lift_baseline_random,
    "random_extend":        _lift_random_extend,
    "modular_tile":         _lift_modular_tile,
    "block_tile_collapse":  _lift_block_tile_collapse,
    "block_tile_shift_iso": _lift_block_tile_shift_iso,
    "block_tile_permute":   _lift_block_tile_permute,
}


def lift_population(
    small_pool: Sequence[AbstractBaseUnit],
    nd_small: int,
    nd_big: int,
    variant: str,
    pop_size: int,
) -> List[AbstractBaseUnit]:
    if variant not in LIFT_VARIANTS:
        raise KeyError(f"Unknown lift variant: {variant!r}. "
                       f"Available: {sorted(LIFT_VARIANTS)}")
    big_cls = _make_big_class(nd_big)
    fn = LIFT_VARIANTS[variant]
    return [fn(small_pool, nd_small, nd_big, big_cls) for _ in range(pop_size)]
