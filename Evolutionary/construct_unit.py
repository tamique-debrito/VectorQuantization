"""
Deterministic construction of `BaseUnit`s for arbitrary scalar functions.

A unit's forward output for input v is `FORWARD_TABLE[v][w_idx]`. If we make
every column of `FORWARD_TABLE` equal to f, then `forward(v) == f(v)` for any
`w_idx`. The other tables are then irrelevant to forward correctness — `update`
can change `w` to anything and the output still tracks f. This is the simplest
proof that the 5-table mechanic is *expressive enough* to encode any function
on the index set; it does not by itself prove the EA can *find* such tables.
"""

from typing import Callable, Optional

from Evolutionary.abstract_unit import Mat, Vec
from Evolutionary.base_unit import base_unit_factory


def construct_for_function(
    f: Callable[[int], int],
    nd: int,
    num_op: Optional[int] = None,
):
    """Return a BaseUnit instance whose forward(v) is guaranteed to equal f(v).

    Strategy: set every column of FORWARD_TABLE to f. The unit's `w_idx` becomes
    irrelevant for forward correctness, so update() can do whatever it wants.

    Returns: an instance whose `forward(Vec(v)).v_idx == f(v)` for all v.
    """
    if num_op is None:
        num_op = nd * nd
    BaseUnit = base_unit_factory(nd, num_mat=num_op)
    unit = BaseUnit(Mat(0))
    # Instance attribute shadows the class-level table, leaves other units
    # produced by other factory calls untouched.
    unit.FORWARD_TABLE = [[f(v) for _ in range(num_op)] for v in range(nd)]
    return unit


def construct_for_permutation(perm: list[int], num_op: Optional[int] = None):
    """Convenience wrapper. `perm[v]` is f(v). Validates perm is a permutation."""
    nd = len(perm)
    assert sorted(perm) == list(range(nd)), f"Not a permutation of [0,{nd}): {perm}"
    return construct_for_function(lambda v, p=perm: p[v], nd, num_op)
