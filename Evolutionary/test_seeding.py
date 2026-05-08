"""Unit tests for Evolutionary/seeding.py — checks shape, range, and the
structural invariants each lift variant is supposed to provide."""

import copy
import random
import unittest

from Evolutionary.abstract_unit import Mat
from Evolutionary.base_unit import base_unit_factory
from Evolutionary.seeding import (
    LIFT_VARIANTS,
    TABLE_SPECS,
    _kind_size,
    lift_population,
)


def _build_small_pool(nd_small: int, n_units: int):
    """Build a small pool with per-instance copies of the tables (matching
    the post-mutation shape the lift code consumes)."""
    cls = base_unit_factory(nd_small, num_mat=nd_small * nd_small,
                            num_update_obj=nd_small * nd_small)
    pool = []
    for _ in range(n_units):
        # Re-randomize tables each time so units in the pool differ.
        cls = base_unit_factory(nd_small, num_mat=nd_small * nd_small,
                                num_update_obj=nd_small * nd_small)
        u = cls(Mat(random.randrange(nd_small * nd_small)))
        for name in TABLE_SPECS:
            setattr(u, name, copy.deepcopy(getattr(u, name)))
        pool.append(u)
    return pool


class TestLiftShapeAndRange(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        self.nd_small, self.nd_big = 4, 8
        self.pool = _build_small_pool(self.nd_small, n_units=5)

    def test_all_variants_produce_well_shaped_tables(self):
        for variant in LIFT_VARIANTS:
            pop = lift_population(self.pool, self.nd_small, self.nd_big,
                                  variant, pop_size=3)
            self.assertEqual(len(pop), 3, msg=f"{variant} pop_size mismatch")
            for u in pop:
                for name, (k1, k2, ko) in TABLE_SPECS.items():
                    t = getattr(u, name)
                    rows = _kind_size(k1, self.nd_big)
                    cols = _kind_size(k2, self.nd_big)
                    out_max = _kind_size(ko, self.nd_big)
                    self.assertEqual(len(t), rows,
                                     f"{variant}/{name} bad row count")
                    self.assertEqual(len(t[0]), cols,
                                     f"{variant}/{name} bad col count")
                    for r in t:
                        for v in r:
                            self.assertGreaterEqual(v, 0,
                                f"{variant}/{name} produced negative {v}")
                            self.assertLess(v, out_max,
                                f"{variant}/{name} produced out-of-range {v} >= {out_max}")
                self.assertEqual(u.lift_variant, variant)


class TestRandomExtendPreservesSmallCells(unittest.TestCase):
    def test_small_subgrid_is_preserved(self):
        random.seed(0)
        nd_small, nd_big = 4, 8
        pool = _build_small_pool(nd_small, 1)
        big = lift_population(pool, nd_small, nd_big, "random_extend", 1)[0]
        for name, (k1, k2, _ko) in TABLE_SPECS.items():
            small = getattr(pool[0], name)
            big_t = getattr(big, name)
            s1_s, s2_s = _kind_size(k1, nd_small), _kind_size(k2, nd_small)
            for i in range(s1_s):
                for j in range(s2_s):
                    self.assertEqual(big_t[i][j], small[i][j],
                                     f"{name}[{i}][{j}] altered by random_extend")


class TestModularTileMatchesSmallByMod(unittest.TestCase):
    def test_modular_lookup_equivalence(self):
        random.seed(0)
        nd_small, nd_big = 4, 8
        pool = _build_small_pool(nd_small, 1)
        big = lift_population(pool, nd_small, nd_big, "modular_tile", 1)[0]
        for name, (k1, k2, _ko) in TABLE_SPECS.items():
            small = getattr(pool[0], name)
            big_t = getattr(big, name)
            s1_s = _kind_size(k1, nd_small)
            s2_s = _kind_size(k2, nd_small)
            s1_b = _kind_size(k1, nd_big)
            s2_b = _kind_size(k2, nd_big)
            for i in range(s1_b):
                for j in range(s2_b):
                    self.assertEqual(big_t[i][j], small[i % s1_s][j % s2_s],
                                     f"{name}[{i}][{j}] != small[{i%s1_s}][{j%s2_s}]")


class TestBlockTileShiftIsoBlockClosure(unittest.TestCase):
    """For block_tile_shift_iso, when the input is in block b, the OUTPUT
    must land in block b's slice of the output range."""

    def test_outputs_are_block_local_on_axis_0(self):
        random.seed(0)
        nd_small, nd_big = 4, 8
        k = nd_big // nd_small
        pool = _build_small_pool(nd_small, 4)
        big = lift_population(pool, nd_small, nd_big,
                              "block_tile_shift_iso", 1)[0]
        for name, (k1, _k2, ko) in TABLE_SPECS.items():
            big_t = getattr(big, name)
            s1_s = _kind_size(k1, nd_small)
            s_out_s = _kind_size(ko, nd_small)
            for i in range(_kind_size(k1, nd_big)):
                b_in = (i // s1_s) % k
                lo = b_in * s_out_s
                hi = lo + s_out_s
                for v in big_t[i]:
                    self.assertGreaterEqual(
                        v, lo, f"{name} row {i} (block {b_in}): {v} < {lo}")
                    self.assertLess(
                        v, hi, f"{name} row {i} (block {b_in}): {v} >= {hi}")


class TestBlockTilePermuteRouting(unittest.TestCase):
    def test_outputs_land_in_permuted_block(self):
        random.seed(0)
        nd_small, nd_big = 4, 8
        k = nd_big // nd_small
        pool = _build_small_pool(nd_small, 4)
        big = lift_population(pool, nd_small, nd_big,
                              "block_tile_permute", 1)[0]
        perm = big.lift_permutation
        # k=2: with non-identity rejection sampling, perm should be [1, 0].
        self.assertEqual(sorted(perm), list(range(k)))
        for name, (k1, _k2, ko) in TABLE_SPECS.items():
            big_t = getattr(big, name)
            s1_s = _kind_size(k1, nd_small)
            s_out_s = _kind_size(ko, nd_small)
            for i in range(_kind_size(k1, nd_big)):
                b_in = (i // s1_s) % k
                target_block = perm[b_in]
                lo = target_block * s_out_s
                hi = lo + s_out_s
                for v in big_t[i]:
                    self.assertGreaterEqual(v, lo,
                        f"{name} row {i} (block {b_in}->{target_block}): {v} < {lo}")
                    self.assertLess(v, hi,
                        f"{name} row {i} (block {b_in}->{target_block}): {v} >= {hi}")


if __name__ == "__main__":
    unittest.main()
