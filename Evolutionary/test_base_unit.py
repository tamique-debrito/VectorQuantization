import unittest
import random
from base_unit import base_unit_factory
from abstract_unit import Vec, Mat


class TestBaseUnit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Instantiate BaseUnit once for all tests
        cls.BaseUnit = base_unit_factory(
            num_vec=5,
            num_mat=5,
            num_update_obj=5)
        # Define dimensions for testing
        
    def setUp(self):
        # Reset random seed for reproducibility in each test if needed
        random.seed(42)
        # Re-initialize BaseUnit for each test to ensure a clean state
        self.BaseUnit = base_unit_factory(
            num_vec=5,
            num_mat=5,
            num_update_obj=5
        )

    def test_initialization(self):
        mat_idx = 0
        mat_obj = Mat(w_idx=mat_idx)
        unit = self.BaseUnit(w=mat_obj)
        self.assertEqual(unit.w.w_idx, mat_idx)
        self.assertIsNone(unit.v_out)

    def test_forward(self):
        mat_idx = 0
        vec_idx = 0
        mat_obj = Mat(w_idx=mat_idx)
        vec_obj = Vec(v_idx=vec_idx)
        unit = self.BaseUnit(w=mat_obj)

        expected_v_out_idx = self.BaseUnit.FORWARD_TABLE[vec_idx][mat_idx]
        
        v_out = unit.forward(vec_obj)
        assert unit.v_out is not None
        self.assertEqual(v_out.v_idx, expected_v_out_idx)
        self.assertEqual(unit.v_out.v_idx, expected_v_out_idx)

    def test_backward(self):
        mat_idx = 0
        vec_back_info_idx = 1 # Using a different index for backward info
        mat_obj = Mat(w_idx=mat_idx)
        vec_back_info_obj = Vec(v_idx=vec_back_info_idx)
        unit = self.BaseUnit(w=mat_obj)

        expected_v_back_idx = self.BaseUnit.BACKWARD_TABLE[vec_back_info_idx][mat_idx]

        v_back = unit.backward(vec_back_info_obj)
        self.assertEqual(v_back.v_idx, expected_v_back_idx)

    def test_update(self):
        mat_idx = 0
        vec_idx = 0
        vec_back_info_idx = 1

        mat_obj = Mat(w_idx=mat_idx)
        vec_obj = Vec(v_idx=vec_idx)
        vec_back_info_obj = Vec(v_idx=vec_back_info_idx)
        unit = self.BaseUnit(w=mat_obj)

        unit.forward(vec_obj)
        assert unit.v_out is not None

        expected_update_obj = self.BaseUnit.FIND_UPDATE_TABLE[vec_back_info_idx][unit.v_out.v_idx]
        expected_w_idx_after_update = self.BaseUnit.APPLY_UPDATE_TABLE[expected_update_obj][unit.w.w_idx]

        unit.update(vec_back_info_obj)
        self.assertEqual(unit.w.w_idx, expected_w_idx_after_update)

    def test_update_without_forward(self):
        mat_idx = 0
        vec_back_info_idx = 1

        mat_obj = Mat(w_idx=mat_idx)
        vec_back_info_obj = Vec(v_idx=vec_back_info_idx)
        unit = self.BaseUnit(w=mat_obj)

        with self.assertRaisesRegex(AssertionError, "No out vector during update"):
            unit.update(vec_back_info_obj)

if __name__ == "__main__":
    unittest.main() 