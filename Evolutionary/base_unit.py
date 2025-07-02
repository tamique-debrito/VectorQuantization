from __future__ import annotations
import random
from typing import Optional

from Evolutionary.abstract_unit import AbstractBaseUnit, TABLE_2D, Vec, Mat

def base_unit_factory(num_vec, num_mat=None, num_update_obj=None):
    class BaseUnit(AbstractBaseUnit):
        def __init__(self, w: Mat) -> None:
            self.w = w
            self.v_out: Optional[Vec] = None

        def random_init(self) -> None:
            self.w = Mat(random.randrange(BaseUnit.NUM_OPERATOR_OBJ))

        def forward(self, v: Vec) -> Vec:
            self.v_out = Vec(v_idx=BaseUnit.FORWARD_TABLE[v.v_idx][self.w.w_idx])
            return self.v_out
        
        def find_back_info(self, target: Vec):
            assert self.v_out is not None, "No out vector when finding back info"
            return Vec(v_idx=BaseUnit.FIND_BACK_INFO_TABLE[target.v_idx][self.v_out.v_idx])
        
        def backward(self, v_back_info: Vec):
            return Vec(v_idx=BaseUnit.BACKWARD_TABLE[v_back_info.v_idx][self.w.w_idx])
        
        def update(self, v_back_info: Vec):
            assert self.v_out is not None, "No out vector during update"
            update_obj = BaseUnit.FIND_UPDATE_TABLE[v_back_info.v_idx][self.v_out.v_idx]
            self.w.w_idx = BaseUnit.APPLY_UPDATE_TABLE[update_obj][self.w.w_idx]

        @staticmethod
        def init_table(in_n_1, in_n_2, out_n) -> TABLE_2D:
            return [[random.randrange(out_n) for _ in range(in_n_2)] for _ in range(in_n_1)]
        
        @staticmethod
        def initialize_tables(num_vec, num_mat=None, num_update_obj=None):
            BaseUnit.NUM_DATA_OBJ = num_vec
            if num_mat is None:
                num_mat = num_vec ** 2
            BaseUnit.NUM_OPERATOR_OBJ = num_mat
            if num_update_obj is None:
                num_update_obj = num_mat
            BaseUnit.NUM_UPDATE_OBJ = num_update_obj
            
            BaseUnit.FORWARD_TABLE = BaseUnit.init_table(BaseUnit.NUM_DATA_OBJ, BaseUnit.NUM_OPERATOR_OBJ, BaseUnit.NUM_DATA_OBJ)
            BaseUnit.FIND_BACK_INFO_TABLE = BaseUnit.init_table(BaseUnit.NUM_DATA_OBJ, BaseUnit.NUM_DATA_OBJ, BaseUnit.NUM_DATA_OBJ)
            BaseUnit.BACKWARD_TABLE = BaseUnit.init_table(BaseUnit.NUM_DATA_OBJ, BaseUnit.NUM_OPERATOR_OBJ, BaseUnit.NUM_DATA_OBJ)
            BaseUnit.FIND_UPDATE_TABLE = BaseUnit.init_table(BaseUnit.NUM_DATA_OBJ, BaseUnit.NUM_DATA_OBJ, BaseUnit.NUM_UPDATE_OBJ)
            BaseUnit.APPLY_UPDATE_TABLE = BaseUnit.init_table(BaseUnit.NUM_UPDATE_OBJ, BaseUnit.NUM_OPERATOR_OBJ, BaseUnit.NUM_OPERATOR_OBJ)

    BaseUnit.initialize_tables(num_vec, num_mat, num_update_obj)

    return BaseUnit