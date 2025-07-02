from abc import ABC, abstractmethod
from typing import List, Optional, Any
from dataclasses import dataclass

# Assuming TABLE_2D, Vec, Mat are defined elsewhere or will be imported
# For now, let's just define a placeholder for TABLE_2D
TABLE_2D = List[List[int]]

@dataclass
class Vec:
    v_idx: int
@dataclass
class Mat:
    w_idx: int

class AbstractBaseUnit(ABC):
    # table dimensions
    NUM_DATA_OBJ: int # Vectors
    NUM_OPERATOR_OBJ: int # Matrices
    NUM_UPDATE_OBJ: int # Something else, but could be thought of as matrices

    # lookup tables
    FORWARD_TABLE: TABLE_2D # NUM_DATA_OBJ x NUM_OPERATOR_OBJ -> NUM_DATA_OBJ
    FIND_BACK_INFO_TABLE: TABLE_2D # NUM_DATA_OBJ x NUM_DATA_OBJ -> NUM_DATA_OBJ (v_target, v_out) -> v_back_info
    BACKWARD_TABLE: TABLE_2D # NUM_DATA_OBJ x NUM_OPERATOR_OBJ -> NUM_DATA_OBJ
    FIND_UPDATE_TABLE: TABLE_2D # NUM_DATA_OBJ x NUM_DATA_OBJ -> NUM_UPDATE_OBJ; (v_back_info, v_out) -> update_val
    APPLY_UPDATE_TABLE: TABLE_2D # NUM_OPERATOR_OBJ x NUM_UPDATE_OBJ -> NUM_OPERATOR_OBJ

    @abstractmethod
    def __init__(self, w) -> None:
        # w should be of type Mat, but for abstract class, we can leave it general
        pass

    def random_init(self) -> None:
        pass

    @abstractmethod
    def forward(self, v) -> Any:
        # v should be of type Vec, return type Vec
        pass
    
    @abstractmethod
    def find_back_info(self, target) -> Any:
        # target should be of type Vec, return type Vec
        pass
    
    @abstractmethod
    def backward(self, v_back_info) -> Any:
        # v_back_info should be of type Vec, return type Vec
        pass
    
    @abstractmethod
    def update(self, v_back_info) -> None:
        # v_back_info should be of type Vec
        pass

    # Static methods can be implemented in the abstract class or overridden
    @staticmethod
    @abstractmethod
    def init_table(in_n_1, in_n_2, out_n) -> TABLE_2D:
        pass

    @staticmethod
    @abstractmethod
    def initialize_tables(num_vec, num_mat=None, num_update_obj=None) -> None:
        pass 