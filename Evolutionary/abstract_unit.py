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
    # Table dimensions. There are four semantic "kinds" of object an axis can
    # range over: DATA (the vectors a unit reads/writes), OPERATOR (the
    # parameter `w`), BACK_INFO (the analogue of a gradient — produced by
    # `find_back_info` and propagated by `backward`), and UPDATE (an
    # intermediate object consumed by `apply_update`).
    #
    # BACK_INFO is conceptually distinct from DATA, but defaults to the same
    # count because in standard backprop the gradient lives in the same
    # space as the input vector. Nothing forces them to be equal.
    NUM_DATA_OBJ: int       # Vectors
    NUM_OPERATOR_OBJ: int   # Matrices
    NUM_BACK_INFO_OBJ: int  # Gradient-analogues; defaults to NUM_DATA_OBJ
    NUM_UPDATE_OBJ: int     # Intermediate update objects

    # Lookup tables. Each annotation reads as (axis_in1, axis_in2) -> axis_out.
    FORWARD_TABLE: TABLE_2D        # (DATA,      OPERATOR) -> DATA          ; (v_in,        w)     -> v_out
    FIND_BACK_INFO_TABLE: TABLE_2D # (DATA,      DATA)     -> BACK_INFO     ; (v_target,    v_out) -> v_back_info
    BACKWARD_TABLE: TABLE_2D       # (BACK_INFO, OPERATOR) -> BACK_INFO     ; (v_back_info, w)     -> v_back_info' (propagated to the input side)
    FIND_UPDATE_TABLE: TABLE_2D    # (BACK_INFO, DATA)     -> UPDATE        ; (v_back_info, v_out) -> update_obj
    APPLY_UPDATE_TABLE: TABLE_2D   # (UPDATE,    OPERATOR) -> OPERATOR      ; (update_obj,  w)     -> w'

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