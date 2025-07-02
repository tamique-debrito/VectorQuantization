import random
import numpy as np

class NontabledObject:
    d = None
    lr = None
    @staticmethod
    def set_params(d, lr):
        NontabledObject.d = d
        NontabledObject.lr = lr


class Scalar(NontabledObject):
    def __init__(self, s=None) -> None:
        if s is None:
            s = np.random.uniform(-1, 1)
        self.s = s
        

class Vec(NontabledObject):
    def __init__(self, v=None):
        if v is None:
            v = np.random.uniform(-1, 1, self.d)
        self.v = v

    def scale(self, s):
        return Vec(self.v * s.s)

    def __add__(self, other_vec):
        return Vec(self.v + other_vec.v)

    def __sub__(self, other_vec):
        return Vec(self.v - other_vec.v)

    def __mul__(self, mat):
        return Vec(np.matmul(mat.m, self.v))

    def __pow__(self, other_vec):
        return Mat(self.v[:, np.newaxis] * other_vec.v[np.newaxis, :])

    def grad_l1_wrt(self, other_vec):
        return Vec((self.v - other_vec.v) * ((self.v > other_vec.v) * 2 - 1) * NontabledObject.lr)

    def activ(self):
        return Vec(self.v * (self.v > np.zeros_like(self.v)))

    def mask_by(self, other):
        return Vec(self.v * (other.v > np.zeros_like(other.v)))

    def __str__(self):
        return f'vec_{self.v}'

    def __eq__(self, other):
        return self.v == other.v


class Mat(NontabledObject):
    def __init__(self, m=None):
        if m is None:
            m = np.random.uniform(-1, 1, (self.d, self.d))

        self.m = m

    def scale(self, s):
        return Mat(self.m * s.s)

    def __add__(self, other_mat):
        return Mat(self.m + other_mat.m)

    def __sub__(self, other_mat):
        return Mat(self.m - other_mat.m)

    def __mul__(self, vec):
        return Vec(np.matmul(self.m, vec.v))

    def __str__(self):
        return f'mat_{self.m}'

    def __eq__(self, other):
        return self.m == other.m