import random

from CreationScript import NAMES

class TabledObject:
    tables = None
    n_scalar = None
    n_vec = None
    n_mat = None
    @staticmethod
    def load_tables():
        import pickle
        names = NAMES
        tables = dict()
        for name in names:
            with open('precomputed/' + name, 'rb') as f:
                tables[name] = pickle.load(f)

        TabledObject.tables = tables
        TabledObject.n_scalar = len(TabledObject.tables['scale_vec'])
        TabledObject.n_vec = len(TabledObject.tables['activ'])
        TabledObject.n_mat = len(TabledObject.tables['add_mat'])

class Scalar(TabledObject):
    s: int
    def __init__(self, s=None):
        if s is None:
            s = random.randrange(self.n_scalar)
        self.s = s
    

class Vec(TabledObject):
    def __init__(self, v=None):
        if v is None:
            v = random.randrange(self.n_vec)
        assert isinstance(v, int), "Must have integer index"
        self.v = v

    def mult_int(self, n: int):
        v = Vec(self.v)
        for _ in range(n - 1):
            v = v + self
        return v
    
    def scale(self, scalar):
        return Vec(self.tables['scale_vec'][scalar.s][self.v])

    def __add__(self, other_vec):
        return Vec(self.tables['add_vec'][self.v][other_vec.v])

    def __sub__(self, other_vec):
        return Vec(self.tables['sub_vec'][self.v][other_vec.v])

    def __mul__(self, mat):
        return Vec(self.tables['prod_t'][self.v][mat.w])

    def __pow__(self, other_vec):
        return Mat(self.tables['prod_outer'][self.v][other_vec.v])

    def grad_l1_wrt(self, other_vec):
        return Vec(self.tables['grad_l1'][self.v][other_vec.v])

    def activ(self):
        return Vec(self.tables['activ'][self.v])

    def mask_by(self, other):
        return Vec(self.tables['grad_mask'][self.v][other.v])

    def l1_err_from(self, other_vec):
        return self.tables['l1_err'][self.v][other_vec.v]

    def __str__(self):
        return f'vec_{self.v}'

    def __eq__(self, other):
        return self.v == other.v


class Mat(TabledObject):
    def __init__(self, w=None):
        if w is None:
            w = random.randrange(self.n_mat)
        self.w = w

    def mult_int(self, n: int):
        w = Mat(self.w)
        for _ in range(n - 1):
            w = w + self
        return w

    def scale(self, scalar):
        return Mat(self.tables['scale_mat'][scalar.s][self.w])

    def __add__(self, other_mat):
        return Mat(self.tables['add_mat'][self.w][other_mat.w])

    def __sub__(self, other_mat):
        return Mat(self.tables['sub_mat'][self.w][other_mat.w])

    def __mul__(self, vec):
        return Vec(self.tables['prod_t'][vec.v][self.w])

    def __str__(self):
        return f'mat_{self.w}'

    def __eq__(self, other):
        return self.w == other.w