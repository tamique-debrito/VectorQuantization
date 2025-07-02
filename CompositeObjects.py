#TODO: Actually try out using this concept

import random

class Composite:
    add_vec = None
    sub_vec = None
    add_mat = None
    sub_mat = None
    prod = None
    prod_t = None
    prod_outer = None

    @staticmethod
    def init(add_vec, sub_vec, add_mat, sub_mat, prod, prod_t, prod_outer):
        Composite.add_vec = add_vec
        Composite.sub_vec = sub_vec
        Composite.add_mat = add_mat
        Composite.sub_mat = sub_mat
        Composite.prod = prod
        Composite.prod_t = prod_t
        Composite.prod_outer = prod_outer

    def __init__(self):
        pass

class CompositeVec(Composite):
    def __init__(self, d, v=None):
        super().__init__()
        if v is None:
            v = [random.randrange(len(self.add_vec)) for _ in range(d)]
        self.v = v

        self.d = d

    def __add__(self, other):
        assert self.d == other.D, f'dim mismatch on add ({self.d} {other.D})'
        v = [self.add_vec[i, j] for i, j in zip(self.v, other.v)]
        return CompositeVec(d=self.d, v=v)

    def __sub__(self, other):
        assert self.d == other.D, f'dim mismatch on add ({self.d} {other.D})'
        v = [self.sub_vec[i, j] for i, j in zip(self.v, other.v)]
        return CompositeVec(d=self.d, v=v)

    def __pow__(self, other):  # outer product
        assert self.d == other.D, f'dim mismatch on add ({self.d} {other.D})'
        m = [[self.prod_outer[i, j] for j in other.v] for i in self.v]
        return CompositeMat(d_in=other.D, d_out=self.d, m=m)

class CompositeMat(Composite):
        def __init__(self, d_in, d_out, m=None):
            super().__init__()
            if m is None:
                m = [[random.randrange(len(self.add_mat)) for _ in range(d_out)] for _ in range(d1)]
            self.m = m

            self.d1 = d1
            self.d2 = d2

        def __prod__(self, vec):
            assert self.d == other.D, f'dim mismatch on add ({self.d} {other.D})'
            v = [self.add_vec[i, j] for i, j in zip(self.v, other.v)]
            return CompositeVec(d=self.d, v=v)

        def __sub__(self, other):
            assert self.d == other.D, f'dim mismatch on add ({self.d} {other.D})'
            v = [self.sub_vec[i, j] for i, j in zip(self.v, other.v)]
            return CompositeVec(d=self.d, v=v)

        def __pow__(self, other):  # outer product
            assert self.d == other.D, f'dim mismatch on add ({self.d} {other.D})'
            v = [self.prod_outer[i, j] for i, j in zip(self.v, other.v)]
            return CompositeMat(d=self.d, v=v)