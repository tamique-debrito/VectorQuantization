import numpy as np
from utils import project
import random


# Product tables are indexed by (vector_index, matrix_index)

def gen_l2_grads(vec_table):
    grads = dict()
    for i in range(vec_table.N):
        for j in range(vec_table.N):
            grads[(i, j)] = project(2.5 * (vec_table.true_vec(i) - vec_table.true_vec(j)), vec_table.vecs)
    return grads

def gen_vec_items(n, d):
    vecs = np.random.uniform(-1, 1, (n, d))
    add = dict()
    sub = dict()
    activ = dict()
    for i in range(n):
        for j in range(n):
            add[(i, j)] = project(vecs[i] + vecs[j], vecs)
            sub[(i, j)] = project(vecs[i] - vecs[j], vecs)

    for i in range(n):
        activ[i] = project(vecs[i] * (vecs[i] > 0), vecs)

    return vecs, add, sub, activ


def gen_mat_items(n, d_in, d_out):
    mats = np.random.uniform(-1, 1, (n, d_out, d_in))
    add = dict()
    sub = dict()
    for i in range(n):
        for j in range(n):
            add[(i, j)] = project(mats[i] + mats[j], mats)
            sub[(i, j)] = project(mats[i] - mats[j], mats)

    return mats, add, sub


def gen_inner_prod_items(vecs1, vecs2, mats):
    prods = dict()
    transposed_prods = dict()
    for i in range(vecs1.shape[0]):
        for j in range(mats.shape[0]):
            prods[(i, j)] = project(np.matmul(mats[j], vecs1[i]), vecs2)

    for i in range(vecs2.shape[0]):
        for j in range(mats.shape[0]):
            transposed_prods[(i, j)] = project(np.matmul(vecs2[i], mats[j]), vecs1)

    return prods, transposed_prods


def gen_outer_prod_items(vecs1, vecs2, mats):
    outer_prods = dict()
    for i in range(vecs1.shape[0]):
        for j in range(vecs2.shape[0]):
            outer_prods[(j, i)] = project(np.outer(vecs2[j], vecs1[i]), mats)

    return outer_prods


class LinLayer:
    def __init__(self, mat_table, w_i=None, b_i=None):
        if w_i is None:
            w_i = random.randrange(mat_table.N)
        if b_i is None:
            b_i = random.randrange(mat_table.out_vecs.N)
        self.w_i = w_i
        self.b_i = b_i
        self.mat_table = mat_table

        self.d_w = None
        self.d_b = None

    def forward(self, x_i):
        y_i = self.mat_table.prod_op(x_i, self.w_i)
        y_i = self.mat_table.out_vecs.add_op(y_i, self.b_i)
        y_i = self.mat_table.out_vecs.activ_op(y_i)
        return y_i

    def backward(self, grad_i, x_i):
        grad_i = self.mat_table.out_vecs.activ_op(grad_i)

        self.d_w = self.mat_table.outer_prod_op(grad_i, x_i)
        self.d_b = grad_i

        new_grad_i = self.mat_table.prod_t_op(grad_i, self.w_i)

        return new_grad_i

    def step(self):
        assert self.d_w is not None and self.d_b is not None, "Attempted to step when grad is None"
        self.w_i = self.mat_table.sub_op(self.w_i, self.d_w)
        self.b_i = self.mat_table.out_vecs.sub_op(self.b_i, self.d_b)
        self.d_w = None
        self.d_b = None


class Vec:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.vecs, self.add, self.sub, self.activ = gen_vec_items(n, d)

    def add_op(self, i, j):
        return self.add[(i, j)]

    def sub_op(self, i, j):
        return self.sub[(i, j)]

    def activ_op(self, i):
        return self.activ[i]

    def true_vec(self, i):
        return self.vecs[i]


class Mat:
    def __init__(self, in_vecs, n, out_vecs):
        self.n = n
        self.in_vecs = in_vecs
        self.out_vecs = out_vecs
        self.mats, self.add, self.sub = gen_mat_items(n, in_vecs.D, out_vecs.D)
        self.prod, self.prod_t = gen_inner_prod_items(in_vecs.vecs, out_vecs.vecs, self.mats)
        self.outer_prod = gen_outer_prod_items(in_vecs.vecs, out_vecs.vecs, self.mats)

    def add_op(self, i, j):
        return self.add[(i, j)]

    def sub_op(self, i, j):
        return self.sub[(i, j)]

    def prod_op(self, vec_i, mat_i):
        return self.prod[(vec_i, mat_i)]

    def prod_t_op(self, vec_i, mat_i):
        return self.prod_t[(vec_i, mat_i)]

    def outer_prod_op(self, i, j):
        return self.outer_prod[(i, j)]

    def true_mat(self, i):
        return self.mats[i]
