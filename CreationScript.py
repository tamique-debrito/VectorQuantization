import torch
import random

N = 2 ** 10
D = 8

X = torch.rand(N, D) * 2 - 1

mask = (torch.rand(N, D) > 0.2)

X = X * ((X > 0) | mask)

for _ in range(3):
    for i in range(len(X)):
        if random.random() > 0.8:
            a, b = random.randrange(len(X)), random.randrange(len(X))
            X[i] = X[a] + X[b] * (1 if random.random() > 0 else -1)

W = torch.rand(N, D, D)
#w_mask = torch.rand(N, D, D)
#W = W * (w_mask > 0.3)
for _ in range(3):
    for i in range(len(W)):
        if random.random() > 0.8:
            a, b = random.randrange(len(W)), random.randrange(len(W))
            W[i] = W[a] + W[b] * (1 if random.random() > 0 else -1)

X = X.cuda()
W = W.cuda()

Activ = []

Grad_mask = []

Grad_L1 = []

Add_x = []
Sub_x = []

Add_w = []
Sub_w = []

Prod = []
Prod_t = []

Prod_outer = []

L1_err = []

################### Generation

# ReLU activation
# table is i -> ReLU(xi)
print(f'\rReLU: 1/1', end='')
act = X * (X > 0)
Activ = (X.unsqueeze(0) - act.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
print('\rReLU done!')

# l1 errors
for i, x in enumerate(X):  # table is (i, j) -> D/dxi ||xi - xj||1
    print(f'\rl1 errorrs: {i + 1}/{len(X)}', end='')
    e = (x.unsqueeze(0) - X).abs().mean(dim=1).tolist()
    L1_err.append(e)
print('\rl1 errors done!')

# l1 grads
lr = 1.0
for i, x in enumerate(X):  # table is (i, j) -> D/dxi ||xi - xj||1
    print(f'\rl1 grad: {i + 1}/{len(X)}', end='')
    g = (2 * (x.unsqueeze(0) > X) - 1) * lr
    y = (X.unsqueeze(0) - g.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
    Grad_L1.append(y)
print('\rl1 grad done!')

# mask by activation (for gradients)
for i, x in enumerate(X):  # table is (i, j) -> gi * (xj > 0)
    print(f'\rgrad mask: {i + 1}/{len(X)}', end='')
    m = x.unsqueeze(0) * (X > 0)
    y = (X.unsqueeze(0) - m.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
    Grad_mask.append(y)
print('\rgrad mask done!')

# vec addition
for i, x in enumerate(X):  # table is (i, j) -> xi + xj
    print(f'\rvec addition: {i + 1}/{len(X)}', end='')
    s = x.unsqueeze(0) + X
    y = (X.unsqueeze(0) - s.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
    Add_x.append(y)
print('\rvec addition done!')

# vec subtraction
for i, x in enumerate(X):  # table is (i, j) -> xi - xj
    print(f'\rvec subtraction: {i + 1}/{len(X)}', end='')
    s = x.unsqueeze(0) - X
    y = (X.unsqueeze(0) - s.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
    Sub_x.append(y)
print('\rvec subtraction done!')

# mat addition
for i, w in enumerate(W):  # table is (i, j) -> Wi + Wj
    print(f'\rmat addition: {i + 1}/{len(W)}', end='')
    s = w.unsqueeze(0) + W
    y = (W.unsqueeze(0) - s.unsqueeze(1)).abs().mean(dim=(2, 3)).min(dim=1).indices.tolist()
    Add_w.append(y)
print('\rmat addition done!')

# mat subtraction
for i, w in enumerate(W):  # table is (i, j) -> Wi - Wj
    print(f'\rmat subtraction: {i + 1}/{len(W)}', end='')
    s = w.unsqueeze(0) - W
    y = (W.unsqueeze(0) - s.unsqueeze(1)).abs().mean(dim=(2, 3)).min(dim=1).indices.tolist()
    Sub_w.append(y)
print('\rmat subtraction done!')

# product
for i, x in enumerate(X):  # table is (i, j) -> Wj xi
    print(f'\rproduct: {i + 1}/{len(X)}', end='')
    v = torch.matmul(W, x)
    t = (X.unsqueeze(0) - v.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
    Prod.append(t)
print('\rproduct done!')

# transposed product
for i, x in enumerate(X):  # table is (i, j) -> xiT Wj
    print(f'\rtransposed product: {i + 1}/{len(X)}', end='')
    v = torch.matmul(x, W)
    t = (X.unsqueeze(0) - v.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()
    Prod_t.append(t)
print('\rtransposed product done!')

#outer product
for i, x in enumerate(X):  # table is (i, j) -> xi xjT
    print(f'\rvec outer prod: {i + 1}/{len(W)}', end='')
    w = x.unsqueeze(0).unsqueeze(2) * X.unsqueeze(1)
    w_p = (W.unsqueeze(0) - w.unsqueeze(1)).abs().mean(dim=(2, 3)).min(dim=1).indices.tolist()
    Prod_outer.append(w_p)
print('\rvec outer prod done!')


################### Evaluation

err = 0
print(f'\rchecking ReLU error: 1/1', end='')
true_vecs = X * (X > 0)
approxes = X[[Activ[i] for i in range(N)]]
err = (true_vecs - approxes).abs().mean()
print(f'\rReLU l1 err={err / N ** 2:.6f}')

err = 0
for i, x in enumerate(X):
    print(f'\rchecking l1 grad error: {i + 1}/{N}', end='')
    true_grads = (2 * (x.unsqueeze(0) > X) - 1) * lr
    approxes = X[[Grad_L1[i][j] for j in range(N)]]
    err += (true_grads - approxes).abs().mean()

print(f'\rl1 grad l1 err={err / N ** 2:.6f}')

err = 0
for i, x in enumerate(X):
    print(f'\rchecking vec add error: {i + 1}/{N}', end='')
    true_vecs = x.unsqueeze(0) + X
    approxes = X[[Add_x[i][j] for j in range(N)]]
    err += (true_vecs - approxes).abs().mean()

print(f'\rvec add l1 err={err / N ** 2:.6f}')

err = 0
for i, x in enumerate(X):
    print(f'\rchecking vec sub error: {i + 1}/{N}', end='')
    true_vecs = x.unsqueeze(0) - X
    approxes = X[[Sub_x[i][j] for j in range(N)]]
    err += (true_vecs - approxes).abs().mean()

print(f'\rvec sub l1 err={err / N ** 2:.6f}')

err = 0
for i, w in enumerate(W):
    print(f'\rchecking mat add error: {i + 1}/{N}', end='')
    true_mats = w.unsqueeze(0) + W
    approxes = W[[Add_w[i][j] for j in range(N)]]
    err += (true_mats - approxes).abs().mean()

print(f'\rmat add l1 err={err / N ** 2:.6f}')

err = 0
for i, w in enumerate(W):
    print(f'\rchecking mat sub error: {i + 1}/{N}', end='')
    true_mats = w.unsqueeze(0) - W
    approxes = W[[Sub_w[i][j] for j in range(N)]]
    err += (true_mats - approxes).abs().mean()

print(f'\rmat sub l1 err={err / N ** 2:.6f}')


err = 0
for i, x in enumerate(X):
    print(f'\rchecking prod error: {i + 1}/{N}', end='')
    true_vecs = torch.matmul(W, x)
    approxes = X[[Prod[i][j] for j in range(N)]]
    err += (true_vecs - approxes).abs().mean()

print(f'\rprod l1 err={err / N ** 2:.6f}')

err = 0
for i, x in enumerate(X):
    print(f'\rchecking prod_t error: {i + 1}/{N}', end='')
    true_vecs = torch.matmul(x, W)
    approxes = X[[Prod_t[i][j] for j in range(N)]]
    err += (true_vecs - approxes).abs().mean()

print(f'\rprod_t l1 err={err / N ** 2:.6f}')


err = 0
for i, x in enumerate(X):
    print(f'\rchecking outer prod error: {i + 1}/{N}', end='')
    true_mats = x.unsqueeze(0).unsqueeze(2) * X.unsqueeze(1)
    approxes = W[[Prod_outer[i][j] for j in range(N)]]
    err += (true_mats - approxes).abs().mean()

print(f'\router prod l1 err={err / N ** 2:.6f}')


############## Preservation

import pickle
NAMES = ['activ', 'grad_l1', 'grad_mask', 'add_vec', 'sub_vec', 'add_mat', 'sub_mat', 'prod', 'prod_t', 'prod_outer', 'l1_err']
DATA = [Activ, Grad_L1, Grad_mask, Add_x, Sub_x, Add_w, Sub_w, Prod, Prod_t, Prod_outer, L1_err]

assert len(NAMES) == len(DATA), 'names should match data'

for name, data in zip(NAMES, DATA):
    with open('precomputed/' + name, 'wb') as f:
        pickle.dump(data, f)