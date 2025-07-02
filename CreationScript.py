import torch
import random
NAMES = ['scale_vec', 'scale_mat', 'activ', 'grad_l1', 'grad_mask', 'add_vec', 'sub_vec', 'add_mat', 'sub_mat', 'prod', 'prod_t', 'prod_outer', 'l1_err']

NUM_VECTORS = 2 ** 11
NUM_SCALARS = 10
NUM_MATRICES = NUM_VECTORS # Start with just using the same number of matrices as vectors
VECTOR_DIMENSION = 4

def run_precompute():
    SCALARS = torch.tensor([1 / NUM_SCALARS * (i + 1) for i in range(NUM_SCALARS)])
    VECTORS = torch.rand(NUM_VECTORS, VECTOR_DIMENSION) * 2 - 1
    MATRICES = torch.rand(NUM_MATRICES, VECTOR_DIMENSION, VECTOR_DIMENSION) * 2 - 1


    def quantize_vec(vec2):
        return (VECTORS.unsqueeze(0) - vec2.unsqueeze(1)).abs().mean(dim=2).min(dim=1).indices.tolist()

    def quantize_mat(mat2):
        return (MATRICES.unsqueeze(0) - mat2.unsqueeze(1)).abs().mean(dim=(2, 3)).min(dim=1).indices.tolist()
    
    MATRICES = MATRICES * torch.triu(torch.ones(VECTOR_DIMENSION, VECTOR_DIMENSION)).unsqueeze(0)

    # Do various things to try and make the distribution better
    mask = (torch.rand(NUM_VECTORS, VECTOR_DIMENSION) > 0.4)

    VECTORS = VECTORS * ((VECTORS > 0) | mask) # Get some zeros in there (for relu activations I guess) and bias towards positivity


    for _ in range(3):# Try to make sure that some vecs are the addition/subtraction of other vecs
        for i in range(len(VECTORS)):
            if random.random() > 0.8:
                a, b = random.randrange(len(VECTORS)), random.randrange(len(VECTORS))
                VECTORS[i] = VECTORS[a] + VECTORS[b] * (1 if random.random() > 0 else -1)

    #w_mask = torch.rand(N, D, D)
    #W = W * (w_mask > 0.3)
    for _ in range(3): # Try to make sure that some matrices are the addition/subtraction of other matrices
        for i in range(len(MATRICES)):
            if random.random() > 0.8:
                a, b = random.randrange(len(MATRICES)), random.randrange(len(MATRICES))
                MATRICES[i] = MATRICES[a] + MATRICES[b] * (1 if random.random() > 0 else -1)

    VECTORS = VECTORS.cuda()
    MATRICES = MATRICES.cuda()

    Scale_vec = []

    Activ = []

    Grad_mask = []

    Grad_L1 = []

    Add_x = []
    Sub_x = []

    Scale_mat = []

    Add_w = []
    Sub_w = []

    Prod = []
    Prod_t = []

    Prod_outer = []

    L1_err = []

    ################### Generation

    # Vector scaling
    # table is (scalar, vector) -> scalar * vector
    for i, s in enumerate(SCALARS):
        print(f'\rScaling vectors: {i + 1}/{len(SCALARS)}', end='')
        scaled = VECTORS * s
        scaled_quantized = quantize_vec(scaled)
        Scale_vec.append(scaled_quantized)
    print('\rScaling vectors done!                            ')


    # ReLU activation
    # table is i -> ReLU(xi)
    print(f'\rReLU: 1/1', end='')
    act = VECTORS * (VECTORS > 0)
    Activ = quantize_vec(act)
    print('\rReLU done!                            ')

    # l1 errors
    for i, x in enumerate(VECTORS):  # table is (i, j) -> D/dxi ||xi - xj||1
        print(f'\rl1 errorrs: {i + 1}/{len(VECTORS)}', end='')
        e = (x.unsqueeze(0) - VECTORS).abs().mean(dim=1).tolist()
        L1_err.append(e)
    print('\rl1 errors done!                            ')

    # l1 grads
    lr = 1.0
    for i, x in enumerate(VECTORS):  # table is (i, j) -> D/dxi ||xi - xj||1
        print(f'\rl1 grad: {i + 1}/{len(VECTORS)}', end='')
        g = (2 * (x.unsqueeze(0) > VECTORS) - 1) * lr
        y = quantize_vec(g)
        Grad_L1.append(y)
    print('\rl1 grad done!                            ')

    # mask by activation (for gradients)
    for i, x in enumerate(VECTORS):  # table is (i, j) -> gi * (xj > 0)
        print(f'\rgrad mask: {i + 1}/{len(VECTORS)}', end='')
        m = x.unsqueeze(0) * (VECTORS > 0)
        y = quantize_vec(m)
        Grad_mask.append(y)
    print('\rgrad mask done!                            ')

    # vec addition
    for i, x in enumerate(VECTORS):  # table is (i, j) -> xi + xj
        print(f'\rvec addition: {i + 1}/{len(VECTORS)}', end='')
        s = x.unsqueeze(0) + VECTORS
        y = quantize_vec(s)
        Add_x.append(y)
    print('\rvec addition done!                            ')

    # vec subtraction
    for i, x in enumerate(VECTORS):  # table is (i, j) -> xi - xj
        print(f'\rvec subtraction: {i + 1}/{len(VECTORS)}', end='')
        s = x.unsqueeze(0) - VECTORS
        y = quantize_vec(s)
        Sub_x.append(y)
    print('\rvec subtraction done!                            ')


    # matrix scaling
    # table is (scalar, matrix) -> scalar * matrix
    for i, s in enumerate(SCALARS):
        print(f'\rScaling matrices: {i + 1}/{len(SCALARS)}', end='')
        scaled = MATRICES * s
        scaled_quantized = quantize_mat(scaled)
        Scale_mat.append(scaled_quantized)
    print('\rScaling matrices done!                            ')

    # mat addition
    for i, w in enumerate(MATRICES):  # table is (i, j) -> Wi + Wj
        print(f'\rmat addition: {i + 1}/{len(MATRICES)}', end='')
        s = w.unsqueeze(0) + MATRICES
        y = quantize_mat(s)
        Add_w.append(y)
    print('\rmat addition done!                            ')

    # mat subtraction
    for i, w in enumerate(MATRICES):  # table is (i, j) -> Wi - Wj
        print(f'\rmat subtraction: {i + 1}/{len(MATRICES)}', end='')
        s = w.unsqueeze(0) - MATRICES
        y = quantize_mat(s)
        Sub_w.append(y)
    print('\rmat subtraction done!                            ')

    # product
    for i, x in enumerate(VECTORS):  # table is (i, j) -> Wj xi
        print(f'\rproduct: {i + 1}/{len(VECTORS)}', end='')
        v = torch.matmul(MATRICES, x)
        t = quantize_vec(v)
        Prod.append(t)
    print('\rproduct done!                            ')

    # transposed product
    for i, x in enumerate(VECTORS):  # table is (vec_i, mat_j) -> xiT Wj
        print(f'\rtransposed product: {i + 1}/{len(VECTORS)}', end='')
        v = torch.matmul(x, MATRICES)
        t = quantize_vec(v)
        Prod_t.append(t)
    print('\rtransposed product done!                            ')

    #outer product
    for i, x in enumerate(VECTORS):  # table is (i, j) -> xi xjT
        print(f'\rvec outer prod: {i + 1}/{len(MATRICES)}', end='')
        w = x.unsqueeze(0).unsqueeze(2) * VECTORS.unsqueeze(1)
        w_p = quantize_mat(w)
        Prod_outer.append(w_p)
    print('\rvec outer prod done!                            ')


    ################### Evaluation

    err = 0
    print(f'\rchecking scale vec error: 1/1', end='')
    for i, s in enumerate(SCALARS):
        true_vecs = VECTORS * s
        approxes = VECTORS[[Scale_vec[i][j] for j in range(NUM_VECTORS)]]
        err += (true_vecs - approxes).abs().mean()
    print(f'\rScale vec l1 err={err / NUM_SCALARS:.6f}                  ')

    err = 0
    print(f'\rchecking ReLU error: 1/1', end='')
    true_vecs = VECTORS * (VECTORS > 0)
    approxes = VECTORS[[Activ[i] for i in range(NUM_VECTORS)]]
    err = (true_vecs - approxes).abs().mean()
    print(f'\rReLU l1 err={err:.6f}                  ')

    err = 0
    for i, x in enumerate(VECTORS):
        print(f'\rchecking l1 grad error: {i + 1}/{NUM_VECTORS}', end='')
        true_grads = (2 * (x.unsqueeze(0) > VECTORS) - 1) * lr
        approxes = VECTORS[[Grad_L1[i][j] for j in range(NUM_VECTORS)]]
        err += (true_grads - approxes).abs().mean()
    print(f'\rl1 grad l1 err={err / NUM_VECTORS:.6f}                  ')

    err = 0
    for i, x in enumerate(VECTORS):
        print(f'\rchecking vec add error: {i + 1}/{NUM_VECTORS}', end='')
        true_vecs = x.unsqueeze(0) + VECTORS
        approxes = VECTORS[[Add_x[i][j] for j in range(NUM_VECTORS)]]
        err += (true_vecs - approxes).abs().mean()
    print(f'\rvec add l1 err={err / NUM_VECTORS:.6f}                  ')

    err = 0
    for i, x in enumerate(VECTORS):
        print(f'\rchecking vec sub error: {i + 1}/{NUM_VECTORS}', end='')
        true_vecs = x.unsqueeze(0) - VECTORS
        approxes = VECTORS[[Sub_x[i][j] for j in range(NUM_VECTORS)]]
        err += (true_vecs - approxes).abs().mean()
    print(f'\rvec sub l1 err={err / NUM_VECTORS:.6f}                  ')


    err = 0
    print(f'\rchecking scale mat error: 1/1', end='')
    for i, s in enumerate(SCALARS):
        true_vecs = MATRICES * s
        approxes = MATRICES[[Scale_mat[i][j] for j in range(NUM_MATRICES)]]
        err += (true_vecs - approxes).abs().mean()
    print(f'\rscale mat l1 err={err / NUM_SCALARS:.6f}                  ')

    err = 0
    for i, w in enumerate(MATRICES):
        print(f'\rchecking mat add error: {i + 1}/{NUM_VECTORS}', end='')
        true_mats = w.unsqueeze(0) + MATRICES
        approxes = MATRICES[[Add_w[i][j] for j in range(NUM_MATRICES)]]
        err += (true_mats - approxes).abs().mean()
    print(f'\rmat add l1 err={err / NUM_MATRICES:.6f}                  ')

    err = 0
    for i, w in enumerate(MATRICES):
        print(f'\rchecking mat sub error: {i + 1}/{NUM_VECTORS}', end='')
        true_mats = w.unsqueeze(0) - MATRICES
        approxes = MATRICES[[Sub_w[i][j] for j in range(NUM_MATRICES)]]
        err += (true_mats - approxes).abs().mean()
    print(f'\rmat sub l1 err={err / NUM_MATRICES:.6f}                  ')


    err = 0
    for i, x in enumerate(VECTORS):
        print(f'\rchecking prod error: {i + 1}/{NUM_VECTORS}', end='')
        true_vecs = torch.matmul(MATRICES, x)
        approxes = VECTORS[[Prod[i][j] for j in range(NUM_MATRICES)]]
        err += (true_vecs - approxes).abs().mean()
    print(f'\rprod l1 err={err / NUM_VECTORS:.6f}                  ')

    err = 0
    for i, x in enumerate(VECTORS):
        print(f'\rchecking prod_t error: {i + 1}/{NUM_VECTORS}', end='')
        true_vecs = torch.matmul(x, MATRICES)
        approxes = VECTORS[[Prod_t[i][j] for j in range(NUM_MATRICES)]]
        err += (true_vecs - approxes).abs().mean()
    print(f'\rprod_t l1 err={err / NUM_VECTORS:.6f}                  ')


    err = 0
    for i, x in enumerate(VECTORS):
        print(f'\rchecking outer prod error: {i + 1}/{NUM_VECTORS}', end='')
        true_mats = x.unsqueeze(0).unsqueeze(2) * VECTORS.unsqueeze(1)
        approxes = MATRICES[[Prod_outer[i][j] for j in range(NUM_VECTORS)]]
        err += (true_mats - approxes).abs().mean()
    print(f'\router prod l1 err={err / NUM_VECTORS:.6f}                  ')


    ############## Preservation

    import pickle

    DATA = [Scale_vec, Scale_mat, Activ, Grad_L1, Grad_mask, Add_x, Sub_x, Add_w, Sub_w, Prod, Prod_t, Prod_outer, L1_err]

    assert len(NAMES) == len(DATA), 'names should match data'

    for name, data in zip(NAMES, DATA):
        with open('precomputed/' + name, 'wb') as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    with torch.no_grad():
        run_precompute()