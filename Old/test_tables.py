from Old.Tables import *


def test_basic_ops(n_in_vec, n_mat, n_out_vec, d_in_vec, d_out_vec):
    v_in = Vec(n_in_vec, d_in_vec)
    v_out = Vec(n_out_vec, d_out_vec)
    w = Mat(v_in, n_mat, v_out)

    print('Add:')
    print(v_in.vecs[0])
    print(v_in.vecs[1])
    print(v_in.true_vec(v_in.add_op(0, 1)))
    print(v_in.true_vec(0) + v_in.true_vec(1))

    print('Multiply:')
    print(v_in.vecs[0])
    print(w.mats[0])
    print(v_out.true_vec(w.prod_op(0, 0)))
    print(np.matmul(w.true_mat(0), v_in.true_vec(0)))

    print('Activ:')
    print(v_in.true_vec(0))
    print(v_in.true_vec(v_in.activ_op(0)))


def test_error(n_in_vec, n_mat, n_out_vec, d_in_vec, d_out_vec):
    v_in = Vec(n_in_vec, d_in_vec)
    v_out = Vec(n_out_vec, d_out_vec)
    w = Mat(v_in, n_mat, v_out)

    activ_error = 0
    for i in range(v_in.n):
        v_approx = v_in.true_vec(v_in.activ_op(i))
        v_true = v_in.true_vec(i) * (v_in.true_vec(i) > 0)
        activ_error += ((v_approx - v_true) ** 2).mean()

    add_error = 0
    for i in range(v_in.n):
        for j in range(v_in.n):
            v_approx = v_in.true_vec(v_in.add_op(i, j))
            v_true = v_in.true_vec(i) + v_in.true_vec(j)
            add_error += ((v_approx - v_true) ** 2).mean()

    prod_error = 0
    for i in range(v_in.n):
        for j in range(w.n):
            v_approx = w.out_vecs.true_vec(w.prod_op(i, j))
            v_true = np.matmul(w.true_mat(j), v_in.true_vec(i))
            prod_error += ((v_approx - v_true) ** 2).mean()

    print(f"activ error (mse)={activ_error / v_in.n:.3f}", end=' | ')
    print(f"add error (mse)={add_error / (v_in.n ** 2):.3f}", end=' | ')
    print(f"prod error (mse)={prod_error / (v_in.n * w.n):.3f}")


def test_error_multi():
    dim_list = [8]
    n_vec_list = [512]
    n_mat_list = [16]

    for n_mat in n_mat_list:
        for n_vec in n_vec_list:
            for d in dim_list:
                print(f'dim={d}, n_vec={n_vec}, n_mat={n_mat}', end=': ')
                test_error(n_vec, n_mat, n_vec, d, d)


def test_lin_layer(n_in_vec, n_mat, n_out_vec, d_in_vec, d_out_vec):
    v_in = Vec(n_in_vec, d_in_vec)
    v_out = Vec(n_out_vec, d_out_vec)
    w = Mat(v_in, n_mat, v_out)

    lin = LinLayer(w)

    x = 0

    print(f'forward: {lin.forward(x)}')
    print(f'backward: {lin.backward(0, x)}')


def test_simple_train():
    d = 8
    n1 = 128
    n_mat = 64
    n2 = 16
    v_in = Vec(n1, d)
    v_out = Vec(n2, d)
    w = Mat(v_in, n_mat, v_out)

    lin = LinLayer(w)

    toy_ds = [(random.randrange(n1), random.randrange(n2)) for _ in range(50)]

    grads = gen_l2_grads(v_out)

    correct = 0
    for input, target in toy_ds:
        pred = lin.forward(input)
        correct += 1 if pred == target else 0

    print(f'base:  n_correct={correct}, lin_w={lin.w_i}, lin_b={lin.b_i}')

    for e in range(15):
        correct = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            pred = lin.forward(input)
            grad = grads[(pred, target)]
            lin.backward(grad, input)
            lin.step()
            correct += 1 if pred == target else 0
        print(f'train: n_correct={correct}, lin_w={lin.w_i}, lin_b={lin.b_i}')


def test_simple_train_2_layer():
    d = 16
    n1 = 64
    n2 = 128
    n3 = 16
    v_in = Vec(n1, d)
    v_mid = Vec(n2, d)
    v_out = Vec(n3, d)
    w1 = Mat(v_in, n2, v_mid)
    w2 = Mat(v_mid, n3, v_out)

    lin1 = LinLayer(w1)
    lin2 = LinLayer(w2)

    toy_ds = [(random.randrange(n1), random.randrange(n3)) for _ in range(50)]

    grads = gen_l2_grads(v_out)

    correct = 0
    for input, target in toy_ds:
        pred = lin2.forward(lin1.forward(input))
        correct += 1 if pred == target else 0

    print(f'base:  n_correct={correct}, lin1_w={lin1.w_i}, lin1_b={lin1.b_i}, lin2_w={lin2.w_i}, lin2_b={lin2.b_i}')

    for e in range(15):
        correct = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            z = lin1.forward(input)
            pred = lin2.forward(z)
            grad = grads[(pred, target)]
            grad_z = lin2.backward(grad, z)
            grad_in = lin1.backward(grad_z, input)
            lin1.step()
            lin2.step()
            correct += 1 if pred == target else 0
        print(f'train: n_correct={correct}, lin1_w={lin1.w_i}, lin1_b={lin1.b_i}, lin2_w={lin2.w_i}, lin2_b={lin2.b_i}')


if __name__ == "__main__":
    # test_basic_ops(16, 4, 16, 6, 5)
    # test_simple_train()
    test_simple_train_2_layer()
