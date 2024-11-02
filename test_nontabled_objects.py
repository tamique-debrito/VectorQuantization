from Model import LinLayer
from NontabledObjects import NontabledObject, Vec, Mat
import random

import numpy as np

def test_lin_layer_single():
    x = Vec()
    g = Vec()
    lin = LinLayer()

    print(f'before grad_step: {lin}')

    y = lin.forward(x)
    lin.backward(g, x)
    lin.step()

    print(f'after grad_step: {lin}')

    print('done')

def test_lin_layer_multi():
    x = Vec()
    g = Vec()
    lin1 = LinLayer()
    lin2 = LinLayer()

    print(f'before grad_step: {lin1}, {lin2}')

    z = lin1.forward(x)
    y = lin2.forward(x)
    g = lin2.backward(g, z)
    lin1.backward(g, x)

    lin1.step()
    lin2.step()


    print(f'after grad_step: {lin1}, {lin2}')

    print('done')

def test_simple_train():
    lin = LinLayer()


    toy_ds = [(Vec(), Vec()) for _ in range(20)]

    err = 0
    for input, target in toy_ds:
        pred = lin.forward(input)
        err += np.abs((pred - target).v).mean()
    print(f'base: err={err:.4f}')

    for e in range(20):
        err = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            pred = lin.forward(input)
            grad = pred.grad_l1_wrt(target)
            lin.backward(grad, input)
            lin.step()
            err += np.abs((pred - target).v).mean()
        print(f'train: err={err:.4f}')

    print('done')

def test_simple_train_2_layer():
    lin1 = LinLayer()
    lin2 = LinLayer()

    toy_ds = [(Vec(), Vec()) for _ in range(10)]

    err = 0
    for input, target in toy_ds:
        z = lin1.forward(input)
        pred = lin2.forward(z)
        err += np.abs((pred - target).v).mean()
    print(f'base: err={err:.4f}')

    for e in range(20):
        err = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            z = lin1.forward(input)
            pred = lin2.forward(z)

            grad = pred.grad_l1_wrt(target)
            grad = lin2.backward(grad, z)
            grad = lin1.backward(grad, input)
            lin1.step()
            lin2.step()
            err += np.abs((pred - target).v).mean()
        print(f'train: err={err:.4f}')


    print('done')


def test_simple_train_3_layer():
    lin1 = LinLayer()
    lin2 = LinLayer()
    lin3 = LinLayer()

    toy_ds = [(Vec(), Vec()) for _ in range(10)]

    err = 0
    for input, target in toy_ds:
        z1 = lin1.forward(input)
        z2 = lin2.forward(z1)
        pred = lin3.forward(z2)
        err += np.abs((pred - target).v).mean()
    print(f'base: err={err:.4f}')

    for e in range(20):
        err = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            z1 = lin1.forward(input)
            z2 = lin2.forward(z1)
            pred = lin3.forward(z2)

            grad = pred.grad_l1_wrt(target)
            grad = lin3.backward(grad, z2)
            grad = lin2.backward(grad, z1)
            grad = lin1.backward(grad, input)
            lin1.step()
            lin2.step()
            lin3.step()
            err += np.abs((pred - target).v).mean()
        print(f'train: err={err:.4f}')


    print('done')



if __name__ == "__main__":
    LinLayer.set_classes(Vec, Mat)
    NontabledObject.set_params(8, 0.001)
    # test_lin_layer_single()
    # test_lin_layer_multi()
    # test_simple_train()
    # test_simple_train_2_layer()
    test_simple_train_3_layer()
