from Model import LinLayer
from TabledObjects import Scalar, TabledObject, Vec, Mat
import random

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


    toy_ds = get_toy_dataset(2, n=20)

    err = 0
    corr = 0
    for input, target in toy_ds:
        pred = lin.forward(input)
        err += pred.l1_err_from(target)
        if check_accuracy(pred, target, toy_ds):
            corr += 1
    print(f'Base err: err={err:.4f}')
    print(f'Base accuracy: {100 * corr / len(toy_ds):.1f}%')

    n_epoch = 200
    for e in range(n_epoch):
        err = 0
        corr = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            pred = lin.forward(input)
            grad = pred.grad_l1_wrt(target)
            lin.backward(grad, input)
            lin.step(Scalar(int((n_epoch - e) / n_epoch * 9)))
            err += pred.l1_err_from(target)
            if check_accuracy(pred, target, toy_ds):
                corr += 1
        if e % (n_epoch // 20 + 1) == 0:
            print(f'{e+1}/{n_epoch} train: err={err:.4f} | acc={100 * corr / len(toy_ds):.1f}%')

    print('done')

def test_simple_train_2_layer():
    lin1 = LinLayer()
    lin2 = LinLayer()


    toy_ds = get_toy_dataset(2, n=50)

    err = 0
    corr = 0
    for input, target in toy_ds:
        z = lin1.forward(input)
        pred = lin2.forward(z)
        err += pred.l1_err_from(target)
        if check_accuracy(pred, target, toy_ds):
            corr += 1
    print(f'Base err: err={err:.4f}')
    print(f'Base accuracy: {100 * corr / len(toy_ds):.1f}%')

    n_epoch = 50
    for e in range(n_epoch):
        err = 0
        corr = 0
        random.shuffle(toy_ds)
        for input, target in toy_ds:
            lr = Scalar(int((n_epoch - e) / n_epoch * 9))
            z = lin1.forward(input)
            pred = lin2.forward(z)

            grad = pred.grad_l1_wrt(target)
            #grad = (grad + grad) + grad + grad # This was probably just to scale the gradient
            grad = lin2.backward(grad, z)
            grad = lin1.backward(grad, input)
            lin1.step(lr)
            lin2.step(lr)
            err += pred.l1_err_from(target)
            if check_accuracy(pred, target, toy_ds):
                corr += 1
        if e % (n_epoch // 20 + 1) == 0:
            print(f'{e+1}/{n_epoch} train: err={err:.4f} | acc={100 * corr / len(toy_ds):.1f}%')



    print('done')

def check_accuracy(prediction, true_target, dataset):
    """
    Checks if the prediction is closer to the true target than any other target in the dataset.
    Returns True if the l1_err from the prediction to the true target is the minimum among all targets, False otherwise.
    """
    true_error = prediction.l1_err_from(true_target)
    min_error = float('inf')

    for _, dataset_target in dataset:
        error = prediction.l1_err_from(dataset_target)
        if error < min_error:
            min_error = error
    return true_error == min_error

def get_toy_dataset(dataset_type: int, n=10):
    x = [Vec() for _ in range(n)]
    dataset_name = ["identity", "linear", 'network'][dataset_type]

    if dataset_type == 0:
        toy_ds = [(v, v) for v in x]
    elif dataset_type == 1:
        s = Scalar()
        b = Vec()
        toy_ds = [(v, v.scale(s) + b) for v in x]
    elif dataset_type == 2:
        ref_lin1 = LinLayer()
        ref_lin2 = LinLayer()
        toy_ds = [(v, ref_lin2.forward(ref_lin1.forward(v))) for v in x]
    else:
        raise ValueError("invalid dataset type")

    print(f"created dataset type={dataset_name}")
    return toy_ds



if __name__ == "__main__":
    LinLayer.set_classes(Vec, Mat)
    TabledObject.load_tables()
    # test_lin_layer_single()
    # test_lin_layer_multi()
    # test_simple_train()
    test_simple_train_2_layer()
