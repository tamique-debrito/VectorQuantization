from typing import Optional
from TabledObjects import Scalar, Mat, Vec


class LinLayer:
    @staticmethod
    def set_classes(Vec, Mat):
        LinLayer.Vec = Vec
        LinLayer.Mat = Mat
    def __init__(self, w=None, b=None):
        if w is None:
            w = LinLayer.Mat()
        if b is None:
            b = LinLayer.Vec()
        self.w = w
        self.b = b

        self.d_w: Optional[Mat] = None
        self.d_b: Optional[Vec] = None
        self.a = None

    def forward(self, x):
        x = self.w * x
        x = x + self.b
        x = x.activ()
        self.a = x
        return x

    def backward(self, grad, x):
        assert self.a is not None, "tried to pass backward with no stored activation"
        grad = grad.mask_by(self.a)

        self.d_w = grad ** x
        self.d_b = grad

        new_grad = grad * self.w

        self.a = None

        return new_grad

    def step(self, lr=None):
        if lr is None:
            lr = 1
        assert self.d_w is not None and self.d_b is not None, "Attempted to step when saved gradients are None"
        if isinstance(lr, Scalar):
            self.w = self.w - self.d_w.scale(lr)
            self.b = self.b - self.d_b.scale(lr)
        else:
            self.w = self.w - self.d_w.mult_int(lr)
            self.b = self.b - self.d_b.mult_int(lr)
        self.d_w = None
        self.d_b = None

    def __str__(self):
        return f'lin (w={self.w}, b={self.b})'
