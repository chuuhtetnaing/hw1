"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return (multiply(mul_scalar(out_grad, self.scalar), power_scalar(x, (self.scalar - 1))), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs
        return (multiply(out_grad, (power_scalar(y, -1))), multiply(out_grad, multiply(-x, (power_scalar(y, -2)))))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (mul_scalar(out_grad, (1/self.scalar)),)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return array_api.swapaxes(a, *self.axes)

        return array_api.swapaxes(a, -1, -2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return (transpose(out_grad, self.axes), )

        return (transpose(out_grad, (-1, -2)), )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, newshape=self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return (reshape(out_grad, x.shape), )
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # x, = node.inputs
        # x_shape = x.shape if isinstance(x, Tensor) else []
        # grad_shape = out_grad.shape
        # sum_index = None
        #
        # if len(x_shape) > 1:
        #     sum_index = -1
        #     for x_i, y_i in zip(x_shape, grad_shape):
        #         sum_index += 1
        #         if x_i != y_i:
        #             break
        #
        # z = summation(Tensor(out_grad), sum_index)
        # if len(x_shape) > 0:
        #     z = reshape(z, x_shape)
        #
        # return (z, )
        ### END YOUR SOLUTION
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        x_ori_shape = x.shape

        if len(x_ori_shape) > 0:
            grad_shape = out_grad.shape
            sum_index = list()

            diff = len(grad_shape) - len(x_ori_shape)
            x_shape = (None,) * diff + x_ori_shape

            for none_i, (x_i, y_i) in enumerate(zip(x_shape, grad_shape)):
                if x_i != y_i:
                    if none_i >= diff:
                        sum_index.append(none_i - diff)
                    else:
                        sum_index.append(none_i)

            for index in sum_index:
                out_grad = summation(out_grad, index)

            if len(x_shape) > 0:
                out_grad = reshape(out_grad, x_ori_shape)

        else:
            out_grad = summation(out_grad)

        return (out_grad, )
        ### END YOUR SOLUTION

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].data
        out_shape = out_grad.shape
        input_shape = [1] * len(a.shape)
        j = 0
        for i in range(len(a.shape)):
            if j < len(out_shape) and out_shape[j] == a.shape[i]:
                input_shape[i] = a.shape[i]
                j += 1
        return (broadcast_to(reshape(out_grad, tuple(input_shape)), a.shape), )
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)

class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs
        x_dim, y_dim = (len(x.shape), len(y.shape))
        x_grad = matmul(out_grad, transpose(y))
        y_grad = matmul(transpose(x), out_grad)
        x_grad_dim, y_grad_dim = (len(x_grad.shape), len(y_grad.shape))

        if x_dim != x_grad_dim:
            x_grad = summation(x_grad, axes=tuple(range(x_dim)))
        if y_dim != y_grad_dim:
            y_grad = summation(y_grad, axes=tuple(range(y_dim)))
        return (x_grad, y_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # x,  = node.inputs
        # return (multiply(out_grad, x / array_api.abs(x.numpy())), )
        return (mul_scalar(out_grad, -1), )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return (multiply(out_grad, power_scalar(x, -1)), )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return multiply(out_grad, exp(x)),
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        mask = (a <= 0)
        out = a.copy()
        out[mask] = 0
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.inputs[0].realize_cached_data() > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

