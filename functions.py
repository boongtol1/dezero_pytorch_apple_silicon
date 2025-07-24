# Modified functions.py for DeZero to use PyTorch MPS backend on Apple Silicon

import torch
import dezero
from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array
from dezero.utils import sum_to

# 수정: dezero.layers_torch 에서 가져오도록 변경
from dezero.layers import conv2d, max_pool2d, relu, reshape, matmul, mean_squared_error, get_conv_outsize


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        y = torch.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = torch.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = torch.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        y = torch.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        y = torch.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: sum / reshape / broadcast_to / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = torch.sum(x, dim=self.axis, keepdim=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        if isinstance(gy, Variable):
            return Variable(gy.data.expand(self.x_shape))
        else:
            return gy.expand(self.x_shape)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.expand(self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)

def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        return torch.matmul(x, W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.data.T)
        gW = matmul(x.data.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
        y = torch.matmul(x, W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.data.T)
        gW = matmul(x.data.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


# =============================================================================
# Activation functions
# =============================================================================
class ReLU(Function):
    def forward(self, x):
        return torch.clamp(x, min=0)

    def backward(self, gy):
        x, = self.inputs
        return gy * (x.data > 0).float()

def relu(x):
    return ReLU()(x)


class Sigmoid(Function):
    def forward(self, x):
        return torch.sigmoid(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)

def sigmoid(x):
    return Sigmoid()(x)

def leaky_relu(x, negative_slope=0.2):
    x_data = x.data if isinstance(x, Variable) else x
    return Variable(torch.nn.functional.leaky_relu(x_data, negative_slope=negative_slope))

def flatten(x):
    x_data = x.data if isinstance(x, Variable) else x
    return Variable(x_data.view(x_data.size(0), -1))

def tanh(x):
    x_data = x.data if isinstance(x, Variable) else x
    return Variable(torch.tanh(x_data))


# =============================================================================
# Loss functions
# =============================================================================
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)
