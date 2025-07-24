# Apple Silicon GPU (MPS) 지원용 dezero/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from dezero.core import Variable

# Conv2D wrapper
def conv2d(x, weight, bias=None, stride=1, padding=0):
    x_data = x.data if isinstance(x, Variable) else x
    w_data = weight.data if isinstance(weight, Variable) else weight
    b_data = bias.data if (bias is not None and isinstance(bias, Variable)) else bias

    y = torch_F.conv2d(
        x_data, w_data, b_data,
        stride=(stride, stride) if isinstance(stride, int) else stride,
        padding=(padding, padding) if isinstance(padding, int) else padding
    )
    return Variable(y)

# MaxPool2D wrapper
def max_pool2d(x, kernel_size, stride=None, padding=0):
    x_data = x.data if isinstance(x, Variable) else x

    y = torch_F.max_pool2d(
        x_data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    return Variable(y)

# ReLU wrapper
def relu(x):
    x_data = x.data if isinstance(x, Variable) else x
    return Variable(torch_F.relu(x_data))

# Reshape wrapper
def reshape(x, shape):
    x_data = x.data if isinstance(x, Variable) else x
    return Variable(x_data.reshape(shape))

# MatMul wrapper
def matmul(x, W):
    x_data = x.data if isinstance(x, Variable) else x
    W_data = W.data if isinstance(W, Variable) else W
    return Variable(torch.matmul(x_data, W_data))

# Mean Squared Error wrapper
def mean_squared_error(x0, x1):
    x0_data = x0.data if isinstance(x0, Variable) else x0
    x1_data = x1.data if isinstance(x1, Variable) else x1
    diff = x0_data - x1_data
    return Variable(torch.mean(diff ** 2))

# Utility for conv size
def get_conv_outsize(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1

# Optional: Layer classes for modular modeling
class Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        raise NotImplementedError()


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x.data if isinstance(x, Variable) else x)


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x.data if isinstance(x, Variable) else x)


class Deconv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.deconv(x.data if isinstance(x, Variable) else x)


class BatchNorm(Layer):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x.data if isinstance(x, Variable) else x)