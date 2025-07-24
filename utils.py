# layers_torch.py (for Apple Silicon + PyTorch backend)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import dezero.core


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
        return self.linear(x)


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Deconv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.deconv(x)


class BatchNorm(Layer):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


# Dummy function to replace plot_dot_graph to prevent import errors in Apple Silicon PyTorch environment

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    print("plot_dot_graph is not supported in torch-only backend. Skipped.")


# Dummy get_file and cache_dir for compatibility with datasets.py

def get_file(url, file_name=None):
    print("get_file is not supported in torch-only backend. Skipped.")
    return ""

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """역전파 시 sum의 gradient 모양을 원래대로 되돌리기 위한 함수"""
    ndim = len(x_shape)
    tupled_axis = axis if isinstance(axis, tuple) else (axis,) if axis is not None else None

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
        gy = gy.reshape(*shape)
    return gy

def sum_to(x, shape):
    if isinstance(x, dezero.core.Variable):
        x_data = x.data
    else:
        x_data = x

    ndim = len(shape)
    lead = x_data.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    
    y = x_data.sum(lead_axis + axis, keepdim=True)

    if lead > 0:
        y = y.squeeze(tuple(range(lead)))

    return y
