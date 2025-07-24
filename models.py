# models_torch.py (PyTorch backend for Apple Silicon GPU)
import torch
import torch.nn.functional as F
from dezero.layers import Layer, Linear, Conv2d, Deconv2d, BatchNorm
from dezero.utils import plot_dot_graph

# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)


class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=torch.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# =============================================================================
# Utility: Global Average Pooling
# =============================================================================
def _global_average_pooling_2d(x):
    return torch.mean(x, dim=(2, 3))


# =============================================================================
# ResNet Blocks (simplified for compatibility)
# =============================================================================
class BottleneckA(Layer):
    def __init__(self, in_channels, mid_channels, out_channels, stride=2):
        super().__init__()
        self.conv1 = Conv2d(mid_channels, kernel_size=1, stride=1, pad=0, in_channels=in_channels)
        self.bn1 = BatchNorm()
        self.conv2 = Conv2d(mid_channels, kernel_size=3, stride=stride, pad=1, in_channels=mid_channels)
        self.bn2 = BatchNorm()
        self.conv3 = Conv2d(out_channels, kernel_size=1, stride=1, pad=0, in_channels=mid_channels)
        self.bn3 = BatchNorm()
        self.shortcut = Conv2d(out_channels, kernel_size=1, stride=stride, pad=0, in_channels=in_channels)
        self.bn_sc = BatchNorm()

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        sc = self.bn_sc(self.shortcut(x))
        return torch.relu(h + sc)


class BottleneckB(Layer):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = Conv2d(mid_channels, kernel_size=1, stride=1, pad=0, in_channels=in_channels)
        self.bn1 = BatchNorm()
        self.conv2 = Conv2d(mid_channels, kernel_size=3, stride=1, pad=1, in_channels=mid_channels)
        self.bn2 = BatchNorm()
        self.conv3 = Conv2d(in_channels, kernel_size=1, stride=1, pad=0, in_channels=mid_channels)
        self.bn3 = BatchNorm()

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return torch.relu(h + x)


class BuildingBlock(Layer):
    def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        self.layers = [BottleneckA(in_channels, mid_channels, out_channels, stride)]
        for _ in range(n_layers - 1):
            self.layers.append(BottleneckB(out_channels, mid_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# ResNet
# =============================================================================
class ResNet(Model):
    def __init__(self, n_layers=50):
        super().__init__()
        if n_layers == 50:
            blocks = [3, 4, 6, 3]
        elif n_layers == 101:
            blocks = [3, 4, 23, 3]
        elif n_layers == 152:
            blocks = [3, 8, 36, 3]
        else:
            raise ValueError('Unsupported ResNet depth.')

        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=3)
        self.bn1 = BatchNorm()
        self.res2 = BuildingBlock(blocks[0], 64, 64, 256, 1)
        self.res3 = BuildingBlock(blocks[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(blocks[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(blocks[3], 1024, 512, 2048, 2)
        self.fc6 = Linear(1000, in_size=2048)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = _global_average_pooling_2d(x)
        return self.fc6(x)


class ResNet50(ResNet):
    def __init__(self):
        super().__init__(n_layers=50)


class ResNet101(ResNet):
    def __init__(self):
        super().__init__(n_layers=101)


class ResNet152(ResNet):
    def __init__(self):
        super().__init__(n_layers=152)


