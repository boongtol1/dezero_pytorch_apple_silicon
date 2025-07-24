# transforms_torch.py (PyTorch backend for Apple Silicon)
import torch
import numpy as np
from PIL import Image
from dezero.utils import pair


class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'BGR':
            img = img.convert('RGB')
            r, g, b = img.split()
            return Image.merge('RGB', (b, g, r))
        return img.convert(self.mode)


class Resize:
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = left + OW
        top = (H - OH) // 2
        bottom = top + OH
        return img.crop((left, top, right, bottom))


class ToArray:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img
        if isinstance(img, Image.Image):
            arr = np.asarray(img).transpose(2, 0, 1)
            return torch.tensor(arr, dtype=self.dtype)
        raise TypeError("Unsupported input type")


class ToPIL:
    def __call__(self, tensor):
        if isinstance(tensor, torch.Tensor):
            array = tensor.numpy().transpose(1, 2, 0).astype(np.uint8)
            return Image.fromarray(array)
        raise TypeError("Input should be a torch.Tensor")


class Normalize:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device)

        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]

        return (tensor - mean) / std


class Flatten:
    def __call__(self, tensor):
        return tensor.view(-1)


class AsType:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=torch.int32):
        super().__init__(dtype)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if isinstance(img, Image.Image):
            if torch.rand(1).item() < self.p:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            return img
        elif isinstance(img, torch.Tensor):
            if torch.rand(1).item() < self.p:
                return img.flip(-1)
            return img
        raise TypeError("Unsupported type for flip")
