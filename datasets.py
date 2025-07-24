import os
import gzip
import tarfile
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from dezero.utils import get_file, cache_dir
from dezero.transforms import Compose, Flatten, ToFloat, Normalize

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# =============================================================================
# Dataset base class
# =============================================================================
class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform or (lambda x: x)
        self.target_transform = target_transform or (lambda x: x)
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        x = self.transform(self.data[index])
        t = self.target_transform(self.label[index]) if self.label is not None else None
        return x, t

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


# =============================================================================
# Toy dataset: Spiral
# =============================================================================
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_data * num_class
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int64)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            r = rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = [r * np.sin(theta), r * np.cos(theta)]
            t[ix] = j

    idx = np.random.permutation(data_size)
    x = torch.tensor(x[idx], device=device)
    t = torch.tensor(t[idx], device=device)
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


# =============================================================================
# MNIST dataset
# =============================================================================
class MNIST(Dataset):
    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'data': 'train-images-idx3-ubyte.gz', 'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'data': 't10k-images-idx3-ubyte.gz', 'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['data'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_data(self, path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        return torch.tensor(data, dtype=torch.float32, device=device)

    def _load_label(self, path):
        with gzip.open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return torch.tensor(labels, dtype=torch.int64, device=device)

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = torch.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                idx = np.random.randint(len(self.data))
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[idx].view(H, W)
        plt.imshow(img.cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {i: str(i) for i in range(10)}


# =============================================================================
# SinCurve dataset
# =============================================================================
class SinCurve(Dataset):
    def prepare(self):
        x = np.linspace(0, 2 * np.pi, 1000)
        noise = np.random.uniform(-0.05, 0.05, size=x.shape)
        y = np.sin(x) + noise if self.train else np.cos(x)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        self.data = y[:-1].unsqueeze(1)
        self.label = y[1:].unsqueeze(1)


# =============================================================================
# Shakespeare dataset
# =============================================================================
class Shakespear(Dataset):
    def prepare(self):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        path = get_file(url, 'shakespear.txt')
        with open(path, 'r') as f:
            text = f.read()
        chars = list(text)

        char_to_id = {ch: i for i, ch in enumerate(sorted(set(chars)))}
        id_to_char = {i: ch for ch, i in char_to_id.items()}
        indices = torch.tensor([char_to_id[c] for c in chars], dtype=torch.int64, device=device)

        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


# =============================================================================
# Cache utils
# =============================================================================
def load_cache_npz(filename, train=False):
    name = filename[filename.rfind('/')+1:]
    suffix = '.train.npz' if train else '.test.npz'
    path = os.path.join(cache_dir, name + suffix)
    if not os.path.exists(path):
        return None, None
    loaded = np.load(path)
    data = torch.tensor(loaded['data'], dtype=torch.float32, device=device)
    label = torch.tensor(loaded['label'], dtype=torch.int64, device=device)
    return data, label


def save_cache_npz(data, label, filename, train=False):
    name = filename[filename.rfind('/')+1:]
    suffix = '.train.npz' if train else '.test.npz'
    path = os.path.join(cache_dir, name + suffix)
    if os.path.exists(path):
        return
    print("Saving:", name + suffix)
    np.savez_compressed(path, data=data.cpu().numpy(), label=label.cpu().numpy())
    print("Done")
