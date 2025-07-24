import math
from dezero import cuda
import torch

pil_available = True
try:
    from PIL import Image
except:
    pil_available = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = torch.randperm(len(self.dataset)).tolist()
        else:
            self.index = list(range(len(self.dataset)))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[idx] for idx in batch_index]

        x = torch.stack([torch.tensor(ex[0], dtype=torch.float32) for ex in batch]).to(device)
        t = torch.tensor([ex[1] for ex in batch], dtype=torch.int64).to(device)

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        x = torch.stack([torch.tensor(ex[0], dtype=torch.float32) for ex in batch]).to(device)
        t = torch.tensor([ex[1] for ex in batch], dtype=torch.int64).to(device)

        self.iteration += 1
        return x, t
