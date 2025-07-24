# dezero/backend.py

import torch

device = torch.device('cpu')  # 초기값 (init.py에서 mps로 설정될 것)

def get_array_module(x):
    return torch

def as_array(x):
    if isinstance(x, float):
        return torch.tensor(x, dtype=torch.float32, device=device)
    elif isinstance(x, int):
        return torch.tensor(float(x), dtype=torch.float32, device=device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)
