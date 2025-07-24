import torch
from dezero import Variable

# Apple Silicon GPU (MPS) 우선 사용
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def get_array_module(x):
    """
    PyTorch에서는 불필요하지만, 호환성 위해 torch 반환.
    """
    return torch

def as_numpy(x):
    """
    PyTorch Tensor를 numpy ndarray로 변환
    """
    if isinstance(x, Variable):
        x = x.data
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x  # 이미 numpy인 경우

def as_torch(x):
    """
    numpy → torch.Tensor 변환 및 MPS 장치 이동
    """
    if isinstance(x, Variable):
        x = x.data
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)
