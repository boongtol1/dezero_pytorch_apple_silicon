# optimizers_torch.py (PyTorch backend)
import math
import torch


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad += self.rate * param.data


class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += torch.sum(param.grad ** 2)
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad *= rate


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = torch.zeros_like(param.data)

        v = self.vs[v_key]
        v.mul_(self.momentum).sub_(self.lr * param.grad)
        param.data.add_(v)


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = torch.zeros_like(param.data)

        grad = param.grad
        h = self.hs[h_key]
        h.add_(grad ** 2)
        param.data -= self.lr * grad / (torch.sqrt(h) + self.eps)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        key = id(param)
        if key not in self.msg:
            self.msg[key] = torch.zeros_like(param.data)
            self.msdx[key] = torch.zeros_like(param.data)

        msg = self.msg[key]
        msdx = self.msdx[key]
        grad = param.grad

        msg.mul_(self.rho).add_((1 - self.rho) * grad ** 2)
        dx = torch.sqrt((msdx + self.eps) / (msg + self.eps)) * grad
        msdx.mul_(self.rho).add_((1 - self.rho) * dx ** 2)
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self):
        self.t += 1
        super().update()

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = torch.zeros_like(param.data)
            self.vs[key] = torch.zeros_like(param.data)

        m = self.ms[key]
        v = self.vs[key]
        grad = param.grad

        m.mul_(self.beta1).add_((1 - self.beta1) * grad)
        v.mul_(self.beta2).add_((1 - self.beta2) * (grad ** 2))
        param.data -= self.lr * m / (torch.sqrt(v) + self.eps)