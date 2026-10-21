import numpy as np


class SGD:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.requires_grad:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad:
                p.grad = np.zeros_like(p.data)


class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if not p.requires_grad:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad:
                p.grad = np.zeros_like(p.data)
