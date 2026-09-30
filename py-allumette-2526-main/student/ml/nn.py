import numpy as np
from autodiff.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features):
        # He initialization (good default for ReLU networks); weights are
        # stored as (in_features, out_features) so forward is x @ W + b.
        scale = np.sqrt(2.0 / in_features)
        self.weights = Tensor(np.random.randn(in_features, out_features) * scale,
                              requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights + self.bias

    def parameters(self):
        return [self.weights, self.bias]


class MLP:
    def __init__(self, in_dim=8, hidden_dim=64, out_dim=6):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.fc1.forward(x).relu()
        h2 = self.fc2.forward(h1).relu()
        return self.fc3.forward(h2)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()
