import numpy as np


def unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)

    # ------------------------------------------------------------------
    # Backward pass: iterative topological sort (avoids recursion limits
    # on the deep graphs produced by long MPC rollouts), then reverse
    # traversal calling each node's local _backward closure.
    # ------------------------------------------------------------------
    def backward(self):
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        # Seed: dL/dL = 1 (ones for non-scalar outputs, torch-like sum semantics)
        self.grad = self.grad + np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _wrap(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))

        def _backward():
            G = out.grad
            a, b = self.data, other.data
            if self.requires_grad:
                if a.ndim == 1 and b.ndim == 1:
                    grad_a = G * b
                elif a.ndim == 1:
                    grad_a = b @ G                      # (n,m) @ (m,) -> (n,)
                elif b.ndim == 1:
                    grad_a = np.outer(G, b)             # (m,) x (n,) -> (m,n)
                else:
                    grad_a = G @ np.swapaxes(b, -1, -2)
                    grad_a = unbroadcast(grad_a, a.shape)
                self.grad += grad_a.reshape(a.shape)
            if other.requires_grad:
                if a.ndim == 1 and b.ndim == 1:
                    grad_b = G * a
                elif a.ndim == 1:
                    grad_b = np.outer(a, G)             # (n,) x (m,) -> (n,m)
                elif b.ndim == 1:
                    grad_b = a.T @ G                    # (n,m) @ (m,)? -> (n,)
                else:
                    grad_b = np.swapaxes(a, -1, -2) @ G
                    grad_b = unbroadcast(grad_b, b.shape)
                other.grad += grad_b.reshape(b.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return (self * -1) + other

    def __neg__(self):
        return self * -1

    def __pow__(self, exp):
        out = Tensor(self.data ** exp,
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += (exp * self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(a % self.data.ndim for a in axes)
                for a in sorted(axes):
                    g = np.expand_dims(g, a)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sin(self):
        out = Tensor(np.sin(self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += np.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def cos(self):
        out = Tensor(np.cos(self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += -np.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** -1.0)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx],
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                grad_full = np.zeros_like(self.data)
                np.add.at(grad_full, idx, out.grad)
                self.grad += grad_full

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    @staticmethod
    def stack(tensors, axis=0):
        tensors = [Tensor._wrap(t) for t in tensors]
        out = Tensor(np.stack([t.data for t in tensors], axis=axis),
                     requires_grad=any(t.requires_grad for t in tensors),
                     _children=tuple(tensors))

        def _backward():
            for i, t in enumerate(tensors):
                if t.requires_grad:
                    t.grad += np.take(out.grad, i, axis=axis)

        out._backward = _backward
        return out

    def __abs__(self):
        out = Tensor(np.abs(self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def arctan2(self, other):
        other = Tensor._wrap(other)
        out = Tensor(np.arctan2(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))

        def _backward():
            denom = self.data ** 2 + other.data ** 2
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.data / denom,
                                         self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(-out.grad * self.data / denom,
                                          other.data.shape)

        out._backward = _backward
        return out

    def clip(self, a_min, a_max):
        out = Tensor(np.clip(self.data, a_min, a_max),
                     requires_grad=self.requires_grad,
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                mask = np.ones_like(self.data)
                if a_min is not None:
                    mask = mask * (self.data >= a_min)
                if a_max is not None:
                    mask = mask * (self.data <= a_max)
                self.grad += mask * out.grad

        out._backward = _backward
        return out
