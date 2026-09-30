# tests/test_autodiff.py [FULLY PROVIDED TO STUDENTS]
import numpy as np
import pytest
from autodiff.tensor import Tensor


def numerical_grad(f, x_np, eps=1e-5, method="central"):
    """
    Numerical gradient of scalar function f w.r.t. flat array x_np.

    Parameters
    ----------
    f      : callable that takes a numpy array and returns a scalar float
    x_np   : numpy array, the point at which the gradient is evaluated
    eps    : finite difference step size
    method : one of
             "forward"  — f(x+h) - f(x)          / h     [O(h)   error]
             "backward" — f(x)   - f(x-h)         / h     [O(h)   error]
             "central"  — f(x+h) - f(x-h)         / 2h    [O(h²)  error]
    """
    assert method in ("forward", "backward", "central"), \
        f"Unknown method '{method}'. Choose from 'forward', 'backward', 'central'."

    grad   = np.zeros_like(x_np, dtype=np.float64)
    f_x    = f(x_np) if method in ("forward", "backward") else None

    for i in range(x_np.size):
        xp = x_np.copy()
        xm = x_np.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps

        if method == "forward":
            grad.flat[i] = (f(xp) - f_x) / eps
        elif method == "backward":
            grad.flat[i] = (f_x - f(xm)) / eps
        elif method == "central":
            grad.flat[i] = (f(xp) - f(xm)) / (2 * eps)

    return grad


def check_and_log(op_name, auto_g, num_g, method, rtol=1e-4, atol=1e-4):
    """Helper to log the expected vs actual gradients and assert they match."""
    print(f"\n[{op_name}] Checking gradients (method='{method}')...")
    print(f"  Expected (Numerical / {method:8s}): {num_g.flatten()}")
    print(f"  Actual   (Autodiff):               {auto_g.flatten()}")
    diff = np.abs(auto_g.flatten() - num_g.flatten())
    print(f"  Max Absolute Diff:                  {diff.max():.2e}")

    try:
        np.testing.assert_allclose(auto_g, num_g, rtol=rtol, atol=atol)
        print(f"  ✅ PASS: {op_name}")
    except AssertionError as e:
        print(f"  ❌ FAIL: {op_name}")
        raise e


# ------------------------------------------------------------------
# We run every test for all three methods to help students understand
# the accuracy difference between forward, backward and central.
# ------------------------------------------------------------------
@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_add_backward(method):
    x_np = np.random.randn(3)
    y_np = np.random.randn(3)

    x = Tensor(x_np.copy(), requires_grad=True)
    loss = (x + Tensor(y_np)).sum()
    loss.backward()

    num_g = numerical_grad(lambda xv: (Tensor(xv) + Tensor(y_np)).sum().data,
                           x_np, method=method)
    check_and_log("Addition", x.grad, num_g, method)


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_mul_backward(method):
    x_np = np.random.randn(4)
    y_np = np.random.randn(4)

    x = Tensor(x_np.copy(), requires_grad=True)
    loss = (x * Tensor(y_np)).sum()
    loss.backward()

    num_g = numerical_grad(lambda xv: (Tensor(xv) * Tensor(y_np)).sum().data,
                           x_np, method=method)
    check_and_log("Multiplication", x.grad, num_g, method)


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_matmul_backward(method):
    A_np = np.random.randn(4, 3)
    B_np = np.random.randn(3, 2)

    A = Tensor(A_np.copy(), requires_grad=True)
    loss = (A @ Tensor(B_np)).sum()
    loss.backward()

    num_g = numerical_grad(
        lambda av: (Tensor(av.reshape(4, 3)) @ Tensor(B_np)).sum().data,
        A_np.flatten(), method=method)
    check_and_log("Matrix Multiplication", A.grad.flatten(), num_g, method)


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_relu_backward(method):
    x_np = np.random.randn(5)

    x = Tensor(x_np.copy(), requires_grad=True)
    loss = x.relu().sum()
    loss.backward()

    num_g = numerical_grad(lambda xv: Tensor(xv).relu().sum().data,
                           x_np, method=method)
    check_and_log("ReLU", x.grad, num_g, method)


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_tanh_backward(method):
    x_np = np.random.randn(5)

    x = Tensor(x_np.copy(), requires_grad=True)
    loss = x.tanh().sum()
    loss.backward()

    num_g = numerical_grad(lambda xv: Tensor(xv).tanh().sum().data,
                           x_np, method=method)
    check_and_log("Tanh", x.grad, num_g, method)


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_chain_rule_deep(method):
    """Tests that the backward pass correctly chains through 3 operations."""
    x_np = np.random.randn(3)

    x = Tensor(x_np.copy(), requires_grad=True)
    loss = ((x * x).relu() + x).sum()
    loss.backward()

    num_g = numerical_grad(
        lambda xv: ((Tensor(xv) * Tensor(xv)).relu() + Tensor(xv)).sum().data,
        x_np, method=method)
    check_and_log("Deep Chain Rule", x.grad, num_g, method)