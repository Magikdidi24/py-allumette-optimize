"""
tests/test_torch_compat.py
==========================
PyTorch compatibility tests for the student autodiff / ml / optim stack.

Each test:
  1. Runs the SAME computation with the student Tensor API.
  2. Runs the SAME computation with torch (CPU, float64).
  3. Asserts that forward values AND gradients agree to tight tolerances.

This validates that the student implementation is a faithful re-implementation
of a real-world autodiff engine, not just a toy that passes numerical checks.

Requirements (already installed by the CI):
    torch (CPU wheel, float64)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from autodiff.tensor import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RTOL = 1e-5
ATOL = 1e-5


def _t(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    """Wrap a numpy array into a torch float64 tensor."""
    return torch.tensor(arr, dtype=torch.float64, requires_grad=requires_grad)


def check(name: str, student_val, torch_val,
          student_grad, torch_grad, rtol=RTOL, atol=ATOL):
    """Assert forward value and gradient match between student and torch."""
    print(f"\n[{name}]")
    print(f"  student fwd : {np.array(student_val).flatten()[:6]}")
    print(f"  torch   fwd : {torch_val.detach().numpy().flatten()[:6]}")
    print(f"  student grad: {np.array(student_grad).flatten()[:6]}")
    print(f"  torch   grad: {torch_grad.numpy().flatten()[:6]}")

    np.testing.assert_allclose(
        np.array(student_val), torch_val.detach().numpy(),
        rtol=rtol, atol=atol,
        err_msg=f"{name}: forward value mismatch"
    )
    np.testing.assert_allclose(
        np.array(student_grad), torch_grad.numpy(),
        rtol=rtol, atol=atol,
        err_msg=f"{name}: gradient mismatch"
    )
    print(f"  ✅ PASS: {name}")


# ===========================================================================
# 1. ELEMENT-WISE OPS
# ===========================================================================

def test_add_scalar_torch():
    x_np = np.random.randn(4).astype(np.float64)

    # Student
    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = (x_s + 3.0).sum()
    loss_s.backward()

    # Torch
    x_t = _t(x_np, requires_grad=True)
    loss_t = (x_t + 3.0).sum()
    loss_t.backward()

    check("add_scalar", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_sub_backward_torch():
    x_np = np.random.randn(5).astype(np.float64)
    y_np = np.random.randn(5).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = (x_s - Tensor(y_np)).sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = (x_t - _t(y_np)).sum()
    loss_t.backward()

    check("subtraction", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_mul_backward_torch():
    x_np = np.random.randn(6).astype(np.float64)
    y_np = np.random.randn(6).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = (x_s * Tensor(y_np)).sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = (x_t * _t(y_np)).sum()
    loss_t.backward()

    check("multiplication", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_div_backward_torch():
    x_np = np.abs(np.random.randn(5).astype(np.float64)) + 0.5  # avoid /0
    y_np = np.abs(np.random.randn(5).astype(np.float64)) + 0.5

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = (x_s / Tensor(y_np)).sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = (x_t / _t(y_np)).sum()
    loss_t.backward()

    check("division", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_pow_backward_torch():
    x_np = np.abs(np.random.randn(4).astype(np.float64)) + 0.1

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = (x_s ** 3).sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = (x_t ** 3).sum()
    loss_t.backward()

    check("power", loss_s.data, loss_t, x_s.grad, x_t.grad)


# ===========================================================================
# 2. REDUCTIONS
# ===========================================================================

def test_mean_backward_torch():
    x_np = np.random.randn(3, 4).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = x_s.mean()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = x_t.mean()
    loss_t.backward()

    check("mean", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_sum_backward_torch():
    x_np = np.random.randn(3, 4).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = x_s.sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = x_t.sum()
    loss_t.backward()

    check("sum", loss_s.data, loss_t, x_s.grad, x_t.grad)


# ===========================================================================
# 3. MATRIX OPS
# ===========================================================================

def test_matmul_backward_torch():
    A_np = np.random.randn(4, 3).astype(np.float64)
    B_np = np.random.randn(3, 2).astype(np.float64)

    A_s = Tensor(A_np.copy(), requires_grad=True)
    loss_s = (A_s @ Tensor(B_np)).sum()
    loss_s.backward()

    A_t = _t(A_np, requires_grad=True)
    loss_t = (A_t @ _t(B_np)).sum()
    loss_t.backward()

    check("matmul wrt A", loss_s.data, loss_t, A_s.grad, A_t.grad)


def test_matmul_both_grads_torch():
    """Both operands require grad — exercises the product rule of matmul."""
    A_np = np.random.randn(3, 4).astype(np.float64)
    B_np = np.random.randn(4, 2).astype(np.float64)

    A_s = Tensor(A_np.copy(), requires_grad=True)
    B_s = Tensor(B_np.copy(), requires_grad=True)
    loss_s = (A_s @ B_s).sum()
    loss_s.backward()

    A_t = _t(A_np, requires_grad=True)
    B_t = _t(B_np, requires_grad=True)
    loss_t = (A_t @ B_t).sum()
    loss_t.backward()

    np.testing.assert_allclose(A_s.grad, A_t.grad.numpy(), rtol=RTOL, atol=ATOL,
                                err_msg="matmul: grad wrt A mismatch")
    np.testing.assert_allclose(B_s.grad, B_t.grad.numpy(), rtol=RTOL, atol=ATOL,
                                err_msg="matmul: grad wrt B mismatch")
    print("\n  ✅ PASS: matmul both grads")


# ===========================================================================
# 4. ACTIVATION FUNCTIONS
# ===========================================================================

def test_relu_backward_torch():
    x_np = np.random.randn(8).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = x_s.relu().sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = torch.relu(x_t).sum()
    loss_t.backward()

    check("relu", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_tanh_backward_torch():
    x_np = np.random.randn(8).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = x_s.tanh().sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = torch.tanh(x_t).sum()
    loss_t.backward()

    check("tanh", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_sigmoid_backward_torch():
    """sigmoid(x) = 1 / (1 + exp(-x)); often used in binary classification."""
    x_np = np.random.randn(6).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    # expressed via existing ops: 1 / (1 + exp(-x))
    # students must implement __neg__ and exp, or a dedicated sigmoid
    one = Tensor(np.ones_like(x_np))
    loss_s = (one / (one + (x_s * Tensor(np.full_like(x_np, -1.0))).exp())).sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = torch.sigmoid(x_t).sum()
    loss_t.backward()

    check("sigmoid (via ops)", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_exp_backward_torch():
    x_np = (np.random.randn(5) * 0.5).astype(np.float64)  # keep small to avoid overflow

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = x_s.exp().sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = torch.exp(x_t).sum()
    loss_t.backward()

    check("exp", loss_s.data, loss_t, x_s.grad, x_t.grad)


def test_log_backward_torch():
    x_np = np.abs(np.random.randn(5).astype(np.float64)) + 0.1

    x_s = Tensor(x_np.copy(), requires_grad=True)
    loss_s = x_s.log().sum()
    loss_s.backward()

    x_t = _t(x_np, requires_grad=True)
    loss_t = torch.log(x_t).sum()
    loss_t.backward()

    check("log", loss_s.data, loss_t, x_s.grad, x_t.grad)


# ===========================================================================
# 5. BROADCASTING
# ===========================================================================

def test_broadcast_add_torch():
    """(N, M) + (M,) — broadcast over first axis, like adding a bias."""
    A_np = np.random.randn(4, 3).astype(np.float64)
    b_np = np.random.randn(3).astype(np.float64)

    A_s = Tensor(A_np.copy(), requires_grad=True)
    b_s = Tensor(b_np.copy(), requires_grad=True)
    loss_s = (A_s + b_s).sum()
    loss_s.backward()

    A_t = _t(A_np, requires_grad=True)
    b_t = _t(b_np, requires_grad=True)
    loss_t = (A_t + b_t).sum()
    loss_t.backward()

    np.testing.assert_allclose(A_s.grad, A_t.grad.numpy(), rtol=RTOL, atol=ATOL,
                                err_msg="broadcast add: grad wrt A")
    np.testing.assert_allclose(b_s.grad, b_t.grad.numpy(), rtol=RTOL, atol=ATOL,
                                err_msg="broadcast add: grad wrt b (sum over batch)")
    print("\n  ✅ PASS: broadcast add")


def test_broadcast_mul_torch():
    """(N, M) * (M,) broadcast — mimics per-feature scaling."""
    A_np = np.random.randn(5, 4).astype(np.float64)
    s_np = np.random.randn(4).astype(np.float64)

    A_s = Tensor(A_np.copy(), requires_grad=True)
    s_s = Tensor(s_np.copy(), requires_grad=True)
    loss_s = (A_s * s_s).sum()
    loss_s.backward()

    A_t = _t(A_np, requires_grad=True)
    s_t = _t(s_np, requires_grad=True)
    loss_t = (A_t * s_t).sum()
    loss_t.backward()

    np.testing.assert_allclose(A_s.grad, A_t.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s_s.grad, s_t.grad.numpy(), rtol=RTOL, atol=ATOL)
    print("\n  ✅ PASS: broadcast mul")


# ===========================================================================
# 6. LINEAR LAYER — forward output matches torch.nn.Linear
# ===========================================================================

def test_linear_layer_forward_torch():
    """
    Given the SAME weights and bias, the student Linear layer must produce
    the SAME output as torch.nn.Linear (without training — only forward).
    """
    from ml.nn import Linear

    np.random.seed(0)
    in_f, out_f, N = 4, 3, 8
    W_np = np.random.randn(in_f, out_f).astype(np.float64)
    b_np = np.random.randn(out_f).astype(np.float64)
    X_np = np.random.randn(N, in_f).astype(np.float64)

    # Student
    layer = Linear(in_f, out_f)
    layer.weights = Tensor(W_np.copy())
    layer.bias    = Tensor(b_np.copy())
    out_s = layer.forward(Tensor(X_np)).data

    # Torch
    fc = torch.nn.Linear(in_f, out_f, dtype=torch.float64)
    with torch.no_grad():
        fc.weight.copy_(_t(W_np.T))   # torch stores W as (out, in)
        fc.bias.copy_(_t(b_np))
    out_t = fc(_t(X_np)).detach().numpy()

    np.testing.assert_allclose(out_s, out_t, rtol=1e-4, atol=1e-4,
                                err_msg="Linear layer forward mismatch vs torch.nn.Linear")
    print("\n  ✅ PASS: Linear layer forward matches torch.nn.Linear")


# ===========================================================================
# 7. MSE LOSS — matches torch.nn.MSELoss(reduction='mean')
# ===========================================================================

def test_mse_loss_value_and_grad_torch():
    np.random.seed(7)
    pred_np = np.random.randn(6, 3).astype(np.float64)
    tgt_np  = np.random.randn(6, 3).astype(np.float64)

    # Student
    pred_s = Tensor(pred_np.copy(), requires_grad=True)
    diff_s = pred_s - Tensor(tgt_np)
    loss_s = (diff_s * diff_s).mean()
    loss_s.backward()

    # Torch
    pred_t = _t(pred_np, requires_grad=True)
    loss_t = torch.nn.MSELoss()(pred_t, _t(tgt_np))
    loss_t.backward()

    np.testing.assert_allclose(loss_s.data, loss_t.item(), rtol=RTOL, atol=ATOL,
                                err_msg="MSE loss value mismatch")
    np.testing.assert_allclose(pred_s.grad, pred_t.grad.numpy(), rtol=RTOL, atol=ATOL,
                                err_msg="MSE loss gradient mismatch")
    print("\n  ✅ PASS: MSE loss value and gradient match torch.nn.MSELoss")


# ===========================================================================
# 8. SGD OPTIMIZER — one step matches torch.optim.SGD (no momentum)
# ===========================================================================

def test_sgd_step_torch():
    """
    After one SGD step, the student parameter must equal the torch parameter.
    Verifies: param <- param - lr * grad
    """
    from ml.optim import SGD

    np.random.seed(3)
    lr = 0.05
    W_np = np.random.randn(3, 2).astype(np.float64)
    X_np = np.random.randn(4, 3).astype(np.float64)
    Y_np = np.random.randn(4, 2).astype(np.float64)

    # Student
    from ml.nn import Linear
    layer_s = Linear(3, 2)
    layer_s.weights = Tensor(W_np.copy(), requires_grad=True)
    layer_s.bias    = Tensor(np.zeros(2), requires_grad=True)
    opt_s = SGD(layer_s.parameters(), lr=lr)

    opt_s.zero_grad()
    diff_s = layer_s.forward(Tensor(X_np)) - Tensor(Y_np)
    loss_s = (diff_s * diff_s).mean()
    loss_s.backward()
    opt_s.step()

    W_student_after = layer_s.weights.data.copy()

    # Torch
    fc = torch.nn.Linear(3, 2, dtype=torch.float64)
    with torch.no_grad():
        fc.weight.copy_(_t(W_np.T))
        fc.bias.zero_()
    opt_t = torch.optim.SGD(fc.parameters(), lr=lr)

    opt_t.zero_grad()
    loss_t = torch.nn.MSELoss()(_t(X_np) @ _t(W_np) + _t(np.zeros(2)), _t(Y_np))
    # recompute through fc for correct grads
    loss_t2 = torch.nn.MSELoss()(fc(_t(X_np)), _t(Y_np))
    loss_t2.backward()
    opt_t.step()

    W_torch_after = fc.weight.detach().numpy().T  # transpose back to (in, out)

    np.testing.assert_allclose(W_student_after, W_torch_after, rtol=1e-3, atol=1e-3,
                                err_msg="SGD: weight after one step mismatch")
    print("\n  ✅ PASS: SGD step matches torch.optim.SGD")


# ===========================================================================
# 9. GRADIENT ACCUMULATION — calling backward() twice accumulates grads
# ===========================================================================

def test_grad_accumulation_torch():
    """
    In PyTorch (and standard autodiff), calling backward() without zeroing
    grads accumulates them. This is used in gradient accumulation training.
    """
    x_np = np.random.randn(4).astype(np.float64)

    x_s = Tensor(x_np.copy(), requires_grad=True)
    for _ in range(2):
        loss = (x_s * Tensor(x_np)).sum()
        loss.backward()

    x_t = _t(x_np, requires_grad=True)
    for _ in range(2):
        loss_t = (x_t * _t(x_np)).sum()
        loss_t.backward()

    np.testing.assert_allclose(x_s.grad, x_t.grad.numpy(), rtol=RTOL, atol=ATOL,
                                err_msg="Gradient accumulation mismatch")
    print("\n  ✅ PASS: gradient accumulation matches torch")


# ===========================================================================
# 10. DEEP COMPUTATIONAL GRAPH — MLP-like chain
# ===========================================================================

def test_deep_graph_mlp_like_torch():
    """
    Three Linear(matmul + add) + ReLU layers, the same architecture as MLP.
    Checks that gradients flow correctly through a deep graph, matching torch.
    """
    np.random.seed(42)
    N, d0, d1, d2, d3 = 5, 8, 16, 16, 4
    W1 = np.random.randn(d0, d1).astype(np.float64) * 0.1
    b1 = np.zeros(d1, dtype=np.float64)
    W2 = np.random.randn(d1, d2).astype(np.float64) * 0.1
    b2 = np.zeros(d2, dtype=np.float64)
    W3 = np.random.randn(d2, d3).astype(np.float64) * 0.1
    b3 = np.zeros(d3, dtype=np.float64)
    X  = np.random.randn(N, d0).astype(np.float64)
    Y  = np.random.randn(N, d3).astype(np.float64)

    # --- Student ---
    W1_s = Tensor(W1.copy(), requires_grad=True)
    W2_s = Tensor(W2.copy(), requires_grad=True)
    W3_s = Tensor(W3.copy(), requires_grad=True)

    h1 = (Tensor(X) @ W1_s + Tensor(b1)).relu()
    h2 = (h1 @ W2_s + Tensor(b2)).relu()
    out = h2 @ W3_s + Tensor(b3)
    diff = out - Tensor(Y)
    loss_s = (diff * diff).mean()
    loss_s.backward()

    # --- Torch ---
    W1_t = _t(W1, requires_grad=True)
    W2_t = _t(W2, requires_grad=True)
    W3_t = _t(W3, requires_grad=True)

    h1_t = torch.relu(_t(X) @ W1_t + _t(b1))
    h2_t = torch.relu(h1_t @ W2_t + _t(b2))
    out_t = h2_t @ W3_t + _t(b3)
    loss_t = torch.nn.MSELoss()(out_t, _t(Y))
    loss_t.backward()

    for name, gs, gt in [("W1", W1_s.grad, W1_t.grad.numpy()),
                          ("W2", W2_s.grad, W2_t.grad.numpy()),
                          ("W3", W3_s.grad, W3_t.grad.numpy())]:
        np.testing.assert_allclose(gs, gt, rtol=1e-4, atol=1e-4,
                                    err_msg=f"Deep graph: {name} gradient mismatch")
    print("\n  ✅ PASS: deep MLP-like graph gradients match torch")
