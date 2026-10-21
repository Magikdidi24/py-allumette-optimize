import numpy as np
import pytest
from autodiff.tensor import Tensor
from ml.nn import Linear, MLP
from ml.optim import SGD

def test_linear_regression():
    # We create a simple random dataset for linear regression
    # y = X @ w_true + b_true
    np.random.seed(42)
    in_features = 2
    out_features = 1
    N = 100
    
    w_true = np.array([[1.5], [-2.0]])
    b_true = np.array([0.5])
    
    X_np = np.random.randn(N, in_features)
    Y_np = X_np @ w_true + b_true
    
    # create a linear model via MLP modules
    model = Linear(in_features, out_features)
    
    # Check that initial loss is something
    X_t = Tensor(X_np)
    Y_t = Tensor(Y_np)
    
    # Train for a few epochs
    lr = 0.05
    for epoch in range(100):
        # forward
        preds = model.forward(X_t)
        # MSE loss
        diff = preds - Y_t
        loss = (diff * diff).mean()
        
        # zero grad
        for p in model.parameters():
            p.zero_grad()
            
        loss.backward()
        
        # Optimizer step
        for p in model.parameters():
            p.data -= lr * p.grad
            
    # Check that the weights have approximated the true weights
    w_learned = model.weights.data
    b_learned = model.bias.data
    
    print("\n[Linear Regression] Checking weights...")
    print(f"  w_true: {w_true.flatten()}, w_learned: {w_learned.flatten()}")
    print(f"  b_true: {b_true}, b_learned: {b_learned}")
    
    np.testing.assert_allclose(w_learned, w_true, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(b_learned, b_true, rtol=1e-1, atol=1e-1)

def test_mlp_forward_backward():
    np.random.seed(42)
    N = 5
    in_dim = 8
    
    X_np = np.random.randn(N, in_dim)
    Y_np = np.random.randn(N, 6) # Target output
    
    X_t = Tensor(X_np)
    Y_t = Tensor(Y_np)
    
    model = MLP(in_dim=in_dim, hidden_dim=16, out_dim=6)
    
    preds = model.forward(X_t)
    diff = preds - Y_t
    loss = (diff * diff).mean()
    
    loss.backward()
    
    # check that gradients flow at least
    for p in model.parameters():
        assert p.grad is not None
        assert np.any(np.abs(p.grad) > 1e-6)

def test_mlp_linear_approximation():
    """
    Test a 3-layer MLP on approximating a linear line and evaluate using R^2.
    The MLP class already contains 3 layers (Linear -> ReLU -> Linear -> ReLU -> Linear).
    """
    np.random.seed(42)
    N = 200
    
    # 1D Random data
    X_np = np.random.uniform(-5, 5, (N, 1))
    
    # Target function: y = 2x + 1
    w_true = 2.0
    b_true = 1.0
    Y_np = w_true * X_np + b_true
    
    X_t = Tensor(X_np)
    Y_t = Tensor(Y_np)
    
    # MLP with 1 input feature and 1 output target
    model = MLP(in_dim=1, hidden_dim=32, out_dim=1)
    
    lr = 0.005 # Small learning rate to avoid exploding gradients with ReLUs
    for epoch in range(500):
        preds = model.forward(X_t)
        diff = preds - Y_t
        loss = (diff * diff).mean()
        
        for p in model.parameters():
            p.zero_grad()
            
        loss.backward()
        
        for p in model.parameters():
            p.data -= lr * p.grad
            
    # Calculate R^2 Coefficient of Determination
    preds_np = model.forward(X_t).data
    
    ss_res = np.sum((Y_np - preds_np) ** 2)
    ss_tot = np.sum((Y_np - np.mean(Y_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n[MLP Linear Approx] Final MSE Loss: {loss.data:.4f}, R^2 Score: {r2:.4f}")
    
    # Accept the model if it captures a large majority of the variance
    assert r2 > 0.90, f"R^2 ({r2:.4f}) is lower than expected for a linear approximation"


