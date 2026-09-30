"""
Load collected residuals and train the MLP to predict them.

Input features : [X, Y, Phi, vx, vy, omega, p, delta]      (8)
Targets        : residual state rate (x_true - x_nom) / dt (6)

Both are standardized; the normalization stats are saved next to the weights
so the controller can reload everything for inference without retraining.
Outputs: student/mlp_weights.npy, student/norm_stats.npy
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodiff.tensor import Tensor
from ml.nn import MLP
from ml.optim import Adam

STUDENT_DIR = Path(__file__).resolve().parent
DATA_DIR = STUDENT_DIR / "data"


def train(n_epochs=200, batch_size=32, lr=1e-3):
    rng = np.random.default_rng(42)

    states = np.load(DATA_DIR / "states.npy")
    actions = np.load(DATA_DIR / "actions.npy")
    residuals = np.load(DATA_DIR / "residuals.npy")

    X = np.hstack([states, actions])          # (N, 8)
    Y = residuals                             # (N, 6)
    print(f"Dataset: X {X.shape}, Y {Y.shape}")

    # Standardization (saved for inference)
    x_mean, x_std = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-8)
    y_mean, y_std = Y.mean(axis=0), np.maximum(Y.std(axis=0), 1e-8)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    # Train / validation split
    n = len(Xn)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, Y_tr = Xn[tr_idx], Yn[tr_idx]
    X_val, Y_val = Xn[val_idx], Yn[val_idx]

    model = MLP(in_dim=8, hidden_dim=64, out_dim=6)
    opt = Adam(model.parameters(), lr=lr)

    def mse(a, b):
        d = a - b
        return (d * d).mean()

    best_val = np.inf
    best_weights = None
    for epoch in range(n_epochs):
        order = rng.permutation(len(X_tr))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(X_tr), batch_size):
            idx = order[start:start + batch_size]
            xb = Tensor(X_tr[idx])
            yb = Tensor(Y_tr[idx])
            loss = mse(model.forward(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.data)
            n_batches += 1

        val_loss = float(mse(model.forward(Tensor(X_val)), Tensor(Y_val)).data)
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {
                "W1": model.fc1.weights.data.copy(), "b1": model.fc1.bias.data.copy(),
                "W2": model.fc2.weights.data.copy(), "b2": model.fc2.bias.data.copy(),
                "W3": model.fc3.weights.data.copy(), "b3": model.fc3.bias.data.copy(),
            }
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d} | train {epoch_loss / n_batches:.5f} "
                  f"| val {val_loss:.5f} | best {best_val:.5f}")

    # Validation R^2 per dimension with the best weights
    model.fc1.weights.data = best_weights["W1"]; model.fc1.bias.data = best_weights["b1"]
    model.fc2.weights.data = best_weights["W2"]; model.fc2.bias.data = best_weights["b2"]
    model.fc3.weights.data = best_weights["W3"]; model.fc3.bias.data = best_weights["b3"]
    pred = model.forward(Tensor(X_val)).data
    ss_res = ((Y_val - pred) ** 2).sum(axis=0)
    ss_tot = np.maximum(((Y_val - Y_val.mean(axis=0)) ** 2).sum(axis=0), 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    labels = ["dX", "dY", "dPhi", "dvx", "dvy", "domega"]
    print("Validation R^2:", ", ".join(f"{l}={v:.3f}" for l, v in zip(labels, r2)))

    norm_stats = {
        "x_mean": x_mean, "x_std": x_std, "x_inv_std": 1.0 / x_std,
        "y_mean": y_mean, "y_std": y_std,
    }
    np.save(STUDENT_DIR / "mlp_weights.npy", best_weights, allow_pickle=True)
    np.save(STUDENT_DIR / "norm_stats.npy", norm_stats, allow_pickle=True)
    print(f"Saved weights -> {STUDENT_DIR / 'mlp_weights.npy'}")
    print(f"Saved stats   -> {STUDENT_DIR / 'norm_stats.npy'}")
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train residual model.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
