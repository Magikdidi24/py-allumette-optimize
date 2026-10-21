from pathlib import Path

import numpy as np
from autodiff.tensor import Tensor
from ml.nn import MLP


class MPCController:
    def __init__(self, horizon, dt, dynamics_fn, cost_fn, action_bounds=None, lr=0.02, n_iter=50):
        self.N = horizon
        self.dt = dt
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.action_bounds = action_bounds
        self.lr = lr
        self.n_iter = n_iter
        self.action_dim = None
        self._U_prev = None
        # Tunables (attributes, not signature args): the car factory switches
        # the inner optimizer to Adam and enables early stopping for speed.
        self.optimizer = "gd"
        self.tol = 0.0

    def _project(self, U):
        if self.action_bounds is None:
            return U
        for j, (low, high) in enumerate(self.action_bounds):
            U[:, j] = np.clip(U[:, j], low, high)
        return U

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float64)
        if self.action_dim is None:
            self.action_dim = len(self.action_bounds) if self.action_bounds else 1

        # Warm start: shift the previous optimal sequence one step forward
        if self._U_prev is None:
            U = np.zeros((self.N, self.action_dim))
        else:
            U = np.roll(self._U_prev, -1, axis=0)
            U[-1] = self._U_prev[-1]
        U = self._project(U)

        m = np.zeros_like(U)
        v = np.zeros_like(U)
        best_cost, best_U = np.inf, U.copy()
        prev_cost, flat_iters = None, 0

        for it in range(self.n_iter):
            # Fresh graph every iteration: roll the dynamics out over the
            # horizon, evaluate the cost, and backprop down to the actions.
            U_t = Tensor(U, requires_grad=True)
            x = Tensor(state)
            states = []
            for k in range(self.N):
                x = self.dynamics_fn(x, U_t[k], self.dt)
                states.append(x)

            cost = self.cost_fn(state, states, U_t)
            cost.backward()
            g = U_t.grad

            c = float(np.sum(cost.data))
            if c < best_cost:
                best_cost, best_U = c, U.copy()

            if self.optimizer == "adam":
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * (g * g)
                m_hat = m / (1.0 - 0.9 ** (it + 1))
                v_hat = v / (1.0 - 0.999 ** (it + 1))
                U = self._project(U - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8))
            else:
                U = self._project(U - self.lr * g)

            # Early stop once the cost stops improving (only when enabled)
            if self.tol > 0.0 and prev_cost is not None:
                if prev_cost - c < self.tol * max(abs(prev_cost), 1.0):
                    flat_iters += 1
                    if flat_iters >= 3:
                        break
                else:
                    flat_iters = 0
            prev_cost = c

        self._U_prev = best_U.copy()
        return best_U[0].copy()


class MLMPCController(MPCController):
    """MPC whose internal model is the nominal bicycle model corrected by a
    learned residual MLP:  x_{k+1} = f_nom(x_k, u_k) + dt * g_theta(x_k, u_k).

    The MLP weights are loaded from student/mlp_weights.npy and
    student/norm_stats.npy (inference only, no retraining needed).
    """

    WEIGHTS_FILE = "mlp_weights.npy"
    NORM_FILE = "norm_stats.npy"

    def __init__(self, horizon, dt, nominal_dynamics_fn, cost_fn, action_bounds=None, lr=0.02, n_iter=50):
        self.nominal_dynamics_fn = nominal_dynamics_fn
        self.model, self.norm = self._load_residual_model()
        super().__init__(horizon, dt, self._learned_dynamics, cost_fn,
                         action_bounds=action_bounds, lr=lr, n_iter=n_iter)

    def _load_residual_model(self):
        root = Path(__file__).resolve().parents[1]
        w_path = root / self.WEIGHTS_FILE
        n_path = root / self.NORM_FILE
        if not (w_path.exists() and n_path.exists()):
            print(f"[MLMPC] WARNING: no trained weights found at {w_path}; "
                  f"falling back to the nominal model. Run "
                  f"student/main_collect_data.py then student/main_train_model.py.")
            return None, None

        weights = np.load(w_path, allow_pickle=True).item()
        norm = np.load(n_path, allow_pickle=True).item()

        in_dim = weights["W1"].shape[0]
        hidden = weights["W1"].shape[1]
        out_dim = weights["W3"].shape[1]
        model = MLP(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim)
        model.fc1.weights = Tensor(weights["W1"])
        model.fc1.bias = Tensor(weights["b1"])
        model.fc2.weights = Tensor(weights["W2"])
        model.fc2.bias = Tensor(weights["b2"])
        model.fc3.weights = Tensor(weights["W3"])
        model.fc3.bias = Tensor(weights["b3"])
        return model, norm

    def _residual(self, x, u):
        """Normalized MLP prediction of the residual state derivative."""
        feats = Tensor.stack([x[0], x[1], x[2], x[3], x[4], x[5], u[0], u[1]])
        feats = (feats - self.norm["x_mean"]) * self.norm["x_inv_std"]
        out = self.model.forward(feats.reshape(1, -1))[0]
        return out * self.norm["y_std"] + self.norm["y_mean"]

    def _learned_dynamics(self, x, u, dt):
        x_next = self.nominal_dynamics_fn(x, u, dt)
        if self.model is None:
            return x_next
        return x_next + self._residual(x, u) * dt
