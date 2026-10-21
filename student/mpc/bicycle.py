import numpy as np
from autodiff.tensor import Tensor


class NominalBicycleModel:
    """Dynamic bicycle model, usable both with plain numpy arrays (fast
    rollouts, numerics validation) and with autodiff Tensors (differentiable
    MPC rollouts). The Tensor path mirrors the numpy equations one-to-one so
    gradients flow from the predicted states back to the actions.
    """

    def __init__(self, params):
        self.mass = params["mass"]
        self.I_z = params["I_z"]
        self.l_f = params["l_f"]
        self.l_r = params["l_r"]
        self.C_f = params["C_f"]
        self.C_r = params["C_r"]
        self.C_m = params["C_m"]
        self.C_drag = params["C_drag"]
        self.C_roll = params["C_roll"]

    def continuous_dynamics(self, state, action):
        if isinstance(state, Tensor) or isinstance(action, Tensor):
            return self._continuous_dynamics_tensor(state, action)
        return self._continuous_dynamics_numpy(np.asarray(state, dtype=np.float64),
                                               np.asarray(action, dtype=np.float64))

    def euler_step(self, state, action, dt=0.03):
        # Forward Euler: x_{k+1} = x_k + dt * f_c(x_k, u_k).
        # Always returns a Tensor so the step stays differentiable end-to-end.
        state = state if isinstance(state, Tensor) else Tensor(state)
        action = action if isinstance(action, Tensor) else Tensor(action)
        dX = self._continuous_dynamics_tensor(state, action)
        return state + dX * dt

    # ------------------------------------------------------------------
    # numpy path (mirrors src/physics/bicycle.py)
    # ------------------------------------------------------------------
    def _continuous_dynamics_numpy(self, state, action):
        X, Y, Phi, vx, vy, omega = state
        p, delta = action[0], action[1]

        vx_safe = max(abs(vx), 0.3)
        alpha_f = delta - np.arctan2(vy + self.l_f * omega, vx_safe)
        alpha_r = -np.arctan2(vy - self.l_r * omega, vx_safe)

        F_rx = self.C_m * p - self.C_drag * vx * abs(vx) - self.C_roll * vx
        F_fy = self.C_f * alpha_f
        F_ry = self.C_r * alpha_r

        m, I_z, l_f, l_r = self.mass, self.I_z, self.l_f, self.l_r
        dX = vx * np.cos(Phi) - vy * np.sin(Phi)
        dY = vx * np.sin(Phi) + vy * np.cos(Phi)
        dPhi = omega
        dvx = (F_rx - F_fy * np.sin(delta) + m * vy * omega) / m
        dvy = (F_ry + F_fy * np.cos(delta) - m * vx * omega) / m
        domega = (F_fy * l_f * np.cos(delta) - F_ry * l_r) / I_z
        return np.array([dX, dY, dPhi, dvx, dvy, domega])

    # ------------------------------------------------------------------
    # Tensor path (differentiable)
    # ------------------------------------------------------------------
    def _continuous_dynamics_tensor(self, state, action):
        state = state if isinstance(state, Tensor) else Tensor(state)
        action = action if isinstance(action, Tensor) else Tensor(action)

        X, Y, Phi = state[0], state[1], state[2]
        vx, vy, omega = state[3], state[4], state[5]
        p, delta = action[0], action[1]

        # Differentiable stand-in for max(|vx|, 0.3)
        vx_safe = abs(vx).clip(0.3, None)
        alpha_f = delta - (vy + omega * self.l_f).arctan2(vx_safe)
        alpha_r = -((vy - omega * self.l_r).arctan2(vx_safe))

        F_rx = p * self.C_m - vx * abs(vx) * self.C_drag - vx * self.C_roll
        F_fy = alpha_f * self.C_f
        F_ry = alpha_r * self.C_r

        m, I_z, l_f, l_r = self.mass, self.I_z, self.l_f, self.l_r
        sin_phi, cos_phi = Phi.sin(), Phi.cos()
        sin_d, cos_d = delta.sin(), delta.cos()

        dX = vx * cos_phi - vy * sin_phi
        dY = vx * sin_phi + vy * cos_phi
        dPhi = omega
        dvx = (F_rx - F_fy * sin_d + vy * omega * m) * (1.0 / m)
        dvy = (F_ry + F_fy * cos_d - vx * omega * m) * (1.0 / m)
        domega = (F_fy * (l_f * cos_d) - F_ry * l_r) * (1.0 / I_z)
        return Tensor.stack([dX, dY, dPhi, dvx, dvy, domega])
