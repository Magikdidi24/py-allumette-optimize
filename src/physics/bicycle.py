import numpy as np
from physics.params import GOOD_BICYCLE_PARAMS, BAD_BICYCLE_PARAMS


class NominalBicycleModel:
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

    def tire_forces(self, state, action):
        _, _, _, vx, vy, omega = state[0], state[1], state[2], state[3], state[4], state[5]
        p, delta = action[0], action[1]

        vx_safe = max(abs(vx), 0.3)
        alpha_f = delta - np.arctan2(vy + self.l_f * omega, vx_safe)
        alpha_r = -np.arctan2(vy - self.l_r * omega, vx_safe)

        F_rx = self.C_m * p - self.C_drag * vx * abs(vx) - self.C_roll * vx
        F_fy = self.C_f * alpha_f
        F_ry = self.C_r * alpha_r
        return F_rx, F_fy, F_ry

    def continuous_dynamics(self, state, action):
        X, Y, Phi, vx, vy, omega = state[0], state[1], state[2], state[3], state[4], state[5]
        _, delta = action[0], action[1]

        F_rx, F_fy, F_ry = self.tire_forces(state, action)
        m, I_z, l_f, l_r = self.mass, self.I_z, self.l_f, self.l_r

        dX = vx * np.cos(Phi) - vy * np.sin(Phi)
        dY = vx * np.sin(Phi) + vy * np.cos(Phi)
        dPhi = omega
        dvx = (F_rx - F_fy * np.sin(delta) + m * vy * omega) / m
        dvy = (F_ry + F_fy * np.cos(delta) - m * vx * omega) / m
        domega = (F_fy * l_f * np.cos(delta) - F_ry * l_r) / I_z
        return np.array([dX, dY, dPhi, dvx, dvy, domega])

    def euler_step(self, state, action, dt=0.03):
        dX = self.continuous_dynamics(state, action)
        return state + np.array(dX) * dt
