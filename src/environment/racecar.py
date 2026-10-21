# environment/racecar.py
import numpy as np
from scipy.integrate import solve_ivp
from physics.bicycle import NominalBicycleModel
from physics.params import GOOD_BICYCLE_PARAMS

class RaceCarEnvironment:
    def __init__(self):
        self._model = NominalBicycleModel(params=GOOD_BICYCLE_PARAMS)

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        solution = solve_ivp(
            lambda _t, x: self._model.continuous_dynamics(x, action),
            [0.0, dt],
            state,
            method="RK45",
            max_step=dt / 10,
        )
        next_state = solution.y[:, -1]
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        return next_state