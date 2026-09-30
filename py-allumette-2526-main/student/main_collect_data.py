"""
Run the nominal MPC for one full lap and save the state/action/residual
training data.

The residual target follows the SAE definition:
    y_k = (x_true_{k+1} - x_nom_{k+1}) / dt
i.e. the state-space one-step error of the nominal (bad-parameter, Euler)
model against the true environment (good-parameter, RK45), expressed as a
rate so it can be rescaled by the controller's dt at inference time.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# pyrefly: ignore [missing-import]
from environment.track import Track
# pyrefly: ignore [missing-import]
from environment.racecar import RaceCarEnvironment
# pyrefly: ignore [missing-import]
from environment.visualizer import RaceVisualizer
from mpc.bicycle import NominalBicycleModel
from mpc.car_env_config import create_car_mpc
from autodiff.tensor import Tensor
# pyrefly: ignore [missing-import]
from physics.params import BAD_BICYCLE_PARAMS

DATA_DIR = Path(__file__).resolve().parent / "data"
DT = 0.1               # control interval used at race time (src/main_race.py)
NOISE_THROTTLE = 0.06  # exploration noise so the dataset covers a band
NOISE_STEER = 0.03     # around the nominal racing line


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def collect_data(n_steps=500):
    rng = np.random.default_rng(0)
    track = Track()
    env = RaceCarEnvironment()
    nominal = NominalBicycleModel(params=BAD_BICYCLE_PARAMS)
    ctrl = create_car_mpc(track, nominal, dt=DT, learned_model=False)

    start_state = np.array([track.x_center[0], track.y_center[0], 0.1, 0.5, 0.0, 0.0])
    state = start_state.copy()

    states, actions, residuals = [], [], []
    for step in range(n_steps):
        action = ctrl.get_action(state)
        # Exploration noise (applied to BOTH the env and the nominal
        # prediction, so the residual stays consistent).
        action = action + rng.normal(0.0, [NOISE_THROTTLE, NOISE_STEER])
        action = np.clip(action, [-1.0, -0.45], [1.0, 0.45])

        # Prediction of the exact model used inside the MPC rollout
        x_nom = ctrl.dynamics_fn(Tensor(state), Tensor(action), DT).data
        x_true = env.step(state, action, DT)

        res = (x_true - x_nom) / DT
        res[2] = wrap_angle(x_true[2] - x_nom[2]) / DT

        states.append(state.copy())
        actions.append(action.copy())
        residuals.append(res)

        state = x_true
        if not track.is_on_track(state):
            print(f"  off-track at step {step}; resetting to the start line")
            state = start_state.copy()
            ctrl._U_prev = None

        if step % 50 == 0:
            print(f"  step {step:4d}/{n_steps} | v={state[3]:.2f} m/s | "
                  f"|res|={np.abs(res).max():.4f}")

    states = np.array(states)
    actions = np.array(actions)
    residuals = np.array(residuals)

    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "states.npy", states)
    np.save(DATA_DIR / "actions.npy", actions)
    np.save(DATA_DIR / "residuals.npy", residuals)
    print(f"\nSaved {len(states)} samples to {DATA_DIR}")
    print(f"residual rate stats (mean |.| per dim): {np.abs(residuals).mean(axis=0)}")
    return states, actions, residuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect residual training data.")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    collect_data(n_steps=args.steps)
