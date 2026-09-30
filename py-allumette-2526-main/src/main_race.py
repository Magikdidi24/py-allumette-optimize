# main_race.py [SKELETON PROVIDED TO STUDENTS]
"""
Race the MPC. You can choose to run the nominal model, the learned model,
or both sequentially to compare lap times.

Usage:
    python main_race.py --mode mpc
    python main_race.py --mode mlmpc
"""
import sys
import argparse
from pathlib import Path
import numpy as np
from environment.track import Track
from environment.racecar import RaceCarEnvironment
from environment.visualizer import RaceVisualizer
from physics.params import BAD_BICYCLE_PARAMS, GOOD_BICYCLE_PARAMS


def resolve_impl_path(root, impl):
    return root / impl


def configure_implementation(impl):
    root = Path(__file__).resolve().parents[1]
    impl_path = resolve_impl_path(root, impl)
    impl_path_str = str(impl_path)
    if impl_path_str in sys.path:
        sys.path.remove(impl_path_str)
    sys.path.insert(0, impl_path_str)


def run_race(controller, env, track, max_steps=600, label="MPC"):
    start_state = np.array([track.x_center[0], track.y_center[0], 0.1, 0.5, 0.0, 0.0])
    state = start_state.copy()

    state_history = [state.copy()]
    crashes = 0
    steps_taken = 0

    n_centerline = len(track.x_center)
    prev_idx = 0
    lap_done = False
    # Skip lap detection for the first few steps so the car moves away from start
    WARMUP_STEPS = 20

    for step in range(max_steps):
        action = controller.get_action(state)
        state  = env.step(state, action, controller.dt)
        state_history.append(state.copy())
        steps_taken += 1

        # --- Lap completion detection ---
        # The track is a closed loop. When the closest centerline index jumps
        # from near the END of the array back to near the START, the car has
        # crossed the finish line.
        curr_idx = track.get_closest_idx(state)
        if step > WARMUP_STEPS:
            if prev_idx > n_centerline * 0.9 and curr_idx < n_centerline * 0.1:
                lap_done = True
        prev_idx = curr_idx

        # --- Real-time Progress Bar ---
        progress = (step + 1) / max_steps
        bar_len = 20
        filled = int(bar_len * progress)
        bar = '█' * filled + '-' * (bar_len - filled)
        sys.stdout.write(
            f"\r  🏎️  [{label}] |{bar}| {step+1}/{max_steps}"
            f" | idx={curr_idx}/{n_centerline} | v: {state[3]:.1f} m/s"
        )
        sys.stdout.flush()

        if lap_done:
            sys.stdout.write("\n")
            break

        if not track.is_on_track(state):
            crashes += 1
            sys.stdout.write("\n")
            print(f"  💥 CRASH at step {step+1}! Race terminated.")
            break

    sys.stdout.write("\n")
    elapsed = steps_taken * controller.dt

    if lap_done:
        print(f"[{label}] 🏁 Lap complete! time={elapsed:.2f}s")
    elif crashes == 0:
        print(f"[{label}] Finished (max steps reached). time={elapsed:.2f}s")
    else:
        print(f"[{label}] DNF (Did Not Finish). Survived for time={elapsed:.2f}s")

    return np.array(state_history), elapsed, crashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAE Autonomous Racing MPC.")
    parser.add_argument("--impl", choices=["student", "teacher"], default="teacher",
                        help="Select implementation package used by MPC/autodiff/ML modules.")
    parser.add_argument("--mode", type=str, choices=["mpc", "mlmpc"], default="mpc",
                        help="Choose which controller to run and visualize.")
    parser.add_argument("--parameters", type=str, choices=["good", "bad"], default="bad",
                        help="Choose which bicycle parameters to use.")
    args = parser.parse_args()

    configure_implementation(args.impl)
    from mpc.car_env_config import create_car_mpc
    from mpc.bicycle import NominalBicycleModel
    nominal = NominalBicycleModel(
        params=GOOD_BICYCLE_PARAMS if args.parameters == "good" else BAD_BICYCLE_PARAMS
    )

    track = Track()
    env   = RaceCarEnvironment()
    vis   = RaceVisualizer(track)

    if args.mode == 'mpc':
        print(f"\n--- Running MPC ---")
        ctrl_nom = create_car_mpc(track, nominal, dt=0.1, learned_model=False)
        hist_nom, t_nom, crashes_nom = run_race(ctrl_nom, env, track, label="MPC")
        vis.animate(hist_nom, dt=ctrl_nom.dt, title="MPC")

    if args.mode == 'mlmpc':
        print(f"\n--- Running Learned MPC ---")
        ctrl_ml = create_car_mpc(track, nominal, dt=0.1, learned_model=True)
        hist_ml, t_ml, crashes_ml = run_race(ctrl_ml, env, track, label="Learned MPC")
        vis.animate(hist_ml, dt=ctrl_ml.dt, title="Learned MPC")
