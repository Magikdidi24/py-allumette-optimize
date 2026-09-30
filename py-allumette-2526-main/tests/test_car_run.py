import numpy as np
import pytest
from environment.track import Track
from environment.racecar import RaceCarEnvironment
from mpc.car_env_config import create_car_mpc
from physics.params import GOOD_BICYCLE_PARAMS
from mpc.bicycle import NominalBicycleModel

def test_nominal_mpc_track_following():
    # Only simulate for a subset of steps to keep test fast, or full lap
    # We want to check if it actually reduces distance to the track centerline.
    track = Track()
    env = RaceCarEnvironment()
    controller = create_car_mpc(track, NominalBicycleModel(GOOD_BICYCLE_PARAMS), horizon=15, dt=0.08) # reduced horizon
    
    # Start slightly off center to see if it converges
    start_state = np.array([track.x_center[0] - 0.4, track.y_center[0], 0.1, 0.5, 0.0, 0.0])
    state = start_state.copy()
    
    def get_dist(s):
        idx = track.get_closest_idx(s)
        cx, cy = track.x_center[idx], track.y_center[idx]
        return np.hypot(s[0] - cx, s[1] - cy)

    init_dist = get_dist(state)
    print(f"\n[Test Car Run] Starts at: ({state[0]:.2f}, {state[1]:.2f}) | dist: {init_dist:.3f}")
    
    max_steps = 15
    for step in range(max_steps):
        action = controller.get_action(state)
        print(action)
        state = env.step(state, action, controller.dt)
        
        # Check if severely off track
        if not track.is_on_track(state):
            pytest.fail(f"CRASH at step {step}! Position: ({state[0]:.2f}, {state[1]:.2f})")
            
    final_dist = get_dist(state)
    print(f"Ended at state: {state} | dist: {final_dist:.3f}")
    
    assert final_dist < init_dist, f"The car didn't get closer to the track! Init dist: {init_dist:.3f}, Final dist: {final_dist:.3f}"

