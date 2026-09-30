import numpy as np
import pytest
from autodiff.tensor import Tensor
from mpc.controller import MPCController

def test_simple_mpc_dist_only():
    """
    Test the generic MPC optimizing only for reaching the target (no control cost).
    It should apply max control effort to reach the target rapidly.
    """
    N = 10
    dt = 0.2
    target = 10.0
    
    def dynamics_fn(x_t, u_t, dt_val):
        return x_t + u_t[0] * dt_val
        
    def horizon_cost_fn(initial_state, states, U_t):
        cost = Tensor([0.0])
        for x_t in states:
            cost = cost + (x_t - target) ** 2
        return cost

    mpc = MPCController(horizon=N, dt=dt, dynamics_fn=dynamics_fn, 
                     cost_fn=horizon_cost_fn, action_bounds=[(-10.0, 10.0)], 
                     lr=0.01, n_iter=50)
    
    x_curr = np.array([0.0])
    print(f"\n--- [Generic MPC Autodiff] Dist Only ---")
    print(f"Start: {x_curr[0]} -> Target: {target}")
    
    for step in range(30):
        action = mpc.get_action(x_curr)
        x_curr[0] = x_curr[0] + action[0] * dt
        print(f"Step {step+1:2d} | Action: {action[0]:5.2f} | State: {x_curr[0]:.3f}")
        if np.abs(x_curr[0] - target) < 0.05:
            break
            
    assert np.abs(x_curr[0] - target) < 0.5, f"Missed target. End state: {x_curr[0]:.3f}"


def test_simple_mpc_with_ctrl_cost():
    """
    Test the generic MPC with both target distance and control cost, yielding a smoother
    but slower approach to the target.
    """
    N = 10
    dt = 0.2
    target = 10.0
    
    def dynamics_fn(x_t, u_t, dt_val):
        return x_t + u_t[0] * dt_val
        
    def horizon_cost_fn(initial_state, states, U_t):
        cost = Tensor([0.0])
        for k, x_t in enumerate(states):
            u_t = U_t[k]
            dist_cost = (x_t - target) ** 2
            ctrl_cost = u_t[0] ** 2 * 0.1  # Stronger control cost
            cost = cost + dist_cost + ctrl_cost
        return cost

    mpc = MPCController(horizon=N, dt=dt, dynamics_fn=dynamics_fn, 
                     cost_fn=horizon_cost_fn, action_bounds=[(-10.0, 10.0)], 
                     lr=0.01, n_iter=50)
    
    x_curr = np.array([0.0])
    print(f"\n--- [Generic MPC Autodiff] Dist + Control Cost ---")
    print(f"Start: {x_curr[0]} -> Target: {target}")
    
    for step in range(30):
        action = mpc.get_action(x_curr)
        x_curr[0] = x_curr[0] + action[0] * dt
        print(f"Step {step+1:2d} | Action: {action[0]:5.2f} | State: {x_curr[0]:.3f}")
        if np.abs(x_curr[0] - target) < 0.05:
            break
            
    assert np.abs(x_curr[0] - target) < 0.5, f"Missed target. End state: {x_curr[0]:.3f}"
