# tests/test_numerics.py [FULLY PROVIDED TO STUDENTS]
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from mpc.bicycle import NominalBicycleModel
from physics.params import GOOD_BICYCLE_PARAMS
from physics.bicycle import NominalBicycleModel as ReferenceBicycleModel


def test_euler_vs_rk45():
    """
    The Euler step must match the reference RK45 integrator to within 2%
    (as specified in the SAE success criteria).
    """
    model  = NominalBicycleModel(params=GOOD_BICYCLE_PARAMS)
    dt     = 0.01  # Small dt so Euler is relatively accurate
    x0     = np.array([0.0, 0.0, 0.1, 1.5, 0.05, 0.02])
    u      = np.array([0.5, 0.1])

    # Student's Euler prediction
    x_euler = model.euler_step(x0, u, dt)
    x_euler = np.array([x.data if hasattr(x, 'data') else x for x in x_euler])  # Convert Tensor to numpy if needed

    # Scipy's highly accurate RK45 reference
    sol = solve_ivp(lambda t, x: ReferenceBicycleModel(params=GOOD_BICYCLE_PARAMS).continuous_dynamics(x, u),
                    [0, dt], x0, method='RK45', max_step=dt/10)
    x_ref = sol.y[:, -1]

    rel_error = np.abs(x_euler - x_ref) / (np.abs(x_ref) + 1e-8)
    max_err = rel_error.max()

    print("\n[Euler Discretization] Checking accuracy...")
    print(f"  Expected State (RK45): \n  {x_ref}")
    print(f"  Actual State   (Euler):\n  {x_euler}")
    print(f"  Max Relative Error:    {max_err:.2%}")

    try:
        assert max_err < 0.02, f"Euler error too large: {max_err:.2%}"
        print("  ✅ PASS: Euler approximation is within 2% of continuous integral")
    except AssertionError as e:
        print("  ❌ FAIL: Euler approximation diverged too much")
        raise e


def test_dynamics_shape():
    model = NominalBicycleModel(params=GOOD_BICYCLE_PARAMS)
    x = np.zeros(6); x[3] = 1.0  # Non-zero velocity
    u = np.array([0.5, 0.1])
    
    xdot = model.continuous_dynamics(x, u)
    
    print("\n[Dynamics Shape] Checking dimensions...")
    print(f"  Expected len: 6")
    print(f"  Actual len:   {len(xdot)}")
    
    try:
        assert len(xdot) == 6, f"Expected dynamics output of length 6, got {len(xdot)}"
        print("  ✅ PASS: Shape is correct")
    except AssertionError as e:
        print("  ❌ FAIL: Incorrect output shape")
        raise e