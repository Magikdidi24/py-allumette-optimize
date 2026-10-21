import numpy as np
from autodiff.tensor import Tensor
from mpc.controller import MPCController, MLMPCController


# ---------------------------------------------------------------------------
# Tuning knobs (cost weights, speed profile, optimizer settings)
# ---------------------------------------------------------------------------
TUNING = {
    # Speed profile
    "a_lat_max": 4.3,     # max lateral accel used to derive corner speeds [m/s^2]
    "v_cap": 2.8,         # top speed on straights [m/s]
    "a_brake": 1.5,       # decel used in the backward pass [m/s^2]
    "a_acc": 1.2,         # accel used in the forward pass [m/s^2]
    # Cost weights
    "w_contour": 8.0,     # lateral (contouring) error
    "w_lag": 2.0,         # longitudinal (lag) error
    "w_speed": 1.5,       # (vx - v_ref)^2
    "w_boundary": 80.0,   # hinge penalty outside the safe corridor
    "margin": 0.26,       # half-width of the safe corridor [m]
    "w_dsteer": 0.5,      # steering rate smoothness
    "w_steer": 0.05,      # steering magnitude
    "w_dthrottle": 0.3,   # throttle rate smoothness
    "w_progress": 0.0,    # explicit forward-progress reward (via lag term)
    # Stability envelope (anti-spin): hinge penalties that only activate
    # beyond normal cornering values.
    "w_yaw": 3.0,         # hinge on yaw rate^2
    "yaw_lim": 2.6,       # [rad/s] (normal cornering peaks ~2.2; spins reach 3+)
    "w_slip": 5.0,        # hinge on lateral velocity^2
    "slip_lim": 0.75,     # [m/s]  (normal cornering peaks ~0.6)
    # Rollout / reference behaviour
    "substeps": 2,        # Euler sub-steps per control interval (accuracy)
    "v_floor": 0.3,       # reference never plans slower than this [m/s]
    # Action bounds
    "throttle_bounds": (-1.0, 1.0),
    "steer_bounds": (-0.45, 0.45),
    # Optimizer
    "lr": 0.08,
    "n_iter": 100,
    "tol": 0.0,
    "warm_throttle": 0.6,
}


class _TrackGeometry:
    """Precomputed centerline geometry: tangents, normals, curvature and a
    curvature-limited reference speed profile (with braking/accel passes)."""

    def __init__(self, track, cfg):
        self.x = np.asarray(track.x_center)
        self.y = np.asarray(track.y_center)
        self.n = len(self.x)
        self.arc = np.asarray(track.arc_lengths)          # length n
        self.total = float(track.total_length)

        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        norm = np.hypot(dx, dy)
        self.tx = dx / norm
        self.ty = dy / norm
        self.nx = -self.ty
        self.ny = self.tx

        # Curvature = d(heading)/d(arc length)
        theta = np.unwrap(np.arctan2(dy, dx))
        ds = np.gradient(self.arc)
        kappa = np.abs(np.gradient(theta) / ds)

        # Curvature-limited speed, then wrap-around braking / accel passes
        v = np.minimum(cfg["v_cap"], np.sqrt(cfg["a_lat_max"] / np.maximum(kappa, 1e-6)))
        for _ in range(3):  # iterate to convergence around the loop
            for i in range(self.n - 2, -1, -1):
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * cfg["a_brake"] * ds[i]))
            v[-1] = min(v[-1], np.sqrt(v[0] ** 2 + 2 * cfg["a_brake"] * ds[-1]))
            for i in range(self.n - 1):
                v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * cfg["a_acc"] * ds[i]))
            v[0] = min(v[0], np.sqrt(v[-1] ** 2 + 2 * cfg["a_acc"] * ds[-1]))
        self.v_prof = v

    def index_at_arc(self, s):
        s = s % self.total
        return int(np.searchsorted(self.arc, s) % self.n)

    def reference(self, state, horizon, dt, cfg):
        """Reference indices and speed targets ahead of the car, spaced by
        integrating the speed profile over the control interval (with an
        accel ramp from the current speed so a slow car is not asked to jump
        to a far-away reference)."""
        px, py, v0 = state[0], state[1], state[3]
        idx = int(np.argmin((self.x - px) ** 2 + (self.y - py) ** 2))
        s = self.arc[idx]
        idxs = np.empty(horizon, dtype=int)
        v_refs = np.empty(horizon)
        v_base = max(v0, cfg["v_floor"])
        for k in range(horizon):
            # Reference point advances at profile speed, softened by an accel
            # ramp from the car's current speed (matters when starting off).
            v_ref = min(self.v_prof[idx], v_base + cfg["a_acc"] * (k + 1) * dt)
            s += v_ref * dt
            idx = self.index_at_arc(s)
            idxs[k] = idx
            # Speed target is the full profile speed: a unique attractor that
            # always pulls the car toward the racing pace.
            v_refs[k] = self.v_prof[idx]
        return idxs, v_refs


def create_car_mpc(track, bycicle_model, horizon=15, dt=0.08, learned_model=False, use_good_params=False):
    """Factory for the racing MPC (nominal or residual-corrected).

    The stage cost follows the MPCC formulation: contouring error (lateral
    offset to the reference line) and lag error (longitudinal delay w.r.t. a
    point advancing at the reference speed), plus speed tracking, a soft
    track-boundary barrier and control smoothness terms.
    """
    cfg = TUNING
    geom = _TrackGeometry(track, cfg)

    def dynamics_fn(x, u, dt_val):
        # Optional sub-stepping keeps forward Euler accurate over the (fairly
        # long) control interval, especially in tight corners.
        n_sub = int(cfg["substeps"])
        h = dt_val / n_sub
        for _ in range(n_sub):
            x = bycicle_model.euler_step(x, u, h)
        return x

    def cost_fn(initial_state, states, U_t):
        N = len(states)
        idxs, ref_v = geom.reference(initial_state, N, dt, cfg)
        ref_x, ref_y = geom.x[idxs], geom.y[idxs]
        t_x, t_y = geom.tx[idxs], geom.ty[idxs]
        n_x, n_y = geom.nx[idxs], geom.ny[idxs]

        S = Tensor.stack(states)          # (N, 6)
        px, py, vx = S[:, 0], S[:, 1], S[:, 3]
        vy, omega = S[:, 4], S[:, 5]
        dx = px - ref_x
        dy = py - ref_y

        e_c = dx * n_x + dy * n_y         # contouring (lateral) error
        e_l = dx * t_x + dy * t_y         # lag (longitudinal) error
        dv = vx - ref_v

        cost = (e_c * e_c).sum() * cfg["w_contour"]
        cost = cost + (e_l * e_l).sum() * cfg["w_lag"]
        cost = cost + (dv * dv).sum() * cfg["w_speed"]

        # Soft barrier: quadratic hinge once the lateral error leaves the
        # safe corridor (track half-width is 0.5).
        barrier = (e_c * e_c - cfg["margin"] ** 2).relu()
        cost = cost + barrier.sum() * cfg["w_boundary"]

        # Stability envelope: keep yaw rate and lateral slip out of the
        # spin region (hinges are inactive during normal cornering).
        yaw_pen = (omega * omega - cfg["yaw_lim"] ** 2).relu()
        slip_pen = (vy * vy - cfg["slip_lim"] ** 2).relu()
        cost = cost + yaw_pen.sum() * cfg["w_yaw"] + slip_pen.sum() * cfg["w_slip"]

        if cfg["w_progress"] > 0.0:
            cost = cost - e_l.sum() * cfg["w_progress"]

        throttle, steer = U_t[:, 0], U_t[:, 1]
        d_steer = steer[1:] - steer[:-1]
        d_thr = throttle[1:] - throttle[:-1]
        cost = cost + (d_steer * d_steer).sum() * cfg["w_dsteer"]
        cost = cost + (steer * steer).sum() * cfg["w_steer"]
        cost = cost + (d_thr * d_thr).sum() * cfg["w_dthrottle"]
        return cost

    action_bounds = [cfg["throttle_bounds"], cfg["steer_bounds"]]
    ctrl_cls = MLMPCController if learned_model else MPCController
    ctrl = ctrl_cls(horizon, dt, dynamics_fn, cost_fn,
                    action_bounds=action_bounds,
                    lr=cfg["lr"], n_iter=cfg["n_iter"])
    ctrl.optimizer = "adam"
    ctrl.tol = cfg["tol"]
    # Cold-start warm guess: gentle throttle, straight wheels
    ctrl._U_prev = np.tile([cfg["warm_throttle"], 0.0], (horizon, 1))
    return ctrl
