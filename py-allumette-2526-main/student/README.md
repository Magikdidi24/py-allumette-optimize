# Student Workspace

Complete implementation of the Py-Allumette learning-based MPC stack.

## Modules

- `autodiff/tensor.py` — reverse-mode autodiff engine (numpy only): arithmetic
  ops, matmul, activations (relu/tanh), trig (sin/cos/arctan2), exp/log,
  reductions with axis support, shape ops (getitem/reshape/stack), abs/clip,
  broadcasting with gradient un-broadcasting, gradient accumulation, and an
  iterative topological backward pass (no recursion limit on long rollouts).
- `ml/nn.py` — `Linear` (He init, `x @ W + b`) and 3-layer `MLP` with ReLU.
- `ml/optim.py` — `SGD` and `Adam` (bias-corrected moments).
- `mpc/bicycle.py` — dynamic bicycle model; numpy path for fast evaluation and
  a Tensor path so MPC rollouts stay differentiable end-to-end.
- `mpc/controller.py` — `MPCController`: warm-started projected gradient
  descent (optional Adam inner optimizer) over the action sequence, best-cost
  solution tracking, optional early stop. `MLMPCController`: same loop with
  the learned dynamics `x_next = f_nom(x, u) + dt * g_theta(x, u)`; loads
  `mlp_weights.npy` / `norm_stats.npy` for inference only.
- `mpc/car_env_config.py` — MPC factory: MPCC-style cost (contouring + lag
  errors against a curvature-limited speed profile, speed tracking, soft
  track-boundary barrier, anti-spin yaw/slip hinges, control smoothness),
  action bounds and all tuning knobs in `TUNING`.

## Reproducing the trained weights

```bash
python student/main_collect_data.py --steps 600   # nominal (bad-params) laps + noise
python student/main_train_model.py --epochs 300   # trains MLP, saves weights + stats
```

Outputs `student/mlp_weights.npy` and `student/norm_stats.npy` (checked in;
the race reloads them directly — no retraining needed).

## Race

```bash
python src/main_race.py --impl student --mode mlmpc --parameters bad
```

## Design notes

- The environment integrates the good-parameter model with RK45; the MPC uses
  the (possibly bad-parameter) Euler model with 2 sub-steps. The MLP learns
  the residual state error per unit time — parameter mismatch plus remaining
  discretization error — from one noisy exploratory run.
- The car is an oversteer vehicle (l_f·C_f > l_r·C_r): open-loop unstable
  above ~1.9 m/s, so the cost includes hinge penalties on yaw rate and
  lateral velocity that only activate outside the normal cornering envelope —
  they prevent the optimizer from commanding spins while leaving racing
  behaviour untouched.
