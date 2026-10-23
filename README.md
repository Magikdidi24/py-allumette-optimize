# py-allumette-optimize

[![tests](https://github.com/Magikdidi24/py-allumette-optimize/actions/workflows/ci.yml/badge.svg)](https://github.com/Magikdidi24/py-allumette-optimize/actions/workflows/ci.yml)

School project at EPITA (SAE Py-Allumette, 2025-26): drive a small simulated race car around a track with MPC, then with an MPC that has a learned model. No PyTorch allowed in the controller, so the autodiff, the MLP, the optimizers and the MPC are written by hand in NumPy.

![Learned MPC, one lap](docs/race_mlmpc.gif)

Every 0.1 s the MPC optimizes the next 15 actions (throttle and steering) with Adam, by backprop through a rollout of a bicycle model. The tests check my autodiff against PyTorch.

The simulator moves the car with the good parameters and RK45, but the controller only has Euler steps and the wrong ("bad") parameters. For the learned version I trained an MLP on one noisy run of the normal MPC to predict that error, and the controller plans with `f_nominal(x, u) + dt * mlp(x, u)`.

## Results

One lap each, all 4 finish without leaving the track. Full video of the learned one: [race_mlmpc.mp4](race_mlmpc.mp4).

| controller | model params | lap time |
|---|---|---:|
| MPC | bad | 14.00 s |
| learned MPC | bad | 14.20 s |
| MPC | good | 14.00 s |
| learned MPC | good | 14.40 s |

The learned model doesn't make the lap faster. Plain MPC is 0.2 s quicker with the bad params and 0.4 s with the good ones. The MLP was trained against the bad params, so the "good" rows use it outside what it was trained for. Autodiff in pure NumPy is also slow, one lap took 15 to 25 min to simulate.

## How to run

Python 3.10, from the repo root. `--impl student` is needed because the default `teacher` is the course reference code, which isn't in this repo.

```bash
pip install -r src/requirements.txt
python src/main_race.py --impl student --mode mlmpc --parameters bad   # or --mode mpc
pytest --impl student

# retrain the MLP (overwrites the weights in student/)
python student/main_collect_data.py --steps 600
python student/main_train_model.py --epochs 300
```

`src/`, `tests/` and `docs/course.md` come from the course. Everything in `student/` is mine.
