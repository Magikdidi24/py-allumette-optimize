# SAE: Organized Workspace Layout

The project is organized into student-facing and runtime folders, plus a private reference implementation.

```text
SAE/
├── src/        # Infrastructure/runtime assets only (environment, perfect physics, race entrypoint, data)
├── student/    # Student implementations + data collection / training scripts
└── tests/      # External tests, selectable target implementation
```

## Folder Roles

- `src/`: structural and runtime files (`environment`, `physics`, `main_race.py`, `data`, `requirements.txt`).
- `student/`: student `autodiff`, `ml`, `mpc` modules plus `main_collect_data.py` and `main_train_model.py`.
- private reference implementation: solved `autodiff`, `ml`, `mpc` modules plus the same training scripts.
- `tests/`: external test suite with a selector to target `student` or a private reference implementation.

## Setup

Use Python 3.9+ and install dependencies from `src/requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```

## Run Tests

From the repository root:

```bash
pytest --impl student tests/test_autodiff.py
pytest --impl student tests/test_numerics.py
pytest --impl student
```

`--impl` controls which package is used for `autodiff`, `ml`, and `mpc` imports (`student` or private reference).

## Run Main Scripts

From the repository root:

```bash
python src/main_race.py --impl student --mode both
python src/main_race.py --impl student --mode nominal --nominal-params good
python src/main_race.py --impl student --mode learned --learned-params bad
python student/main_collect_data.py
python student/main_train_model.py
```

Use `--nominal-params {good,bad}` and `--learned-params {good,bad}` to compare MPC and ML-MPC performance under different internal model parameters.

To validate student code directly, switch `--impl student`.