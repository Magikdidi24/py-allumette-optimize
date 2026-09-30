from __future__ import annotations

import sys
from pathlib import Path


def resolve_impl_path(root: Path, impl: str) -> Path:
    return root / impl


def pytest_addoption(parser):
    parser.addoption(
        "--impl",
        action="store",
        default="teacher",
        choices=["student", "teacher"],
        help="Select implementation package for autodiff/ml/mpc imports.",
    )


def pytest_configure(config):
    root = Path(__file__).resolve().parent
    impl = config.getoption("impl")

    src_path = str(root / "src")
    impl_path = str(resolve_impl_path(root, impl))

    sys.path[:] = [path for path in sys.path if path not in {src_path, impl_path}]
    sys.path.insert(0, impl_path)
    sys.path.insert(1, src_path)
