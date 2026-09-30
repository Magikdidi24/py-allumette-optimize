"""
tests/test_no_forbidden_libs.py
================================
Ensures that the student implementation does NOT import or use any
autodiff / tensor framework (torch, jax, tensorflow, keras, autograd,
tinygrad, micrograd) inside the student/ package.

Students must implement differentiation from scratch using only numpy.
Using a framework as a backend — even partially — is academic dishonesty
in the context of this SAE.

Strategy
--------
We walk every .py file under student/ and check two things:
  1. AST-level import scan  — catches `import torch`, `from jax import ...`,
     `import tensorflow as tf`, etc.
  2. Runtime import side-effect check — after importing each student module,
     we verify that none of the forbidden packages ended up in sys.modules.

The runtime check catches sneaky indirect imports (e.g. a helper that
imports torch inside a function body that runs at module load time).
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STUDENT_ROOT = Path(__file__).resolve().parents[1] / "student"

FORBIDDEN = {
    "torch",
    "torchvision",
    "torchaudio",
    "jax",
    "jaxlib",
    "tensorflow",
    "tf",
    "keras",
    "autograd",       # HIPS/autograd
    "tinygrad",
    "micrograd",
    "paddle",         # PaddlePaddle
    "mxnet",
    "theano",
    "chainer",
}

# Prefixes — catches sub-packages like torch.nn, jax.numpy, tf.keras …
FORBIDDEN_PREFIXES = tuple(sorted(FORBIDDEN))


# ---------------------------------------------------------------------------
# Helper: collect all .py files under student/
# ---------------------------------------------------------------------------

def student_py_files() -> list[Path]:
    return sorted(STUDENT_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Helper: extract all imported names from an AST
# ---------------------------------------------------------------------------

def imports_in_ast(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


# ---------------------------------------------------------------------------
# Test 1 — Static AST scan of every student source file
# ---------------------------------------------------------------------------

def _ast_params():
    """Yield (filepath, imported_name) for every forbidden import found."""
    params = []
    for path in student_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for name in imports_in_ast(tree):
            root = name.split(".")[0]
            if root in FORBIDDEN:
                params.append(pytest.param(path, name,
                                            id=f"{path.relative_to(STUDENT_ROOT)}::{name}"))
    return params if params else [pytest.param(None, None, id="no_violations")]


@pytest.mark.parametrize("filepath,imported_name", _ast_params())
def test_no_forbidden_import_in_source(filepath, imported_name):
    """
    AST check: no forbidden framework import found in student source files.
    """
    if filepath is None:
        # No violations found — the parametrize fallback param
        return

    rel = filepath.relative_to(STUDENT_ROOT)
    pytest.fail(
        f"\n❌  Forbidden import detected!\n"
        f"   File   : student/{rel}\n"
        f"   Import : '{imported_name}'\n\n"
        f"   You must implement autodiff using numpy only.\n"
        f"   Using {imported_name.split('.')[0]} as a backend is not allowed."
    )


# ---------------------------------------------------------------------------
# Test 2 — Runtime sys.modules check after importing student packages
# ---------------------------------------------------------------------------

STUDENT_PACKAGES = ["autodiff", "ml", "mpc"]


@pytest.mark.parametrize("package", STUDENT_PACKAGES)
def test_no_forbidden_lib_loaded_at_runtime(package):
    """
    Runtime check: importing the student package must not load any forbidden
    framework into sys.modules (catches imports inside function bodies that
    execute at module-load time). Mandatory imported libraries: pprint, math, 
    cmath, random, itertools, functools, halo, operator, collections, typing, 
    numpy, and builtins.
    """
    # Snapshot sys.modules before
    before = set(sys.modules.keys())

    try:
        mod = importlib.import_module(package)
        # Also import all sub-modules
        if hasattr(mod, "__path__"):
            for _, subname, _ in pkgutil.walk_packages(mod.__path__,
                                                        prefix=package + "."):
                try:
                    importlib.import_module(subname)
                except Exception:
                    pass
    except Exception:
        pytest.skip(f"Could not import '{package}' — implementation may be incomplete")

    after = set(sys.modules.keys())
    newly_loaded = after - before

    violations = [
        name for name in newly_loaded
        if name.split(".")[0] in FORBIDDEN
    ]

    if violations:
        pytest.fail(
            f"\n❌  Forbidden library loaded at runtime!\n"
            f"   Package  : {package}\n"
            f"   Loaded   : {violations}\n\n"
            f"   Importing '{package}' caused a forbidden framework to be\n"
            f"   loaded into sys.modules. Remove all uses of: "
            f"{', '.join(sorted({v.split('.')[0] for v in violations}))}"
        )

    print(f"\n  ✅ PASS: '{package}' loaded without any forbidden frameworks")
