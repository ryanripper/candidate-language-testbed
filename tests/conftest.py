"""Shared test fixtures and import paths.

The repo is a research archive, not a package, so tests import the two
maintained modules by path: ws0-harness/metrics.py and
data/synthetic-candidate-tweets/generate_synthetic_candidates.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# metrics.py lives in ws0-harness/ (added to sys.path via pyproject
# pythonpath as well; this keeps direct pytest invocations working from
# any cwd).
sys.path.insert(0, str(ROOT / "ws0-harness"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def gen():
    """The corpus generator module (directory name has hyphens, so it is
    loaded by file path)."""
    return _load_module(
        "generate_synthetic_candidates",
        ROOT / "data" / "synthetic-candidate-tweets" / "generate_synthetic_candidates.py",
    )


@pytest.fixture()
def fresh_gen(gen):
    """The generator module with its RNG re-seeded to the shipped SEED, so
    each test sees the exact random stream the pinned corpus was built
    from."""
    import random

    gen.rng = random.Random(gen.SEED)
    return gen


@pytest.fixture(scope="session")
def corpus_path():
    return ROOT / "data" / "synthetic-candidate-tweets" / "synthetic_candidate_tweets_2022.csv.gz"


@pytest.fixture(scope="session")
def manifest_path():
    return ROOT / "ws0-harness" / "seal_manifest.json"
