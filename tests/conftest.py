from pathlib import Path

import numpy as np
import pytest

BASELINE_DIR = Path(__file__).parent.parent / "verifications" / "baseline_v1"


@pytest.fixture(scope="session")
def orion():
    from pnicer.demo import load_orion

    return load_orion()


@pytest.fixture(scope="session")
def control():
    from pnicer.demo import load_control

    return load_control()


@pytest.fixture(scope="session")
def baseline_dir():
    if not BASELINE_DIR.exists():
        pytest.skip("Legacy baseline outputs not available on this machine")
    return BASELINE_DIR


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260702)
