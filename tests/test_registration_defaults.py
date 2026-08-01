from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_registration_module():
    source = PROJECT_ROOT / "registration" / "2021.py"
    spec = importlib.util.spec_from_file_location("registration_2021_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hmm_and_framework_use_paper_sigma_defaults():
    module = load_registration_module()

    hmm = module.HMMPoseEstimator()
    framework = module.LUSCTRegistrationFramework()

    assert (hmm.sigma_x, hmm.sigma_y, hmm.sigma_z, hmm.sigma_theta) == (
        0.6,
        0.6,
        3.0,
        2.0,
    )
    assert (
        framework.sigma_x,
        framework.sigma_y,
        framework.sigma_z,
        framework.sigma_theta,
    ) == (0.6, 0.6, 3.0, 2.0)
