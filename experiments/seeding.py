"""Deterministic seeding for EdgeAIBus verification runs.

The Datacenter and SimEdgeEnv both consume global ``random`` / ``numpy`` state:
- ``DatacenterGeneration.__init__`` draws ``containers_model`` via
  ``np.random.randint`` (the original ``np.random.seed``/``random.seed`` lines
  are commented out in Datacenter.py).
- ``SimEdgeEnv.__init__`` and ``SimEdgeEnv.step`` draw ``yolo_random_samples``
  via ``random.sample`` (global module state).

Historical runs (author PT runs and earlier ad-hoc evals) were **unseeded**;
we keep those results frozen. New verification runs MUST be seeded so that
arms are comparable run-to-run. Seeding is symmetric across all arms
(PatchTST / DLinear / no-predictor), so it does not bias one arm over another:
we only compare seeded-vs-seeded or historical-vs-historical.

This module provides one entry point that seeds every consumer we use, and
must be called BEFORE constructing a Datacenter (i.e. before
``DatacenterGeneration(config)``).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: Optional[int] = 42) -> int:
    """Seed all global RNG consumers involved in EdgeAIBus runs.

    Call this before ``DatacenterGeneration`` / ``SimEdgeEnv`` creation so the
    initial ``containers_model`` and per-step ``yolo_random_samples`` are
    reproducible.  Returns the resolved seed.

    Notes
    -----
    - ``torch`` is seeded too when importable (only used when training the
      DLinear forecaster / policies).
    - ``PYTHONHASHSEED`` cannot be changed after interpreter start; we set it
      anyway for spawned subprocesses (harmless if already fixed).
    """
    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved)
    os.environ['PYTHONHASHSEED'] = str(resolved)
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(resolved)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved)
    except ImportError:
        pass
    return resolved
