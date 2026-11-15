# Derived from GeomLoss v0.3.1 (geomloss/ot/__init__.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
from ._implementations.matrix import solve, solve_batch, barycenter, OTResultMatrix
from ._implementations.sample import (
    solve_sample,
    solve_sample_batch,
    barycenter_sample,
    OTResultSample,
)
from ._implementations.grid import solve_grid, barycenter_grid
from ._ot_result import OTResult, LinearOperator

__all__ = sorted(
    [
        "solve",
        "solve_batch",
        "barycenter",
        "OTResultMatrix",
        "solve_sample",
        "solve_sample_batch",
        "barycenter_sample",
        "OTResultSample",
        "solve_grid",
        "barycenter_grid",
        "OTResult",
        "LinearOperator",
    ]
)
