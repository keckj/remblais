# Derived from GeomLoss v0.3.1 (geomloss/ot/_implementations/grid.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
from ...sinkhorn_images import (  # noqa: F401  (kernox implementations)
    C_transform, dimension, log_dens, pyramid, softmin_grid, subsample, upsample, upsample_mode)

_STUB = ("{}: only a skeleton in GeomLoss 0.3.1 (no implementation); not provided by remblais. "
         "For images, see remblais.sinkhorn_divergence and remblais.ImagesBarycenter.")


def solve_grid(*args, **kwargs):
    """Skeleton in GeomLoss 0.3.1: raises ``NotImplementedError``."""
    raise NotImplementedError(_STUB.format("ot.solve_grid"))


def barycenter_grid(*args, **kwargs):
    """Skeleton in GeomLoss 0.3.1: raises ``NotImplementedError``."""
    raise NotImplementedError(_STUB.format("ot.barycenter_grid"))
