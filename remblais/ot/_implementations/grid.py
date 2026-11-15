# Derived from GeomLoss v0.3.1 (geomloss/ot/_implementations/grid.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
from ...sinkhorn_images import (  # noqa: F401  (implémentations kernox)
    C_transform, dimension, log_dens, pyramid, softmin_grid, subsample, upsample, upsample_mode)

_STUB = ("{} : simple squelette dans GeomLoss 0.3.1 (aucune implémentation) ; non fourni par remblais. "
         "Pour des images, voir remblais.sinkhorn_divergence et remblais.ImagesBarycenter.")


def solve_grid(*args, **kwargs):
    """Skeleton in GeomLoss 0.3.1: raises ``NotImplementedError``."""
    raise NotImplementedError(_STUB.format("ot.solve_grid"))


def barycenter_grid(*args, **kwargs):
    """Skeleton in GeomLoss 0.3.1: raises ``NotImplementedError``."""
    raise NotImplementedError(_STUB.format("ot.barycenter_grid"))
