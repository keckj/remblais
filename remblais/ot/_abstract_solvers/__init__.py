# Derived from GeomLoss v0.3.1 (geomloss/ot/_abstract_solvers/__init__.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
from .sinkhorn_ot import sinkhorn_loop
from .sinkhorn_barycenters import sinkhorn_barycenter_loop
from .annealing import max_diameter, annealing_parameters
