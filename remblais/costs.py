"""Transport costs: defaults for p = 1, 2, and symbolic kernox costs.

A custom cost can be given:

- as a **Python function of kernox formulas**: it receives ``x`` (``Vi``) and
  ``y`` (``Vj``) and returns a scalar formula. This is the standard way: the
  formula is simplified, differentiated and compiled by kernox/symox for the
  ``online`` backend, which ``backend="auto"`` then always selects;
- as a ``(formula, dense routine)`` pair (the GeomLoss multiscale convention):
  the dense routine, on (B, N, D), (B, M, D) tensors, serves the
  ``tensorized`` and ``multiscale`` backends;
- with ``backend="tensorized"``, as a Python function on tensors returning the
  (B, N, M) cost matrix (GeomLoss semantics);
- as a `remblais.Cost`.

KeOps formula strings (``"Norm2(X-Y)"``) are not supported: write the cost as
a kernox formula instead, e.g. ``lambda x, y: (x - y).norm2()``.
"""

from ._kp import DEFAULT_COSTS, Cost, sqdist
from .utils import distances, squared_distances

_DENSE_DEFAULTS = {1: lambda x, y: distances(x, y), 2: lambda x, y: squared_distances(x, y) / 2}

def _formula_only(fn):
    def dense(x, y):
        raise ValueError("this cost is a kernox formula: use backend='online' (or 'auto'), "
                         "or provide a dense routine as a (formula, routine) pair")
    return Cost(("callable", fn), fn, dense)


def resolve_cost(cost, p, backend):
    """Normalizes the ``cost`` argument of SamplesLoss into a `Cost`."""
    if isinstance(cost, Cost):
        return cost
    if cost is None:
        if p not in DEFAULT_COSTS:
            raise ValueError(f"p = {p} is not supported without a custom cost (p = 1 or 2)")
        if p == 2:  # ½‖x − y‖²: the ½ goes into the parameter s = 1/(2ε)
            return Cost(("p", p), DEFAULT_COSTS[p], _DENSE_DEFAULTS[p], kernel_formula=sqdist, factor=0.5)
        return Cost(("p", p), DEFAULT_COSTS[p], _DENSE_DEFAULTS[p])
    if isinstance(cost, str):
        raise TypeError(f"cost={cost!r}: KeOps formula strings are not supported; give the cost as a "
                        "function of kernox formulas, e.g. cost=lambda x, y: (x - y).norm2()")
    if isinstance(cost, tuple) and len(cost) == 2:
        formula, routine = cost
        if isinstance(formula, str):
            resolve_cost(formula, p, backend)  # same error as above
        return Cost(("callable", formula, "dense", routine), formula, routine)
    if callable(cost):
        if backend == "tensorized":
            return Cost(("dense", cost), None, cost)
        return _formula_only(cost)
    raise TypeError(f"cost: expected a function, a (formula, routine) pair or a remblais.Cost, got {cost!r}")


def is_symbolic_only(cost):
    """True when the cost has no dense routine (a function of kernox formulas,
    or a Cost without ``dense``): only the ``online`` backend can use it."""
    if isinstance(cost, Cost):
        return cost.dense is None
    return callable(cost) and not isinstance(cost, tuple)


__all__ = ["Cost", "resolve_cost", "is_symbolic_only"]
