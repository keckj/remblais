# Python API reference

Generated from the docstrings of the public API. For explanations and
examples, start with [Concepts](concepts.md) and the how-to guides.

## Losses between point clouds

::: remblais.SamplesLoss
    options:
      members: false

## Images and volumes

The function below is exported as `remblais.sinkhorn_divergence`, the name
it has in GeomLoss.

::: remblais.sinkhorn_images.sinkhorn_divergence

::: remblais.ImagesBarycenter

## Custom costs and kernels

::: remblais.keops_string_cost

::: remblais.Cost
    options:
      members: false

::: remblais.Kernel
    options:
      members: false

## The ot module

The docstrings of this module come from GeomLoss 0.3.1; their references to
`geomloss.ot` apply to `remblais.ot`.

::: remblais.ot.solve
    options:
      docstring_style: numpy

::: remblais.ot.solve_batch
    options:
      docstring_style: numpy

::: remblais.ot.solve_sample
    options:
      docstring_style: numpy

::: remblais.ot.OTResult
    options:
      docstring_style: numpy

::: remblais.ot.OTResultMatrix
    options:
      docstring_style: numpy

::: remblais.ot.OTResultSample
    options:
      docstring_style: numpy

::: remblais.ot.LinearOperator
    options:
      docstring_style: numpy

## LazyTensor

::: remblais.lazy
    options:
      members: false
