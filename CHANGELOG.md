# Changelog

All notable changes to remblais. Versions follow
[semantic versioning](https://semver.org/): a major version may break existing
code, a minor version only adds features. Each section below is also the text
of the corresponding [GitHub release](https://github.com/keckj/remblais/releases).

## Unreleased

### Added

- `tools/bench_geomloss.py`: benchmark against GeomLoss on a GPU or on the
  CPU, one fresh process and one fresh PyKeOps build folder per measurement.
- Documentation: measurements on an RTX 5090 (× 1.15 to × 9 against GeomLoss)
  and on 2 × AMD EPYC 9654 (× 1.3 to × 22); speed-ups are now given as
  GeomLoss time / remblais time.

## 2.1.0 (2026-11-20)

remblais now runs fast on CPUs, through the new native CPU backend of kernox
1.1, and gets a `flush_denormals` option.

### Added

- **Fast CPU execution.** On CPU tensors, the `online` and `multiscale`
  backends run on the [CPU backend of kernox](https://keckj.github.io/kernox/backend-cpu/):
  each formula is compiled to C++ that uses every core (OpenMP) and the
  vector units of the processor. A Sinkhorn divergence between two clouds of
  10⁴ points in 3D, forward and backward, on 8 threads of a Xeon:

  | Backend | 2.0.0 | 2.1.0 |
  | --- | --- | --- |
  | `online` | 31 s (kernox oracle) | 1.0 s |
  | `tensorized` (dense PyTorch) | 19 s | 19 s |

  It needs an x86-64 Linux machine with `g++` 12 or later; the first call of
  each formula compiles it (a few seconds), later calls and later sessions
  reuse it. Elsewhere, remblais falls back to the kernox oracle, as before.
  `REMBLAIS_CPU_BACKEND=cpu` or `oracle` forces the choice.
- **`SamplesLoss(..., flush_denormals=None)`.** Subnormal numbers (below about
  10⁻³⁸ in FP32) are many times slower to compute on CPUs, and narrow
  Gaussian kernels produce many of them. They do not change the losses and
  gradients of remblais in practice.
  - `None` (default): flushed to zero on the CPU and on NVIDIA GPUs in FP32,
    as PyKeOps' `-use_fast_math` does under GeomLoss; kept elsewhere.
  - `False`: kept everywhere, as IEEE arithmetic requires.
  - `True`: asks kernox to flush them explicitly (refused on AMD and Intel
    GPUs).

  The Gaussian kernel sums behind a `gaussian` loss with a small blur run up
  to 4.8 times faster on an Intel Xeon (1.5 on an AMD EPYC). Sinkhorn losses are not affected: their
  log-sum-exp produces no subnormal numbers. See
  [Subnormal numbers](https://keckj.github.io/remblais/geomloss/#subnormal-numbers).
- Documentation: a table from KeOps cost strings to kernox formulas in
  [Coming from GeomLoss](https://keckj.github.io/remblais/geomloss/).

### Changed

- Requires kernox ≥ 1.1.0.
- On NVIDIA GPUs in FP32 and on the CPU, subnormal numbers are now flushed
  to zero by default, as PyKeOps does (remblais 2.0 kept them). Losses and
  gradients do not change in practice; `flush_denormals=False` restores the
  2.0 behaviour.

## 2.0.0 (2026-11-18)

Custom costs are written as kernox formulas instead of KeOps formula strings.

### Breaking changes

- **KeOps formula strings are no longer accepted.** `cost="..."` raises a
  `TypeError` that points to kernox formulas, e.g.
  `cost=lambda x, y: (x - y).norm2()`. The formula is simplified,
  differentiated and compiled by kernox and symox, so any formula kernox
  supports can be used as a cost.
- `remblais.keops_string_cost` is removed, with its parser and its function
  table.

### Changed

- `backend="auto"` selects the `online` backend for a custom cost without a
  dense implementation, whatever the problem size.
- The `tensorized` and `multiscale` backends take a pair (kernox formula,
  dense PyTorch function); a plain function on tensors still works with
  `backend="tensorized"`.
- Documentation: Intel GPU users are pointed to the SYCL runtime step of the
  kernox installation.

### Upgrading

Replace each KeOps string by a function of kernox formulas:
`cost="SqDist(X,Y)"` becomes `cost=lambda x, y: x.sqdist(y)`. The
[GeomLoss migration page](https://keckj.github.io/remblais/geomloss/) lists
the translation of each KeOps operation.

## 1.0.1 (2026-11-15)

remblais is installable from PyPI: `pip install remblais`, with kernox and
symox from PyPI too (no git install, no Rust toolchain).

### Changed

- kernox ≥ 1.0.1 is a regular dependency installed from PyPI (symox comes
  with it).
- All messages, comments, docstrings and tests are in English.
- Documentation: the GPUs remblais runs on (NVIDIA, AMD and Intel: CUDA,
  HIP, SYCL), and `ImagesBarycenter` returns an unnormalized image, as
  GeomLoss does.

### Added

- Release workflow: each version is tested, built, published to PyPI through
  Trusted Publishing with build provenance attestations (PEP 740), and
  released on GitHub.

## 1.0.0 (2026-11-15)

First release. remblais computes geometric losses between measures on GPUs:
Sinkhorn divergences (entropic optimal transport), kernel norms (MMD) and
Hausdorff divergences between point clouds, images and volumes. It is a
drop-in replacement for [GeomLoss](https://www.kernel-operations.io/geomloss/)
0.3.1 built on kernox instead of PyKeOps: `import remblais as geomloss`.

- Same API as GeomLoss: `SamplesLoss`, `sinkhorn_divergence`,
  `ImagesBarycenter` and the `ot` module, with the same signatures and
  outputs.
- Faster: each softmin is one fused kernox reduction, compiled once for all
  values of ε; the debiased Sinkhorn loop fuses its four softmins into two
  reductions per iteration.
- NVIDIA, AMD and Intel GPUs, through kernox.
- A few GeomLoss bugs fixed (gradient of unbalanced transport, `"hausdorff"`
  without a kernel, …), each listed in Coming from GeomLoss.
