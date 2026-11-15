# remblais

<p align="center">
  <img src="docs/images/remblais-banner.png" alt="remblais logo" width="560">
</p>

remblais computes geometric losses between measures on NVIDIA (CUDA), AMD
(ROCm/HIP) and Intel (SYCL) GPUs: Sinkhorn divergences (entropic optimal
transport), kernel norms (MMD) and Hausdorff divergences between point clouds,
images and volumes. It is a drop-in
replacement for [GeomLoss](https://www.kernel-operations.io/geomloss/) 0.3.1,
built on [kernox](https://github.com/keckj/kernox) instead of PyKeOps: write
`import remblais as geomloss` and your code runs unchanged.

- Documentation: <https://keckj.github.io/remblais/>
- Getting started: <https://keckj.github.io/remblais/getting-started/>

## Why remblais

- **Same API as GeomLoss.** `SamplesLoss`, `sinkhorn_divergence`,
  `ImagesBarycenter` and the `ot` module have the GeomLoss 0.3.1 signatures,
  calling conventions and output shapes. With `backend="tensorized"`, values
  and gradients match GeomLoss bit for bit, apart from the documented fixes.
- **Faster.** Every softmin is one fused kernox reduction, compiled once and
  reused for all values of ε; the debiased Sinkhorn loop fuses its four
  softmins into two reductions per iteration.
- **NVIDIA, AMD and Intel GPUs.** The `online` and `multiscale` backends,
  which GeomLoss runs through PyKeOps on NVIDIA GPUs only, run here on CUDA,
  ROCm/HIP and SYCL through kernox. remblais's own test suite runs on NVIDIA
  and Intel GPUs; on AMD GPUs it relies on kernox's tested HIP backend.
- **Fixes, not surprises.** A few GeomLoss bugs are fixed (the gradient of
  unbalanced transport, `"hausdorff"` without a kernel, ...). Every
  difference is listed in [Coming from GeomLoss](docs/geomloss.md).

## Installation

remblais needs Python 3.10 or later, PyTorch and kernox. Install PyTorch for
your platform, kernox from PyPI with the extra that matches your GPU, then
remblais from GitHub. For an NVIDIA GPU:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "kernox[cuda]"
pip install "remblais @ git+https://github.com/keckj/remblais"
```

For AMD and Intel GPUs, install PyTorch from the `rocm` or `xpu` index and
kernox with the `rocm` or `xpu` extra; the details are in the
[kernox installation guide](https://keckj.github.io/kernox/getting-started/)
and in [Getting started](docs/getting-started.md).

## Performance

The goal of remblais is to be **at least as fast as GeomLoss** wherever
GeomLoss runs. In the cases measured so far, it is already faster.

Sinkhorn divergence, p = 2, FP32, forward and backward passes, on an RTX 2080
Ti; remblais with `fast_math`, its default setting, against GeomLoss 0.3.1 in
its best configuration (each PyKeOps kernel compiled alone in a fresh
process). Time ratios remblais / GeomLoss, lower than 1 when remblais is
faster:

| Case | N = 10⁴ | N = 10⁵ | N = 10⁶ |
| --- | --- | --- | --- |
| multiscale, D = 2 | ×0.45 | ×0.40 | ×0.79 |
| multiscale, D = 3 | ×0.26 | ×0.92 | ×0.96 |
| online, D = 2 | ×0.33 | ×0.84 | ×0.89 |
| online, D = 3 | ×0.31 | ×0.85 | ×0.90 |
| online, D = 64 | ×0.45 | ×0.54 | not measured |

For small clouds (N = 10³ to 3·10⁴), the online backend runs at ×0.31 to
×0.61 of the GeomLoss time in D = 2 and 3, and at ×0.17 to ×0.45 in D = 64.

These figures cover only the cases listed; they are not a general claim of
superiority over GeomLoss on every workload or card. The full tables and the
protocol are on the [Performance](docs/performance.md) page.

## About the name and logo

In 1781, Gaspard Monge asked how to move a pile of earth, the *déblais*, into
a hole of the same volume, the *remblais*, at the lowest cost: the founding
problem of optimal transport. remblais is where the mass ends up. The logo
shows the ox of symox and kernox, harnessed this time to a scraper, carrying
the déblais to the remblais: Monge's optimal transport.

## License

At your option, [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Code
derived from GeomLoss stays under the MIT license of GeomLoss; its provenance
and notice are in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Related projects

- GeomLoss, the original library: <https://github.com/jeanfeydy/geomloss>
- kernox, fused GPU reductions: <https://github.com/keckj/kernox>
- symox, symbolic engine and code generation: <https://github.com/keckj/symox>
