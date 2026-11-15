# Third-party code and algorithms

remblais is dual-licensed under MIT or Apache-2.0, at your option
([`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE)). Part of
its code is derived from GeomLoss, which is released under the MIT license:
that code remains under the MIT license, with the notice below. The table
lists the code and algorithms taken from third-party projects; the files
copied from GeomLoss start with an attribution header.

| remblais file | Source | Revision | License | Nature |
| --- | --- | --- | --- | --- |
| `remblais/samples_loss.py` | GeomLoss `src/geomloss/_legacy/samples_loss.py` (jeanfeydy/geomloss) | v0.3.1 (tag `6a29db8`, identical to the PyPI package) | MIT | `SamplesLoss` interface (signature, calling conventions, shape checks and error messages, `backend="auto"` selection rule) taken and adapted |
| `remblais/sinkhorn_divergence.py` | GeomLoss `src/geomloss/_legacy/sinkhorn_divergence.py` | v0.3.1 | MIT | symmetric Sinkhorn loop, ε-scaling, debiased and unbalanced formulas, taken and adapted (grad state restored, unbalanced gradient fixed) |
| `remblais/sinkhorn_samples.py` | GeomLoss `src/geomloss/_legacy/sinkhorn_samples.py` | v0.3.1 | MIT | tensorized / online / multiscale backends (truncation, extrapolation); softmins rewritten on kernox |
| `remblais/kernel_samples.py` | GeomLoss `src/geomloss/_legacy/kernel_samples.py` | v0.3.1 | MIT | kernel norms, `DoubleGrad`, truncated multiscale; products rewritten on kernox |
| `remblais/utils.py` | GeomLoss `src/geomloss/_legacy/utils.py` | v0.3.1 | MIT | `scal`, dense distances, diameter |
| `remblais/clustering.py` | PyKeOps `pykeops/torch/cluster/{grid_cluster,matrix}.py` (getkeops/keops) | `ef72bc907df78cd18bff866d3dc7cb6375f11a4d` | MIT | semantics of `grid_cluster`, `cluster_ranges_centroids`, `sort_clusters`, `from_matrix`; rewritten in Torch (no line copied) |
| `remblais/ot/**`, `remblais/_backends/*` (except `keops.py`, rewritten on `remblais.lazy`), `remblais/_input_validation/*`, `remblais/_arguments.py`, `remblais/_typing.py`, `remblais/_cache.py` | GeomLoss `src/geomloss/` (same paths) | v0.3.1 | MIT | copied with an attribution header; the `barycenter*` and `solve_grid` skeletons replaced by explicit `NotImplementedError`s, grid softmins redirected to `remblais.sinkhorn_images` |
| `tests/ot/**` | GeomLoss `tests/` (`check_ot_result.py`, `generators/`, `test_ot_solve_matrix.py`, `test_ot_solve_sample.py`) | v0.3.1 | MIT | copied, `geomloss` replaced by `remblais` in the imports |
| `remblais/sinkhorn_images.py`, `remblais/wasserstein_barycenter_images.py` | GeomLoss `src/geomloss/_legacy/{sinkhorn_images,wasserstein_barycenter_images,utils}.py` | v0.3.1 | MIT | algorithms taken; softmin and C-transform rewritten on kernox |
| `tests/test_samples_loss.py` | GeomLoss (reference behavior, run from the PyPI package `geomloss==0.3.1`) | 0.3.1 | MIT | no line copied |

## GeomLoss notice

```
MIT License

Copyright (c) 2019 Jean Feydy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The KeOps license (MIT) is in its repository
(<https://github.com/getkeops/keops>, `licence.txt`); see also the
`THIRD_PARTY.md` file of kernox (<https://github.com/keckj/kernox>).
