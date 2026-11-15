"""Clusters sur grille : étiquettes, ranges et centroïdes identiques sur tous les devices."""

import pytest
import torch

from remblais.clustering import Clusters, cluster_ranges_centroids, grid_cluster

from .conftest import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("N,D,scale", [(300, 2, 0.07), (2000, 3, 0.1), (5000, 2, 0.01)])
def test_ranges_match_cpu(device, N, D, scale):
    g = torch.Generator().manual_seed(N)
    x = torch.rand(N, D, generator=g, dtype=torch.float64)
    w = torch.rand(N, generator=g, dtype=torch.float64)
    ref = Clusters(w, x, labels=grid_cluster(x, scale))
    got = Clusters(w.to(device), x.to(device), labels=grid_cluster(x.to(device), scale))
    assert torch.equal(got.labels.cpu(), ref.labels)
    assert torch.equal(got.ranges.cpu(), ref.ranges)
    # ranges : partition contiguë de 0 … N
    r = got.ranges.cpu()
    assert r[0, 0] == 0 and r[-1, 1] == N and torch.equal(r[1:, 0], r[:-1, 1]) and bool((r[:, 1] > r[:, 0]).all())
    counts = torch.bincount(ref.labels)  # sur CPU : référence
    assert torch.equal(r[:, 1] - r[:, 0], counts)
    torch.testing.assert_close(got.centroids.cpu(), ref.centroids, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(got.masses.cpu(), ref.masses, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_counts_with_empty_clusters(device):
    lab = torch.tensor([0, 0, 2, 2, 2, 5], device=device)
    x = torch.arange(6, dtype=torch.float64, device=device).view(-1, 1)
    ranges, x_c, w_c = cluster_ranges_centroids(x, lab, nlab=7)
    assert ranges.cpu().tolist() == [[0, 2], [2, 2], [2, 5], [5, 5], [5, 5], [5, 6], [6, 6]]
    assert x_c.view(-1).cpu().tolist() == [0.5, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0]


def test_ranged_cost_ratio():
    """Gros clusters : rapport ≈ fraction gardée ; petits clusters : pénalité de tuile."""
    from remblais.clustering import _TILE_M, ranged_cost_ratio

    class C:
        def __init__(self, sizes):
            e = torch.cumsum(torch.tensor(sizes), 0)
            self.ranges = torch.stack((e - torch.tensor(sizes), e), 1)

    big = C([4 * _TILE_M] * 4)
    keep = torch.eye(4, dtype=torch.bool)
    assert ranged_cost_ratio(big, big, keep) == pytest.approx(0.25)
    small = C([4] * 64)
    keep = torch.eye(64, dtype=torch.bool)
    assert ranged_cost_ratio(small, small, keep) == pytest.approx(_TILE_M / 4 / 64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loss", ["sinkhorn", "gaussian"])
def test_dense_fallback(device, loss, monkeypatch):
    """Repli dense des niveaux fins : même perte qu'avec les ranges, à la troncature près
    (exactement la perte en ligne pour les noyaux)."""
    import remblais.clustering as cl
    from remblais import SamplesLoss

    from .conftest import measures

    a, x, b, y = measures(N=300, M=250, D=2, device=device, grad=True)
    vals = {}
    for ratio in (0.0, float("inf")):
        monkeypatch.setattr(cl, "DENSE_RATIO", ratio)
        v = SamplesLoss(loss, blur=0.05, backend="multiscale", truncate=3)(a, x, b, y)
        vals[ratio] = (v.detach(), torch.autograd.grad(v, [x])[0])
    online = SamplesLoss(loss, blur=0.05, backend="online")(a, x, b, y)
    torch.testing.assert_close(vals[0.0][0], vals[float("inf")][0], rtol=1e-3, atol=1e-6)
    if loss == "gaussian":
        torch.testing.assert_close(vals[0.0][0], online.detach(), rtol=1e-12, atol=1e-14)
