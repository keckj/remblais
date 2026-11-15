"""Grid clustering and block-sparse ranges for the multiscale backend.

Functional equivalents of ``pykeops.torch.cluster`` (``grid_cluster``,
``cluster_ranges_centroids``, ``sort_clusters``, ``from_matrix``), rewritten in
Torch; the ranges are built by :func:`kernox.ranges_from_labels`.
"""

import math
import os

import torch

import kernox as kp


def _dense_rank(key):
    """Dense rank (0 … C−1) of each integer of ``key``, by sorting: equivalent to
    ``torch.unique(key, sorted=True, return_inverse=True)[1]`` without ``unique``
    (broken on XPU with Torch 2.15: "-fsycl-id-queries-range=int")."""
    sk, perm = torch.sort(key)
    new = torch.ones_like(sk)
    new[1:] = (sk[1:] != sk[:-1]).to(sk.dtype)
    rank = torch.cumsum(new, 0) - 1
    lab = torch.empty_like(rank)
    lab[perm] = rank
    return lab


def grid_cluster(x, size):
    """Labels each point of ``x`` (N, D) with its cell in a grid of step ``size``.

    The labels are contiguous (0 … C−1) and follow the lexicographic order of the
    cells, like ``pykeops.torch.cluster.grid_cluster`` (without its D ≤ 3 limit
    or the risk of collision of its 32-bit integer encoding).
    """
    with torch.no_grad():
        cells = torch.floor(x / size).to(torch.int64)
        cells = cells - cells.min(dim=0).values
        ext = (cells.max(dim=0).values + 1).tolist()
        if math.prod(ext) < 2**62:  # code entier lexicographique exact : unique 1D (tous devices)
            key = torch.zeros(len(cells), dtype=torch.int64, device=x.device)
            for d in range(cells.shape[1]):
                key = key * ext[d] + cells[:, d]
            lab = _dense_rank(key)
        else:  # très grand nombre de cellules : unique lexicographique sur CPU
            _, lab = torch.unique(cells.cpu(), dim=0, sorted=True, return_inverse=True)
            lab = lab.to(x.device)
    return lab.view(-1)


def cluster_ranges_centroids(x, lab, weights=None, nlab=None):
    """(C, 2) ranges of the clusters (points assumed sorted by label), centroids, masses.

    Centroid = barycenter weighted by ``weights`` (uniform otherwise), mass = sum
    of the weights. An empty or zero-mass cluster has the unweighted mean of its
    points as centroid (0 if empty) and a zero mass; a singleton has its point as
    centroid, exactly (no rounding of x·w/w).
    """
    with torch.no_grad():
        lab = lab.view(-1)
        C = int(lab.max().item()) + 1 if nlab is None else nlab
        # pas de bincount : faux sur XPU (Torch 2.15), d'où des ranges hors bornes
        counts = torch.zeros(C, dtype=torch.int64, device=lab.device).index_add_(0, lab, torch.ones_like(lab))
        ends = torch.cumsum(counts, 0)
        ranges = torch.stack((ends - counts, ends), dim=1)
    w = torch.ones(len(x), dtype=x.dtype, device=x.device) if weights is None else weights
    w_c = torch.zeros(C, dtype=x.dtype, device=x.device).index_add(0, lab, w)
    with torch.no_grad():
        D = x.shape[1]
        sw = torch.zeros(C, D, dtype=x.dtype, device=x.device).index_add(0, lab, x * w.view(-1, 1))
        su = torch.zeros(C, D, dtype=x.dtype, device=x.device).index_add(0, lab, x)
        cnt = counts.to(x.dtype).clamp_min(1).view(-1, 1)
        wc = w_c.view(-1, 1)
        x_c = torch.where(wc > 0, sw / torch.where(wc > 0, wc, torch.ones_like(wc)), su / cnt)
        # singleton : le point lui-même, par indexation (exact sur tout device ;
        # une somme atomique index_add n'est pas garantie exacte partout)
        first = x[ranges[:, 0].clamp(max=len(x) - 1)]
        x_c = torch.where(counts.view(-1, 1) == 1, first, x_c)
    return ranges, x_c, w_c


def sort_clusters(x, lab):
    """Sorts ``x`` (a tensor or a tuple of tensors) by label; returns (sorted x, sorted labels)."""
    lab_sorted, perm = torch.sort(lab.view(-1), stable=True)
    if isinstance(x, tuple):
        return tuple(t[perm] for t in x), lab_sorted
    return x[perm], lab_sorted


class Clusters:
    """Points sorted by cluster: sorted labels, ranges, centroids, masses, permutation."""

    def __init__(self, weights, points, scale=None, labels=None):
        lab = grid_cluster(points, scale) if labels is None else labels.view(-1).to(torch.int64)
        lab_sorted, perm = torch.sort(lab, stable=True)
        self.perm = perm
        self.weights = weights[perm]
        self.points = points[perm]
        self.labels = lab_sorted
        self.ranges, self.centroids, self.masses = cluster_ranges_centroids(
            self.points, lab_sorted, weights=self.weights)

    def __len__(self):
        return len(self.centroids)


def from_matrix(clusters_i, clusters_j, keep):
    """kernox ranges (two triplets) of the blocks (ci, cj) such that ``keep[ci, cj]``."""
    return kp.ranges_from_labels(clusters_i.labels, clusters_j.labels, keep)


# Repli dense des niveaux fins multiscale : si le coût estimé des blocs gardés
# dépasse cette fraction de N·M, on lance la réduction sans ranges (toutes les
# paires, résultat plus exact que la troncature). REMBLAIS_DENSE_RATIO=inf garde
# toujours les ranges (comportement de GeomLoss).
# Seuil calibré sur RTX 2080 Ti (D = 3, blur = 0,05, 10⁴ à 3·10⁵ points) : coût réel
# des ranges ≈ 1,3 à 1,7 × l'estimation ; équilibre vers 0,75.
DENSE_RATIO = float(os.environ.get("REMBLAIS_DENSE_RATIO", "0.75"))
_TILE_M = 128  # hauteur de tuile de lignes des noyaux ranges (modèle de coût)


def ranged_cost_ratio(clusters_i, clusters_j, keep):
    """Estimated cost of the kept blocks, relative to N·M.

    Each cluster i takes ⌈n_i / T⌉ tiles of T rows: a cluster of 5 points costs
    as much as T rows. Σ_ij keep_ij · T⌈n_i/T⌉ · n_j / (N·M).
    """
    with torch.no_grad():
        ni = (clusters_i.ranges[:, 1] - clusters_i.ranges[:, 0]).to(torch.float64)
        nj = (clusters_j.ranges[:, 1] - clusters_j.ranges[:, 0]).to(torch.float64)
        padded = torch.ceil(ni / _TILE_M) * _TILE_M
        est = (padded * (keep.to(torch.float64) @ nj)).sum()
        return float(est) / max(1.0, float(ni.sum()) * float(nj.sum()))


def from_matrix_or_dense(clusters_i, clusters_j, keep, ratio=None):
    """Ranges of the kept blocks, or ``None`` (all pairs) if the dense fallback is cheaper."""
    ratio = ranged_cost_ratio(clusters_i, clusters_j, keep) if ratio is None else ratio
    if ratio > DENSE_RATIO:
        return None
    return from_matrix(clusters_i, clusters_j, keep)
