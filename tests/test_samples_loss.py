"""SamplesLoss: parity with GeomLoss v0.3.1 and accuracy of the kernox backends.

References:
- installed GeomLoss 0.3.1 (tensorized backend, pure Torch): bit-identical
  values and gradients for the tensorized backend of remblais;
- dense FP64 oracle (tensorized backend of remblais, on exact matrices) for the
  online and multiscale backends; on the CPU, kernox runs its blockwise Torch
  oracle, on GPUs the native kernels.
"""

import pytest
import torch

import remblais
from remblais import SamplesLoss

from .conftest import measures

try:
    import geomloss
except ImportError:  # pragma: no cover
    geomloss = None

needs_geomloss = pytest.mark.skipif(geomloss is None, reason="geomloss 0.3.1 not installed")

CASES = [
    dict(),
    dict(p=1),
    dict(reach=0.3),
    dict(reach=0.3, debias=False),
    dict(debias=False),
    dict(blur=0.01, scaling=0.9),
    dict(blur=1.0),
    dict(loss="gaussian"),
    dict(loss="laplacian"),
    dict(loss="energy"),
    dict(loss="gaussian", blur=0.3),
]


def _sq(x, y):
    """‖x − y‖² from differences (exactly zero on the diagonal)."""
    return ((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2).sum(-1)


def _dist(x, y):
    d2 = _sq(x, y)
    pos = d2 > 0
    return torch.where(pos, torch.where(pos, d2, torch.ones_like(d2)).sqrt(), torch.zeros_like(d2))


def exact_reference(kw):
    """Tensorized FP64 SamplesLoss whose distances are exactly 0 on the diagonal.

    GeomLoss tensorized puts a 1e-8 floor under the square root (distance 1e-4
    at x = y); KeOps and kernox compute ‖x − y‖ = 0 exactly, so the reference
    for the online backends follows the KeOps convention.
    """
    kw = dict(kw)
    loss, p = kw.get("loss", "sinkhorn"), kw.get("p", 2)
    if loss == "sinkhorn" and p == 1:
        kw["cost"] = _dist
    elif loss == "sinkhorn" and p == 2:
        kw["cost"] = lambda x, y: _sq(x, y) / 2
    elif loss == "laplacian":
        kw["loss"], kw["kernel"] = "hausdorff", lambda x, y, blur, **_: (-_dist(x / blur, y / blur)).exp()
    elif loss == "energy":
        kw["loss"], kw["kernel"] = "hausdorff", lambda x, y, **_: -_dist(x, y)
    return SamplesLoss(backend="tensorized", **kw)


def _value_and_grads(loss, inputs):
    out = loss(*inputs)
    if isinstance(out, tuple):
        s = (out[0] ** 2).sum() + (out[1] * out[1].detach().cos()).sum()
        val = torch.cat([o.reshape(-1) for o in out])
    else:
        s, val = out.sum(), out.reshape(-1)
    grads = torch.autograd.grad(s, inputs, allow_unused=True)
    return val.detach(), [torch.zeros_like(t) if g is None else g for g, t in zip(grads, inputs)]


def _close(u, v, rtol, atol):
    torch.testing.assert_close(u.cpu().to(torch.float64), v.cpu().to(torch.float64), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------------
# Tensorized parity with GeomLoss
# ---------------------------------------------------------------------------------
@needs_geomloss
@pytest.mark.parametrize("B", [None, 3])
@pytest.mark.parametrize("potentials", [False, True])
@pytest.mark.parametrize("kw", CASES, ids=str)
def test_tensorized_matches_geomloss(kw, potentials, B):
    inputs = measures(B=B, grad=True)
    v_ref, g_ref = _value_and_grads(geomloss.SamplesLoss(backend="tensorized", potentials=potentials, **kw), inputs)
    v, g = _value_and_grads(SamplesLoss(backend="tensorized", potentials=potentials, **kw), inputs)
    _close(v, v_ref, 0, 0)
    if "reach" in kw and not potentials:
        return  # corrected unbalanced gradient (intentional difference): see test_sinkhorn_gradient_at_convergence
    for u, w in zip(g, g_ref):
        _close(u, w, 0, 0)


@needs_geomloss
@pytest.mark.parametrize("args", ["xy", "axby", "weights_n1"])
def test_call_conventions_match_geomloss(args):
    a, x, b, y = measures()
    call = {"xy": (x, y), "axby": (a, x, b, y), "weights_n1": (a.view(-1, 1), x, b.view(-1, 1), y)}[args]
    ref = geomloss.SamplesLoss(backend="tensorized")(*call)
    out = SamplesLoss(backend="tensorized")(*call)
    assert out.shape == ref.shape == ()
    _close(out, ref, 0, 0)


# ---------------------------------------------------------------------------------
# kernox backends against the dense FP64 oracle
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("B", [None, 1, 3])
@pytest.mark.parametrize("potentials", [False, True])
@pytest.mark.parametrize("kw", CASES, ids=str)
def test_online_matches_dense_oracle(kw, potentials, B, device):
    inputs = measures(B=B, grad=True)
    v_ref, g_ref = _value_and_grads(exact_reference(dict(kw, potentials=potentials)), inputs)
    dev_inputs = [t.detach().to(device).requires_grad_() for t in inputs]
    v, g = _value_and_grads(SamplesLoss(backend="online", potentials=potentials, **kw), dev_inputs)
    _close(v, v_ref, 1e-9, 1e-12)
    for u, w in zip(g, g_ref):
        _close(u, w, 1e-8, 1e-11)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_online_float32(dtype, device):
    a, x, b, y = measures(N=300, M=200, D=3, dtype=dtype, device=device)
    ref = SamplesLoss(backend="tensorized")(*[t.double().cpu() for t in (a, x, b, y)])
    out = SamplesLoss(backend="online")(a, x, b, y)
    assert out.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 1e-11
    _close(out, ref, tol, tol * 1e-2)


@pytest.mark.parametrize("kw", [dict(), dict(reach=0.5), dict(debias=False), dict(p=1)], ids=str)
def test_multiscale_without_clustering_equals_online(kw, device):
    """One cluster per point and no truncation: multiscale reproduces the online computation exactly."""
    inputs = [t.to(device).requires_grad_() for t in measures(N=80, M=70, D=2)]
    v_ref, g_ref = _value_and_grads(SamplesLoss(backend="online", **kw), inputs)
    ms = SamplesLoss(backend="multiscale", cluster_scale=1e-9, truncate=None, **kw)
    v, g = _value_and_grads(ms, inputs)
    _close(v, v_ref, 1e-10, 1e-13)
    for u, w in zip(g, g_ref):
        _close(u, w, 1e-9, 1e-12)


@pytest.mark.parametrize("loss", ["sinkhorn", "gaussian", "laplacian"])
def test_multiscale_potentials_in_input_order(loss, device):
    a, x, b, y = [t.to(device) for t in measures(N=300, M=250, D=2)]
    kw = dict(loss=loss, blur=0.05, potentials=True)
    F_ref, G_ref = SamplesLoss(backend="online", **kw)(a, x, b, y)
    F, G = SamplesLoss(backend="multiscale", truncate=8, **kw)(a, x, b, y)
    F_ref, G_ref = F_ref.view_as(F), G_ref.view_as(G)  # (1, N): GeomLoss shape of the online backend
    scale = F_ref.abs().max()
    _close(F / scale, F_ref / scale, 0, 2e-3)
    _close(G / scale, G_ref / scale, 0, 2e-3)


def test_multiscale_close_to_exact(device):
    a, x, b, y = [t.to(device) for t in measures(N=2000, M=1500, D=2, seed=3)]
    ref = SamplesLoss(backend="online", blur=0.01)(a, x, b, y)
    out = SamplesLoss(backend="multiscale", blur=0.01)(a, x, b, y)
    assert out.shape == ()
    _close(out, ref, 2e-2, 0)


def test_multiscale_explicit_labels(device):
    a, x, b, y = [t.to(device) for t in measures(N=400, M=300, D=2, seed=4)]
    l_x = (x[:, 0] * 4).floor().long() * 4 + (x[:, 1] * 4).floor().long()
    l_y = ((y[:, 0] - 0.2) * 4).floor().long() * 4 + ((y[:, 1] - 0.2) * 4).floor().long()
    out = SamplesLoss()(l_x, a, x, l_y, b, y)
    ref = SamplesLoss(backend="online")(a, x, b, y)
    _close(out, ref, 5e-2, 0)
    with pytest.raises(ValueError):
        SamplesLoss(backend="online")(l_x, a, x, l_y, b, y)


# ---------------------------------------------------------------------------------
# Custom costs
# ---------------------------------------------------------------------------------
SYMBOLIC_COSTS = [
    (lambda x, y: x.sqdist(y) / 2, 2),
    (lambda x, y: (x - y).norm2(), 1),
    (lambda x, y: ((x - y) ** 2).sum() * 0.5, 2),
]


@pytest.mark.parametrize("cost,p", SYMBOLIC_COSTS, ids=["sqdist/2", "norm2", "sum-squares/2"])
@pytest.mark.parametrize("backend", ["auto", "online"])
def test_symbolic_costs(cost, p, backend, device):
    inputs = [t.to(device) for t in measures(B=2)]
    ref = SamplesLoss(backend="online", p=p)(*inputs)
    out = SamplesLoss(backend=backend, p=p, cost=cost)(*inputs)
    _close(out, ref, 1e-9, 1e-12)


def test_symbolic_cost_selects_online():
    loss = SamplesLoss(cost=lambda x, y: (x - y).norm2())
    assert loss.select_backend(1, 10, 10, 2, False) == "online"
    assert SamplesLoss().select_backend(1, 10, 10, 2, False) == "tensorized"
    with_routine = SamplesLoss(cost=(lambda x, y: (x - y).norm2(), lambda x, y: torch.cdist(x, y)))
    assert with_routine.select_backend(1, 10, 10, 2, False) == "tensorized"


def test_kernox_formula_cost_batched(device):
    a, x, b, y = [t.to(device) for t in measures(B=3, D=3)]
    w = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    def cost(x, y):  # kernox formulas: x = Vi, y = Vj
        d2 = (x - y) ** 2
        return (d2[0] * 1.0 + d2[1] * 2.0 + d2[2] * 0.5) / 2

    def dense(x, y):
        return ((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2 * w.to(x)).sum(-1) / 2

    out = SamplesLoss(backend="online", cost=cost)(a, x, b, y)
    ref = SamplesLoss(backend="tensorized", cost=dense)(a, x, b, y)
    _close(out, ref, 1e-9, 1e-12)
    one = (a[0], x[0], b[0], y[0])  # the diameter of a batch covers all of its elements
    out_ms = SamplesLoss(backend="multiscale", cost=(cost, dense), cluster_scale=1e-9, truncate=None)(*one)
    _close(out_ms, SamplesLoss(backend="tensorized", cost=dense)(*one), 1e-9, 1e-12)


def test_keops_strings_are_rejected():
    for cost in ("Norm2(X-Y)", ("Norm2(X-Y)", lambda x, y: torch.cdist(x, y))):
        with pytest.raises(TypeError, match="kernox formulas"):
            SamplesLoss(backend="online", cost=cost)(*measures())


# ---------------------------------------------------------------------------------
# Hausdorff
# ---------------------------------------------------------------------------------
@needs_geomloss
def test_hausdorff_with_kernel_matches_geomloss():
    a, x, b, y = measures()

    def k(x, y, blur=0.05, **_):
        return (-((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2).sum(-1) / blur).exp()

    ref = geomloss.SamplesLoss("hausdorff", kernel=k, backend="tensorized")(a, x, b, y)
    out = SamplesLoss("hausdorff", kernel=k, backend="tensorized")(a, x, b, y)
    _close(out, ref, 0, 0)
    with pytest.raises(KeyError):  # the broken GeomLoss 0.3.1 code path
        geomloss.SamplesLoss("hausdorff", backend="tensorized")(a, x, b, y)


@pytest.mark.parametrize("p", [1, 2])
def test_hausdorff_definition(p, device):
    a, x, b, y = [t.to(device) for t in measures(N=40, M=30)]
    blur = 0.1
    eps = blur**p
    cost = (lambda u, v: _dist(u, v)) if p == 1 else (lambda u, v: _sq(u, v) / 2)

    def smin(u, v, w):
        return -eps * (w.log()[None, :] - cost(u, v) / eps).logsumexp(1)

    a_x, b_x, a_y, b_y = smin(x, x, a), smin(x, y, b), smin(y, x, a), smin(y, y, b)
    ref = 0.5 * ((a * (b_x - a_x)).sum() + (b * (a_y - b_y)).sum())
    for backend in ("tensorized", "online", "multiscale"):
        out = SamplesLoss("hausdorff", p=p, blur=blur, backend=backend)(a, x, b, y)
        # tensorized p = 1: 1e-4 distance floor on the diagonal (GeomLoss convention)
        _close(out, ref, 2e-3 if (p == 1 and backend == "tensorized") else 1e-10, 1e-12)


def test_hausdorff_icp_limit():
    a, x, b, y = measures(N=30, M=20)
    out = SamplesLoss("hausdorff", p=1, blur=1e-4, backend="online")(a, x, b, y)
    d = _dist(x, y)
    ref = 0.5 * ((a * d.min(1).values).sum() + (b * d.min(0).values).sum())
    _close(out, ref, 1e-3, 0)


def test_hausdorff_gradcheck():
    inputs = [t.requires_grad_() for t in measures(N=7, M=5)]
    loss = SamplesLoss("hausdorff", blur=0.3, backend="online")
    assert torch.autograd.gradcheck(lambda *t: loss(*t), inputs)


# ---------------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["gaussian", "laplacian", "energy"])
@pytest.mark.parametrize("backend", ["online", "tensorized"])
def test_kernel_gradcheck(name, backend):
    # x and y apart: the distance is never zero off the diagonal (smooth gradient)
    inputs = [t.requires_grad_() for t in measures(N=6, M=5, B=2)]
    loss = SamplesLoss(name, blur=0.4, backend=backend)
    if name != "gaussian":  # ‖x − y‖ is not differentiable at x = y: test α, β and y
        a, x, b, y = inputs
        x.requires_grad_(False)
        assert torch.autograd.gradcheck(lambda a, b, y: loss(a, x, b, y), (a, b, y))
    else:
        assert torch.autograd.gradcheck(lambda *t: loss(*t), inputs)


@pytest.mark.parametrize("kw", [dict(), dict(p=1), dict(reach=1.0), dict(reach=0.3, p=1),
                                dict(reach=0.5, debias=False), dict(debias=False)], ids=str)
def test_sinkhorn_gradient_at_convergence(kw, monkeypatch):
    """At convergence, the gradient in x, y (envelope theorem) is the derivative of the value.

    400 iterations at ε = blur^p are forced to reach convergence. In unbalanced
    transport, this test fails with the effective gradient of GeomLoss v0.3.1
    (weight ρ + ε/2 instead of ρ + ε).
    """
    import importlib

    sd = importlib.import_module("remblais.sinkhorn_divergence")  # the name is shadowed by the exported function

    monkeypatch.setattr(sd, "epsilon_schedule", lambda p, d, blur, s: [blur**p] * 400)
    a, x, b, y = measures(N=8, M=6)
    x.requires_grad_(), y.requires_grad_()
    loss = SamplesLoss(blur=0.5, backend="online", **kw)
    grads = torch.autograd.grad(loss(a, x, b, y), (x, y))
    h = 1e-6
    for t, g in zip((x, y), grads):
        for idx in [(0, 0), (1, 1), (3, 0)]:
            with torch.no_grad():
                t[idx] += h
                up = loss(a, x, b, y)
                t[idx] -= 2 * h
                down = loss(a, x, b, y)
                t[idx] += h
            assert abs((up - down) / (2 * h) - g[idx]) < 1e-7 * g.abs().max()


def test_grad_state_restored():
    a, x, b, y = measures()
    with torch.no_grad():
        SamplesLoss(backend="online")(a, x, b, y)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


# ---------------------------------------------------------------------------------
# Shapes and errors
# ---------------------------------------------------------------------------------
def test_output_shapes():
    a, x, b, y = measures(B=4)
    assert SamplesLoss(backend="online")(a, x, b, y).shape == (4,)
    assert SamplesLoss(backend="online")(a[0], x[0], b[0], y[0]).shape == ()
    assert SamplesLoss(backend="multiscale")(a[:1], x[:1], b[:1], y[:1]).shape == (1,)
    assert SamplesLoss("energy", backend="multiscale")(a[0], x[0], b[0], y[0]).shape == ()
    F, G = SamplesLoss(backend="online", potentials=True)(a, x, b, y)
    assert F.shape == a.shape and G.shape == b.shape


def test_errors():
    a, x, b, y = measures()
    with pytest.raises(ValueError):
        SamplesLoss(loss="wasserstein")
    with pytest.raises(ValueError):
        SamplesLoss()(x, y, x)
    with pytest.raises(ValueError):
        SamplesLoss()(a, x, b, y[:, :1])
    with pytest.raises(ValueError):
        SamplesLoss()(a[:-1], x, b, y)
    with pytest.warns(UserWarning):
        a2, x2, b2, y2 = measures(B=2)
        SamplesLoss(backend="multiscale")(a2, x2, b2, y2)


@pytest.mark.parametrize("B", [None, 2])
@pytest.mark.parametrize("kw", [dict(), dict(reach=0.4), dict(p=1), dict(debias=False), dict(debias=False, reach=0.4)],
                         ids=str)
@pytest.mark.parametrize("NM", [(40, 30), (35, 35), (60, 30)], ids=str)
def test_paired_loop_equals_four_softmins(kw, B, NM, device):
    """The loop with two reductions per iteration reproduces the GeomLoss loop exactly
    (N ≠ M: lse2 with selectors; N = M: LSE with a batch of 2; N = 2M: no mix-up)."""
    from remblais.sinkhorn_samples import sinkhorn_online

    inputs = [t.to(device).requires_grad_() for t in measures(B=B or 1, N=NM[0], M=NM[1])]
    outs = []
    for paired in (True, False):
        val = sinkhorn_online(*inputs, paired=paired, **kw)
        outs.append((val.detach(), torch.autograd.grad(val.sum(), inputs)))
    _close(outs[0][0], outs[1][0], 1e-13, 1e-15)
    for u, w in zip(outs[0][1], outs[1][1]):
        _close(u, w, 1e-12, 1e-15)


@pytest.mark.parametrize("B", [1, 3])
def test_stacked_equals_selectors(B, device, monkeypatch):
    """N = M: the LSE with a batch of 2 (broadcast columns) gives the same values and gradients as lse2."""
    import remblais.sinkhorn_samples as rs

    inputs = [t.to(device).requires_grad_() for t in measures(B=B, N=45, M=45)]
    res = []
    for stacked in (True, False):
        monkeypatch.setattr(rs, "STACKED", stacked)
        val = rs.sinkhorn_online(*inputs, blur=0.1)
        res.append((val.detach(), torch.autograd.grad(val.sum(), inputs)))
    _close(res[0][0], res[1][0], 1e-13, 1e-15)
    for u, w in zip(res[0][1], res[1][1]):
        _close(u, w, 1e-12, 1e-15)


@pytest.mark.parametrize("which", ["x", "y", "both", "weights"])
def test_one_sided_gradient_matches_paired(which, device):
    """A single differentiable cloud: final extrapolation with four LSEs, same values and
    gradients as the GeomLoss loop (paired=False)."""
    from remblais.sinkhorn_samples import sinkhorn_online

    a, x, b, y = (t.to(device) for t in measures(B=1, N=40, M=40))
    need = {"x": [x], "y": [y], "both": [x, y], "weights": [a, b]}[which]
    for t in need:
        t.requires_grad_()
    out = []
    for paired in (True, False):
        val = sinkhorn_online(a, x, b, y, blur=0.1, paired=paired)
        out.append((val.detach(), torch.autograd.grad(val.sum(), need)))
    _close(out[0][0], out[1][0], 1e-13, 1e-15)
    for u, w in zip(out[0][1], out[1][1]):
        _close(u, w, 1e-12, 1e-15)


@pytest.mark.parametrize("D", [3, 20])
def test_dot_form_equals_direct(D, device):
    """Dot-product form (½‖x‖² + ½‖y‖² − ⟨x, y⟩): same result in FP64, gradients included."""
    from remblais.sinkhorn_samples import sinkhorn_online

    inputs = [t.to(device).requires_grad_() for t in measures(B=1, N=50, M=40, D=D)]
    res = []
    for dot in (False, True):
        val = sinkhorn_online(*inputs, blur=0.1, dot=dot)
        res.append((val.detach(), torch.autograd.grad(val.sum(), inputs)))
    _close(res[0][0], res[1][0], 1e-11, 1e-14)
    for u, w in zip(res[0][1], res[1][1]):
        _close(u, w, 1e-9, 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs: requires an NVIDIA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_graph_iterations_equal_eager(dtype, monkeypatch):
    import importlib

    rs = importlib.import_module("remblais.sinkhorn_samples")
    vals = []
    for graphs in (False, True, True):  # the 2nd graph call replays the captured graph
        monkeypatch.setattr(rs, "USE_CUDA_GRAPHS", graphs)
        a, x, b, y = [t.to("cuda").requires_grad_() for t in measures(N=300, M=200, D=3, dtype=dtype, seed=len(vals))]
        val = SamplesLoss(backend="online")(a, x, b, y)
        vals.append((val.detach(), torch.autograd.grad(val, (a, x, b, y)), (a, x, b, y)))
    # same data for the 1st (eager) call and an eager recomputation of the 3rd
    monkeypatch.setattr(rs, "USE_CUDA_GRAPHS", False)
    for v, g, inp in vals[1:]:
        ref = SamplesLoss(backend="online")(*inp)
        _close(v, ref.detach(), 1e-6 if dtype == torch.float32 else 1e-13, 0)
        for u, w in zip(g, torch.autograd.grad(ref, inp)):
            _close(u, w, 1e-5 if dtype == torch.float32 else 1e-12, 1e-9 if dtype == torch.float32 else 1e-15)
    assert len(rs._GRAPH_FAILED) == 0


def test_flush_denormals_option(device):
    """flush_denormals: enabled by default on the CPU and on CUDA FP32; the losses
    and gradients do not depend on subnormal terms (small blur: many of them)."""
    from remblais import _kp

    with pytest.raises(ValueError):
        SamplesLoss(flush_denormals="yes")
    cuda32 = device.type == "cuda" and torch.version.hip is None
    assert _kp.default_flush_denormals(device, torch.float32) == (device.type == "cpu" or cuda32)
    assert _kp.default_flush_denormals(device, torch.float64) == (device.type == "cpu")
    for dtype in (torch.float32, torch.float64):
        a, x, b, y = measures(N=300, M=280, D=3, dtype=dtype, device=device)
        if device.type == "xpu" or (device.type == "cuda" and torch.version.hip is not None):
            import kernox as kp

            with pytest.raises(kp.UnsupportedError):
                SamplesLoss("sinkhorn", blur=0.01, backend="online", flush_denormals=True)(a, x, b, y)
            continue
        tol = dict(rtol=1e-5, atol=1e-7) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-15)
        for loss in ("sinkhorn", "gaussian"):
            res = {}
            for fd in (None, True, False):
                xr = x.clone().requires_grad_()
                L = SamplesLoss(loss, blur=0.01, backend="online", flush_denormals=fd)(a, xr, b, y)
                res[fd] = (L.detach(), torch.autograd.grad(L, [xr])[0])
            dflt = bool(_kp.default_flush_denormals(device, dtype))
            assert torch.equal(res[None][0], res[dflt][0]) and torch.equal(res[None][1], res[dflt][1])
            torch.testing.assert_close(res[True][0], res[False][0], **tol)
            torch.testing.assert_close(res[True][1], res[False][1], **tol)


def test_fast_math_option(device):
    """fast_math: enabled by default only on CUDA FP32; approximate but close to IEEE;
    no effect in FP64 and on the CPU."""
    from remblais import _kp

    with pytest.raises(ValueError):
        SamplesLoss(fast_math="yes")
    assert _kp.default_fast_math(device, torch.float32) == (device.type == "cuda" and torch.version.hip is None)
    assert not _kp.default_fast_math(device, torch.float64)
    for dtype in (torch.float32, torch.float64):
        a, x, b, y = measures(N=300, M=280, D=3, dtype=dtype, device=device)
        res = {}
        if device.type == "xpu":  # sycl backend: kernox explicitly refuses fast_math
            import kernox as kp

            with pytest.raises(kp.UnsupportedError):
                SamplesLoss("sinkhorn", blur=0.05, backend="online", fast_math=True)(a, x, b, y)
            res_none = SamplesLoss("sinkhorn", blur=0.05, backend="online")(a, x, b, y)
            assert torch.equal(res_none, SamplesLoss("sinkhorn", blur=0.05, backend="online", fast_math=False)(a, x, b, y))
            continue
        for fm in (None, True, False):
            xr = x.clone().requires_grad_()
            L = SamplesLoss("sinkhorn", blur=0.05, backend="online", fast_math=fm)(a, xr, b, y)
            res[fm] = (L.detach(), torch.autograd.grad(L, [xr])[0])
        dflt = True if _kp.default_fast_math(device, dtype) else False
        assert torch.equal(res[None][0], res[dflt][0]) and torch.equal(res[None][1], res[dflt][1])
        if dtype == torch.float64 or device.type == "cpu":
            assert torch.equal(res[True][0], res[False][0])
        else:
            torch.testing.assert_close(res[True][0], res[False][0], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(res[True][1], res[False][1], rtol=1e-3, atol=1e-5)
