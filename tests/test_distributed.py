"""SamplesLoss(distributed=True): the kernox reductions split across a Gloo group of
local processes give the value and the gradients of the local computation."""

import os
import traceback

import pytest
import torch

dist = pytest.importorskip("torch.distributed")

pytestmark = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="torch.distributed with Gloo required")

CASES = [
    ("sinkhorn", "online", {}, None),
    ("sinkhorn", "online", {"reach": 0.5}, None),
    ("sinkhorn", "online", {"debias": False}, None),
    ("sinkhorn", "online", {}, 3),
    ("sinkhorn", "online", {"potentials": True}, None),
    ("sinkhorn", "multiscale", {"blur": 0.02, "truncate": 4}, None),
    ("sinkhorn", "multiscale", {"blur": 0.02, "potentials": True}, None),
    ("hausdorff", "online", {}, None),
    ("gaussian", "online", {"blur": 0.3}, None),
    ("energy", "multiscale", {}, None),
]


def _measures(N, M, B=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = (B,) if B else ()
    x = torch.rand(*s, N, 2, generator=g, dtype=torch.float64)
    y = torch.rand(*s, M, 2, generator=g, dtype=torch.float64) + 0.2
    a = torch.rand(*s, N, generator=g, dtype=torch.float64) + 0.1
    b = torch.rand(*s, M, generator=g, dtype=torch.float64) + 0.1
    return a / a.sum(-1, keepdim=True), x, b / b.sum(-1, keepdim=True), y


def _run(loss, inputs):
    leaves = [t.clone().requires_grad_() for t in inputs]
    out = loss(*leaves)
    outs = out if isinstance(out, tuple) else (out,)
    g = torch.Generator().manual_seed(5)
    total = sum((o * torch.randn(o.shape, generator=g, dtype=o.dtype)).sum() for o in outs)
    grads = torch.autograd.grad(total, leaves, allow_unused=True)
    return [o.detach() for o in outs], grads


def _worker(rank, world, init_file, errors_dir):
    import remblais._kp
    import remblais.clustering
    from remblais import SamplesLoss

    remblais._kp.VALIDATE_RANGES = True
    remblais.clustering.DENSE_RATIO = float("inf")
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    failures = []
    try:
        for name, backend, kw, B in CASES:
            label = f"{name}/{backend}/{kw}/B={B}"
            try:
                inputs = _measures(70, 53, B=B)
                ref = _run(SamplesLoss(name, backend=backend, **kw), inputs)
                got = _run(SamplesLoss(name, backend=backend, distributed=True, **kw), inputs)
                for p, q in zip(got[0], ref[0]):
                    torch.testing.assert_close(p, q, rtol=1e-9, atol=1e-12)
                for k, (p, q) in enumerate(zip(got[1], ref[1])):
                    assert (p is None) == (q is None), (label, k)
                    if p is not None:
                        torch.testing.assert_close(p, q, rtol=1e-8, atol=1e-11, msg=lambda s: f"grad {k}: {s}")
            except Exception:  # noqa: BLE001 - reported by the parent
                failures.append(f"{label}:\n{traceback.format_exc()}")
    finally:
        dist.barrier()
        dist.destroy_process_group()
    if failures:
        with open(os.path.join(errors_dir, f"rank{rank}.txt"), "w") as f:
            f.write("\n".join(failures))


@pytest.mark.parametrize("world", [2, 3])
def test_distributed_samples_loss(world, tmp_path):
    import torch.multiprocessing as mp

    errors = tmp_path / "errors"
    errors.mkdir()
    mp.spawn(_worker, args=(world, str(tmp_path / "init"), str(errors)), nprocs=world, join=True)
    reports = sorted(errors.iterdir())
    if reports:
        pytest.fail("\n".join(p.read_text() for p in reports)[:20000])


def test_distributed_option_is_checked():
    from remblais import SamplesLoss

    with pytest.raises(ValueError, match="distributed"):
        SamplesLoss(distributed="yes")
    # without an initialized group, distributed=True computes locally
    a, x, b, y = _measures(20, 30)
    ref = SamplesLoss(backend="online")(a, x, b, y)
    torch.testing.assert_close(SamplesLoss(backend="online", distributed=True)(a, x, b, y), ref, rtol=0, atol=0)
