import os

import pytest
import torch

# The tests validate every block-sparse range sent to kernox.
import remblais._kp  # noqa: E402

remblais._kp.VALIDATE_RANGES = True

# The multiscale tests exercise the ranges: no dense fallback (small clouds).
import remblais.clustering  # noqa: E402

remblais.clustering.DENSE_RATIO = float("inf")


torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


def _devices():
    devs = ["cpu"]
    if os.environ.get("REMBLAIS_GPU") == "1" and torch.cuda.is_available():
        devs.append("cuda")
    if os.environ.get("REMBLAIS_GPU") == "1" and hasattr(torch, "xpu") and torch.xpu.is_available():
        devs.append("xpu")
    return devs


DEVICES = _devices()


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


def measures(N=60, M=50, D=2, B=None, dtype=torch.float64, device="cpu", seed=0, grad=False, shift=0.2):
    """Two random weighted measures (positions in [0,1]^D, shifted)."""
    g = torch.Generator().manual_seed(seed)
    s = (B,) if B else ()
    x = torch.rand(*s, N, D, generator=g, dtype=torch.float64)
    y = torch.rand(*s, M, D, generator=g, dtype=torch.float64) + shift
    a = torch.rand(*s, N, generator=g, dtype=torch.float64) + 0.1
    b = torch.rand(*s, M, generator=g, dtype=torch.float64) + 0.1
    a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
    out = [t.to(device=device, dtype=dtype) for t in (a, x, b, y)]
    if grad:
        out = [t.requires_grad_() for t in out]
    return out


@pytest.fixture(autouse=True)
def _sync_after_test(request):
    """Synchronizes the GPU after each test, so that an asynchronous fault is attributed to the right case."""
    yield
    dev = request.node.callspec.params.get("device") if hasattr(request.node, "callspec") else None
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "xpu":
        torch.xpu.synchronize()
