"""remblais: a GeomLoss clone built on kernox (fused GPU reductions).

remblais is a drop-in replacement for GeomLoss 0.3.1: replace
``from geomloss import SamplesLoss`` with ``from remblais import SamplesLoss``,
or ``import remblais as geomloss``.
"""

__version__ = "2.2.0"
GEOMLOSS_VERSION = "0.3.1"  # reference version for parity

from .costs import Cost  # noqa: E402
from .kernel_samples import Kernel  # noqa: E402
from .samples_loss import SamplesLoss  # noqa: E402
from .sinkhorn_images import sinkhorn_divergence  # noqa: E402
from .wasserstein_barycenter_images import ImagesBarycenter  # noqa: E402

__all__ = ["SamplesLoss", "ImagesBarycenter", "sinkhorn_divergence", "Cost", "Kernel"]
