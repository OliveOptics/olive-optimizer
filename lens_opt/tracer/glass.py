"""Material models: constant index and Sellmeier dispersion."""
from dataclasses import dataclass
import torch
from torch import Tensor


class Material:
    """Base class. Subclasses implement n(wavelength_um) -> Tensor."""

    def n(self, wavelength_um: Tensor) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantIndex(Material):
    """Wavelength-independent refractive index (e.g. air at n=1)."""
    n_value: Tensor

    def n(self, wavelength_um: Tensor) -> Tensor:
        return self.n_value.expand_as(wavelength_um)


@dataclass(frozen=True)
class Sellmeier(Material):
    """Sellmeier dispersion: n^2 = 1 + sum_i B_i * lam^2 / (lam^2 - C_i).

    B, C: tensors of shape (3,). C is in micrometers^2.
    """
    B: Tensor
    C: Tensor

    def n(self, wavelength_um: Tensor) -> Tensor:
        lam2 = (wavelength_um * wavelength_um).unsqueeze(-1)
        n_sq = 1.0 + (self.B * lam2 / (lam2 - self.C)).sum(-1)
        return torch.sqrt(n_sq)


def _sellmeier(B, C, *, device=None, dtype=torch.float64) -> Sellmeier:
    return Sellmeier(
        B=torch.tensor(B, device=device, dtype=dtype),
        C=torch.tensor(C, device=device, dtype=dtype),
    )


def Air(*, device=None, dtype=torch.float64) -> ConstantIndex:
    return ConstantIndex(n_value=torch.tensor(1.0, device=device, dtype=dtype))


def BK7(*, device=None, dtype=torch.float64) -> Sellmeier:
    """Schott N-BK7."""
    return _sellmeier(
        [1.03961212, 0.231792344, 1.01046945],
        [0.00600069867, 0.0200179144, 103.560653],
        device=device, dtype=dtype,
    )


def SF2(*, device=None, dtype=torch.float64) -> Sellmeier:
    """Schott SF2."""
    return _sellmeier(
        [1.40301821, 0.231767504, 0.939056586],
        [0.0105795466, 0.0493226978, 112.405955],
        device=device, dtype=dtype,
    )


def F2(*, device=None, dtype=torch.float64) -> Sellmeier:
    """Schott F2 (flint glass used in Cooke triplet middle element)."""
    return _sellmeier(
        [1.34533359, 0.209073176, 0.937357162],
        [0.00997743871, 0.0470450767, 111.886764],
        device=device, dtype=dtype,
    )


def SK16(*, device=None, dtype=torch.float64) -> Sellmeier:
    """Schott SK16 (crown glass used in Cooke triplet outer elements)."""
    return _sellmeier(
        [1.34317774, 0.241144399, 0.994317969],
        [0.00704687339, 0.0229005000, 92.7508526],
        device=device, dtype=dtype,
    )
