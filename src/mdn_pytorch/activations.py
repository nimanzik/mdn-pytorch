from __future__ import annotations

from torch import Tensor as TorchTensor
from torch import nn

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
}


class ElevatedELU(nn.ELU):
    """ELU activation shifted upwards by 1 to get strictly positive outputs."""

    def __init__(self):
        super().__init__(alpha=1.0)

    def forward(self, input: TorchTensor) -> TorchTensor:
        return super().forward(input) + 1.0
