"""Tests for the ElevatedELU activation function."""

from __future__ import annotations

import torch

from mdn_pytorch.activations import ElevatedELU


def test_positive_and_zero_inputs() -> None:
    x = torch.tensor([0.0, 1.0, 2.5])
    assert torch.allclose(ElevatedELU()(x), x + 1.0)


def test_negative_inputs_converge_to_exp() -> None:
    x = torch.tensor([-2.0, -1.0, -0.5])
    assert torch.allclose(ElevatedELU()(x), torch.exp(x), rtol=1e-5)


def test_strictly_positive_output() -> None:
    x = torch.linspace(-10.0, 10.0, 100)
    assert torch.all(ElevatedELU()(x) > 0)


def test_dtype_preserved() -> None:
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, -1.0], dtype=dtype)
        assert ElevatedELU()(x).dtype == dtype
