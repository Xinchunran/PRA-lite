from __future__ import annotations

import torch

from src.model.pragma_lite import PragmaLiteConfig, PragmaLiteModel


def test_calendar_proj_shape_with_mlp_enabled() -> None:
    cfg = PragmaLiteConfig(vocab_size=100, d_model=64, n_heads=4, calendar_mlp=True, calendar_hidden_dim=48)
    model = PragmaLiteModel(cfg)
    x = torch.randn(2, 10, 6)
    y = model.calendar_proj(x)

    assert y.shape == (2, 10, 64)


def test_calendar_proj_falls_back_to_linear_when_disabled() -> None:
    cfg = PragmaLiteConfig(vocab_size=100, d_model=64, n_heads=4, calendar_mlp=False)
    model = PragmaLiteModel(cfg)

    assert isinstance(model.calendar_proj, torch.nn.Linear)
