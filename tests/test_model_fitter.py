"""Unit tests for BPR model fitting."""
import numpy as np
import pytest

from src.model_fitter import model


def test_bpr_model_basic():
    """BPR formula: d(x) = fft * (1 + a * (x/cap)^b) — paper eq. 15."""
    fft, a, b, cap = 30.0, 0.15, 4.0, 1000.0
    assert model(0, a, b, cap, fft) == pytest.approx(fft)
    assert model(cap, a, b, cap, fft) == pytest.approx(fft * (1 + a))
    assert model(cap / 2, a, b, cap, fft) == pytest.approx(fft * (1 + a * 0.5**b))


def test_bpr_model_monotonic():
    """BPR delay must be non-decreasing in x (paper assumption)."""
    fft, a, b, cap = 30.0, 0.15, 4.0, 1000.0
    xs = np.linspace(0, 2000, 50)
    ys = [model(x, a, b, cap, fft) for x in xs]
    assert all(ys[i] <= ys[i + 1] + 1e-10 for i in range(len(ys) - 1))


def test_bpr_model_positive():
    """BPR delay must be non-negative."""
    for x in [0, 100, 500, 1000, 5000]:
        assert model(x, 0.15, 4.0, 1000.0, 30.0) >= 0
