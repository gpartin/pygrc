"""
Copyright (c) 2026 G. Partin. MIT License.
Author: G. Partin, Date: March 2026
"""
from pygrc.lfm import LFM, v_baryonic, lfm_rar_velocity
import numpy as np
import pandas as pd


def _make_galaxy(n=20):
    """Create a synthetic SPARC-like DataFrame with a rising rotation curve."""
    r = np.linspace(1, 20, n)
    return pd.DataFrame(
        {
            "Rad": r,
            "Vobs": 100.0 + 5.0 * r,
            "errV": np.full(n, 5.0),
            "Vgas": 20.0 + 2.0 * r,
            "Vdisk": 90.0 + 3.0 * r,
            "Vbul": np.zeros(n),
        }
    )


def test_lfm_predict():
    data = _make_galaxy()
    m = LFM(data, name="Synthetic")
    v = m.predict()
    assert len(v) == 20, "Prediction length should match input"
    assert all(v > 0), "All predicted velocities should be positive"
    assert m.r_squared() > 0.0, "R-squared should be positive"


def test_lfm_calibrate():
    data = _make_galaxy()
    m = LFM(data, name="Synthetic")
    best = m.calibrate()
    assert 0.1 <= best <= 1.5, \
        "Calibrated Upsilon should be within bounds"
    assert m.rmse() >= 0, "RMSE should be non-negative"


def test_v_baryonic():
    data = _make_galaxy()
    vb = v_baryonic(data)
    assert len(vb) == 20, "v_baryonic length should match input"
    assert all(vb > 0), "Baryonic velocities should be positive"


def test_lfm_rar_velocity():
    r = np.array([5.0, 10.0, 15.0])
    vbar = np.array([100.0, 120.0, 130.0])
    vpred = lfm_rar_velocity(r, vbar)
    assert len(vpred) == 3, "Output length should match input"
    assert all(vpred >= vbar), \
        "LFM-RAR velocity should be >= baryonic velocity"
