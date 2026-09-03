"""Unit tests for BPR model fitting."""
import numpy as np
import pandas as pd
import pytest

from src.model_fitter import TrafficModelFitter, convert_string_to_array, model, validate_bpr_fit_table


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


def test_constant_fit_is_explicit_and_reports_actual_r2():
    frame = pd.DataFrame([{
        'link_id': 0,
        'x_vector': np.arange(5, dtype=float),
        'y_vector': np.ones(5, dtype=float) * 10.0,
    }])
    fitter = TrafficModelFitter(pandas_df=frame)
    link_id, a, b, cap, fft, r2 = fitter.fit_and_evaluate(frame.iloc[0].to_dict())

    assert link_id == 0
    assert (a, b, cap, fft) == pytest.approx((0.0, 0.0, 1.0, 10.0))
    assert r2 == pytest.approx(0.0)


def test_fitter_csv_arrays_round_trip_as_json(tmp_path):
    frame = pd.DataFrame([{
        'link_id': 0,
        'x_vector': np.arange(5, dtype=float),
        'y_vector': np.arange(5, dtype=float) + 10.0,
    }])
    fitter = TrafficModelFitter(pandas_df=frame)
    fitter.parallel_fit_and_evaluate(workers=1, output_dir=tmp_path, save_plots=False)

    loaded = pd.read_csv(tmp_path / 'fitter_results.csv')
    convert_string_to_array(loaded, 'x_vector')
    convert_string_to_array(loaded, 'y_vector')
    assert np.array_equal(loaded.loc[0, 'x_vector'], np.arange(5, dtype=float))
    assert np.array_equal(loaded.loc[0, 'y_vector'], np.arange(5, dtype=float) + 10.0)
    assert loaded.loc[0, 'fit_status'] in {'full', 'constant'}


def test_variation_screen_is_not_scaled_by_flow_units():
    frame = pd.DataFrame([{
        'link_id': 0,
        'x_vector': np.array([570.0, 1425.0, 2850.0, 5700.0, 11400.0]),
        'y_vector': np.array([10.0, 12.0, 15.0, 20.0, 24.0]),
        'calibration_capacity': 5700.0,
    }])
    fitter = TrafficModelFitter(pandas_df=frame)
    result = fitter.fit_and_evaluate(frame.iloc[0].to_dict())

    # A changing travel-time curve must not be classified as constant merely
    # because flow is expressed in vehicles/hour rather than seconds.
    assert result[1:4] != (0, 0, 1)


def test_strict_bpr_validation_rejects_constant_fit():
    frame = pd.DataFrame([{
        'link_id': 0,
        'a_fit': 0.0,
        'b_fit': 0.0,
        'cap_fit': 1.0,
        'fft_fit': 10.0,
        'R^2': 0.0,
        'fit_status': 'constant',
    }])
    with pytest.raises(ValueError, match='Strict BPR fit validation failed'):
        validate_bpr_fit_table(frame, expected_link_ids=[0], require_full_fit=True)


def test_fixed_reference_fit_recovers_positive_bpr_parameters(tmp_path):
    fft, capacity, a_true, b_true = 20.0, 1000.0, 0.15, 4.0
    x = np.array([100, 250, 500, 750, 1000, 1250, 1500, 2000], dtype=float)
    y = model(x, a_true, b_true, capacity, fft)
    frame = pd.DataFrame([{
        'link_id': 0,
        'x_vector': x,
        'y_vector': y,
        'calibration_capacity': capacity,
        'calibration_fft': fft,
    }])
    fitter = TrafficModelFitter(pandas_df=frame)
    summary = fitter.parallel_fit_and_evaluate(
        workers=1,
        output_dir=tmp_path,
        save_plots=False,
        require_full_fit=True,
        r2_threshold=0.99,
        expected_link_ids=[0],
        fixed_references=True,
    )
    assert summary['full_fit_count'] == 1
    row = fitter.df.iloc[0]
    assert row['a_fit'] == pytest.approx(a_true, rel=1e-3)
    assert row['b_fit'] == pytest.approx(b_true, rel=1e-3)
    assert row['cap_fit'] == pytest.approx(capacity)
    assert row['fft_fit'] == pytest.approx(fft)
