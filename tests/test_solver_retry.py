"""Contract tests for the post-solve validation retry in the CG solver."""

import cvxpy as cp

from src.traffic_optimizer import Network


class _Stub:
    solver_metadata = {"attempts": []}


def test_retry_solve_passes_when_validation_accepts():
    stub = _Stub()
    x = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 2)), [x >= 0])
    calls = {"count": 0}

    def validate_solution():
        calls["count"] += 1
        return calls["count"] <= 1

    assert Network._retry_solve_after_validation_failure(
        stub, prob, "CLARABEL",
        {"tol_gap_abs": 1e-9, "tol_feas": 1e-9, "max_iter": 500},
        validate_solution,
    ) is True
    assert calls["count"] == 1
    assert stub.solver_metadata["attempts"][-1]["stage"] == "validation_retry"
    assert stub.solver_metadata["attempts"][-1]["status"] == "optimal"


def test_retry_solve_fails_when_validation_keeps_rejecting():
    stub = _Stub()
    x = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 2)), [x >= 0])
    assert Network._retry_solve_after_validation_failure(
        stub, prob, "CLARABEL",
        {"tol_feas": 1e-9, "max_iter": 200},
        lambda: False,
    ) is False
    assert stub.solver_metadata["attempts"][-1]["status"] == "optimal"


def test_retry_solve_records_solver_exception():
    stub = _Stub()
    x = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 2)), [x >= 0])
    assert Network._retry_solve_after_validation_failure(
        stub, prob, "CLARABEL", {"no_such_option": 1}, lambda: True,
    ) is False
    assert stub.solver_metadata["attempts"][-1]["status"] == "error"
    assert stub.solver_metadata["attempts"][-1]["stage"] == "validation_retry"
