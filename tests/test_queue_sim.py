"""Unit tests for queue simulation availability and basic imports."""
from types import SimpleNamespace

import pytest


def test_queue_sim_import():
    """Test that queue_sim can be imported (may be unavailable on non-macOS)."""
    from queue_sim import QUEUE_SIM_AVAILABLE, _QUEUE_SIM_ERROR
    if QUEUE_SIM_AVAILABLE:
        from queue_sim import Runner
        assert Runner is not None
    else:
        pytest.skip(f"Queue sim not available on this platform: {_QUEUE_SIM_ERROR}")


def test_queue_sim_helpers():
    """Test the helper functions from find_nash and comparison modules."""
    from queue_sim.find_nash import _collapse_repeats, _rounded_counts, _prune_flow_data
    assert _collapse_repeats([1, 1, 2, 2, 3, 3, 1]) == [1, 2, 3, 1]
    assert _collapse_repeats([]) == []
    assert _collapse_repeats([5]) == [5]

    routes = [{'flow': 10.0}, {'flow': 20.0}, {'flow': 30.0}]
    counts = _rounded_counts(routes, 6)
    assert sum(counts) == 6
    assert len(counts) == 3

    from queue_sim.comparison import _placement_seed
    s1 = _placement_seed(0, [14, 20])
    s2 = _placement_seed(0, [20, 14])
    s3 = _placement_seed(1, [14, 20])
    assert s1 == s2
    assert s1 != s3


def test_unused_route_uses_current_link_cost_not_free_flow():
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip("queue simulator unavailable")
    from queue_sim.runner_EV import Runner

    runner = Runner.__new__(Runner)
    runner.sim = SimpleNamespace(
        all_agents={},
        all_links={
            10: SimpleNamespace(ave_travel_time=18.0, fft=5.0),
            11: SimpleNamespace(ave_travel_time=12.0, fft=4.0),
        },
        resolve_link_id=lambda start, end: {(0, 1): 10, (1, 2): 11}[(start, end)],
    )
    runner.route_groups = [{
        'od_pair': (0, 2), 'vehicle_type': 'F1',
        'paths': [[(0, 1), (1, 2)]],
        'route_agent_ids': {0: []},
        'entries': [{'station_cost': 0.0}],
        'route_ids': ['unused'],
    }]
    details = runner._check_route_details()
    assert details[((0, 2), 'F1')][0]['travel_time'] == pytest.approx(30.0)
