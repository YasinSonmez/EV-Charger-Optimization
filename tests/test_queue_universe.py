"""Contract tests for the queue-stage placement universe expansion."""

from types import SimpleNamespace

import pytest

from queue_sim.find_nash import _expanded_queue_configs
from src.contracts import enumerate_placements


def _config(num_chargers, route_source='independent_network_routes'):
    return SimpleNamespace(
        num_chargers=num_chargers,
        possible_charger_positions=[1, 2, 3, 4],
        queue_simulation={'route_source': route_source},
    )


def test_expansion_covers_every_intermediate_size():
    data = {'configurations': {(1,): {}, (1, 2): {}, (1, 2, 3): {}, (1, 2, 4): {}}}
    configs = _expanded_queue_configs(_config(3), data)
    assert configs == [
        (1,), (2,), (3,), (4,),
        (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
    ]


def test_cg_recovered_routes_require_the_complete_universe():
    data = {'configurations': {(1,): {}, (1, 2): {}, (1, 2, 3): {}, (1, 2, 4): {}}}
    with pytest.raises(ValueError, match='missing placements'):
        _expanded_queue_configs(_config(3, 'cg_recovered_top_k'), data)


def test_six_candidate_three_charger_universe_has_41_placements():
    candidates = [0, 20, 28, 59, 80, 101]
    config = SimpleNamespace(
        num_chargers=3,
        possible_charger_positions=candidates,
        queue_simulation={'route_source': 'cg_recovered_top_k'},
    )
    placements = {
        value: {}
        for size in range(1, 4)
        for value in enumerate_placements(candidates, size)
    }
    configs = _expanded_queue_configs(config, {'configurations': placements})
    assert len(configs) == 41
    assert (0, 80) in configs
