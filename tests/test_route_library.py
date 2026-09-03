"""Route-library tests for multi-OD and multi-vehicle coverage."""

from __future__ import annotations

import numpy as np

from src.utils import analyze_route_reconstruction


class _ReconstructionFixture:
    def reconstruct_route_flows(self, *args, **kwargs):
        return {
            (0, 2): {
                "non_charging": [{"path": [0, 1], "link_ids": [0], "flow": 2.0}],
                "charging": {4: [{"path": [0, 1], "link_ids": [0, 1], "flow": 1.0}]},
            },
            (1, 3): {
                "non_charging": [{"path": [1, 2], "link_ids": [1], "flow": 1.0}],
                "charging": {4: [{"path": [1, 2], "link_ids": [1, 2], "flow": 2.0}]},
            },
        }

    def _path_to_link_ids(self, path):
        return list(path)


def test_top_k_route_library_preserves_every_od_and_vehicle_class():
    link_flows = {
        link_id: {"total_flow": 1.0, "start_node_id": 0, "end_node_id": 1}
        for link_id in range(3)
    }

    result = analyze_route_reconstruction(_ReconstructionFixture(), link_flows, k_values=[1])
    routes = result["k_metrics"][1]["routes"]
    groups = {(route["origin"], route["destination"], route["type"]) for route in routes}

    assert groups == {
        (0, 2, "non_charging"),
        (0, 2, "charging"),
        (1, 3, "non_charging"),
        (1, 3, "charging"),
    }
    assert all(np.isfinite(route["flow"]) for route in routes)
