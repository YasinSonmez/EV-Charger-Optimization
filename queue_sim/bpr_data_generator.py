"""Generate per-link BPR fitting data by running the queue simulator at varying flows.

This is Step 1 of the paper's methodology (Section V-B): for each physical link,
simulate over a range of aggregate flows to obtain {(x_l, d_l)} pairs, then fit
BPR delay functions.

Uses the simulator's init_sq_simulation_for_bpr_function_fitting_V2 to force
agents through each link, and return_traffic_data to extract the results.

Output: traffic_data.csv with columns [link_id, x_vector, y_vector] where
x_vector and y_vector are lists of flows and travel times.

Requires macOS (liblsp.dylib).
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.road_network import RoadNet
from queue_sim import Runner, QUEUE_SIM_AVAILABLE

# Consistent network name - must match what RoadNet.save_data generates
NETWORK_NAME = 'college_park'


def generate_bpr_data(coordinates, num_samples=25, max_flow=250,
                      work_dir=None, highway_types=None, prune_dead_ends=False):
    """Generate BPR fitting data for every link in the network.

    For each link, forces agents through the straight-ahead path:
      sa_in_link -> target_link -> sa_out_link

    The OD pair is set per-link to match: origin = sa_in_link.start, dest = sa_out_link.end.

    Args:
        coordinates: [north, south, east, west] bbox.
        num_samples: Number of flow levels to sample per link.
        max_flow: Maximum flow (number of vehicles) to push through each link.
        work_dir: Directory for network CSVs and sim outputs.
        highway_types: If provided, filter OSM to these road types only.
        prune_dead_ends: If True, iteratively remove dead-end nodes before BPR generation.

    Returns:
        Tuple (pd.DataFrame with columns [link_id, x_vector, y_vector], n_links_total).
    """
    if not QUEUE_SIM_AVAILABLE:
        raise RuntimeError("Queue simulation not available (requires macOS)")

    if work_dir is None:
        work_dir = os.getcwd()
    for sub in ('t_stats', 'link_stats', 'node_stats'):
        os.makedirs(os.path.join(work_dir, 'traffic_outputs', sub), exist_ok=True)

    print("Downloading OSM network...")
    road_net = RoadNet('College Park')
    road_net.get_map(*coordinates, highway_types=highway_types, prune_dead_ends=prune_dead_ends)

    edges = road_net.edges
    n_links = len(edges)
    n_nodes = len(road_net.nodes)
    print(f"Network: {n_nodes} nodes, {n_links} links")

    # Save network CSVs ONCE - these don't change per link iteration
    # Use a dummy OD to satisfy the CSV format requirement
    orig = int(road_net.nodes.iloc[0]['node_id'])
    dest = int(road_net.nodes.iloc[-1]['node_id'])
    road_net.set_exit(False, dest)
    road_net.create_demand_with_orig_dest(1, orig, dest)
    road_net.save_data(save_dir=work_dir)

    nodes_csv = os.path.join(work_dir, f'traffic_inputs_{NETWORK_NAME}_nodes.csv')
    edges_csv = os.path.join(work_dir, f'traffic_inputs_{NETWORK_NAME}_edges.csv')
    od_csv = os.path.join(work_dir, f'traffic_inputs_{NETWORK_NAME}_od.csv')

    flow_levels = np.linspace(1, max_flow, num_samples).astype(int)
    all_data = []
    links_with_data = 0
    links_failed = 0

    for link_idx in range(n_links):
        link = edges.iloc[link_idx]
        link_id = int(link['link_id'])
        start_nid = int(link['start_node_id'])
        end_nid = int(link['end_node_id'])

        if start_nid == end_nid:
            continue

        x_vector = []
        y_vector = []

        # Find straight-ahead in/out links using a temporary Runner
        try:
            r = Runner(nodes_csv=nodes_csv, links_csv=edges_csv, od_csv=od_csv)
            sa_il = r.find_sa_in_link(link)
            sa_ol = r.find_sa_out_link(link)
        except Exception:
            sa_il = sa_ol = None

        if sa_il is None or sa_ol is None:
            links_failed += 1
            continue

        # Determine OD for this link: origin = sa_il.start, dest = sa_ol.end
        link_orig = int(sa_il['start_node_id'])
        link_dest = int(sa_ol['end_node_id'])

        for flow in flow_levels:
            try:
                # Update demand for this link's OD pair
                road_net.set_exit(False, link_dest)
                road_net.create_demand_with_orig_dest(int(flow), link_orig, link_dest)
                road_net.save_data(save_dir=work_dir)

                r = Runner(nodes_csv=nodes_csv, links_csv=edges_csv, od_csv=od_csv)
                r.add_charging_info(0, 0)
                r.init_sq_simulation_for_bpr_function_fitting_V2(link, sa_il, sa_ol)
                r.spatial_queue_simulation(NETWORK_NAME,
                                           output_dir=os.path.join(work_dir, 'traffic_outputs'))

                traffic_df = r.return_traffic_data(int(flow), link_orig, link_dest)
                target_row = traffic_df[traffic_df['link_id'] == link_id]
                if len(target_row) > 0 and not pd.isna(target_row.iloc[0]['travel_time']):
                    x_vector.append(float(flow))
                    y_vector.append(float(target_row.iloc[0]['travel_time']))
            except Exception:
                continue

        if len(x_vector) >= 2:
            all_data.append({
                'link_id': link_id,
                'x_vector': x_vector,
                'y_vector': y_vector,
            })
            links_with_data += 1
        else:
            links_failed += 1
            print(f"  WARNING: Link {link_id} produced no BPR data "
                  f"(no straight-ahead in/out link or sim failure)")

        if (links_with_data + links_failed) % 10 == 0:
            print(f"  Processed {links_with_data + links_failed}/{n_links} "
                  f"({links_with_data} with data, {links_failed} failed)...")

    print(f"Generated sim BPR data for {links_with_data} links (out of {n_links})")

    # For links where simulation failed, compute proxy data from physical properties.
    # Uses the link's real length, maxspeed, lanes — not fabricated values.
    # The BPR fitter will calibrate a and b from these (x,y) pairs as it does for all links.
    sim_link_ids = set(d['link_id'] for d in all_data)
    proxy_added = 0
    for link_idx in range(n_links):
        link = edges.iloc[link_idx]
        link_id = int(link['link_id'])
        if link_id in sim_link_ids:
            continue
        if int(link['start_node_id']) == int(link['end_node_id']):
            continue
        length = float(link.get('length', 1))
        maxmph = float(link.get('maxmph', 25.0))
        lanes = float(link.get('lanes', 1.0))
        fft = length / (maxmph * 2.2369) if maxmph > 0 else 1.0
        # Generate proxy (flow, travel_time) pairs at free-flow
        x_vec = [float(x) for x in np.linspace(1, max_flow, num_samples)]
        y_vec = [float(fft) for _ in x_vec]  # constant = free-flow time
        all_data.append({
            'link_id': link_id,
            'x_vector': x_vec,
            'y_vector': y_vec,
        })
        proxy_added += 1

    if proxy_added > 0:
        print(f"Added proxy BPR data for {proxy_added} links (physical properties).")
    print(f"Total links with BPR data: {links_with_data + proxy_added} (out of {n_links})")

    df = pd.DataFrame(all_data)
    return df, n_links


def generate_and_save_bpr_data(coordinates, output_path, **kwargs):
    """Generate BPR data and save to CSV.

    Args:
        coordinates: [north, south, east, west] bbox.
        output_path: Path to save traffic_data.csv.
        **kwargs: Passed to generate_bpr_data.

    Returns:
        Tuple (pd.DataFrame, n_links_total).
    """
    df, n_links = generate_bpr_data(coordinates, **kwargs)
    df.to_csv(output_path, index=False)
    # Save network link count alongside the data
    meta_path = output_path.replace('.csv', '_meta.json')
    import json
    with open(meta_path, 'w') as f:
        json.dump({'n_links': n_links}, f)
    print(f"BPR data saved to {output_path}")
    return df, n_links


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate BPR fitting data from queue simulation')
    parser.add_argument('--coordinates', type=str, default=None,
                        help='Comma-separated bbox: north,south,east,west')
    parser.add_argument('--output', type=str, default='data/traffic_data.csv',
                        help='Output CSV path')
    parser.add_argument('--num-samples', type=int, default=25,
                        help='Number of flow levels per link')
    parser.add_argument('--max-flow', type=int, default=250,
                        help='Maximum flow (vehicles) per link')
    args = parser.parse_args()

    if args.coordinates:
        coords = [float(x) for x in args.coordinates.split(',')]
    else:
        coords = [38.98211, 38.975, -76.93006, -76.93704]

    generate_and_save_bpr_data(coords, args.output, num_samples=args.num_samples, max_flow=args.max_flow)