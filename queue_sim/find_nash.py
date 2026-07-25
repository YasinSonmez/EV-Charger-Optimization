"""Find Nash equilibrium path assignments for queue-based simulation.

Adapted from find_nash_assignments.py to read from Config and write to an experiment directory.
Reuses: RoadNet, Runner, _prune_flow_data, _rounded_counts, _simulate, better-response heuristic.
"""
import copy
import os
import pickle
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

from src.road_network import RoadNet
from queue_sim import Runner, QUEUE_SIM_AVAILABLE


def _collapse_repeats(lst):
    if not lst:
        return []
    out = [lst[0]]
    for x in lst[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def _prune_flow_data(data, charger_locs_tuple, k):
    flow_data = {'no charging type': [], 'charging type': []}
    for r in data['configurations'][charger_locs_tuple]['reconstruction_results']['k_metrics'][k]['routes']:
        if r['type'] == 'non_charging':
            flow_data['no charging type'].append({'path': r['links'], 'flow': r['flow'], 'station node': None})
        else:
            flow_data['charging type'].append({'path': _collapse_repeats(r['links']), 'flow': r['flow'], 'station node': r['charger']})
    return flow_data


def _rounded_counts(routes, total):
    raw = [r['flow'] for r in routes]
    tot = sum(raw)
    if tot == 0:
        return [0] * len(routes)
    floored = [int(f / tot * total) for f in raw]
    remainder = total - sum(floored)
    fracs = sorted(enumerate(raw[i] / tot * total - floored[i] for i in range(len(raw))), key=lambda x: -x[1])
    for i in range(remainder):
        floored[fracs[i][0]] += 1
    return floored


def _simulate(num_iters, charger_locs_tuple, flow_data, ch_counts, no_ch_counts,
              orig, dest, num_of_vehs, num_charging, name, work_dir,
              ent_cap, ch_cap, ex_cap, cost):
    travel_times = []
    for _ in range(num_iters):
        r = Runner(
            nodes_csv=os.path.join(work_dir, f'traffic_inputs_{name}_nodes.csv'),
            links_csv=os.path.join(work_dir, f'traffic_inputs_{name}_edges.csv'),
            od_csv=os.path.join(work_dir, f'traffic_inputs_{name}_od.csv'),
        )
        r.add_charging_info(1, num_charging / num_of_vehs)
        for loc in charger_locs_tuple:
            r.create_EV_charging_station_at_node(loc, ent_cap, ch_cap, ex_cap, cost)
        r.init_sq_simulation_with_path_assignment({(orig, dest): flow_data}, ch_counts, no_ch_counts)
        r.spatial_queue_simulation(name, output_dir=os.path.join(work_dir, 'traffic_outputs'))
        travel_times.append(r.check_NE())
    ch_all, no_ch_all = zip(*travel_times)
    avg = lambda tl: [sum(t) / num_iters for t in zip(*tl)]
    return avg(ch_all), avg(no_ch_all)


def _nash_for_config(args):
    (charger_locs_tuple, file_path, k, thresh, num_iters,
     orig, dest, num_of_vehs, num_charging, name, work_dir,
     ent_cap, ch_cap, ex_cap, cost) = args

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f'  Nash for [{",".join(map(str, charger_locs_tuple))}]: FAILED to load pkl: {e}')
        return ",".join(map(str, charger_locs_tuple)), [], [], []

    flow_data = _prune_flow_data(data, charger_locs_tuple, k)
    ed_ch = _rounded_counts(flow_data['charging type'], num_charging)
    ed_no_ch = _rounded_counts(flow_data['no charging type'], num_of_vehs - num_charging)

    diff = thresh + 1
    iters = 0
    diff_history = []

    try:
        while diff > thresh and iters < 200:
            ave_ch, ave_no_ch = _simulate(
                num_iters, charger_locs_tuple, flow_data, ed_ch, ed_no_ch,
                orig, dest, num_of_vehs, num_charging, name, work_dir,
                ent_cap, ch_cap, ex_cap, cost,
            )

            m_ch = [ave_ch[i] if ed_ch[i] > 0 else 0 for i in range(len(ave_ch))]
            m_nch = [ave_no_ch[i] if ed_no_ch[i] > 0 else 0 for i in range(len(ave_no_ch))]

            max_ch_i = max(range(len(m_ch)), key=lambda i: m_ch[i])
            min_ch_i = min(range(len(ave_ch)), key=lambda i: ave_ch[i])
            ed_ch[max_ch_i] -= 1
            ed_ch[min_ch_i] += 1

            max_nch_i = max(range(len(m_nch)), key=lambda i: m_nch[i])
            min_nch_i = min(range(len(ave_no_ch)), key=lambda i: ave_no_ch[i])
            ed_no_ch[max_nch_i] -= 1
            ed_no_ch[min_nch_i] += 1

            vals_ch = [v for v in m_ch if v > 0]
            vals_nch = [v for v in m_nch if v > 0]
            diff = (max(vals_ch) - min(vals_ch) if len(vals_ch) > 1 else 0) + \
                   (max(vals_nch) - min(vals_nch) if len(vals_nch) > 1 else 0)
            diff_history.append(diff)
            iters += 1
    except Exception as e:
        loc_str = ','.join(map(str, charger_locs_tuple))
        print(f'  Nash for [{loc_str}]: SKIPPED (sim error: {e})')
        return loc_str, [], [], []

    loc_str = ','.join(map(str, charger_locs_tuple))
    print(f'  Nash for [{loc_str}]: {iters} iters, diff={diff:.1f}')
    return loc_str, ed_ch, ed_no_ch, diff_history


def find_nash_assignments(config, experiment_dir, all_opt_results_path, network_name='college_park'):
    """Find NE path assignments for all charger configs.

    Args:
        config: Config instance with queue_simulation params.
        experiment_dir: Path to the experiment directory (CG results already saved here).
        all_opt_results_path: Path to all_optimization_results.pkl.
        network_name: Name for the network (affects CSV filenames).

    Returns:
        (ne_pkl_path, convergence_data) where convergence_data is {config_str: [diff per iter]}.
    """
    if not QUEUE_SIM_AVAILABLE:
        raise RuntimeError(f"Queue simulation not available: {__import__('queue_sim')._QUEUE_SIM_ERROR}")

    q = config.queue_simulation
    k = q['K']
    thresh = q['THRESH']
    num_iters = q['NUM_ITERS']
    workers = q['WORKERS']
    ent_cap = q['ENT_CAPACITY']
    ch_cap = q['CHARGING_CAPACITY']
    ex_cap = q['EXIT_CAPACITY']
    cost = q['COST']

    work_dir = os.path.join(experiment_dir, 'queue')
    os.makedirs(work_dir, exist_ok=True)
    for sub in ('t_stats', 'link_stats', 'node_stats'):
        os.makedirs(os.path.join(work_dir, 'traffic_outputs', sub), exist_ok=True)

    with open(all_opt_results_path, 'rb') as f:
        data = pickle.load(f)

    od = data['run_configuration']['od_demand']
    orig, dest = next(iter(od))
    num_charging = od[(orig, dest)][1]
    num_not_charging = od[(orig, dest)][0]
    num_of_vehs = num_charging + num_not_charging

    print(f'Origin: {orig}  Destination: {dest}')
    print(f'Vehicles: {num_of_vehs}  ({num_charging} charging, {num_not_charging} non-charging)')

    coords = data['run_configuration']['coordinates']
    rf = config.road_filter
    highway_types = rf.get('highway_types') if rf.get('enabled', True) else None
    prune_de = rf.get('prune_dead_ends', True)
    road_net = RoadNet('College Park')
    road_net.get_map(*coords, highway_types=highway_types, prune_dead_ends=prune_de)
    road_net.set_exit(False, dest)
    road_net.create_demand_with_orig_dest(num_of_vehs, orig, dest)
    road_net.save_data(save_dir=work_dir)
    print('Network CSV files saved.')

    all_configs = list(data['configurations'].keys())
    print(f'Finding Nash equilibria for {len(all_configs)} configurations '
          f'(K={k}, THRESH={thresh}, NUM_ITERS={num_iters}) ...')

    args_list = [
        (locs, all_opt_results_path, k, thresh, num_iters, orig, dest,
         num_of_vehs, num_charging, network_name, work_dir,
         ent_cap, ch_cap, ex_cap, cost)
        for locs in all_configs
    ]

    with Pool(workers) as pool:
        results = pool.map(_nash_for_config, args_list)

    ne_path_assignments = {}
    convergence_data = {}
    for loc_str, ch, no_ch, diff_hist in results:
        if ch or no_ch:
            ne_path_assignments[loc_str] = {'ch': ch, 'no_ch': no_ch}
            convergence_data[loc_str] = diff_hist

    ne_pkl_path = os.path.join(work_dir, 'NE_path_assignments.pkl')
    with open(ne_pkl_path, 'wb') as f:
        pickle.dump(ne_path_assignments, f)

    print(f'\nNash assignments saved to {ne_pkl_path}')
    return ne_pkl_path, convergence_data
