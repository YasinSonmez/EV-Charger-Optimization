#!/usr/bin/env python3
"""Generate BPR cache for the new merge pipeline. Run once per topology change."""
import sys, os, warnings, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')


def main():
    import pandas as pd, numpy as np
    from src.model_fitter import TrafficModelFitter, convert_string_to_array
    from queue_sim.bpr_data_generator import generate_and_save_bpr_data

    coord_hash = abs(hash(tuple([38.98211, 38.975, -76.93006, -76.93704]))) % (10 ** 8)
    tag = '_pruned_merged'
    data_path = f'data/traffic_data_{coord_hash}{tag}.csv'
    cache_path = f'data/cached_results_{coord_hash}{tag}.pkl'

    print(f'Generating BPR data for CP {tag}...')
    df, n_links = generate_and_save_bpr_data(
        coordinates=[38.98211, 38.975, -76.93006, -76.93704],
        output_path=data_path, num_samples=3, max_flow=30,
        prune_dead_ends=True,
    )
    print(f'  {len(df)}/{n_links} links with sim data')

    print('Fitting BPR models...')
    df2 = pd.read_csv(data_path)
    convert_string_to_array(df2, 'x_vector')
    convert_string_to_array(df2, 'y_vector')
    mf = TrafficModelFitter(pandas_df=df2)
    mf.parallel_fit_and_evaluate()
    mf.fill_missing_link_ids()
    pdf = mf.df
    missing = sorted(set(range(n_links)) - set(pdf['link_id'].unique()))
    if missing:
        rows = [{'link_id': lid, 'x_vector': np.zeros(3), 'y_vector': np.zeros(3),
                 'a_fit': 0.0, 'b_fit': 0.0, 'cap_fit': 1.0, 'fft_fit': 1.0, 'R^2': 1.0}
                for lid in missing]
        pdf = pd.concat([pdf, pd.DataFrame(rows)], ignore_index=True).sort_values('link_id').reset_index(drop=True)
        print(f'  Filled {len(missing)} missing links')
    with open(cache_path, 'wb') as f:
        pickle.dump((pdf, mf), f)
    print(f'Cached {len(pdf)} links to {cache_path}')


if __name__ == '__main__':
    main()