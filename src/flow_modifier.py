#!/usr/bin/env python3
"""
Minimal EV Charger Flow Heatmap Modifier
========================================
Standalone script to load results, modify flows, and plot heatmaps

INSTRUCTIONS:
1. Change the pickle_path and config_data variable to point to your pickle file
2. Modify the flow editing section (marked with ###) to suit your needs
3. Run: python flow_modifier.py

REQUIREMENTS: pip install numpy matplotlib pandas geopandas shapely
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

def load_pickle_results(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def get_configuration_data(results, charger_config):
    if isinstance(charger_config, (list, frozenset, set)):
        config_key = tuple(sorted(charger_config))
    elif isinstance(charger_config, tuple):
        config_key = charger_config
    else:
        config_key = (charger_config,)
    
    if config_key not in results['configurations']:
        available_configs = list(results['configurations'].keys())
        raise ValueError(f"Configuration {charger_config} not found. Available: {available_configs}")
    
    return results['configurations'][config_key]

def create_network_structure(results):
    if 'network' not in results:
        raise ValueError("Network geometry not found in pickle file")
    
    network = results['network']
    nodes_df = network.net.nodes.copy()
    edges_df = network.net.edges.copy().sort_values('link_id').reset_index(drop=True)
    return nodes_df, edges_df

def plot_flow_heatmap_comparison(nodes_df, edges_df, original_flows, modified_flows, 
                                charger_config, title_suffix="", figsize=(16, 8)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    _plot_single_heatmap(ax1, nodes_df, edges_df, original_flows, charger_config, f"Original Flows {title_suffix}")
    _plot_single_heatmap(ax2, nodes_df, edges_df, modified_flows, charger_config, f"Modified Flows {title_suffix}")
    
    plt.tight_layout()
    plt.show()

def _plot_single_heatmap(ax, nodes_df, edges_df, link_flows, charger_config, title):
    edges_plot = edges_df.copy()
    flow_values = np.zeros(len(edges_plot))
    
    for _, row in edges_plot.iterrows():
        link_id = row['link_id']
        if link_id in link_flows:
            flow_values[link_id] = link_flows[link_id]['total_flow']
    
    edges_plot['flow'] = flow_values
    edges_plot['geometry'] = gpd.GeoSeries.from_wkt(edges_plot['geometry'])
    gdf = gpd.GeoDataFrame(edges_plot, geometry='geometry')
    
    max_linewidth, min_linewidth, flow_threshold = 8, 0.5, 1.0
    
    if gdf['flow'].max() > 0:
        norm = mcolors.Normalize(vmin=0, vmax=gdf['flow'].max())
        linewidths = min_linewidth + (gdf['flow'] / max(1e-6, gdf['flow'].max())) * (max_linewidth - min_linewidth)
        sm = ScalarMappable(norm=norm, cmap="plasma")
        gdf['color'] = gdf['flow'].apply(lambda f: sm.to_rgba(f))
        gdf['is_low_flow'] = gdf['flow'] < flow_threshold
        
        # Plot links
        for geom, color, lw, is_low_flow in zip(gdf['geometry'], gdf['color'], linewidths, gdf['is_low_flow']):
            x, y = geom.xy
            if is_low_flow:
                color_rgba = list(color)
                color_rgba[3] = 0.3
                ax.plot(x, y, color=color_rgba, linewidth=min_linewidth, linestyle='--')
            else:
                ax.plot(x, y, color=color, linewidth=lw)
                if len(x) >= 2:
                    try:
                        mid_idx = len(x) // 2 - 1
                        ax.annotate('', xy=(x[mid_idx + 1], y[mid_idx + 1]), xytext=(x[mid_idx], y[mid_idx]),
                                   arrowprops=dict(arrowstyle='->', color=color, lw=min(lw, 3)))
                    except:
                        pass
        
        plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04).set_label('Link Flow', fontsize=10)
    
    # Plot nodes and chargers
    ax.scatter(nodes_df['lon'], nodes_df['lat'], color='black', s=3, zorder=3)
    
    if charger_config:
        charger_list = list(charger_config) if isinstance(charger_config, (frozenset, set)) else charger_config
        for i, charger_id in enumerate(charger_list):
            if charger_id < len(nodes_df):
                charger_lon = nodes_df.iloc[charger_id]['lon']
                charger_lat = nodes_df.iloc[charger_id]['lat']
                ax.scatter(charger_lon, charger_lat, c='red', s=100, marker='*', edgecolors='black', zorder=5,
                          label='Charger' if i == 0 else "")
        ax.legend(loc='upper right', fontsize=8)
    
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

# Main execution and flow modification
if __name__ == "__main__":
    # STEP 1: Load results and select configuration
    pickle_path = "results/2025-07-17_16-37-51_n=48 d=1 possible_charger_positions=5 num_chargers=2/all_optimization_results.pkl"
    results = load_pickle_results(pickle_path)
    
    print("Available configurations:")
    for config_key in results['configurations'].keys():
        obj_value = results['configurations'][config_key]['objective_value']
        print(f"  Chargers {list(config_key)}: Objective = {obj_value:.2f}")
    
    # Select configuration (change this to use different charger combinations)
    best_config = min(results['configurations'].keys(), key=lambda k: results['configurations'][k]['objective_value'])
    print(f"\nUsing configuration: {list(best_config)}")
    config_data = get_configuration_data(results, best_config)
    
    # STEP 2: Create modified flows starting from original
    original_link_flows = config_data['link_flows']
    modified_link_flows = {}
    
    # Initialize modified flows with original values
    for link_id, link_data in original_link_flows.items():
        modified_link_flows[link_id] = {
            'start_node_id': link_data['start_node_id'],
            'end_node_id': link_data['end_node_id'],
            'total_flow': link_data['total_flow'],
            'original_flow': link_data['total_flow']
        }
    
    ##################### FLOW MODIFICATION SECTION #####################
    # MODIFY THIS SECTION TO CHANGE FLOWS AS NEEDED
    
    k_metrics = config_data['reconstruction_results']['k_metrics']
    if 32 in k_metrics:  # Change k value here (1, 2, 4, 8, 16, 32, 64)
        print("\nSetting custom flows for top-32 routes...")
        top_k_routes = k_metrics[32]['routes']
        
        # YOUR CUSTOM ROUTE FLOWS - MODIFY THIS LIST:
        # Set new flow values for each route (must have same length as top_k_routes)
        # EXAMPLES:
        # custom_route_flows = [10, 20, 15, 25, 30, 12, 18, 22, 28, 14, ...] # Specific values
        # custom_route_flows = [route['flow'] * 1.5 for route in top_k_routes]  # 50% increase
        # custom_route_flows = [max(route['flow'], 20) for route in top_k_routes]  # Minimum 20
        custom_route_flows = [route['flow'] * 2.0 for route in top_k_routes]  # Default: double all flows
        
        if len(custom_route_flows) != len(top_k_routes):
            print(f"Warning: custom_route_flows length ({len(custom_route_flows)}) doesn't match routes ({len(top_k_routes)})")
            custom_route_flows = custom_route_flows[:len(top_k_routes)]  # Truncate if too long
        
        # Apply custom flows to routes
        for i, route in enumerate(top_k_routes):
            path = route['links']
            original_flow = route['flow']
            new_flow = custom_route_flows[i] if i < len(custom_route_flows) else original_flow
            
            # Calculate the difference to add to links
            flow_change = new_flow - original_flow
            
            # Apply flow change to each link in the route
            for j in range(len(path) - 1):
                start_node, end_node = path[j], path[j + 1]
                for link_id, link_data in modified_link_flows.items():
                    if link_data['start_node_id'] == start_node and link_data['end_node_id'] == end_node:
                        modified_link_flows[link_id]['total_flow'] += flow_change
                        break
        
        print(f"Applied custom flows to {len(top_k_routes)} routes")
        print(f"Total flow change: {sum(custom_route_flows) - sum(route['flow'] for route in top_k_routes):.2f}")
    
    ##################### END MODIFICATION SECTION #####################
    
    # STEP 3: Create network structure and plot comparison
    nodes_df, edges_df = create_network_structure(results)
    
    print("\nPlotting comparison...")
    plot_flow_heatmap_comparison(nodes_df, edges_df, original_link_flows, modified_link_flows, 
                                best_config, "(Modified flows)")
    
    print("\nDone! Left: Original flows, Right: Modified flows, Red stars: Chargers") 