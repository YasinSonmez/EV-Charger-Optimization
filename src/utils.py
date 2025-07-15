import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
import re
import glob
from datetime import datetime
from itertools import combinations
from math import comb
import shutil # For copying the file
import json # Added for loading config file

from src.traffic_optimizer import Network

def save_flow_heatmap(grid, output_path, title, use_cvxpy=True, flows=None, flow_threshold=1.0):
    """Save a flow heatmap for a single configuration without displaying it"""
    # Create a figure and suppress display
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get flow data
    if flows is None:
        flows = grid.cvxpy_link_flows if use_cvxpy else grid.flow
    if flows is None:
        print(f"No flow data available for grid with chargers {grid.chargers}.")
        plt.close(fig)
        return False
    
    # Create GeoDataFrame for plotting
    edges_df = grid.net.edges.sort_values("link_id").copy()
    if len(flows) != len(edges_df):
        print(f"Flow length {len(flows)} does not match number of edges {len(edges_df)}.")
        plt.close(fig)
        return False
    
    from matplotlib.cm import ScalarMappable
    import matplotlib.colors as mcolors
    import geopandas as gpd
    
    # Setup the plot
    edges_df["flow"] = flows
    edges_df["geometry"] = gpd.GeoSeries.from_wkt(edges_df["geometry"])
    gdf = gpd.GeoDataFrame(edges_df, geometry="geometry")

    # Setup the coloring and sizing
    max_linewidth = 10
    min_linewidth = 1
    cmap = "plasma"
    norm = mcolors.Normalize(vmin=gdf["flow"].min(), vmax=gdf["flow"].max())
    linewidths = min_linewidth + (gdf["flow"] - gdf["flow"].min()) / max(1e-6, gdf["flow"].max() - gdf["flow"].min()) * (max_linewidth - min_linewidth)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    gdf["color"] = gdf["flow"].apply(lambda f: sm.to_rgba(f))
    
    # Identify low-flow links for special treatment
    gdf["is_low_flow"] = gdf["flow"] < flow_threshold

    # Plot links
    for geom, color, lw, is_low_flow in zip(gdf["geometry"], gdf["color"], linewidths, gdf["is_low_flow"]):
        x, y = geom.xy
        
        # Handle low flow links differently
        if is_low_flow:
            # Make low flow links very transparent (alpha=0.2)
            color_rgba = list(color)
            color_rgba[3] = 0.2  # Set alpha to 0.2
            ax.plot(x, y, color=color_rgba, linewidth=min_linewidth, linestyle='--')
        else:
            # Plot normal flow links
            ax.plot(x, y, color=color, linewidth=lw)
            
            # Add direction arrow only for links with flow above threshold
            if len(x) >= 2:
                try:
                    mid_idx = len(x) // 2 - 1
                    ax.annotate(
                        '', xy=(x[mid_idx + 1], y[mid_idx + 1]),
                        xytext=(x[mid_idx], y[mid_idx]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw)
                    )
                except IndexError:
                    continue  # skip problematic geometries

    # Plot all nodes as black circles
    ax.scatter(grid.net.nodes["lon"], grid.net.nodes["lat"], color="black", s=10, zorder=3, label="Node")

    # Plot OD nodes with demand annotation
    if hasattr(grid, 'od_pairs'):
        for i, (o, d) in enumerate(grid.od_pairs):
            o_lon = grid.net.nodes.at[o, "lon"]
            o_lat = grid.net.nodes.at[o, "lat"]
            d_lon = grid.net.nodes.at[d, "lon"]
            d_lat = grid.net.nodes.at[d, "lat"]
            
            demand_no_charging = grid.b[2 * i]
            demand_charging = grid.b[2 * i + 1]

            ax.scatter(o_lon, o_lat, c='blue', s=100, edgecolors='black', label='Origin' if i == 0 else "", zorder=4)
            ax.text(o_lon, o_lat + 0.0002, f"O{i}: {demand_no_charging:.0f}/{demand_charging:.0f}",
                    fontsize=9, ha='center', color='blue', zorder=5)

            ax.scatter(d_lon, d_lat, c='red', s=100, edgecolors='black', label='Destination' if i == 0 else "", zorder=4)
            ax.text(d_lon, d_lat - 0.0002, f"D{i}: {demand_no_charging:.0f}/{demand_charging:.0f}",
                    fontsize=9, ha='center', color='red', zorder=5)

    # Final plot elements
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    # Add colorbar
    sm.set_array(gdf["flow"])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label("Link Flow", fontsize=12)

    # Add legend if needed
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='lower left', fontsize=10)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True

def save_optimization_pickle(grid):
    """Constructs a dictionary of optimization results for a given grid."""
    import numpy as np

    results_data = {}
    results_data['charger_combination'] = list(grid.chargers) if grid.chargers is not None else []

    link_flows_dict = {}
    all_edges = grid.net.edges.sort_values("link_id").copy()

    if hasattr(grid, 'cvxpy_link_flows') and grid.cvxpy_link_flows is not None: # CVXPY case
        results_data['objective_value'] = grid.best_objective_value
        results_data['method'] = 'cvxpy'
        raw_link_flows = grid.cvxpy_link_flows
        charger_throughputs = getattr(grid, 'cvxpy_charger_throughput', None)
        
        # Get charger contributions if available
        charger_contributions = grid.get_charger_contributions() if hasattr(grid, 'get_charger_contributions') else None
        
        charger_node_to_throughput_map = {}
        if charger_throughputs is not None and grid.chargers is not None and len(grid.chargers) == len(charger_throughputs):
            for i, charger_node_id in enumerate(grid.chargers):
                 charger_node_to_throughput_map[charger_node_id] = charger_throughputs[i]
        elif charger_throughputs is not None:
            print(f"Warning: Charger throughputs available but cannot map to charger nodes for {grid.chargers}")

        for _, edge_row in all_edges.iterrows():
            link_id = int(edge_row['link_id'])
            start_node = int(edge_row['start_node_id'])
            end_node = int(edge_row['end_node_id'])
            
            # Initialize flow components
            flow_data = {
                'start_node_id': start_node,
                'end_node_id': end_node,
                'total_flow': 0.0,
                'non_charging_flow': 0.0,
                'charging_flows': {}
            }
            
            is_charger_self_link = (start_node == end_node and start_node in charger_node_to_throughput_map)

            if is_charger_self_link:
                flow_data['total_flow'] = charger_node_to_throughput_map.get(start_node, 0.0)
                # For charger self-links, all flow is charging flow
                flow_data['charging_flows'][start_node] = flow_data['total_flow']
            elif link_id < len(raw_link_flows): # Regular road link
                flow_data['total_flow'] = float(raw_link_flows[link_id])
                
                # Add non-charging flow if available
                if charger_contributions and 'non_charging' in charger_contributions:
                    flow_data['non_charging_flow'] = float(charger_contributions['non_charging'][link_id])
                
                # Add individual charger contributions if available
                if charger_contributions:
                    for charger_key, flows in charger_contributions.items():
                        if charger_key != 'non_charging':
                            flow_data['charging_flows'][charger_key] = float(flows[link_id])
            
            link_flows_dict[link_id] = flow_data

    elif hasattr(grid, 'flow') and grid.flow is not None:
        results_data['objective_value'] = grid.travel_time_obj
        results_data['method'] = 'scipy'
        
        # Create a full flow array initialized to zeros
        full_scipy_flows = np.zeros(grid.l)
        if hasattr(grid, 'active_link_indices') and grid.active_link_indices is not None and len(grid.active_link_indices) == len(grid.flow):
            full_scipy_flows[grid.active_link_indices] = grid.flow
        elif len(grid.flow) == grid.l:
            full_scipy_flows = grid.flow
        else:
            print(f"Warning: SciPy flow array (len {len(grid.flow)}) and active_link_indices mismatch or grid.l (len {grid.l}) issue for {grid.chargers}. Some flows may be zero.")
            if len(grid.flow) < grid.l:
                full_scipy_flows[:len(grid.flow)] = grid.flow

        for _, edge_row in all_edges.iterrows():
            link_id = int(edge_row['link_id'])
            flow_val = 0.0
            if link_id < len(full_scipy_flows):
                flow_val = full_scipy_flows[link_id]
            else:
                print(f"Warning: link_id {link_id} out of bounds for SciPy full_scipy_flows (len {len(full_scipy_flows)}). Flow set to 0.")

            link_flows_dict[link_id] = {
                'start_node_id': int(edge_row['start_node_id']),
                'end_node_id': int(edge_row['end_node_id']),
                'total_flow': float(flow_val),
                'non_charging_flow': float(flow_val),  # In SciPy case, we can't separate flows
                'charging_flows': {}  # Empty dict for SciPy case as we don't have this information
            }

    else: # Fallback if no flow data is found
        print(f"Warning: No flow data found for grid with chargers {grid.chargers}. Cannot save link flows.")
        results_data['objective_value'] = getattr(grid, 'best_objective_value', getattr(grid, 'travel_time_obj', float('inf')))
        results_data['method'] = 'unknown'
        for _, edge_row in all_edges.iterrows():
            link_id = int(edge_row['link_id'])
            link_flows_dict[link_id] = {
                'start_node_id': int(edge_row['start_node_id']),
                'end_node_id': int(edge_row['end_node_id']),
                'total_flow': 0.0,
                'non_charging_flow': 0.0,
                'charging_flows': {}
            }

    results_data['link_flows'] = link_flows_dict
    return results_data

def save_charger_flow_heatmaps(grid, output_folder, base_filename, use_cvxpy=True, flow_threshold=1.0):
    """Save a set of flow heatmaps showing contributions from each charger"""
    if not use_cvxpy or not hasattr(grid, 'get_charger_contributions'):
        print("Charger flow contributions can only be visualized with CVXPY optimization.")
        return False
    
    # Get flow contributions from each charger
    contributions = grid.get_charger_contributions()
    if contributions is None:
        return False
    
    # Create a figure with subplots for total flow, non-charging flow, and each charger
    n_plots = len(contributions) + 1  # +1 for the total flow
    n_cols = min(n_plots, 3)  # Limit to at most 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.ioff()  # Turn off interactive mode
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle the case where axes is a 1D array or even a single axis
    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten the axes array for easier indexing if it's 2D
    if n_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes
    
    # Get the total flows and determine global min/max for consistent colormaps
    total_flows = grid.cvxpy_link_flows
    all_flows = [total_flows]
    all_flows.append(contributions["non_charging"])
    for charger_id in grid.chargers:
        all_flows.append(contributions[charger_id])
    
    global_max_flow = max([flow.max() for flow in all_flows])
    global_min_flow = 0  # Usually 0 is the minimum flow

    # Plot the total flow first
    ax_idx = 0
    title = f"Total Flow - TT: {grid.best_objective_value:.2f}"
    _plot_flow_on_axis(grid, axes_flat[ax_idx], title, total_flows, 
                      global_min_flow, global_max_flow, flow_threshold=flow_threshold)
    
    # Plot non-charging flow second
    ax_idx += 1
    if ax_idx < len(axes_flat):
        title = "Non-Charging Flow"
        _plot_flow_on_axis(grid, axes_flat[ax_idx], title, contributions["non_charging"], 
                          global_min_flow, global_max_flow, flow_threshold=flow_threshold)
    
    # Plot each charger's contribution
    for charger_id in grid.chargers:
        ax_idx += 1
        if ax_idx < len(axes_flat):
            title = f"Charger {charger_id} Flow"
            _plot_flow_on_axis(grid, axes_flat[ax_idx], title, contributions[charger_id], 
                              global_min_flow, global_max_flow, flow_threshold=flow_threshold)
    
    # Add a note about the threshold in the figure title
    fig.suptitle(f"Flow Heatmaps (flows < {flow_threshold} shown as dashed lines)", fontsize=14)
    
    # Adjust the layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    # Save the figure
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{base_filename}_charger_flows.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return True

def _plot_flow_on_axis(grid, ax, title, flows, min_flow=None, max_flow=None, flow_threshold=1.0):
    """Helper function to plot a flow heatmap on a specific axis
    
    Parameters:
    -----------
    grid : Network
        The network object containing nodes and edges
    ax : matplotlib.axes.Axes
        The axis to plot on
    title : str
        The title for the plot
    flows : numpy.ndarray
        Array of flow values for each link
    min_flow : float, optional
        Minimum flow value for colormap normalization. If None, uses min of flows.
    max_flow : float, optional
        Maximum flow value for colormap normalization. If None, uses max of flows.
    flow_threshold : float, optional
        Flows below this threshold will be shown with high transparency and dashed lines
    """
    from matplotlib.cm import ScalarMappable
    import matplotlib.colors as mcolors
    import geopandas as gpd
    
    # Make sure ax is a proper matplotlib axis object, not a numpy.ndarray
    if isinstance(ax, np.ndarray):
        print(f"Warning: ax is a numpy.ndarray of shape {ax.shape}. Using plt.gca() instead.")
        ax = plt.gca()
    
    # Create GeoDataFrame for plotting
    edges_df = grid.net.edges.sort_values("link_id").copy()
    edges_df["flow"] = flows
    edges_df["geometry"] = gpd.GeoSeries.from_wkt(edges_df["geometry"])
    gdf = gpd.GeoDataFrame(edges_df, geometry="geometry")

    # Setup the coloring and sizing
    max_linewidth = 8
    min_linewidth = 1
    cmap = "plasma"
    
    # Use provided min/max flow values for consistent colormaps if provided
    if min_flow is None:
        min_flow = gdf["flow"].min()
    if max_flow is None:
        max_flow = gdf["flow"].max()
    
    norm = mcolors.Normalize(vmin=min_flow, vmax=max_flow)
    linewidths = min_linewidth + (gdf["flow"] - min_flow) / max(1e-6, max_flow - min_flow) * (max_linewidth - min_linewidth)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    gdf["color"] = gdf["flow"].apply(lambda f: sm.to_rgba(f))
    
    # Identify low-flow links for special treatment
    gdf["is_low_flow"] = gdf["flow"] < flow_threshold

    # Plot links with directional arrows
    for geom, color, lw, is_low_flow in zip(gdf["geometry"], gdf["color"], linewidths, gdf["is_low_flow"]):
        x, y = geom.xy
        
        # Handle low flow links differently
        if is_low_flow:
            # Make low flow links very transparent (alpha=0.2)
            color_rgba = list(color)
            color_rgba[3] = 0.2  # Set alpha to 0.2
            ax.plot(x, y, color=color_rgba, linewidth=min_linewidth, linestyle='--')
        else:
            # Plot normal flow links
            ax.plot(x, y, color=color, linewidth=lw)
            
            # Add direction arrow only for links with flow above threshold
            if len(x) >= 2:
                try:
                    mid_idx = len(x) // 2 - 1
                    ax.annotate(
                        '', xy=(x[mid_idx + 1], y[mid_idx + 1]),
                        xytext=(x[mid_idx], y[mid_idx]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=min(lw, 3))
                    )
                except IndexError:
                    continue  # skip problematic geometries

    # Plot all nodes as black circles
    ax.scatter(grid.net.nodes["lon"], grid.net.nodes["lat"], color="black", s=5, zorder=3)

    # Plot OD nodes with demand annotation
    if hasattr(grid, 'od_pairs'):
        for i, (o, d) in enumerate(grid.od_pairs):
            o_lon = grid.net.nodes.at[o, "lon"]
            o_lat = grid.net.nodes.at[o, "lat"]
            d_lon = grid.net.nodes.at[d, "lon"]
            d_lat = grid.net.nodes.at[d, "lat"]
            
            demand_no_charging = grid.b[2 * i]
            demand_charging = grid.b[2 * i + 1]

            ax.scatter(o_lon, o_lat, c='blue', s=80, edgecolors='black', label='Origin' if i == 0 else "", zorder=4)
            ax.text(o_lon, o_lat + 0.0002, f"O{i}: {demand_no_charging:.0f}/{demand_charging:.0f}",
                    fontsize=9, ha='center', color='blue', zorder=5)

            ax.scatter(d_lon, d_lat, c='red', s=80, edgecolors='black', label='Destination' if i == 0 else "", zorder=4)
            ax.text(d_lon, d_lat - 0.0002, f"D{i}: {demand_no_charging:.0f}/{demand_charging:.0f}",
                    fontsize=9, ha='center', color='red', zorder=5)

    # Plot charger nodes as special markers
    for charger_id in grid.chargers:
        charger_lon = grid.net.nodes.at[charger_id, "lon"]
        charger_lat = grid.net.nodes.at[charger_id, "lat"]
        ax.scatter(charger_lon, charger_lat, c='green', s=80, edgecolors='black', marker='*', zorder=5)
        # Add label for charger
        ax.text(charger_lon, charger_lat + 0.0002, f"C{charger_id}", 
                fontsize=9, ha='center', color='green', zorder=5, fontweight='bold')

    # Final plot elements
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

    # Add colorbar
    sm.set_array(gdf["flow"])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Link Flow", fontsize=10)
    
    # Add a note about the threshold
    ax.text(0.01, 0.01, f"Links with flow < {flow_threshold} shown as dashed lines",
            transform=ax.transAxes, fontsize=8, ha='left', va='bottom')

def save_all_flow_heatmaps(grids, config, results_folder, time_history=None):
    """Generate and save flow heatmaps for all configurations in the results folder"""
    # Find the best configuration
    best_idx = np.argmin([grid.travel_time_obj for grid in grids])
    
    # Generate file with summary information
    summary_path = os.path.join(results_folder, "heatmap_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"EV Charger Optimization Results\n")
        f.write(f"===============================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Coordinates: {config['coordinates']}\n")
        f.write(f"Number of chargers: {config['num_chargers']}\n")
        f.write(f"Possible positions: {config['possible_charger_positions']}\n\n")
        
        f.write(f"Results Summary:\n")
        f.write(f"---------------\n")
        f.write(f"Total configurations evaluated: {len(grids)}\n")
        f.write(f"Best configuration: {grids[best_idx].chargers}\n")
        f.write(f"Best travel time objective: {grids[best_idx].travel_time_obj:.4f}\n\n")
        
        # Add network statistics
        f.write(f"Network Statistics:\n")
        f.write(f"------------------\n")
        f.write(f"Number of nodes (n): {grids[0].n}\n")
        f.write(f"Number of links (l): {grids[0].l}\n")
        f.write(f"Dimensions (d): {grids[0].d}\n")
        
        # Calculate route statistics
        route_counts = [grid.r for grid in grids]
        f.write(f"Number of routes: {np.mean(route_counts):.2f} ± {np.sqrt(np.var(route_counts)):.2f}\n\n")
        
        # Add computation statistics if time_history is available
        if time_history is not None:
            f.write(f"Computation Statistics:\n")
            f.write(f"----------------------\n")
            
            # Try to determine phase boundaries for greedy and full optimization
            possible_charger_positions = len(config['possible_charger_positions'])
            num_chargers = config['num_chargers']
            vertical_line_pos = calculate_phase_boundaries(possible_charger_positions, num_chargers)
            
            # Greedy results
            if 'Greedy' in vertical_line_pos:
                greedy_boundary_idx = vertical_line_pos['Greedy']
                if greedy_boundary_idx >= len(grids):
                    greedy_boundary_idx = len(grids)-1
                
                greedy_result_idx = np.argmin([grid.travel_time_obj for grid in grids[:greedy_boundary_idx+1]])
                
                if greedy_boundary_idx < len(time_history):
                    greedy_time = time_history[greedy_boundary_idx]
                else:
                    greedy_time = time_history[-1] if time_history else 0
                    
                f.write(f"Greedy algorithm result: {grids[greedy_result_idx].travel_time_obj:.4f}\n")
                f.write(f"Greedy algorithm chargers: {grids[greedy_result_idx].chargers}\n")
                f.write(f"Greedy algorithm computation time: {greedy_time:.2f} seconds\n\n")
            
            # Full optimization results
            full_idx_key = list(vertical_line_pos.keys())[-3] if len(vertical_line_pos) >= 3 else None
            if full_idx_key and vertical_line_pos[full_idx_key] < len(grids):
                full_chargers_idx = vertical_line_pos[full_idx_key]
                full_charger_results = [grid.travel_time_obj for grid in grids[full_chargers_idx:]]
                full_best_idx = full_chargers_idx + np.argmin(full_charger_results)
                
                if full_chargers_idx < len(time_history):
                    full_time = time_history[-1] - time_history[full_chargers_idx]
                else:
                    full_time = 0
                
                f.write(f"Full optimization minimum result: {min(full_charger_results):.4f}\n")
                f.write(f"Full optimization mean result: {np.mean(full_charger_results):.4f} ± {np.sqrt(np.var(full_charger_results)):.4f}\n")
                f.write(f"Full optimization best chargers: {grids[full_best_idx].chargers}\n")
                f.write(f"Full optimization computation time: {full_time:.2f} seconds\n\n")
            
            # Total computation time
            if time_history:
                f.write(f"Total computation time: {time_history[-1]:.2f} seconds\n\n")
        
        # Add Braess paradoxes
        paradoxes = find_braess_paradoxes(grids)
        f.write(f"Braess Paradoxes:\n")
        f.write(f"----------------\n")
        f.write(f"Number of paradoxes found: {len(paradoxes)}\n")
        for i, p in enumerate(paradoxes):
            f.write(f"{i+1}. Subset: {sorted(p[0])}, Superset: {sorted(p[1])}\n")
            f.write(f"   Subset Value: {p[2]:.4f}, Superset Value: {p[3]:.4f}\n")
            f.write(f"   Efficiency Loss: {(p[3] - p[2])/p[2]*100:.2f}%\n")
        
        if not paradoxes:
            f.write("No Braess paradoxes found in this configuration.\n\n")
        
        f.write(f"\nAll Configurations:\n")
        f.write(f"-----------------\n")
        
        # Sort grids by travel time objective
        sorted_indices = np.argsort([grid.travel_time_obj for grid in grids])
        for i, idx in enumerate(sorted_indices):
            f.write(f"{i+1}. Chargers: {grids[idx].chargers}, Travel Time: {grids[idx].travel_time_obj:.4f}\n")
    
    print(f"Generating flow heatmaps for {len(grids)} configurations...")
    
    all_results_data = {} # Initialize dictionary to store all results
    all_results_data['configurations'] = {} # Sub-dictionary for individual configurations

    # Get link connectivity information once from the first grid
    # Assuming the network structure is the same for all grids in this run.
    if grids: 
        first_grid = grids[0]
        if hasattr(first_grid, 'net') and hasattr(first_grid.net, 'edges') and not first_grid.net.edges.empty:
            sorted_edges = first_grid.net.edges.sort_values("link_id").copy()
            link_connectivity_data = []
            for _, row in sorted_edges.iterrows():
                link_connectivity_data.append({
                    'link_id': int(row['link_id']),
                    'start_node_id': int(row['start_node_id']),
                    'end_node_id': int(row['end_node_id'])
                })
            all_results_data['network_link_connectivity'] = link_connectivity_data
        else:
            all_results_data['network_link_connectivity'] = []
            print("Warning: Could not retrieve edge data from the first grid. Link connectivity will be empty.")
    else:
        all_results_data['network_link_connectivity'] = []
        print("Warning: No grids provided. Link connectivity will be empty.")

    # Add the run configuration to the results dictionary
    all_results_data['run_configuration'] = config

    # Generate heatmaps for all configurations
    for i, grid in enumerate(grids):
        is_best = (i == best_idx)
        marker = "BEST_" if is_best else ""
        config_name = f"{marker}config_{i+1}_chargers_{grid.chargers}_tt_{grid.travel_time_obj:.4f}"
        
        # Create configuration-specific directory
        config_dir = os.path.join(results_folder, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        # Save the standard flow heatmap in the config directory
        filepath = os.path.join(config_dir, "flow_heatmap.png")
        title = f"Charger Placement: {grid.chargers}, Travel Time: {grid.travel_time_obj:.4f}"
        if is_best:
            title = f"BEST {title}"
        
        success = save_flow_heatmap(
            grid=grid,
            output_path=filepath,
            title=title,
            use_cvxpy=config['use_cvxpy']
        )
        
        # Get the optimization results data for the current grid
        current_grid_data = save_optimization_pickle(grid)
        charger_key = frozenset(sorted(grid.chargers)) if grid.chargers is not None else frozenset()
        all_results_data['configurations'][charger_key] = current_grid_data

        # For CVXPY optimized grids, also save the charger contribution plots in the config directory
        if config['use_cvxpy'] and hasattr(grid, 'get_charger_contributions'):
            charger_success = save_charger_flow_heatmaps(
                grid=grid,
                output_folder=config_dir,
                base_filename="charger_flows",
                use_cvxpy=True,
                flow_threshold=1.0
            )
            if charger_success:
                print(f"Saved charger flow heatmaps for configuration {i+1}/{len(grids)}")
        
        if success:
            print(f"Saved configuration {i+1}/{len(grids)}: {filepath}")
        else:
            print(f"Failed to save configuration {i+1}/{len(grids)}")
    
    # Save the aggregated results to a single pickle file
    global_pickle_path = os.path.join(results_folder, "all_optimization_results.pkl")
    try:
        with open(global_pickle_path, 'wb') as f:
            pickle.dump(all_results_data, f)
        print(f"All optimization results saved to {global_pickle_path}")
    except Exception as e:
        print(f"Failed to save global optimization results to {global_pickle_path}: {e}")

    print(f"All flow heatmaps saved to {results_folder}")
    return True

def outer_optimization(coordinates, num_chargers=None, possible_charger_positions=None, 
                       calculate_on_all_possible_positions=False, plot_info=False, use_derivatives=True, 
                       max_iter=1000, parameter_fit_results=None, single_swap=False, use_cvxpy=False, od_demand=None,
                       config_filepath=None):
    """
    Outer optimization to find the best charger locations.
    Includes configurable route analysis with top-k routes and parameter sweep.
    """
    time_history = []
    iteration_count = 0
    grids = []  # array of grids with different chargers
    configurations = {}  # Dictionary to store results for each configuration

    chargers_set = set()
    best_charger = None
    best_grid = None
    best_travel_time = float('inf')
    possible_charger_positions_j = possible_charger_positions.copy()
    t0 = time.time()

    # Step 1: Perform grid search to find the initial best placement
    for num_chargers_j in range(1, num_chargers + 1):
        best_charger_position_j = None
        best_charger_position_travel_time = 1e100
        for charger_ji in possible_charger_positions_j:
            if best_charger is None:
                chargers_i = (charger_ji,)
            else:
                chargers_i = (best_charger) + (charger_ji,)
                chargers_i = tuple(sorted(chargers_i))
            chargers_set.add(chargers_i)
            grid_i = Network(coordinates, chargers=chargers_i, parameter_fit_results=parameter_fit_results, od_demand=od_demand)
            iteration_count += 1
            print("*" * 80)
            print('Iteration: ', iteration_count)
            grid_i.optimize(use_cvxpy=use_cvxpy, use_derivatives=use_derivatives, max_iter=max_iter)
            time_history.append(time.time() - t0)
            if plot_info:
                grid_i.plot_info()
            grids.append(grid_i)

            if grid_i.travel_time_obj < best_charger_position_travel_time:
                best_charger_position_travel_time = grid_i.travel_time_obj
                best_charger_tmp = chargers_i
                best_charger_position_j = charger_ji
                best_grid = grid_i
        best_charger = best_charger_tmp
        possible_charger_positions_j.remove(best_charger_position_j)

    # Step 2: Single Swap Optimization (if enabled)
    if single_swap:
        best_chargers = best_charger
        all_possible_swaps = []
        for idx1, charger1 in enumerate(best_chargers):
            for charger2 in possible_charger_positions:
                if charger2 not in best_chargers:
                    swapped_chargers = list(best_chargers)
                    swapped_chargers[idx1] = charger2
                    swapped_chargers = tuple(sorted(swapped_chargers))
                    if swapped_chargers not in chargers_set:
                        all_possible_swaps.append(swapped_chargers)
                        chargers_set.add(swapped_chargers)

        # Evaluate each swap
        for swapped_chargers in all_possible_swaps:
            grid_i = Network(coordinates, chargers=swapped_chargers, parameter_fit_results=parameter_fit_results, od_demand=od_demand)
            grid_i.optimize(use_cvxpy=use_cvxpy, use_derivatives=use_derivatives, max_iter=max_iter)
            time_history.append(time.time() - t0)
            if plot_info:
                grid_i.plot_info()
            grids.append(grid_i)

            if grid_i.travel_time_obj < best_travel_time:
                best_travel_time = grid_i.travel_time_obj
                best_charger = swapped_chargers
                best_grid = grid_i

    # Step 3: Full Search on All Possible Charger Combinations (if enabled)
    if calculate_on_all_possible_positions:
        for chargers_i in list(combinations(possible_charger_positions, num_chargers)):
            chargers_i = tuple(sorted(chargers_i))
            if chargers_i in chargers_set:
                continue
            chargers_set.add(chargers_i)
            grid_i = Network(coordinates, chargers=chargers_i, parameter_fit_results=parameter_fit_results, od_demand=od_demand)
            iteration_count += 1
            print('Iteration: ', iteration_count)
            grid_i.optimize(use_cvxpy=use_cvxpy, use_derivatives=use_derivatives, max_iter=max_iter)
            time_history.append(time.time() - t0)
            if plot_info:
                grid_i.plot_info()
            grids.append(grid_i)

            if grid_i.travel_time_obj < best_travel_time:
                best_travel_time = grid_i.travel_time_obj
                best_charger = chargers_i
                best_grid = grid_i

    # Create the results directory with timestamp
    filename = 'n=' + str(grid_i.n) + ' d=' + str(grids[-1].d) + ' possible_charger_positions=' + str(len(possible_charger_positions)) + ' num_chargers=' + str(num_chargers)
    today = datetime.now()
    foldername = "results/" + today.strftime('%Y-%m-%d_%H-%M-%S_') + filename
    filename = foldername + '/' + filename
    os.makedirs(foldername, exist_ok=True)

    # Calculate phase boundaries for plotting
    possible_charger_positions_count = len(possible_charger_positions)
    phases = calculate_phase_boundaries(possible_charger_positions_count, num_chargers)
    
    # Generate travel time objectives plot
    plot_travel_time_objectives(grids, time_history, phases, filename, single_swap)
    
    # Copy the config file if provided
    if config_filepath and os.path.exists(config_filepath):
        try:
            shutil.copy(config_filepath, os.path.join(foldername, "run_config.json"))
            print(f"Copied config file to {os.path.join(foldername, 'run_config.json')}")
        except Exception as e:
            print(f"Error copying config file: {e}")
    else:
        if config_filepath:
            print(f"Warning: Config file path provided ({config_filepath}) but file not found. Not copied.")

    # Create config dictionary for saving and visualization
    run_config_params = {
        'coordinates': coordinates,
        'num_chargers': num_chargers,
        'possible_charger_positions': possible_charger_positions,
        'calculate_on_all_possible_positions': calculate_on_all_possible_positions,
        'use_derivatives': use_derivatives,
        'single_swap': single_swap,
        'use_cvxpy': use_cvxpy,
        'max_iter': max_iter,
        'plot_info': plot_info,
        'od_demand': od_demand
    }

    # Generate flow heatmaps and save all results
    save_all_flow_heatmaps(grids, run_config_params, foldername, time_history)
    
    # Print the best result
    best_grid = grids[np.argmin([grid.travel_time_obj for grid in grids])]
    print("\nResults:")
    print(f"Best charger configuration: {best_grid.chargers}")
    print(f"Best travel time objective: {best_grid.travel_time_obj:.4f}")
    print(f"All results saved to: {foldername}")

    # Load route analysis configuration
    route_analysis_config = {}
    if config_filepath and os.path.exists(config_filepath):
        with open(config_filepath, 'r') as f:
            config_data = json.load(f)
            route_analysis_config = config_data.get('route_analysis', {})

    # Store results for each configuration
    if route_analysis_config.get('analyze_top_k_routes', True):
        print("\nAnalyzing route reconstruction for each configuration...")
        configurations = {}
        for i, grid in enumerate(grids):
            # Get optimization results including properly structured link flows
            grid_results = save_optimization_pickle(grid)
            chargers_tuple = tuple(sorted(grid.chargers))
            
            # Determine if this is the best configuration
            is_best = (grid.travel_time_obj == best_grid.travel_time_obj)
            marker = "BEST_" if is_best else ""
            config_name = f"{marker}config_{i+1}_chargers_{grid.chargers}_tt_{grid.travel_time_obj:.4f}"
            
            # Use the existing configuration directory
            config_dir = os.path.join(foldername, config_name)
            recon_dir = os.path.join(config_dir, "reconstruction")
            os.makedirs(recon_dir, exist_ok=True)
            
            reconstruction_results = None
            param_sweep_results = None
            
            # Run top-k route analysis if enabled
            k_values = route_analysis_config.get('k_values', [1, 2, 4, 8, 16, 32, 64])
            print(f"\nAnalyzing top-{max(k_values)} routes for configuration {i+1}/{len(grids)}...")
            reconstruction_results = analyze_route_reconstruction(
                network=grid,
                link_flows_dict=grid_results['link_flows'],
                k_values=k_values,
                save_dir=recon_dir
            )
                
            # Run parameter sweep analysis if both flags are enabled
            if route_analysis_config.get('run_parameter_sweep', True):
                print(f"\nRunning parameter sweep analysis for configuration {i+1}/{len(grids)}...")
                param_sweep_dir = os.path.join(recon_dir, "parameter_sweep")
                os.makedirs(param_sweep_dir, exist_ok=True)
                
                # Get parameter sweep configuration
                param_sweep_config = route_analysis_config.get('parameter_sweep', {})
                paths_per_od_values = param_sweep_config.get('paths_per_od_values', [5, 10, 15, 20, 25, 30])
                paths_per_oc_cd_values = param_sweep_config.get('paths_per_oc_cd_values', [3, 6, 9, 12, 15])
                
                param_sweep_results = analyze_path_parameters(
                    network=grid,
                    link_flows_dict=grid_results['link_flows'],
                    paths_per_od_values=paths_per_od_values,
                    paths_per_oc_cd_values=paths_per_oc_cd_values,
                    save_dir=param_sweep_dir
                )
            
            # Store all results
            config_results = {
                'charger_combination': chargers_tuple,
                'objective_value': float(grid.travel_time_obj),
                'link_flows': grid_results['link_flows'],
                'method': grid_results.get('method', 'unknown')
            }
            
            # Only include analysis results if they were generated
            if reconstruction_results is not None:
                config_results['reconstruction_results'] = reconstruction_results
            if param_sweep_results is not None:
                config_results['param_sweep_results'] = param_sweep_results
                
            configurations[chargers_tuple] = config_results

        # Add route analysis config to run configuration
        run_config_params['route_analysis'] = route_analysis_config

        # Save all results to pickle file
        results_dict = {
            'run_configuration': run_config_params,
            'configurations': configurations,
            'time_history': time_history,
            'iteration_count': iteration_count
        }
        
        with open(os.path.join(foldername, 'all_optimization_results.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)

    return grids, time_history


def calculate_phase_boundaries(n, k):
    idx = -1
    phases = {}
    for i in range(1, k+2):
        idx += n+1-i
        if i < k:
            phases[str(i) + ' Chargers'] = idx
        if i == k:
            phases['Greedy'] = idx
        if i == k+1:
            phases['Greedy + Single Swap'] = phases['Greedy'] + (n-k)*(k-1)

    phases['Exhaustive'] = phases[list(phases.keys())[-3]] + comb(n, k)
    print("Phase boundaries are: ", phases)
    return phases


def plot_travel_time_objectives(grids, time_history, phases, filename='1', single_swap=False):
    """Plot travel time objectives and optimization progress as a static image"""
    travel_times = [grid.travel_time_obj for grid in grids]

    # Find indices for different phases
    greedy_idx = np.argmin(travel_times[0:phases['Greedy']+1])
    greedy_result = grids[greedy_idx].travel_time_obj

    if single_swap:
        single_swap_idx = np.argmin(travel_times[0:phases['Greedy + Single Swap']+1])
        single_swap_result = grids[single_swap_idx].travel_time_obj

    exhaustive_idx = np.argmin(travel_times)
    exhaustive_result = grids[exhaustive_idx].travel_time_obj

    charger_names = [str(grid.chargers) for grid in grids]
    colors = ['tab:purple', 'tab:green', 'tab:red', 'tab:gray', 'tab:blue']

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot computation time history
    ax = axes[0]
    ax.plot(range(1, len(time_history)+1), time_history, marker='o')
    ax.set_xlabel('Charger Combinations')
    ax.set_ylabel('Cumulative Computation Time (s)')
    
    # Add vertical lines for phase boundaries
    for i, (label, vertical_line_pos_i) in enumerate(phases.items()):
        ax.axvline(x=vertical_line_pos_i+1, label=label, linestyle='--', c=colors[i])
    
    ax.legend(loc='upper left')
    ax.set_xticks(range(1, len(charger_names)+1))
    ax.set_xticklabels(charger_names, rotation=45, ha='right')
    ax.grid(True)

    # Plot travel time objectives
    ax = axes[1]
    for j, grid_j in enumerate(grids):
        ax.scatter(j + 1, grid_j.travel_time_obj, c='tab:blue', s=50)
        
        if j == greedy_idx:
            ax.scatter(j + 1, greedy_result, marker='D', s=100, c='tab:gray', label='Greedy Min')
        if single_swap and j == single_swap_idx:
            ax.scatter(j + 1, single_swap_result, marker='s', s=100, c='tab:green', label='Greedy + Single Swap Min')
        if j == exhaustive_idx:
            ax.scatter(j + 1, exhaustive_result, marker='*', s=200, c='tab:red', label='Exhaustive Min')

    # Add vertical lines for phase boundaries
    for i, (label, vertical_line_pos_i) in enumerate(phases.items()):
        ax.axvline(x=vertical_line_pos_i+1, label=label, linestyle='--', c=colors[i])

    ax.set_xlabel('Charger Combinations')
    ax.set_ylabel('Travel Time Objective')
    ax.legend(loc='upper right')
    ax.set_xticks(range(1, len(charger_names)+1))
    ax.set_xticklabels(charger_names, rotation=45, ha='right')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def find_braess_paradoxes(grids):
    """Find instances of Braess's paradoxes in the given set of chargers"""
    chargers_values = {frozenset(grids[i].chargers):grids[i].travel_time_obj for i in range(len(grids))}
    paradoxes = []
    for subset in chargers_values.keys():
        for superset in chargers_values.keys():
            # Check if subset is a proper subset of superset and value is less
            if subset.issubset(superset) and subset != superset and chargers_values[subset] * 1.001 < chargers_values[superset]:
                paradoxes.append((subset, superset, chargers_values[subset], chargers_values[superset]))

    print("*" * 80)
    print(" Number of Braess Paradoxes: ", len(paradoxes))
    for p in paradoxes:
        print(f"Subset: {p[0]}, Superset: {p[1]}, Subset Value: {p[2]}, Superset Value: {p[3]}")
    return paradoxes


def print_stats(grids, time_history, phases):
    """Print statistics about optimization results"""
    # vertical_line_pos and related calculations removed, using phases dict directly.
    # num_charger = len(grids[-1].chargers) # Can get from phases or config if needed

    # Extract necessary indices from the phases dictionary
    # Ensure keys exist before accessing, or provide defaults.
    greedy_idx = phases.get('Greedy', len(grids) -1) # Default to end if key missing
    
    # For full_chargers_idx, we need the start of the exhaustive phase.
    # This logic assumes 'Exhaustive' marks the end of the exhaustive set of runs, 
    # and a prior key like 'Greedy + Single Swap' or 'Greedy' marks the start of more runs.
    # A more robust way would be to have calculate_phase_boundaries return start/end for each phase.
    # For now, let's find a key that likely marks the start of the exhaustive part.
    # The 'Exhaustive' value in phases is the *end* index of that section.
    # The start of 'Exhaustive' is often after 'Greedy + Single Swap' or 'Greedy'.
    
    # Find the starting index for the 'full' or 'exhaustive' part.
    # This is a heuristic: the index after 'Greedy + Single Swap' if it exists, else after 'Greedy'.
    if 'Greedy + Single Swap' in phases:
        full_chargers_start_idx = phases['Greedy + Single Swap'] + 1
    elif 'Greedy' in phases:
        full_chargers_start_idx = phases['Greedy'] + 1
    else:
        full_chargers_start_idx = 0 # Default to start if no other phase found
    
    # Ensure full_chargers_start_idx is within bounds
    full_chargers_start_idx = min(full_chargers_start_idx, len(grids) -1)
    if full_chargers_start_idx < 0: full_chargers_start_idx = 0

    od_pair_route_counts = [] # Renamed from od_pairss to avoid confusion
    # Collect route counts from the 'full' optimization part if it's meaningful
    # Or perhaps all grids if the distinction isn't critical here.
    # For now, let's use all grids for route stats as before.
    for grid in grids:
        if hasattr(grid, 'r') and grid.r is not None:
             od_pair_route_counts.append(grid.r)
        
    # print(phases) # Debug: check phases dictionary
    if grids: # Check if grids list is not empty
        print('l: ', grids[-1].l, 'd: ', grids[-1].d)
    else:
        print('l: N/A, d: N/A (no grids found)')

    if od_pair_route_counts:
        print("Number of routes: ", np.mean(od_pair_route_counts), '+-', np.sqrt(np.var(od_pair_route_counts)))
    else:
        print("Number of routes: N/A (no route data found)")

    # Greedy results: use up to greedy_idx (inclusive)
    if grids and greedy_idx < len(grids) and greedy_idx >=0:
        greedy_results_slice = [grid.travel_time_obj for grid in grids[:greedy_idx+1]]
        if greedy_results_slice:
            best_greedy_in_slice_idx = np.argmin(greedy_results_slice)
            print('Greedy result: ', grids[best_greedy_in_slice_idx].travel_time_obj, 
                  ' using chargers: ', grids[best_greedy_in_slice_idx].chargers, 
                  'in', time_history[greedy_idx] if greedy_idx < len(time_history) else 'N/A', 'seconds')
        else:
            print("Greedy result: No data in slice")
    else:
        print("Greedy result: N/A (index out of bounds or no grids)")

    # Full optimization results: use from full_chargers_start_idx to end
    if grids and full_chargers_start_idx < len(grids):
        full_charger_results_slice = np.array([grid.travel_time_obj for grid in grids[full_chargers_start_idx:]])
        if full_charger_results_slice.size > 0:
            best_full_in_slice_idx = np.argmin(full_charger_results_slice)
            actual_best_idx_in_grids = full_chargers_start_idx + best_full_in_slice_idx
            
            computation_time_for_full = 'N/A'
            if full_chargers_start_idx < len(time_history) and time_history:
                computation_time_for_full = time_history[-1] - time_history[full_chargers_start_idx]
            elif not time_history:
                computation_time_for_full = 'N/A (no time history)'
            else: # full_chargers_start_idx might be too large for time_history
                computation_time_for_full = time_history[-1] # or some other sensible default

            print('Full optimization result: ', full_charger_results_slice.min(), ',', 
                  full_charger_results_slice.mean(), '+-', np.sqrt(full_charger_results_slice.var()), 
                  ' using chargers: ', grids[actual_best_idx_in_grids].chargers, 
                  'in', computation_time_for_full, 'seconds')
        else:
            print("Full optimization result: No data in slice")
    else:
        print("Full optimization result: N/A (index out of bounds or no grids)")

    # Braess paradoxes are found using all grids, so this remains unchanged.
    paradoxes = find_braess_paradoxes(grids)

def analyze_route_reconstruction(network, link_flows_dict, k_values=[1, 2, 4, 8, 16, 32, 64], save_dir=None):
    """
    Analyze route reconstruction for a given network configuration.
    Returns reconstruction metrics and route information for different k values.
    
    Args:
        network: Network object with the current configuration
        link_flows_dict: Dictionary of link flows from optimization
        k_values: List of k values to analyze
        save_dir: Directory to save visualizations (optional)
    
    Returns:
        dict: Dictionary containing reconstruction metrics and routes
    """
    print("\nAnalyzing route reconstruction...")
    
    # Get original flows from link_flows_dict
    original_flows = np.zeros(len(link_flows_dict))
    for link_id, link_data in link_flows_dict.items():
        original_flows[link_id] = link_data['total_flow']
    
    # Store metrics for each k value
    k_metrics = {}
    
    # Get the maximum k value for full route analysis
    max_k = max(k_values)
    
    # Reconstruct flows with maximum k to get all route information
    reconstruction_result = network.reconstruct_route_flows(
        link_flows_dict,
        paths_per_od=max_k,
        paths_per_oc_cd=max_k,
        use_od_constraints=True,
        use_charger_constraints=True
    )
    
    if reconstruction_result is not None:
        # Convert reconstruction result to flat list of routes with flows
        routes_with_flows = []
        route_id = 0
        
        for od_pair, flows in reconstruction_result.items():
            # Add non-charging routes
            for route_data in flows['non_charging']:
                if route_data['flow'] > 1e-6:  # Only include routes with non-zero flow
                    route_info = {
                        'route_id': route_id,
                        'flow': float(route_data['flow']),
                        'links': route_data['path'],
                        'type': 'non_charging',
                        'origin': od_pair[0],
                        'destination': od_pair[1]
                    }
                    routes_with_flows.append(route_info)
                    route_id += 1
            
            # Add charging routes
            for charger, charger_routes in flows['charging'].items():
                for route_data in charger_routes:
                    if route_data['flow'] > 1e-6:  # Only include routes with non-zero flow
                        route_info = {
                            'route_id': route_id,
                            'flow': float(route_data['flow']),
                            'links': route_data['path'],
                            'type': 'charging',
                            'origin': od_pair[0],
                            'destination': od_pair[1],
                            'charger': charger
                        }
                        routes_with_flows.append(route_info)
                        route_id += 1
        
        if routes_with_flows:  # Only proceed if we have valid routes
            # Sort routes by flow in descending order
            sorted_routes = sorted(routes_with_flows, key=lambda x: x['flow'], reverse=True)
            total_flow = sum(route['flow'] for route in sorted_routes)
            
            if total_flow > 0:  # Only update metrics if we have valid flow
                # Calculate and store top-k metrics
                for k in k_values:
                    k = min(k, len(sorted_routes))  # Don't exceed available routes
                    top_k_routes = sorted_routes[:k]
                    k_flow = sum(route['flow'] for route in top_k_routes)
                    
                    # Calculate reconstructed flows for this k
                    flows = np.zeros(len(link_flows_dict))
                    for route in top_k_routes:
                        path = route['links']
                        flow = route['flow']
                        for i in range(len(path) - 1):
                            start_node = path[i]
                            end_node = path[i + 1]
                            for link_id, link_data in link_flows_dict.items():
                                if link_data['start_node_id'] == start_node and link_data['end_node_id'] == end_node:
                                    flows[link_id] += flow
                                    break
                    
                    # Calculate error metrics
                    flow_difference = flows - original_flows
                    k_metrics[k] = {
                        'coverage': float(k_flow / total_flow * 100),
                        'mae': float(np.mean(np.abs(flow_difference))),
                        'rmse': float(np.sqrt(np.mean(np.square(flow_difference)))),
                        'max_diff': float(np.max(np.abs(flow_difference))),
                        'correlation': float(np.corrcoef(original_flows, flows)[0, 1]),
                        'routes': [
                            {
                                'route_id': route['route_id'],
                                'flow': float(route['flow']),
                                'links': route['links'],
                                'type': route['type'],
                                'origin': route['origin'],
                                'destination': route['destination'],
                                'charger': route.get('charger')  # Only included for charging routes
                            }
                            for route in top_k_routes
                        ]
                    }
    
    # Create visualizations if save_dir is provided and we have results
    if save_dir and k_metrics:
        print("\nGenerating visualizations...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create metrics plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        metrics = ['coverage', 'mae', 'rmse', 'max_diff', 'correlation']
        titles = ['Flow Coverage (%)', 'Mean Absolute Error', 'Root Mean Square Error', 
                 'Maximum Absolute Difference', 'Correlation']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            y = [k_metrics[k][metric] for k in k_values if k in k_metrics]
            k_vals = [k for k in k_values if k in k_metrics]
            if k_vals:  # Only plot if we have data
                ax.plot(k_vals, y, 'o-', linewidth=2)
                ax.set_xlabel('Number of Routes (k)')
                ax.set_ylabel(title)
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'k_routes_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Create flow visualizations in a single figure
        n_plots = len(k_metrics) + 1  # +1 for original flows
        n_cols = 4  # We want 4 columns
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(20, 5*n_rows))
        
        # Plot original flows
        ax = plt.subplot(n_rows, n_cols, 1)
        network.cvxpy_link_flows = original_flows
        _plot_flow_on_axis(network, ax, 'Original Flows', original_flows)
        
        # Plot flows for each k value
        for idx, (k, metrics) in enumerate(sorted(k_metrics.items()), start=2):
            ax = plt.subplot(n_rows, n_cols, idx)
            # Calculate reconstructed flows
            flows = np.zeros(len(link_flows_dict))
            for route in metrics['routes']:
                path = route['links']
                flow = route['flow']
                for i in range(len(path) - 1):
                    start_node = path[i]
                    end_node = path[i + 1]
                    for link_id, link_data in link_flows_dict.items():
                        if link_data['start_node_id'] == start_node and link_data['end_node_id'] == end_node:
                            flows[link_id] += flow
                            break
            network.cvxpy_link_flows = flows
            _plot_flow_on_axis(network, ax, 
                             f'k={k}\nCoverage={metrics["coverage"]:.1f}%\nMAE={metrics["mae"]:.3f}', 
                             flows)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'flow_reconstructions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return {'k_metrics': k_metrics}

def analyze_path_parameters(network, link_flows_dict, 
                          paths_per_od_values=[5, 10, 15, 20, 25, 30],
                          paths_per_oc_cd_values=[3, 6, 9, 12, 15],
                          save_dir=None):
    """
    Analyze the impact of different path parameter combinations on reconstruction accuracy.
    
    Args:
        network: Network object with the current configuration
        link_flows_dict: Dictionary of link flows from optimization
        paths_per_od_values: List of values to test for paths_per_od
        paths_per_oc_cd_values: List of values to test for paths_per_oc_cd
        save_dir: Directory to save visualizations (optional)
    """
    print("\nAnalyzing path parameter combinations...")
    
    # Store results for each combination
    results = {}
    original_flows = np.array([data['total_flow'] for data in link_flows_dict.values()])
    
    for paths_per_oc_cd in paths_per_oc_cd_values:
        results[paths_per_oc_cd] = {
            'mae': [],
            'rmse': [],
            'max_diff': [],
            'correlation': []
        }
        
        for paths_per_od in paths_per_od_values:
            print(f"\nTesting paths_per_od={paths_per_od}, paths_per_oc_cd={paths_per_oc_cd}")
            
            # Get reconstruction result for this parameter combination
            reconstruction_result = network.reconstruct_route_flows(
                link_flows_dict,
                paths_per_od=paths_per_od,
                paths_per_oc_cd=paths_per_oc_cd,
                use_od_constraints=True,
                use_charger_constraints=True
            )
            
            # Convert to format expected by analysis
            all_routes = []
            for od_pair, flows in reconstruction_result.items():
                # Add non-charging routes
                for route_data in flows['non_charging']:
                    all_routes.append((route_data['path'], route_data['flow']))
                
                # Add charging routes
                for charger_routes in flows['charging'].values():
                    for route_data in charger_routes:
                        all_routes.append((route_data['path'], route_data['flow']))
            
            # Calculate reconstructed flows
            flows = np.zeros(len(link_flows_dict))
            for route_data in all_routes:
                route = route_data[0]  # path
                flow = route_data[1]   # flow
                for i in range(len(route) - 1):
                    start_node = route[i]
                    end_node = route[i + 1]
                    for link_id, link_data in link_flows_dict.items():
                        if link_data['start_node_id'] == start_node and link_data['end_node_id'] == end_node:
                            flows[link_id] += flow
                            break
            
            # Calculate metrics
            flow_difference = flows - original_flows
            results[paths_per_oc_cd]['mae'].append(float(np.mean(np.abs(flow_difference))))
            results[paths_per_oc_cd]['rmse'].append(float(np.sqrt(np.mean(np.square(flow_difference)))))
            results[paths_per_oc_cd]['max_diff'].append(float(np.max(np.abs(flow_difference))))
            results[paths_per_oc_cd]['correlation'].append(float(np.corrcoef(original_flows, flows)[0, 1]))
    
    if save_dir:
        print("\nGenerating visualizations...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['mae', 'rmse', 'max_diff', 'correlation']
        titles = ['MAE vs Paths per OD', 'RMSE vs Paths per OD', 
                 'MAX_DIFF vs Paths per OD', 'CORRELATION vs Paths per OD']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            for paths_per_oc_cd in paths_per_oc_cd_values:
                ax.plot(paths_per_od_values, results[paths_per_oc_cd][metric], 
                       'o-', label=f'paths_per_oc_cd={paths_per_oc_cd}')
            
            ax.set_xlabel('Paths per OD')
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'path_parameter_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return results