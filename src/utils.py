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
        
        charger_node_to_throughput_map = {}
        if charger_throughputs is not None and grid.chargers is not None and len(grid.chargers) == len(charger_throughputs):
            for i, charger_node_id in enumerate(grid.chargers):
                 charger_node_to_throughput_map[charger_node_id] = charger_throughputs[i]
        elif charger_throughputs is not None:
            # This case might occur if grid.chargers is None or length mismatch.
            # For now, we cannot map throughputs to specific charger nodes if so.
            print(f"Warning: Charger throughputs available but cannot map to charger nodes for {grid.chargers}")

        for _, edge_row in all_edges.iterrows():
            link_id = int(edge_row['link_id'])
            start_node = int(edge_row['start_node_id'])
            end_node = int(edge_row['end_node_id'])
            flow_value = 0.0

            is_charger_self_link = (start_node == end_node and start_node in charger_node_to_throughput_map)

            if is_charger_self_link:
                flow_value = charger_node_to_throughput_map.get(start_node, 0.0)
            elif link_id < len(raw_link_flows): # Regular road link
                flow_value = raw_link_flows[link_id]
            
            link_flows_dict[link_id] = {
                'start_node_id': start_node,
                'end_node_id': end_node,
                'flow': float(flow_value)
            }

    elif hasattr(grid, 'flow') and grid.flow is not None: # SciPy case
        results_data['objective_value'] = grid.travel_time_obj
        results_data['method'] = 'scipy'
        
        # Create a full flow array initialized to zeros
        full_scipy_flows = np.zeros(grid.l)
        if hasattr(grid, 'active_link_indices') and grid.active_link_indices is not None and len(grid.active_link_indices) == len(grid.flow):
            # Populate flows for active links
            full_scipy_flows[grid.active_link_indices] = grid.flow
        elif len(grid.flow) == grid.l:
            # grid.flow is already a full array
            full_scipy_flows = grid.flow
        else:
            # Fallback: This case indicates a potential issue with SciPy flow data structure.
            # Log a warning and save what's available, possibly resulting in missing flows for some links.
            print(f"Warning: SciPy flow array (len {len(grid.flow)}) and active_link_indices mismatch or grid.l (len {grid.l}) issue for {grid.chargers}. Some flows may be zero.")
            # Try to assign flows if grid.flow is shorter but covers initial link_ids
            # This is a best-effort assignment if active_link_indices is not usable.
            if len(grid.flow) < grid.l:
                full_scipy_flows[:len(grid.flow)] = grid.flow
            # If grid.flow is longer, it's an undefined state, so full_scipy_flows remains mostly zeros.

        for _, edge_row in all_edges.iterrows():
            link_id = int(edge_row['link_id'])
            flow_val = 0.0
            if link_id < len(full_scipy_flows):
                flow_val = full_scipy_flows[link_id]
            else:
                # This should ideally not happen if all_edges corresponds to grid.l and full_scipy_flows is sized to grid.l
                print(f"Warning: link_id {link_id} out of bounds for SciPy full_scipy_flows (len {len(full_scipy_flows)}). Flow set to 0.")

            link_flows_dict[link_id] = {
                'start_node_id': int(edge_row['start_node_id']),
                'end_node_id': int(edge_row['end_node_id']),
                'flow': float(flow_val)
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
                'flow': 0.0
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
    for c in range(len(grid.chargers)):
        all_flows.append(contributions[f"charger_{c}"])
    
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
    for c in range(len(grid.chargers)):
        ax_idx += 1
        if ax_idx < len(axes_flat):
            charger_id = grid.chargers[c]
            title = f"Charger {charger_id} Flow"
            _plot_flow_on_axis(grid, axes_flat[ax_idx], title, contributions[f"charger_{c}"], 
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
    """Helper function to plot a flow heatmap on a specific axis"""
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
        filename = f"{marker}config_{i+1}_chargers_{grid.chargers}_tt_{grid.travel_time_obj:.4f}"
        filepath = os.path.join(results_folder, filename + ".png")
        # pickle_filepath = os.path.join(results_folder, filename + ".pkl") # Removed individual pickle path
        
        title = f"Charger Placement: {grid.chargers}, Travel Time: {grid.travel_time_obj:.4f}"
        if is_best:
            title = f"BEST {title}"
        
        # Save the standard flow heatmap
        success = save_flow_heatmap(
            grid=grid,
            output_path=filepath,
            title=title,
            use_cvxpy=config['use_cvxpy']
        )
        
        # Get the optimization results data for the current grid
        current_grid_data = save_optimization_pickle(grid)
        # Use a frozenset of a sorted tuple of chargers as the key for the dictionary
        charger_key = frozenset(sorted(grid.chargers)) if grid.chargers is not None else frozenset()
        all_results_data['configurations'][charger_key] = current_grid_data

        # For CVXPY optimized grids, also save the charger contribution plots
        if config['use_cvxpy'] and hasattr(grid, 'get_charger_contributions'):
            charger_success = save_charger_flow_heatmaps(
                grid=grid,
                output_folder=results_folder,
                base_filename=filename,
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

def find_latest_results_dir():
    """Find the latest results directory matching the expected pattern"""
    # Pattern to match: "YYYY-MM-DD_HH-MM-SS_n=xx d=yy possible_charger_positions=zz num_chargers=ww"
    pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_n=\d+ d=\d+ possible_charger_positions=\d+ num_chargers=\d+"
    
    # Fix path due to src/ directory structure differences
    result_dirs = glob.glob(os.path.join("results", "*"))
    matching_dirs = [d for d in result_dirs if re.search(pattern, d)]
    
    if matching_dirs:
        # Get the latest matching directory
        return max(matching_dirs, key=os.path.getmtime)
    else:
        print("ERROR: Could not find a results directory matching the expected pattern.")
        print("Available directories:")
        for d in result_dirs:
            print(f"  - {d}")
        return None

def outer_optimization(coordinates, num_chargers=None, possible_charger_positions=None, 
                       calculate_on_all_possible_positions=False, plot_info=False, use_derivatives=True, 
                       max_iter=1000, parameter_fit_results=None, single_swap=False, use_cvxpy=False, od_demand=None,
                       config_filepath=None):
    """Run the outer optimization process to find optimal charger placements"""
    # Parameters d, od_pairs, demands removed as od_demand is the primary source.
    time_history = []
    iteration_count = 0
    grids = []  # array of grids with different chargers

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

    # Create the results directory
    filename = 'n=' + str(grid_i.n) + ' d=' + str(grids[-1].d) + ' possible_charger_positions=' + str(len(possible_charger_positions)) + ' num_chargers=' + str(num_chargers)
    today = datetime.now()
    foldername = "results/" + today.strftime('%Y-%m-%d_%H-%M-%S_') + filename
    filename = foldername + '/' + filename
    os.makedirs(foldername, exist_ok=True)
    
    # Copy the config file to the results folder
    if config_filepath and os.path.exists(config_filepath):
        try:
            shutil.copy(config_filepath, os.path.join(foldername, "run_config.json"))
            print(f"Copied config file to {os.path.join(foldername, 'run_config.json')}")
        except Exception as e:
            print(f"Error copying config file: {e}")
    else:
        if config_filepath:
            print(f"Warning: Config file path provided ({config_filepath}) but file not found. Not copied.")
        # else: # No config_filepath provided, so nothing to copy. This is fine if called programmatically without one.
            # print("No config file path provided to outer_optimization. Not copied.")

    # Calculate phase boundaries once
    phases = calculate_phase_boundaries(len(possible_charger_positions), num_chargers)

    # Generate and save plots
    plot_travel_time_objectives(grids, time_history, phases, filename=filename, single_swap=single_swap)
    print_stats(grids, time_history, phases) # Pass phases dict
    
    # Create config dictionary for saving and visualization, from outer_optimization parameters
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
        'od_demand': od_demand # od_demand is already in the correct format here
        # config_filepath is also available if needed: 'config_filepath': config_filepath
    }

    # Generate flow heatmaps and save all results
    save_all_flow_heatmaps(grids, run_config_params, foldername, time_history)
    
    # Print the best result
    best_grid = grids[np.argmin([grid.travel_time_obj for grid in grids])]
    print("\nResults:")
    print(f"Best charger configuration: {best_grid.chargers}")
    print(f"Best travel time objective: {best_grid.travel_time_obj:.4f}")
    print(f"All results saved to: {foldername}")

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
    for i, (label, vertical_line_pos_i) in enumerate(phases.items()):
        ax.axvline(x=vertical_line_pos_i+1, label=label, linestyle='--', c=colors[i])
    ax.legend(loc='upper left')
    ax.set_xticks(range(1, len(charger_names)+1))
    ax.set_xticklabels(charger_names, rotation=45, ha='right')
    ax.grid(True)

    # Plot travel time objectives
    ax = axes[1]
    for j, grid_j in enumerate(grids):
        if j == greedy_idx:
            ax.scatter(j + 1, greedy_result, marker='D', s=100, c='tab:gray', label='Greedy Min')
        if single_swap and j == single_swap_idx:
            ax.scatter(j + 1, single_swap_result, marker='s', s=100, c='tab:green', label='Greedy + Single Swap Min')
        if j == exhaustive_idx:
            ax.scatter(j + 1, exhaustive_result, marker='*', s=200, c='tab:red', label='Exhaustive Min')
        else:
            ax.scatter(j + 1, grid_j.travel_time_obj)
    
    ax.set_xlabel('Charger Combinations')
    ax.set_ylabel('Travel Time Objective')
    ax.grid(True)

    for i, (label, vertical_line_pos_i) in enumerate(phases.items()):
        ax.axvline(x=vertical_line_pos_i+1, label=label, linestyle='--', c=colors[i])
    ax.set_xticks(range(1, len(charger_names)+1))
    ax.set_xticklabels(charger_names, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(loc='upper left')

    # Adjust layout and save as PNG
    plt.tight_layout()
    plt.savefig(filename + '.png', dpi=300)
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