import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
import copy
from itertools import islice
from scipy.optimize import minimize, LinearConstraint
from shapely.geometry import Point
import geopandas as gpd
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

from src.road_network import RoadNet

class Network(RoadNet):
    def __init__(self, coordinates=[38.98211, 38.979, -76.93006, -76.93704], chargers=None, parameter_fit_results=None, od_demand=None):
        # Parameters
        self.charging_to_no_charging_ratio = 0.5
        self.charger_self_link_length = 100  # this is important
        self.paths_per_od = 9
        self.paths_per_oc_and_cd = 4
        self.parameter_fit_results = parameter_fit_results
        self.use_cvxpy = False  # Flag to determine which optimization method to use

        self.net = RoadNet('College Park')
        self.net.get_map(coordinates[0], coordinates[1], coordinates[2], coordinates[3])
        self.DiGraph = nx.DiGraph(self.net.graph)  # DiGraph to find routes
        self.net.rearrange_data()

        self.n = len(self.net.nodes)
        self.l = len(self.net.edges)  # Number of total links

        # Chargers can be a list of node IDs
        if chargers is not None:
            self.chargers = np.array(chargers)
        else:
            # If no chargers provided, create random ones if an integer is passed
            if isinstance(chargers, int):
                self.chargers = self.create_random_chargers(chargers)
            else:
                self.chargers = None

        self.car_types = ["no charging type"] + ["charging type"]
        print("chargers: ", self.chargers)

        if self.chargers is not None:
            self.create_self_links()

        self.d = None  # Number of demand constraints
        self.r = None  # Number of total routes

        self.od_pairs_and_routes = {}
        self.A = None  # Demand matrix
        self.R = None  # Route flow to link flow conversion matrix (Routing matrix)

        self.objective_count = 0
        self.objective_first_derivative_count = 0
        self.objective_second_derivative_count = 0

        self.obj_history = []

        # CVXPY related attributes
        self.cvxpy_link_flows = None
        self.cvxpy_link_flows_data = {}

        # Process OD demand if provided
        if od_demand:
            self.od_demand = od_demand
            self.process_od_demand(od_demand)

    def process_od_demand(self, od_demand):
        """Process OD demand dictionary in the format {(o,d): (demand_type1, demand_type2)}"""
        od_pairs = []
        demands = []

        for (o, d), (demand_type1, demand_type2) in od_demand.items():
            od_pairs.append((o, d))
            demands.extend([demand_type1, demand_type2])

        self.od_pairs = od_pairs
        self.get_od_pairs_and_demands(od_pairs, demands)

    def print_info(self):
        print("Number of total nodes n: ", self.n)
        if self.chargers is not None:
            print("Charger Positions: \n", self.chargers)
        print("Number of total links l: ", self.l)
        print("Number of OD pairs d: ", self.d)
        print("Number of routes r: ", self.r)
        print("OD pairs and corresponding routes: \n", self.od_pairs_and_routes)
        print("Routing matrix R: \n", self.R)
        print("Demand matrix A: \n", self.A)
        print("Demands vector b: \n", self.b)

    def plot_info(self):
        # Plots the route flows before and after optimization
        plt.show()
        plt.plot(self.x0, label='Initial (uniform distributed)')
        plt.plot(self.res.x, label='Optimized')
        plt.xlabel("Route #")
        plt.ylabel("Route flow")
        plt.legend()
        plt.show()

        # Plots the link flows before and after optimization
        plt.figure()
        plt.plot(self.flow, label='Link flows after optimization')
        initial_link_flows = self.R@self.x0
        plt.plot(initial_link_flows, label='Initial link flows')
        plt.xlabel("Link #")
        plt.ylabel("Link flow")
        plt.legend()

        # Plot the objective function progress during the optimization
        plt.figure()
        plt.plot(self.obj_history, label='Objective Value')
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.legend()

    def plot_link_flow_heatmap(self, title="Link Flow Heatmap", use_cvxpy=False, figsize=(10, 10), cmap="plasma",
                            max_linewidth=10, min_linewidth=1, show_od_nodes=True, show_all_nodes=True, 
                            flow_threshold=2):
        """
        Plot a heatmap of link flows using color and thickness
        
        Parameters:
        -----------
        flow_threshold : float
            Flows below this threshold will be shown with high transparency and no arrows
        """
        flows = self.cvxpy_link_flows if use_cvxpy else self.flow
        if flows is None:
            print("No flow data available to plot.")
            return

        edges_df = self.net.edges.sort_values("link_id").copy()
        if len(flows) != len(edges_df):
            print(f"Flow length {len(flows)} does not match number of edges {len(edges_df)}.")
            return
        edges_df["flow"] = flows
        edges_df["geometry"] = gpd.GeoSeries.from_wkt(edges_df["geometry"])
        gdf = gpd.GeoDataFrame(edges_df, geometry="geometry")

        norm = mcolors.Normalize(vmin=gdf["flow"].min(), vmax=gdf["flow"].max())
        linewidths = min_linewidth + (gdf["flow"] - gdf["flow"].min()) / max(1e-6, gdf["flow"].max() - gdf["flow"].min()) * (max_linewidth - min_linewidth)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        gdf["color"] = gdf["flow"].apply(lambda f: sm.to_rgba(f))
        
        # Identify low-flow links for special treatment
        gdf["is_low_flow"] = gdf["flow"] < flow_threshold

        fig, ax = plt.subplots(figsize=figsize)

        # Plot links
        for geom, color, lw, flow, is_low_flow in zip(gdf["geometry"], gdf["color"], linewidths, gdf["flow"], gdf["is_low_flow"]):
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
        if show_all_nodes:
            ax.scatter(self.net.nodes["lon"], self.net.nodes["lat"], color="black", s=10, zorder=3, label="Node")

        # Plot OD nodes with demand annotation
        if show_od_nodes and hasattr(self, 'od_pairs'):
            for i, (o, d) in enumerate(self.od_pairs):
                o_lon = self.net.nodes.at[o, "lon"]
                o_lat = self.net.nodes.at[o, "lat"]
                d_lon = self.net.nodes.at[d, "lon"]
                d_lat = self.net.nodes.at[d, "lat"]

                demand_no_charging = self.b[2 * i]
                demand_charging = self.b[2 * i + 1]

                ax.scatter(o_lon, o_lat, c='blue', s=100, edgecolors='black', label='Origin' if i == 0 else "", zorder=4)
                ax.text(o_lon, o_lat + 0.0002, f"O{i}: {demand_no_charging:.0f}/{demand_charging:.0f}",
                        fontsize=9, ha='center', color='blue', zorder=5)

                ax.scatter(d_lon, d_lat, c='red', s=100, edgecolors='black', label='Destination' if i == 0 else "", zorder=4)
                ax.text(d_lon, d_lat - 0.0002, f"D{i}: {demand_no_charging:.0f}/{demand_charging:.0f}",
                        fontsize=9, ha='center', color='red', zorder=5)

        ax.set_title(title, fontsize=16)
        ax.set_axis_off()

        sm.set_array(gdf["flow"])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label("Link Flow", fontsize=12)
        
        # Add a note about the threshold
        ax.text(0.01, 0.01, f"Links with flow < {flow_threshold} shown with dashed lines",
                transform=ax.transAxes, fontsize=8, ha='left', va='bottom')

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='lower left', fontsize=10)

        plt.show()

    def create_self_links(self):
        # Coordinates of the centers of the circles
        center_coords = list(zip(self.net.nodes['lon'][self.chargers], self.net.nodes['lat'][self.chargers]))

        # Radius of the circles
        radius = 0.0002  # in degrees (adjust according to your coordinate system)

        # Create Shapely Points representing the centers
        center_points = [Point(coords) for coords in center_coords]

        # Create circular shapes (Polygons) around the centers
        circles = [point.buffer(radius).exterior for point in center_points]

        # Convert the geometries to WKT strings
        circle_wkts = [circle.wkt for circle in circles]

        # Create a DataFrame for self links
        chargers_list = list(self.chargers)
        chargers_osmid_list = [self.net.nid_to_osmid_dict[charger] for charger in chargers_list]
        self_links_df = pd.DataFrame({
            'link_id': range(self.l, self.l + len(chargers_list)),
            'start_node_id': chargers_list,
            'end_node_id': chargers_list,
            'length': [self.charger_self_link_length] * len(chargers_list),
            'start_osmid': chargers_osmid_list,
            'end_osmid': chargers_osmid_list,
            'geometry': circle_wkts,
        })
        # charger link parameters
        if self.parameter_fit_results is not None:
            self_links_parameters_df = pd.DataFrame({
                'link_id': range(self.l, self.l + len(chargers_list)),
                'link_length': self.charger_self_link_length,
                'fft_fit': self.charger_self_link_length,
                'free_flow_speed': 1,
                'capacity': 0.527778,
                'cap_fit': 0.527778,
                'a_fit': 0.0,
                'b_fit': 1,
                })
            self.parameter_fit_results = pd.concat([self.parameter_fit_results, self_links_parameters_df], ignore_index=True)

        # Update the value of self.l after creating self-links
        self.l += len(chargers_list)

        # Append self_links_df to self.net.edges
        self.net.edges = pd.concat([self.net.edges, self_links_df], ignore_index=True)

    def create_random_chargers(self, c):
        chargers = set()
        while len(chargers)!=c:
            chargers.add(np.random.randint(0, self.n))
        return chargers

    def find_k_shortest_paths(self, origin_node_id, destination_node_id, k, weight='length'):
        paths = list(islice(nx.shortest_simple_paths(
            self.DiGraph,
            self.net.nid_to_osmid_dict[origin_node_id],
            self.net.nid_to_osmid_dict[destination_node_id],
            weight=weight), k))
        return paths

    def create_OD_paths(self, od_pair, k):
        # creates different paths for the given OD pair
        paths = self.find_k_shortest_paths(od_pair[0], od_pair[1], k)
        n_paths = len(paths)
        return paths, n_paths

    def create_OCD_paths(self, od_pair):
        # creates different paths that go through the carger for the given OD pair
        # crate a set to remove redundancies in paths
        o, d = od_pair
        ocd_paths = set()
        ocd_paths_charger_not_repeated = set()
        for charger_i in self.chargers:
            oc_pair = (o, charger_i)
            cd_pair = (charger_i, d)

            oc_paths, k_oc = self.create_OD_paths(oc_pair, self.paths_per_oc_and_cd)
            cd_paths, k_cd = self.create_OD_paths(cd_pair, self.paths_per_oc_and_cd)

            for oc_paths_i in oc_paths:
                for cd_paths_j in cd_paths:
                    # Notice this accounts for the self loop at the charger
                    ocd_paths.add(tuple(oc_paths_i + cd_paths_j))
                    ocd_paths_charger_not_repeated.add(tuple(oc_paths_i + cd_paths_j[1:]))

        n_paths = len(ocd_paths)
        ocd_paths_list_of_lists = [list(path) for path in list(ocd_paths)]
        ocd_paths_charger_not_repeated_list_of_lists = [list(path) for path in list(ocd_paths_charger_not_repeated)]
        return ocd_paths_list_of_lists, ocd_paths_charger_not_repeated_list_of_lists, n_paths

    def get_od_pairs_and_demands(self, od_pairs, demands):
        if self.chargers is not None:
            self.d = int(len(demands)/2)
        else:
            self.d = len(demands)
        # create b
        self.b = demands
        # create routes
        n_paths_total = 0
        for od_pair in od_pairs:
            self.od_pairs_and_routes[od_pair] = {}
            paths, n_paths_i = self.create_OD_paths(od_pair, self.paths_per_od)

            self.od_pairs_and_routes[od_pair]["no charging type"] = list(map(lambda sublist: list(map(self.net.osmid_to_nid_dict.get, sublist)), paths))

            n_paths_total += n_paths_i
            if self.chargers is not None:
                paths, paths_charger_not_repeated, n_paths_i = self.create_OCD_paths(od_pair)
                self.od_pairs_and_routes[od_pair]["charging type"] = list(map(lambda sublist: list(map(self.net.osmid_to_nid_dict.get, sublist)), paths))

                n_paths_total += n_paths_i
        # Create A and R
        self.r = n_paths_total
        self.A = np.zeros((2*self.d, self.r), dtype=int)
        self.R = np.zeros((self.l, self.r), dtype=int)
        route_idx = 0
        for i, od_pair in enumerate(self.od_pairs_and_routes.keys()):
            for c, car_type in enumerate(self.car_types):
                od_pair_routes = self.od_pairs_and_routes[od_pair][car_type]
                t = len(od_pair_routes)
                self.A[2*i+c][route_idx:route_idx+t] = 1
                for j, od_pair_route_j in enumerate(od_pair_routes):
                    for k, route_j_elem in enumerate(od_pair_route_j):
                        if k == len(od_pair_route_j)-1:
                            break
                        links_df = self.net.edges
                        link_idx = int(links_df[(links_df['start_node_id'] == route_j_elem) & (links_df['end_node_id'] == od_pair_route_j[k+1])]['link_id'].iloc[0])
                        self.R[link_idx][route_idx+j] = 1
                route_idx += t

        # Identify active links: links that are used by at least one route and remove others
        active_links = np.any(self.R != 0, axis=1)

        # Filter the Routing matrix to include only these active links
        self.R = self.R[active_links, :]

        # Optionally, you can create a mapping or list of active link indices if needed for other purposes
        self.active_link_indices = np.where(active_links)[0]

        # BPR function values for each link (should be same size as number of links)
        # Extracting the 'a_fit' values for the active links using the active link indices
        self.a = self.parameter_fit_results.loc[self.parameter_fit_results['link_id'].isin(self.active_link_indices), 'a_fit'].values
        self.p = self.parameter_fit_results.loc[self.parameter_fit_results['link_id'].isin(self.active_link_indices), 'b_fit'].values
        self.cap = self.parameter_fit_results.loc[self.parameter_fit_results['link_id'].isin(self.active_link_indices), 'cap_fit'].values
        self.fft = self.parameter_fit_results.loc[self.parameter_fit_results['link_id'].isin(self.active_link_indices), 'fft_fit'].values
        self.link_length = self.parameter_fit_results.loc[self.parameter_fit_results['link_id'].isin(self.active_link_indices), 'link_length'].values

        # cvxpy objects
        param_df_sorted = self.parameter_fit_results.sort_values('link_id')
        self.a_cvxpy = param_df_sorted['a_fit'].values
        self.p_cvxpy = param_df_sorted['b_fit'].values
        self.cap_cvxpy = param_df_sorted['cap_fit'].values
        self.fft_cvxpy = param_df_sorted['fft_fit'].values
        self.link_length_cvxpy = param_df_sorted['link_length']
        # Set self.flow to zero initially for all links
        self.flow = np.zeros(self.l)

        assert len(self.a) == len(self.active_link_indices), (len(self.a), len(self.active_link_indices))

    def create_random_od_pairs_and_demands(self, d = 3):
        # Create a random demand for charging and no charging cars
        demands = np.random.randint(1, 10, size=2*d)
        for i in range(1, len(demands), 2):
            demands[i] = demands[i] * self.charging_to_no_charging_ratio
        od_pairs = set()
        while(len(od_pairs) < d):
            origin = (np.random.randint(0, self.n))
            destination = (np.random.randint(0, self.n))
            if  origin!=destination:
                od_pairs.add((origin,destination))
        self.od_pairs = od_pairs
        self.b = demands
        self.get_od_pairs_and_demands(od_pairs, demands)

    def record_paths_and_flows(self):
        self.paths_and_flows = {}
        route_idx = 0
        for od_pair in self.od_pairs_and_routes.keys():
            self.paths_and_flows[od_pair] = {}
            for car_type in self.car_types:
                od_pair_routes = self.od_pairs_and_routes[od_pair][car_type]
                self.paths_and_flows[od_pair][car_type] = []
                for route in od_pair_routes:
                    flow = self.res.x[route_idx]
                    self.paths_and_flows[od_pair][car_type].append({
                        'path': route,
                        'flow': flow
                    })
                    route_idx += 1

    def travel_time_function(self, x):
        # BPR function
        return self.fft*(1+self.a*(x/self.cap)**self.p)

    def potential(self, x, fft=0):
        # BPR function
        return (self.fft*(1+self.a/(self.p+1)*(x/self.cap)**self.p))*x

    def potential_first_derivative(self, x, fft=0):
        # BPR function derivative
        return self.fft*(1+self.a*(x/self.cap)**self.p)

    def potential_second_derivative(self, x, fft=0):
        # BPR function second derivative
        return self.fft*(self.a*self.p/self.cap*(x/self.cap)**(self.p-1))

    # Define the objective function
    def objective(self, x):
        self.objective_count += 1
        if np.any(x<0):
            return 1e100
        self.flow = np.dot(self.R, x)
        self.obj = np.sum(self.potential(self.flow))
        return self.obj

    # Define the travel time objective function
    def travel_time_objective(self):
        self.travel_time_obj = self.flow @ self.travel_time_function(self.flow)
        return self.travel_time_obj

    # Define the first derivative of the objective function
    def objective_first_derivative(self, x):
        self.objective_first_derivative_count += 1
        self.flow = np.dot(self.R, x)
        derivatives_vec = self.potential_first_derivative(self.flow)
        first_der =  self.R.T@derivatives_vec
        return first_der

    # Define the second derivative of the objective function
    def objective_second_derivative(self, x):
        self.objective_second_derivative_count += 1
        self.flow = np.dot(self.R, x)
        second_derivatives_vec = self.potential_second_derivative(self.flow)
        second_der =  self.R.T@np.diag(second_derivatives_vec)@self.R
        return second_der

    def optimization_callback(self, x, intermediate_result):
        # Callback function to create a history of objective values
        self.obj_history.append(intermediate_result.fun)

    def optimize(self, use_derivatives=False, disp=True, max_iter=1000, use_cvxpy=False):
        """Wrapper for optimization methods"""
        if use_cvxpy:
            return self.optimize_with_cvxpy(disp=True)
        else:
            try:
                # Original optimization method
                repetitions = np.sum(self.A, axis=1)
                equal_divider = np.divide(self.b, repetitions)
                # Initialize x0
                self.x0 = np.zeros(self.r)

                j = 0
                for i, repeat_i in enumerate(repetitions):
                    self.x0[j:j + repeat_i] = equal_divider[i]
                    j += repeat_i

                # Define the bounds for the route flows
                bounds = [(0, None) for _ in range(self.r)]
                lc1 = LinearConstraint(self.A, self.b, self.b)

                # Either use the derivatives or not
                if use_derivatives:
                    self.res = minimize(self.objective, self.x0,
                                        jac=self.objective_first_derivative, hess=self.objective_second_derivative,
                                        bounds=bounds, constraints=lc1,
                                        method="trust-constr", options={'disp': disp, 'maxiter': max_iter},
                                        callback=self.optimization_callback)
                else:
                    self.res = minimize(self.objective, self.x0,
                                        bounds=bounds, constraints=lc1,
                                        method="trust-constr", options={'disp': disp, 'maxiter': max_iter},
                                        callback=self.optimization_callback)
                # Calculate the travel time objective
                self.travel_time_objective()

                # Record the paths and their route flows after optimization
                self.record_paths_and_flows()
                return True
            except Exception as e:
                print(f"Optimization failed: {e}")
                # Set default values for failure case
                self.travel_time_obj = float('inf')
                return False

    def _generate_cvxpy_node_edge_mapping(self):
        """Create mappings between the original network and CVXPY formulation"""
        # Map nodes to sequential integers
        self.node_id_map = {self.net.nid_to_osmid_dict[i]: i for i in range(self.n)}
        self.inverse_node_id_map = {v: k for k, v in self.node_id_map.items()}

        # Create edge mapping
        self.edge_map = {}
        for idx, row in self.net.edges.iterrows():
            start_node = row['start_node_id']
            end_node = row['end_node_id']
            link_id = row['link_id']
            self.edge_map[link_id] = (start_node, end_node)

    def _create_cvxpy_incidence_matrix(self):
        """Create incidence matrix for CVXPY formulation"""
        A = np.zeros((self.n, self.l))

        for link_id, (start_node, end_node) in self.edge_map.items():
            A[start_node, link_id] = 1
            A[end_node, link_id] = -1

        return A

    def _create_cvxpy_charger_matrix(self):
        """Create charger incidence matrix for CVXPY formulation"""
        n_s = len(self.chargers)
        A_prime = np.zeros((n_s, self.l))

        for charger_idx, node in enumerate(self.chargers):
            for link_id, (start_node, end_node) in self.edge_map.items():
                if start_node == node:
                    A_prime[charger_idx, link_id] = 1

        return A_prime

    def _compute_objective(self, link_flows, charger_flows=None):
        """
        Unified objective function that works with both CVXPY variables and numerical arrays.
        
        Parameters:
        -----------
        link_flows : array-like or cvxpy.Variable
            Flow values for each link in the network (symbolic or numeric)
        charger_flows : array-like or cvxpy.Variable, optional
            Flow values for each charger (symbolic or numeric)
            
        Returns:
        --------
        Expression or float
            CVXPY expression if inputs are CVXPY variables, or float if inputs are numeric
        """
        # Determine if we're working with CVXPY variables (symbolic computation)
        is_symbolic = isinstance(link_flows, cp.Variable) or (
            hasattr(link_flows, '__len__') and len(link_flows) > 0 and 
            isinstance(link_flows[0], cp.expressions.expression.Expression)
        )
        
        # Initialize objective term
        if is_symbolic:
            obj_term = 0  # CVXPY will add to this expression
        else:
            obj_term = 0.0  # Regular float for numeric computation
        
        # Calculate link flow contribution
        for l in range(self.l):
            p_val = self.p_cvxpy[l] + 1
            flow = link_flows[l]
            fft = self.fft_cvxpy[l]
            a = self.a_cvxpy[l]
            cap = self.cap_cvxpy[l]
            
            if is_symbolic:
                # Symbolic computation for CVXPY
                term = fft * (flow + a * cap * cp.power(flow/cap, p_val) / p_val)
                obj_term += term
            else:
                # Numeric computation 
                if flow > 0 and cap > 0:
                    obj_term += fft * (flow + a * cap * (flow/cap)**p_val / p_val)
        
        # Add charger costs if provided
        if charger_flows is not None:
            # Handle different types of charger_flows
            if is_symbolic:
                # For CVXPY variables, we need to use the array dimension
                # If it's a single Variable, we iterate once
                # If it's an array of Variables, we iterate over each element
                if isinstance(charger_flows, cp.Variable):
                    if charger_flows.ndim == 0:  # Scalar
                        obj_term += self.charger_self_link_length * charger_flows
                    else:  # Vector
                        for c in range(charger_flows.size):
                            obj_term += self.charger_self_link_length * charger_flows[c]
                else:
                    # For other CVXPY expressions or sequences
                    for c_flow in charger_flows:
                        obj_term += self.charger_self_link_length * c_flow
            else:
                # Numeric computation - we can safely use len()
                charger_count = len(charger_flows)
                for c in range(charger_count):
                    if charger_flows[c] is not None:
                        obj_term += self.charger_self_link_length * charger_flows[c]
        
        return obj_term

    def optimize_with_cvxpy(self, disp=True):
        print("Running structured CVXPY optimization with link-specific BPR delay...")

        T = [1, 2]  # T=1: non-charging, T=2: charging
        od_pairs = list(self.od_demand.keys())
        N = len(od_pairs)
        S = list(self.chargers)
        n_s = len(S)
        I = np.eye(self.n)

        self._generate_cvxpy_node_edge_mapping()
        self.A = self._create_cvxpy_incidence_matrix()

        # === Total OD demand per (i, t)
        q_total = {(i, t): self.od_demand[od_pairs[i]][t - 1] for i in range(N) for t in T}
        
        # Calculate total network demand as an upper bound for flows
        total_network_demand = sum(demand for demands in q_total.values() for demand in [demands] if isinstance(demands, (int, float)))
        print(f"Total network demand: {total_network_demand}")
        
        # === Generate warm start values
        warm_start_values = self._generate_warm_start_values(od_pairs, q_total, N, n_s, S)
        
        # === Variables - with warm start values
        # Non-charging flows only exist at t=1
        x_nc = {(i, 1): cp.Variable(self.l, nonneg=True, value=warm_start_values['x_nc'].get((i, 1), np.zeros(self.l))) 
                for i in range(N)}
        q_nc = {(i, 1): q_total[i, 1] for i in range(N)}  # Fixed values, not variables
        
        # Charging flows only exist at t=2
        x_plus = {(i, 2, c): cp.Variable(self.l, nonneg=True, value=warm_start_values['x_plus'].get((i, 2, c), np.zeros(self.l))) 
                 for i in range(N) for c in range(n_s)}
        x_minus = {(i, 2, c): cp.Variable(self.l, nonneg=True, value=warm_start_values['x_minus'].get((i, 2, c), np.zeros(self.l))) 
                  for i in range(N) for c in range(n_s)}
        q_c = {(i, 2, c): cp.Variable(nonneg=True, value=warm_start_values['q_c'].get((i, 2, c), 0.0)) 
              for i in range(N) for c in range(n_s)}

        constraints = []

        # === Flow conservation constraints
        for i in range(N):
            o, d = od_pairs[i]
            
            # t=1: Non-charging vehicles only
            constraints.append(self.A @ x_nc[i, 1] == q_nc[i, 1] * (I[o] - I[d]))
            
            # Add upper bounds for non-charging flows
            for l in range(self.l):
                constraints.append(x_nc[i, 1][l] <= q_nc[i, 1])
            
            # t=2: Charging vehicles only - split demand among chargers
            constraints.append(cp.sum([q_c[i, 2, c] for c in range(n_s)]) == q_total[i, 2])
            
            for c in range(n_s):
                s_c = S[c]
                constraints.append(self.A @ x_plus[i, 2, c] == q_c[i, 2, c] * (I[o] - I[s_c]))
                constraints.append(self.A @ x_minus[i, 2, c] == q_c[i, 2, c] * (I[s_c] - I[d]))

        # === Total link flows - directly compute expressions
        x_total = cp.Variable(self.l, nonneg=True, value=warm_start_values['x_total'])
        
        # Build the total flow expression more efficiently
        total_flow_expr = sum(x_nc[i, 1] for i in range(N))
        for i in range(N):
            for c in range(n_s):
                total_flow_expr += x_plus[i, 2, c] + x_minus[i, 2, c]
        
        constraints.append(x_total == total_flow_expr)
        
        # Ensure all arrays have the same length (equal to the number of links)
        assert x_total.shape[0] == self.l, f"x_total dimension {x_total.shape[0]} != {self.l}"
        assert len(self.fft_cvxpy) == self.l, f"fft_cvxpy dimension {len(self.fft_cvxpy)} != {self.l}"
        assert len(self.a_cvxpy) == self.l, f"a_cvxpy dimension {len(self.a_cvxpy)} != {self.l}" 
        assert len(self.cap_cvxpy) == self.l, f"cap_cvxpy dimension {len(self.cap_cvxpy)} != {self.l}"
        assert len(self.p_cvxpy) == self.l, f"p_cvxpy dimension {len(self.p_cvxpy)} != {self.l}"

        # === Charger flows (self-link delay)
        x_hat = cp.Variable(n_s, nonneg=True, value=warm_start_values['x_hat'])
        for c in range(n_s):
            charger_total = cp.sum([q_c[i, 2, c] for i in range(N)])
            constraints.append(x_hat[c] == charger_total)
            
        # === Build the symbolic objective function for CVXPY
        objective_expr = self._compute_objective(x_total, x_hat)
        total_obj = cp.Minimize(objective_expr)
        prob = cp.Problem(total_obj, constraints)

        # Store flow contributions for later visualization
        self.flow_components = {
            "x_nc": x_nc,
            "x_plus": x_plus,
            "x_minus": x_minus
        }

        # Define solver parameters for better numerical stability
        clarabel_params = {
            'tol_gap_abs': 1e-5,  # Relaxed from 1e-6
            'tol_gap_rel': 1e-5,  # Relaxed from 1e-6
            'tol_feas': 1e-5,     # Relaxed from 1e-6
            'max_iter': 500,
        }
        
        # Get list of available solvers
        available_solvers = cp.installed_solvers()
        print(f"Available solvers: {available_solvers}")
        
        solver_order = []
        if 'CLARABEL' in available_solvers:
            solver_order.append(("CLARABEL", clarabel_params))
        
        # If CLARABEL is not available or if it fails, CVXPY will try its default solver.
        # The loop below handles trying solvers from solver_order first.
        # If solver_order is empty or all in it fail, a final attempt without specifying a solver can be made,
        # or simply let the error propagate if no solver works.
        # For now, we ensure that if CLARABEL is not in solver_order, the list remains empty, prompting CVXPY default.
            
        if not solver_order and available_solvers: # This will now only trigger if CLARABEL isn't available
            print("CLARABEL not found or not in available_solvers. Will attempt CVXPY default solver if primary fails.")
            # solver_order.append((available_solvers[0], {})) # No, let CVXPY pick its default if CLARABEL fails
            
        # Try solvers in priority order (CLARABEL if present)
        solver_success = False
        last_exception = None
        
        for solver, params in solver_order:
            try:
                print(f"Trying solver: {solver}")
                result = prob.solve(solver=solver, verbose=True, **params)
                
                # Check the problem status to determine if it actually succeeded
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    print(f"Solver {solver} succeeded with status: {prob.status}")
                    solver_success = True
                    break  # Exit the loop if solver succeeds with optimal solution
                else:
                    print(f"Solver {solver} completed but returned status: {prob.status}")
                    # Save the solution anyway if it's usable
                    if (x_total.value is not None and 
                        not np.any(np.isnan(x_total.value)) and 
                        prob.value is not None and 
                        np.isfinite(prob.value)):
                        print(f"Solution appears usable with objective value: {prob.value}")
                        solver_success = True
                        break
                    # Continue to next solver since this one didn't get an optimal solution
            except Exception as e:
                last_exception = e
                print(f"Solver {solver} failed with exception: {e}")
                # Continue to the next solver automatically
        
        # Check if any solver succeeded
        if not solver_success:
            # Try with CVXPY default if CLARABEL (or specified solvers) failed and solver_order was attempted
            if solver_order: # Only try default if specific solvers were attempted and failed
                print("Trying CVXPY default solver...")
                try:
                    result = prob.solve(verbose=True) # Try CVXPY default
                    if prob.status in ["optimal", "optimal_inaccurate"]:
                        print(f"CVXPY default solver succeeded with status: {prob.status}")
                        solver_success = True
                    else:
                        print(f"CVXPY default solver completed but returned status: {prob.status}")
                        if (x_total.value is not None and 
                            not np.any(np.isnan(x_total.value)) and 
                            prob.value is not None and 
                            np.isfinite(prob.value)):
                            print(f"Solution from default solver appears usable with objective value: {prob.value}")
                            solver_success = True
                except Exception as e:
                    print(f"CVXPY default solver failed: {e}")
                    last_exception = e # Update last_exception to reflect default solver failure

        if not solver_success:
            print("All solvers (including CVXPY default if attempted) failed to find an optimal solution. Check problem formulation.")
            if last_exception:
                 print(f"Last error: {last_exception}") # Print the last error encountered
            # Set default values so the program can continue
            self.cvxpy_link_flows = np.zeros(self.l)
            self.best_objective_value = float('inf')
            self.travel_time_obj = float('inf')
            return None
            
        print(f"CVXPY optimization complete. Objective value: {prob.value}")
        print(f"Solver status: {prob.status}")
        print(f"Solve time: {prob.solver_stats.solve_time} seconds")
        print(f"Iterations: {prob.solver_stats.num_iters}")

        # If prob.value is None or NaN, compute it manually using our method
        if prob.value is None or np.isnan(prob.value):
            computed_obj_value = self._compute_objective(x_total.value, x_hat.value)
            print(f"Computed objective value manually: {computed_obj_value}")
            self.best_objective_value = computed_obj_value
            self.travel_time_obj = computed_obj_value
        else:
            self.best_objective_value = prob.value
            self.travel_time_obj = prob.value
            
        # Store optimization results
        self.cvxpy_link_flows = x_total.value if x_total is not None else np.zeros(self.l)
        self.cvxpy_charger_throughput = x_hat.value if x_hat is not None and x_hat.value is not None else np.zeros(n_s)
        self.cvxpy_link_flows_data = self.cvxpy_link_flows

        # Store demand splits for analysis
        self.q_c_values = {k: v.value for k, v in q_c.items()}
        
        # Save individual flow contributions
        self._save_individual_flow_contributions()
        
        print("Structured CVXPY optimization complete.")
        return self.best_objective_value

    def _generate_warm_start_values(self, od_pairs, q_total, N, n_s, S):
        """Generate warm start values for CVXPY optimization using shortest paths"""
        print("Generating warm start values for CVXPY optimization...")
        
        # Initialize dictionaries for warm start values
        x_nc_warm = {}
        x_plus_warm = {}
        x_minus_warm = {}
        q_c_warm = {}
        x_total_warm = np.zeros(self.l)
        x_hat_warm = np.zeros(n_s)
        
        # Process non-charging flows (t=1)
        for i in range(N):
            o, d = od_pairs[i]
            demand = q_total[i, 1]
            
            if demand > 0:
                # Find shortest path for this OD pair
                try:
                    path = nx.shortest_path(self.DiGraph, 
                                          self.net.nid_to_osmid_dict[o],
                                          self.net.nid_to_osmid_dict[d], 
                                          weight='length')
                    
                    # Convert OSM IDs to node IDs
                    path_nids = [self.net.osmid_to_nid_dict[osmid] for osmid in path]
                    
                    # Initialize flow array for this OD pair
                    flow_array = np.zeros(self.l)
                    
                    # Assign demand to links along the shortest path
                    for j in range(len(path_nids) - 1):
                        start_node = path_nids[j]
                        end_node = path_nids[j + 1]
                        
                        # Find the link ID for this edge
                        link_indices = self.net.edges[
                            (self.net.edges['start_node_id'] == start_node) & 
                            (self.net.edges['end_node_id'] == end_node)
                        ]['link_id'].values
                        
                        if len(link_indices) > 0:
                            link_id = link_indices[0]
                            flow_array[link_id] = demand
                    
                    # Store the flow array
                    x_nc_warm[i, 1] = flow_array
                    
                    # Add to total flow
                    x_total_warm += flow_array
                    
                except (nx.NetworkXNoPath, KeyError) as e:
                    print(f"Warning: No path found for OD pair ({o}, {d}). Error: {e}")
                    x_nc_warm[i, 1] = np.zeros(self.l)
            else:
                x_nc_warm[i, 1] = np.zeros(self.l)
                
        # Process charging flows (t=2)
        for i in range(N):
            o, d = od_pairs[i]
            charging_demand = q_total[i, 2]
            
            if charging_demand > 0:
                # Distribute demand evenly among chargers
                demand_per_charger = charging_demand / n_s
                
                for c in range(n_s):
                    charger_node = S[c]
                    q_c_warm[i, 2, c] = demand_per_charger
                    x_hat_warm[c] += demand_per_charger
                    
                    # Find shortest path from origin to charger
                    try:
                        path_to_charger = nx.shortest_path(self.DiGraph, 
                                                         self.net.nid_to_osmid_dict[o],
                                                         self.net.nid_to_osmid_dict[charger_node], 
                                                         weight='length')
                        
                        # Convert OSM IDs to node IDs
                        path_to_charger_nids = [self.net.osmid_to_nid_dict[osmid] for osmid in path_to_charger]
                        
                        # Initialize flow arrays
                        flow_to_charger = np.zeros(self.l)
                        
                        # Assign demand to links along the path to charger
                        for j in range(len(path_to_charger_nids) - 1):
                            start_node = path_to_charger_nids[j]
                            end_node = path_to_charger_nids[j + 1]
                            
                            # Find the link ID
                            link_indices = self.net.edges[
                                (self.net.edges['start_node_id'] == start_node) & 
                                (self.net.edges['end_node_id'] == end_node)
                            ]['link_id'].values
                            
                            if len(link_indices) > 0:
                                link_id = link_indices[0]
                                flow_to_charger[link_id] = demand_per_charger
                        
                        # Store the flow array
                        x_plus_warm[i, 2, c] = flow_to_charger
                        
                        # Add to total flow
                        x_total_warm += flow_to_charger
                        
                    except (nx.NetworkXNoPath, KeyError) as e:
                        print(f"Warning: No path found from origin {o} to charger {charger_node}. Error: {e}")
                        x_plus_warm[i, 2, c] = np.zeros(self.l)
                        
                    # Find shortest path from charger to destination
                    try:
                        path_from_charger = nx.shortest_path(self.DiGraph, 
                                                           self.net.nid_to_osmid_dict[charger_node],
                                                           self.net.nid_to_osmid_dict[d], 
                                                           weight='length')
                        
                        # Convert OSM IDs to node IDs
                        path_from_charger_nids = [self.net.osmid_to_nid_dict[osmid] for osmid in path_from_charger]
                        
                        # Initialize flow array
                        flow_from_charger = np.zeros(self.l)
                        
                        # Assign demand to links along the path from charger
                        for j in range(len(path_from_charger_nids) - 1):
                            start_node = path_from_charger_nids[j]
                            end_node = path_from_charger_nids[j + 1]
                            
                            # Find the link ID
                            link_indices = self.net.edges[
                                (self.net.edges['start_node_id'] == start_node) & 
                                (self.net.edges['end_node_id'] == end_node)
                            ]['link_id'].values
                            
                            if len(link_indices) > 0:
                                link_id = link_indices[0]
                                flow_from_charger[link_id] = demand_per_charger
                        
                        # Store the flow array
                        x_minus_warm[i, 2, c] = flow_from_charger
                        
                        # Add to total flow
                        x_total_warm += flow_from_charger
                        
                    except (nx.NetworkXNoPath, KeyError) as e:
                        print(f"Warning: No path found from charger {charger_node} to destination {d}. Error: {e}")
                        x_minus_warm[i, 2, c] = np.zeros(self.l)
        
        # Return all warm start values
        return {
            'x_nc': x_nc_warm,
            'x_plus': x_plus_warm,
            'x_minus': x_minus_warm,
            'q_c': q_c_warm,
            'x_total': x_total_warm,
            'x_hat': x_hat_warm
        }

    def _save_individual_flow_contributions(self):
        """Extract and save individual flow contributions from optimization results"""
        if not hasattr(self, 'flow_components'):
            return
            
        # Initialize flow arrays for each component
        n_links = self.l
        non_charging_flows = np.zeros(n_links)
        charger_contributions = {"non_charging": non_charging_flows}
        
        # Add up non-charging flows (from t=1)
        for i in range(len(self.od_demand)):
            if (i, 1) in self.flow_components["x_nc"]:
                flow_value = self.flow_components["x_nc"][i, 1].value
                if flow_value is not None:
                    non_charging_flows += flow_value
        
        # Extract individual charger contributions (from t=2)
        for c in range(len(self.chargers)):
            charger_flows = np.zeros(n_links)
            
            for i in range(len(self.od_demand)):
                # Add flow to the charger
                if (i, 2, c) in self.flow_components["x_plus"]:
                    flow_value = self.flow_components["x_plus"][i, 2, c].value
                    if flow_value is not None:
                        charger_flows += flow_value
                
                # Add flow from the charger
                if (i, 2, c) in self.flow_components["x_minus"]:
                    flow_value = self.flow_components["x_minus"][i, 2, c].value
                    if flow_value is not None:
                        charger_flows += flow_value
            
            charger_contributions[f"charger_{c}"] = charger_flows
        
        # Store the results
        self.charger_flow_contributions = charger_contributions
        
    def get_charger_contributions(self):
        """Return the flow contributions from each charger"""
        if hasattr(self, 'charger_flow_contributions'):
            return self.charger_flow_contributions
            
        if not hasattr(self, 'flow_components'):
            print("Charger contributions not available. Run optimization with CVXPY first.")
            return None
            
        self._save_individual_flow_contributions()
        return self.charger_flow_contributions 