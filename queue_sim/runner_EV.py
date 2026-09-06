###################################################################### ADDITION STARTS

#from .queue_model import Simulation, Node, Link
# from .ev_queue_model import Simulation, Node, Link, EV_Charging_Station
from .queue_model_EV import Simulation, Node, Link, EV_Charging_Station
import geopandas as gpd
from matplotlib import pyplot as plt
import random
import networkx as nx
from itertools import islice
import osmnx as ox
import itertools
import os

###################################################################### ADDITION ENDS

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse


def _legacy_flow_counts(routes, od_df, od, charging):
    """Convert legacy aggregate flow records into deterministic route counts."""
    if isinstance(od, str):
        origin, destination = [int(value) for value in od.split(',')]
    else:
        origin, destination = [int(value) for value in od]
    if 'is_EV' in od_df.columns and 'need_to_charge' in od_df.columns:
        mask = (od_df['origin_node_id'] == origin) & (od_df['destin_node_id'] == destination)
        if charging:
            mask &= od_df['is_EV'].astype(bool) & od_df['need_to_charge'].astype(bool)
        else:
            mask &= ~(od_df['is_EV'].astype(bool) & od_df['need_to_charge'].astype(bool))
        total = int(mask.sum())
    else:
        total = int(round(sum(float(route.get('flow', 0.0)) for route in routes)))
    weights = [max(0.0, float(route.get('flow', 0.0))) for route in routes]
    if not routes:
        return []
    weight_sum = sum(weights)
    quotas = [weight / weight_sum * total for weight in weights] if weight_sum else [0.0] * len(weights)
    counts = [int(np.floor(value)) for value in quotas]
    for index in sorted(range(len(counts)), key=lambda i: (-(quotas[i] - counts[i]), i))[:total - sum(counts)]:
        counts[index] += 1
    return counts


class Runner:
    def __init__(self,
                 links_csv: str, nodes_csv: str, od_csv: str,
                 contraflow_csv='',
                 NodeClass=Node, LinkClass=Link, ChargingStationClass=EV_Charging_Station,
                 reroute_freq=10800, seed=None): ###################################################################### ADDITION INLINE
        self.nodes_df = pd.read_csv(nodes_csv)
        self.links_df = pd.read_csv(links_csv)
        self.od_df = pd.read_csv(od_csv)
        self.sim = Simulation(NodeClass, LinkClass, ChargingStationClass) ###################################################################### ADDITION INLINE
        self.seed = seed
        self.sim.random_seed = seed
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))
        self.reroute_freq = reroute_freq
        self.running = True

        ###################################################################### ADDITION STARTS

        self.charging_stations_df = pd.DataFrame() #self.charging_stations_df = []
        self.tot_travel_time = 0
        self.simulation_status = 'not_started'
        self.simulation_error = None

        self.strategies = [] # For bpr function fitting
        self.ave_link_densities = {} # For bpr function fitting

        self.active_links = []

        self.no_ch_routes = []
        self.ch_routes = []
        self.route_groups = []
        self.route_agent_ids = {}

        ###################################################################### ADDITION ENDS

        if contraflow_csv:
            contraflow_df = pd.read_csv(contraflow_csv)
            self.links_df = self.links_df.merge(
                contraflow_df[['link_id', 'new_lanes']], how='left', on='link_id')

    def reset_for_new_simulation(self):
        """Reset mutable simulation state while retaining immutable CSV inputs.

        This is used by batched BPR workers so one Runner can execute a full
        flow sweep without carrying agents, queues, or statistics between
        samples.
        """
        self.sim = Simulation(self.sim.NodeClass, self.sim.LinkClass, self.sim.ChargingStationClass)
        self.sim.random_seed = self.seed
        self.running = True
        self.tot_travel_time = 0
        self.simulation_status = 'not_started'
        self.simulation_error = None
        self.strategies = []
        self.ave_link_densities = {}
        self.active_links = []
        self.no_ch_routes = []
        self.ch_routes = []
        self.route_groups = []
        self.route_agent_ids = {}

###################################################################### ADDITION STARTS

    def create_EV_charging_station_at_node(self, station_node_id, ent_capacity, charging_capacity, exit_capacity, cost):  # Create station at node_id

        in_links = self.links_df.loc[self.links_df.end_node_id == station_node_id]
        if len(in_links) == 0:  # There are no roads that enter the station node
            print("There aren't any roads that enter the specified node, try a different node.")
            return
        main_link = in_links.iloc[random.choice(range(len(in_links)))]
        mask = in_links['link_id'] == main_link.link_id
        sub_links = in_links[~mask]

        node_index = self.nodes_df.index[self.nodes_df.node_id == station_node_id].tolist()[0]
        x_end, y_end = self.nodes_df.iloc[node_index]['lon'], self.nodes_df.iloc[node_index]['lat']

        self.nodes_df.loc[node_index,'type'] = 'Station_Entrance/Exit'

        main_start_node_index = self.nodes_df.index[self.nodes_df.node_id == main_link.start_node_id].tolist()[0]
        main_x_start, main_y_start = self.nodes_df.iloc[main_start_node_index]['lon'], self.nodes_df.iloc[main_start_node_index]['lat']

        in_vec = (main_x_start - x_end, main_y_start - y_end)

        ang = []
        for sub_link in sub_links.itertuples():
            sub_start_node_index = self.nodes_df.index[self.nodes_df.node_id == sub_link.start_node_id].tolist()[0]
            sub_x_start, sub_y_start = self.nodes_df.iloc[sub_start_node_index]['lon'], self.nodes_df.iloc[sub_start_node_index]['lat']

            out_vec = (sub_x_start - x_end, sub_y_start - y_end)

            dot, det = (in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1]), (in_vec[0] * out_vec[1] - in_vec[1] * out_vec[0])
            ang.append(np.arctan2(det, dot) * 180 / np.pi)

        # With multiple incoming roads, preserve the historical angular
        # placement.  A valid directed-network node can have exactly one
        # incoming road (and still belong to the SCC).  In that case there is
        # no second angle to bisect, so place the symbolic station link
        # perpendicular to the sole incoming road instead of calling
        # ``min`` on an empty sequence.  This geometry is only a visual and
        # topological offset for the virtual station links; it does not alter
        # road travel times or capacities.
        if ang:
            station_ang = min(ang) / 2
        else:
            incoming_angle = np.degrees(np.arctan2(in_vec[1], in_vec[0]))
            station_ang = incoming_angle + 90.0

        # Build two links: One is from the station entrance to the station, the other from the station to the station's exit
        # These links are symbolic (virtual), but we still add them to the dataframes for simulation purposes
        station_link_len = 0.0003
        station_end_lon = x_end + station_link_len * np.cos(station_ang * np.pi / 180)
        station_end_lat = y_end + station_link_len * np.sin(station_ang * np.pi / 180)
        station_node_data = {'node_id': len(self.nodes_df), 'lon': station_end_lon, 'lat': station_end_lat,
                             'node_osmid': 'Station', 'type': 'Station'}
        station_node = pd.DataFrame(data=station_node_data, index=[0])
        self.nodes_df = pd.concat([self.nodes_df, station_node]).reset_index(drop=True)
        station_inlink_data = {'link_id': len(self.links_df), 'start_node_id': station_node_id,
                               'end_node_id': station_node_data['node_id'],
                               'type': 'In_Station', 'length': station_link_len, 'maxmph': np.nan, 'lanes': 1,
                               'capacity': np.nan,
                               'start_osmid': 'Station_Entrance/Exit', 'end_osmid': 'Station_Entrance/Exit',
                               'geometry': 'LINESTRING({} {}, {} {})'.format(x_end, y_end, station_end_lon,
                                                                             station_end_lat)}
        station_outlink_data = {'link_id': len(self.links_df) + 1, 'start_node_id': station_node_data['node_id'],
                                'end_node_id': station_node_id,
                                'type': 'Out_Station', 'length': station_link_len, 'maxmph': np.nan, 'lanes': 1,
                                'capacity': np.nan,
                                'start_osmid': 'Station_Entrance/Exit', 'end_osmid': 'Station_Entrance/Exit',
                                'geometry': 'LINESTRING({} {}, {} {})'.format(station_end_lon, station_end_lat, x_end,
                                                                              y_end)}
        self.links_df = pd.concat([self.links_df, pd.DataFrame([station_inlink_data])]).reset_index(drop=True)
        self.links_df = pd.concat([self.links_df, pd.DataFrame([station_outlink_data])]).reset_index(drop=True)
        # Reset the link ids
        self.links_df = self.links_df.assign(link_id=range(len(self.links_df)))

        # Append the EV_Charging_Station object to the charging station dataframe
        charging_station_data = {'station_id': len(self.charging_stations_df),
                                 'in_link_id': station_inlink_data['link_id'], 'out_link_id': station_outlink_data['link_id'],
                                 'node_id': station_node_data['node_id'], 'ent_ex_node_id' : station_node_id,
                                 'lon': x_end, 'lat': y_end,
                                 'ent_capacity': ent_capacity,
                                 'charging_capacity': charging_capacity, 'exit_capacity': exit_capacity, 'cost': cost}
        new_charging_station = pd.DataFrame(data=charging_station_data, index=[0])
        self.charging_stations_df = pd.concat([self.charging_stations_df, new_charging_station]).reset_index(drop=True)

        # Display the charging stations dataframe
        # display(self.charging_stations_df)

        # Plot the stations
        # nodesOX = gpd.GeoDataFrame(
        #     geometry=gpd.points_from_xy(x=self.nodes_df['lon'].to_list(), y=self.nodes_df['lat'].to_list()))
        # edgesOX = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.links_df['geometry']).tolist())
        # station_locs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=self.charging_stations_df['lon'].to_list(),
        #                                                             y=self.charging_stations_df['lat'].to_list()))
        # station_links = gpd.GeoDataFrame(
        #     geometry=gpd.GeoSeries.from_wkt(self.links_df[self.links_df['type'] == 'In_Station'].geometry.tolist()))
        # station_nodes = gpd.GeoDataFrame(
        #     geometry=gpd.points_from_xy(x=self.nodes_df.loc[self.nodes_df['type'] == 'Station', 'lon'].to_list(),
        #                                 y=self.nodes_df.loc[self.nodes_df['type'] == 'Station', 'lat'].to_list()))
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # fig.patch.set_facecolor('black')
        # ax.set_axis_off()
        # nodesOX.plot(ax=ax, color='white')
        # edgesOX.plot(ax=ax, color='white')
        # station_locs.plot(ax=ax, color='g', zorder=2)
        # station_links.plot(ax=ax, color='g', zorder=2)
        # station_nodes.plot(ax=ax, color='g', zorder=2)


    def create_EV_charging_station(self, node1_id, node2_id, ent_capacity, charging_capacity, exit_capacity, cost): # Create station on the link(s) that connect node1 and node2
        # Get the candidate links that connect node1 and node2
        links_from_1_to_2 = self.links_df[(self.links_df['start_node_id'] == node1_id) & (self.links_df['end_node_id'] == node2_id)]
        links_from_2_to_1 = self.links_df[(self.links_df['start_node_id'] == node2_id) & (self.links_df['end_node_id'] == node1_id)]
        candid_links = pd.concat([links_from_1_to_2, links_from_2_to_1])
        
        # Get the candidate link
        #link = self.links_df[self.links_df['link_id'] == candid_link_id]
        
        # Check if the candidate links are admissible
        if len(candid_links) == 0: # There are no links that connect the specified nodes
            print("Specified nodes are not neighbors, try different nodes.")
            return
        elif len(candid_links) > 3: # There are more than 2 links that connect the specified nodes
            print("There are more than 2 links that connect the specified nodes, try different nodes.")
            return
        else:
            for index, link in candid_links.iterrows():
                if gpd.GeoSeries.from_wkt([link['geometry']]).type.iloc[0] != 'LineString': # There is a nonlinear link that connect the nodes
                    print("At least one of the links that connect the nodes is not a LineString, try different nodes.")
                    return
        # At this point, we know that the candidate link is admissible
        
        # Find the midpoint of the nodes, this is location of the station's entrance/exit
        link = candid_links.iloc[0]
        midpo = gpd.GeoSeries.from_wkt([link['geometry']]).centroid.iloc[0]
        # Create a new node at the midpoint
        station_ent_ex_data = {'node_id': len(self.nodes_df), 'lon': midpo.x, 'lat': midpo.y, 'node_osmid': 'Station_Entrance/Exit', 'type': 'Station_Entrance/Exit'}
        station_ent_ex = pd.DataFrame(data = station_ent_ex_data, index=[0])
        # Append the new node to the list of all nodes
        self.nodes_df = pd.concat([self.nodes_df, station_ent_ex]).reset_index(drop=True)
        
        # Find the angle of the candidate link
        x_start = self.nodes_df.loc[node1_id]['lon']
        y_start = self.nodes_df.loc[node1_id]['lat']
        x_end = self.nodes_df.loc[node2_id]['lon']
        y_end = self.nodes_df.loc[node2_id]['lat']
        dely = y_end-y_start
        delx = x_end-x_start
        ang = np.arctan2(dely, delx)*180/np.pi
        # Find an angle that is perpendicular to the candidate link's angle (choose between +90 or -90 randomly)
        perp_ang = ang+90*random.choice([-1,1])
        
        sub_inlinks = []
        sub_outlinks = []
        for index, link in candid_links.iterrows():
            # Create a link that starts from the start node of the candidate link and ends at the station_ent_ex
            sub_inlink_data = {'link_id': len(self.links_df), 'start_node_id': link['start_node_id'], 'end_node_id': station_ent_ex.iloc[0]['node_id'],
                               'type': 'Link_to_Station_Entrance', 'length': link['length']/2, 'maxmph': link['maxmph'], 'lanes': link['lanes'],
                               'capacity': link['capacity'], 'start_osmid': link['start_osmid'], 'end_osmid': 'Station_Entrance/Exit',
                               'geometry': 'LINESTRING({} {}, {} {})'.format(self.nodes_df.loc[link['start_node_id']]['lon'], self.nodes_df.loc[link['start_node_id']]['lat'], midpo.x, midpo.y)}
            sub_inlinks.append(sub_inlink_data)
            # Create a link that starts from the station_ent_ex and ends at the end node of the candidate link
            sub_outlink_data = {'link_id': len(self.links_df)+1, 'start_node_id': station_ent_ex.iloc[0]['node_id'], 'end_node_id': link['end_node_id'],
                                'type': 'Link_from_Station_Exit', 'length': link['length']/2, 'maxmph': link['maxmph'], 'lanes': link['lanes'],
                                'capacity': link['capacity'], 'start_osmid': 'Station_Entrance/Exit', 'end_osmid': link['end_osmid'],
                                'geometry': 'LINESTRING({} {}, {} {})'.format(midpo.x, midpo.y, self.nodes_df.loc[link['end_node_id']]['lon'], self.nodes_df.loc[link['end_node_id']]['lat'])}
            sub_outlinks.append(sub_outlink_data)
            
        # Remove the links that were split
        mask = [False if i in candid_links['link_id'].tolist() else True for i in self.links_df['link_id'].tolist()]
        self.links_df = self.links_df[mask]
        # Concatenate the new links (resulting from splitting) to the link dataframe
        self.links_df = pd.concat([self.links_df, pd.DataFrame(sub_inlinks), pd.DataFrame(sub_outlinks)]).reset_index(drop=True)
        # Reset the link ids
        self.links_df = self.links_df.assign(link_id = range(len(self.links_df)))
        
        # Build one node and two links: Node represents the station and
        # One line is from station entrance to station, the other from station to station's exit
        # These node and links are symbolic (virtual), but we still add them to the dataframes for simulation purposes
        station_link_len = 0.0003
        station_end_lon = midpo.x + station_link_len*np.cos(perp_ang*np.pi/180)
        station_end_lat = midpo.y + station_link_len*np.sin(perp_ang*np.pi/180)
        station_node_data = {'node_id': len(self.nodes_df), 'lon': station_end_lon, 'lat': station_end_lat, 'node_osmid': 'Station', 'type': 'Station'}
        station_node = pd.DataFrame(data=station_node_data, index=[0])
        self.nodes_df = pd.concat([self.nodes_df, station_node]).reset_index(drop=True)
        station_inlink_data = {'link_id': len(self.links_df), 'start_node_id': station_ent_ex.iloc[0]['node_id'], 'end_node_id': station_node_data['node_id'],
                          'type': 'In_Station', 'length': station_link_len, 'maxmph': np.nan, 'lanes': 1, 'capacity': np.nan,
                          'start_osmid': 'Station_Entrance/Exit', 'end_osmid': 'Station_Entrance/Exit',
                          'geometry': 'LINESTRING({} {}, {} {})'.format(midpo.x, midpo.y, station_end_lon, station_end_lat)}
        station_outlink_data = {'link_id': len(self.links_df)+1, 'start_node_id': station_node_data['node_id'], 'end_node_id': station_ent_ex.iloc[0]['node_id'],
                          'type': 'Out_Station', 'length': station_link_len, 'maxmph': np.nan, 'lanes': 1, 'capacity': np.nan,
                          'start_osmid': 'Station_Entrance/Exit', 'end_osmid': 'Station_Entrance/Exit',
                          'geometry': 'LINESTRING({} {}, {} {})'.format(station_end_lon, station_end_lat, midpo.x, midpo.y)}
        self.links_df = pd.concat([self.links_df, pd.DataFrame([station_inlink_data])]).reset_index(drop=True)
        self.links_df = pd.concat([self.links_df, pd.DataFrame([station_outlink_data])]).reset_index(drop=True)
        # Reset the link ids
        self.links_df = self.links_df.assign(link_id = range(len(self.links_df)))

        # Append the EV_Charging_Station object to the charging station dataframe
        charging_station_data = {'station_id': len(self.charging_stations_df), 'in_link_id': -1, 'out_link_id': -1, 'node_id': station_node_data['node_id'],
                                 'ent_ex_node_id': station_ent_ex_data['node_id'], 'parent_node1_id': node1_id, 'parent_node2_id': node2_id, 'lon': midpo.x, 'lat': midpo.y, 'ent_capacity': ent_capacity,
                                 'charging_capacity': charging_capacity, 'exit_capacity': exit_capacity, 'cost': cost}
        new_charging_station = pd.DataFrame(data=charging_station_data, index=[0])
        if charging_station_data['station_id'] == 0:
            self.charging_stations_df = pd.DataFrame(new_charging_station, index=[0])
            self.charging_stations_df.loc[0, 'in_link_id'] = len(self.links_df)-2
            self.charging_stations_df.loc[0, 'out_link_id'] = len(self.links_df)-1
        else:
            new_charging_station = pd.DataFrame(data=charging_station_data, index=[0])
            self.charging_stations_df = pd.concat([self.charging_stations_df, new_charging_station]).reset_index(drop=True)
            # Need to readjust previously added stations' link_ids due to the removal of the candidate link
            for index, link in self.links_df.iterrows():
                if link.type == 'In_Station':
                    self.charging_stations_df.loc[self.charging_stations_df['node_id'] == link.end_node_id, 'in_link_id'] = link.link_id
                elif link.type == 'Out_Station':
                    self.charging_stations_df.loc[self.charging_stations_df['node_id'] == link.start_node_id, 'out_link_id'] = link.link_id

        # Display the charging stations dataframe
        # display(self.charging_stations_df)
            
        # Plot the stations
        # nodesOX = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=self.nodes_df['lon'].to_list(), y=self.nodes_df['lat'].to_list()))
        # edgesOX = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.links_df['geometry']).tolist())
        # station_locs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=self.charging_stations_df['lon'].to_list(), y=self.charging_stations_df['lat'].to_list()))
        # station_links = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.links_df[self.links_df['type']=='In_Station'].geometry.tolist()))
        # station_nodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=self.nodes_df.loc[self.nodes_df['type']=='Station','lon'].to_list(), y=self.nodes_df.loc[self.nodes_df['type']=='Station','lat'].to_list()))
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # fig.patch.set_facecolor('black')
        # ax.set_axis_off()
        # nodesOX.plot(ax=ax, color='white')
        # edgesOX.plot(ax=ax, color='white')
        # station_locs.plot(ax=ax,color='g',zorder=2)
        # station_links.plot(ax=ax,color='g',zorder=2)
        # station_nodes.plot(ax=ax, color='g', zorder=2)

    def add_charging_info(self, per_of_EVs, per_need_to_charge): # Add agents' charging-related information to the o-d dataframe
        # Calculate the number of electric vehicles
        num_of_EVs = round(per_of_EVs*len(self.od_df))
        # Randomly select num_of_EVs many agents, these will be the electric vehicles
        EV_agent_ids = random.sample(range(len(self.od_df)), num_of_EVs)
        # Add 'is_EV' column to the o-d dataframe (value is True if the corresponding agent is EV)
        self.od_df['is_EV'] = [True if i in EV_agent_ids else False for i in range(len(self.od_df))]
        # Calculate the number of EVs that need to charge
        num_need_to_charge = round(per_need_to_charge*num_of_EVs)
        # Randomly select num_need_to_charge many EVs
        need_to_charge_ids = random.sample(EV_agent_ids, num_need_to_charge)
        # Add 'need_to_charge' column to the o-d dataframe (value is True if the corresponding EV needs to charge)
        self.od_df['need_to_charge'] = [True if i in need_to_charge_ids else False for i in range(len(self.od_df))]
        
        # Set the current and targer charge levels to 0 and 100 respectively
        self.od_df['current_charge'] = 0
        self.od_df['target_charge'] = 100
        # Add the 'go_to_station_id' column, its values are populated later
        self.od_df['go_to_station_id'] = np.nan

###################################################################### ADDITION ENDS
    
    def init_sq_simulation(self):
        self.sim.create_network(self.nodes_df, self.links_df, self.charging_stations_df) ###################################################################### ADDITION INLINE
        self.sim.create_demand(self.od_df)

        # remove vehicles from self.sim w/o path to the end
        cannot_find_path = []
        for vehicle_id, vehicle in self.sim.all_agents.items():
            routing_status = vehicle.get_path(g=self.sim.g)
            if routing_status == 'no_path_found':
                cannot_find_path.append(vehicle_id)

        [self.sim.all_agents.pop(vh_id) for vh_id in cannot_find_path]

        # print(
        #     f'# o-d pairs whose paths cannot be found: {len(cannot_find_path)}')
        # print(f'# o-d pairs/trips {len(self.sim.all_agents)}')

###################################################################### ADDITION STARTS

    def find_k_shortest_paths(self, orig, dest, k, DiGraph, raw_nodes, weight='length'):
        paths = list(islice(nx.shortest_simple_paths(DiGraph, orig, dest, weight=weight), k))
        path_node_ids = []
        for path in paths:
            path_node_ids.append([raw_nodes.loc[raw_nodes.osmid == osmid, 'node_id'].values[0] for osmid in path])
        return path_node_ids

    def init_sq_simulation_for_bpr_function_fitting_V1(self, x, k, map):
        self.sim.create_network(self.nodes_df, self.links_df, self.charging_stations_df)
        self.sim.create_demand(self.od_df)

        nodesOX, edgesOX = ox.graph_to_gdfs(map)
        raw_nodes = nodesOX.copy().reset_index()
        raw_nodes['node_id'] = np.arange(raw_nodes.shape[0])
        od_pairs = []
        od_pairs_osmid = []
        for od in self.od_df.itertuples():
            od_pairs.append([raw_nodes.loc[raw_nodes.node_id==od.origin_node_id, 'node_id'].values[0],
                             raw_nodes.loc[raw_nodes.node_id==od.destin_node_id, 'node_id'].values[0]])
            od_pairs_osmid.append([raw_nodes.loc[raw_nodes.node_id==od.origin_node_id, 'osmid'].values[0],
                             raw_nodes.loc[raw_nodes.node_id==od.destin_node_id, 'osmid'].values[0]])
        od_pairs = [list(od) for od in set(tuple(od) for od in od_pairs)]
        od_pairs_osmid = [list(od) for od in set(tuple(od) for od in od_pairs_osmid)]

        DiGraph = nx.DiGraph(map)  # DiGraph to find routes
        paths = []
        for od in od_pairs_osmid:
            paths.append(self.find_k_shortest_paths(od[0], od[1], k, DiGraph, raw_nodes, weight='length'))

        paths_as_tuples = []
        for paths_od in paths:
            paths_od_as_tuples = []
            for path_od in paths_od:
                paths_od_as_tuples.append([(path_od[i],path_od[i+1]) for i in range(len(path_od)-1)])
            paths_as_tuples.append(paths_od_as_tuples)
        self.strategies = paths_as_tuples

        paths_as_links = [[self.sim.node2link_dict[(i, j)] for (i, j) in path] for path in paths_as_tuples[0]]
        # print("Available route choices: " + str(paths_as_links))

        if len(x) != len(od_pairs):
            print("Dimension of x is not compatible with the number of od pairs. Change x and try again.")
            return
        for i in range(len(x)):
            if len(x[i]) != k:
                print("Dimension of the " + str(i+1) + "-st entry of x is not compatible with the number of paths. Change x and try again.")
                return


        # Assign paths to the agents, so that their distribution on the paths is x
        agent_ids = []
        x_ind = 0
        for od in od_pairs:
            agent_ids.append([agent.aid for agent in self.sim.all_agents.values() if agent.origin_nid == od[0] and agent.destin_nid == od[1]])
            N = len(agent_ids[x_ind]) # Number of agents with the current od pair
            x_od = np.array(x[x_ind])
            num_of_agents = list(np.append(np.round(x_od[0:-1]*N),N-np.sum(np.round(x_od[0:-1]*N))))
            num_of_agents = [int(num) for num in num_of_agents]
            assigned_agents = 0
            for num_ind in range(len(num_of_agents)):
                masked_agent_ids = agent_ids[x_ind][assigned_agents:assigned_agents+num_of_agents[num_ind]]
                for agent in self.sim.all_agents.values():
                    if agent.aid in masked_agent_ids:
                        agent.route_igraph = [(agent.cls, agent.cle)] + [(start_nid, end_nid)
                                                              for (start_nid, end_nid) in paths_as_tuples[x_ind][num_ind]]
                assigned_agents = assigned_agents+num_of_agents[num_ind]
            x_ind += 1


        for link in self.links_df.itertuples():
            self.ave_link_densities[link.link_id] = 0

    # def init_sq_simulation_for_bpr_function_fitting_V2(self, od_pairs, k, network):
    def init_sq_simulation_for_bpr_function_fitting_V2(self, link, sa_il, sa_ol):

        self.reset_for_new_simulation()
        self.sim.create_network(self.nodes_df, self.links_df, self.charging_stations_df)
        self.sim.create_demand(self.od_df)

        '''
        all_paths = []
        for od in od_pairs:
            all_paths = all_paths + list(itertools.chain.from_iterable(self.find_paths(od[0],od[1],k,network)))
        
        link_list = list(set(all_paths))

        sa_link_list=[]
        for link in self.links_df.iloc[link_list].itertuples():
            sa_link_list.append({'id':link.link_id, 
                                'sa_il':self.find_sa_in_link(link), 
                                'sa_ol':self.find_sa_out_link(link)})
        '''
        
        # We want the agents to originate from the start node of the sa in-link,
        # use the main link, and arrive at the end node of the sa out-link.
        # For links without a valid straight-ahead predecessor/successor, use
        # a direct one-link route.  This is required for one-way boundary
        # turns and avoids treating a physically valid canonical link as an
        # unmeasurable BPR failure merely because the simulator forbids U-turns.
        for agent in self.sim.all_agents.values():
            if sa_il is not None and sa_ol is not None:
                agent.route_igraph = [
                    (agent.cls, agent.cle),
                    (sa_il.start_node_id, sa_il.end_node_id),
                    (sa_il.end_node_id, link.end_node_id),
                    (link.end_node_id, sa_ol.end_node_id),
                ]
                agent.route_link_ids = [
                    int(sa_il.link_id), int(link.link_id), int(sa_ol.link_id),
                ]
            elif sa_ol is not None:
                # Link-level BPR probe: inject at the target link's start,
                # measure the target link alone, and use one continuation
                # link so the target's downstream capacity is enforced.  The
                # continuation may be a U-turn at a boundary, but its travel
                # time is never included in the target observation.
                continuation_links = (
                    list(sa_ol)
                    if isinstance(sa_ol, (list, tuple))
                    else [sa_ol]
                )
                agent.route_igraph = [
                    (agent.cls, agent.cle),
                    (int(link.start_node_id), int(link.end_node_id)),
                ] + [
                    (int(continuation.start_node_id), int(continuation.end_node_id))
                    for continuation in continuation_links
                ]
                agent.route_link_ids = [
                    int(link.link_id),
                ] + [int(continuation.link_id) for continuation in continuation_links]
            else:
                agent.route_igraph = [
                    (agent.cls, agent.cle),
                    (int(link.start_node_id), int(link.end_node_id)),
                ]
                agent.route_link_ids = [int(link.link_id)]
            # print(agent.route_igraph)

        for link in self.links_df.itertuples():
            self.ave_link_densities[link.link_id] = 0


################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

    def find_sa_in_link(self, link):

        # Get the start and end nodes of l*
        start_node = self.nodes_df.loc[self.nodes_df.node_id == link.start_node_id]
        end_node = self.nodes_df.loc[self.nodes_df.node_id == link.end_node_id]
        # Get the longitudes and lattitudes of the start and end nodes
        x_start = start_node.lat.values[0]
        y_start = start_node.lon.values[0]
        x_end = end_node.lat.values[0]
        y_end = end_node.lon.values[0]
        # Vector representation of l*
        out_vec = (x_start-x_end, y_start-y_end)
        # Initialize straight-ahead in-link (SA((l*)^-1) as none
        sa_il = None
        # Initialize the angle between l* and (SA((l*)^-1) as 180
        il_dir = 180


        # Find all the in-links of l*
        in_links = self.links_df.loc[self.links_df.end_node_id == link.start_node_id]
        # For each in-link, find the angle that it makes with l*
        for ind,in_link in in_links.iterrows():
            # The end node of the in-link is the start node of l*
            in_link_end_node = start_node
            # Get the start node of the in-link
            in_link_start_node = self.nodes_df.loc[self.nodes_df.node_id == in_link.start_node_id]
            # Get the longitudes and lattitudes of the start and end nodes of the in-link
            in_link_x_start = in_link_start_node.lat.values[0]
            in_link_y_start = in_link_start_node.lon.values[0]
            in_link_x_end = in_link_end_node.lat.values[0]
            in_link_y_end = in_link_end_node.lon.values[0]
            # Represent the in-link as a vector
            in_vec = (in_link_x_start-in_link_x_end, in_link_y_start-in_link_y_end)
            # Calculate the angle that l* makes with the out-link
            dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
            det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
            new_il_dir = np.arctan2(det, dot)*180/np.pi
    #         print('Angle with link id ' + str(in_link.link_id) + ': ' + str(new_il_dir))
            # If this angle is smaller than the previous recorded angles, than set (SA((l*)^-1) to the in-link
            if abs(new_il_dir) < abs(il_dir):
                sa_il = in_link
                il_dir = new_il_dir
        
        return sa_il

    def find_sa_out_link(self, link):

        # Get the start and end nodes of l*
        start_node = self.nodes_df.loc[self.nodes_df.node_id == link.start_node_id]
        end_node = self.nodes_df.loc[self.nodes_df.node_id == link.end_node_id]
        # Get the longitudes and lattitudes of the start and end nodes
        x_start = start_node.lat.values[0]
        y_start = start_node.lon.values[0]
        x_end = end_node.lat.values[0]
        y_end = end_node.lon.values[0]
        # Vector representation of l*
        in_vec = (x_start-x_end, y_start-y_end)
        # Initialize straight-ahead out-link (SA(l*)) as none
        sa_ol = None
        # Initialize the angle between l* and SA(l*) as 180
        ol_dir = 180


        # Find all the out-links of l*
        out_links = self.links_df.loc[self.links_df.start_node_id == link.end_node_id]
        # For each out-link, find the angle that it makes with l*
        for ind,out_link in out_links.iterrows():
            # The start node of the out-link is the end node of l*
            out_link_start_node = end_node
            # Get the end node of the out-link
            out_link_end_node = self.nodes_df.loc[self.nodes_df.node_id == out_link.end_node_id]
            # Get the longitudes and lattitudes of the start and end nodes of the out-link
            out_link_x_start = out_link_start_node.lat.values[0]
            out_link_y_start = out_link_start_node.lon.values[0]
            out_link_x_end = out_link_end_node.lat.values[0]
            out_link_y_end = out_link_end_node.lon.values[0]
            # Represent the out-link as a vector
            out_vec = (out_link_x_start-out_link_x_end, out_link_y_start-out_link_y_end)
            # Calculate the angle that l* makes with the out-link
            dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
            det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
            new_ol_dir = np.arctan2(det, dot)*180/np.pi
    #         print('Angle with link id ' + str(out_link.link_id) + ': ' + str(new_ol_dir))
            # If this angle is smaller than the previous recorded angles, than set SA(l*) to the out-link
            if abs(new_ol_dir) < abs(ol_dir):
                sa_ol = out_link
                ol_dir = new_ol_dir
        
        return sa_ol

    def find_paths(self, orig, dest, k, network):
        # Create node and edge objects from the osmnx road graph 
        nodesOX, edgesOX = ox.graph_to_gdfs(network)
        raw_nodes = nodesOX.copy().reset_index()
        raw_nodes['node_id'] = np.arange(raw_nodes.shape[0])
        # Find the OSM ids of the origin and destination
        od_pair_osmid = [raw_nodes.loc[raw_nodes.node_id == orig, 'osmid'].values[0],
                        raw_nodes.loc[raw_nodes.node_id == dest, 'osmid'].values[0]]
        # Translate the osmnx road graph to an nx digraph
        DiGraph = nx.DiGraph(network)
        
        # Find the k-shortest paths
        paths = []
        paths.append(self.find_k_shortest_paths(od_pair_osmid[0], od_pair_osmid[1], k, DiGraph, raw_nodes, weight='length'))
        paths = paths[0]
        # Represent the paths in terms of link ids
        paths_as_links = []
        for path in paths:
            path_as_links = []
            for i in range(len(path)-1):
                # Find the row index of the link with start_node_id == path[i] and end_node_id == path[i+1]
                row_ind = np.where([a and b for a, b in zip(list(self.links_df.start_node_id == path[i]), 
                                                            list(self.links_df.end_node_id == path[i+1]))])
                # Append the id of the link corresponding to this row
                path_as_links.append(self.links_df.loc[row_ind,'link_id'].values[0])
            paths_as_links.append(path_as_links)

        self.active_links = paths_as_links
        # print(paths_as_links)
        
        return paths_as_links
        
###################################################################### ADDITION ENDS

    def single_step_sq_sim(self, t):
        # load agents
        for agent in self.sim.all_agents.values():
            agent.load_trips(t)
            # reroute
            if t > 0 and t % self.reroute_freq == 0:
                agent.get_path(g=self.sim.g)

        # run link model
        for link in self.sim.all_links.values():
            link.run_link_model(t)

###################################################################### ADDITION STARTS
            for ind in range(len(link.running_travel_time_list)):
                link.running_travel_time_list[ind][1] += 1
                # print("Time: " + str(t))
                # print("Travel time list for link id " + str(link.lid) + ": " + str(link.running_travel_time_list))

            if link.ltype != 'vl_in':         # UNCOMMENT FOR TRAFFIC DATA GENERATION
                self.ave_link_densities[link.lid] = self.ave_link_densities[link.lid] + link.density         # UNCOMMENT FOR TRAFFIC DATA GENERATION

        # run node model
        standard_node_ids_to_run = {
            link.end_nid for link in self.sim.all_links.values() if len(link.queue_veh) > 0 and self.sim.all_nodes[link.end_nid].ntype != 'Station_Entrance/Exit' and self.sim.all_nodes[link.end_nid].ntype != 'Station'}

        station_ent_ex_node_ids = {node.nid for node in self.sim.all_nodes.values() if node.ntype == 'Station_Entrance/Exit'}

        for nid in standard_node_ids_to_run:
            node = self.sim.all_nodes[nid]
            node.run_node_model(t)
        for nid in station_ent_ex_node_ids:
            node = self.sim.all_nodes[nid]
            node.run_station_ent_ex_node_model(t)

        # for node in self.sim.all_nodes.values():
        #     if node.ntype == 'Station_Entrance/Exit':
        #         node.run_station_ent_ex_node_model(t)
        #     else:
        #         node.run_node_model(t)


        # run charging station model
        charging_station_ids_to_run = {station.station_id for station in self.sim.all_charging_stations.values() if
            len(station.ent_queue) > 0 or len(station.charging_vehicles) > 0 or len(station.exit_queue) > 0}
        for stat_id in charging_station_ids_to_run: # self.sim.all_charging_stations.keys():
            station = self.sim.all_charging_stations[stat_id]
            station.run_station_model(t)

###################################################################### ADDITION ENDS

    # count the number of evacuees that have successfully reach their destination

    def arrival_counts(self, t, save_path):
        arrival_cnts = np.sum(
            [1 for a in self.sim.all_agents.values() if a.status == 'arr'])
        # print(
        #     f'At {t} seconds, {arrival_cnts} evacuees successfully reached the destination.')
        if arrival_cnts == len(self.sim.all_agents):
            # print(f"All agents arrive at destinations at time {t} seconds.")
            self.running = False
            return False
        with open(save_path, 'a') as t_stats_outfile:
            t_stats_outfile.write(f"{t},{arrival_cnts}\n")
        return True

    def write_link_outputs(self, save_path):
        link_output = pd.DataFrame([(link.lid, len(link.queue_veh), len(link.run_veh), np.round((len(link.queue_veh) + len(link.run_veh)) / (link.length * link.lanes+0.00001) * 100, 2), link.geometry)
                                   for link in self.sim.all_links.values() if link.ltype[0:2] != 'vl'], columns=['link_id', 'queue_vehicle_count', 'run_vehicle_count', 'vehicle_per_100m', 'geometry'])
        link_output = link_output[(link_output['queue_vehicle_count'] > 0) | (
            link_output['run_vehicle_count'] > 0)].reset_index(drop=True)
        link_output.to_csv(save_path, index=False)

    def write_node_outputs(self, save_path):
        node_predepart = pd.DataFrame([(agent.cle, 1) for agent in self.sim.all_agents.values() if (agent.status in [
                                      None, 'loaded'])], columns=['node_id', 'predepart_cnt']).groupby('node_id').agg({'predepart_cnt': np.sum}).reset_index()
        if node_predepart.shape[0] > 0:
            node_predepart = node_predepart.merge(
                self.nodes_df[['node_id', 'lat', 'lon']], how='left', on='node_id')
            node_predepart.to_csv(save_path, index=False)

    def _finalize_simulation(self):
        """Compute completed-agent/link statistics exactly once."""
        self.tot_travel_time = 0
        for agent in self.sim.all_agents.values():
            if np.isnan(agent.arrival_time):
                raise RuntimeError(f'Agent {agent.aid} has no arrival time')
            self.tot_travel_time += agent.arrival_time - agent.dept_time
        for link in self.sim.all_links.values():
            if link.ltype == 'vl_in' or not link.completed_travel_time_list:
                continue
            total = sum(travel_time for _, travel_time in link.completed_travel_time_list)
            link.ave_travel_time = total / len(link.completed_travel_time_list)
            occupancy = max(1, link.occup_time)
            link.ave_flow = len(link.completed_travel_time_list) / occupancy
            self.ave_link_densities[link.lid] = self.ave_link_densities[link.lid] / occupancy
        self.simulation_status = 'completed'

    def spatial_queue_simulation(self, scenario_name, t_end=10801, output_dir='traffic_outputs'):
        os.makedirs(output_dir, exist_ok=True)
        for subdirectory in ('t_stats', 'link_stats', 'node_stats'):
            os.makedirs(os.path.join(output_dir, subdirectory), exist_ok=True)
        self.simulation_status = 'running'
        arrival_output_path = f'{output_dir}/t_stats/arrivals_{scenario_name}.csv'
        with open(arrival_output_path, 'w') as t_stats_outfile:
            t_stats_outfile.write("t,arrival_count"+"\n")

        # iterate through each time step
        # Include the horizon boundary in the completion check so vehicles
        # arriving exactly at ``t_end`` are not misreported as timed out.
        for t in range(t_end + 1):
            # run the spatial-queue simulation for one step
            self.single_step_sq_sim(t)

###################################################################### ADDITION STARTS

            # Plot the links with critical storage capacity every 100 steps
            # if t % 500 == 0 and t > 0:
            #     link_ids_to_plot = [link.lid for link in self.sim.all_links.values() if link.st_c < 8]
            #     self.plot_links_and_nodes(link_ids_to_plot, [], 'red')
            #     # print("Congested roads: " + str(link_ids_to_plot))

            for link in self.sim.all_links.values():
                if len(link.queue_veh) + len(link.run_veh) > 0:
                    link.occup_time += 1

###################################################################### ADDITION ENDS

            # break if all agents have reached their destinations
            if not self.running:
                self._finalize_simulation()
                return
            # output time-step results every 100 seconds
            if t % 100 == 0:
                if not self.arrival_counts(t, arrival_output_path):
                    self._finalize_simulation()
                    return
                link_output_path = f'{output_dir}/link_stats/l{scenario_name}_at_{t}.csv'
                node_output_path = f'{output_dir}/node_stats/n{scenario_name}_at_{t}.csv'
                self.write_link_outputs(link_output_path)
                self.write_node_outputs(node_output_path)

        self.simulation_status = 'timeout'
        self.simulation_error = (
            f'No completion by t_end={t_end}; '
            f'{sum(agent.status == "arr" for agent in self.sim.all_agents.values())}/'
            f'{len(self.sim.all_agents)} agents arrived'
        )
        raise TimeoutError(self.simulation_error)

###################################################################### ADDITION STARTS

    def plot_links_and_nodes(self, link_ids_to_plot, node_ids_to_plot, color_choice):
        link_geos_to_plot = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(self.links_df.iloc[link_ids_to_plot].geometry.tolist()))
        node_geos_to_plot = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=self.nodes_df.loc[node_ids_to_plot, 'lon'].to_list(),
                                        y=self.nodes_df.loc[node_ids_to_plot, 'lat'].to_list()))

        nodesOX = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=self.nodes_df['lon'].to_list(), y=self.nodes_df['lat'].to_list()))
        edgesOX = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.links_df['geometry']).tolist())
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.patch.set_facecolor('black')
        ax.set_axis_off()
        nodesOX.plot(ax=ax, color='white')
        edgesOX.plot(ax=ax, color='white')

        link_geos_to_plot.plot(ax=ax, color=color_choice, zorder=2)
        node_geos_to_plot.plot(ax=ax, color=color_choice, zorder=2)

    def return_traffic_data(self, x, orig, dest, density = None):
        if density == 0:
            traffic_df = pd.DataFrame(columns = ['od_pair', 'x', 'link_id', 'link_length', 'free_flow_speed', 'capacity', 'flow', 'travel_time'])
            for link in self.sim.all_links.values():
                if link.ltype != 'vl_in':
                    data = {'od_pair': [[orig, dest]], 'x': [[x]], 'link_id': link.lid, 'link_length': link.length,
                            'free_flow_speed': link.maxmph/2.2369, 'capacity': link.ou_c,
                            'flow': link.ave_flow, 'travel_time': link.ave_travel_time}
                    traffic_df = pd.concat([traffic_df, pd.DataFrame(data)], ignore_index=True)

        else:
            traffic_df = pd.DataFrame(
            columns=['od_pair', 'x', 'link_id', 'link_length', 'free_flow_speed', 'capacity', 'density', 'travel_time'])
            for link in self.sim.all_links.values():
                if link.ltype != 'vl_in':
                    data = {'od_pair': [[orig, dest]], 'x': [[x]], 'link_id': link.lid, 'link_length': link.length,
                            'free_flow_speed': link.maxmph / 2.2369, 'capacity': link.ou_c,
                            'density': self.ave_link_densities[link.lid], 'travel_time': link.ave_travel_time}
                    traffic_df = pd.concat([traffic_df, pd.DataFrame(data)], ignore_index=True)

        return traffic_df

    def init_sq_simulation_with_Nash_flows(self, data, num_of_vehs, num_need_to_charge):
        # Route assignments are now always initialized through the shared
        # multi-OD contract.  Keep this legacy entry point for callers that
        # still provide aggregate vehicle counts.
        assignments_ch = {}
        assignments_no = {}
        for od, group in data.items():
            assignments_no[od] = _legacy_flow_counts(group.get('no charging type', []), self.od_df, od, False)
            assignments_ch[od] = _legacy_flow_counts(group.get('charging type', []), self.od_df, od, True)
        return self._init_multi_od_path_assignment(data, assignments_ch, assignments_no)

    def init_sq_simulation_with_switched_agent(self, data, num_of_vehs, num_need_to_charge):
        assignments_ch = {}
        assignments_no = {}
        for od, group in data.items():
            assignments_no[od] = _legacy_flow_counts(group.get('no charging type', []), self.od_df, od, False)
            assignments_ch[od] = _legacy_flow_counts(group.get('charging type', []), self.od_df, od, True)
        return self._init_multi_od_path_assignment(data, assignments_ch, assignments_no)

    def return_Nash_path_assignment(self, data, num_of_vehs, num_need_to_charge):
        assignments_ch = {}
        assignments_no = {}
        for od, group in data.items():
            assignments_no[od] = _legacy_flow_counts(group.get('no charging type', []), self.od_df, od, False)
            assignments_ch[od] = _legacy_flow_counts(group.get('charging type', []), self.od_df, od, True)
        return {'F1': assignments_no, 'F2': assignments_ch}

    def _init_multi_od_path_assignment(self, data, num_for_ch_paths, num_for_no_ch_paths):
        """Assign exact F1/F2 route counts for all OD pairs jointly."""
        self.sim.create_network(self.nodes_df, self.links_df, self.charging_stations_df)
        self.sim.create_demand(self.od_df)
        self.no_ch_routes = []
        self.ch_routes = []
        self.route_groups = []
        self.route_agent_ids = {}

        def normalize_key(value):
            if isinstance(value, str):
                origin, destination = value.split(',')
                return int(origin), int(destination)
            return tuple(value)

        def entries_to_pairs(entries):
            pairs = []
            for entry in entries:
                path = entry.get('path', entry.get('links', []))
                if path and isinstance(path[0], (tuple, list)) and len(path[0]) == 2:
                    pairs.append([tuple(edge) for edge in path])
                else:
                    pairs.append([(path[i], path[i + 1]) for i in range(len(path) - 1)])
            return pairs

        grouped_agents = {}
        for agent in self.sim.all_agents.values():
            vehicle_type = 'F2' if agent.is_EV and agent.need_to_charge else 'F1'
            grouped_agents.setdefault(((agent.origin_nid, agent.destin_nid), vehicle_type), []).append(agent)
        for agents in grouped_agents.values():
            agents.sort(key=lambda agent: agent.aid)

        assigned_agent_ids = set()

        for raw_od, group_data in data.items():
            od = normalize_key(raw_od)
            no_entries = group_data.get('no charging type', group_data.get('F1', []))
            ch_entries = group_data.get('charging type', group_data.get('F2', []))
            no_paths = entries_to_pairs(no_entries)
            ch_paths = entries_to_pairs(ch_entries)
            self.no_ch_routes.extend(no_paths)
            self.ch_routes.extend(ch_paths)

            for vehicle_type, entries, paths, counts_input in (
                ('F1', no_entries, no_paths, num_for_no_ch_paths),
                ('F2', ch_entries, ch_paths, num_for_ch_paths),
            ):
                if vehicle_type == 'F1' and any(
                    entry.get('station node') is not None for entry in entries
                ):
                    raise ValueError(f'F1 route for {od} illegally contains a charger')
                if vehicle_type == 'F2' and any(
                    entry.get('station node') is None for entry in entries
                ):
                    raise ValueError(f'F2 route for {od} must contain exactly one charger')
                if isinstance(counts_input, dict):
                    counts = counts_input.get(od, counts_input.get(f'{od[0]},{od[1]}', []))
                else:
                    counts = counts_input if len(data) == 1 else []
                counts = [int(value) for value in (counts or [0] * len(paths))]
                agents = grouped_agents.get((od, vehicle_type), [])
                if len(counts) != len(paths):
                    raise ValueError(f'Count/route mismatch for {od} {vehicle_type}')
                if sum(counts) != len(agents):
                    raise ValueError(
                        f'Assigned {sum(counts)} vehicles for {od} {vehicle_type}, '
                        f'but demand contains {len(agents)}'
                    )
                group = {
                    'od_pair': od,
                    'vehicle_type': vehicle_type,
                    'entries': entries,
                    'paths': paths,
                    'route_ids': [
                        entry.get('route_id', f'{od[0]}_{od[1]}_{vehicle_type}_{index}')
                        for index, entry in enumerate(entries)
                    ],
                    'agent_ids': [],
                    'route_agent_ids': {i: [] for i in range(len(paths))},
                }
                self.route_groups.append(group)
                assignments = [index for index, count in enumerate(counts) for _ in range(count)]
                for agent, route_index in zip(agents, assignments):
                    link_ids = entries[route_index].get('link_ids', [])
                    if link_ids and len(link_ids) != len(paths[route_index]):
                        raise ValueError(
                            f'Route/link identity mismatch for {od} {vehicle_type} '
                            f'route {route_index}'
                        )
                    agent.route_igraph = [(agent.cls, agent.cle)] + list(paths[route_index])
                    agent.route_link_ids = [int(value) for value in link_ids]
                    agent.route_group_key = (od, vehicle_type)
                    agent.route_index = route_index
                    if vehicle_type == 'F2':
                        agent.go_to_station_id = entries[route_index].get('station node')
                    group['agent_ids'].append(agent.aid)
                    assigned_agent_ids.add(agent.aid)
                    group['route_agent_ids'][route_index].append(agent.aid)
                self.route_agent_ids[(od, vehicle_type)] = group['route_agent_ids']

        for link in self.links_df.itertuples():
            self.ave_link_densities[link.link_id] = 0
        if assigned_agent_ids != set(self.sim.all_agents):
            missing = sorted(set(self.sim.all_agents) - assigned_agent_ids)
            raise ValueError(
                f'Not every demand agent received a route; missing agent IDs: {missing[:10]}'
            )

    def init_sq_simulation_with_path_assignment(self,data,num_for_ch_paths,num_for_no_ch_paths):

        # Multi-OD assignments use exact per-OD/type counts for every caller.
        return self._init_multi_od_path_assignment(data, num_for_ch_paths, num_for_no_ch_paths)

    def _check_route_details(self):
        """Return agent-wise route travel-time summaries for multi-OD runs."""
        details = {}
        link_by_id = self.sim.all_links
        for group in self.route_groups:
            key = (tuple(group['od_pair']), group['vehicle_type'])
            route_results = []
            for route_index, path in enumerate(group['paths']):
                agent_ids = group['route_agent_ids'].get(route_index, [])
                observed = []
                for agent_id in agent_ids:
                    agent = self.sim.all_agents.get(agent_id)
                    if agent is not None and np.isfinite(agent.arrival_time):
                        observed.append(float(agent.arrival_time - agent.dept_time))
                if observed:
                    travel_time = float(np.mean(observed))
                else:
                    travel_time = 0.0
                    for start_nid, end_nid in path:
                        try:
                            link_id = self.sim.resolve_link_id(start_nid, end_nid)
                        except ValueError:
                            # Route records with explicit link IDs are allowed.
                            entry = group['entries'][route_index]
                            link_ids = entry.get('link_ids', [])
                            edge_index = list(path).index((start_nid, end_nid))
                            link_id = link_ids[edge_index]
                        link = link_by_id[int(link_id)]
                        # Wardrop/better-response cost under the current
                        # network state.  A route need not carry an agent for
                        # its cost to be assembled from its links; links with
                        # no observations retain their initialized FFT.
                        travel_time += float(
                            link.ave_travel_time
                            if np.isfinite(link.ave_travel_time) and link.ave_travel_time > 0
                            else link.fft
                        )
                    if group['vehicle_type'] == 'F2':
                        travel_time += float(group['entries'][route_index].get('station_cost', 0.0) or 0.0)
                details.setdefault(key, []).append({
                    'route_id': group.get('route_ids', [])[route_index]
                    if route_index < len(group.get('route_ids', [])) else None,
                    'route_index': route_index,
                    'travel_time': travel_time,
                    'agent_count': len(agent_ids),
                    'used': bool(agent_ids),
                })
        return details

    def check_NE(self, return_details=False):
        if return_details and self.route_groups:
            return self._check_route_details()
        
        ch_routes_links = []
        for route in self.ch_routes:
            route_links = []
            for (start_nid, end_nid) in route:
                for link in self.sim.all_links.values():
                    if link.start_nid == start_nid and link.end_nid == end_nid:
                        route_links.append(link.lid)
            ch_routes_links.append(route_links)
        
        no_ch_routes_links = []
        for route in self.no_ch_routes:
            route_links = []
            for (start_nid, end_nid) in route:
                for link in self.sim.all_links.values():
                    if link.start_nid == start_nid and link.end_nid == end_nid:
                        route_links.append(link.lid)
            no_ch_routes_links.append(route_links)


        ch_routes_travel_times = []
        no_ch_routes_travel_times = []
        ch_routes_flows = []
        no_ch_routes_flows = []
        ch_routes_occup_time = []
        no_ch_routes_occup_time = []
        
        for route in ch_routes_links:
            route_travel_time = 0
            route_flow = 0
            route_occup_time = 0
            for lid in route:
                for link in self.sim.all_links.values():
                    if link.lid == lid:
                        route_travel_time = route_travel_time + link.ave_travel_time
                        route_flow = route_flow + link.ave_flow
                        route_occup_time = route_occup_time + link.occup_time
            ch_routes_travel_times.append(route_travel_time)
            ch_routes_flows.append(route_flow)
            ch_routes_occup_time.append(route_occup_time)
        for route in no_ch_routes_links:
            route_travel_time = 0
            route_flow = 0
            route_occup_time = 0
            for lid in route:
                for link in self.sim.all_links.values():
                    if link.lid == lid:
                        route_travel_time = route_travel_time + link.ave_travel_time
                        route_flow = route_flow + link.ave_flow
                        route_occup_time = route_occup_time + link.occup_time
            no_ch_routes_travel_times.append(route_travel_time)
            no_ch_routes_flows.append(route_flow)
            no_ch_routes_occup_time.append(route_occup_time)
        

        return [ch_routes_occup_time, no_ch_routes_occup_time] #, [ch_routes_travel_times, no_ch_routes_travel_times], [ch_routes_flows, no_ch_routes_flows]

###################################################################### ADDITION ENDS

def cli():
    parser = argparse.ArgumentParser(
        description='command line tool for running spatial queue model')
    parser.add_argument('--nodes', required=True,
                        help='path to nodes csv that represents all the intersections of your model')
    parser.add_argument('--links', required=True, help='path to link csv')
    parser.add_argument('--ods', required=True,
                        help='path to travel demand csv')
    parser.add_argument(
        '--cf', help='path to contraflow links csv', default='')
    parser.add_argument('--name', default='berkeley-evac',
                        help='path to travel demand csv')
    args = parser.parse_args()

    runner = Runner(args.links, args.nodes, args.ods, args.cf)
    runner.init_sq_simulation()
    runner.spatial_queue_simulation(args.name)
