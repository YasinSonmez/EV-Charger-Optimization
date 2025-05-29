import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt


class RoadNet:
    def __init__(self, name):
        self.name = name
        self.graph = []
        self.nodes = []
        self.edges = []
        self.demand = []

    def get_map(self, no_lat, so_lat, east_long, west_long):
        self.graph = ox.graph_from_bbox([west_long, so_lat, east_long, no_lat], network_type='drive')

    def plot_links_and_nodes(self, link_ids_to_plot):
        link_geos_to_plot = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(self.edges.iloc[link_ids_to_plot].geometry.tolist()))

        edgesOX = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.edges['geometry']).tolist())
        nodesOX = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                x=self.nodes['lon'].to_list(),
                y=self.nodes['lat'].to_list()))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.patch.set_facecolor('white')
        ax.set_axis_off()
        nodesOX.plot(ax=ax, color='black', alpha=0.7)
        edgesOX.plot(ax=ax, color='black', alpha=0.7)
        link_geos_to_plot.plot(ax=ax, color='g', zorder=2, linewidth=8.0, alpha=0.8)

    def rearrange_data(self):
        nodesOX, edgesOX = ox.graph_to_gdfs(self.graph)

        raw_nodes = nodesOX.copy().reset_index()
        raw_nodes['node_id'] = np.arange(raw_nodes.shape[0])
        raw_nodes['lon'] = raw_nodes['x']
        raw_nodes['lat'] = raw_nodes['y']
        raw_nodes['node_osmid'] = raw_nodes['osmid']

        self.osmid_to_nid_dict = {getattr(node, 'osmid'): getattr(node, 'node_id') for node in raw_nodes.itertuples()}
        self.nid_to_osmid_dict = {getattr(node, 'node_id'): getattr(node, 'osmid') for node in raw_nodes.itertuples()}

        raw_edges = edgesOX.copy().reset_index()
        raw_edges['link_id'] = np.arange(raw_edges.shape[0])
        raw_edges['start_node_id'] = raw_edges['u'].map(self.osmid_to_nid_dict)
        raw_edges['end_node_id'] = raw_edges['v'].map(self.osmid_to_nid_dict)
        raw_edges['start_osmid'] = raw_edges['u']
        raw_edges['end_osmid'] = raw_edges['v']
        raw_edges['length'] = raw_edges['length'].astype(float)
        raw_edges['geometry'] = raw_edges['geometry'].apply(wkt.dumps)

        raw_nodes = raw_nodes.drop(['x', 'y', 'street_count', 'geometry', 'highway', 'osmid'], axis=1)
        raw_edges = raw_edges[['link_id', 'start_node_id', 'end_node_id', 'length', 'start_osmid', 'end_osmid', 'geometry']]

        self.nodes = raw_nodes
        self.edges = raw_edges 